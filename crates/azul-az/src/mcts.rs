//! 单树 MCTS（PUCT）——AlphaZero 轨道的搜索核心（plan stage 5）。
//!
//! # ⚠️ 视角翻转规则（worklog bug #4 的教训，绝不能重蹈）
//!
//! `FastState::apply()` 在游戏结束时**不翻转 `turn`**，且补砖后的新回合由
//! 先手标记（first-player marker）持有者开始——**同一玩家可以合法连走两手**。
//! 因此任何"父 → 子取反"的备份逻辑（MCTS backup、minimax negamax）都
//! **只在 `turn` 改变时取反**，否则赢/输符号会翻转（"我赢"被读成 -1）。
//! 参见 azul-search/src/search.rs 的 `child_value` 注释。本文件的备份
//! (`simulate`) 与选择 (`select_action`) 都显式比较 turn 后再决定符号。
//!
//! # 设计
//! - 节点统计：N、W、Q、先验 P；子节点按 180 索引稀疏存储（只展开合法着法）。
//! - 选择：`Q + c_puct * P * sqrt(N_parent) / (1 + N)`，Q 恒从**该节点待走方**
//!   视角存储；选择时换算到父方视角（turn 相同则不换算——同上警告）。
//! - 叶评估：直接取 net value（不做随机 rollout——已证伪）；终局节点用真实
//!   结果 `GameResult::value_for(turn)`（±1）。
//! - 补砖随机性：着法触发回合结束（triggers_refill）时用确定性 chance seed
//!   （由 position hash + move identity 派生），单样本起步，留 `chance_samples`
//!   参数。同一（局面, 着法）永远抽到同一补砖，逐字节可复现。
//! - 根 Dirichlet 噪声（α=0.3, ε=0.25）仅自对弈时开（`dirichlet` 参数化）。
//! - 无 transposition table（起步口径）：不合并转置，每条边独立子树。

use crate::action::{self, ACTION_DIM};
use crate::net::PvNet;
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::types::{Dest, Move, Source, NUM_FACTORIES, PATTERN_LINES};

/// PUCT 探索常数（AlphaZero 论文口径）。
pub const C_PUCT: f32 = 1.5;
/// Dirichlet 噪声浓度 α。
pub const DIRICHLET_ALPHA: f32 = 0.3;
/// 根噪声混合比例 ε。
pub const DIRICHLET_EPS: f32 = 0.25;
/// 默认补砖样本数（单样本确定性起步）。
pub const DEFAULT_CHANCE_SAMPLES: u64 = 1;
/// 默认每手模拟数。
pub const DEFAULT_SIMS: u32 = 160;

// ---------------------------------------------------------------------------
// 补砖确定性：复制 azul-search/src/search.rs 的 triggers_refill / move_identity /
// chance_seed 思路（那边是私有函数，此处独立实现，不动 azul-search）。
// ---------------------------------------------------------------------------

/// 着法 `m` 是否清空工厂+中央、触发回合结束与补砖（chance 事件）。
pub fn triggers_refill(st: &FastState, m: &Move) -> bool {
    let mut in_play = 0u32;
    for f in 0..NUM_FACTORIES {
        for c in 0..5u8 {
            in_play += st.factory_count(f, c) as u32;
        }
    }
    for c in 0..5u8 {
        in_play += st.centre_count(c) as u32;
    }
    let taken = match m.source {
        Source::Factory(f) => st.factory_count(f, m.colour),
        Source::Centre => st.centre_count(m.colour),
    } as u32;
    in_play == taken
}

/// 着法的稳定身份（用于派生 chance seed）。
#[inline]
fn move_identity(m: &Move) -> u64 {
    let s = match m.source {
        Source::Factory(f) => f as u64,
        Source::Centre => NUM_FACTORIES as u64,
    };
    let d = match m.dest {
        Dest::Row(r) => r as u64,
        Dest::Floor => PATTERN_LINES as u64,
    };
    (s << 16) | ((m.colour as u64) << 8) | d
}

/// 由 (position hash, move identity, sample 序号) 派生确定性补砖 seed。
/// 同一 (局面, 着法, 样本) 永远得到同一补砖 —— 树内一致、可复现。
#[inline]
pub fn chance_seed(pos_key: u64, m: &Move, sample: u64) -> u64 {
    pos_key
        .wrapping_add(move_identity(m).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(sample.wrapping_mul(0x853C_49E6_748F_EA9B))
        .wrapping_add(0xA5A5_1234_5678_9ABC)
}

// ---------------------------------------------------------------------------
// 树结构
// ---------------------------------------------------------------------------

/// 子节点未分配的哨兵。
const NO_CHILD: usize = usize::MAX;

/// 稀疏子边：动作索引 + 先验 P + 子节点槽位（惰性分配）。
struct Edge {
    action: usize,
    p: f32,
    child: usize,
}

/// 树节点。Q/W 从**该节点待走方**视角（负值 = 待走方劣势）。
struct Node {
    /// 待走方（FastState.turn）；终局节点保留最后待走方。
    turn: u8,
    /// 是否终局。
    terminal: bool,
    /// 终局价值（待走方视角 ±1/0）；非终局为 0。
    terminal_value: f32,
    /// 是否已展开（edges 就绪）。
    expanded: bool,
    /// 访问次数。
    n: u32,
    /// 价值累计（本节点待走方视角）。
    w: f32,
    /// 子边（先验 P 存边上）。
    edges: Vec<Edge>,
}

impl Node {
    #[inline]
    fn q(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.w / self.n as f32
        }
    }
}

// ---------------------------------------------------------------------------
// 搜索配置与搜索器
// ---------------------------------------------------------------------------

/// 搜索配置。`dirichlet = true` 时根节点加 Dirichlet 噪声（自对弈用）。
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub dirichlet: bool,
    /// 补砖样本数（参数化；>1 时对同一着法的不同补砖样本求期望，起步单样本）。
    pub chance_samples: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            sims: DEFAULT_SIMS,
            c_puct: C_PUCT,
            dirichlet: false,
            chance_samples: DEFAULT_CHANCE_SAMPLES,
        }
    }
}

/// 单树 MCTS 搜索器。每次 `search` 以新局面重建根（不做树复用，起步口径）。
pub struct Mcts<'a> {
    net: &'a PvNet,
    cfg: MctsConfig,
    nodes: Vec<Node>,
}

impl<'a> Mcts<'a> {
    pub fn new(net: &'a PvNet, cfg: MctsConfig) -> Self {
        Mcts { net, cfg, nodes: Vec::new() }
    }

    /// 从 `st` 出发做 `cfg.sims` 次模拟，返回根节点访问分布（长度 180：
    /// 各合法着法的 N/ΣN，非法/未访问为 0，总和 ≈ 1）。
    /// `root_seed` 只用于根 Dirichlet 噪声（确定性可复现）。
    pub fn search(&mut self, st: &FastState, root_seed: u64) -> [f32; ACTION_DIM] {
        self.nodes.clear();
        self.nodes.push(new_node(st)); // 根
        if !st.ended {
            let (logits, _v) = self.net.forward_state(st);
            self.expand_node(0, st, &logits);
        }
        if self.cfg.dirichlet {
            self.add_dirichlet_noise(0, root_seed);
        }
        for _ in 0..self.cfg.sims {
            self.simulate(*st);
        }
        self.root_policy(0)
    }

    /// 根节点的 Q（根待走方视角的估计价值）。`search` 后调用。
    pub fn root_q(&self) -> f32 {
        self.nodes.first().map(|r| r.q()).unwrap_or(0.0)
    }

    /// 根访问分布 -> 180 维数组。
    fn root_policy(&self, root: usize) -> [f32; ACTION_DIM] {
        let node = &self.nodes[root];
        let mut out = [0.0f32; ACTION_DIM];
        let total: u32 = node
            .edges
            .iter()
            .map(|e| if e.child == NO_CHILD { 0 } else { self.nodes[e.child].n })
            .sum();
        if total == 0 {
            return out;
        }
        for e in &node.edges {
            if e.child != NO_CHILD {
                out[e.action] = self.nodes[e.child].n as f32 / total as f32;
            }
        }
        out
    }

    /// 一次模拟：选择 -> 展开/评估叶 -> 视角正确地备份整条路径。
    fn simulate(&mut self, mut st: FastState) {
        // 路径：(节点索引, 到达该节点的着法；根为 None)。
        let mut path: Vec<(usize, Option<Move>)> = vec![(0, None)];
        let leaf_value = loop {
            let cur = path.last().unwrap().0;
            if self.nodes[cur].terminal {
                break self.nodes[cur].terminal_value;
            }
            if !self.nodes[cur].expanded {
                // 叶节点：net 前向评估（value 用于备份），并立即展开
                // （子节点仍惰性分配，只做 legal movegen）。
                let (logits, v) = self.net.forward_state(&st);
                self.expand_node(cur, &st, &logits);
                break v;
            }
            // PUCT 选择。
            let action = self.select_action(cur);
            let m = action::from_index(action);
            // 补砖确定性：触发回合结束的着法用 chance seed；其余不需要 rng。
            let seed = if triggers_refill(&st, &m) {
                chance_seed(st.hash(), &m, 0)
            } else {
                0
            };
            st.apply(m, &mut Rnd::new(seed));
            // 惰性分配子节点（一次只分配选中的那条边，兄弟不提前展开）。
            let slot = self.nodes[cur]
                .edges
                .iter()
                .find(|e| e.action == action)
                .expect("selected action has an edge")
                .child;
            let child = if slot == NO_CHILD {
                self.nodes.push(new_node(&st));
                let c = self.nodes.len() - 1;
                let e = self.nodes[cur]
                    .edges
                    .iter_mut()
                    .find(|e| e.action == action)
                    .expect("selected action has an edge");
                e.child = c;
                c
            } else {
                slot
            };
            path.push((child, Some(m)));
        };

        // 备份：`leaf_value` 是叶节点待走方视角。沿路径向上，v[i-1] 与 v[i]
        // 的符号关系**只在 turn 改变时取反**（同一玩家可连走两手——见文件头
        // 警告）。每个路径节点都按自己的待走方视角累计。
        let k = path.len() - 1;
        let mut v = leaf_value;
        self.nodes[path[k].0].n += 1;
        self.nodes[path[k].0].w += v;
        for i in (0..k).rev() {
            let (parent_i, _m) = path[i];
            let child_i = path[i + 1].0;
            if self.nodes[child_i].turn != self.nodes[parent_i].turn {
                v = -v;
            }
            self.nodes[parent_i].n += 1;
            self.nodes[parent_i].w += v;
        }
    }

    /// 从节点子边中按 PUCT `Q_parent + c_puct * P * sqrt(N_parent) / (1 + N)`
    /// 选一条，返回动作索引。Q 换算到父方视角时遵守 turn 比较规则。
    fn select_action(&self, node_i: usize) -> usize {
        let node = &self.nodes[node_i];
        let sqrt_n = (node.n.max(1) as f32).sqrt();
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for e in &node.edges {
            // 未访问子节点：N=0、Q=0（符号无关），无需 turn 信息。
            let (n_child, q_parent) = if e.child == NO_CHILD {
                (0u32, 0.0f32)
            } else {
                let child = &self.nodes[e.child];
                // 子节点 Q 是子待走方视角。换算到父方视角：turn 改变才取反
                //（同一玩家连走两手时两边是同一人，不能取反——bug #4 教训）。
                let qp = if child.turn == node.turn { child.q() } else { -child.q() };
                (child.n, qp)
            };
            let score = q_parent + self.cfg.c_puct * e.p * sqrt_n / (1.0 + n_child as f32);
            if score > best_score {
                best_score = score;
                best = Some(e.action);
            }
        }
        best.expect("expanded node has at least one legal edge")
    }

    /// 展开节点：合法着法 softmax 先验（只在合法动作上归一化）+ 稀疏子边。
    /// 子节点在被选中时惰性分配（避免对每个兄弟做 net 前向）。
    fn expand_node(&mut self, node_i: usize, st: &FastState, logits: &[f32; ACTION_DIM]) {
        if self.nodes[node_i].expanded {
            return;
        }
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        let mut acts: Vec<usize> = Vec::with_capacity(buf.len);
        for m in buf.iter() {
            acts.push(action::to_index(&m));
        }
        if acts.is_empty() {
            // 无合法着法（apply 内部会结算回合，理论上不出现）：视作和棋终局。
            let node = &mut self.nodes[node_i];
            node.terminal = true;
            node.terminal_value = 0.0;
            node.expanded = true;
            return;
        }
        let mut max_logit = f32::NEG_INFINITY;
        for &a in &acts {
            if logits[a] > max_logit {
                max_logit = logits[a];
            }
        }
        let mut sum = 0.0f32;
        let mut ps = Vec::with_capacity(acts.len());
        for &a in &acts {
            let p = (logits[a] - max_logit).exp();
            sum += p;
            ps.push(p);
        }
        for p in ps.iter_mut() {
            *p /= sum;
        }
        let node = &mut self.nodes[node_i];
        node.edges = acts
            .into_iter()
            .zip(ps)
            .map(|(a, p)| Edge { action: a, p, child: NO_CHILD })
            .collect();
        node.expanded = true;
    }

    /// 根节点 Dirichlet 噪声：P ← (1-ε)·P + ε·Dir(α)，确定性由 root_seed 驱动。
    fn add_dirichlet_noise(&mut self, root: usize, seed: u64) {
        let node = &mut self.nodes[root];
        if node.edges.is_empty() {
            return;
        }
        let mut rng = Rnd::new(seed ^ 0xD151_C71E_5EED_0001);
        let n = node.edges.len();
        let mut noise = Vec::with_capacity(n);
        let mut sum = 0.0f32;
        for _ in 0..n {
            let g = gamma_sample(&mut rng, DIRICHLET_ALPHA);
            sum += g;
            noise.push(g);
        }
        for (i, e) in node.edges.iter_mut().enumerate() {
            let d = noise[i] / sum.max(1e-12);
            e.p = (1.0 - DIRICHLET_EPS) * e.p + DIRICHLET_EPS * d;
        }
    }
}

/// 新节点骨架（未展开）。turn/terminal/terminal_value 从局面直接读出。
fn new_node(st: &FastState) -> Node {
    let turn = st.turn;
    let terminal = st.ended;
    let terminal_value = if terminal {
        st.result().map(|r| r.value_for(turn as usize) as f32).unwrap_or(0.0)
    } else {
        0.0
    };
    Node {
        turn,
        terminal,
        terminal_value,
        expanded: false,
        n: 0,
        w: 0.0,
        edges: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// 确定性随机数工具：Dirichlet 抽样（Marsaglia-Tsang Gamma + 归一化）
// ---------------------------------------------------------------------------

/// U(0,1)（避开 0/1 端点，防 log(0)）。
#[inline]
fn uniform01(rng: &mut Rnd) -> f32 {
    (rng.next_u64() as f32 / u64::MAX as f32).clamp(f32::MIN_POSITIVE, 1.0 - 1e-7)
}

/// 标准正态（Box-Muller）。
fn normal01(rng: &mut Rnd) -> f32 {
    let u1 = uniform01(rng);
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Gamma(α, 1) 抽样（Marsaglia-Tsang；α < 1 用 boost 乘法技巧）。
fn gamma_sample(rng: &mut Rnd, alpha: f32) -> f32 {
    if alpha < 1.0 {
        let u = uniform01(rng);
        return gamma_sample(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal01(rng);
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        v = v * v * v;
        let u = uniform01(rng);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use azul_core::fast::FastPlayer;
    use azul_core::types::{Dest, Move, Source, NUM_COLOURS};

    fn random_net(seed: u64) -> PvNet {
        let mut net = PvNet::new();
        net.init_random(seed);
        net
    }

    /// 空白中期局面（bag 满，测试用局面不会触发真实补砖以外的分支）。
    fn base_state() -> FastState {
        let mut st = FastState::new(0x5EED);
        st.factories = [0; NUM_FACTORIES];
        st.centre = 0;
        st.marker_in_centre = true;
        st.marker_owner = 1;
        st.turn = 0;
        st.ended = false;
        st.players = [FastPlayer::new(); 2];
        st
    }

    /// 与 azul-search/src/search/tests.rs 的 win_in_two_position 同构：
    /// 玩家 0 走"中央 colour-4 -> 行 0"完成图案行，对手唯一回着后墙铺砖
    /// 完成墙行 0 -> 游戏结束，玩家 0 获胜。
    fn win_in_two_position() -> FastState {
        let mut st = base_state();
        st.players[0].wall = 0b0_1111; // 行 0 已放 colour 0,1,2,3；缺 4
        st.players[0].score = 10;
        st.players[1].score = 8;
        st.centre = 1u32 << (4 * NUM_COLOURS); // 1x colour 4
        st.factories[0] = 2u32 << (1 * 3); // 2x colour 1
        st
    }

    /// 镜像局面：玩家 1 即将完成墙行 0，玩家 0 无论如何都会输。
    fn forced_loss_position() -> FastState {
        let mut st = base_state();
        st.players[1].wall = 0b0_1111;
        st.players[1].set_line(0, Some(4), 1); // 补砖后墙铺 colour 4 -> 完成行 0
        st.players[1].score = 30;
        st.players[0].score = 5;
        st.centre = 1u32 << (4 * NUM_COLOURS);
        st.factories[0] = 2u32 << (1 * 3);
        st
    }

    /// 视角回归（bug #4）：MCTS 必须看到"一步即胜"——选中赢棋着法且根
    /// value > 0.5。若备份符号错误，赢棋会被读成大负值。
    #[test]
    fn mcts_takes_immediate_win() {
        let st = win_in_two_position();
        let net = random_net(1);
        let cfg = MctsConfig { sims: 64, dirichlet: false, ..Default::default() };
        let mut mcts = Mcts::new(&net, cfg);
        let policy = mcts.search(&st, 0xDEAD);
        let win_idx = action::to_index(&Move { source: Source::Centre, colour: 4, dest: Dest::Row(0) });
        assert!(
            policy[win_idx] > 0.5,
            "MCTS must concentrate on the winning move, got P={} (visited: {:?})",
            policy[win_idx],
            policy
                .iter()
                .enumerate()
                .filter(|(_, &p)| p > 0.0)
                .map(|(a, &p)| (action::from_index(a), p))
                .collect::<Vec<_>>()
        );
        assert!(mcts.root_q() > 0.5, "root value must read the win (>0.5), got {}", mcts.root_q());
    }

    /// 镜像回归：全部着法都输时根 value 必须 < -0.5（符号翻转 bug 会读成
    /// 大正值）。
    #[test]
    fn mcts_avoids_forced_loss() {
        let st = forced_loss_position();
        let net = random_net(1);
        let cfg = MctsConfig { sims: 64, dirichlet: false, ..Default::default() };
        let mut mcts = Mcts::new(&net, cfg);
        let _ = mcts.search(&st, 0xDEAD);
        assert!(mcts.root_q() < -0.5, "root value must read the loss (<-0.5), got {}", mcts.root_q());
    }

    /// 根访问分布只在合法着法上有质量、总和 ≈ 1。
    #[test]
    fn root_policy_legal_only() {
        let st = FastState::new(0x1234);
        let net = random_net(2);
        let cfg = MctsConfig { sims: 32, dirichlet: false, ..Default::default() };
        let mut mcts = Mcts::new(&net, cfg);
        let policy = mcts.search(&st, 7);
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        let legal: Vec<usize> = buf.iter().map(|m| action::to_index(&m)).collect();
        let mut sum = 0.0f32;
        for a in 0..ACTION_DIM {
            if policy[a] > 0.0 {
                assert!(legal.contains(&a), "illegal action {} got mass", a);
            }
            sum += policy[a];
        }
        assert!((sum - 1.0).abs() < 1e-3, "policy mass must be ~1, got {}", sum);
    }

    /// 同 seed 搜索逐字节可复现（含 Dirichlet 噪声路径）。
    #[test]
    fn mcts_deterministic() {
        let st = FastState::new(0xABCDE);
        let net = random_net(3);
        let cfg = MctsConfig { sims: 32, dirichlet: true, ..Default::default() };
        let mut a = Mcts::new(&net, cfg);
        let cfg2 = MctsConfig { sims: 32, dirichlet: true, ..Default::default() };
        let mut b = Mcts::new(&net, cfg2);
        let pa = a.search(&st, 42);
        let pb = b.search(&st, 42);
        assert_eq!(pa, pb);
    }
}
