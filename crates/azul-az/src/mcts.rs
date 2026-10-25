//! Single-tree MCTS (PUCT) — the search core of the AlphaZero track
//! (plan stage 5).
//!
//! # ⚠️ Perspective flip rule (the lesson of worklog bug #4; never regress)
//!
//! `FastState::apply()` **does not flip `turn`** when the game ends, and after
//! the tile refill the new turn starts with the holder of the first-player
//! marker — **the same player may legally take two moves in a row**. Therefore
//! any "parent → child negation" backup logic (MCTS backup, minimax negamax)
//! **negates only when `turn` changes**, otherwise win/loss signs flip
//! ("I win" is read as -1). See the `child_value` comment in
//! azul-search/src/search.rs. In this file both the backup (`simulate`) and the
//! selection (`select_action`) explicitly compare turns before deciding the
//! sign.
//!
//! # Design
//! - Node statistics: N, W, Q, prior P; children are stored sparsely by 180
//!   indices (only legal moves are expanded).
//! - Selection: `Q + c_puct * P * sqrt(N_parent) / (1 + N)`, with Q always
//!   stored from the **side to move at that node**; converted to the parent's
//!   perspective at selection time (no conversion when the turn is the
//!   same — see the warning above).
//! - Leaf evaluation: take the net value directly (no random rollout — that was
//!   disproven); terminal nodes use the real result
//!   `GameResult::value_for(turn)` (±1).
//! - Refill randomness: when a move triggers the end of a round
//!   (triggers_refill), a deterministic chance seed is used (derived from the
//!   position hash + move identity), starting with a single sample, with a
//!   `chance_samples` parameter left for later. The same (position, move) always
//!   draws the same refill, byte-for-byte reproducible.
//! - Root Dirichlet noise (α=0.3, ε=0.25) is enabled only for self-play
//!   (parameterised by `dirichlet`).
//! - No transposition table (initial scope): transpositions are not merged;
//!   every edge is an independent subtree.

use crate::action::{self, ACTION_DIM};
use crate::net::PvNet;
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::types::{Dest, Move, Source, NUM_FACTORIES, PATTERN_LINES};

/// PUCT exploration constant (AlphaZero paper values).
pub const C_PUCT: f32 = 1.5;
/// Dirichlet noise concentration α.
pub const DIRICHLET_ALPHA: f32 = 0.3;
/// Root noise mixing ratio ε.
pub const DIRICHLET_EPS: f32 = 0.25;
/// Default number of refill samples (starting with a deterministic single
/// sample).
pub const DEFAULT_CHANCE_SAMPLES: u64 = 1;
/// Default simulations per move.
pub const DEFAULT_SIMS: u32 = 160;

// ---------------------------------------------------------------------------
// 补砖确定性：复制 azul-search/src/search.rs 的 triggers_refill / move_identity /
// chance_seed 思路（那边是私有函数，此处独立实现，不动 azul-search）。
// ---------------------------------------------------------------------------

/// Whether move `m` empties the factories + centre, triggering the end of the
/// round and a tile refill (a chance event).
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

/// Stable identity of a move (used to derive the chance seed).
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

/// Derive a deterministic refill seed from (position hash, move identity,
/// sample index). The same (position, move, sample) always yields the same
/// refill — consistent within the tree and reproducible.
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

/// Sentinel for an unassigned child.
const NO_CHILD: usize = usize::MAX;

/// Sparse child edge: action index + prior P + child slot (lazily allocated).
struct Edge {
    action: usize,
    p: f32,
    child: usize,
}

/// Tree node. Q/W are from the **side to move at this node** (negative =
/// worse for the side to move).
struct Node {
    /// Side to move (FastState.turn); terminal nodes keep the last side to move.
    turn: u8,
    /// Whether the game is over.
    terminal: bool,
    /// Terminal value (±1/0 from the side to move's perspective); 0 when
    /// non-terminal.
    terminal_value: f32,
    /// Whether the node is expanded (edges are ready).
    expanded: bool,
    /// Visit count.
    n: u32,
    /// Accumulated value (from this node's side to move).
    w: f32,
    /// Child edges (the prior P is stored on the edges).
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

/// Search configuration. With `dirichlet = true` the root node gets Dirichlet
/// noise (used for self-play).
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub dirichlet: bool,
    /// Number of refill samples (parameterised; with >1 the expectation is taken
    /// over different refill samples of the same move, starting with one).
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

/// Single-tree MCTS searcher. Each `search` rebuilds the root from the new
/// position (no tree reuse, initial scope).
pub struct Mcts<'a> {
    net: &'a PvNet,
    cfg: MctsConfig,
    nodes: Vec<Node>,
}

impl<'a> Mcts<'a> {
    pub fn new(net: &'a PvNet, cfg: MctsConfig) -> Self {
        Mcts { net, cfg, nodes: Vec::new() }
    }

    /// Run `cfg.sims` simulations from `st` and return the root visit
    /// distribution (length 180: N/ΣN for each legal move, 0 for
    /// illegal/unvisited, summing to ≈ 1). `root_seed` is only used for the
    /// root Dirichlet noise (deterministic and reproducible).
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

    /// The root node's Q (estimated value from the root's side to move). Call
    /// after `search`.
    pub fn root_q(&self) -> f32 {
        self.nodes.first().map(|r| r.q()).unwrap_or(0.0)
    }

    /// Root visit distribution -> 180-dim array.
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

    /// One simulation: select -> expand/evaluate leaf -> back up the whole path
    /// with correct perspectives.
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

    /// Pick an edge from the node's children by PUCT
    /// `Q_parent + c_puct * P * sqrt(N_parent) / (1 + N)` and return the action
    /// index. When converting Q to the parent's perspective, the turn-comparison
    /// rule is observed.
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

    /// Expand a node: softmax priors over legal moves (normalised over legal
    /// actions only) + sparse child edges. Children are lazily allocated when
    /// selected (avoiding a net forward pass for every sibling).
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

    /// Root Dirichlet noise: P ← (1-ε)·P + ε·Dir(α), deterministically driven
    /// by root_seed.
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

/// Fresh node skeleton (unexpanded). turn/terminal/terminal_value are read
/// straight from the position.
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

/// U(0,1) (avoiding the 0/1 endpoints to prevent log(0)).
#[inline]
fn uniform01(rng: &mut Rnd) -> f32 {
    (rng.next_u64() as f32 / u64::MAX as f32).clamp(f32::MIN_POSITIVE, 1.0 - 1e-7)
}

/// Standard normal (Box-Muller).
fn normal01(rng: &mut Rnd) -> f32 {
    let u1 = uniform01(rng);
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Gamma(α, 1) sampling (Marsaglia-Tsang; for α < 1 the boost multiplication
/// trick is used).
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

    /// Blank mid-game position (full bag; the test positions avoid any branch
    /// other than real refills).
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

    /// Isomorphic to win_in_two_position in azul-search/src/search/tests.rs:
    /// player 0 plays "centre colour-4 -> row 0" to complete a pattern line,
    /// after the opponent's only reply the wall tiling completes wall row 0 ->
    /// the game ends and player 0 wins.
    fn win_in_two_position() -> FastState {
        let mut st = base_state();
        st.players[0].wall = 0b0_1111; // 行 0 已放 colour 0,1,2,3；缺 4
        st.players[0].score = 10;
        st.players[1].score = 8;
        st.centre = 1u32 << (4 * NUM_COLOURS); // 1x colour 4
        st.factories[0] = 2u32 << (1 * 3); // 2x colour 1
        st
    }

    /// Mirrored position: player 1 is about to complete wall row 0, and player
    /// 0 loses no matter what.
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

    /// Perspective regression (bug #4): MCTS must see the "win in one move" —
    /// it must select the winning move and have root value > 0.5. If the backup
    /// sign is wrong, the win is read as a large negative value.
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

    /// Mirrored regression: when every move loses, the root value must be
    /// < -0.5 (a sign-flip bug would read it as a large positive value).
    #[test]
    fn mcts_avoids_forced_loss() {
        let st = forced_loss_position();
        let net = random_net(1);
        let cfg = MctsConfig { sims: 64, dirichlet: false, ..Default::default() };
        let mut mcts = Mcts::new(&net, cfg);
        let _ = mcts.search(&st, 0xDEAD);
        assert!(mcts.root_q() < -0.5, "root value must read the loss (<-0.5), got {}", mcts.root_q());
    }

    /// The root visit distribution has mass only on legal moves and sums to ≈ 1.
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

    /// Search with the same seed is byte-for-byte reproducible (including the
    /// Dirichlet noise path).
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
