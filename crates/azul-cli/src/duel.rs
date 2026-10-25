//! Arena duel command (plan stage 2/6): pit two named agents against each
//! other with paired seeds and report win/loss + Elo delta.

use azul_arena::agent::{Agent, GreedyAgent, MinimaxAgent, RandomAgent};
use azul_arena::elo::duel;
use azul_az::mcts::{Mcts, MctsConfig, DEFAULT_SIMS};
use azul_az::net::PvNet;
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::types::Move;
use azul_nnue::net::Mlp;
use azul_nnue::{HybridEvaluator, NnueEvaluator};
use azul_search::eval::{Heuristic0, Heuristic11};
use std::fs;

/// AlphaZero agent：policy+value 网 + MCTS（PUCT），每手跑 `sims` 次模拟，
/// 取根访问分布的 argmax（对局用途不加 Dirichlet 噪声，走子确定）。
struct AzAgent {
    net: PvNet,
    sims: u32,
}

impl AzAgent {
    fn new(net: PvNet, sims: u32) -> Self {
        AzAgent { net, sims }
    }
}

impl Agent for AzAgent {
    fn choose(&mut self, st: &FastState, rng: &mut Rnd) -> Move {
        let cfg = MctsConfig { sims: self.sims, dirichlet: false, ..Default::default() };
        let mut mcts = Mcts::new(&self.net, cfg);
        let policy = mcts.search(st, rng.next_u64());
        // argmax（平局取最小索引）；兜底：非法则取首个合法着法。
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        let legal: Vec<(usize, Move)> = buf.iter().map(|m| (azul_az::to_index(&m), m)).collect();
        let best = legal
            .iter()
            .max_by(|a, b| policy[a.0].total_cmp(&policy[b.0]))
            .map(|(_, m)| *m)
            .unwrap_or_else(|| legal.first().expect("legal move").1);
        best
    }
    fn name(&self) -> String {
        format!("az-s{}", self.sims)
    }
}

/// 载入 policy+value 网："random" 或文件缺失时用确定性随机初始化。
fn load_pv_net(path: &str, seed: u64) -> PvNet {
    if path == "random" {
        let mut net = PvNet::new();
        net.init_random(seed ^ 0x9E37_79B9);
        return net;
    }
    match fs::read(path) {
        Ok(bytes) => PvNet::from_bytes(&bytes).unwrap_or_else(|| {
            panic!("cannot parse pv net at {}", path)
        }),
        Err(_) => {
            let mut net = PvNet::new();
            net.init_random(seed ^ 0x9E37_79B9);
            net
        }
    }
}

pub struct DuelConfig {
    pub a: String,
    pub b: String,
    pub games: u32,
    pub seed: u64,
    pub depth_a: u32,
    pub depth_b: u32,
    pub nnue_path: String,
    /// Optional separate weights path for agent B (for nnue-vs-nnue duels).
    pub nnue_path_b: Option<String>,
}

fn make_agent(name: &str, depth: u32, nnue_path: &str) -> Box<dyn Agent> {
    match name {
        "random" => Box::new(RandomAgent),
        "greedy" => Box::new(GreedyAgent::new()),
        "h0" => Box::new(MinimaxAgent::new(depth, Heuristic0, "h0")),
        "h11" => Box::new(MinimaxAgent::new(depth, Heuristic11::tuned(), "h11")),
        "nnue" => {
            let bytes = fs::read(nnue_path).unwrap_or_else(|e| {
                panic!("cannot read nnue weights at {}: {}", nnue_path, e)
            });
            let net = Mlp::from_bytes(&bytes).expect("parse nnue weights");
            // scale=25 matches the search-value/25 target mapping used in training.
            Box::new(MinimaxAgent::new(depth, NnueEvaluator::new(net, 25.0), "nnue"))
        }
        n if n.starts_with("hybrid") => {
            // "hybrid[:alpha][r]" — h0 at full strength + alpha * nnue
            // correction; trailing 'r' enables greedy-rollout leaves.
            let rollout = n.ends_with('r');
            let core = n.strip_suffix('r').unwrap_or(n);
            let alpha: f32 = core.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let bytes = fs::read(nnue_path).unwrap_or_else(|e| {
                panic!("cannot read nnue weights at {}: {}", nnue_path, e)
            });
            let net = Mlp::from_bytes(&bytes).expect("parse nnue weights");
            let agent = MinimaxAgent::new(depth, HybridEvaluator::new(net, 25.0, alpha), "hybrid");
            Box::new(if rollout { agent.with_rollout() } else { agent })
        }
        n if n == "az" || n.starts_with("az:") => {
            // "az" | "az:<sims>" — policy+value 网 + MCTS；sims 默认 160。
            // 网络权重走 nnue_path 参数（"random" 或缺失 = 确定性随机初始化）。
            let sims: u32 = n.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SIMS);
            let net = load_pv_net(nnue_path, 0xA20C_0FFEE);
            Box::new(AzAgent::new(net, sims))
        }
        other => panic!("unknown agent '{}'", other),
    }
}

pub fn run(cfg: &DuelConfig) {
    let mut a = make_agent(&cfg.a, cfg.depth_a, &cfg.nnue_path);
    let mut b = make_agent(&cfg.b, cfg.depth_b, &cfg.nnue_path_b.as_deref().unwrap_or(&cfg.nnue_path));
    let res = duel(a.as_mut(), b.as_mut(), cfg.games, cfg.seed);
    println!(
        "duel {}(d{}) vs {}(d{}): {}-{}-{} over {} games (score {:.3}, elo {:+.0})",
        cfg.a,
        cfg.depth_a,
        cfg.b,
        cfg.depth_b,
        res.wins_a,
        res.wins_b,
        res.draws,
        cfg.games * 2,
        res.score_a(),
        res.elo_delta()
    );
}
