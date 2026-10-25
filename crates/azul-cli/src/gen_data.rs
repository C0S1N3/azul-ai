//! Self-play data generation (plan stage 2 §2.3 / stage 3 step 1).
//! Plays games with a chosen agent, extracts NNUE features at each position
//! (from the side-to-move's perspective) + the eventual game outcome, and
//! writes a binary training set readable by train/train_nnue.py.
//!
//! Binary record format (little-endian), one per position:
//!   [FEATURE_DIM f32 features][1 f32 target]
//! target ∈ {0.0 (loss), 0.5 (draw), 1.0 (win)} from the side-to-move view.
//!
//! Games are independent (seed per game), so generation runs multi-threaded;
//! results are collected in game order, so the output file is byte-identical
//! to a single-threaded run.

use azul_arena::agent::{Agent, MinimaxAgent, RandomAgent, GreedyAgent};
use azul_core::fast::FastState;
use azul_core::rng::Rnd;
use azul_nnue::features::{extract, FEATURE_DIM};
use azul_nnue::{HybridEvaluator, Mlp};
use azul_search::eval::{Heuristic0, Heuristic11};
use azul_search::Searcher;
use std::fs::File;
use std::io::{BufWriter, Write};

pub struct GenConfig {
    pub games: u32,
    pub base_seed: u64,
    pub agent: String,
    pub depth: u32,
    pub out: String,
    /// "outcome" = game result {0,0.5,1}; "search" = minimax search value
    /// distilled into a probability (Rzepecki-style value distillation).
    pub target_mode: String,
    /// Search depth used to produce search-value labels (target_mode=search).
    pub label_depth: u32,
    /// NNUE weights path for hybrid teachers ("hybrid[:alpha]" agent).
    pub nnue_path: String,
}

fn game_seed(base_seed: u64, g: u32) -> u64 {
    base_seed.wrapping_add(g as u64 * 0x9E37_79B9) | 1
}

/// Build a fresh (agent_a, agent_b) pair — searchers accumulate TT state, so
/// each game gets fresh agents.
fn make_agents(cfg: &GenConfig) -> (Box<dyn Agent>, Box<dyn Agent>) {
    match cfg.agent.as_str() {
        "random" => (Box::new(RandomAgent), Box::new(RandomAgent)),
        "greedy" => (Box::new(GreedyAgent::new()), Box::new(GreedyAgent::new())),
        "h0" => (
            Box::new(MinimaxAgent::new(cfg.depth, Heuristic0, "h0")),
            Box::new(MinimaxAgent::new(cfg.depth, Heuristic0, "h0")),
        ),
        n if n.starts_with("hybrid") => {
            // "hybrid[:alpha]" teacher: h0 full strength + nnue correction.
            let alpha: f32 = n.split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let make = || {
                let bytes = std::fs::read(&cfg.nnue_path).expect("read nnue weights");
                let net = Mlp::from_bytes(&bytes).expect("parse nnue weights");
                Box::new(MinimaxAgent::new(cfg.depth, HybridEvaluator::new(net, 25.0, alpha), "hybrid"))
                    as Box<dyn Agent>
            };
            (make(), make())
        }
        _ => (
            Box::new(MinimaxAgent::new(cfg.depth, Heuristic11::tuned(), "h11")),
            Box::new(MinimaxAgent::new(cfg.depth, Heuristic11::tuned(), "h11")),
        ),
    }
}

/// Label evaluator for search-value modes: the SAME evaluator as the game
/// agents (hybrid with the given net, or h11 otherwise), so distilled values
/// reflect the strongest available search.
fn make_label_ev(cfg: &GenConfig) -> Box<dyn azul_search::eval::Evaluator> {
    if cfg.agent.starts_with("hybrid") {
        let alpha: f32 = cfg.agent.strip_suffix('r').unwrap_or(&cfg.agent)
            .split(':').nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let bytes = std::fs::read(&cfg.nnue_path).expect("read nnue weights");
        let net = Mlp::from_bytes(&bytes).expect("parse nnue weights");
        Box::new(HybridEvaluator::new(net, 25.0, alpha))
    } else {
        Box::new(Heuristic11::tuned())
    }
}

/// Play one game and return its raw record bytes.
fn play_game(cfg: &GenConfig, seed: u64) -> Vec<u8> {
    let (mut agent_a, mut agent_b) = make_agents(cfg);
    let mut st = FastState::new(seed);
    let mut game_rng = Rnd::new(seed ^ 0x5A17);
    // Label producer for search-value modes: a fresh deep searcher.
    let mut labeler = Searcher::new();
    let label_ev = make_label_ev(cfg);
    let mut out = Vec::new();
    let mut positions: Vec<([f32; FEATURE_DIM], usize)> = Vec::new();

    while !st.ended {
        let feat = extract(&st);
        if cfg.target_mode == "search" {
            // Distill (legacy, sigmoid-compressed): target = sigmoid(value/25).
            let (_m, value) = labeler.think(&st, cfg.label_depth, &*label_ev, &mut game_rng);
            let target = 1.0 / (1.0 + (-(value / 25.0)).exp());
            for v in feat.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&target.to_le_bytes());
        } else if cfg.target_mode == "searchreg" {
            // Distill, LINEAR map: keeps the full dynamic range of search
            // values (score units) in the target — the sigmoid/25 compression
            // squashed target std to 0.11 and was falsified for that reason.
            // target = (value/50 + 1) / 2, clamped to [0,1].
            let (_m, value) = labeler.think(&st, cfg.label_depth, &*label_ev, &mut game_rng);
            let target = ((value / 50.0 + 1.0) / 2.0).clamp(0.0, 1.0);
            for v in feat.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&target.to_le_bytes());
        } else {
            positions.push((feat, st.turn as usize));
        }
        let m = if st.turn == 0 {
            agent_a.choose(&st, &mut game_rng)
        } else {
            agent_b.choose(&st, &mut game_rng)
        };
        st.apply(m, &mut Rnd::new(game_rng.next_u64()));
    }

    // Outcome mode: resolve targets from each position's side-to-move view.
    if cfg.target_mode != "search" && cfg.target_mode != "searchreg" {
        let result = st.result();
        for (feat, stm) in &positions {
            let target = match result {
                Some(r) => (r.value_for(*stm) as f32 + 1.0) / 2.0, // {-1,0,1} -> {0,0.5,1}
                None => 0.5,
            };
            for v in feat.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&target.to_le_bytes());
        }
    }
    out
}

pub fn run(cfg: &GenConfig) {
    // Create the parent directory if needed (e.g. data/sample.bin).
    if let Some(parent) = std::path::Path::new(&cfg.out).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let games = cfg.games as usize;
    let chunk = games.div_ceil(threads);

    let chunks: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..games)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(games);
                let cfg = cfg;
                scope.spawn(move || {
                    let mut bytes = Vec::new();
                    for g in start..end {
                        bytes.extend_from_slice(&play_game(cfg, game_seed(cfg.base_seed, g as u32)));
                    }
                    (start, bytes)
                })
            })
            .collect();
        let mut results: Vec<(usize, Vec<u8>)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        results.sort_by_key(|(start, _)| *start);
        results.into_iter().map(|(_, b)| b).collect()
    });

    let mut writer = BufWriter::new(File::create(&cfg.out).expect("create dataset"));
    let mut total_bytes = 0usize;
    for c in &chunks {
        writer.write_all(c).unwrap();
        total_bytes += c.len();
    }
    writer.flush().unwrap();
    let rec = (FEATURE_DIM + 1) * 4;
    println!(
        "gen-data: wrote {} samples from {} games to {} (agent={} depth={} threads={})",
        total_bytes / rec, cfg.games, cfg.out, cfg.agent, cfg.depth, threads
    );
}
