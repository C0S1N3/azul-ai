//! azul-cli: play / duel / gen-data / serve / perft commands.

mod duel;
mod gen_data;

use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_az::net::PvNet;
use std::time::Instant;

/// 载入 policy+value 网。`path` 不存在或为 "random" 时用确定性随机初始化
/// （seed 驱动），便于管线 smoke 与无训练基线对局。
fn load_pv_net(path: &str, seed: u64) -> PvNet {
    if path == "random" {
        let mut net = PvNet::new();
        net.init_random(seed ^ 0x9E37_79B9);
        return net;
    }
    match std::fs::read(path) {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    match cmd {
        "perft" => {
            let iters: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
            let mut st = FastState::new(0xC0FFEE);
            let mut buf = MoveBuf::new();
            let mut rng = Rnd::new(0xDEAD);
            let mut generated = 0u64;
            let start = Instant::now();
            for _ in 0..iters {
                if st.ended {
                    st = FastState::new(rng.next_u64());
                }
                st.legal_into(&mut buf);
                generated += buf.len as u64;
                let m = buf.moves[rng.next_u64() as usize % buf.len].inner.unwrap();
                st.apply(m, &mut Rnd::new(rng.next_u64()));
            }
            let elapsed = start.elapsed();
            let gen_per_sec = generated as f64 / elapsed.as_secs_f64();
            let apply_per_sec = iters as f64 / elapsed.as_secs_f64();
            println!("perft: {} applies in {:.3}s", iters, elapsed.as_secs_f64());
            println!("  move generation: {:>12.0} moves/sec", gen_per_sec);
            println!("  apply:           {:>12.0} moves/sec", apply_per_sec);
            println!("  acceptance >= 10k/s: {}", if apply_per_sec > 10_000.0 { "PASS" } else { "FAIL" });
        }
        "gen-data" => {
            // usage: azul gen-data <games> <agent> <depth> <out> [seed] [target_mode] [label_depth]
            //   target_mode: outcome (default) | search
            let games: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            let agent = args.get(3).cloned().unwrap_or_else(|| "h11".to_string());
            let depth: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
            let out = args.get(5).cloned().unwrap_or_else(|| "data.bin".to_string());
            let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);
            let target_mode = args.get(7).cloned().unwrap_or_else(|| "outcome".to_string());
            let label_depth: u32 = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(5);
            let nnue_path = args.get(9).cloned().unwrap_or_else(|| "out/nnue.bin".to_string());
            gen_data::run(&gen_data::GenConfig {
                games,
                base_seed: seed,
                agent,
                depth,
                out,
                target_mode,
                label_depth,
                nnue_path,
            });
        }
        "duel" => {
            // usage: azul duel <agentA> <agentB> <games> [seed] [depthA] [depthB] [nnue_path]
            // agents: random | greedy | h0 | h11 | nnue
            let a_name = args.get(2).cloned().unwrap_or_else(|| "h11".to_string());
            let b_name = args.get(3).cloned().unwrap_or_else(|| "h0".to_string());
            let games: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
            let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0x1234_5678);
            let depth_a: u32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2);
            let depth_b: u32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(2);
            let nnue_path = args.get(8).cloned().unwrap_or_else(|| "out/nnue.bin".to_string());
            let nnue_path_b = args.get(9).cloned();
            duel::run(&duel::DuelConfig {
                a: a_name,
                b: b_name,
                games,
                seed,
                depth_a,
                depth_b,
                nnue_path,
                nnue_path_b,
            });
        }
        "az-selfplay" => {
            // usage: azul az-selfplay <games> <out> <net_path> [sims] [seed] [temp_cutoff]
            // net_path 不存在或为 "random" 时用确定性随机初始化网络。
            let games: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
            let out = args.get(3).cloned().unwrap_or_else(|| "out/az_selfplay.bin".to_string());
            let net_path = args.get(4).cloned().unwrap_or_else(|| "random".to_string());
            let sims: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(azul_az::mcts::DEFAULT_SIMS);
            let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0xA20C_0FFEE);
            let temp_cutoff: u32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(8);
            let net = load_pv_net(&net_path, seed);
            azul_az::selfplay_run(
                &net,
                &azul_az::SelfplayConfig {
                    games,
                    base_seed: seed,
                    sims,
                    temp_cutoff,
                    out,
                    dirichlet: true,
                },
            );
        }
        _ => {
            println!("azul CLI");
            println!("usage: azul <command> [args]");
            println!("  perft [iters]                     move-gen & apply speed benchmark");
            println!("  gen-data <games> <agent> <depth> <out> [seed]");
            println!("  duel <agentA> <agentB> <games> [seed] [depthA] [depthB] [nnue]");
            println!("       agents: random | greedy | h0 | h11 | nnue | az[:<sims>]");
            println!("  az-selfplay <games> <out> <net_path> [sims] [seed] [temp_cutoff]");
            println!("       net_path= random | <file>（缺失时确定性随机初始化）");
        }
    }
}
