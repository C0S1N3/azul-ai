//! 自对弈生成训练数据（plan stage 5）。
//!
//! 每局：双方共用同一 policy+value 网，各自每手跑 MCTS（带根 Dirichlet 噪声），
//! 按根访问分布走子（温度 τ：前 `temp_cutoff` 手 τ=1 抽样，之后 argmax）。
//!
//! 数据格式（小端），每个位置一条记录：
//!   [171 f32 obs][180 f32 policy target（根访问分布归一化）]
//!   [1 f32 value target（该位置待走方的终局 {-1,0,1}）]
//! 共 (171 + 180 + 1) × 4 = 1408 字节/条。
//!
//! 并行模型与 azul-cli/src/gen_data.rs 一致：对局级并行（std::thread::scope
//! 按局分片、按局序合并），每局 seed 由 base_seed 派生，输出逐字节确定。

use crate::action::ACTION_DIM;
use crate::mcts::{Mcts, MctsConfig, DEFAULT_SIMS};
use crate::net::PvNet;
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_nnue::features::FEATURE_DIM;
use std::fs::File;
use std::io::{BufWriter, Write};

/// 每条记录字节数：obs + policy + value。
pub const RECORD_BYTES: usize = (FEATURE_DIM + ACTION_DIM + 1) * 4;
/// value target 的取值集合 {-1, 0, 1}（待走方视角）。
pub const VALUE_SCALE: f32 = 1.0;

pub struct SelfplayConfig {
    pub games: u32,
    pub base_seed: u64,
    /// 每手 MCTS 模拟数。
    pub sims: u32,
    /// 前 temp_cutoff 手 τ=1 抽样（按根访问分布），之后 argmax。
    pub temp_cutoff: u32,
    pub out: String,
    /// Dirichlet 根噪声（自对弈探索）。1 = 开，0 = 关。
    pub dirichlet: bool,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        SelfplayConfig {
            games: 1,
            base_seed: 0,
            sims: DEFAULT_SIMS,
            temp_cutoff: 8,
            out: "out/az_selfplay.bin".to_string(),
            dirichlet: true,
        }
    }
}

/// 每条记录 = 一个位置：obs、根访问分布、终局价值（待走方视角）。
pub struct Sample {
    pub obs: [f32; FEATURE_DIM],
    pub policy: [f32; ACTION_DIM],
    pub value: f32,
}

impl Sample {
    pub fn write(&self, out: &mut Vec<u8>) {
        for v in self.obs.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.policy.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.value.to_le_bytes());
    }
}

/// 游戏级 seed（与 gen_data::game_seed 同配方，保持全库一致）。
fn game_seed(base_seed: u64, g: u32) -> u64 {
    base_seed.wrapping_add(g as u64 * 0x9E37_79B9) | 1
}

/// 从根访问分布选着法：前 temp_cutoff 手 τ=1 按分布抽样，之后 argmax。
/// `rng` 决定抽样（确定性可复现）。
fn pick_move(policy: &[f32; ACTION_DIM], rng: &mut Rnd, temp_cutoff: u32, ply: u32) -> usize {
    if ply < temp_cutoff {
        // τ=1：按访问分布抽样。
        let mut sum = 0.0f32;
        for &p in policy.iter() {
            sum += p;
        }
        if sum <= 0.0 {
            return 0;
        }
        let mut u = (rng.next_u64() as f32 / u64::MAX as f32) * sum;
        for (a, &p) in policy.iter().enumerate() {
            u -= p;
            if u <= 0.0 {
                return a;
            }
        }
        ACTION_DIM - 1
    } else {
        // argmax（平局取最小索引，确定性）。
        let mut best = 0;
        let mut best_p = f32::NEG_INFINITY;
        for (a, &p) in policy.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = a;
            }
        }
        best
    }
}

/// 确保抽样出的动作是合法着法：policy 只在合法着法上有质量，但浮点抽样
/// 可能落到 0 质量非法位；保险起见把非法结果夹回首个合法着法。
fn legal_indices(st: &FastState) -> Vec<usize> {
    let mut buf = MoveBuf::new();
    st.legal_into(&mut buf);
    buf.iter().map(|m| crate::action::to_index(&m)).collect()
}

/// 打一局自对弈，返回全部位置样本（含终局回填 value target）。
pub fn play_game(net: &PvNet, cfg: &SelfplayConfig, seed: u64) -> Vec<Sample> {
    let mut st = FastState::new(seed);
    let mut game_rng = Rnd::new(seed ^ 0x5A17);
    let mcts_cfg = MctsConfig {
        sims: cfg.sims,
        dirichlet: cfg.dirichlet,
        ..Default::default()
    };
    let mut mcts = Mcts::new(net, mcts_cfg);
    // 位置记录：(obs, policy target, 待走方)。value 在终局回填。
    let mut positions: Vec<(Sample, usize)> = Vec::new();
    let mut ply = 0u32;

    while !st.ended {
        let obs = azul_nnue::features::extract(&st);
        let policy = mcts.search(&st, game_rng.next_u64());
        let stm = st.turn as usize;
        positions.push((Sample { obs, policy, value: 0.0 }, stm));

        let legal = legal_indices(&st);
        debug_assert!(!legal.is_empty(), "no legal moves in a live position");
        let mut action = pick_move(&policy, &mut game_rng, cfg.temp_cutoff, ply);
        if !legal.contains(&action) {
            // 数值兜底：退到访问最多的合法着法。
            action = *legal
                .iter()
                .max_by(|&&a, &&b| policy[a].total_cmp(&policy[b]))
                .expect("legal moves");
        }
        let m = crate::action::from_index(action);
        st.apply(m, &mut Rnd::new(game_rng.next_u64()));
        ply += 1;
    }

    // 终局回填：value target = 该位置待走方的终局 {-1,0,1}。
    let result = st.result();
    for (sample, stm) in positions.iter_mut() {
        sample.value = match result {
            Some(r) => r.value_for(*stm) as f32 * VALUE_SCALE,
            None => 0.0,
        };
    }
    positions.into_iter().map(|(s, _)| s).collect()
}

/// 生成 `cfg.games` 局自对弈数据写入 `cfg.out`（创建父目录）。
/// 对局级并行、按局序合并 —— 输出逐字节确定。返回写入的样本条数。
pub fn run(net: &PvNet, cfg: &SelfplayConfig) -> usize {
    if let Some(parent) = std::path::Path::new(&cfg.out).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let games = cfg.games as usize;
    let chunk = games.div_ceil(threads).max(1);

    let chunks: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..games)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(games);
                let cfg = cfg;
                let net = net;
                scope.spawn(move || {
                    let mut bytes = Vec::new();
                    for g in start..end {
                        let samples = play_game(net, cfg, game_seed(cfg.base_seed, g as u32));
                        for s in &samples {
                            s.write(&mut bytes);
                        }
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
    let samples = total_bytes / RECORD_BYTES;
    println!(
        "az-selfplay: wrote {} samples from {} games to {} (sims={} temp_cutoff={} threads={})",
        samples, cfg.games, cfg.out, cfg.sims, cfg.temp_cutoff, threads
    );
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 自对弈 smoke：2 局跑得完、数据条数 = 位置数、记录字节数对齐。
    #[test]
    fn selfplay_smoke_two_games() {
        let mut net = PvNet::new();
        net.init_random(0xA2);
        let cfg = SelfplayConfig {
            games: 2,
            base_seed: 123,
            sims: 8,
            temp_cutoff: 4,
            out: "target/az_test_smoke.bin".to_string(),
            dirichlet: true,
        };
        let samples = run(&net, &cfg);
        assert!(samples > 0, "must produce samples");
        // 文件字节数 = 条数 × 记录大小。
        let meta = std::fs::metadata(&cfg.out).expect("output file exists");
        assert_eq!(meta.len() as usize, samples * RECORD_BYTES);

        // 逐局独立打一遍，条数必须相加吻合（确定性 + 位置计数一致）。
        let mut expected = 0usize;
        for g in 0..2u32 {
            let seed = game_seed(cfg.base_seed, g);
            expected += play_game(&net, &cfg, seed).len();
        }
        assert_eq!(samples, expected, "sample count must equal total positions");
    }

    /// value target 只能取 {-1, 0, 1}，且每条 obs/policy 维度正确。
    #[test]
    fn selfplay_targets_sane() {
        let mut net = PvNet::new();
        net.init_random(0xB3);
        let cfg = SelfplayConfig { sims: 4, temp_cutoff: 2, ..Default::default() };
        let samples = play_game(&net, &cfg, 99);
        assert!(!samples.is_empty());
        for s in &samples {
            assert!(s.value == -1.0 || s.value == 0.0 || s.value == 1.0, "bad value {}", s.value);
            let psum: f32 = s.policy.iter().sum();
            assert!((psum - 1.0).abs() < 1e-3 || psum == 0.0, "policy sum {}", psum);
        }
    }
}
