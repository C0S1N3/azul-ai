//! azul-az: AlphaZero 轨道基础设施（plan stage 5）。
//!
//! policy+value 双头网络（Rust 推理）+ PUCT MCTS + 自对弈数据管线。
//! 训练在 train/train_az.py（PyTorch），推理/自对弈在本 crate。
//!
//! ⚠️ 备份视角规则（worklog bug #4）：`FastState::apply()` 终局不翻转
//! `turn`、补砖后先手标记持有者先走（同一玩家可连走两手），所有"父→子
//! 取反"逻辑只在 turn 改变时取反。详见 mcts.rs 文件头。

pub mod action;
pub mod mcts;
pub mod net;
pub mod replay;
pub mod selfplay;

pub use action::{from_index, to_index, ACTION_DIM};
pub use mcts::{Mcts, MctsConfig};
pub use net::PvNet;
pub use selfplay::{run as selfplay_run, play_game, Sample, SelfplayConfig, RECORD_BYTES};
