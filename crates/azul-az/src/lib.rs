//! azul-az: AlphaZero pipeline infrastructure (plan stage 5).
//!
//! Policy+value two-headed net (Rust inference) + PUCT MCTS + self-play data
//! pipeline. Training lives in train/train_az.py (PyTorch); inference and
//! self-play live in this crate.
//!
//! ⚠️ Backup perspective rule (worklog bug #4): `FastState::apply()` does not
//! flip `turn` at game end, and after the tile refill the holder of the
//! first-move marker plays first (the same player may take two moves in a
//! row), so all "parent -> child negation" logic only negates when the turn
//! changes. See the mcts.rs file header for details.

pub mod action;
pub mod mcts;
pub mod net;
pub mod replay;
pub mod selfplay;

pub use action::{from_index, to_index, ACTION_DIM};
pub use mcts::{Mcts, MctsConfig};
pub use net::PvNet;
pub use selfplay::{run as selfplay_run, play_game, Sample, SelfplayConfig, RECORD_BYTES};
