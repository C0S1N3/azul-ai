//! azul-core: Azul (2-player) rules engine.
//!
//! Dual implementation per plan stage 0:
//! - `slow`: clarity-first reference engine (source of truth)
//! - `fast`: bitpacked engine (later), verified against `slow` via differential tests
//!
//! Rule authority: official Plan B Games rulebook (en-azul-rules.pdf),
//! conclusions recorded in docs/worklog.md and the plan (stage 0 §5).

pub mod types;
pub mod rng;
pub mod slow;
pub mod fast;

pub use types::*;
