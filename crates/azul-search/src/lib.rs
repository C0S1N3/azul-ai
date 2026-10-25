//! azul-search: minimax (PVS + MTDF + TT + iterative deepening + branching).
//! Stage 1 of the plan.

pub mod eval;
pub mod search;
pub mod tt;

pub use eval::{Eval, Evaluator, Heuristic0, Heuristic11, H11Weights};
pub use search::Searcher;
