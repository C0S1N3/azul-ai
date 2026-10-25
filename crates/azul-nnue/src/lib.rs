//! azul-nnue: NNUE value-net features + inference (stage 2 of the plan).
//! Trained via train/ (PyTorch); this crate does inference (f32 MLP now,
//! int8 quantised + SIMD later).

pub mod features;
pub mod net;

pub use features::{extract, FEATURE_DIM};
pub use net::{HybridEvaluator, Mlp, NnueEvaluator};

#[cfg(test)]
mod net_tests;
