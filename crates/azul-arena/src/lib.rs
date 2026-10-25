//! azul-arena: evaluation harness — Elo anchor pool, duels, gating, reports.
//! Anchor pool v1 (plan stage 1): random / greedy / minimax-h0 at depths.

pub mod agent;
pub mod elo;

pub use agent::{Agent, GreedyAgent, MinimaxAgent, RandomAgent};
pub use elo::{duel, DuelResult};

#[cfg(test)]
mod arena_tests;
