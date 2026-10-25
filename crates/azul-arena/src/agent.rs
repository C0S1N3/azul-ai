//! Agents for the anchor pool (plan stage 1): Random, Greedy (h0-greedy one
//! ply), and Minimax(h0) at various depths.

use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::types::Move;
use azul_search::eval::{Evaluator, Heuristic0};
use azul_search::Searcher;

/// An agent chooses a move for the side to move.
pub trait Agent {
    fn choose(&mut self, st: &FastState, rng: &mut Rnd) -> Move;
    fn name(&self) -> String;
}

/// Uniform random legal mover (baseline anchor).
pub struct RandomAgent;

impl Agent for RandomAgent {
    fn choose(&mut self, st: &FastState, rng: &mut Rnd) -> Move {
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        buf.moves[rng.next_u64() as usize % buf.len].inner.unwrap()
    }
    fn name(&self) -> String {
        "random".to_string()
    }
}

/// Greedy: pick the move maximising the 1-ply h0 evaluation.
pub struct GreedyAgent {
    ev: Heuristic0,
}

impl GreedyAgent {
    pub fn new() -> Self {
        GreedyAgent { ev: Heuristic0 }
    }
}

impl Agent for GreedyAgent {
    fn choose(&mut self, st: &FastState, rng: &mut Rnd) -> Move {
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        let mut best = None;
        let mut best_val = f32::MIN;
        for i in 0..buf.len {
            let m = buf.moves[i].inner.unwrap();
            let mut next = *st;
            next.apply(m, &mut Rnd::new(rng.next_u64()));
            // After our move the opponent is to move; negate their advantage.
            let val = if next.ended {
                match next.result() {
                    Some(r) => (r.value_for(st.turn as usize)) as f32 * 1000.0,
                    None => 0.0,
                }
            } else {
                -self.ev.eval(&next)
            };
            if val > best_val {
                best_val = val;
                best = Some(m);
            }
        }
        best.unwrap()
    }
    fn name(&self) -> String {
        "greedy".to_string()
    }
}

/// Minimax agent with any evaluator at a fixed depth.
pub struct MinimaxAgent<E: Evaluator> {
    searcher: Searcher,
    depth: u32,
    ev: E,
    name: String,
}

impl<E: Evaluator> MinimaxAgent<E> {
    pub fn new(depth: u32, ev: E, name: &str) -> Self {
        MinimaxAgent { searcher: Searcher::new(), depth, ev, name: name.to_string() }
    }

    /// Override the number of bag-refill samples averaged at chance nodes.
    pub fn with_branch_samples(mut self, n: u64) -> Self {
        self.searcher.branch_samples = n;
        self
    }

    /// Use greedy rollout to true terminal score at search leaves.
    pub fn with_rollout(mut self) -> Self {
        self.searcher.rollout = true;
        self
    }
}

impl<E: Evaluator> Agent for MinimaxAgent<E> {
    fn choose(&mut self, st: &FastState, rng: &mut Rnd) -> Move {
        self.searcher.think(st, self.depth, &self.ev, rng).0
    }
    fn name(&self) -> String {
        format!("{}-d{}", self.name, self.depth)
    }
}
