//! Minimax search (plan stage 1 §1.1): negamax + alpha-beta + transposition
//! table (hash move) + iterative deepening. PVS/MTDF and branching for bag
//! randomness land in stage 1 §1.2 refinements.

use crate::eval::Evaluator;
use crate::tt::{TtFlag, TranspositionTable};
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::types::{Dest, Move, Source, NUM_FACTORIES, PATTERN_LINES};

const INF: f32 = 1e9;
/// Score assigned to a decisive terminal win — dominates heuristic values.
pub const TERMINAL_WIN: f32 = 1000.0;

/// Stable identity for a move — used to derive chance seeds so the SAME move
/// in the SAME position always draws the same tiles (TT-consistent), while
/// different moves/positions vary (no overfitting to one refill sequence).
#[inline]
fn move_identity(m: &Move) -> u64 {
    let s = match m.source {
        Source::Factory(f) => f as u64,
        Source::Centre => NUM_FACTORIES as u64,
    };
    let d = match m.dest {
        Dest::Row(r) => r as u64,
        Dest::Floor => PATTERN_LINES as u64,
    };
    (s << 16) | ((m.colour as u64) << 8) | d
}

#[inline]
fn chance_seed(pos_key: u64, m: &Move, sample: u64) -> u64 {
    pos_key
        .wrapping_add(move_identity(m).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(sample.wrapping_mul(0x853C_49E6_748F_EA9B))
        .wrapping_add(0xA5A5_1234_5678_9ABC)
}

/// Default number of bag-refill samples averaged at chance nodes (plan §1.2
/// branching). Rzepecki: branchingFactor=1 was best WITH a strong NNUE
/// evaluator; with a weaker heuristic the trade-off may differ, so this is a
/// runtime knob (`Searcher::branch_samples`) swept by the depth-trend harness.
pub const DEFAULT_BRANCH_SAMPLES: u64 = 1;

/// True iff playing `m` empties factories+centre, ending the round and
/// triggering the bag refill (the chance event branching averages over).
/// A move removes exactly the taken tiles from play (factory rejects go to
/// centre), so the round ends iff tiles in play == tiles taken.
fn triggers_refill(st: &FastState, m: &Move) -> bool {
    let mut in_play = 0u32;
    for f in 0..NUM_FACTORIES {
        for c in 0..5u8 {
            in_play += st.factory_count(f, c) as u32;
        }
    }
    for c in 0..5u8 {
        in_play += st.centre_count(c) as u32;
    }
    let taken = match m.source {
        Source::Factory(f) => st.factory_count(f, m.colour),
        Source::Centre => st.centre_count(m.colour),
    } as u32;
    in_play == taken
}

/// Greedy rollout to a TRUE terminal score difference (plan §1.2 option c:
/// "roll out to true value before terminal"). Kills leaf-evaluation noise at
/// the frontier: instead of a heuristic snapshot, each leaf plays out greedily
/// and reports the actual final score gap in points. The guiding policy is the
/// CHEAP h0 settle heuristic (not `ev` — NNUE-forward-per-step is ~100x cost);
/// only the terminal value is exact. Deterministic (seeds derived from
/// position hash) so TT-storable.
fn rollout_value(st: &FastState, _ev: &dyn Evaluator) -> f32 {
    use crate::eval::{settle_score, Heuristic0};
    use crate::eval::Evaluator;
    const ROLLOUT_PLY_CAP: u32 = 120;
    let me = st.turn as usize;
    let mut st = *st;
    for ply in 0..ROLLOUT_PLY_CAP {
        if st.ended {
            break;
        }
        let base = st.hash() ^ (ply as u64).wrapping_mul(0x9E37_79B9);
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        if buf.len == 0 {
            break;
        }
        let mut best = buf.moves[0].inner.expect("move");
        let mut best_v = f32::MIN;
        for i in 0..buf.len {
            let m = buf.moves[i].inner.expect("move");
            let mut next = st;
            next.apply(m, &mut Rnd::new(base.wrapping_add(i as u64)));
            let v = -Heuristic0.eval(&next);
            if v > best_v {
                best_v = v;
                best = m;
            }
        }
        st.apply(best, &mut Rnd::new(base ^ 0xBEEF));
    }
    // Exact terminal value: final score gap. If the ply cap was hit without
    // ending (rare), add the pending settlement as approximation.
    let gap = (st.players[me].score - st.players[1 - me].score) as f32;
    if st.ended {
        gap
    } else {
        gap + (settle_score(&st.players[me]) - settle_score(&st.players[1 - me])) as f32
    }
}

pub struct Searcher {
    tt: TranspositionTable,
    pub nodes: u64,
    /// Bag-refill samples averaged at chance nodes (1 = single deterministic
    /// sample). Each extra sample multiplies the cost of refill nodes only.
    pub branch_samples: u64,
    /// If true, depth-0 leaves use greedy rollout to true terminal score diff
    /// instead of raw heuristic evaluation (slower, much less leaf noise).
    pub rollout: bool,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            tt: TranspositionTable::new(),
            nodes: 0,
            branch_samples: DEFAULT_BRANCH_SAMPLES,
            rollout: false,
        }
    }

    #[inline]
    fn samples_for(&self, st: &FastState, m: &Move) -> u64 {
        if self.branch_samples > 1 && triggers_refill(st, m) { self.branch_samples } else { 1 }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.nodes = 0;
    }

    /// Iterative deepening search to `max_depth`. Returns best move for the
    /// side to move and its value (side-to-move perspective).
    pub fn think(&mut self, st: &FastState, max_depth: u32, ev: &dyn Evaluator, _rng: &mut Rnd) -> (Move, f32) {
        let mut best_move = None;
        let mut best_value = -INF;
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        if buf.len == 0 {
            panic!("think on position with no legal moves");
        }
        let root_key = st.hash();
        for depth in 1..=max_depth {
            self.nodes = 0;
            let mut local_best = None;
            let mut alpha = -INF;
            for i in 0..buf.len {
                let m = buf.moves[i].inner.expect("move");
                let samples = self.samples_for(st, &m);
                let mut acc = 0.0;
                for s in 0..samples {
                    let mut next = *st;
                    next.apply(m, &mut Rnd::new(chance_seed(root_key, &m, s)));
                    acc += self.child_value(&next, st.turn, depth - 1, alpha, INF, ev);
                }
                let value = acc / samples as f32;
                if value > alpha {
                    alpha = value;
                    local_best = Some(m);
                }
            }
            if let Some(m) = local_best {
                best_move = Some(m);
                best_value = alpha;
            }
        }
        (best_move.expect("best move found"), best_value)
    }

    /// Value of the child node `next` from the perspective of the player who
    /// just moved (`parent_turn`).
    ///
    /// negamax's convention is "value from the side to move", so the parent
    /// must negate ONLY when the perspective actually flips. `apply()` does
    /// not flip `turn` at game end, and a refill lets the first-marker holder
    /// start the new round — i.e. the same player can move twice in a row.
    /// Unconditional negation inverted wins/losses at every such node
    /// (the search read "I win" as -1000) — root cause of "deeper is worse".
    fn child_value(&mut self, next: &FastState, parent_turn: u8, depth: u32, alpha: f32, beta: f32, ev: &dyn Evaluator) -> f32 {
        if next.turn == parent_turn {
            self.negamax(next, depth, alpha, beta, ev)
        } else {
            -self.negamax(next, depth, -beta, -alpha, ev)
        }
    }

    fn negamax(&mut self, st: &FastState, depth: u32, mut alpha: f32, beta: f32, ev: &dyn Evaluator) -> f32 {
        // Terminal positions must be scored by the TRUE game result (a decisive
        // win/loss dominates any heuristic). Without this, deeper search can't
        // see wins and only accumulates bag-refill noise — the root cause of
        // "deeper is worse".
        if st.ended {
            if let Some(r) = st.result() {
                return (r.value_for(st.turn as usize) as f32) * TERMINAL_WIN;
            }
        }
        let key = st.hash();
        if let Some(entry) = self.tt.probe(key) {
            if entry.depth >= depth {
                match entry.flag {
                    TtFlag::Exact => return entry.value,
                    TtFlag::Lower if entry.value >= beta => return entry.value,
                    TtFlag::Upper if entry.value <= alpha => return entry.value,
                    _ => {}
                }
            }
        }

        if depth == 0 {
            return if self.rollout { rollout_value(st, ev) } else { ev.eval(st) };
        }

        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        if buf.len == 0 {
            // Should not happen (round-end resolves within apply), but guard.
            return ev.eval(st);
        }

        let mut best_value = -INF;
        let mut best_move = None;
        let tt_move = self.tt.probe(key).and_then(|e| e.best);
        let alpha_orig = alpha;

        // Move ordering: hash move first.
        if let Some(tt_move) = tt_move {
            if let Some(pos) = (0..buf.len).find(|&i| buf.moves[i].inner == Some(tt_move)) {
                buf.moves.swap(0, pos);
            }
        }

        for i in 0..buf.len {
            let m = buf.moves[i].inner.expect("move");
            // Chance seed from (position, move) identity — deterministic per node
            // (TT-consistent) but varies across positions. When the move ends the
            // round, average over `branch_samples` refill draws so deep searches
            // are not whipsawed by a single lucky/unlucky bag sequence.
            let samples = self.samples_for(st, &m);
            let mut acc = 0.0;
            for s in 0..samples {
                let mut next = *st;
                next.apply(m, &mut Rnd::new(chance_seed(key, &m, s)));
                let v = if i == 0 {
                    // First child (hash move): full-window search.
                    self.child_value(&next, st.turn, depth - 1, alpha, beta, ev)
                } else {
                    // PVS: null-window probe first, re-search full window only on
                    // a fail-high (plan stage 1 §1.2).
                    let mut v = self.child_value(&next, st.turn, depth - 1, alpha, alpha + 1.0, ev);
                    if v > alpha && v < beta {
                        v = self.child_value(&next, st.turn, depth - 1, alpha, beta, ev);
                    }
                    v
                };
                acc += v;
            }
            let value = acc / samples as f32;
            if value > best_value {
                best_value = value;
                best_move = Some(m);
            }
            if value > alpha {
                alpha = value;
            }
            if alpha >= beta {
                break;
            }
        }

        let flag = if best_value <= alpha_orig {
            TtFlag::Upper
        } else if best_value >= beta {
            TtFlag::Lower
        } else {
            TtFlag::Exact
        };
        self.tt.store(key, depth, best_value, flag, best_move);
        best_value
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
