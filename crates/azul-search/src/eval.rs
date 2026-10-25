//! Heuristic evaluations. Plan stage 1 §1.4: h0 = simulate immediate
//! settlement (wall-tiling scoring + floor penalty + end bonuses) and return
//! the score difference from the side-to-move's perspective.
//!
//! All helpers operate on the raw packed wall (u32) as pure functions so the
//! engine's move logic stays untouched.

use azul_core::fast::{FastPlayer, FastState};
use azul_core::types::*;

/// Side-to-move perspective score difference (higher = better for `st.turn`).
pub type Eval = f32;

pub trait Evaluator {
    fn eval(&self, st: &FastState) -> Eval;
}

/// Points gained placing `colour` at `row` on wall `wall` (colour-bitmask per
/// row), mirroring the rulebook scoring. Pure — does not mutate.
pub fn placement_score(wall: u32, row: usize, colour: Colour) -> i32 {
    let col = wall_col_for_colour(row, colour);
    let filled = |r: usize, c: usize| -> bool { wall & (1 << (r * 5 + wall_colour(r, c) as usize)) != 0 };
    let h_adj = (col > 0 && filled(row, col - 1)) || (col + 1 < NUM_COLOURS && filled(row, col + 1));
    let v_adj = (row > 0 && filled(row - 1, col)) || (row + 1 < PATTERN_LINES && filled(row + 1, col));
    if !h_adj && !v_adj {
        return 1;
    }
    let mut gained = 0i32;
    if h_adj {
        let mut h = 1i32;
        let mut c = col;
        while c > 0 && filled(row, c - 1) {
            h += 1;
            c -= 1;
        }
        let mut c = col;
        while c + 1 < NUM_COLOURS && filled(row, c + 1) {
            h += 1;
            c += 1;
        }
        gained += h;
    }
    if v_adj {
        let mut v = 1i32;
        let mut r = row;
        while r > 0 && filled(r - 1, col) {
            v += 1;
            r -= 1;
        }
        let mut r = row;
        while r + 1 < PATTERN_LINES && filled(r + 1, col) {
            v += 1;
            r += 1;
        }
        gained += v;
    }
    gained
}

/// End-of-game bonuses on a wall (+2/row, +7/column, +10/full colour).
pub fn end_bonus_of(wall: u32) -> i32 {
    let rows = (0..PATTERN_LINES)
        .filter(|&r| (wall >> (r * 5)) & 0b1_1111 == WALL_ROW_FULL as u32)
        .count() as i32;
    let cols = (0..NUM_COLOURS)
        .filter(|&c| (0..PATTERN_LINES).all(|r| wall & (1 << (r * 5 + wall_colour(r, c) as usize)) != 0))
        .count() as i32;
    let colours = (0..NUM_COLOURS)
        .filter(|&c| (0..PATTERN_LINES).all(|r| wall & (1 << (r * 5 + c)) != 0))
        .count() as i32;
    rows * 2 + cols * 7 + colours * 10
}

/// Simulated settlement score for one player: score every complete pattern
/// line onto the wall (sequentially, so runs chain), add end bonuses, subtract
/// floor penalties, clamp at 0 (rulebook: never below 0).
pub fn settle_score(p: &FastPlayer) -> i32 {
    let mut wall = p.wall;
    let mut score = p.score;
    for row in 0..PATTERN_LINES {
        let (colour, count) = p.line(row);
        if count == (row + 1) as u8 {
            let c = colour.expect("complete line has colour") as usize;
            score += placement_score(wall, row, c as Colour);
            wall |= 1 << (row * 5 + c);
        }
    }
    score += end_bonus_of(wall);
    for i in 0..p.floor_len as usize {
        score += FLOOR_PENALTY[i];
    }
    score.max(0)
}

/// h0: Domwil-style heuristic — immediate-settlement score difference, from
/// the side to move's perspective.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heuristic0;

impl Evaluator for Heuristic0 {
    fn eval(&self, st: &FastState) -> Eval {
        let s0 = settle_score(&st.players[0]) as f32;
        let s1 = settle_score(&st.players[1]) as f32;
        if st.turn == 0 {
            s0 - s1
        } else {
            s1 - s0
        }
    }
}

// ---------------------------------------------------------------------------
// h11: feature-based heuristic (plan stage 1 §1.4). Features × coefficients,
// coefficients vary with gamePhase. Tunable via `H11Weights`.
// ---------------------------------------------------------------------------

/// Progress phase 0..4 from total wall tiles placed (0..50).
#[inline]
pub fn game_phase(st: &FastState) -> f32 {
    let placed = (st.players[0].wall.count_ones() + st.players[1].wall.count_ones()) as f32;
    (placed / 50.0 * 4.0).clamp(0.0, 4.0)
}

/// Per-player feature vector. These are FUTURE-VALUE signals not captured by
/// the h0 settle-score baseline (which only rewards already-complete lines).
pub fn features(st: &FastState, me: usize) -> [f32; 10] {
    let p = &st.players[me];

    // f0: future value of staged pattern lines. A line holding k of (row+1)
    // tiles toward a wall colour not yet placed is worth a discounted future
    // wall placement (~1-2 pts) once completed.
    let mut line_future = 0.0f32;
    for row in 0..PATTERN_LINES {
        let (colour, count) = p.line(row);
        if count > 0 {
            if let Some(c) = colour {
                if !p.wall_has(row, c) {
                    let frac = count as f32 / (row + 1) as f32;
                    // Expected placement value ~1.5 (avg), discounted by how
                    // close the line is to completion.
                    line_future += frac * 1.5;
                }
            }
        }
    }

    // f1: column-completion potential. Each column with k filled is worth a
    // discounted +7 when complete (h0 only pays +7 when already full).
    let mut col_pot = 0.0f32;
    for c in 0..NUM_COLOURS {
        let filled = (0..PATTERN_LINES)
            .filter(|&r| p.wall & (1 << (r * 5 + wall_colour(r, c) as usize)) != 0)
            .count();
        if filled > 0 {
            col_pot += (filled as f32 / PATTERN_LINES as f32) * 7.0;
        }
    }

    // f2: row-completion potential (discounted +2 per near-complete row).
    let mut row_pot = 0.0f32;
    for r in 0..PATTERN_LINES {
        let filled = ((p.wall >> (r * 5)) & 0b1_1111).count_ones();
        if filled > 0 && filled < 5 {
            row_pot += (filled as f32 / 5.0) * 2.0;
        }
    }

    // f3: near-complete lines (k = row, i.e. one tile from filling) — these are
    // about to convert to a wall placement next turn; high leverage.
    let mut hot_lines = 0.0f32;
    for row in 0..PATTERN_LINES {
        let (colour, count) = p.line(row);
        if count == row as u8 && count > 0 {
            if let Some(c) = colour {
                if !p.wall_has(row, c) {
                    hot_lines += 1.0;
                }
            }
        }
    }

    // f4: starter marker held (tempo: controls next-round opening).
    let marker = if p.holds_marker { 1.0 } else { 0.0 };

    // f5..f9: padding for stable indexing (unused).
    [line_future, col_pot, row_pot, hot_lines, marker, 0.0, 0.0, 0.0, 0.0, 0.0]
}

/// Linear weights per feature, phase-adjusted. Base weights are a starting
/// point; per the plan they need arena tuning (h11 vs h0 duel).
#[derive(Clone, Copy, Debug)]
pub struct H11Weights {
    /// Base coefficient per feature (10).
    pub base: [f32; 10],
    /// Extra weight added per phase unit (0..1) for late-rising features.
    pub late: [f32; 10],
}

impl Default for H11Weights {
    fn default() -> Self {
        //        line_fut col_pot row_pot hot  marker
        let base = [1.0, 1.0, 1.0, 1.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0];
        let late = [0.5, 1.5, 0.5, -0.5, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0];
        H11Weights { base, late }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Heuristic11 {
    pub w: H11Weights,
}

impl Default for Heuristic11 {
    fn default() -> Self {
        Heuristic11 { w: H11Weights::default() }
    }
}

impl Heuristic11 {
    /// Best-known weights from the arena tuning harness (verified 39-20-1 vs
    /// h0, 65.8%, +114 Elo at depth 2). Use this for production agents.
    /// Weights from the 2026-10-14 staged tuning campaign (screen 500 / re-eval
    /// top-12 / climb 8x12 / verify 240 fresh games): 145-92-3 (0.610, +78 Elo)
    /// vs h0 at d2. This is near the 5-feature linear h11 ceiling (~60% vs h0);
    /// the earlier config verified 138-95-7 (0.590, +63 Elo) on the same seed.
    pub fn tuned() -> Self {
        Heuristic11 {
            w: H11Weights {
                base: [0.683_892_9, 0.003_294_172_5, 1.600_989_3, 0.166_832_72, 0.938_883_96, 0.0, 0.0, 0.0, 0.0, 0.0],
                late: [1.894_686_5, 0.838_773_67, 0.014_957_009, -0.650_346_04, -0.350_390_43, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
    }
}

impl Evaluator for Heuristic11 {
    fn eval(&self, st: &FastState) -> Eval {
        // Baseline = h0 (settle-score difference), then add feature deltas as
        // a CORRECTION (so default h11 is at least as strong as h0).
        let s0 = settle_score(&st.players[0]) as f32;
        let s1 = settle_score(&st.players[1]) as f32;
        let baseline = if st.turn == 0 { s0 - s1 } else { s1 - s0 };

        let phase = game_phase(st);
        let f_me = features(st, st.turn as usize);
        let f_opp = features(st, 1 - st.turn as usize);
        let mut correction = 0.0f32;
        // Sum the 5 active future-value features (indices 0..5).
        for i in 0..5 {
            let coeff = self.w.base[i] + self.w.late[i] * (phase / 4.0);
            correction += coeff * (f_me[i] - f_opp[i]);
        }
        baseline + correction
    }
}
