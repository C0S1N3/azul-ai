//! NNUE input features (plan stage 2 §2.1: "编码 5" — rich hand-crafted
//! features, whole-position encoding, side-to-move perspective).
//!
//! The feature vector must be IDENTICAL between Rust (inference) and the
//! PyTorch training side (train/train_nnue.py). This layout is authoritative.
//!
//! Layout (171 dims):
//!   0..50    wall cells (25 per player: me 0..25, opp 25..50) — 1.0 if filled
//!  50..80    pattern lines (5 rows × 6 per player: me, opp)
//!            per row: [fill/1..5, colour/5, staged_bool, hot_bool, 0, 0]
//!  80..94    floor slots (7 per player: penalty-magnitude /3)
//!  94..99    centre colour counts (/20)
//!  99..124   factory colour counts (25 = 5 factories × 5 colours, /4)
//! 124..129   bag colour counts (/20)
//! 129..134   lid colour counts (/20)
//! 134..139   me wall column fill (/5)
//! 139..144   opp wall column fill (/5)
//! 144..149   me wall row fill (/5)
//! 149..154   opp wall row fill (/5)
//! 154..171   scalar context:
//!            154 score_diff/100, 155 phase(0..1), 156 me marker,
//!            157 opp marker, 158 factory_tile_frac, 159 bag_frac,
//!            160..164 me line fill fractions (row 0..5),
//!            165..169 opp line fill fractions, 169 = bias 1.0,
//!            170 settle_diff/100 (h0 simulated-settlement score diff — the
//!            strongest hand-crafted signal; lets the net learn CORRECTIONS
//!            on top of h0 instead of rediscovering settlement from scratch)
//!
//! NOTE: keep FEATURE_DIM in sync with train/train_nnue.py.

use azul_core::fast::FastState;
use azul_core::types::*;

/// Feature vector length. Must match train/train_nnue.py (FEATURE_DIM).
/// v2 = 171: adds settle_diff at index 170 (2026-10-14, see worklog).
pub const FEATURE_DIM: usize = 171;

/// Extract the NNUE feature vector for `st`, from the side-to-move's view.
pub fn extract(st: &FastState) -> [f32; FEATURE_DIM] {
    let mut f = [0.0f32; FEATURE_DIM];
    let me = st.turn as usize;
    let opp = 1 - me;

    // 0..50: wall cells, 25 per player.
    for (base, pi) in [(0usize, me), (25usize, opp)] {
        for row in 0..PATTERN_LINES {
            for col in 0..NUM_COLOURS {
                let colour = wall_colour(row, col);
                let filled = (st.players[pi].wall >> (row * 5)) & (1 << colour) != 0;
                f[base + row * NUM_COLOURS + col] = if filled { 1.0 } else { 0.0 };
            }
        }
    }

    // 50..80: pattern lines (5 rows × 6 per player).
    for (base, pi) in [(50usize, me), (65usize, opp)] {
        for row in 0..PATTERN_LINES {
            let (colour, count) = st.players[pi].line(row);
            let off = base + row * 6;
            f[off] = count as f32 / (row + 1) as f32; // fill fraction
            f[off + 1] = colour.map(|c| (c as f32 + 1.0) / 6.0).unwrap_or(0.0);
            f[off + 2] = if count > 0 { 1.0 } else { 0.0 }; // staged
            // hot = one tile from filling, toward a placeable wall cell
            let hot = count == row as u8
                && count > 0
                && colour.map(|c| !st.players[pi].wall_has(row, c)).unwrap_or(false);
            f[off + 3] = if hot { 1.0 } else { 0.0 };
        }
    }

    // 80..94: floor slots (penalty magnitude /3).
    for (base, pi) in [(80usize, me), (87usize, opp)] {
        for i in 0..FLOOR_CAPACITY {
            let pen = if i < st.players[pi].floor_len as usize {
                FLOOR_PENALTY[i].abs() as f32 / 3.0
            } else {
                0.0
            };
            f[base + i] = pen;
        }
    }

    // 94..99: centre colour counts.
    for c in 0..NUM_COLOURS {
        f[94 + c] = st.centre_count(c as Colour) as f32 / 20.0;
    }

    // 99..124: factory colour counts (5×5).
    for fac in 0..NUM_FACTORIES {
        for c in 0..NUM_COLOURS {
            f[99 + fac * NUM_COLOURS + c] = st.factory_count(fac, c as Colour) as f32 / 4.0;
        }
    }

    // 124..134: bag + lid colour counts.
    for c in 0..NUM_COLOURS {
        f[124 + c] = st.bag[c] as f32 / 20.0;
        f[129 + c] = st.lid[c] as f32 / 20.0;
    }

    // 134..154: wall column & row fill fractions.
    for (base, pi) in [(134usize, me), (139usize, opp)] {
        for c in 0..NUM_COLOURS {
            let filled = (0..PATTERN_LINES)
                .filter(|&r| (st.players[pi].wall >> (r * 5)) & (1u32 << wall_colour(r, c) as usize) != 0)
                .count();
            f[base + c] = filled as f32 / 5.0;
        }
    }
    for (base, pi) in [(144usize, me), (149usize, opp)] {
        for r in 0..PATTERN_LINES {
            f[base + r] = ((st.players[pi].wall >> (r * 5)) & 0b1_1111).count_ones() as f32 / 5.0;
        }
    }

    // 154..170: scalar context.
    let score_diff = (st.players[me].score - st.players[opp].score) as f32 / 100.0;
    let phase = (st.players[0].wall.count_ones() + st.players[1].wall.count_ones()) as f32 / 50.0;
    f[154] = score_diff.clamp(-1.0, 1.0);
    f[155] = phase;
    f[156] = if st.players[me].holds_marker { 1.0 } else { 0.0 };
    f[157] = if st.players[opp].holds_marker { 1.0 } else { 0.0 };
    let factory_tiles: u32 = (0..NUM_FACTORIES)
        .map(|fw| (0..NUM_COLOURS).map(|c| st.factory_count(fw, c as Colour) as u32).sum::<u32>())
        .sum();
    f[158] = factory_tiles as f32 / (NUM_FACTORIES * TILES_PER_FACTORY) as f32;
    let bag_total: u32 = st.bag.iter().map(|&n| n as u32).sum();
    f[159] = bag_total as f32 / 100.0;
    for r in 0..PATTERN_LINES {
        let (_, cme) = st.players[me].line(r);
        let (_, copp) = st.players[opp].line(r);
        f[160 + r] = cme as f32 / (r + 1) as f32;
        f[165 + r] = copp as f32 / (r + 1) as f32;
    }
    f[169] = 1.0; // bias
    // 170: h0 settle-score diff — simulated-settlement value of the position.
    let settle = azul_search::eval::settle_score(&st.players[me])
        - azul_search::eval::settle_score(&st.players[opp]);
    f[170] = (settle as f32 / 100.0).clamp(-1.0, 1.0);

    f
}
