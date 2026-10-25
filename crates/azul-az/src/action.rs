//! Action encoding: Azul moves <--> 180-dim policy action space (aligned with
//! the garden 180-action space).
//!
//! Encoding formula: `idx = src * 30 + colour * 6 + dest`
//!   - `src`   ∈ 0..6  : 0-4 = factory Factory(f), 5 = centre Centre
//!   - `colour`∈ 0..5  : tile colour 0..4
//!   - `dest`  ∈ 0..6  : 0-4 = pattern line Row(r), 5 = floor line Floor
//!
//! 6 × 5 × 6 = 180 actions in total. The policy head on the training side
//! (train/train_az.py) outputs 180 logits and must use the same encoding order
//! (dest innermost, then colour, then src).

use azul_core::types::{Dest, Move, Source, NUM_COLOURS, NUM_FACTORIES, PATTERN_LINES};

/// Action-space size (garden 180).
pub const ACTION_DIM: usize = 180;

/// Number of sources: 5 factories + 1 centre.
pub const NUM_SOURCES: usize = NUM_FACTORIES + 1;
/// Number of destinations: 5 pattern lines + 1 floor.
pub const NUM_DESTS: usize = PATTERN_LINES + 1;

/// Move -> action index. Illegal moves (out-of-range fields) are not validated
/// here; the caller must guarantee legality.
#[inline]
pub fn to_index(m: &Move) -> usize {
    let src = match m.source {
        Source::Factory(f) => f,
        Source::Centre => NUM_FACTORIES,
    };
    let dest = match m.dest {
        Dest::Row(r) => r,
        Dest::Floor => PATTERN_LINES,
    };
    src * (NUM_COLOURS * NUM_DESTS) + m.colour as usize * NUM_DESTS + dest
}

/// Action index -> move (the inverse of `to_index`). The index must be < 180.
#[inline]
pub fn from_index(idx: usize) -> Move {
    debug_assert!(idx < ACTION_DIM, "action index out of range: {}", idx);
    let src = idx / (NUM_COLOURS * NUM_DESTS);
    let rest = idx % (NUM_COLOURS * NUM_DESTS);
    let colour = rest / NUM_DESTS;
    let dest = rest % NUM_DESTS;
    Move {
        source: if src < NUM_FACTORIES { Source::Factory(src) } else { Source::Centre },
        colour: colour as u8,
        dest: if dest < PATTERN_LINES { Dest::Row(dest) } else { Dest::Floor },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip identity: index -> Move -> index must hold for all 180 indices.
    #[test]
    fn index_roundtrip_identity() {
        for idx in 0..ACTION_DIM {
            let m = from_index(idx);
            assert_eq!(to_index(&m), idx, "roundtrip failed at index {}", idx);
        }
    }

    /// Move -> index -> Move round-trip identity (covers all src/colour/dest
    /// combinations).
    #[test]
    fn move_roundtrip_identity() {
        for src in 0..NUM_SOURCES {
            for colour in 0..NUM_COLOURS {
                for dest in 0..NUM_DESTS {
                    let m = Move {
                        source: if src < NUM_FACTORIES {
                            Source::Factory(src)
                        } else {
                            Source::Centre
                        },
                        colour: colour as u8,
                        dest: if dest < PATTERN_LINES { Dest::Row(dest) } else { Dest::Floor },
                    };
                    assert_eq!(from_index(to_index(&m)), m);
                }
            }
        }
    }
}
