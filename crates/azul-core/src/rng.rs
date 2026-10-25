//! Deterministic RNG + canonical tile-drawing procedure.
//!
//! Both slow and fast engines MUST draw tiles through these functions so that
//! seeded games replay identically and differential tests are meaningful.

use crate::types::{Colour, NUM_COLOURS};

/// splitmix64 — tiny, deterministic, dependency-free.
#[derive(Clone, Debug)]
pub struct Rnd {
    state: u64,
}

impl Rnd {
    pub fn new(seed: u64) -> Self {
        Rnd { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, n). n must be > 0.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next_u64() % n
    }
}

/// Canonical draw: pick uniformly among the multiset represented by `bag`
/// counts (colour order 0..5). If bag is empty but lid has tiles, refill bag
/// from lid first (rulebook: refill mid-filling when bag empties).
///
/// Returns None only when both bag and lid are empty.
pub fn draw_tile(bag: &mut [u8; NUM_COLOURS], lid: &mut [u8; NUM_COLOURS], rng: &mut Rnd) -> Option<Colour> {
    let total: u64 = bag.iter().map(|&c| c as u64).sum();
    if total == 0 {
        let lid_total: u64 = lid.iter().map(|&c| c as u64).sum();
        if lid_total == 0 {
            return None;
        }
        *bag = *lid;
        *lid = [0; NUM_COLOURS];
    }
    let total: u64 = bag.iter().map(|&c| c as u64).sum();
    let mut k = rng.below(total);
    for c in 0..NUM_COLOURS {
        if (k as u8) < bag[c] {
            bag[c] -= 1;
            return Some(c as Colour);
        }
        k -= bag[c] as u64;
    }
    unreachable!("draw_tile: k out of range");
}
