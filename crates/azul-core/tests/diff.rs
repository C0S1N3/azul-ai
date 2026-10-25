//! Differential tests: slow reference engine vs fast bitpacked engine.
//! Plan stage 0 §4 — random games + seeded moves must match exactly.

use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::slow::SlowState;
use azul_core::types::*;

/// Normalise a slow state into a comparable tuple of its rule-relevant fields.
fn slow_sig(st: &SlowState) -> Vec<String> {
    let mut v = Vec::new();
    v.push(format!("bag={:?}", st.bag));
    v.push(format!("lid={:?}", st.lid));
    v.push(format!("centre={:?}", st.centre));
    for f in 0..NUM_FACTORIES {
        v.push(format!("fac{}={:?}", f, st.factories[f]));
    }
    for pi in 0..2 {
        let p = &st.players[pi];
        v.push(format!("p{}.score={}", pi, p.score));
        v.push(format!("p{}.wall={:?}", pi, p.wall));
        v.push(format!("p{}.lines={:?}", pi, p.lines));
        let floor: Vec<String> = (0..p.floor_len)
            .map(|i| match p.floor[i] {
                Some(azul_core::slow::FloorItem::Tile(c)) => format!("T{}", c),
                Some(azul_core::slow::FloorItem::Marker) => "M".to_string(),
                None => "-".to_string(),
            })
            .collect();
        v.push(format!("p{}.floor={}", pi, floor.join(",")));
        v.push(format!("p{}.holds_marker={}", pi, p.holds_marker));
    }
    v.push(format!("turn={}", st.turn));
    v.push(format!("marker_in_centre={}", st.marker_in_centre));
    v.push(format!("marker_owner={}", st.marker_owner));
    v.push(format!("ended={}", st.ended));
    v
}

fn fast_sig(st: &FastState) -> Vec<String> {
    let mut v = Vec::new();
    v.push(format!("bag={:?}", st.bag));
    v.push(format!("lid={:?}", st.lid));
    let centre: Vec<u8> = (0..NUM_COLOURS).map(|c| st.centre_count(c as Colour)).collect();
    v.push(format!("centre={:?}", centre));
    for f in 0..NUM_FACTORIES {
        let fac: Vec<u8> = (0..NUM_COLOURS).map(|c| st.factory_count(f, c as Colour)).collect();
        v.push(format!("fac{}={:?}", f, fac));
    }
    for pi in 0..2 {
        let p = &st.players[pi];
        v.push(format!("p{}.score={}", pi, p.score));
        let wall: Vec<u32> = (0..PATTERN_LINES)
            .map(|r| (p.wall >> (r * 5)) & 0b1_1111)
            .collect();
        v.push(format!("p{}.wall={:?}", pi, wall));
        let lines: Vec<(Option<u8>, u8)> = (0..PATTERN_LINES).map(|r| p.line(r)).collect();
        v.push(format!("p{}.lines={:?}", pi, lines));
        let floor: Vec<String> = (0..p.floor_len as usize)
            .map(|i| match p.floor_slot(i) {
                Some(azul_core::slow::FloorItem::Tile(c)) => format!("T{}", c),
                Some(azul_core::slow::FloorItem::Marker) => "M".to_string(),
                None => "-".to_string(),
            })
            .collect();
        v.push(format!("p{}.floor={}", pi, floor.join(",")));
        v.push(format!("p{}.holds_marker={}", pi, p.holds_marker));
    }
    v.push(format!("turn={}", st.turn));
    v.push(format!("marker_in_centre={}", st.marker_in_centre));
    v.push(format!("marker_owner={}", st.marker_owner));
    v.push(format!("ended={}", st.ended));
    v
}

/// Deterministic sort key for moves (enums have payload; can't `as` cast).
fn move_key(m: &Move) -> (u8, usize, u8, u8, usize) {
    let (s1, s2) = match m.source {
        Source::Factory(f) => (0u8, f),
        Source::Centre => (1u8, 0usize),
    };
    let (d1, d2) = match m.dest {
        Dest::Row(r) => (0u8, r),
        Dest::Floor => (1u8, 0usize),
    };
    (s1, s2, m.colour, d1, d2)
}

#[test]
fn differential_random_games() {
    for seed in 0..50u64 {
        let mut slow = SlowState::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut fast = FastState::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut game_rng = Rnd::new(seed ^ 0x1234_5678); // move choice only

        assert_eq!(slow_sig(&slow), fast_sig(&fast), "seed {}: initial state mismatch", seed);

        let mut plies = 0;
        while !slow.ended && plies < 3000 {
            // Compare move sets (order-insensitive).
            let mut slow_moves = slow.legal_moves();
            slow_moves.sort_by_key(move_key);
            let mut fast_buf = MoveBuf::new();
            fast.legal_into(&mut fast_buf);
            let mut fast_moves: Vec<Move> = fast_buf.iter().collect();
            fast_moves.sort_by_key(move_key);
            assert_eq!(
                slow_moves, fast_moves,
                "seed {} ply {}: move set mismatch",
                seed, plies
            );

            let chosen = slow_moves[game_rng.next_u64() as usize % slow_moves.len()];
            slow.apply(chosen, &mut Rnd::new(seed ^ (plies as u64 + 0xAAAA)));
            fast.apply(chosen, &mut Rnd::new(seed ^ (plies as u64 + 0xAAAA)));

            let s_sig = slow_sig(&slow);
            let f_sig = fast_sig(&fast);
            if s_sig != f_sig {
                for (a, b) in s_sig.iter().zip(f_sig.iter()) {
                    if a != b {
                        eprintln!("  mismatch: slow={} fast={}", a, b);
                    }
                }
                panic!("seed {} ply {}: state mismatch", seed, plies);
            }
            plies += 1;
        }
        assert!(slow.ended && fast.ended, "seed {}: game did not end", seed);
        assert_eq!(slow.result(), fast.result(), "seed {}: result mismatch", seed);
    }
}

/// Hash consistency: identical states must hash equal, differing states must
/// (almost surely) hash different.
#[test]
fn hash_tracks_state() {
    let seed = 7u64;
    let a = FastState::new(seed);
    let b = FastState::new(seed);
    assert_eq!(a.hash(), b.hash(), "same state must hash equal");

    let c = FastState::new(seed + 1);
    assert_ne!(a.hash(), c.hash(), "different state must hash differently");
}

/// Move-generation speed smoke check (no strict threshold here; the release
/// benchmark in azul-cli `perft` carries the ≥数万/秒 acceptance).
#[test]
fn movegen_smoke() {
    let mut st = FastState::new(99);
    let mut buf = MoveBuf::new();
    let mut rng = Rnd::new(0xBEEF);
    let mut total = 0u64;
    for _ in 0..200 {
        if st.ended {
            break;
        }
        st.legal_into(&mut buf);
        total += buf.len as u64;
        let m = buf.moves[rng.next_u64() as usize % buf.len].inner.unwrap();
        st.apply(m, &mut Rnd::new(rng.next_u64()));
    }
    assert!(total > 0);
}
