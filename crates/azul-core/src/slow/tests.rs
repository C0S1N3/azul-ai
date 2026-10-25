//! Rulebook-conformance unit tests (plan stage 0 §5: 规则书核对结论).
//! Each test verifies one conclusion from the official rulebook.

use super::*;
use crate::rng::Rnd;

#[test]
fn wall_colour_layout() {
    // Row 0: 0 1 2 3 4; row 1: 4 0 1 2 3; ...
    assert_eq!(wall_colour(0, 0), 0);
    assert_eq!(wall_colour(0, 4), 4);
    assert_eq!(wall_colour(1, 0), 4);
    assert_eq!(wall_colour(4, 0), 1);
    for row in 0..PATTERN_LINES {
        for c in 0..NUM_COLOURS {
            assert_eq!(wall_colour(row, wall_col_for_colour(row, c as Colour)), c as Colour);
        }
    }
}

#[test]
fn setup_two_player() {
    let st = SlowState::new(42);
    let placed: u32 = st.factories.iter().flatten().map(|&n| n as u32).sum();
    assert_eq!(placed, (NUM_FACTORIES * TILES_PER_FACTORY) as u32);
    let bag: u32 = st.bag.iter().map(|&n| n as u32).sum();
    assert_eq!(bag, 100 - 20);
    assert!(st.marker_in_centre);
    assert_eq!(st.turn, 0);
    assert!(!st.ended);
}

/// Rulebook: taking from a factory takes all tiles of that colour and sweeps
/// the rest into the centre.
#[test]
fn factory_take_sweeps_remainder_to_centre() {
    let mut st = SlowState::new(42);
    let mut rng = Rnd::new(1);
    let f = 0;
    let colour = (0..NUM_COLOURS).find(|&c| st.factories[f][c] > 0).unwrap() as Colour;
    let before_factory = st.factories[f];
    let before_centre = st.centre;
    let m = Move { source: Source::Factory(f), colour, dest: Dest::Row(4) };
    st.apply(m, &mut rng);
    for c in 0..NUM_COLOURS {
        let expected = before_centre[c] + if c == colour as usize { 0 } else { before_factory[c] };
        assert_eq!(st.centre[c], expected, "colour {} centre count wrong", c);
    }
    assert!(st.factories[f].iter().all(|&n| n == 0));
}

/// Rulebook: first take from the centre grants the starter marker.
#[test]
fn centre_first_take_grants_marker() {
    let mut st = SlowState::new(42);
    let mut rng = Rnd::new(1);
    // Factory take first (sweeps tiles into the centre).
    let f = 0;
    let colour = (0..NUM_COLOURS).find(|&c| st.factories[f][c] > 0).unwrap() as Colour;
    st.apply(Move { source: Source::Factory(f), colour, dest: Dest::Row(4) }, &mut rng);
    assert!(st.marker_in_centre, "marker must still be in the centre");
    // Now take from the centre -> marker granted to the taker.
    let me = st.turn;
    let cc = (0..NUM_COLOURS).find(|&c| st.centre[c] > 0).unwrap() as Colour;
    st.apply(Move { source: Source::Centre, colour: cc, dest: Dest::Row(3) }, &mut rng);
    assert!(!st.marker_in_centre);
    assert_eq!(st.marker_owner, me);
}

/// Rulebook scoring: no adjacency = +1; adjacency = horizontal run + vertical
/// run (each counted including the newly placed tile; only directions WITH
/// links contribute a run).
#[test]
fn wall_scoring_runs() {
    let mut p = Player::new();
    // Colour 0 goes to (row 0, col 0): isolated => +1.
    let g = score_place(&mut p, 0, 0);
    assert_eq!(g, 1);
    // Colour 1 goes to (row 0, col 1): links horizontally to (0,0) => +2.
    let g = score_place(&mut p, 0, 1);
    assert_eq!(g, 2);
    // Colour 4 goes to (row 1, col 0): links vertically to (0,0), no horizontal
    // links => vertical run only, +2.
    let g = score_place(&mut p, 1, 4);
    assert_eq!(g, 2);
    // Colour 4 goes to (row 0, col 4): isolated => +1.
    let g = score_place(&mut p, 0, 4);
    assert_eq!(g, 1);
    // Colour 0 goes to (row 1, col 1): links h to (1,0) and v to (0,1)
    // => h-run 2 + v-run 2 = +4.
    let g = score_place(&mut p, 1, 0);
    assert_eq!(g, 4);
}

/// Place via the real wall logic; returns the points gained.
fn score_place(p: &mut Player, row: usize, colour: Colour) -> i32 {
    let mut st = SlowState::new(0);
    st.players[0] = p.clone();
    let before = st.players[0].score;
    st.place_on_wall(0, row, colour);
    *p = st.players[0].clone();
    st.players[0].score - before
}

/// Rulebook: end-of-game bonuses +2/row, +7/column, +10/full colour.
#[test]
fn end_bonuses() {
    let mut p = Player::new();
    // One complete row => +2.
    for c in 0..NUM_COLOURS {
        p.wall[0] |= 1 << c;
    }
    assert_eq!(SlowState::end_bonus(&p), 2);

    // One complete column (all rows filled at column 0) => +7.
    let mut p = Player::new();
    for r in 0..PATTERN_LINES {
        let colour = wall_colour(r, 0);
        p.wall[r] |= 1 << colour;
    }
    assert_eq!(SlowState::end_bonus(&p), 7);

    // All five tiles of colour 0 (bit 0 set in every row) => +10.
    let mut p = Player::new();
    for r in 0..PATTERN_LINES {
        p.wall[r] |= 1 << 0;
    }
    assert_eq!(SlowState::end_bonus(&p), 10);
}

/// Rulebook: floor penalties never take the score below 0.
#[test]
fn floor_score_never_negative() {
    let mut st = SlowState::new(7);
    st.players[0].score = 2;
    st.players[0].floor[0] = Some(FloorItem::Marker);
    for i in 1..5 {
        st.players[0].floor[i] = Some(FloorItem::Tile(0));
    }
    st.players[0].floor_len = 5; // penalty -1-1-2-2-2 = -8
    st.centre = [0; NUM_COLOURS];
    st.factories = [[0; NUM_COLOURS]; NUM_FACTORIES];
    st.wall_tiling_and_floor();
    assert!(st.players[0].score >= 0);
    assert_eq!(st.players[0].floor_len, 0, "floor must be cleared after scoring");
}

/// Rulebook: COMPLETE pattern lines lose their remaining tiles to the lid;
/// INCOMPLETE lines keep their tiles for the next round (勘误已固化).
#[test]
fn wall_tiling_keeps_incomplete_lines() {
    let mut st = SlowState::new(7);
    // Line 4 (5 slots): 5 red = complete -> 1 to wall, 4 to lid.
    st.players[0].lines[4] = (Some(0), 5);
    // Line 2 (3 slots): 2 blue = incomplete -> tiles must stay.
    st.players[0].lines[2] = (Some(1), 2);
    let lid_before = st.lid;
    st.wall_tiling_and_floor();
    assert_eq!(st.players[0].lines[4], (None, 0), "complete line cleared");
    assert_eq!(st.players[0].lines[2], (Some(1), 2), "incomplete line keeps tiles");
    assert_eq!(st.lid[0], lid_before[0] + 4, "4 leftover red tiles to lid");
    assert_eq!(st.lid[1], lid_before[1], "blue tiles must NOT go to the lid");
}

#[test]
fn legal_moves_nonempty_and_have_floor() {
    let st = SlowState::new(42);
    let moves = st.legal_moves();
    assert!(!moves.is_empty());
    let floor_moves = moves.iter().filter(|m| m.dest == Dest::Floor).count();
    assert!(floor_moves > 0);
}

/// Random full games must terminate, never panic, and respect invariants.
#[test]
fn random_games_terminate() {
    for seed in 0..20u64 {
        let mut st = SlowState::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut rng = Rnd::new(seed ^ 0xDEAD_BEEF);
        let mut plies = 0;
        while !st.ended && plies < 3000 {
            let moves = st.legal_moves();
            assert!(!moves.is_empty(), "seed {}: no legal moves but not ended", seed);
            let m = moves[(rng.next_u64() as usize) % moves.len()];
            st.apply(m, &mut rng);
            plies += 1;
        }
        assert!(st.ended, "seed {}: game did not terminate", seed);
        assert!(st.result().is_some());
        // Invariant: tile conservation (in play + bag + lid == 100).
        let mut total: u32 = st.bag.iter().map(|&n| n as u32).sum();
        total += st.lid.iter().map(|&n| n as u32).sum::<u32>();
        total += st.centre.iter().map(|&n| n as u32).sum::<u32>();
        total += st.factories.iter().flatten().map(|&n| n as u32).sum::<u32>();
        for p in &st.players {
            for &(c, n) in &p.lines {
                if c.is_some() {
                    total += n as u32;
                }
            }
            for item in p.floor.iter().flatten() {
                if let FloorItem::Tile(_) = item {
                    total += 1;
                }
            }
            for w in &p.wall {
                total += w.count_ones();
            }
        }
        assert_eq!(total, 100, "seed {}: tile conservation violated", seed);
    }
}
