//! Stage-1 smoke tests: the searcher must return legal moves and beat random.

use crate::eval::Heuristic0;
use crate::search::Searcher;
use azul_core::fast::{FastPlayer, FastState};
use azul_core::rng::Rnd;
use azul_core::types::{Dest, Move, Source, NUM_COLOURS, NUM_FACTORIES};

#[test]
fn think_returns_legal_move() {
    let st = FastState::new(7);
    let mut searcher = Searcher::new();
    let ev = Heuristic0;
    let mut rng = Rnd::new(1);
    let (m, value) = searcher.think(&st, 2, &ev, &mut rng);
    let mut buf = azul_core::fast::MoveBuf::new();
    st.legal_into(&mut buf);
    let legal = buf.iter().any(|lm| lm == m);
    assert!(legal, "think returned illegal move {:?}", m);
    assert!(value.is_finite());
}

/// h0 minimax (depth 2) must beat a random mover over several games.
#[test]
fn minimax_beats_random() {
    let mut wins = 0;
    let games = 10;
    for seed in 0..games as u64 {
        let mut st = FastState::new(seed.wrapping_mul(0x1234_5678) | 1);
        let mut searcher = Searcher::new();
        let ev = Heuristic0;
        let mut rng = Rnd::new(seed ^ 0xABCD);
        let mut plies = 0;
        while !st.ended && plies < 3000 {
            let mut buf = azul_core::fast::MoveBuf::new();
            st.legal_into(&mut buf);
            let m = if st.turn == 0 {
                searcher.think(&st, 2, &ev, &mut rng).0
            } else {
                buf.moves[rng.next_u64() as usize % buf.len].inner.unwrap()
            };
            st.apply(m, &mut Rnd::new(rng.next_u64()));
            plies += 1;
        }
        assert!(st.ended);
        if let Some(azul_core::types::GameResult::Win(0)) = st.result() {
            wins += 1;
        }
    }
    // Player 0 = minimax. Random play gives ~50%; require a clear edge.
    assert!(wins * 2 >= games, "minimax won {}/{} games vs random", wins, games);
}

// ---------------------------------------------------------------------------
// Regression tests for the perspective bug (worklog 2026-10-14, bug #4):
// `apply()` keeps `turn` unchanged at game end (and a refill can let the same
// player move twice), so negamax must NOT unconditionally negate child values.
// Otherwise the search reads "I win" as -1000 and "I lose" as +1000.
// ---------------------------------------------------------------------------

/// Blank mid-game state (bag full, no refills will be needed for these tests).
fn base_state() -> FastState {
    let mut st = FastState::new(0x5EED);
    st.factories = [0; NUM_FACTORIES];
    st.centre = 0;
    st.marker_in_centre = true;
    st.marker_owner = 1;
    st.turn = 0;
    st.ended = false;
    st.players = [FastPlayer::new(), FastPlayer::new()];
    st
}

/// Player 0 to move and can force a win in 2 plies: taking the centre
/// colour-4 tile into pattern line 0 completes it; the opponent's only reply
/// (factory colour-1 tiles) empties the table, wall-tiling places colour 4 on
/// wall row 0 (cols 0..3 already filled) -> row 0 complete -> game over.
fn win_in_two_position() -> FastState {
    let mut st = base_state();
    st.players[0].wall = 0b0_1111; // row 0: colours 0,1,2,3 filled; colour 4 missing
    st.players[0].score = 10;
    st.players[1].score = 8;
    st.centre = 1u32 << (4 * NUM_COLOURS); // 1x colour 4
    st.factories[0] = 2u32 << (1 * 3); // 2x colour 1
    st
}

/// Mirror position: player 1 completes wall row 0 at the coming wall-tiling,
/// so EVERY line for player 0 loses within 2 plies. The root value must be
/// strongly negative — with the bug it read as +1000.
fn forced_loss_position() -> FastState {
    let mut st = base_state();
    st.players[1].wall = 0b0_1111;
    st.players[1].set_line(0, Some(4), 1); // complete line -> wall row 0 at tiling
    st.players[1].score = 30;
    st.players[0].score = 5;
    st.centre = 1u32 << (4 * NUM_COLOURS);
    st.factories[0] = 2u32 << (1 * 3);
    st
}

#[test]
fn search_takes_immediate_win() {
    let st = win_in_two_position();
    let mut searcher = Searcher::new();
    let ev = Heuristic0;
    let mut rng = Rnd::new(1);
    let (m, value) = searcher.think(&st, 2, &ev, &mut rng);
    assert_eq!(
        m,
        Move { source: Source::Centre, colour: 4, dest: Dest::Row(0) },
        "search must take the winning move, got {:?}",
        m
    );
    assert!(value > 500.0, "winning line must score ~+1000, got {}", value);
}

#[test]
fn search_avoids_forced_loss() {
    let st = forced_loss_position();
    let mut searcher = Searcher::new();
    let ev = Heuristic0;
    let mut rng = Rnd::new(1);
    let (_m, value) = searcher.think(&st, 2, &ev, &mut rng);
    assert!(value < -500.0, "all lines lose; root value must be ~-1000, got {}", value);
}
