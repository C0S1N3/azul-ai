//! Duel harness + Elo (plan stage 1 §anchor pool v1). Games use paired seeds
//! (each seed plays both sides) to cut variance — the plan's 配对种子 discipline.

use crate::agent::Agent;
use azul_core::fast::FastState;
use azul_core::rng::Rnd;
use azul_core::types::GameResult;

pub struct DuelResult {
    pub wins_a: u32,
    pub wins_b: u32,
    pub draws: u32,
}

impl DuelResult {
    /// Score for A in [0, 1] (win=1, draw=0.5).
    pub fn score_a(&self) -> f64 {
        let total = (self.wins_a + self.wins_b + self.draws) as f64;
        (self.wins_a as f64 + 0.5 * self.draws as f64) / total.max(1.0)
    }

    /// Elo of A relative to B (positive = A stronger).
    pub fn elo_delta(&self) -> f64 {
        let s = self.score_a().clamp(0.001, 0.999);
        -400.0 * (1.0 / s - 1.0).log10()
    }
}

/// Play `games` between agent A (seat varies) and B. Each seed plays twice with
/// sides swapped (paired seeds).
pub fn duel(a: &mut dyn Agent, b: &mut dyn Agent, games: u32, base_seed: u64) -> DuelResult {
    let mut res = DuelResult { wins_a: 0, wins_b: 0, draws: 0 };
    for g in 0..games {
        // Long duels (rollout leaves etc.) print progress so callers can observe.
        if g > 0 && g % 10 == 0 {
            println!("  duel progress {}/{}: {}-{}-{}", g, games, res.wins_a, res.wins_b, res.draws);
        }
        for swap in [false, true] {
            let seed = base_seed.wrapping_add(g as u64 * 0x9E37_79B9) | 1;
            let outcome = play_game(a, b, seed, swap);
            match outcome {
                Some(r) => match r {
                    GameResult::Win(p) => {
                        let a_seat = if swap { 1 } else { 0 };
                        if p == a_seat {
                            res.wins_a += 1;
                        } else {
                            res.wins_b += 1;
                        }
                    }
                    GameResult::Draw => res.draws += 1,
                },
                None => res.draws += 1,
            }
        }
    }
    res
}

/// One game. If `swap`, agent A plays seat 1 instead of seat 0.
fn play_game(a: &mut dyn Agent, b: &mut dyn Agent, seed: u64, swap: bool) -> Option<GameResult> {
    let mut st = FastState::new(seed);
    let mut rng = Rnd::new(seed ^ 0x5A17);
    let mut plies = 0;
    while !st.ended && plies < 3000 {
        let a_turn = (st.turn == 0) ^ swap;
        let m = if a_turn {
            a.choose(&st, &mut rng)
        } else {
            b.choose(&st, &mut rng)
        };
        st.apply(m, &mut Rnd::new(rng.next_u64()));
        plies += 1;
    }
    st.result()
}
