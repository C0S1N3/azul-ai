//! Arena smoke tests + h11 weight-tuning harness.

use crate::agent::{MinimaxAgent, RandomAgent};
use crate::elo::duel;
use azul_search::eval::{Heuristic0, Heuristic11, H11Weights};
#[allow(unused_imports)]
use azul_core::rng::Rnd;

#[test]
fn minimax_dominates_random() {
    let mut a = MinimaxAgent::new(2, Heuristic0, "minimax-h0");
    let mut b = RandomAgent;
    let res = duel(&mut a, &mut b, 4, 0xC0FFEE);
    let total = (res.wins_a + res.wins_b + res.draws) as f64;
    println!(
        "duel: minimax-h0-d2 vs random: {}-{}-{} (score {:.2}, elo {:+.0})",
        res.wins_a, res.wins_b, res.draws, res.score_a(), res.elo_delta()
    );
    assert!(total == 8.0);
    assert!(res.wins_a > res.wins_b, "minimax should beat random");
}

/// Large-scale staged tuning campaign for the 5 active h11 weights.
/// Run manually: `cargo test -p azul-arena --release tune_h11 -- --ignored --nocapture`
///
/// Stages (Dan Luu discipline: never trust small-sample selection):
///   1. screen   — many jittered candidates, same-seed paired small duels;
///   2. re-eval  — top-K over larger duels with a fresh seed (de-noise);
///   3. climb    — shrinking-scale hill-climb around the winner;
///   4. verify   — final large duel with another fresh seed (honest number).
#[test]
#[ignore = "long-running tuning harness; run manually"]
fn tune_h11() {
    const SCREEN_TRIALS: u64 = 500;
    const SCREEN_GAMES: u32 = 8;
    const SCREEN_SEED: u64 = 0x5C0E_E111;
    const TOP_K: usize = 12;
    const REEVAL_GAMES: u32 = 40;
    const REEVAL_SEED: u64 = 0x5C0E_E222;
    const CLIMB_ROUNDS: u64 = 8;
    const CLIMB_TRIES: u64 = 12;
    const CLIMB_GAMES: u32 = 24;
    const VERIFY_GAMES: u32 = 120;
    const VERIFY_SEED: u64 = 0x5C0E_E999;

    fn eval_cfg(w: &H11Weights, games: u32, seed: u64) -> (f64, crate::elo::DuelResult) {
        let mut a = MinimaxAgent::new(2, Heuristic11 { w: *w }, "h11");
        let mut b = MinimaxAgent::new(2, Heuristic0, "h0");
        let res = duel(&mut a, &mut b, games, seed);
        (res.score_a(), res)
    }

    fn jitter(rng: &mut Rnd, w: &H11Weights, scale: f32) -> H11Weights {
        let mut w = *w;
        for i in 0..5 {
            let j1 = ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * scale;
            w.base[i] = (w.base[i] + j1).clamp(0.0, 4.0);
            let j2 = ((rng.next_u64() % 2000) as f32 / 1000.0 - 1.0) * scale * 0.75;
            w.late[i] = (w.late[i] + j2).clamp(-2.0, 2.0);
        }
        w
    }

    let mut rng = Rnd::new(0xDEAD_BEEF);
    let start_w = Heuristic11::tuned().w;

    // Stage 1: screen around the current tuned() weights (same seed = paired).
    println!("stage1 screen: {} trials x {} game-pairs", SCREEN_TRIALS, SCREEN_GAMES);
    let (s0, r0) = eval_cfg(&start_w, SCREEN_GAMES, SCREEN_SEED);
    println!("  start(tuned): score {:.3} ({}-{}-{})", s0, r0.wins_a, r0.wins_b, r0.draws);
    let mut cands: Vec<(f64, H11Weights)> = vec![(s0, start_w)];
    for trial in 0..SCREEN_TRIALS {
        let w = jitter(&mut rng, &start_w, 0.8);
        let (s, _) = eval_cfg(&w, SCREEN_GAMES, SCREEN_SEED);
        cands.push((s, w));
        if trial % 100 == 99 {
            let mut sorted = cands.clone();
            sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("  ...trial {}/{}: current screen-best {:.3}", trial + 1, SCREEN_TRIALS, sorted[0].0);
        }
    }
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top: Vec<H11Weights> = cands.iter().take(TOP_K).map(|c| c.1).collect();

    // Stage 2: re-eval top-K on fresh seed, larger duels.
    println!("stage2 re-eval: top {} x {} game-pairs (fresh seed)", TOP_K, REEVAL_GAMES);
    let mut best_w = top[0];
    let mut best_score = -1.0f64;
    for (k, w) in top.iter().enumerate() {
        let (s, r) = eval_cfg(w, REEVAL_GAMES, REEVAL_SEED);
        println!("  top{}: score {:.3} ({}-{}-{})", k, s, r.wins_a, r.wins_b, r.draws);
        if s > best_score {
            best_score = s;
            best_w = *w;
        }
    }

    // Stage 3: hill-climb with shrinking scale; fresh seed each round.
    println!("stage3 climb: {} rounds x {} tries x {} game-pairs", CLIMB_ROUNDS, CLIMB_TRIES, CLIMB_GAMES);
    for round in 0..CLIMB_ROUNDS {
        let scale = 0.4 * 0.7f32.powi(round as i32);
        let seed = 0xC11B_0000 + round;
        let (center_s, _) = eval_cfg(&best_w, CLIMB_GAMES, seed);
        let mut round_best = (center_s, best_w);
        for _ in 0..CLIMB_TRIES {
            let w = jitter(&mut rng, &best_w, scale);
            let (s, _) = eval_cfg(&w, CLIMB_GAMES, seed);
            if s > round_best.0 {
                round_best = (s, w);
            }
        }
        if round_best.1.base != best_w.base || round_best.1.late != best_w.late {
            println!(
                "  round {}: improved {:.3} -> {:.3} (scale {:.3})",
                round, center_s, round_best.0, scale
            );
            best_w = round_best.1;
        } else {
            println!("  round {}: no improvement (center {:.3}, scale {:.3})", round, center_s, scale);
        }
    }
    println!("BEST base={:?} late={:?}", best_w.base, best_w.late);

    // Stage 4: honest verification on a fresh seed.
    let (s, r) = eval_cfg(&best_w, VERIFY_GAMES, VERIFY_SEED);
    println!(
        "verify: h11-tuned vs h0: {}-{}-{} (score {:.3}, elo {:+.0})",
        r.wins_a, r.wins_b, r.draws, s, r.elo_delta()
    );
    let (s0v, r0v) = eval_cfg(&start_w, VERIFY_GAMES, VERIFY_SEED);
    println!(
        "verify: h11-prev  vs h0: {}-{}-{} (score {:.3}, elo {:+.0})",
        r0v.wins_a, r0v.wins_b, r0v.draws, s0v, r0v.elo_delta()
    );
}

/// Pathology experiment (worklog "deeper is worse"): with refill branching,
/// deeper search should beat shallower (d3>d2, d4>d3 monotonic). bf=1 cells
/// re-measure the pathology baseline with the current tuned weights.
/// Run manually:
/// `cargo test -p azul-arena --release depth_trend -- --ignored --nocapture`
#[test]
#[ignore = "long-running pathology experiment; run manually"]
fn depth_trend() {
    for &bf in &[1u64, 4, 8] {
        for &(da, db) in &[(3u32, 2u32), (4u32, 3u32)] {
            let mut a = MinimaxAgent::new(da, Heuristic11::tuned(), "deep").with_branch_samples(bf);
            let mut b = MinimaxAgent::new(db, Heuristic11::tuned(), "shallow").with_branch_samples(bf);
            let res = duel(&mut a, &mut b, 40, 0xD3D2_0000 + bf * 256 + da as u64);
            println!(
                "bf={} d{} vs d{}: {}-{}-{} (score {:.3}, elo {:+.0})",
                bf, da, db, res.wins_a, res.wins_b, res.draws, res.score_a(), res.elo_delta()
            );
        }
    }
}

/// Locked acceptance: a known-good h11 config (produced by tune_h11) beats h0.
/// Current tuned() weights verified 145-92-3 (0.610, +78 Elo) over 240 fresh
/// games at depth 2 (2026-10-14 campaign). The 5-feature linear h11 ceilings
/// near ~60% vs h0 — the plan's original ≥80% target is not reachable in this
/// structure; further strength comes from the NNUE/hybrid route instead.
#[test]
fn h11_beats_h0() {
    // Verified weights via Heuristic11::tuned() (145-92-3 vs h0 over 240 games).
    let mut a = MinimaxAgent::new(2, Heuristic11::tuned(), "h11");
    let mut b = MinimaxAgent::new(2, Heuristic0, "h0");
    let res = duel(&mut a, &mut b, 20, 0x1234_5678);
    println!(
        "duel: h11-d2 vs h0-d2: {}-{}-{} (score {:.2}, elo {:+.0})",
        res.wins_a, res.wins_b, res.draws, res.score_a(), res.elo_delta()
    );
    assert!(
        res.wins_a > res.wins_b,
        "h11 should beat h0: got {}-{}-{}",
        res.wins_a,
        res.wins_b,
        res.draws
    );
}
