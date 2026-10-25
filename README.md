# Azul AI (azul-ai)

An Azul board-game AI built on a Rust engine with AlphaZero self-play training. The full pipeline is implemented — rules engine, heuristic baselines, NNUE evaluation, Rzepecki iteration, and the AZ self-play track. The champion network **az_net7** is validated on an internal Elo ladder and in cross-project matches, beating open-source pretrained networks of comparable size.

> **Strength disclaimer**: every strength figure in this project comes from internal engine-vs-engine matches and has **never been certified through play against humans**; the true strength is not guaranteed. Conservatively speaking, it **has not exceeded human playing strength** — please do not read any internal Elo number as evidence that it "can beat humans".

🎮 **[Play in the browser](https://azul-ai-f9d6h843g0z.qoder.zone/)** — the champion network az_net7 compiled to WebAssembly, running directly on the page. Open it and play against the AI.

[简体中文 README](README.zh-CN.md)

## Project layout

```
crates/
  azul-core    rules engine (slow reference implementation + fast bit-packed implementation,
               cross-checked by differential testing)
  azul-search  minimax search + heuristic evaluation (h0 / h11)
  azul-nnue    NNUE hybrid evaluation (hybrid engine)
  azul-az      AlphaZero track (PvNet policy-value net + PUCT MCTS + self-play + replay)
  azul-arena   match / Elo evaluation
  azul-cli     CLI entry (perft / gen-data / duel / az-selfplay)
train/
  train_az.py    AZ net training (self-play data + cumulative replay)
  train_nnue.py  NNUE training
  diag_nnue.py   NNUE diagnostics
docs/
  index.html     browser play page (Azul board UI)
  azul_wasm.wasm wasm build of the engine + champion network (served by GitHub Pages)
docs/worklog.md            full engineering log (Chinese; decisions, incidents, Elo data)
azul-ai-plan.md overall plan, phases 0-6 (Chinese)
out/
  az_net7.bin    AZ champion policy-value net (256 channels x3 residual blocks)
  nnue_v4.bin    NNUE main evaluation weights (used by the hybrid engine)
```

## Training path (details in the worklog)

1. **Phases 0-1**: rules engine + minimax heuristic baselines (h0/h11); differential tests guarantee engine correctness.
2. **Phase 2**: NNUE hybrid evaluation, large net wins over the heuristics.
3. **Phase 3**: Rzepecki iteration (teacher generates data → train → swap generation), with ≥ +30 Elo net gain per generation as the acceptance bar; below that, switch to a stronger teacher instead.
4. **Phase 5**: AZ self-play track. Key lessons: **replay must be cumulative** (without replay it always fails — taken from a failed experiment in another project); search strictly follows the turn-flip rule (invert the perspective only when the child node's player differs from the current node's player).
5. **Iteration history**: net1 (349K samples) → net2 (cumulative replay, +436 Elo) → … → **net7 becomes the champion**.

## Weights and match data

- The repo contains only the **final champion weights**: `az_net7.bin` (AZ track) and `nnue_v4.bin` (NNUE track).
- Per-generation training data (self-play game records, GB-sized bins) is not checked in and can be fully regenerated with the commands below.

### Regenerating training data

```bash
cargo build --release

# generate self-play records (with generation-N net; random = untrained random net)
target/release/azul.exe az-selfplay <games> out/az_selfplay.bin out/az_net<N>.bin <sims> <seed>

# train (train/train_az.py reads the records; supports cumulative replay across generations)
python train/train_az.py
```

### Match evaluation

```bash
# agent: random | greedy | h0 | h11 | nnue | az:<sims>
target/release/azul.exe duel az:2560 h11 100 <seed>
```

## Main results (internal relative Elo; see the worklog)

- net2 vs net1 **+436 Elo** (37-3 — the power of cumulative replay); net1 vs random 39-0 (+759).
- Every AZ generation gains steadily with no sign of plateauing; sims scale linearly with time (2560→81920) and higher settings give diminishing gains.
- Cross-project match (vs the cestpasphoto open-source pretrained net): 80 games, 50 wins / 1 draw / 29 losses (63% win rate, about +93 Elo) — this project's self-trained net is stronger.

## Usage

```bash
cargo build --release
target/release/azul.exe duel random greedy 10        # quick taste
target/release/azul.exe perft                        # engine benchmark
```

## Hard constraints (engineering red lines — every violation was paid for)

1. The search perspective flips **only when the child node's player ≠ the current node's player**; unconditional negamax flipping is forbidden (it caused deep-search pathology and poisoned all data).
2. Self-play training must use **cumulative replay**.
3. Acceptance bar: net gain ≥ +30 Elo per generation; if not met, switch the teacher rather than piling on more data.
