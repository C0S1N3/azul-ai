# Azul AI (azul-ai)

An Azul engine written in Rust, plus the training pipeline that produced its two evaluators: an **NNUE-style value net** and an **AlphaZero-style policy-value net** trained by self-play. The whole path lives in this repo — bit-packed rules engine, minimax search, NNUE, teacher-student iteration, PUCT MCTS self-play, an Elo arena, a CLI, and a WebAssembly build that runs the champion net in the browser.

> **On strength.** Every number below comes from engine-vs-engine matches inside this project, plus one 80-game cross-project match against a publicly available pretrained net. None of it has been validated against human players, and none of it should be read as a claim about how this AI would fare against people.

🎮 **Play in the browser** — [GitHub Pages](https://c0sin3.github.io/azul-ai/). The champion net `az_net7` compiled to WebAssembly, running entirely inside the page.

[简体中文 README](README.zh-CN.md)

## What's in the box

```
crates/
  azul-core    rules engine — a slow reference implementation plus a fast bit-packed one,
               cross-checked against each other by differential testing
  azul-search  negamax + alpha-beta + transposition table, iterative deepening,
               hand-written evaluations h0 / h11
  azul-nnue    NNUE-style value net (171-dim features -> small MLP), plus a hybrid evaluator
  azul-az      AlphaZero track: policy+value net, PUCT MCTS, self-play, replay buffer
  azul-arena   match runner and Elo ladder
  azul-cli     CLI entry: perft / gen-data / duel / az-selfplay
  azul-wasm    WebAssembly bindings used by the browser build
train/
  train_az.py    AZ net training (self-play records, cumulative replay)
  train_nnue.py  NNUE training
  diag_nnue.py   NNUE diagnostics
docs/
  index.html     browser play page (board UI)
  azul_wasm.wasm engine + champion net, compiled to WebAssembly
  worklog.md     full engineering log (Chinese; decisions, incidents, Elo data)
azul-ai-plan.md  overall plan, phases 0-6 (Chinese)
out/
  az_net7.bin    AZ champion policy-value net
  nnue_v4.bin    NNUE main evaluation weights (used by the hybrid engine)
```

## How it works, in plain language

### 1. The rules engine

Two independent implementations: a readable one that mirrors the rulebook, and a bit-packed one that is fast enough to be searched. They are run against each other on random positions (differential testing), so a rules bug shows up as a disagreement instead of a silently wrong evaluation. This matters more than it sounds: every later stage trains on positions produced by this engine, so a rules bug poisons the whole pipeline.

### 2. Minimax search with a hand-written evaluator

The baseline is **negamax with alpha-beta pruning**, a transposition table, and iterative deepening. Alpha-beta is the classic trick that lets a depth-*d* search skip branches that cannot change the result; in the best case it roughly doubles the reachable depth for the same time. Two hand-written evaluators are included: `h0` (simple) and `h11` (larger). Both are still useful as fast baselines and as teachers early in training.

### 3. NNUE: a cheap neural evaluator

**NNUE** ("Efficiently Updatable Neural Network") comes from the shogi/chess world. The idea is to keep the network small and the input *sparse*: encode the position as a fixed-length vector in which only a few entries change per move, then use a tiny network. That keeps evaluation cheap enough to call at every leaf.

A full NNUE also updates the first layer incrementally (subtract the features that disappeared, add the ones that appeared) instead of recomputing. This project implements the simpler half of the idea: a 171-dimensional hand-crafted feature vector (wall cells, pattern lines, floor penalties, …) read from the side-to-move's perspective, fed into an MLP with hidden layers 128 → 16 → 16. The vector is small enough that recomputing it is cheap, so incremental updates are not implemented. Training uses binary cross-entropy against win/draw/loss = 1 / 0.5 / 0.

The **hybrid** evaluator is a simple linear mix — `h0 + α · NNUE` — so the hand-written term keeps the evaluation anchored while the net adds resolution.

### 4. The AlphaZero track

AlphaZero replaces the hand-written evaluation with a network that outputs two things at once:

- a **policy** — a probability for each of the 180 action slots (which colour to take from which factory, and where to place it),
- a **value** — one number in (−1, 1) estimating the result from the side to move.

Instead of alpha-beta, it uses **PUCT MCTS**: a tree search that picks the next branch by balancing the network's prior `P` with how good the branch has looked so far `Q`, using `Q + c·P·√N_parent / (1 + N)` with `c = 1.5`. During self-play the root gets Dirichlet noise (α = 0.3, ε = 0.25) so the game does not always follow the same line. The played moves, the visit distribution and the final result become training data; the net is trained on them and the improved net plays the next generation. (This is the loop described in the AlphaZero papers; PUCT itself comes from Rosin's bandit work.)

Two engineering details turned out to be load-bearing:

1. **Cumulative replay.** Training on only the newest generation's games collapses; keeping all previous generations in the buffer is what makes each generation improve. In this project, generation 2 alone gained **+436 Elo** over generation 1 with that change (37–3).
2. **The turn-flip rule.** When the search passes from a node to a child, the value perspective flips **only if the child's player differs** from the current node's player. Unconditional negamax-style flipping caused a deep-search pathology and poisoned the generated data.

### 5. Reading the Elo numbers

Elo is a **relative** scale: +100 Elo means roughly a 64% score between two players *inside the same pool*. It says nothing absolute. A net that gains +400 Elo against its own previous generation may still be far from a human, and an internal ladder that only ever plays itself can drift. That is why every number here is labelled *internal*, and why the only outside check is a single 80-game match against one public net.

## Training pipeline

1. **Rules engine** + differential tests.
2. **Minimax baselines** (`h0` / `h11`).
3. **NNUE** — the value net wins clearly over the hand-written evaluations.
4. **Teacher-student iteration** (following Rzepecki's thesis): the current strongest engine generates data, a new net is trained on it, and the new net becomes the teacher. The acceptance bar is **≥ +30 Elo per generation**; if a generation fails it, the answer is a stronger teacher, not more data.
5. **AlphaZero self-play** — net1 (349K samples) → net2 (+436 Elo with cumulative replay) → … → **net7 becomes the champion**.

## Results (internal, relative Elo)

- net1 vs a random agent: 39–0 (+759).
- net2 vs net1: 37–3, **+436 Elo** — the effect of cumulative replay.
- Each AZ generation gained steadily, with no plateau observed at the sample counts used.
- Search scale: 256 → 1024 → 2560 simulations per move scales roughly linearly in time; gains diminish at the top end.
- Cross-project match against the `cestpasphoto/alpha-zero-general` pretrained net: 80 games, 50 wins / 1 draw / 29 losses (63% score, ~+93 Elo in that match).

## Weights and data

- Only the final weights are checked in: `out/az_net7.bin` (AZ track, 171 → 256 → 256 trunk, 180 policy logits, 1 value output; ~625 KB) and `out/nnue_v4.bin` (NNUE track).
- Per-generation self-play data is GB-sized and not checked in; it can be regenerated with the commands below.

### Regenerating training data

```bash
cargo build --release

# generate self-play records (with generation-N net; random = untrained net)
target/release/azul.exe az-selfplay <games> out/az_selfplay.bin out/az_net<N>.bin <sims> <seed>

# train (train/train_az.py reads the records; supports cumulative replay across generations)
python train/train_az.py
```

### Match evaluation

```bash
# agent: random | greedy | h0 | h11 | nnue | az:<sims>
target/release/azul.exe duel az:2560 h11 100 <seed>
```

## Usage

```bash
cargo build --release
target/release/azul.exe duel random greedy 10        # quick sanity check
target/release/azul.exe perft                        # engine benchmark
```

## Engineering constraints

1. The search perspective flips **only when the child node's player ≠ the current node's player**; unconditional flipping is forbidden (it caused deep-search pathology and poisoned all generated data).
2. Self-play training must use **cumulative replay**.
3. Acceptance bar: a generation must gain **≥ +30 Elo**; if it does not, replace the teacher instead of adding more data.

## References

Papers and methods this project is based on:

- **M. Rzepecki**, master's thesis on Azul with a NNUE variation, University of Wrocław, 2025 (supervisor: Jakub Kowalski). [PDF](https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf) — the engine, search, NNUE and teacher-student iteration used here.
- **D. Silver et al.**, *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*, arXiv:1712.01815, 2017. [arXiv](https://arxiv.org/abs/1712.01815)
- **D. Silver et al.**, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 362(6419):1140–1146, 2018. [doi:10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)
- **D. Silver et al.**, *Mastering the game of Go without human knowledge*, Nature 550:354–359, 2017. [doi:10.1038/nature24270](https://doi.org/10.1038/nature24270) — AlphaGo Zero, the "no human data" formulation that AlphaZero generalises.
- **C. D. Rosin**, *Multi-armed bandits with episode context*, Annals of Mathematics and Artificial Intelligence 61(3):203–230, 2011 — the PUCT selection rule used by AlphaZero.
- **D. E. Knuth, R. W. Moore**, *An analysis of alpha-beta pruning*, Artificial Intelligence 6(4):293–326, 1975.
- **Y. Nasu**, *Efficiently Updatable Neural Networks* (NNUE), 2018 — the original NNUE proposal from the shogi community, later adopted by chess engines.
- **A. E. Elo**, *The Rating of Chessplayers, Past and Present*, Arco, 1978 — the rating model behind the Elo numbers.

Open-source projects and write-ups used as references:

- [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) — the best-known open-source AlphaZero implementation for chess.
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) and [cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general) — compact AlphaZero reference implementations, the latter also used as a cross-project opponent.
- [bkase/azul](https://github.com/bkase/azul) — a Rust Azul AlphaZero project; useful notes on batch inference, virtual loss and arena gating.
- [jw1912/bullet](https://github.com/jw1912/bullet) — Rust NNUE training library (AdamW, binpack pipeline, i16 quantisation).
- [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) — Stockfish's NNUE training tooling.
- [domw95/azul-tiles](https://domwil.co.uk/posts/azul-ai/) — a minimax Azul write-up; the `h0` baseline follows its evaluation.
- [D. Luu, *Agentic test processes…*](https://danluu.com/ai-coding) and [*There's no reason for software to be slow anymore*](https://danluu.com/perf-opt/) — engineering discipline and optimisation notes that shaped the search/eval work.
