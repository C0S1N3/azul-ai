# Azul AI（azul-ai）

一个用 Rust 写的 Azul（花砖）引擎，以及训练出两套评估器（**NNUE 风格的价值网** 与 **AlphaZero 风格的策略价值网**）的完整流水线。仓库里包含从规则引擎、minimax 搜索、NNUE、教师-学生迭代、PUCT MCTS 自博弈、Elo 评测到命令行与浏览器 WebAssembly 版的全链路。

> **关于强度**：下面所有数字都来自本项目内部的引擎对引擎对局，外加一场 80 局的跨项目对局（对手是一个公开的预训练网络）。这些数据**没有经过人类对弈验证**，也不构成"能赢人类"的结论。

🎮 **在线试玩** —— [Qoder Sites](https://azul-ai-f9d6h843g0z.qoder.zone/) · [GitHub Pages](https://c0sin3.github.io/azul-ai/)。冠军网络 `az_net7` 编译成 WebAssembly，完全在页面里本地运行。

[English README](README.md)

## 项目结构

```
crates/
  azul-core    规则引擎：慢速参考实现 + 位打包快速实现，两者做差分测试互验
  azul-search  negamax + alpha-beta + 置换表、迭代加深，手工评估 h0 / h11
  azul-nnue    NNUE 风格价值网（171 维特征 → 小型 MLP），以及混合评估器
  azul-az      AlphaZero 轨道：策略价值网、PUCT MCTS、自博弈、累积 replay
  azul-arena   对局运行器与 Elo 阶梯
  azul-cli     命令行入口：perft / gen-data / duel / az-selfplay
  azul-wasm    浏览器版使用的 WebAssembly 绑定
train/
  train_az.py    AZ 网训练（自博弈棋谱，支持跨代累积 replay）
  train_nnue.py  NNUE 训练
  diag_nnue.py   NNUE 诊断
docs/
  index.html     在线对弈页（棋盘 UI）
  azul_wasm.wasm 引擎 + 冠军网络的 WebAssembly 构建
  worklog.md     全程工程记录（决策、事故、Elo 数据）
azul-ai-plan.md  总体规划（0~6 阶段）
out/
  az_net7.bin    AZ 冠军策略价值网
  nnue_v4.bin    NNUE 主力评估权重（混合引擎用）
```

## 它是怎么下棋的（附带一点机学习科普）

### 1. 规则引擎

同一套规则写了两遍：一份照着规则书写、可读性好；一份做了位打包、快到能拿去做搜索。两份实现互相跑随机局面比对（差分测试），有分歧就说明规则写错了——不会静默地算错分。

这件事比听起来重要：后面所有阶段都在这个引擎产出的局面上训练，规则错一处，整条流水线的数据都会被污染。

### 2. Minimax 搜索 + 手工评估

基线是 **negamax + alpha-beta 剪枝**，配置换表与迭代加深。alpha-beta 是搜索里的经典技巧：如果某条分支已经不可能影响最终结果，就整枝跳过；理想情况下同样时间能多搜一倍的深度。

项目里有两个手工评估函数：`h0`（简单）和 `h11`（更大）。它们既是速度基线，也是训练早期的教师。

### 3. NNUE：一种便宜的神经网络评估

**NNUE**（Efficiently Updatable Neural Network）最早出现在将棋/国际象棋引擎里。思路是让网络小、让输入**稀疏**：把局面编码成一个定长向量，走一步只有少数几维会变，于是可以用很小的网络做评估——便宜到能在搜索的每个叶子节点调用。

完整的 NNUE 还会增量更新第一层（把消失的特征减掉、新增的加上），而不是整向量重算。本项目实现的是这个思路的简化版：171 维手工特征（墙格、图案行、罚分条……）从"轮到走的一方"视角编码，接三层 MLP（隐层 128 → 16 → 16）。向量本身很小，整向量重算已经够快，因此没有做增量更新。训练用二元交叉熵，目标值 胜/平/负 = 1 / 0.5 / 0。

**混合评估器**是一个线性组合：`h0 + α · NNUE` —— 手工项提供稳定的底座，网络补上分辨率。

### 4. AlphaZero 轨道

AlphaZero 把手工评估换成一张一次输出两样东西的网络：

- **策略（policy）**：180 个动作槽各自的概率（从哪个工厂、拿哪个颜色、放到哪里）；
- **价值（value）**：一个 (−1, 1) 的数，估计"轮到走的一方"的胜负。

搜索不用 alpha-beta，而用 **PUCT MCTS**：按 `Q + c·P·√N_parent / (1 + N)` 选择分支，在网络先验 `P` 与已观察到的表现 `Q` 之间做平衡，本项目取 `c = 1.5`。自博弈时根节点加 Dirichlet 噪声（α = 0.3、ε = 0.25），避免每局都走同一条线。下出的着法、访问分布和最终结果一起成为训练数据；训练出的新网再去打下一代。（这套循环出自 AlphaZero 的两篇论文；PUCT 这个选择公式本身来自 Rosin 的 bandit 工作。）

两个工程细节是决定性的：

1. **累积 replay**：只用最新一代的棋谱训练会崩；把历史各代都留在缓冲区里，才有一代代的稳定提升。本项目里仅这一项改动就让 net2 对 net1 **+436 Elo**（37–3）。
2. **turn 翻转规则**：搜索从父节点走到子节点时，**只有子节点的行棋方与当前节点不同**才翻转价值视角。无条件的 negamax 式翻转会造成深层搜索病理，并把生成的数据全部污染。

### 5. Elo 数字怎么读

Elo 是**相对**标尺：+100 Elo 大致等于同一池子里 64% 的得分率。它不含任何绝对含义——一个对自己的上一代 +400 Elo 的网络，离人类水平可能仍然很远；而只跟自己下的内部阶梯还可能整体漂移。

所以这里的数字都标了"内部"，外部参照只有那一场 80 局的跨项目对局。

## 训练路线

1. **规则引擎** + 差分测试。
2. **minimax 基线**（`h0` / `h11`）。
3. **NNUE**：价值网对两个手工评估都有明显净胜。
4. **教师-学生迭代**（参照 Rzepecki 的论文）：当前最强的引擎生成数据，训练新网，新网顶替成为教师。验收线是**每代至少 +30 Elo**；达不到就换更强的教师，而不是继续堆数据。
5. **AlphaZero 自博弈**：net1（349K 样本）→ net2（累积 replay，+436 Elo）→ … → **net7 成为冠军**。

## 主要结果（内部相对 Elo）

- net1 对随机走子：39–0（+759）。
- net2 对 net1：37–3，**+436 Elo**——累积 replay 的效果。
- AZ 每一代都稳定净增，在所用样本量下没有看到平台期。
- 搜索规模：每步 256 → 1024 → 2560 次模拟，耗时大致线性增长，高端增益递减。
- 跨项目对局（对手为 `cestpasphoto/alpha-zero-general` 的预训练网络）：80 局 50 胜 1 平 29 负（得分率 63%，该场约 +93 Elo）。

## 权重与训练数据

- 仓库只放最终权重：`out/az_net7.bin`（AZ 轨道，171 → 256 → 256 主干、180 维策略输出、1 维价值输出，约 625 KB）与 `out/nnue_v4.bin`（NNUE 轨道）。
- 每代自博弈数据是 GB 级，不入库；按下面命令可以完全再生成。

### 再生成训练数据

```bash
cargo build --release

# 自博弈生成棋谱（使用第 N 代网络；random = 未训练随机网）
target/release/azul.exe az-selfplay <games> out/az_selfplay.bin out/az_net<N>.bin <sims> <seed>

# 训练（train/train_az.py 读取棋谱，支持多代累积 replay）
python train/train_az.py
```

### 对局评测

```bash
# agent: random | greedy | h0 | h11 | nnue | az:<sims>
target/release/azul.exe duel az:2560 h11 100 <seed>
```

## 使用

```bash
cargo build --release
target/release/azul.exe duel random greedy 10        # 快速自检
target/release/azul.exe perft                        # 引擎性能基准
```

## 工程红线

1. 搜索视角翻转只在**子节点玩家 ≠ 当前节点玩家**时取反；禁止无条件翻转（曾导致深层搜索病理、污染全部生成数据）。
2. 自博弈训练必须**累积 replay**。
3. 每代验收净增益 ≥ +30 Elo；不达标换教师，而不是堆数据量。

## 参考与出处

本项目的论文与方法来源：

- **M. Rzepecki**，Azul + NNUE 变体的硕士论文，弗罗茨瓦夫大学，2025（导师：Jakub Kowalski）。[PDF](https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf) —— 本项目的引擎、搜索、NNUE 骨架与教师-学生迭代均参照此文。
- **D. Silver 等**，*Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*，arXiv:1712.01815，2017。[arXiv](https://arxiv.org/abs/1712.01815)
- **D. Silver 等**，*A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*，Science 362(6419):1140–1146，2018。[doi:10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)
- **D. Silver 等**，*Mastering the game of Go without human knowledge*，Nature 550:354–359，2017。[doi:10.1038/nature24270](https://doi.org/10.1038/nature24270) —— AlphaGo Zero，即"不依赖人类棋谱"的那套表述，AlphaZero 是它在国际象棋/将棋上的推广。
- **C. D. Rosin**，*Multi-armed bandits with episode context*，Annals of Mathematics and Artificial Intelligence 61(3):203–230，2011 —— AlphaZero 所用的 PUCT 选择公式出处。
- **D. E. Knuth、R. W. Moore**，*An analysis of alpha-beta pruning*，Artificial Intelligence 6(4):293–326，1975。
- **Y. Nasu**，*Efficiently Updatable Neural Networks*（NNUE），2018 —— 将棋社区提出的 NNUE 原始方案，后来被国际象棋引擎广泛采用。
- **A. E. Elo**，*The Rating of Chessplayers, Past and Present*，Arco，1978 —— 本文 Elo 数字所用的评分模型。

参考的开源项目与文章：

- [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0) —— 最知名的开源 AlphaZero 国际象棋实现。
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 与 [cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general) —— 精简版 AlphaZero 参考实现，后者同时作为跨项目对手。
- [bkase/azul](https://github.com/bkase/azul) —— Rust 版 Azul AlphaZero 项目，批量推理、虚拟损失、arena gating 的笔记很有参考价值。
- [jw1912/bullet](https://github.com/jw1912/bullet) —— Rust NNUE 训练库（AdamW、binpack 管线、i16 量化）。
- [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) —— Stockfish 的 NNUE 训练工具。
- [domw95/azul-tiles](https://domwil.co.uk/posts/azul-ai/) —— minimax 版 Azul 的文章，`h0` 基线参考了它的评估思路。
- [D. Luu，*Agentic test processes…*](https://danluu.com/ai-coding) 与 [*There's no reason for software to be slow anymore*](https://danluu.com/perf-opt/) —— 工程纪律与优化经验，影响了本项目的搜索与评估实现。
