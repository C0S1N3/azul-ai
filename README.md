# Azul AI（azul-ai）

用 Rust 引擎 + AlphaZero 自博弈训练的 Azul（花砖）桌游 AI。从规则引擎、启发式基线、NNUE 评估、Rzepecki 迭代到 AZ 自博弈轨道完整实现，冠军网络 **az_net7** 经内部 Elo 阶梯与跨项目对局验证，强于同规模开源预训练网络。

> **实力声明**：本项目的全部强度数据均来自引擎之间的内部对局，**从未经过人类对弈认证**，真实实力不保证。保守估计，它**并没有超过人类的棋力**——请勿将任何内部 Elo 数字解读为"能赢人类"的证据。

## 项目结构

```
crates/
  azul-core    规则引擎（slow 参考实现 + fast 位打包实现，差分测试互验）
  azul-search  minimax 搜索 + 启发式评估（h0 / h11）
  azul-nnue    NNUE 混合评估（hybrid 引擎）
  azul-az      AlphaZero 轨道（PvNet 策略价值网 + PUCT MCTS + 自博弈 + replay）
  azul-arena   对局 / Elo 评测
  azul-cli     命令行入口（perft / gen-data / duel / az-selfplay）
train/
  train_az.py    AZ 网训练（自博弈数据 + 累积 replay）
  train_nnue.py  NNUE 训练
  diag_nnue.py   NNUE 诊断
docs/worklog.md            全程工程记录（决策、事故、Elo 数据）
azul-ai-plan.md 总体规划（0~6 阶段）
out/
  az_net7.bin    AZ 冠军策略价值网（256 通道 x3 残差块）
  nnue_v4.bin    NNUE 主力评估权重（hybrid 引擎用）
```

## 训练路线（详见 worklog）

1. **阶段 0-1**：规则引擎 + minimax 启发式基线（h0/h11），差分测试保证引擎正确性。
2. **阶段 2**：NNUE 混合评估，对启发式大幅净胜。
3. **阶段 3**：Rzepecki 迭代（教师生成数据 → 训练 → 换代），每代净增益 ≥+30 Elo 为验收线，低于即换更强教师。
4. **阶段 5**：AZ 自博弈轨道。关键教训：**必须累积 replay**（无 replay 必败，源自 garden 项目失败复盘）；搜索严格遵循 turn 翻转规则（仅子节点玩家与当前节点不同时取反）。
5. **迭代演进**：net1（349K 样本）→ net2（累积 replay，+436 Elo）→ … → **net7 成为冠军**。

## 权重与对局数据

- 仓库仅含**最终冠军权重** `az_net7.bin`（AZ 轨道）与 `nnue_v4.bin`（NNUE 轨道）。
- 每代训练数据（自博弈棋谱 bin，GB 级）不入库，按下面命令可完全再生成。

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

## 主要结果（内部相对 Elo，详见 worklog）

- net2 vs net1 **+436 Elo**（37-3，replay 累积的威力）；net1 vs random 39-0（+759）。
- AZ 轨道每代稳定净增，无平台化迹象；sims 档位与耗时线性正相关（2560→81920），更高档位增益递减。
- 跨项目对局（vs cestpasphoto 开源预训练网）：80 局 50胜1平29负（胜率 63%，约 +93 Elo），本项目自训网络更强。

## 使用

```bash
cargo build --release
target/release/azul.exe duel random greedy 10        # 快速体验
target/release/azul.exe perft                        # 引擎性能基准
```

## 硬约束（工程红线，违反过的都付过代价）

1. 搜索视角翻转只在**子节点玩家 ≠ 当前节点玩家**时取反，禁止无条件 negamax 翻转（曾导致深层搜索病理、全部数据污染）。
2. 自博弈训练必须**累积 replay**。
3. 每代验收净增益 ≥+30 Elo，不达标换教师而非堆数据量。
