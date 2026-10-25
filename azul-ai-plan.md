# Azul AI 复刻方案（最终增强版）

> **方法论**：Rzepecki（弗罗茨瓦夫大学硕士论文，2025，Minimax+NNUE，BGA 双榜第一，对 top-5 玩家 2.5:0.5）的算法骨架 × Dan Luu（danluu.com/ai-coding + danluu.com/perf-opt，世界冠军认证最强 bot）的工程杠杆与 eval 纪律。
> **本版相对旧方案的改进**：① 落地为具体 Rust workspace 工程结构与模块接口（已定：Rust 实现）；② 按已定"混合双引擎"决策把 AlphaZero 轨道升级为正式阶段（旧方案为可选项）；③ 明确训练技术栈（PyTorch 训练 → 量化导出 → Rust 推理，备选 bullet）与模型导出链路；④ 预留 garden 小程序决策辅助对接契约（180 动作编码 + to_dict JSON）；⑤ 补充开源参考实现的可借鉴点（bkase/azul、cestpasphoto/alpha-zero-general、jw1912/bullet）；⑥ 评估体系显式化为独立 crate 与 Elo 锚点池。
>
> 出处标注：〔R-§x〕= Rzepecki 论文章节；〔D-ai〕= Dan Luu《Agentic test processes…》(danluu.com/ai-coding)；〔D-perf〕= Dan Luu《There's no reason for software to be slow anymore》(danluu.com/perf-opt)。
> 范围：**2 人局 Azul**（两位研究者的主战场）。本期只做引擎+训练+评估；garden 集成留后续，但接口按其契约对齐。

---

## 0. 总体架构与两条路径的分工

| 组件 | 采用谁的做法 | 理由 |
|---|---|---|
| 游戏引擎 | 〔R-5.1〕双引擎模式 | 论文验证的"正确性+速度"两全方案 |
| 搜索算法 | 〔R-5.3〕Minimax 系（起步）→ 〔D-perf〕MCTS 大网络版（双轨） | 论文实测纯随机 rollout MCTS 极差；Dan 的 MCTS 是配大网络的版本 |
| 估值函数 | 〔R-5.3.2→5.3.3〕手工启发式 → NNUE 小网络 | 论文核心贡献；NNUE 无需 GPU、对"NN 只懂大概"的人最友好 |
| 训练循环 | 〔R-7.8〕全自动 11 步训练循环 | 论文第 7–13 轮全自动持续涨分，第 13 轮收敛 |
| 性能优化 | 〔D-perf〕多线程 + 微优化 | Dan 碾压专业团队的核心：**速度翻倍 ≈ +100 Elo**（Azul 平局极少，比国际象棋更陡） |
| 数据增强 | 〔D-ai〕工厂置换 + 列置换 | Dan 的"第 2 列"根因修复；样本效率差一个量级的差距点 |
| Eval 纪律 | 〔R-7.8〕duel 晋级制 + 〔D-ai〕三重防自欺 | 两人共同强调的第一要务 |
| 代码产出方式 | 〔D-ai〕LLM 写代码 + 人定方向 | Dan 实测所有模型无人监督都会失败，必须人把控方向 |

**双引擎路线总览（已定决策）**：
- **主力**：Minimax+NNUE 快速把棋力拉起来（Rzepecki 骨架，本方案阶段 0–4）。
- **副轨**：AlphaZero（policy/value 网 + batched MCTS + 自对弈），工程范式取 bkase/azul + cestpasphoto/alpha-zero-general（阶段 5，与 3/4 并行或后置）。
- **融合**：NNUE 作 MCTS 叶子快速评估/着法排序；强 minimax 当 self-play 教练/陪练；两轨数据互通。

## 0.1 仓库结构（c:\coding\azul，Rust workspace）

```
c:\coding\azul
├── Cargo.toml                  # workspace
├── crates/
│   ├── azul-core/              # 规则引擎（双实现 + 差分测试）
│   │   └── src/{slow.rs, fast.rs, zobrist.rs, scoring_tables.rs, moves.rs, sym.rs}
│   ├── azul-search/            # minimax：pvs.rs, mtdf.rs, tt.rs, ordering.rs, time_mgr.rs, branching.rs
│   ├── azul-nnue/              # 推理：net.rs, quant.rs(int8), simd.rs(AVX2), features.rs(编码5)
│   ├── azul-az/                # AlphaZero：game.rs(Game trait), mcts.rs(batched), replay.rs, selfplay.rs
│   ├── azul-arena/             # 评估：Elo 锚点池、duel、晋级、报告/曲线生成
│   └── azul-cli/               # CLI：play / duel / gen-data / serve(预留 JSON API) / perft
├── train/                      # Python/PyTorch 训练侧
│   ├── encode.py               # 编码转换（编码5 + 增强：工厂/列/颜色置换）
│   ├── train_nnue.py           # NNUE 训练 → export_quant.py → .nnuebin
│   ├── train_az.py             # AZ policy/value 训练
│   └── pipeline/               # 阶段 3 的 11 步循环编排（驱动 azul-cli）
├── tests/                      # 差分测试、perft、规则边界、对拍 C:\coding\garden\engine\azul.py
└── docs/                       # 训练日志、Elo 曲线、决策记录
```

接口契约草案：

```rust
// azul-core
struct FastState;                 // 工厂 3×u64 + 每玩家 1×u64 + Zobrist key
impl FastState {
    fn apply(&mut self, m: Move); fn legal_moves(&self, buf: &mut MoveBuf);
    fn hash(&self) -> u64; fn is_terminal(&self) -> bool; fn result(&self) -> i8; // {-1,0,1}
}
struct Move;                      // 内部 ~500 空间编码；导出层映射 garden 180 编码
// azul-search
trait Evaluator { fn eval(&self, st: &FastState) -> f32; }   // Heuristic / NNUE / AzValue 三实现
fn think(st: &FastState, lim: TimeLimit, ev: &dyn Evaluator) -> (Move, SearchInfo);
// azul-az
trait Game { fn legal(&self)->Vec<Move>; fn apply(&self,m:Move,seed:u64)->Self; fn obs(&self)->Tensor; }
// azul-cli serve（预留 garden 对接）
// POST /suggest {state, time_ms, topk} -> {moves:[{action,factory,colour,line,prob,score,pv}], value}
```

**训练栈取舍**：PyTorch 训练（复用 garden 现成 torch 2.9+cu128 环境；Rzepecki 全 CPU 训练也可行）→ ONNX/裸权重导出 → `azul-nnue` int8 量化前向。备选 jw1912/bullet（Rust 原生 NNUE 训练库：AdamW/ScReLU/binpack 数据管线/i16 量化保存/自定义游戏特征见 examples/ataxx.rs）——若要全 Rust 闭环再切换；特征编码器独立成模块，保证两条路都通。

---

## 0.2 工作记录与防跑偏纪律（长项目必守）

本项目周期长、跨多轮会话，为防记忆缺失造成路线跑偏，执行期间强制遵守：

1. **唯一事实源**：本计划文档 + `docs/worklog.md`（工作日志）。动手前**先读这两个文件**。
2. **worklog.md 结构**：顶部维护「当前阶段指针」（当前在哪个里程碑、下一步）；每次有意义的进展后追加条目（时间、做了什么、验证结果、决策与理由、下一步）；决策变更必须写明"改了什么、为什么"。
3. **阶段自检**：对照里程碑表与验收标准，偏离时立即纠偏或显式更新计划（不允许"默默绕路"）。
4. **决策以计划为准**：Rzepecki 实测值优先于理论推断（见 2.4）；证伪的东西写进 worklog「已证伪清单」，不再重试。
5. **小步验证**：每完成一个子任务跑测试并记录结果，不攒大改动。

---

## 阶段 0：游戏引擎（Rzepecki 双引擎模式）〔R-5.1〕

**目标**：一个绝对正确（慢速参考版）+ 一个极速（位打包快速版）的引擎。这是所有后续工作的地基——隐蔽的规则 bug 会给强度封顶，且让你误以为是 NN 的问题。

1. **慢速参考引擎（slow.rs）**：清晰、正确、只管对不管快。完整 2 人局规则：5 工厂×4 瓷砖 + 中心区、模式行（长度 1–5）、墙面 5×5 固定色排列（墙色 (col−row)%5）、落地行惩罚（−1,−1,−2,−2,−2,−3,−3）、行/列/集齐加成（+2/+7/+10）、先手标记、袋中 20×5 色、终局触发（完成一整行）、终局加成、负分下限 0、平局判定（按 2:0/1:1/0:2 计分口径）。
2. **快速引擎（fast.rs）**：位打包编码——论文实测方案：
   - 工厂区 3 个 uint64（x, y, z）：每工厂 4 bit/色，z 低 25 位存袋余量、次 25 位弃牌堆；
   - 每玩家棋盘 1 个 uint64：25 位墙面 + 各行数量/颜色 + 先手位 + 落地段 + 9 位分数；
   - Zobrist 式哈希（状态切 20 个 16-bit 片查表异或）供置换表用；
   - 计分用编译期生成的查找表（scoring_tables.rs），运行时只做几次查表加法；
   - 工厂补 tile 预先准备，"该补 tile 了吗"只需 `x | y` 判零。
3. **动作编码**：内部 ~500 空间（对齐 bkase/azul 的 ACTION_SPACE_SIZE=500 口径）；导出层映射 garden 180 编码（`factory*30 + colour*6 + line`，见 C:\coding\garden\engine\azul.py），便于日后小程序集成零翻译。
4. **正确性验收**：随机对局 + 变异动作序列，双引擎逐步比对必须完全一致；对局日志可重放；perft 计数基准；与 garden `engine/azul.py` 对拍（其 `to_dict/from_dict` 是现成第三方参照）。
5. **规则书核对结论（已对照官方英文规则书原文逐条核实，2026-10-13）**——以下为权威口径，实现必须逐条覆盖：
   - **终局判定**："The game ends right after the Wall-tiling phase in which at least one player has completed at least one horizontal line of 5 consecutive tiles on her wall"——即**完整打完该轮**（所有玩家都完成 wall-tiling 并扣完落地分），不是提前终止；
   - **先手标记**：首个从中心区取砖的玩家获得标记，**立即放在落地行最左空位**；wall-tiling 后落地行计分时**标记按普通砖计扣分**（"it counts as a normal tile there"），但不进 lid 而是放在自己面前；下轮由持标者补工厂并先行；
   - **落地行计分**：wall-tiling 结束时按位置扣 {1,1,2,2,2,3,3}，标记占一位照扣；**分数永不为负**（"you can never drop below 0 points"）；落地砖全部移出（进 lid）；
   - **pattern line 清理（曾误读，已勘误）**：wall-tiling 时，**满行**上墙后该行**剩余的同色砖移出进 lid**；**未满行的砖保留到下轮继续用**（"any remaining tiles on the pattern lines stay on your player board for the next round"，规则书示例：满行 2/4 的剩余砖进 lid，未满行 3/5 的砖留下）。⚠️ 切勿实现成"未满行清空"；
   - **上墙顺序**：pattern lines 从上到下逐行处理，每行只动最右一枚砖上墙，**每上一枚立即计分**（不是回合末批量计）；
   - **计分规则**：无相邻 +1；有相邻则分别数横向连续段与纵向连续段（均含新砖）**各计一次**（如横 4 纵 3 = 7 分）；
   - **放置约束**：只能放入"该行对应墙行尚无此颜色"的 pattern line；行内已有砖时只能加同色；砖从右向左填；多余砖强制进落地行（可主动全放落地行）；
   - **取砖**：二选一——从某工厂取走全部同色砖（该厂其余砖扫入中心区），或从中心区取走全部同色砖（本回合首个取中心者得先手标记）；
   - **补砖**：中心区和所有工厂均无砖时进入 wall-tiling；下轮由持先手标记者补满工厂（每厂 4 枚）；**袋空时用 lid 中弃砖洗混回袋**；袋与 lid 都空时允许工厂不完整开局；
   - **终局加成**：整行 +2/条、整列 +7/条、集齐 5 枚同色 +10/色；
   - **平局**：分高者胜；同分则**完整横行多者胜**；仍平则共享胜利（2 人局按此实现，竞技口径 2:0/1:1/0:2 仅用于 duel 统计）；
   - **2 人局配置**：5 工厂、袋 100 枚（20×5 色）。
   其余边界人工核对清单：先手标记传递、补砖时机、2 人局 5 工厂。注意 garden 旧实现 `take()` 不维护轮次交替（调用方翻转）——新引擎自身维护执子方，规避此类陷阱。

**验收**：双引擎差分测试全过；快速引擎 ≥ 数万次动作/秒。**预算**：3–5 天（LLM 代理主力编写，人核对规则边界）。

## 阶段 1：Minimax + 手工启发式基线 〔R-5.2, 5.3.1, 5.3.2〕

**目标**：不写一行神经网络代码，先得到一个能赢休闲玩家的 bot，同时为阶段 2 提供数据生成器和 eval 锚点。

**1.1 必装三件套（论文定为 must-have）〔R-5.3.1〕**
- Alpha-Beta 剪枝；
- 置换表（TT）：存最佳走法（hash move），优先探测；
- 迭代加深：配合 TT，浅层为深层供 hash move。

**1.2 论文验证有效的进阶增强**
- 补 tile 分支（branching）：搜索中遇到回合结束补 tile 时，预生成 N 个抽取（branchingFactor），节点值用中位数/截断均值/缩尾均值聚合——处理 Azul 随机性的关键设计（branchingFactor=1 实测最好，作为起点）；
- 初始窗口策略：fullWindow / aspiration / MTDF / safeMTDF 四选一实测（论文中 MTDF/safeMTDF 带来 56% 胜率大增）；
- PVS（主要变例搜索）；
- 边走边生成走法（on the go）实测优于一次性生成。

**1.3 已被论文证伪的方案（不要浪费时间）**
- ❌ 纯随机 rollout 的 MCTS（表现很差）；
- ❌ Killer heuristics（无效）；
- ❌ 遗传算法调启发式系数（耗算力、几乎无提升）。

**1.4 启发式设计模板 〔R-5.3.2〕**
特征 × 系数（随 gamePhase 变化）的线性组合，返回双方差值。heuristic 0（Domwil 同款，起步够用）：模拟回合结束（上墙、计分、扣落地分）→ 模拟终局加成 → 返回分差。最强手工版 heuristic 11（对 h0 胜率 83.36%，置信区间 [82.64%, 84.07%]，n≈10484 局）的特征清单：
- 上三行（墙上砖数最多/满/次多）、靠近中列的砖、下两行不缺砖、下两行的砖与同列成对、能补完下两行列的非空行、先手标记、每块拿到的砖、不同行不同色、未完成列的最大潜力、终局加成；
- 系数随 gamePhase 四型变化：等值(E) / 随阶段递减(D) / 中期最高(M) / 特定阶段例外(P)；gamePhase=4（末轮）时砖类系数调小、分值权重调大。

**验收**：1 秒/步下，h11 级启发式对 h0 胜率 ≥80%（论文 83.36%）；能稳定赢作者本人和休闲玩家。同步建立 `azul-arena` 锚点池 v1：random / greedy / h0 / h11。**预算**：1–2 周（含多轮对局调优）。

## 阶段 2：NNUE 估值网络替换启发式 〔R-5.3.3〕

**目标**：把手工启发式换成小型 NNUE 网络（论文结论：换上后"显著超过"最强手工启发式，最优模型 heuristics350 对最强手工启发式 94.1%、对初始启发式 98.4%）。

**2.1 关键设计决策（全部论文实证）**

| 决策点 | 采用方案 | 依据 |
|---|---|---|
| 网络类型 | 全连接小网络（NNUE 变体），只有估值头 | 无需 policy/MCTS 管道，推理极快，无 GPU 可跑 |
| 编码 | **整局面编码**（区别于象棋 NNUE 的分玩家视角）；7 种编码实测，最优为"大量抽取特征 + 非 one-hot"型（编码 5） | 〔R-5.3.3, R-9〕 |
| 架构 | 7 种实测（64×16×16×1 ~ 512×16×16×1，如 128×16×16×1、256×64×64×1、128×128×32×1），从 128×16×16×1 起步 | 〔R-5.3.3〕 |
| 激活 | 层间 ReLU；输出 tanh（压到 [-1,1]）；**禁用 ClippedReLU**——裁剪摧毁分值信息（Azul 分值至关重要） | 〔R-5.3.3〕⚠️ 已验证的坑 |
| 损失 | MSE / BCE / BCE+sigmoid+目标线性变换 三选一实测；论文 BCE 类明显优 | 〔R-5.3.3〕 |
| 训练目标 | 终局结果 {-1, 0, +1}（论文提出但未验证的改进方向：改用分差，作消融实验） | 〔R-9〕 |
| 部署 | PyTorch 训练 → ONNX 快测 → int8 量化转 Rust NNUE + AVX2 SIMD（论文称"难以置信的加速"，第 8 版比第 1 版快约 8 倍） | 〔R-5.3.3〕 |

**2.2 与 Dan 路线的合流点（数据增强）〔D-ai〕**
自我对弈生成数据时做对称性增强：
- **工厂置换**（5 个工厂位置等价）；
- **列置换**：在游戏各阶段把局面分叉成列重排的新对局（Dan 的"第 2 列"根因修复——让 bot 学到"推进任何半满的列都有价值"，而非死磕平均最优的第 2 列）；颜色置换与墙列置换绑定执行以保持墙色一致。
这是公开 AlphaZero 实现（如 alpha-zero-general）普遍缺失、而 Dan 明确点名差距巨大的一环。

**2.3 训练数据与超参（论文实测量级）**
- 每轮用当前最强 agent 自对弈生成 1.3M–1.9M 样本；每次生成用不同种子；
- batch 1024–4096，LR 0.001 → 3e-5（step/gamma 如 (5,0.1) 或 (3,0.3)），约 20 epoch，weight decay 1e-4；
- 训练流水线：`train/encode.py`（编码+增强）→ `train/train_nnue.py` → `export_quant.py` → `.nnuebin` 供 `azul-nnue` 加载。

**2.4 训练原则（重要）：以 Rzepecki 实测值为默认，不自行发明**
用户决策：Rzepecki 的数据是已验证的，优先于任何理论推断。因此：
- **一切训练默认值照抄论文实测**：目标值 {-1,0,1}、BCE 类损失（sigmoid 输出 + 目标线性变换版）、ReLU+tanh、编码 5、网络结构从 128×16×16×1 起步、batch/LR/epoch/wd 按论文量级。**先跑通论文配方并复现其增益**，这是主线；
- 理论上的改进点（分差/bootstrap 价值目标、WDL 头、数据历史窗口、SPRT 晋级检验、校准图等）**一律降级为后置消融实验**：只在论文配方复现后、且训练曲线暴露出具体问题时（如 v-loss 不降）逐项试，每项单独消融并以 duel 胜率裁决，不得同时改多处；
- 消融裁决标准统一：同种子 duel 对现任 ≥51%（配对种子降方差）才采纳；失败则回退论文配方并记录到 worklog 的"已证伪"清单。

**验收**：新网络 agent 在 1500 局 duel 中对现任最佳手工启发式胜率 ≥51%（晋级门槛），目标 90%+（论文量级）。**预算**：2–3 周（编码工程 + 首轮训练）。

## 阶段 3：全自动训练循环（Rzepecki 7.8 完整 11 步流程）〔R-7.8〕

这是整个方案的发动机。论文第 0–6 轮人工干预较多，第 7–13 轮全自动运行、每轮只需几分钟人工，第 13 轮收敛。参数表：bestHeuristicsId（现任最佳）、历史最佳列表、起始种子、数据集规模、生成数据的每步时限、编码名、NN 架构、batch size、学习率、权重缩放、duel 每步时限。

每轮循环（`train/pipeline/` 编排驱动 `azul-cli`）：
1. 用现任最佳 agent（带每步时限）自我对弈生成数据集；
2. 数据集编码转换；
3. 用最优 batch size / 学习率训练多个模型，存训练日志；
4. 从日志选出 25 个最优模型（含前 7 个 epoch 中的至少 5 个），并更新最优训练参数；
5. 最优模型转成 NNUE "启发式"并编号；
6. 候选 vs 现任最佳 500 局/对；
7. 按胜率取前 10，保存对局分析表；
8. 前 10 vs 现任最佳 1500 局/对；
9. 取前 3；若最优者对现任胜率 ≥51% → 继续；否则把数据生成的每步时限翻倍，结束本轮（质量不够就给生成器更多思考时间）；
10. 前 3 互相 duel + 对抗上轮其他最佳；
11. 选出最终 3 强（综合分 = 各胜率和 + 对上轮最佳胜率×2，奖励对最强对手的表现），选出本轮最佳替代现任。

**关键纪律**：
- 每次生成/对局用不同种子，防止模型反复学同一批数据、防止在自己的训练局上"考试"〔R-7.8〕；
- 每步完成后保存进度，可随时中断续跑〔R-7.8〕；
- **Dan 的三重防自欺**〔D-ai〕叠加在此循环上：① 不用 training loss 选模型（loss 与胜率脱节）；② 不用"打赢自己旧版本"当进步证据（self-play Elo 可虚涨 1000–2000）；③ 保持风格多样的锚点池（随机/贪心/手工启发式/历代候选），每轮对全池测胜率画图。

**验收**：连续 2 轮最优候选对现任胜率 <51% → 收敛，进入阶段 4 冲刺或直接阶段 6 验证。**预算**：CPU 后台挂机数天–2 周/轮量级，人工每轮几分钟。

## 阶段 4：性能优化冲刺（Dan 的胜负手）〔D-perf〕

Dan 的原话级结论：他的 AI 算法层面只"略胜一筹"（他自认比 Rzepecki 论文路线"略好一点"），主要赢在优化——**速度每翻倍 ≈ +100 Elo**（比国际象棋更陡，因 Azul 平局极少），仅多线程一项就足以碾压同等水平的单线程 AI（Rzepecki 的 AI 是单线程的）；叠加 10–20 个"手工做太烦"的微优化后，"手写 AI 已无法合理追赶"。

优先级清单（按 Elo 性价比排序）：
1. **多线程搜索**：minimax 用并行 alpha-beta / Lazy SMP（Rayon）；AZ 轨道用多线程 MCTS + 批量推理。⚠️ Dan 实测 LLM 自己选多线程算法连错 2–3 次——他花 30 分钟读资料定方案，再让 agent 重写多遍。**必须人定方向**；
2. **AVX2/SIMD 推理**（论文同款，NNUE 的天然优势）；
3. **非确定性并行代码的日志重放调试**：标准做法是从调试日志重放复现 bug。人类手工做要数天–一周，对 agent 是"循环里就能干的小事"〔D-perf〕——先建设施后优化；
4. **微优化堆叠**：TT 改进、走法排序（论文指出的自身短板：hash move 失效后按序全扫很慢〔R-9〕——你做了就是超越论文的增量）、更快编码、内联计分查表等；
5. （并入阶段 5）第二版本：更大网络 + MCTS——Dan 的双轨：**小快网络配 minimax，大网络配 MCTS，两个版本互为陪练**〔D-perf〕。

⚠️ 〔D-perf〕额外纪律：游戏 AI 的很多优化会**改变对局结果**且无法廉价判断变好还是变坏；且"性能安全优化会悄悄降棋力"（见 danluu.com/pl-tokens）。因此**每次优化后必须跑锚点池回归**，棋力降即回退。

**验收**：同思考时限下每秒节点数较阶段 3 提升 ≥4×（≈ +200 Elo）；对阶段 3 末代最佳胜率 ≥65%。**预算**：1–2 周（最吃系统工程功力的一段，也是拉开"碾压级"差距的来源）。

## 阶段 5：AlphaZero 双引擎融合轨道（与 3/4 并行，可后置）

按已定"混合双引擎"决策正式立项。工程范式取 bkase/azul（Rust Azul AlphaZero）+ cestpasphoto/alpha-zero-general（即"花园"项目 vendored 的 azg，Dan 认可的最强公开 bot PJF98）：

1. **`azul-az` 核心**：
   - Game trait（legal/apply/obs/symmetries/hash）；
   - batched MCTS：PUCT+FPU、Dirichlet(α=0.3, ε=0.25)、虚拟损失批推理（⚠️ garden 的 batched_mcts.py 教训：虚拟损失冲散先验导致棋力不达标被弃用——选择策略需重做而非调参）、温度截止步数（temp-cutoff-move）；
   - 随机性处理：universes / PC-PIMC 固定 chance seeds〔cestpasphoto〕，与 minimax 侧 branching 机制对照；
   - Rayon 并行自对弈 + replay buffer（旧 garden 失败原因之一：无 replay buffer、每轮仅 ~950 样本）；
   - 无 transposition table 起步（cestpasphoto 同款）。
2. **网络**：trunk + policy/value 双头（policy 输出对齐 garden 180 动作编码）；旧 116K 参数 MobileNetV3 式网络已证伪，直接做大网络档（Dan 的"大网络配 MCTS"路线）；训练 AdamW+OneCycleLR+KL 散度（cestpasphoto 配方）。
3. **互操作（混合路线的价值）**：NNUE 作 MCTS 叶子快速评估/着法排序；强 minimax 当 self-play 教练/陪练；AZ 数据与 NNUE 数据互通喂训练。
4. **arena gating**（bkase 阈值 0.55、800 sims 维护 best checkpoint）。

**验收**：AZ 轨道 best 在锚点池全面 ≥ minimax+NNUE 同限时档；否则保留 minimax 为主力、AZ 退为数据生成器/陪练。**预算**：并行推进或后置，不影响主线达标。

## 阶段 6：强度验证与标定（与 3–5 并行）

1. **锚点池 Elo 追踪**：每轮 checkpoint 对固定锚点池（random/greedy/h0/h11/历代最佳；有条件接入 PJF98=cestpasphoto Azul bot 作公开参照）打 Elo 曲线〔D-ai〕；
2. **北极星对照**：你自己去 danluu.com/game/tile 下 30–50 盘（default 与高线程各半）——"人类世界冠军都下不过它"，这就是目标强度的具体样貌；你下不过自己的 bot、且感觉和下 Dan 的 bot 类似吃力，即为达标〔D-ai〕；
3. **人机实测**：找不同水平的朋友/线上真人分级验证。论文参照系〔R-8.3〕：h19 ≈ BGA 全时 Elo 890（15 局输 1）、h43 ≈ 970（25 局输 1）、h232 = 全时榜第一 Elo 1173 / 赛季榜第一 2340、对 top-5 玩家 2.5:0.5；
4. **对局风格自检**〔R-8.4〕：3.6 万局自战统计——先手胜率 58.32%（≈1.5 分优势）、开局主攻中列（3/4 行）、第 4 轮抢先手标记拿末轮主动权、赢家首行落地惩罚更高（早期牺牲换终盘）。你的 bot 出现同款行为模式是训练充分的信号；
5. ⚠️ **伦理红线**：两人都演示过 BGA 实战，但用 bot 打 BGA 会被判定作弊、成绩作废（Rzepecki 的赛季成绩即被 void，还被社区做视频扒出来）。验证请走朋友、本地社区或自建前端。

---

## 时间线与硬件总览

| 阶段 | 内容 | 预计耗时（业余） | 硬件 | 验收 |
|---|---|---|---|---|
| 0 | 双规则引擎 | 3–5 天 | 笔记本即可 | 差分全绿；≥数万动作/秒 |
| 1 | minimax + h11 | 1–2 周 | 多核 CPU（1s/步基准） | h11 对 h0 ≥80% |
| 2 | NNUE 接入 | 2–3 周 | CPU 训练够用；AVX2 必备；GPU 加速可选 | duel 对 h11 ≥51%（目标 90%+） |
| 3 | 自动训练循环 | 挂机为主，数天–数周 | CPU 后台长跑 | 连续 2 轮 <51% 收敛 |
| 4 | 性能冲刺 | 1–2 周 | 核越多 Elo 越高 | 节点 ≥4×；对 P3 best ≥65% |
| 5 | AZ 融合轨道 | 并行/可后置 | GPU 加速自对弈 | 锚点池 ≥ minimax 主力 |
| 6 | 验证与标定 | 与 3–5 并行 | — | 北极星对照 + Elo 曲线 |

硬件说明：训练侧 PyTorch 复用 garden 现成 torch 2.9+cu128 GPU 环境；Rzepecki 全程 CPU 训练亦可行；部署推理 int8+AVX2。参照系：Dan 无 ML 背景、笔记本 CPU，2–3 小时超过当时最强公开 bot、约 20 小时达世界最强；Rzepecki 全程约一个学期（含大量论文写作）。按本方案（站在两人肩膀上 + LLM 代理写码 + 你已跑通 alpha-zero-general 的经验），**2–3 个月业余时间达到相当强度是现实预期**。

## 风险与回退

| 风险 | 缓解/回退 |
|---|---|
| 规则 bug 封顶强度，伪装成 NN 问题 | 阶段 0 差分测试 + perft + 对拍 garden/azul.py；bug 一律回归用例化 |
| NNUE 增益不及预期 | 编码 5/网络结构网格消融；目标值换分差消融；回退 h11 继续阶段 3 扩数据 |
| 训练死循环（无提升） | 时限翻倍机制〔R-7.8〕；锚点池曲线诊断是数据问题还是搜索问题 |
| 并行化引入隐蔽 bug、悄悄降棋力〔D-perf〕 | 日志重放设施先行；每次优化过锚点池回归，降了即回退 |
| LLM 写码跑偏〔D-ai〕 | 人定方向（多线程算法/网络结构/窗口策略选型），agent 执行 + 人核对边界 |
| AZ 轨道长期不敌 minimax | 降级为数据生成器/陪练，不影响主线达标 |
| 症状式修 bug 造成病态偏好（如"第 2 列"） | 一律查根因 + 数据增强根治〔D-ai〕 |

## 已验证的坑清单（两位研究者亲测，直接绕行）

1. 让 LLM 无人监督自由发挥——GPT-5.1/5.2/5.5、Opus 等全部失败且错法相同〔D-ai〕
2. LLM 自选多线程算法连错 2–3 次，选型人定〔D-perf〕
3. ClippedReLU 摧毁分值信息，Azul 上必用 ReLU+tanh〔R-5.3.3〕
4. 纯随机 rollout MCTS 表现很差〔R-5.2〕；Killer heuristics 无效〔R-5.3.1〕；遗传算法调系数无提升〔R-5.3.2〕
5. 用 training loss 选模型（与胜率脱节）〔D-ai〕
6. 用"打赢自己旧版"当进步证据（self-play Elo 虚涨 1000–2000）〔D-ai〕
7. 症状式修 bug（直接压"第 2 列"估值）→ 要用列置换做根因修复〔D-ai〕
8. 引擎规则 bug 封顶强度——用双引擎差分测试预防〔R-5.1〕
9. 游戏 AI 优化会改变对局结果且无法廉价判优劣；"安全优化"会悄悄降棋力 → 优化后必跑锚点池〔D-perf, pl-tokens〕
10. 旧 garden 实战教训：小网络（116K 参数）+ 粗糙价值目标 clip(分差/25) + 无 replay buffer（每轮 ~950 样本）+ 无 Elo 度量 = 必失败（FAILED_run1 已证伪）；批量 MCTS 的虚拟损失会冲散先验导致棋力不达标
11. 用 bot 打 BGA 等在线平台 = 作弊被 void、被社区实锤〔R-8.3〕

## 参考资料

- Rzepecki, M. 2025，Azul + NNUE 变体的硕士论文（弗罗茨瓦夫大学，导师 Jakub Kowalski）：https://jakubkowalski.tech/Supervising/
- Dan Luu, *Agentic test processes, LLM benchmarks…*：https://danluu.com/ai-coding
- Dan Luu, *There's no reason for software to be slow anymore*：https://danluu.com/perf-opt/
- Dan Luu 公开 bot（北极星）：https://danluu.com/game/tile/
- bkase/azul（Rust Azul AlphaZero：engine/rl-env 分离、批量推理+虚拟损失、arena gating）：https://github.com/bkase/azul
- cestpasphoto/alpha-zero-general（PJF98 bot：Game 抽象、universes 处理随机、batched MCTS、Coach 循环）：https://github.com/cestpasphoto/alpha-zero-general
- jw1912/bullet（NNUE 训练库：AdamW/ScReLU、binpack 管线、i16 量化保存、自定义游戏特征 examples/ataxx.rs）：https://github.com/jw1912/bullet
- domw95/azul-tiles（Dom Wilson minimax 参考，h0 同款）：https://domwil.co.uk/posts/azul-ai/
- garden 项目（视觉管线/前端可复用；`engine/azul.py` 180 动作编码 + to_dict JSON 契约）：C:\coding\garden
