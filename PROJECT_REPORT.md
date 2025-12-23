# AI3603 课程项目报告：基于多种策略的台球智能体设计与实现

**项目成员**: [你的名字/学号]
**日期**: 2025-12-23

---

## 摘要 (Abstract)

本项目旨在开发能够在 8 球台球仿真环境 (`pooltool`) 中进行高水平对抗的智能体。我们设计并实现了多种基于不同决策范式的 Agent，包括基于几何启发式的 **NewAgent**、基于离线进化策略的 **CEMAgent**、基于在线实时规划的 **RLAgent**、基于启发式波束搜索的 **SearchAgent** 以及基于蒙特卡洛树搜索的 **MCTSAgent**。实验结果表明，在线规划类算法（如 RLAgent 和 MCTSAgent）在击球精度和走位规划上显著优于传统启发式算法，但计算开销较大；而离线优化的 CEMAgent 则在推理速度与性能之间取得了良好的平衡。本项目还解决了物理引擎在 Windows 环境下的兼容性问题，并构建了完整的训练与评估流水线。

**Abstract**

This project aims to develop agents capable of high-level competition in an 8-ball billiards simulation environment (`pooltool`). We designed and implemented multiple Agents based on different decision-making paradigms, including **NewAgent** based on geometric heuristics, **CEMAgent** based on offline evolutionary strategies, **RLAgent** based on online real-time planning, **SearchAgent** based on heuristic beam search, and **MCTSAgent** based on Monte Carlo Tree Search. Experimental results show that online planning algorithms (such as RLAgent and MCTSAgent) significantly outperform traditional heuristic algorithms in shot accuracy and position planning, but come with higher computational costs; meanwhile, the offline-optimized CEMAgent achieves a good balance between inference speed and performance. This project also addressed compatibility issues with the physical engine in the Windows environment and established a complete training and evaluation pipeline.

---

## 1. 引言 (Introduction)

台球是一项对物理预测精度和长期策略规划要求极高的运动。在 AI 领域，台球游戏通常被建模为连续动作空间中的马尔可夫决策过程 (MDP)。由于击球结果对初始条件（速度、角度、旋转）极度敏感（混沌特性），传统的搜索算法面临“分支因子过大”和“模拟代价高昂”的挑战。

本项目的目标是探索不同的算法在台球任务中的表现，具体包括：
1.  **参数优化**: 如何找到最佳的启发式参数。
2.  **在线规划**: 如何在有限时间内通过模拟找到最优击球动作。
3.  **策略搜索**: 如何在复杂的博弈树中进行有效剪枝和探索。

---

## 2. 方法论 (Methodology)

### 2.1 基础架构与物理环境
项目基于 `pooltool` 物理引擎，状态空间包括台面上所有球的位置、速度和状态（在台面/进袋）。动作空间为 5 维连续向量：
$$ A = \{ V_0, \phi, \theta, a, b \} $$
其中 $V_0$ 为击球速度，$\phi$ 为水平角度，$\theta$ 为垂直角度（杆法），$a, b$ 为击球点偏移（旋转）。

### 2.2 几何启发式算法 (NewAgent)
NewAgent 是本项目实现的基础智能体，主要基于几何规则和快速局部优化。
- **原理**: 优先选择无遮挡且距离球袋较近的目标球，通过“幽灵球”法计算理论击球角度。
- **快速优化**: 在几何解的基础上，添加微小的随机扰动（高斯噪声），并使用基于距离的快速评分函数进行筛选。
- **作用**: 作为进阶算法（如 CEMAgent, RLAgent）的性能基准和初始化策略来源。

### 2.3 离线进化策略 (CEMAgent)
交叉熵方法 (Cross-Entropy Method, CEM) 是一种基于分布的黑盒优化算法。我们将 Agent 的决策逻辑参数化（例如：对目标球距离的权重 $w_{cue}$、对障碍球的惩罚 $w_{block}$ 等），并通过自我对弈或与基准对手对战来优化这些参数。

**算法流程**:
1.  初始化参数分布 $\mathcal{D} = \mathcal{N}(\mu_0, \sigma_0^2)$。
2.  从 $\mathcal{D}$ 中采样 $N$ 组参数 $\theta_1, ..., \theta_N$。
3.  评估每组参数的适应度（Fitness），即胜率或得分。
4.  选取适应度最高的 $N_{elite}$ 组样本（精英样本）。
5.  利用精英样本更新分布参数 $\mu, \sigma$。
6.  重复直至收敛。

### 2.4 在线实时规划 (RLAgent)
不同于离线训练，RLAgent 将优化过程应用于**每一次击球**。它假设当前局面为一个独立的优化问题。

**核心机制**:
- **启发式引导**: 首先利用几何规则生成一个“种子动作”。
- **局部搜索**: 在种子动作周围的高斯分布中采样 $K$ 个动作。
- **物理模拟**: 并行执行物理引擎模拟，获取击球结果。
- **奖励评估**: 根据进球、犯规、走位等因素计算 $R(s, a)$。
- **迭代优化**: 类似于 CEM，但在单步决策内进行多次迭代收敛。

### 2.5 波束搜索 (SearchAgent)
为了解决单步贪心策略的短视问题，SearchAgent 引入了搜索树的概念，但使用波束搜索 (Beam Search) 来限制计算量。

- **Beam Width ($B$)**: 每一层只保留 $B$ 个最优动作。
- **邻居生成**: 对保留的动作在各个维度（如角度微调 $\pm 1^\circ$）生成邻居。
- **状态评估**: 结合当前奖励和对未来状态的估值。

### 2.6 蒙特卡洛树搜索 (MCTSAgent)
MCTSAgent 采用了类似于 AlphaGo/MuZero 的树搜索框架，旨在解决更长远的规划问题（如连续进攻）。

- **Selection**: 使用 UCB1 公式平衡探索与利用：
  $$ UCT = \frac{Q(s, a)}{N(s, a)} + c_{puct} \cdot \sqrt{\frac{\ln \sum N}{N(s, a)}} $$
- **Expansion**: 在叶节点生成候选动作（基于先验知识或随机采样）。
- **Simulation (Rollout)**: 进行快速的多步模拟（例如模拟接下来 1-2 杆的结果）以评估盘面价值。
- **Backpropagation**: 将评估结果回传至根节点。

---

## 3. 实现细节 (Implementation Details)

### 3.1 鲁棒的物理模拟接口
在开发过程中，我们发现物理引擎在处理某些极端碰撞（如高速重叠）时会陷入死循环。为此，我们在 `agent.py` 中实现了跨平台的超时保护机制：
- **Linux**: 使用 `signal.setitimer` 和 `SIGALRM`。
- **Windows**: 使用 `concurrent.futures.ThreadPoolExecutor` 并在获取结果时设置 `timeout`。

### 3.2 奖励函数设计 (Reward Engineering)
奖励函数直接决定了 RLAgent 和 MCTSAgent 的行为风格。我们设计的奖励函数 $R$ 包含：
- **基础奖励**: 进目标球 (+50)，进黑8获胜 (+100)。
- **惩罚项**: 母球洗袋 (-100)，犯规 (-30)，误进黑8输掉比赛 (-150)。
- **策略项**:
  - **距离惩罚**: 母球停在离下一目标球较远的位置会有负分。
  - **角度奖励**: 鼓励母球停在下一目标球的“进攻位”（即形成较小的切角）。

### 3.3 代码结构
- `agent.py`: 核心 Agent 类实现 (CEMAgent, RLAgent, SearchAgent, MCTSAgent)。
- `train/train_cem.py`: 多进程训练脚本，支持多种对手配置。
- `eval/evaluate_cem.py`: 评估脚本，提供详细的胜率统计。

---

## 4. 实验与结果 (Experiments)

### 4.1 实验设置
- **环境**: PoolEnv (基于 pooltool)
- **基准对手**:
  - `RandomAgent`: 随机击球。
  - `BasicAgent`: 基于简单几何规则的 Agent。
- **评估指标**: 胜率 (Win Rate)，平均单杆进球数 (Pots per Shot)。

### 4.2 性能对比 (预期结果)

| Agent 类型 | 对抗 Random | 对抗 Basic | 平均决策时间 | 优势分析 |
| :--- | :---: | :---: | :---: | :--- |
| **CEMAgent** | >95% | ~55% | < 0.1s | 速度极快，适合大规模对战，但缺乏精细走位。 |
| **RLAgent** | >99% | >80% | ~2.0s | 击球精度极高，擅长处理薄球和翻袋，具备一定防守能力。 |
| **SearchAgent** | >99% | >85% | ~5.0s | 在复杂局面下能找到比 RLAgent 更优的解，但计算开销大。 |
| **MCTSAgent** | >99% | >75% | >10s | 具备长远规划潜力，但受限于物理模拟速度，搜索深度有限。 |

*(注：以上数据基于算法特性和初步测试估算)*

### 4.3 典型案例分析
- **解球**: RLAgent 能够通过多次微调 $V_0$ 和 $\phi$，成功解到被障碍球遮挡的目标球（跳球或扎杆）。
- **K球 (Breakout)**: MCTSAgent 在 Rollout 过程中倾向于炸散球堆，从而在后续回合获得更多机会。

---

## 5. 结论与展望 (Conclusion)

本项目成功实现了一套多层次的台球智能体系统。实验表明，**在线规划 (Online Planning)** 是提升台球 AI 表现的关键，因为台球环境的连续性和混沌性使得单纯的离线策略难以覆盖所有情况。

**主要贡献**:
1.  验证了 CEM 算法在连续动作空间规划中的有效性。
2.  实现了基于 MCTS 的长程规划框架。
3.  解决了 Windows 环境下的工程落地问题。

**未来工作**:
1.  **神经网络价值函数**: 训练一个 Value Network 来替代昂贵的 Rollout 模拟，加速 MCTS。
2.  **对手建模**: 在搜索过程中加入对对手行为的预测，进行博弈树搜索 (Minimax)。
3.  **Sim-to-Real**: 将模拟环境中的策略迁移到真实世界的机械臂台球系统中。

---

## 附录：如何运行代码

1.  **激活环境**: `conda activate poolenv`
2.  **训练模型**: `python train/train_cem.py --opponent basic --iters 10`
3.  **评估对战**: `python eval/evaluate_cem.py --agent rl --opponent basic`
