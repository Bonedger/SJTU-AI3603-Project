# AI3603-Billiards
AI3603课程台球大作业

## 关键文件说明

| 文件 | 作用 | 在最终测试中是否可修改 |
|------|------|-----------|
| `poolenv.py` | 台球环境（游戏规则） | ❌ 不可修改 |
| `agent.py` | Agent 定义（在 `NewAgent` 中实现你的算法） | ✅ 可修改 `NewAgent` |
| `evaluate.py` | 评估脚本（运行对战） | ✅ 可修改 `agent_b` |
| `PROJECT_GUIDE.md` | 项目详细指南 | 📖 参考文档 |
| `GAME_RULES.md` | 游戏规则说明 | 📖 参考文档 |

对作业内容的视频说明：
说明.mp4：https://pan.sjtu.edu.cn/web/share/da9459405eac6252d01c249c3bcb989f
供大家参考，以文字说明为准。

---

## 项目使用说明
在`evaluate.py`中可修改`agent_b`为不同的Agent，以测试不同算法的性能。
- **BasicAgent**: 基于几何规则的基础智能体，作为性能基准。
- **NewAgent**: 基于几何启发式算法的智能体，性能优于 BasicAgent。
- **CEMAgent**: 基于离线进化策略的智能体，性能优于 NewAgent。
- **RLAgent**: 基于在线实时规划的智能体，性能最优。
- **SearchAgent**: 基于启发式搜索的智能体，性能中等。
- **MCTSAgent**: 基于蒙特卡洛树搜索的智能体，性能中等。

### CEMAgent 使用说明

#### 1. 训练命令
使用 `train/train_cem.py` 脚本进行训练。

**常用参数**:
| 参数 | 说明 | 默认值 | 推荐值/选项 |
| :--- | :--- | :--- | :--- |
| `--iters` | 迭代次数（进化的代数） | `12` | `10` ~ `30` |
| `--pop-size` | 种群大小（每一代生成的候选参数组数量） | `18` | `20` ~ `50` |
| `--elite-frac` | 精英比例（选出多少比例的最优个体用于繁衍） | `0.3` | `0.2` ~ `0.3` |
| `--eval-games` | 每一组参数与对手对战的局数 | `8` | 必须是4的倍数 |
| `--eval-repeats` | 对每一代中的每个个体重复评估的次数（降低随机性） | `1` | `1` |
| `--opponent` | 训练时的陪练对手 | `basic` | `random`, `new`, `basic`, `basic_fast`, `search`, `mcts`, `rl` |
| `--opponents` | (高级) 指定多个轮换对手，逗号分隔 | `None` | 例如 `basic,new` |
| `--out` | 训练结果保存的路径 (.json) | `eval/cem_agent.json` | 自定义路径 |
| `--seed` | 随机种子 | `42` | 任意整数 |
| `--seed-disable` | 是否禁用随机种子（完全随机） | `False` | 调试时开启 |
| `--silent` | 是否静默模式（不输出每局对战日志） | `False` | 训练时推荐开启以减少刷屏 |

**示例**:
```bash
python train/train_cem.py --iters 10 --pop-size 24 --elite-frac 0.3 --eval-games 8 --opponent basic --out eval/my_cem_model.json
```

#### 2. 评估命令
使用 `eval/evaluate_cem.py` 脚本评估模型性能。

**常用参数**:
| 参数 | 说明 | 默认值 | 示例 |
| :--- | :--- | :--- | :--- |
| `--agent` | 选择要评估的 Agent 类型 | `cem` | `cem`, `rl`, `mcts`, `search` |
| `--checkpoint` | (仅 CEM) 加载的模型文件路径 | `None` | `eval/my_cem_model.json` |
| `--opponent` | 评估时的对手 | `basic` | `basic`, `rl`, `mcts` |
| `--num-games` | 对战总局数 | `12` | `20`, `100` (必须是4的倍数) |
| `--seed` | 随机种子 | `42` | 任意整数 |

**示例**:
```bash
# 评估训练好的 CEM 模型
python eval/evaluate_cem.py --agent cem --checkpoint eval/my_cem_model.json --opponent rl --num-games 20

# 评估 RLAgent vs BasicAgent
python eval/evaluate_cem.py --agent rl --opponent basic --num-games 12
```

推荐直接在`evaluate.py`中修改`agent_b`和`n_games`进行性能测试。