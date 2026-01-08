import math
import copy
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import signal
import concurrent.futures

from .agent import Agent
from .basic_agent import analyze_shot_for_reward


class SimulationTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("simulation timeout")


def simulate_with_timeout(shot, timeout=3):
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout))
        try:
            pt.simulate(shot, inplace=True)
            signal.alarm(0)
            return True
        except SimulationTimeoutError:
            print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
            return False
        except Exception as e:
            signal.alarm(0)
            raise e
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(pt.simulate, shot, inplace=True)
    try:
        future.result(timeout=float(timeout))
        return True
    except concurrent.futures.TimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    finally:
        executor.shutdown(wait=False)

class NewAgent(Agent):
    """改进的智能台球Agent
    
    核心思想：
    1. 智能目标选择：优先选择距离球袋较近的目标球
    2. 几何分析：计算白球到目标球的最佳击球路径
    3. 快速击球生成：基于几何关系生成初始击球参数
    4. 局部优化：对初始击球进行小幅度调整
    """
    
    def __init__(self):
        self.name = "SmartPoolAgent"
        
        # 球袋位置（标准台球桌的6个球袋）
        self.pocket_positions = [
            (0.0, 0.0),           # 左下角
            (1.42, 0.0),          # 右下角  
            (0.0, 2.84),          # 左上角
            (1.42, 2.84),         # 右上角
            (0.71, 1.42),         # 中间左侧
            (0.71, 1.42),         # 中间右侧（相同点，实际只有4个角袋+2个中袋）
        ]
        
    def _get_ball_position(self, ball_id, balls):
        """获取球的位置坐标"""
        if ball_id in balls and balls[ball_id].state.s != 4:  # 4表示进袋状态
            pos = balls[ball_id].state.rvw[0]
            return (float(pos[0]), float(pos[1]))
        return None
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间的欧氏距离"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _find_best_target(self, balls, my_targets):
        """智能选择最佳目标球"""
        cue_pos = self._get_ball_position('cue', balls)
        if cue_pos is None:
            return None
            
        best_ball = None
        best_score = float('-inf')
        
        for ball_id in my_targets:
            ball_pos = self._get_ball_position(ball_id, balls)
            if ball_pos is None:
                continue
                
            # 计算到最近球袋的距离
            min_pocket_dist = min(self._calculate_distance(ball_pos, pocket) 
                                for pocket in self.pocket_positions)
            
            # 计算到白球的距离
            cue_dist = self._calculate_distance(cue_pos, ball_pos)
            
            # 综合评分：距离球袋越近、距离白球越近越好
            score = -min_pocket_dist - 0.5 * cue_dist
            
            if score > best_score:
                best_score = score
                best_ball = ball_id
                
        return best_ball
    
    def _calculate_shot_parameters(self, cue_pos, target_pos, table):
        """计算击球参数"""
        # 计算方向角度
        dx = target_pos[0] - cue_pos[0]
        dy = target_pos[1] - cue_pos[1]
        phi = math.degrees(math.atan2(dy, dx))
        
        # 计算距离并确定初速度
        distance = self._calculate_distance(cue_pos, target_pos)
        V0 = min(max(1.0 + distance * 2, 0.5), 8.0)  # 距离越远速度越大
        
        # 简化参数：直接瞄准目标球
        a = 0.0  # 横向偏移
        b = 0.0  # 纵向偏移
        theta = 30.0  # 固定垂直角度
        
        return {
            'V0': round(V0, 2),
            'phi': round(phi % 360, 2),
            'theta': round(theta, 2),
            'a': round(a, 3),
            'b': round(b, 3)
        }
    
    def _optimize_shot(self, action, balls, my_targets, table, max_iter=5):
        """对初始击球进行局部优化"""
        try:
            # 保存原始状态
            original_action = action.copy()
            best_action = action.copy()
            best_score = -float('inf')
            
            for _ in range(max_iter):
                # 生成邻近动作
                optimized_action = {
                    'V0': max(0.5, min(8.0, best_action['V0'] + np.random.normal(0, 0.3))),
                    'phi': (best_action['phi'] + np.random.normal(0, 10)) % 360,
                    'theta': max(0, min(90, best_action['theta'] + np.random.normal(0, 5))),
                    'a': max(-0.5, min(0.5, best_action['a'] + np.random.normal(0, 0.1))),
                    'b': max(-0.5, min(0.5, best_action['b'] + np.random.normal(0, 0.1)))
                }
                
                # 快速评估（不完整模拟）
                score = self._quick_evaluate(optimized_action, balls, my_targets)
                
                if score > best_score:
                    best_score = score
                    best_action = optimized_action.copy()
            
            # 格式化返回
            for key in best_action:
                if key in ['V0', 'phi', 'theta']:
                    best_action[key] = round(best_action[key], 2)
                else:
                    best_action[key] = round(best_action[key], 3)
                    
            return best_action
            
        except Exception as e:
            # 优化失败，返回原始动作
            return original_action
    
    def _quick_evaluate(self, action, balls, my_targets):
        """快速评估击球质量（不进行完整物理模拟）"""
        try:
            cue_pos = self._get_ball_position('cue', balls)
            if cue_pos is None:
                return 0
                
            # 检查是否有明确的目标
            if not my_targets:
                return 0
                
            target_ball = my_targets[0] if isinstance(my_targets[0], str) else str(my_targets[0])
            target_pos = self._get_ball_position(target_ball, balls)
            
            if target_pos is None:
                return 10  # 目标球已进袋，给予奖励
                
            # 基于距离的简单评估
            distance = self._calculate_distance(cue_pos, target_pos)
            
            # 距离越近得分越高
            base_score = max(0, 100 - distance * 20)
            
            # 速度适中的奖励
            if 2.0 <= action['V0'] <= 4.0:
                base_score += 20
                
            # 角度合理的奖励
            if 15 <= action['theta'] <= 45:
                base_score += 10
                
            return base_score
            
        except Exception:
            return 0
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        try:
            # 检查输入参数
            if balls is None or my_targets is None:
                print(f"[{self.name}] 缺少关键参数，使用随机动作")
                return self._random_action()
                
            # 处理目标球（如果已清空，切换到黑8）
            remaining_own = [bid for bid in my_targets if self._get_ball_position(bid, balls) is not None]
            if len(remaining_own) == 0:
                if '8' in my_targets or my_targets == ['8']:
                    my_targets = ['8']
                else:
                    my_targets = remaining_own if remaining_own else my_targets
                    
            if not my_targets:
                print(f"[{self.name}] 无有效目标球，使用随机动作")
                return self._random_action()

            # 智能选择最佳目标球
            best_target = self._find_best_target(balls, my_targets)
            if best_target is None:
                print(f"[{self.name}] 无法选择目标球，使用随机动作")
                return self._random_action()

            # 获取位置信息
            cue_pos = self._get_ball_position('cue', balls)
            target_pos = self._get_ball_position(best_target, balls)
            
            if cue_pos is None or target_pos is None:
                print(f"[{self.name}] 无法获取位置信息，使用随机动作")
                return self._random_action()

            # 计算基础击球参数
            action = self._calculate_shot_parameters(cue_pos, target_pos, table)
            
            # 进行局部优化
            optimized_action = self._optimize_shot(action, balls, my_targets, table)
            
            print(f"[{self.name}] 目标: {best_target}, 决策: "
                  f"V0={optimized_action['V0']:.2f}, φ={optimized_action['phi']:.1f}°, "
                  f"θ={optimized_action['theta']:.1f}°")
            
            return optimized_action
            
        except Exception as e:
            print(f"[{self.name}] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class RLAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "RLCEMAgent"
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5),
        }
        self.pop_size = 24
        self.n_iters = 3
        self.elite_frac = 0.25
        self.sim_timeout = 2
        self.min_std = {
            'V0': 0.2,
            'phi': 3.0,
            'theta': 2.0,
            'a': 0.03,
            'b': 0.03,
        }

    def _get_ball_position(self, ball_id, balls):
        if ball_id in balls and balls[ball_id].state.s != 4:
            pos = balls[ball_id].state.rvw[0]
            return (float(pos[0]), float(pos[1]))
        return None

    def _dist(self, p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _table_pockets_xy(self, table):
        if table is None or not hasattr(table, 'pockets'):
            return [(0.0, 0.0), (1.42, 0.0), (0.0, 2.84), (1.42, 2.84), (0.71, 0.0), (0.71, 2.84)]
        pockets = []
        for p in table.pockets.values():
            c = p.center
            pockets.append((float(c[0]), float(c[1])))
        return pockets

    def _choose_target(self, balls, my_targets, table):
        cue_pos = self._get_ball_position('cue', balls)
        if cue_pos is None:
            return None
        pockets = self._table_pockets_xy(table)

        best_bid = None
        best_score = float('-inf')
        for bid in my_targets:
            bpos = self._get_ball_position(bid, balls)
            if bpos is None:
                continue
            d_cue = self._dist(cue_pos, bpos)
            d_pocket = min(self._dist(bpos, pk) for pk in pockets)
            score = -0.6 * d_cue - 1.0 * d_pocket
            if score > best_score:
                best_score = score
                best_bid = bid
        return best_bid

    def _heuristic_action(self, balls, my_targets, table):
        target = self._choose_target(balls, my_targets, table)
        cue_pos = self._get_ball_position('cue', balls)
        target_pos = self._get_ball_position(target, balls) if target is not None else None
        if cue_pos is None or target_pos is None:
            return self._random_action()

        dx = target_pos[0] - cue_pos[0]
        dy = target_pos[1] - cue_pos[1]
        phi = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        distance = self._dist(cue_pos, target_pos)
        V0 = float(np.clip(1.2 + 2.2 * distance, 0.5, 8.0))
        return {
            'V0': V0,
            'phi': phi,
            'theta': 25.0,
            'a': 0.0,
            'b': 0.0,
        }

    def _clip_action(self, action):
        clipped = dict(action)
        clipped['V0'] = float(np.clip(clipped['V0'], *self.pbounds['V0']))
        clipped['phi'] = float(clipped['phi'] % 360)
        clipped['theta'] = float(np.clip(clipped['theta'], *self.pbounds['theta']))
        clipped['a'] = float(np.clip(clipped['a'], *self.pbounds['a']))
        clipped['b'] = float(np.clip(clipped['b'], *self.pbounds['b']))
        return clipped

    def _evaluate_action(self, action, balls, table, last_state_snapshot, my_targets):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(
            V0=action['V0'],
            phi=action['phi'],
            theta=action['theta'],
            a=action['a'],
            b=action['b'],
        )
        if not simulate_with_timeout(shot, timeout=self.sim_timeout):
            return 0.0
        return float(analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets))

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ['8']

            base_action = self._heuristic_action(balls, my_targets, table)
            mean = self._clip_action(base_action)
            std = {
                'V0': 1.0,
                'phi': 18.0,
                'theta': 10.0,
                'a': 0.12,
                'b': 0.12,
            }

            best_action = mean
            best_score = float('-inf')

            elite_k = max(1, int(self.pop_size * self.elite_frac))

            for _ in range(self.n_iters):
                actions = []
                scores = []

                for _ in range(self.pop_size):
                    sampled = {
                        'V0': mean['V0'] + float(np.random.normal(0, std['V0'])),
                        'phi': mean['phi'] + float(np.random.normal(0, std['phi'])),
                        'theta': mean['theta'] + float(np.random.normal(0, std['theta'])),
                        'a': mean['a'] + float(np.random.normal(0, std['a'])),
                        'b': mean['b'] + float(np.random.normal(0, std['b'])),
                    }
                    sampled = self._clip_action(sampled)
                    score = self._evaluate_action(sampled, balls, table, last_state_snapshot, my_targets)
                    actions.append(sampled)
                    scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_action = sampled

                elite_idx = np.argsort(scores)[-elite_k:]
                elite_actions = [actions[i] for i in elite_idx]

                mean = {
                    k: float(np.mean([a[k] for a in elite_actions]))
                    for k in ['V0', 'phi', 'theta', 'a', 'b']
                }
                mean = self._clip_action(mean)

                for k in std.keys():
                    v = float(np.std([a[k] for a in elite_actions]))
                    std[k] = max(v, self.min_std[k])

            action = {
                'V0': round(float(best_action['V0']), 2),
                'phi': round(float(best_action['phi']), 2),
                'theta': round(float(best_action['theta']), 2),
                'a': round(float(best_action['a']), 3),
                'b': round(float(best_action['b']), 3),
            }
            print(f"[{self.name}] 决策(估计得分: {best_score:.2f}): V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action
        except Exception as e:
            print(f"[{self.name}] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class SearchAgent(RLAgent):
    def __init__(self):
        super().__init__()
        self.name = "SearchAgent"
        self.beam_width = 6
        self.max_evals = 72
        self.sim_timeout = 2

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ['8']

            base_action = self._clip_action(self._heuristic_action(balls, my_targets, table))

            score_cache = {}

            def _key(a):
                return (
                    round(float(a['V0']), 2),
                    round(float(a['phi']) % 360.0, 2),
                    round(float(a['theta']), 2),
                    round(float(a['a']), 3),
                    round(float(a['b']), 3),
                )

            def _eval(a):
                k = _key(a)
                if k in score_cache:
                    return score_cache[k]
                s = self._evaluate_action(a, balls, table, last_state_snapshot, my_targets)
                score_cache[k] = float(s)
                return float(s)

            total_evals = 0
            base_score = _eval(base_action)
            total_evals += 1

            best_action = base_action
            best_score = base_score

            beam = [(base_action, base_score)]

            schedule = [
                ('phi', [0.0, 18.0, -18.0, 9.0, -9.0]),
                ('V0', [0.0, 1.0, -1.0, 0.45, -0.45]),
                ('theta', [0.0, 10.0, -10.0, 5.0, -5.0]),
                ('phi', [0.0, 5.0, -5.0, 2.0, -2.0]),
                ('a', [0.0, 0.12, -0.12, 0.06, -0.06]),
                ('b', [0.0, 0.12, -0.12, 0.06, -0.06]),
            ]

            for param, deltas in schedule:
                if total_evals >= self.max_evals:
                    break
                candidates = []

                for a, _ in beam:
                    if total_evals >= self.max_evals:
                        break
                    for d in deltas:
                        if total_evals >= self.max_evals:
                            break

                        na = dict(a)
                        na[param] = float(na[param]) + float(d)
                        na = self._clip_action(na)
                        s = _eval(na)
                        total_evals += 1
                        candidates.append((na, s))

                        if s > best_score:
                            best_score = s
                            best_action = na

                if len(candidates) == 0:
                    break
                candidates.sort(key=lambda x: x[1])
                beam = candidates[-self.beam_width:]

            action = {
                'V0': round(float(best_action['V0']), 2),
                'phi': round(float(best_action['phi']), 2),
                'theta': round(float(best_action['theta']), 2),
                'a': round(float(best_action['a']), 3),
                'b': round(float(best_action['b']), 3),
            }
            print(
                f"[{self.name}] 决策(搜索评估: {best_score:.2f}, evals={total_evals}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}, "
                f"a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action
        except Exception as e:
            print(f"[{self.name}] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class MCTSNode:
    def __init__(self, balls, table, my_targets, parent=None, action_from_parent=None):
        self.balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        self.table = copy.deepcopy(table)
        self.my_targets = list(my_targets) if my_targets is not None else []
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        self.reward_from_parent = 0.0


class MCTSAgent(RLAgent):
    def __init__(self):
        super().__init__()
        self.name = "MCTSAgent"
        self.n_sims = 30
        self.c_puct = 1.4
        self.max_depth = 1
        self.action_samples = 12

    def _simulate_action(self, balls, table, my_targets, action):
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(
            V0=action["V0"],
            phi=action["phi"],
            theta=action["theta"],
            a=action["a"],
            b=action["b"],
        )
        if not simulate_with_timeout(shot, timeout=self.sim_timeout):
            return 0.0, balls, table
        reward = float(
            analyze_shot_for_reward(
                shot=shot,
                last_state=last_state_snapshot,
                player_targets=my_targets,
            )
        )
        next_balls = {bid: copy.deepcopy(ball) for bid, ball in shot.balls.items()}
        next_table = copy.deepcopy(shot.table)
        return reward, next_balls, next_table

    def _generate_actions(self, balls, my_targets, table, n):
        base = self._clip_action(self._heuristic_action(balls, my_targets, table))
        actions = [base]
        std = {
            "V0": 1.0,
            "phi": 18.0,
            "theta": 10.0,
            "a": 0.12,
            "b": 0.12,
        }
        while len(actions) < n:
            sampled = {
                "V0": base["V0"] + float(np.random.normal(0, std["V0"])),
                "phi": base["phi"] + float(np.random.normal(0, std["phi"])),
                "theta": base["theta"] + float(np.random.normal(0, std["theta"])),
                "a": base["a"] + float(np.random.normal(0, std["a"])),
                "b": base["b"] + float(np.random.normal(0, std["b"])),
            }
            sampled = self._clip_action(sampled)
            actions.append(sampled)
        return actions

    def _select_child(self, node):
        best = None
        best_score = float("-inf")
        for child in node.children:
            if child.visits == 0:
                ucb = float("inf")
            else:
                q = child.value / float(child.visits)
                u = math.sqrt(math.log(node.visits + 1.0) / float(child.visits))
                ucb = q + self.c_puct * u
            if ucb > best_score:
                best_score = ucb
                best = child
        return best

    def _backpropagate(self, node, value):
        cur = node
        v = value
        while cur is not None:
            cur.visits += 1
            cur.value += v
            cur = cur.parent

    def _mcts_search(self, balls, my_targets, table):
        root = MCTSNode(balls, table, my_targets)
        root.untried_actions = self._generate_actions(
            balls, my_targets, table, self.action_samples
        )
        for _ in range(self.n_sims):
            node = root
            depth = 0
            while (
                node.children
                and not node.untried_actions
                and depth < self.max_depth
            ):
                node = self._select_child(node)
                depth += 1
            if node.untried_actions:
                action = node.untried_actions.pop()
                reward, next_balls, next_table = self._simulate_action(
                    node.balls, node.table, node.my_targets, action
                )
                child = MCTSNode(
                    next_balls,
                    next_table,
                    node.my_targets,
                    parent=node,
                    action_from_parent=action,
                )
                child.reward_from_parent = reward
                node.children.append(child)
                self._backpropagate(child, reward)
            else:
                self._backpropagate(node, 0.0)
        if not root.children:
            return self._clip_action(self._heuristic_action(balls, my_targets, table))
        best_child = max(root.children, key=lambda c: c.visits)
        return self._clip_action(best_child.action_from_parent)

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        try:
            remaining_own = [
                bid
                for bid in my_targets
                if bid in balls and balls[bid].state.s != 4
            ]
            if len(remaining_own) == 0:
                my_targets = ["8"]
            action = self._mcts_search(balls, my_targets, table)
            result = {
                "V0": round(float(action["V0"]), 2),
                "phi": round(float(action["phi"]), 2),
                "theta": round(float(action["theta"]), 2),
                "a": round(float(action["a"]), 3),
                "b": round(float(action["b"]), 3),
            }
            print(
                f"[{self.name}] 决策: V0={result['V0']:.2f}, phi={result['phi']:.2f}, "
                f"theta={result['theta']:.2f}, a={result['a']:.3f}, b={result['b']:.3f}"
            )
            return result
        except Exception as e:
            print(f"[{self.name}] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class CEMAgent(Agent):
    def __init__(self, params=None, checkpoint_path=None):
        super().__init__()
        self.name = "CEMHeuristicAgent"
        self.ball_diameter = 0.05715
        self.param_bounds = {
            'w_cue': (0.0, 3.0),
            'w_pocket': (0.0, 4.0),
            'w_block': (0.0, 6.0),
            'w_align': (0.0, 6.0),
            'v0_offset': (0.5, 3.0),
            'v0_scale': (0.5, 6.0),
            'v0_align_scale': (0.0, 3.0),
            'theta': (5.0, 45.0),
            'ghost_factor': (0.7, 1.3),
        }

        self.params = self._default_params()
        if params is not None:
            self.params.update(params)
        if checkpoint_path is not None:
            loaded = self._load_checkpoint(checkpoint_path)
            if loaded is not None:
                self.params.update(loaded)
        self.params = self._clip_params(self.params)

    def _default_params(self):
        return {
            'w_cue': 0.8,
            'w_pocket': 1.6,
            'w_block': 2.0,
            'w_align': 1.5,
            'v0_offset': 1.2,
            'v0_scale': 2.2,
            'v0_align_scale': 0.6,
            'theta': 20.0,
            'ghost_factor': 1.0,
        }

    def _clip_params(self, params):
        clipped = dict(params)
        for k, (lo, hi) in self.param_bounds.items():
            if k not in clipped:
                continue
            clipped[k] = float(np.clip(float(clipped[k]), lo, hi))
        return clipped

    def _load_checkpoint(self, checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'params' in data and isinstance(data['params'], dict):
                return data['params']
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None

    def _get_ball_xy(self, ball_id, balls):
        if ball_id in balls and balls[ball_id].state.s != 4:
            pos = balls[ball_id].state.rvw[0]
            return (float(pos[0]), float(pos[1]))
        return None

    def _dist(self, p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _unit_vec(self, src, dst):
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        n = math.sqrt(dx * dx + dy * dy)
        if n < 1e-9:
            return (0.0, 0.0)
        return (dx / n, dy / n)

    def _dot(self, u, v):
        return u[0] * v[0] + u[1] * v[1]

    def _segment_distance(self, p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        cx = ax + t * abx
        cy = ay + t * aby
        dx = px - cx
        dy = py - cy
        return math.sqrt(dx * dx + dy * dy)

    def _pockets_xy(self, table):
        if table is None or not hasattr(table, 'pockets'):
            return [(0.0, 0.0), (1.42, 0.0), (0.0, 2.84), (1.42, 2.84), (0.71, 0.0), (0.71, 2.84)]
        pockets = []
        for p in table.pockets.values():
            c = p.center
            pockets.append((float(c[0]), float(c[1])))
        return pockets

    def _choose_shot(self, balls, my_targets, table):
        cue_xy = self._get_ball_xy('cue', balls)
        if cue_xy is None:
            return None

        pockets = self._pockets_xy(table)
        ball_radius = 0.5 * self.ball_diameter
        ghost_offset = self.params['ghost_factor'] * (2.0 * ball_radius)

        best = None
        best_score = float('-inf')

        alive_ball_ids = [bid for bid, b in balls.items() if b.state.s != 4 and bid not in ['cue']]
        blockers = [bid for bid in alive_ball_ids if bid not in my_targets]

        for target_id in my_targets:
            target_xy = self._get_ball_xy(target_id, balls)
            if target_xy is None:
                continue
            for pocket_xy in pockets:
                u_tp = self._unit_vec(target_xy, pocket_xy)
                ghost_xy = (target_xy[0] - u_tp[0] * ghost_offset, target_xy[1] - u_tp[1] * ghost_offset)

                d_cue = self._dist(cue_xy, ghost_xy)
                d_pocket = self._dist(target_xy, pocket_xy)

                u_cg = self._unit_vec(cue_xy, ghost_xy)
                align = max(-1.0, min(1.0, self._dot(u_cg, u_tp)))
                align_penalty = 1.0 - align

                blocked_count = 0
                for bid in blockers:
                    bxy = self._get_ball_xy(bid, balls)
                    if bxy is None:
                        continue
                    if self._segment_distance(bxy, cue_xy, ghost_xy) < (2.2 * ball_radius):
                        blocked_count += 1

                score = (
                    -self.params['w_cue'] * d_cue
                    -self.params['w_pocket'] * d_pocket
                    -self.params['w_block'] * blocked_count
                    -self.params['w_align'] * align_penalty
                )

                if score > best_score:
                    best_score = score
                    best = {
                        'target_id': target_id,
                        'pocket_xy': pocket_xy,
                        'ghost_xy': ghost_xy,
                        'd_cue': d_cue,
                        'align_penalty': align_penalty,
                    }
        return best

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining_own) == 0:
            my_targets = ['8']

        shot = self._choose_shot(balls, my_targets, table)
        if shot is None:
            return self._random_action()

        cue_xy = self._get_ball_xy('cue', balls)
        ghost_xy = shot['ghost_xy']
        dx = ghost_xy[0] - cue_xy[0]
        dy = ghost_xy[1] - cue_xy[1]
        phi = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        V0 = (
            self.params['v0_offset']
            + self.params['v0_scale'] * float(shot['d_cue'])
            + self.params['v0_align_scale'] * float(shot['align_penalty'])
        )
        V0 = float(np.clip(V0, 0.5, 8.0))
        theta = float(np.clip(self.params['theta'], 0.0, 90.0))

        action = {
            'V0': round(V0, 2),
            'phi': round(float(phi), 2),
            'theta': round(theta, 2),
            'a': 0.0,
            'b': 0.0,
        }
        print(f"[{self.name}] 决策: V0={action['V0']:.2f}, phi={action['phi']:.2f}, theta={action['theta']:.2f}")
        return action
