"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

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