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
import sys
import concurrent.futures
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import json

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
    # 以下注释内容为原版代码，可在Linux上运行，Windows上需要修改为使用线程池实现
    # 最终提交时需要修改为原版代码class RLAgent(Agent):
    def __init__(self):
        # ...
        self.pop_size = 24       # 每次思考产生的随机方案数量（越大越准，越慢）
        self.n_iters = 2         # 反复优化的轮数（越大越准，越慢）
        self.elite_frac = 0.25   # 筛选比例
    # # 设置超时信号处理器
    # old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    # signal.alarm(timeout)  # 设置超时时间
    
    # try:
    #     pt.simulate(shot, inplace=True)
    #     signal.alarm(0)  # 取消超时
    #     return True
    # except SimulationTimeoutError:
    #     print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
    #     return False
    # except Exception as e:
    #     signal.alarm(0)  # 取消超时
    #     raise e
    # finally:
    #     signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

    if hasattr(signal, 'SIGALRM'):
        # Unix/Linux: 使用 signal 实现 (支持真正的中断)
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            pt.simulate(shot, inplace=True)
            signal.alarm(0)
            return True
        except SimulationTimeoutError:
            print(f"[WARNING] 物理模拟超时（>{timeout}秒）")
            return False
        except Exception as e:
            signal.alarm(0)
            raise e
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: 使用线程池实现 (无法真正杀死线程，但能防止主进程卡死)
        def _task():
            pt.simulate(shot, inplace=True)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_task)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"[WARNING] 物理模拟超时（>{timeout}秒 - Windows后台线程）")
                return False
            except Exception as e:
                # 捕获其他异常
                raise e

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


class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, balls, parent=None, action_from_parent=None):
        self.balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        self.reward_from_parent = 0.0

    def expand(self, action, next_balls, reward):
        child = MCTSNode(next_balls, parent=self, action_from_parent=action)
        child.reward_from_parent = reward
        self.children.append(child)
        return child


class MCTSAgent(RLAgent):
    """基于蒙特卡洛树搜索 (MCTS) 的 Agent"""
    def __init__(self):
        super().__init__()
        self.name = "MCTSAgent"
        self.n_sims = 20        # 每次决策的模拟次数（受限于物理引擎速度，设为较小值）
        self.c_puct = 1.414     # 探索常数
        self.sim_timeout = 2    # 模拟超时时间
        self.max_depth = 3      # 最大搜索深度
        self.action_samples = 8 # 每个节点采样的动作数量

    def _get_untried_actions(self, balls, my_targets, table):
        """生成候选动作列表"""
        actions = []
        
        # 1. 启发式基础动作
        base_action = self._heuristic_action(balls, my_targets, table)
        base_action = self._clip_action(base_action)
        actions.append(base_action)

        # 2. 在基础动作周围进行微扰
        variations = [
            {'phi': 1.5}, {'phi': -1.5},
            {'phi': 5.0}, {'phi': -5.0},
            {'V0': 0.8}, {'V0': -0.8},
            {'theta': 5.0}, {'theta': -5.0},
            {'a': 0.05}, {'a': -0.05},
            {'b': 0.05}, {'b': -0.05}
        ]
        
        for var in variations:
            new_a = dict(base_action)
            for k, v in var.items():
                new_a[k] += v
            new_a = self._clip_action(new_a)
            actions.append(new_a)
        
        # 3. 随机动作补充（增加探索性）
        for _ in range(2):
            actions.append(self._random_action())

        # 截断或采样到指定数量
        if len(actions) > self.action_samples:
            actions = actions[:self.action_samples]
            
        return actions

    def _simulate_transition(self, balls, action, table, my_targets):
        """执行动作并返回 (next_balls, reward)"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        shot.cue.set_state(**action)
        
        success = simulate_with_timeout(shot, timeout=self.sim_timeout)
        if not success:
            return None, -100.0
            
        reward = analyze_shot_for_reward(shot, balls, my_targets)
        return shot.balls, reward

    def _rollout(self, node, my_targets, table):
        """执行快速 Rollout 模拟
        
        策略：
        1. 使用简单的启发式策略模拟后续 1-2 步
        2. 返回累积奖励
        """
        current_balls = node.balls
        total_reward = node.reward_from_parent
        depth = 0
        max_rollout_depth = 1  # 仅多看一步，权衡性能
        gamma = 0.8  # 折扣因子

        # 为了避免无限循环或状态问题，这里做一个简单的拷贝
        temp_targets = list(my_targets)

        while depth < max_rollout_depth:
            # 简单检查游戏是否结束
            # 检查目标球是否都在台面上
            remaining = [bid for bid in temp_targets if bid in current_balls and current_balls[bid].state.s != 4]
            
            # 如果目标球清空了，切换到8号球
            if not remaining and '8' not in temp_targets:
                 temp_targets = ['8']
                 remaining = ['8']

            if not remaining: # 赢了
                total_reward += 100
                break
            
            # 使用启发式动作快速决策
            try:
                action = self._heuristic_action(current_balls, temp_targets, table)
                action = self._clip_action(action)
                
                # 模拟
                next_balls, reward = self._simulate_transition(current_balls, action, table, temp_targets)
                if next_balls is None:
                    break
                    
                total_reward += gamma * reward
                current_balls = next_balls
                depth += 1
            except Exception:
                break
            
        return total_reward

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        try:
            # 初始化根节点
            root = MCTSNode(balls)
            root.untried_actions = self._get_untried_actions(balls, my_targets, table)
            
            # 处理目标球
            remaining_own = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ['8']

            print(f"[{self.name}] 开始搜索 (Sims={self.n_sims})...")

            for i in range(self.n_sims):
                node = root
                depth = 0
                
                # 1. Selection (选择)
                # 当节点已完全扩展且有子节点时，继续向下选择
                while not node.untried_actions and node.children:
                    # UCB 公式选择最佳子节点
                    # Value = exploitation + exploration
                    best_score = float('-inf')
                    best_child = None
                    
                    for child in node.children:
                        exploitation = child.value / (child.visits + 1e-6)
                        exploration = self.c_puct * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6))
                        score = exploitation + exploration
                        
                        if score > best_score:
                            best_score = score
                            best_child = child
                    
                    if best_child:
                        node = best_child
                        depth += 1
                    else:
                        break

                # 2. Expansion (扩展)
                # 如果节点还有未尝试的动作，且未达到最大深度，则扩展一个动作
                if node.untried_actions and depth < self.max_depth:
                    action = node.untried_actions.pop(0)
                    next_balls, reward = self._simulate_transition(node.balls, action, table, my_targets)
                    
                    if next_balls:
                        # 创建子节点
                        node = node.expand(action, next_balls, reward)
                        depth += 1
                
                # 3. Simulation (模拟/评估)
                # 使用 Rollout 进行评估，而不是仅仅看单步 reward
                eval_score = self._rollout(node, my_targets, table)
                
                # 4. Backpropagation (回溯)
                while node:
                    node.visits += 1
                    node.value += eval_score
                    # 可以引入折扣因子 gamma，但在单局台球中，直接累加也合理
                    node = node.parent

            # 决策：选择访问次数最多的子节点
            if not root.children:
                print(f"[{self.name}] 搜索未生成有效子节点，使用启发式动作。")
                return self._clip_action(self._heuristic_action(balls, my_targets, table))
            
            best_child = max(root.children, key=lambda c: c.visits)
            action = best_child.action_from_parent
            
            # 格式化输出
            action = {
                'V0': round(float(action['V0']), 2),
                'phi': round(float(action['phi']), 2),
                'theta': round(float(action['theta']), 2),
                'a': round(float(action['a']), 3),
                'b': round(float(action['b']), 3),
            }
            
            print(f"[{self.name}] 决策 (Visits={best_child.visits}/{self.n_sims}, Val={best_child.value/best_child.visits:.1f}): "
                  f"V0={action['V0']}, phi={action['phi']}, theta={action['theta']}")
            return action

        except Exception as e:
            print(f"[{self.name}] 决策异常: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
