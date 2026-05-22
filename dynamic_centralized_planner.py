#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态场景集中规划器 (Dynamic Centralized Planner)

功能描述:
1. 初始阶段生成若干静态任务，使用贪婪算法规划
2. 动态生成新任务，直到达到最大总任务数
3. 每次新任务到达时触发重规划
4. 重规划时，所有智能体都可以被重新分配（包括正在前往任务的智能体）

核心策略:
- 贪婪算法: 为每个智能体选择最近的可行任务
- 重规划触发: 新任务出现时
- 无保护机制: 所有未在工作中的智能体都可以被重新分配
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv


class DynamicCentralizedPlanner:
    """
    动态集中式规划器
    
    核心功能:
    1. 初始规划: 对初始任务集进行贪婪分配
    2. 动态任务生成: 按照设定策略生成新任务
    3. 实时重规划: 新任务到达时重新分配所有可用智能体
    4. 全面重规划: 所有未在工作中的智能体都可以被重新分配
    """
    

    
    def __init__(self, env, max_total_tasks=50, dynamic_task_arrival_rate=0.3,
                 simulation_time_limit=200, verbose=True, random_seed=None,
                 decision_maker=None, max_waiting_time=20, replan_threshold=4,
                 protect_traveling=False, dynamic_deadline_buffer=8.0,
                 dynamic_deadline_horizon=17.0, dynamic_urgent_probability=0.88):
        """
        初始化动态规划器
        
        Args:
            env: TaskEnv环境实例
            max_total_tasks: 最大总任务数（静态+动态）
            dynamic_task_arrival_rate: 动态任务到达率（任务/分钟）
            simulation_time_limit: 仿真时间上限
            verbose: 是否打印详细信息
            random_seed: 随机种子，用于确保动态任务生成的可重复性（None表示不固定）
            decision_maker: 外部决策器（函数或对象），用于决定智能体的任务分配
                          如果为None，则使用默认的贪婪算法
                          函数签名: decision_maker(agent_id, available_tasks, env) -> task_id or None
            max_waiting_time: 智能体最大等待时间（分钟），超过此时间将触发重规划（默认50分钟）
            replan_threshold: 批量重规划阈值，积累多少个动态任务后触发重规划（默认5）
            protect_traveling: 是否保护在途智能体不被重规划（避免任务协作被打断）
            dynamic_deadline_buffer: 动态任务最短可行时间之外的缓冲
            dynamic_deadline_horizon: 动态任务相对出现时间的最大截止时间
            dynamic_urgent_probability: 动态任务被标记为紧急任务的概率
        """
        self.env = env
        self.max_total_tasks = max_total_tasks
        self.dynamic_task_arrival_rate = dynamic_task_arrival_rate
        self.simulation_time_limit = simulation_time_limit
        self.verbose = verbose
        self.random_seed = random_seed
        self.decision_maker = decision_maker  # 外部决策器
        self.max_waiting_time = max_waiting_time  # 最大等待时间
        self.replan_threshold = replan_threshold  # 批量重规划阈值
        self.protect_traveling = protect_traveling  # 是否保护在途智能体
        self.dynamic_deadline_buffer = float(dynamic_deadline_buffer)
        self.dynamic_deadline_horizon = float(dynamic_deadline_horizon)
        self.dynamic_urgent_probability = float(dynamic_urgent_probability)
        # 统一使用环境的移动耗电速率，避免评估尺度不一致
        self.battery_move_rate = getattr(env, 'battery_consume_moving', None)
        if self.battery_move_rate is None:
            self.battery_move_rate = getattr(env, 'battery_consumption_rate', 1.0)
        
        # 如果设置了随机种子，初始化专用的随机数生成器
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
        
        # 规划历史记录
        self.planning_history = []  # 每次规划的记录
        self.task_arrival_history = []  # 任务到达记录
        self.replan_count = 0  # 重规划次数
        
        # CPU时间追踪
        self.total_planning_time = 0.0  # 总规划CPU时间
        self.initial_planning_time = 0.0  # 初始规划时间
        self.replan_times = []  # 每次重规划的时间
        
        # 状态跟踪
        self.next_task_arrival_time = None  # 下一个任务到达时间
        self.dynamic_tasks_generated = 0  # 已生成的动态任务数
        
        # 批量重规划相关
        self.pending_dynamic_tasks = []  # 待处理的动态任务缓冲区
        self.tasks_since_last_replan = 0  # 上次重规划后到达的任务数
        
        # 重规划保护机制
        self.last_forced_replan_time = -1  # 上次强制重规划的时间
        
    def initialize(self):
        """初始化环境和规划器"""
        self.env.init_state()
        
        # 记录初始任务数
        self.initial_task_count = self.env.tasks_num
        
        # 为所有智能体初始化等待时间追踪和任务队列
        for agent_id, agent in self.env.agent_dic.items():
            agent['last_assigned_time'] = 0.0  # 上次被分配任务的时间
            agent['waiting_start_time'] = None  # 等待开始时间（仅当智能体空闲时有效）
            agent['task_queue'] = []  # 任务队列：[(task_id, estimated_arrival_time), ...]
            agent['task_queue'] = []  # 任务队列：存储待执行的任务ID列表
        
        if self.verbose:
            print("=" * 80)
            print("动态集中规划器 - 批量重规划模式")
            print("=" * 80)
            print(f"初始静态任务数: {self.initial_task_count}")
            print(f"智能体总数: {self.env.agents_num}")
            print(f"种类数: {self.env.species_num}")
            print(f"最大总任务数: {self.max_total_tasks}")
            print(f"动态任务到达率: {self.dynamic_task_arrival_rate} 任务/分钟")
            print(f"重规划阈值: 每 {self.replan_threshold} 个动态任务触发一次重规划")
            print(f"仿真时间上限: {self.simulation_time_limit} 分钟")
            print("=" * 80)
        
        # 生成下一个任务到达时间
        self._schedule_next_task_arrival()
    
    def _schedule_next_task_arrival(self):
        """调度下一个动态任务的到达时间（泊松过程）"""
        if self.env.tasks_num >= self.max_total_tasks:
            self.next_task_arrival_time = None  # 不再生成新任务
            return
        
        # 使用指数分布生成下一个到达间隔（使用专用随机数生成器）
        if self.dynamic_task_arrival_rate > 0:
            interval = self.rng.exponential(1.0 / self.dynamic_task_arrival_rate)
            self.next_task_arrival_time = self.env.current_time + interval
        else:
            self.next_task_arrival_time = None
    
    def _generate_dynamic_task(self):
        """生成一个新的动态任务"""
        if self.env.tasks_num >= self.max_total_tasks:
            return None
        
        appear_time = self.env.current_time
        
        # 使用专用随机数生成器生成任务参数
        task_location = self.rng.uniform(0, 1, size=2)
        
        # 生成任务需求（根据环境设置）
        if self.env.binary_task:
            task_requirements = self.rng.randint(0, 2, size=self.env.traits_dim)
            while np.sum(task_requirements) == 0:
                task_requirements = self.rng.randint(0, 2, size=self.env.traits_dim)
        else:
            task_requirements = self.rng.randint(0, self.env.max_task_size, size=self.env.traits_dim)
            while np.sum(task_requirements) == 0:
                task_requirements = self.rng.randint(0, self.env.max_task_size, size=self.env.traits_dim)
        
        # 生成持续时间
        task_duration = float(self.rng.uniform(0, 1) * self.env.duration_scale)
        
        # 计算截止时间 - 动态任务使用适度紧张的deadline
        max_speed = 0.2
        distances_to_depots = [
            np.linalg.norm(task_location - self.env.depot_dic[s]['location']) 
            for s in range(self.env.species_num)
        ]
        max_distance = max(distances_to_depots)
        
        # 动态任务的deadline：默认略紧；可由特定benchmark配置放松。
        min_deadline = appear_time + max_distance / max_speed + self.dynamic_deadline_buffer
        max_deadline = appear_time + self.dynamic_deadline_horizon
        if max_deadline < min_deadline:
            max_deadline = min_deadline
        task_deadline = float(self.rng.uniform(min_deadline, max_deadline))
        
        # 生成紧急程度
        task_is_urgent = bool(self.rng.rand() < self.dynamic_urgent_probability)
        if not task_is_urgent:
            task_deadline = appear_time + self.dynamic_deadline_horizon
        
        # 添加动态任务（传入所有参数以避免内部随机生成）
        new_task_ids = self.env.add_dynamic_task(
            num_tasks=1,
            appear_time=appear_time,
            location=task_location.reshape(1, -1),
            requirements=task_requirements.reshape(1, -1),
            duration=task_duration,
            deadline=task_deadline,
            is_urgent=task_is_urgent
        )
        
        if new_task_ids:
            task_id = new_task_ids[0]
            self.dynamic_tasks_generated += 1
            
            # 添加到待处理缓冲区
            self.pending_dynamic_tasks.append(task_id)
            self.tasks_since_last_replan += 1
            
            # 记录任务到达
            self.task_arrival_history.append({
                'time': appear_time,
                'task_id': task_id,
                'location': self.env.task_dic[task_id]['location'].copy(),
                'requirements': self.env.task_dic[task_id]['requirements'].copy(),
                'deadline': self.env.task_dic[task_id]['deadline']
            })
            
            if self.verbose:
                print(f"\n[动态任务到达] 时间 {appear_time:.2f}")
                print(f"  任务ID: {task_id}")
                print(f"  位置: {self.env.task_dic[task_id]['location']}")
                print(f"  截止时间: {self.env.task_dic[task_id]['deadline']:.2f}")
                print(f"  当前总任务数: {self.env.tasks_num}/{self.max_total_tasks}")
                print(f"  待处理缓冲区: {len(self.pending_dynamic_tasks)} 个任务")
                print(f"  距离下次重规划: {self.replan_threshold - self.tasks_since_last_replan} 个任务")
            
            # 调度下一个任务
            self._schedule_next_task_arrival()
            
            return task_id
        
        return None

    def _dispatch_to_charge(self, agent_id, plan_time=None, planning_record=None):
        """派往充电站，复用集中规划器的充电派发逻辑"""
        if plan_time is None:
            plan_time = self.env.current_time
        agent = self.env.agent_dic[agent_id]
        if agent.get('current_task') == -999:
            return None
        charging_station_loc = agent['charging_station']
        agent['current_task'] = -999
        agent['assigned'] = True
        agent['is_moving'] = True
        agent['is_charging'] = False
        agent['last_assigned_time'] = plan_time
        agent['waiting_start_time'] = None
        distance = np.linalg.norm(agent['location'] - charging_station_loc)
        travel_time = distance / agent['velocity']
        arrival_time = plan_time + travel_time
        agent['next_decision'] = arrival_time
        if planning_record is not None:
            planning_record['assignments'][agent_id] = {
                'task_id': -999,
                'charging_station': charging_station_loc.copy(),
                'distance': distance,
                'arrival_time': arrival_time,
                'is_charging': True,
                'battery': float(agent.get('battery', 0.0))  # 记录派发时电量
            }
        return arrival_time
    
    def _get_agent_status(self, agent_id):
        """
        获取智能体状态
        
        Returns:
            状态字符串: 'idle', 'traveling', 'waiting', 'working', 'returning', 'charging'
        """
        agent = self.env.agent_dic[agent_id]
        current_task = agent['current_task']
        
        if current_task == -999:
            return 'charging'
        if current_task < 0:
            # 在基地
            return 'idle'
        else:
            # 有任务
            task = self.env.task_dic[current_task]
            
            if agent['contributed']:
                if task.get('feasible_assignment', False) and not task['finished']:
                    # 任务已开始执行
                    return 'working'
                if task['finished']:
                    return 'returning'
                # 已到达任务点，等待其他成员
                return 'waiting'
            # 尚未到达任务点
            return 'traveling'
    
    def _get_idle_agents(self, force_waiting=False):
        """
        获取所有可重规划的智能体
        
        在动态多智能体任务分配中，允许重规划以下状态的智能体：
        - 'idle': 空闲智能体
        - 'traveling': 正在前往任务的智能体（可以改变目标）
        - 'waiting': 已到达任务点但任务尚未开始的智能体（仅等待超时才可重规划）
        
        不可重规划的智能体：
        - 'returning': 正在返回基地
        - 'working': 任务已开始执行的智能体
        - 'charging': 正在充电或前往充电站的智能体
        
        Args:
            force_waiting: 是否强制将等待中的智能体纳入重规划
        """
        idle_agents = []
        for agent_id, agent in self.env.agent_dic.items():
            status = self._get_agent_status(agent_id)
            if status == 'idle':
                idle_agents.append(agent_id)
            elif status == 'traveling':
                # 保护在途智能体，避免频繁重规划导致协作失败
                if not self.protect_traveling:
                    idle_agents.append(agent_id)
            elif status == 'waiting':
                if force_waiting:
                    idle_agents.append(agent_id)
                else:
                    # 等待超时才允许重规划
                    if agent.get('waiting_start_time') is not None:
                        waiting_time = self.env.current_time - agent['waiting_start_time']
                        if waiting_time >= self.max_waiting_time:
                            idle_agents.append(agent_id)
        return idle_agents

    def _recompute_task_status(self, task):
        """根据当前成员智能体能力，重新计算任务剩余需求。"""
        if not task['members']:
            task['status'] = task['requirements'].copy()
            return task['status']
        total_abilities = np.zeros(self.env.traits_dim)
        for member_id in task['members']:
            total_abilities += self.env.agent_dic[member_id]['abilities']
        task['status'] = task['requirements'] - total_abilities
        return task['status']

    def _task_ready_to_start(self, task):
        """判断任务是否满足开始执行的条件。"""
        if task.get('feasible_assignment', False) or task['finished']:
            return False
        remaining = self._recompute_task_status(task)
        if not np.all(remaining <= 0):
            return False
        for member_id in task['members']:
            member = self.env.agent_dic[member_id]
            if member['current_task'] != task['ID'] or not member['contributed']:
                return False
        return True

    def _start_task_execution(self, task, start_time):
        """启动任务执行，并锁定成员智能体直到任务完成。"""
        task['time_start'] = float(start_time)
        task['time_finish'] = float(start_time + task['time'])
        task['feasible_assignment'] = True
        for member_id in task['members']:
            member = self.env.agent_dic[member_id]
            member['assigned'] = True
            member['is_moving'] = False
            member['next_decision'] = task['time_finish']
            member['waiting_start_time'] = None

    def _try_assign_next_task_from_queue(self, agent_id, plan_time, planning_record=None):
        """
        尝试从任务队列中为智能体派发下一个任务。
        """
        agent = self.env.agent_dic[agent_id]
        task_queue = list(agent.get('task_queue', []))
        if not task_queue:
            return False

        # 移除已完成的当前任务
        if agent.get('current_task', -1) >= 0 and task_queue and task_queue[0] == agent['current_task']:
            task_queue.pop(0)

        # 清理无效任务，寻找可执行的下一任务
        while task_queue:
            next_task_id = task_queue[0]
            task = self.env.task_dic.get(next_task_id)
            if task is None or task.get('finished', False):
                task_queue.pop(0)
                continue
            if task.get('feasible_assignment', False):
                task_queue.pop(0)
                continue
            if not np.any((agent['abilities'] > 0) & (task['status'] > 0)):
                task_queue.pop(0)
                continue
            if not self.env.can_reach_with_battery(agent_id, task):
                # 电量不足时保留队列，等待后续重规划或充电
                break
            # 找到可执行任务
            distance = np.linalg.norm(agent['location'] - task['location'])
            travel_time = distance / agent['velocity']
            arrival_time = plan_time + travel_time
            agent['current_task'] = next_task_id
            agent['assigned'] = True
            agent['contributed'] = False
            agent['is_moving'] = True
            agent['last_assigned_time'] = plan_time
            agent['waiting_start_time'] = None
            agent['next_decision'] = arrival_time
            agent['task_queue'] = task_queue
            if planning_record is not None:
                planning_record['assignments'][agent_id] = {
                    'task_id': next_task_id,
                    'task_queue': task_queue,
                    'distance': distance,
                    'arrival_time': arrival_time,
                    'battery': float(agent.get('battery', 0.0))
                }
            return True

        agent['task_queue'] = task_queue
        return False
    
    def _get_available_tasks(self):
        """
        获取可分配的任务
        
        Returns:
            list: 可分配任务的ID列表
        """
        available_tasks = []
        
        for task_id, task in self.env.task_dic.items():
            # 任务必须满足以下条件:
            # 1. 尚未完成
            # 2. 已经出现（appear_time <= current_time）
            # 3. 尚未被充分分配（仍有需求未满足）
            
            if (not task['finished'] and 
                task['appear_time'] <= self.env.current_time and
                np.sum(task['status']) > 0):  # 还有需求未满足
                available_tasks.append(task_id)
        
        return available_tasks
    
    def _has_useful_tasks_for_agent(self, agent_id):
        """
        检查是否还有任务需要该智能体的能力
        
        只检查能力匹配，不检查电量（因为智能体可以去充电）
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            bool: 如果还有任务需要该智能体的能力则返回True，否则返回False
        """
        agent = self.env.agent_dic[agent_id]
        agent_abilities = agent['abilities']
        
        # 遍历所有未完成的任务
        for task_id, task in self.env.task_dic.items():
            if (not task['finished'] and 
                task['appear_time'] <= self.env.current_time and
                np.sum(task['status']) > 0):
                
                # 检查能力匹配：智能体是否能为该任务贡献能力
                can_contribute = np.any(
                    (agent_abilities > 0) & (task['status'] > 0)
                )
                
                if can_contribute:
                    return True  # 找到了需要该智能体的任务
        
        return False  # 没有任务需要该智能体
    
    def _greedy_assign_agent(self, agent_id, available_tasks):
        """
        为单个智能体进行任务分配
        
        如果设置了外部decision_maker，则使用外部决策器；
        否则使用默认的贪婪算法。
        
        策略 (默认贪婪算法): 
        1. 首先检查电量，如果电量不足，返回None（后续会安排充电）
        2. 选择距离最近的可选任务（与 task_env.py 的贪婪算法一致）
        
        注意: 虽然原始贪婪算法不检查能力，但在多智能体协同场景中需要确保能够贡献
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务列表
            
        Returns:
            selected_task_id or None (None表示需要充电或无可用任务)
        """
        if not available_tasks:
            return None
        
        # 如果设置了外部决策器，使用外部决策器
        if self.decision_maker is not None:
            try:
                return self.decision_maker(agent_id, available_tasks, self.env)
            except Exception as e:
                if self.verbose:
                    print(f"警告: 外部决策器出错 ({e})，回退到贪婪算法")
                # 出错时回退到贪婪算法
                pass
        
        # 默认使用贪婪算法
        agent = self.env.agent_dic[agent_id]
        
        # 检查电量是否过低
        if self.env.check_battery_critical(agent_id):
            # 电量过低，需要充电，不分配任务
            return None
        
        agent_abilities = agent['abilities']
        
        # 贪心选择：寻找距离最近的可选任务
        min_dist = np.inf  # 初始化最小距离为无穷大
        best_task = None   # 最优任务
        
        # 遍历所有可用任务
        for task_id in available_tasks:
            task = self.env.task_dic[task_id]
            task_status = task['status']
            
            # 检查能力匹配: 智能体至少能满足一项任务需求
            can_contribute = np.any(
                (agent_abilities > 0) & (task_status > 0)
            )
            
            if not can_contribute:
                continue
            
            # 检查电量是否足够到达任务点并返回充电站
            if not self.env.can_reach_with_battery(agent_id, task):
                continue
            
            # 计算到任务的距离（使用 task_env.py 的方法）
            dist = self.env.calculate_eulidean_distance(agent, task)
            
            # 更新最小距离和最优任务
            if dist < min_dist:
                min_dist = dist
                best_task = task_id
        
        return best_task
    
    def _assign_task_queue(self, agent_id, available_tasks, plan_time, max_queue_length=5):
        """
        为单个智能体贪婪地分配一个任务序列
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务列表
            plan_time: 规划时间
            max_queue_length: 队列最大长度
        
        Returns:
            task_queue: [(task_id, estimated_arrival_time), ...]
        """
        agent = self.env.agent_dic[agent_id]
        task_queue = []
        current_location = agent['location'].copy()
        current_time = plan_time
        current_battery = agent['battery']
        
        # 创建可用任务的副本，避免修改原列表
        remaining_tasks = set(available_tasks)
        
        for _ in range(max_queue_length):
            if not remaining_tasks:
                break
            
            # 从剩余任务中贪婪选择最近的任务
            best_task = None
            min_dist = float('inf')
            
            for task_id in remaining_tasks:
                task = self.env.task_dic[task_id]
                
                # 检查能力匹配
                can_contribute = np.any(
                    (agent['abilities'] > 0) & (task['status'] > 0)
                )
                if not can_contribute:
                    continue
                
                # 计算距离
                dist = np.linalg.norm(current_location - task['location'])
                
                # 估算需要的电量（去任务点 + 返回充电站）
                to_task_battery = dist / agent['velocity'] * self.battery_move_rate
                to_station_battery = np.linalg.norm(task['location'] - agent['charging_station']) / agent['velocity'] * self.battery_move_rate
                required_battery = to_task_battery + to_station_battery + 10  # 10%安全余量
                
                # 检查电量是否足够
                if current_battery < required_battery:
                    continue
                
                # 更新最优任务
                if dist < min_dist:
                    min_dist = dist
                    best_task = task_id
            
            if best_task is None:
                # 没有合适的任务，停止分配
                break
            
            # 添加到队列
            task = self.env.task_dic[best_task]
            travel_time = min_dist / agent['velocity']
            arrival_time = current_time + travel_time
            
            task_queue.append((best_task, arrival_time))
            
            # 更新状态
            current_location = task['location'].copy()
            current_time = arrival_time + 1.0  # 假设任务执行时间为1分钟
            current_battery -= min_dist / agent['velocity'] * self.battery_move_rate
            
            # 从可用任务中移除
            remaining_tasks.remove(best_task)
        
        return task_queue
    
    def _plan(self, replan=False, force_waiting=False):
        """
        执行规划（批量分配模式）
        
        核心策略：
        1. 为所有可重规划的智能体分配任务
        2. 每次规划为**所有未完成任务**重新分配智能体
        3. 使用轮询方式为每个智能体分配多个任务（任务序列）
        4. 在重规划时，取消之前的分配并重新分配
        5. 使用外部决策器（如果提供）或默认贪婪算法
        
        Args:
            replan: 是否为重规划
        """
        import time
        planning_start_time = time.time()  # 记录CPU时间开始
        
        plan_time = self.env.current_time
        
        # 获取可重规划智能体和可用任务
        replanning_agents = self._get_idle_agents(force_waiting=force_waiting)
        available_tasks = self._get_available_tasks()
        
        # 如果是重规划，先取消这些智能体之前的任务分配
        if replan:
            for agent_id in replanning_agents:
                agent = self.env.agent_dic[agent_id]
                old_task_id = agent['current_task']
                
                # 如果智能体之前有分配的任务，需要清理
                if old_task_id >= 0:
                    old_task = self.env.task_dic.get(old_task_id)
                    was_contributed = agent['contributed']
                    
                    # 如果智能体已经贡献了能力，需要从任务members中移除
                    if old_task and was_contributed and agent_id in old_task['members']:
                        # 从任务的members中移除
                        old_task['members'].remove(agent_id)
                        # 恢复任务剩余需求
                        self._recompute_task_status(old_task)
                    
                    # 重置智能体状态
                    agent['current_task'] = -agent['species'] - 1
                    agent['assigned'] = False
                    agent['contributed'] = False
                    agent['next_decision'] = float('inf')
                    agent['is_moving'] = False  # 停止移动
                    agent['waiting_start_time'] = None
                    agent['task_queue'] = []
                    # 智能体位置保持不变（可能在任务点或路上）
                    
                    if self.verbose:
                        status = "已贡献" if was_contributed else "前往中"
                        print(f"  [取消分配] 智能体 {agent_id} 从任务 {old_task_id} 释放 ({status})")
        
        # 记录本次规划
        planning_record = {
            'time': plan_time,
            'type': 'replan' if replan else 'initial',
            'replanning_agents': len(replanning_agents),
            'available_tasks': len(available_tasks),
            'assignments': {}
        }
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"{'批量重规划' if replan else '初始规划'} - 时间 {plan_time:.2f}")
            print(f"{'=' * 80}")
            print(f"可重规划智能体: {len(replanning_agents)}")
            print(f"可用任务: {len(available_tasks)}")
        
        # 为每个可重规划智能体分配任务或充电
        assignments_made = 0
        charging_sent = 0
        assigned_tasks = set()  # 记录已分配的任务，避免重复
        
        # 首先，检查所有智能体的电量，电量过低的优先送去充电
        for agent_id in replanning_agents:
            agent = self.env.agent_dic[agent_id]
            
            if self.env.check_battery_critical(agent_id):
                # 电量过低，派往充电站
                charging_station_loc = agent['charging_station']
                
                # 更新智能体状态
                agent['current_task'] = -999  # 特殊标记：充电中
                agent['assigned'] = True
                agent['is_moving'] = True
                agent['is_charging'] = False  # 路上还未开始充电
                agent['last_assigned_time'] = plan_time  # 记录分配时间
                agent['waiting_start_time'] = None  # 清除等待时间
                
                # 计算到达充电站的时间
                distance = np.linalg.norm(agent['location'] - charging_station_loc)
                travel_time = distance / agent['velocity']
                arrival_time = plan_time + travel_time
                
                # 更新下次决策时间（到达充电站）
                agent['next_decision'] = arrival_time
                
                # *** 关键修复：将充电事件记录到 assignments 中 ***
                planning_record['assignments'][agent_id] = {
                    'task_id': -999,  # 特殊标记：充电任务
                    'charging_station': charging_station_loc.copy(),
                    'distance': distance,
                    'arrival_time': arrival_time,
                    'is_charging': True,
                    'battery': float(agent.get('battery', 0.0))  # 记录派发时电量
                }
                
                charging_sent += 1
                
                if self.verbose:
                    print(f"  智能体 {agent_id} → 充电站 (电量: {agent['battery']:.1f}%, "
                          f"距离: {distance:.3f}, 到达: {arrival_time:.2f})")
        
        # 过滤掉已经派去充电的智能体
        available_agents = [aid for aid in replanning_agents 
                           if self.env.agent_dic[aid]['current_task'] != -999]
        
        # 使用轮询方式为所有智能体分配任务
        # 每轮为每个智能体分配一个任务，直到没有任务或没有智能体可分配
        remaining_tasks = set(available_tasks)
        remaining_requirements = {}
        for task_id in available_tasks:
            task = self.env.task_dic[task_id]
            self._recompute_task_status(task)
            remaining_requirements[task_id] = task['status'].copy()
        agent_assignments = {aid: [] for aid in available_agents}  # 每个智能体的任务队列

        # 优先进行任务联盟构建（确保任务需求在同一规划周期内尽量被满足）
        if self.decision_maker is None:
            unassigned_agents = set(available_agents)
            task_order = sorted(
                list(remaining_tasks),
                key=lambda tid: (
                    0 if self.env.task_dic[tid]['members'] else 1,
                    self.env.task_dic[tid]['deadline']
                )
            )
            for task_id in task_order:
                if task_id not in remaining_tasks:
                    continue
                while unassigned_agents:
                    remaining = remaining_requirements.get(task_id)
                    if remaining is None:
                        remaining = self.env.task_dic[task_id]['status'].copy()
                    if np.all(remaining <= 0):
                        remaining_tasks.remove(task_id)
                        break
                    best_agent = None
                    min_dist = np.inf
                    for agent_id in list(unassigned_agents):
                        agent = self.env.agent_dic[agent_id]
                        can_contribute = np.any((agent['abilities'] > 0) & (remaining > 0))
                        if not can_contribute:
                            continue
                        if not self.env.can_reach_with_battery(agent_id, self.env.task_dic[task_id]):
                            continue
                        dist = self.env.calculate_eulidean_distance(
                            agent, self.env.task_dic[task_id]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_agent = agent_id
                    if best_agent is None:
                        break
                    agent_assignments[best_agent].append(task_id)
                    unassigned_agents.remove(best_agent)
                    contribution = np.minimum(
                        self.env.agent_dic[best_agent]['abilities'],
                        remaining
                    )
                    remaining = remaining - contribution
                    remaining_requirements[task_id] = remaining
                if task_id in remaining_tasks:
                    remaining = remaining_requirements.get(task_id)
                    if remaining is not None and np.all(remaining <= 0):
                        remaining_tasks.remove(task_id)
        
        max_rounds = 10  # 最多进行10轮分配，避免无限循环
        for round_num in range(max_rounds):
            if not remaining_tasks:
                break
            
            assigned_this_round = False
            
            for agent_id in available_agents:
                if not remaining_tasks:
                    break
                
                agent = self.env.agent_dic[agent_id]
                
                # 使用外部决策器或默认贪婪算法选择任务
                current_available = list(remaining_tasks)
                if self.decision_maker is not None:
                    # 调用外部决策器（HRLF），仅决定“选哪个任务”
                    try:
                        selected_task = self.decision_maker(
                            agent_id, current_available, self.env, remaining_requirements
                        )
                    except TypeError:
                        selected_task = self.decision_maker(agent_id, current_available, self.env)
                else:
                    # 使用默认贪婪算法（支持同任务多智能体协作）
                    selected_task = None
                    min_dist = np.inf
                    agent_abilities = agent['abilities']
                    for task_id in current_available:
                        if task_id in agent_assignments[agent_id]:
                            continue
                        task = self.env.task_dic[task_id]
                        remaining = remaining_requirements.get(task_id, task['status'])
                        can_contribute = np.any((agent_abilities > 0) & (remaining > 0))
                        if not can_contribute:
                            continue
                        if not self.env.can_reach_with_battery(agent_id, task):
                            continue
                        dist = self.env.calculate_eulidean_distance(agent, task)
                        if dist < min_dist:
                            min_dist = dist
                            selected_task = task_id
                
                if selected_task is not None:
                    # 对HRLF与贪婪统一“多智能体协作”更新逻辑
                    if selected_task in agent_assignments[agent_id]:
                        continue
                    
                    task = self.env.task_dic[selected_task]
                    remaining = remaining_requirements.get(selected_task, task['status'])
                    agent_abilities = agent['abilities']
                    
                    # 防止外部决策器选择不可贡献或无法到达的任务
                    can_contribute = np.any((agent_abilities > 0) & (remaining > 0))
                    if not can_contribute:
                        continue
                    if not self.env.can_reach_with_battery(agent_id, task):
                        continue
                    
                    # 分配任务并更新剩余需求
                    agent_assignments[agent_id].append(selected_task)
                    contribution = np.minimum(agent_abilities, remaining)
                    remaining = remaining - contribution
                    remaining_requirements[selected_task] = remaining
                    if np.all(remaining <= 0):
                        remaining_tasks.remove(selected_task)
                    assigned_this_round = True
            
            # 如果本轮没有任何分配，提前结束
            if not assigned_this_round:
                break
        
        # 为每个智能体设置第一个任务
        for agent_id in available_agents:
            agent = self.env.agent_dic[agent_id]
            
            # 跳过已经去充电的智能体
            if agent['current_task'] == -999:
                continue
            
            task_queue = agent_assignments[agent_id]
            
            if len(task_queue) > 0:
                agent['task_queue'] = list(task_queue)
                # 分配第一个任务
                first_task = task_queue[0]
                task = self.env.task_dic[first_task]
                
                # 更新智能体状态
                agent['current_task'] = first_task
                agent['assigned'] = True
                agent['contributed'] = False  # 尚未到达和贡献
                agent['is_moving'] = True  # 开始移动
                agent['last_assigned_time'] = plan_time  # 记录分配时间
                agent['waiting_start_time'] = None  # 清除等待时间
                
                # 计算到达时间
                distance = np.linalg.norm(
                    agent['location'] - task['location']
                )
                travel_time = distance / agent['velocity']
                arrival_time = plan_time + travel_time
                
                # 更新下次决策时间
                agent['next_decision'] = arrival_time
                
                # 记录分配
                planning_record['assignments'][agent_id] = {
                    'task_id': first_task,
                    'task_queue': task_queue,  # 记录完整队列
                    'distance': distance,
                    'arrival_time': arrival_time,
                    'battery': float(agent.get('battery', 0.0))  # 记录派发时电量
                }
                
                assignments_made += 1
                
                if self.verbose:
                    queue_str = f" [队列: {len(task_queue)}个任务]" if len(task_queue) > 1 else ""
                    print(f"  智能体 {agent_id} → 任务 {first_task} "
                          f"(距离: {distance:.3f}, 到达: {arrival_time:.2f}){queue_str}")
            else:
                agent['task_queue'] = []
                # 未分配任务，标记等待开始时间
                if agent['waiting_start_time'] is None:
                    agent['waiting_start_time'] = plan_time
        
        if self.verbose and charging_sent > 0:
            print(f"\n  派往充电: {charging_sent} 个智能体")
        
        # 保存规划记录
        self.planning_history.append(planning_record)
        
        if replan:
            self.replan_count += 1
        
        # 记录CPU时间
        planning_end_time = time.time()
        planning_cpu_time = planning_end_time - planning_start_time
        self.total_planning_time += planning_cpu_time
        
        if replan:
            self.replan_times.append(planning_cpu_time)
        else:
            self.initial_planning_time = planning_cpu_time
        
        if self.verbose:
            print(f"\n本次规划分配: {assignments_made} 个任务")
            total_tasks_in_queues = sum(len(q) for q in agent_assignments.values())
            print(f"任务队列总数: {total_tasks_in_queues} 个任务")
            print(f"未分配任务: {len(remaining_tasks)} 个")
            print(f"规划CPU时间: {planning_cpu_time:.4f} 秒")
    
    def _update_environment_step(self):
        """
        执行环境的一个决策步
        
        处理智能体到达任务、完成任务等事件
        """
        # 找到下次需要决策的时间点
        next_decision_times = [
            agent['next_decision'] 
            for agent in self.env.agent_dic.values()
            if agent['next_decision'] < float('inf')
        ]
        
        # 如果没有待决策的智能体，检查是否还有动态任务要到达或未完成任务
        if not next_decision_times:
            # 如果还有动态任务要生成，跳转到下一个任务到达时间
            if self.next_task_arrival_time is not None and self.next_task_arrival_time <= self.simulation_time_limit:
                next_time = self.next_task_arrival_time
            else:
                # 没有动态任务要到达了，检查是否还有未完成的任务
                available_tasks = self._get_available_tasks()
                if len(available_tasks) > 0:
                    # 还有未完成任务，但所有智能体都空闲
                    # 这说明需要重规划，但我们不能在这里无限返回True
                    # 如果在当前时间已经触发过强制重规划，说明任务无法分配（能力不匹配或电量不足等）
                    # 此时应该让智能体去充电，而不是直接结束仿真
                    if self.last_forced_replan_time >= self.env.current_time:
                        # 刚刚在这个时间点已经强制重规划过，但仍然无法分配
                        # 可能原因：1) 能力不匹配（无解）2) 电量不足（需要充电）
                        
                        # 检查是否所有智能体都电量充足
                        low_battery_agents = [
                            aid for aid in self.env.agent_dic.keys()
                            if self.env.check_battery_critical(aid)  # 使用标准的低电量检查
                        ]
                        
                        if len(low_battery_agents) > 0:
                            # 有智能体电量过低，让它们去充电
                            if self.verbose:
                                print(f"\n[智能体充电] 时间 {self.env.current_time:.2f} - "
                                      f"{len(low_battery_agents)} 个智能体电量过低，派往充电站")
                            
                            # 让电量不足的智能体去充电
                            for agent_id in low_battery_agents:
                                agent = self.env.agent_dic[agent_id]
                                if agent['current_task'] < 0:  # 只处理空闲的智能体
                                    charging_station_loc = agent['charging_station']
                                    
                                    agent['current_task'] = -999
                                    agent['assigned'] = True
                                    agent['is_moving'] = True
                                    agent['is_charging'] = False
                                    
                                    distance = np.linalg.norm(agent['location'] - charging_station_loc)
                                    travel_time = distance / agent['velocity']
                                    agent['next_decision'] = self.env.current_time + travel_time
                                    
                                    if self.verbose:
                                        print(f"    智能体 {agent_id} → 充电站 (电量: {agent['battery']:.1f}%)")
                            
                            # 重置强制重规划时间，允许充电后再次重规划
                            self.last_forced_replan_time = -1
                            return True  # 继续运行
                        else:
                            # 所有智能体电量充足，但仍无法分配任务
                            # 可能原因：1) 真的能力不匹配 2) 需要多智能体协同，但其他智能体还在工作
                            
                            # 检查是否真的无解（所有智能体的能力加起来都无法满足任务）
                            truly_impossible = []
                            for task_id in available_tasks:
                                task = self.env.task_dic[task_id]
                                # 计算所有智能体的总能力
                                total_abilities = np.zeros(self.env.traits_dim)
                                for agent in self.env.agent_dic.values():
                                    total_abilities += agent['abilities']
                                
                                # 检查总能力是否能满足任务需求
                                can_satisfy = np.all(total_abilities >= task['status'])
                                if not can_satisfy:
                                    truly_impossible.append(task_id)
                            
                            if len(truly_impossible) > 0:
                                # 确实有任务永久无法完成（能力不匹配）
                                if self.verbose:
                                    print(f"\n[警告] 时间 {self.env.current_time:.2f} - "
                                          f"有 {len(truly_impossible)} 个任务永久无法完成（能力不匹配）")
                                    print(f"无法完成的任务ID: {truly_impossible[:10]}...")
                                # 跳转到仿真时间上限，结束仿真
                                if self.env.current_time < self.simulation_time_limit:
                                    self.env.current_time = self.simulation_time_limit
                                return False
                            else:
                                # 任务理论上可以完成，但当前空闲智能体无法分配
                                # 可能原因：需要多智能体协同，但其他智能体还在工作中或正在路上
                                
                                # 检查是否有智能体不在基地（包括工作中、返回中、充电中）
                                busy_agents = [
                                    aid for aid, agent in self.env.agent_dic.items()
                                    if agent['next_decision'] < float('inf')  # 有下次决策时间的智能体（在移动或工作）
                                ]
                                
                                if len(busy_agents) > 0:
                                    # 还有智能体在忙，等待它们完成
                                    # 找到下一个完成的时间点
                                    busy_decision_times = [
                                        self.env.agent_dic[aid]['next_decision']
                                        for aid in busy_agents
                                    ]
                                    
                                    next_busy_time = min(busy_decision_times)
                                    if self.verbose:
                                        print(f"\n[等待智能体] 时间 {self.env.current_time:.2f} - "
                                              f"有 {len(available_tasks)} 个任务等待 {len(busy_agents)} 个智能体完成活动")
                                        print(f"  下次智能体可用时间: {next_busy_time:.2f}")
                                    # 跳转到下一个智能体可用的时间（但不超过时间上限）
                                    next_time = min(next_busy_time, self.simulation_time_limit)
                                    # 更新时间和电量
                                    self.env.current_time = next_time
                                    self.env.update_all_batteries(next_time)
                                    # 重置强制重规划标记，允许智能体完成后重新规划
                                    self.last_forced_replan_time = -1
                                    return True  # 继续运行
                                
                                # 没有智能体在忙，但也无法分配，这是异常情况
                                # 再检查一次：是否有智能体电量不足（可能不够critical但也不够到达任务）
                                agents_need_charging = []
                                for agent_id in self.env.agent_dic.keys():
                                    agent = self.env.agent_dic[agent_id]
                                    if agent['current_task'] < 0:  # 空闲的智能体
                                        # 检查是否能到达任何一个可用任务
                                        can_reach_any = False
                                        for task_id in available_tasks:
                                            task = self.env.task_dic[task_id]
                                            if self.env.can_reach_with_battery(agent_id, task):
                                                can_reach_any = True
                                                break
                                        
                                        if not can_reach_any:
                                            # 无法到达任何任务，需要充电
                                            agents_need_charging.append(agent_id)
                                
                                if len(agents_need_charging) > 0:
                                    # 有智能体电量不足以到达任务，派它们去充电
                                    if self.verbose:
                                        print(f"\n[强制充电] 时间 {self.env.current_time:.2f} - "
                                              f"{len(agents_need_charging)} 个智能体电量不足以到达任务，派往充电站")
                                    
                                    # *** 创建充电规划记录 ***
                                    charging_record = {
                                        'time': self.env.current_time,
                                        'assignments': {}
                                    }
                                    
                                    for agent_id in agents_need_charging:
                                        agent = self.env.agent_dic[agent_id]
                                        charging_station_loc = agent['charging_station']
                                        
                                        agent['current_task'] = -999
                                        agent['assigned'] = True
                                        agent['is_moving'] = True
                                        agent['is_charging'] = False
                                        
                                        distance = np.linalg.norm(agent['location'] - charging_station_loc)
                                        travel_time = distance / agent['velocity']
                                        arrival_time = self.env.current_time + travel_time
                                        agent['next_decision'] = arrival_time
                                        
                                        # *** 将充电事件记录到 assignments 中 ***
                                        charging_record['assignments'][agent_id] = {
                                            'task_id': -999,  # 特殊标记：充电任务
                                            'charging_station': charging_station_loc.copy(),
                                            'distance': distance,
                                            'arrival_time': arrival_time,
                                            'is_charging': True
                                        }
                                        
                                        if self.verbose:
                                            print(f"    智能体 {agent_id} → 充电站 (电量: {agent['battery']:.1f}%)")
                                    
                                    # *** 保存充电规划记录 ***
                                    self.planning_history.append(charging_record)
                                    
                                    # 重置强制重规划标记
                                    self.last_forced_replan_time = -1
                                    return True  # 继续运行
                                
                                # 真的是异常情况：所有智能体都空闲、电量充足，但仍无法分配
                                if self.verbose:
                                    print(f"\n[异常] 时间 {self.env.current_time:.2f} - "
                                          f"任务理论上可完成但无法分配，且所有智能体都空闲")
                                    print(f"  可能是贪婪算法的局限性导致的")
                                # 跳转到仿真时间上限，结束仿真
                                if self.env.current_time < self.simulation_time_limit:
                                    self.env.current_time = self.simulation_time_limit
                                return False
                    else:
                        # 还未在当前时间触发过强制重规划，返回True让主循环处理
                        return True
                else:
                    # 既没有待决策智能体，也没有动态任务，也没有未完成任务
                    # 任务全部完成，跳转到仿真时间上限
                    if self.env.current_time < self.simulation_time_limit:
                        self.env.current_time = self.simulation_time_limit
                    return False
        else:
            next_time = min(next_decision_times)
        
        # 检查是否超过仿真时间限制
        if next_time > self.simulation_time_limit:
            return False
        
        # 更新时间（会自动更新所有智能体电量）
        old_time = self.env.current_time
        self.env.current_time = next_time
        self.env.update_all_batteries(next_time)
        
        # 找出需要决策的智能体
        deciding_agents = [
            agent_id for agent_id, agent in self.env.agent_dic.items()
            if abs(agent['next_decision'] - next_time) < 1e-6
        ]
        
        # 处理每个智能体的决策
        for agent_id in deciding_agents:
            agent = self.env.agent_dic[agent_id]
            current_task_id = agent['current_task']
            
            # 处理充电情况
            if current_task_id == -999:
                # 到达充电站
                old_location = agent['location'].copy()
                agent['location'] = agent['charging_station'].copy()
                agent['is_moving'] = False
                
                # 更新行驶距离
                distance_traveled = np.linalg.norm(old_location - agent['charging_station'])
                agent['travel_dist'] = agent.get('travel_dist', 0) + distance_traveled
                
                # 充电（瞬时充满）
                agent['battery'] = self.env.initial_battery
                agent['total_charging_times'] += 1
                
                # 充电完成，返回基地
                agent['current_task'] = -agent['species'] - 1
                agent['assigned'] = False
                agent['next_decision'] = float('inf')  # 等待新的规划
                
                if self.verbose:
                    print(f"[充电完成] 时间 {next_time:.2f} - 智能体 {agent_id} "
                          f"(第{agent['total_charging_times']}次充电)")
                
                continue
            
            if current_task_id < 0:
                # 在基地，等待规划
                agent['next_decision'] = float('inf')
                continue
            
            task = self.env.task_dic[current_task_id]
            
            if not agent['contributed']:
                # 刚到达任务点，登记为成员并等待任务开始
                old_location = agent['location'].copy()
                agent['location'] = task['location'].copy()
                agent['contributed'] = True
                agent['is_moving'] = False  # 到达任务点，停止移动
                
                # 更新行驶距离
                distance_traveled = self.env.calculate_eulidean_distance(
                    {'location': old_location}, 
                    task
                )
                agent['travel_dist'] = agent.get('travel_dist', 0) + distance_traveled

                # 记录任务到达时间，便于等待时间等指标统计
                if 'route' not in agent:
                    agent['route'] = []
                if 'arrival_time' not in agent:
                    agent['arrival_time'] = []
                if not agent['route'] or agent['route'][-1] != task['ID']:
                    agent['route'].append(task['ID'])
                    agent['arrival_time'].append(float(next_time))
                
                if agent_id not in task['members']:
                    task['members'].append(agent_id)
                self._recompute_task_status(task)
                
                # 检查任务是否满足开始条件
                if self._task_ready_to_start(task):
                    self._start_task_execution(task, next_time)
                    if self.verbose:
                        print(f"[任务开始] 时间 {next_time:.2f} - 任务 {current_task_id} "
                              f"(持续: {task['time']:.2f} 分钟, 成员: {len(task['members'])})")
                else:
                    # 任务未开始，继续等待（任务点等待不触发超时重规划）
                    agent['next_decision'] = float('inf')
                    agent['waiting_start_time'] = None
            
            else:
                if task.get('feasible_assignment', False) and not task['finished']:
                    # 任务正在执行，等待完成
                    if self.env.current_time >= task['time_finish'] - 1e-6 and not task['finished']:
                        # 任务完成
                        task['finished'] = True
                        task['time_finish'] = next_time
                        
                        # 处理所有参与的智能体
                        queue_record = {
                            'time': next_time,
                            'type': 'queue_follow',
                            'assignments': {}
                        }
                        for member_id in task['members']:
                            member = self.env.agent_dic[member_id]
                            # 优先执行队列中的下一个任务，避免空闲滞留
                            if self._try_assign_next_task_from_queue(member_id, next_time, queue_record):
                                continue
                            
                            # 检查是否还有任务需要该智能体，或者还会有动态任务生成
                            has_current_tasks = self._has_useful_tasks_for_agent(member_id)
                            will_generate_more = (self.env.tasks_num < self.max_total_tasks)
                            
                            if has_current_tasks or will_generate_more:
                                # 还有任务需要该智能体，或还会生成新任务，留在原地等待重规划
                                member['current_task'] = -member['species'] - 1
                                member['contributed'] = False
                                member['assigned'] = False
                                member['next_decision'] = float('inf')  # 等待重规划
                                member['is_moving'] = False  # 停留在当前任务点
                                member['waiting_start_time'] = next_time
                                
                                if self.verbose:
                                    reason = "有可用任务" if has_current_tasks else "等待动态任务"
                                    print(f"  智能体 {member_id} 留在任务点 {member['location']} 等待重规划 ({reason})")
                            else:
                                # 没有任务需要该智能体，且不会再生成新任务，返回基地
                                depot_loc = member['depot']
                                
                                # 计算返回时间
                                distance = np.linalg.norm(
                                    member['location'] - depot_loc
                                )
                                travel_time = distance / member['velocity']
                                member['next_decision'] = next_time + travel_time
                                member['is_moving'] = True  # 开始返回基地
                                
                                if self.verbose:
                                    print(f"  智能体 {member_id} 返回基地 (无可用任务且不再生成新任务)")
                        
                        if queue_record['assignments']:
                            self.planning_history.append(queue_record)
                        
                        if self.verbose:
                            print(f"[任务完成] 时间 {next_time:.2f} - 任务 {current_task_id}")
                    else:
                        agent['next_decision'] = task['time_finish']
                elif task['finished']:
                    # 返回基地
                    old_location = agent['location'].copy()
                    agent['location'] = agent['depot'].copy()
                    agent['current_task'] = -agent['species'] - 1
                    agent['contributed'] = False
                    agent['assigned'] = False
                    agent['next_decision'] = float('inf')  # 等待新的规划
                    agent['is_moving'] = False  # 到达基地，停止移动
                    agent['waiting_start_time'] = None
                    
                    # 更新行驶距离
                    distance_traveled = np.linalg.norm(old_location - agent['depot'])
                    agent['travel_dist'] = agent.get('travel_dist', 0) + distance_traveled
                    
                    if self.verbose:
                        print(f"[返回基地] 时间 {next_time:.2f} - 智能体 {agent_id}")
                else:
                    # 任务未开始，继续等待
                    agent['next_decision'] = float('inf')
        
        return True
    
    def run(self):
        """
        运行动态规划仿真
        
        主循环:
        1. 初始规划
        2. 持续运行环境直到下一个事件（任务到达、等待超时、完成等）
        3. 处理事件并触发重规划
        4. 重复直到完成或超时
        """
        # 初始化
        self.initialize()
        
        # 初始规划
        self._plan(replan=False)

        # 主循环
        iteration = 0
        max_iterations = 100000
        
        while iteration < max_iterations:
            iteration += 1
            
            # 执行环境步进（处理一个决策事件）
            can_continue = self._update_environment_step()
            
            if not can_continue:
                break
            
            # 每次步进后，检查是否触发了需要重规划的事件
            
            # 事件1: 检查是否有新任务到达
            if (self.next_task_arrival_time is not None and
                self.env.current_time >= self.next_task_arrival_time):
                
                # 生成动态任务
                new_task_id = self._generate_dynamic_task()
                
                # 检查是否达到批量重规划阈值
                if new_task_id is not None and self.tasks_since_last_replan >= self.replan_threshold:
                    if self.verbose:
                        print(f"\n[批量重规划触发] 时间 {self.env.current_time:.2f}")
                        print(f"  已积累 {self.tasks_since_last_replan} 个动态任务")
                        print(f"  待处理任务: {len(self.pending_dynamic_tasks)} 个")
                    
                    # 触发重规划
                    self._plan(replan=True)
                    
                    # 重置计数器
                    self.tasks_since_last_replan = 0
                    self.pending_dynamic_tasks.clear()
            
            # 事件2: 检查是否有智能体等待超时（需要重规划）
            need_replan_due_to_timeout = False
            for agent_id, agent in self.env.agent_dic.items():
                if agent['current_task'] < 0 and agent.get('waiting_start_time') is not None:
                    waiting_time = self.env.current_time - agent['waiting_start_time']
                    if (waiting_time >= self.max_waiting_time and
                        self._has_useful_tasks_for_agent(agent_id)):
                        need_replan_due_to_timeout = True
                        if self.verbose:
                            print(f"\n[等待超时] 时间 {self.env.current_time:.2f} - "
                                  f"智能体 {agent_id} 已等待 {waiting_time:.2f} 分钟，触发重规划")
                        break
            
            if need_replan_due_to_timeout:
                self._plan(replan=True)
                # 重规划后，清空缓冲区
                self.tasks_since_last_replan = 0
                self.pending_dynamic_tasks.clear()
            
            # 事件3: 检查是否所有智能体都在等待（无下次决策）但仍有未分配任务
            # 这种情况说明系统可能陷入等待状态，需要强制重规划
            # 保护机制：避免在同一时间点重复触发强制重规划
            waiting_agents = [aid for aid, a in self.env.agent_dic.items() 
                            if a['next_decision'] == float('inf')]
            
            if len(waiting_agents) == self.env.agents_num:  # 所有智能体都在等待
                available_tasks = self._get_available_tasks()
                
                # 检查是否需要强制重规划（且避免在同一时间点重复触发）
                if len(available_tasks) > 0 and self.last_forced_replan_time < self.env.current_time:
                    if self.verbose:
                        print(f"\n[强制重规划] 时间 {self.env.current_time:.2f} - "
                              f"所有 {len(waiting_agents)} 个智能体在等待，但仍有 {len(available_tasks)} 个可用任务")
                    self._plan(replan=True, force_waiting=True)
                    # 重规划后，清空缓冲区
                    self.tasks_since_last_replan = 0
                    self.pending_dynamic_tasks.clear()
                    # 记录本次强制重规划时间，避免重复触发
                    self.last_forced_replan_time = self.env.current_time
            
            # 检查 episode 是否真正完成。
            # 如果后续仍有动态任务待到达，则不能因为当前场上任务清空而提前结束。
            stats = self.env.get_task_statistics()
            future_arrivals_pending = (
                self.env.tasks_num < self.max_total_tasks and
                self.next_task_arrival_time is not None
            )
            if stats['finished_tasks'] == stats['total_tasks'] and not future_arrivals_pending:
                if self.verbose:
                    if self.env.tasks_num >= self.max_total_tasks:
                        print(f"\n✓ 达到最大任务数且所有任务完成！")
                    else:
                        print(f"\n✓ 当前无后续动态任务，所有任务完成！")
                break
        
        # 仿真结束
        self._print_summary()
        
        return self._get_results()
    
    def _print_summary(self):
        """打印仿真总结"""
        stats = self.env.get_task_statistics()
        
        print("\n" + "=" * 80)
        print("仿真总结")
        print("=" * 80)
        print(f"仿真结束时间: {self.env.current_time:.2f} 分钟")
        print(f"\n任务统计:")
        print(f"  初始静态任务: {self.initial_task_count}")
        print(f"  动态生成任务: {self.dynamic_tasks_generated}")
        print(f"  总任务数: {stats['total_tasks']}")
        print(f"  已完成任务: {stats['finished_tasks']}")
        print(f"  未完成任务: {stats['unfinished_tasks']}")
        print(f"  完成率: {stats['finished_tasks']/stats['total_tasks']*100:.1f}%")
        
        print(f"\n规划统计:")
        print(f"  总规划次数: {len(self.planning_history)}")
        print(f"  初始规划: 1")
        print(f"  重规划次数: {self.replan_count}")
        print(f"  平均规划间隔: {self.env.current_time/(len(self.planning_history) or 1):.2f} 分钟")
        
        print(f"\n智能体统计:")
        total_distance = sum(
            agent.get('travel_dist', 0) 
            for agent in self.env.agent_dic.values()
        )
        avg_battery = np.mean([
            agent['battery'] 
            for agent in self.env.agent_dic.values()
        ])
        print(f"  智能体总数: {self.env.agents_num}")
        print(f"  总行驶距离: {total_distance:.2f}")
        print(f"  平均剩余电量: {avg_battery:.2f}%")
        
        print("=" * 80)
    
    def _get_results(self):
        """
        获取仿真结果
        
        Returns:
            dict: 包含所有仿真结果的字典
        """
        stats = self.env.get_task_statistics()
        
        # 计算平均规划时间
        avg_planning_time = self.total_planning_time / len(self.planning_history) if self.planning_history else 0
        avg_replan_time = np.mean(self.replan_times) if self.replan_times else 0
        
        results = {
            'simulation_time': self.env.current_time,
            'initial_tasks': self.initial_task_count,
            'dynamic_tasks_generated': self.dynamic_tasks_generated,
            'total_tasks': stats['total_tasks'],
            'finished_tasks': stats['finished_tasks'],
            'unfinished_tasks': stats['unfinished_tasks'],
            'completion_rate': stats['finished_tasks'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0,
            'planning_count': len(self.planning_history),
            'replan_count': self.replan_count,
            'planning_history': self.planning_history,
            'task_arrival_history': self.task_arrival_history,
            # CPU时间统计
            'total_planning_time': self.total_planning_time,
            'initial_planning_time': self.initial_planning_time,
            'avg_planning_time': avg_planning_time,
            'avg_replan_time': avg_replan_time,
            'env': self.env
        }
        
        return results
    
    def visualize_timeline(self, save_path='dynamic_planning_timeline.png'):
        """
        可视化规划时间线
        
        显示任务到达、规划事件、任务完成的时间轴
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 子图1: 任务到达和完成
        ax1 = axes[0]
        
        # 任务到达时间
        static_tasks = [0] * self.initial_task_count
        dynamic_arrivals = [h['time'] for h in self.task_arrival_history]
        
        ax1.scatter([0]*len(static_tasks), range(len(static_tasks)), 
                   marker='s', s=50, c='blue', alpha=0.6, label='Static Tasks')
        ax1.scatter(dynamic_arrivals, 
                   range(self.initial_task_count, 
                         self.initial_task_count + len(dynamic_arrivals)),
                   marker='o', s=50, c='red', alpha=0.6, label='Dynamic Tasks')
        
        # 任务完成时间
        for task_id, task in self.env.task_dic.items():
            if task['finished']:
                y_pos = task_id
                x_pos = task['time_finish']
                ax1.scatter([x_pos], [y_pos], marker='x', s=100, 
                           c='green', alpha=0.8)
        
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('Task ID', fontsize=11)
        ax1.set_title('Task Arrival and Completion Timeline', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 规划事件
        ax2 = axes[1]
        
        planning_times = [h['time'] for h in self.planning_history]
        planning_types = [h['type'] for h in self.planning_history]
        
        initial_times = [t for t, typ in zip(planning_times, planning_types) 
                        if typ == 'initial']
        replan_times = [t for t, typ in zip(planning_times, planning_types) 
                       if typ == 'replan']
        
        ax2.scatter(initial_times, [1]*len(initial_times), 
                   marker='^', s=200, c='blue', label='Initial Planning', zorder=3)
        ax2.scatter(replan_times, [1]*len(replan_times), 
                   marker='v', s=100, c='orange', label='Replanning', zorder=3)
        
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_yticks([])
        ax2.set_title('Planning Events', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim([0.5, 1.5])
        
        # 子图3: 累计任务完成
        ax3 = axes[2]
        
        # 收集完成时间
        completion_times = sorted([
            task['time_finish'] 
            for task in self.env.task_dic.values() 
            if task['finished']
        ])
        
        cumulative_completed = list(range(1, len(completion_times) + 1))
        
        ax3.plot(completion_times, cumulative_completed, 
                linewidth=2, color='green', marker='o', markersize=4)
        ax3.axhline(y=self.initial_task_count, color='blue', 
                   linestyle='--', alpha=0.5, label=f'Initial Tasks ({self.initial_task_count})')
        ax3.axhline(y=self.env.tasks_num, color='red', 
                   linestyle='--', alpha=0.5, label=f'Total Tasks ({self.env.tasks_num})')
        
        ax3.set_xlabel('Time (minutes)', fontsize=11)
        ax3.set_ylabel('Completed Tasks', fontsize=11)
        ax3.set_title('Cumulative Task Completion', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 时间线可视化已保存: {save_path}")
        
        return fig


def demo_scenario_1():
    """
    演示场景1: 小规模动态任务
    
    配置:
    - 3个种类，每种3个智能体，共9个智能体
    - 初始10个静态任务
    - 最多30个总任务
    - 到达率0.3任务/分钟
    """
    print("=" * 80)
    print("演示场景1: 小规模动态任务")
    print("=" * 80)
    
    # 创建环境
    env = TaskEnv(
        per_species_range=(3, 3),
        species_range=(3, 3),
        tasks_range=(10, 10),
        traits_dim=5,
        seed=42
    )
    
    # 创建规划器
    planner = DynamicCentralizedPlanner(
        env=env,
        max_total_tasks=30,
        dynamic_task_arrival_rate=0.3,
        simulation_time_limit=150,
        verbose=True
    )
    
    # 运行仿真
    results = planner.run()
    
    # 可视化
    planner.visualize_timeline('demo_scenario_1_timeline.png')
    
    return results, planner


def demo_scenario_2():
    """
    演示场景2: 大规模动态任务
    
    配置:
    - 5个种类，每种5个智能体，共25个智能体
    - 初始15个静态任务
    - 最多60个总任务
    - 到达率0.5任务/分钟
    """
    print("\n\n")
    print("=" * 80)
    print("演示场景2: 大规模动态任务")
    print("=" * 80)
    
    # 创建环境
    env = TaskEnv(
        per_species_range=(5, 5),
        species_range=(5, 5),
        tasks_range=(15, 15),
        traits_dim=5,
        seed=100
    )
    
    # 创建规划器
    planner = DynamicCentralizedPlanner(
        env=env,
        max_total_tasks=60,
        dynamic_task_arrival_rate=0.5,
        simulation_time_limit=200,
        verbose=False  # 大规模场景关闭详细输出
    )
    
    # 运行仿真
    results = planner.run()
    
    # 可视化
    planner.visualize_timeline('demo_scenario_2_timeline.png')
    
    return results, planner


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    动态场景集中规划器 - 演示程序                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

核心功能:
1. ✓ 初始贪婪规划
2. ✓ 动态任务生成（泊松过程）
3. ✓ 实时重规划（新任务到达时）
4. ✓ 全面重规划（所有智能体都可以被重新分配）

运行两个演示场景...
""")
    
    # 运行演示场景
    results1, planner1 = demo_scenario_1()
    results2, planner2 = demo_scenario_2()
    
    # 打印对比
    print("\n\n")
    print("=" * 80)
    print("两个场景对比")
    print("=" * 80)
    print(f"{'指标':<30} {'场景1':<20} {'场景2':<20}")
    print("-" * 80)
    print(f"{'智能体数量':<30} {planner1.env.agents_num:<20} {planner2.env.agents_num:<20}")
    print(f"{'初始任务数':<30} {results1['initial_tasks']:<20} {results2['initial_tasks']:<20}")
    print(f"{'最大总任务数':<30} {planner1.max_total_tasks:<20} {planner2.max_total_tasks:<20}")
    print(f"{'实际生成动态任务':<30} {results1['dynamic_tasks_generated']:<20} {results2['dynamic_tasks_generated']:<20}")
    print(f"{'完成任务数':<30} {results1['finished_tasks']:<20} {results2['finished_tasks']:<20}")
    print(f"{'完成率':<30} {results1['completion_rate']*100:.1f}%{'':<15} {results2['completion_rate']*100:.1f}%")
    print(f"{'仿真时间':<30} {results1['simulation_time']:.2f} 分钟{'':<10} {results2['simulation_time']:.2f} 分钟")
    print(f"{'重规划次数':<30} {results1['replan_count']:<20} {results2['replan_count']:<20}")
    print("=" * 80)
    
    print("\n✓ 所有演示完成！")
    print("\n生成的文件:")
    print("  - demo_scenario_1_timeline.png")
    print("  - demo_scenario_2_timeline.png")
