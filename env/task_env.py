import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations, product
import copy

from reward_ablation import normalize_reward_config


class TaskEnv:
    """
    多智能体任务分配环境类
    
    这是一个复杂的多智能体强化学习环境，模拟异构智能体的多任务分配问题（HeteroMRTA）。
    主要特点：
    1. 支持多种类型的智能体，每种类型具有不同的能力
    2. 任务需要特定的能力组合才能完成
    3. 智能体需要协作完成复杂任务
    4. 考虑时间、距离、等待时间等实际约束
    5. 支持可视化和动画生成
    
    环境包含：
    - 多种类型的智能体（不同能力配置）
    - 需要特定能力组合的任务
    - 智能体基地（depot）
    - 时间和空间约束
    """
    
    def __init__(self, per_species_range=(10, 10), species_range=(5, 5), tasks_range=(30, 30), traits_dim=5,
                decision_dim=10, max_task_size=2, duration_scale=5, seed=None, plot_figure=False,
                single_skill=False, binary_task=False, use_deadline=True, reward_config=None):
        """
        初始化任务环境
        
        Args:
            per_species_range: 每种类型智能体的数量范围 (最小值, 最大值)
            species_range: 智能体种类数量范围 (最小值, 最大值)
            tasks_range: 任务数量范围 (最小值, 最大值)
            traits_dim: 能力维度数量，例如5种不同的技能
            decision_dim: 决策空间维度
            max_task_size: 单个任务需要的最大能力值
            duration_scale: 任务持续时间的缩放因子
            seed: 随机种子，用于生成可重现的问题实例
            plot_figure: 是否生成可视化图形
            single_skill: 是否使用单技能智能体（SA），False为多技能（MA）
            binary_task: 是否使用二元任务（BT），False为加性任务（AT）
            use_deadline: 是否在任务状态中包含截止时间特征（用于兼容不同模型）
        """
        # 随机数生成器
        self.rng = None
        
        # 环境参数设置
        self.per_species_range = per_species_range    # 每种智能体数量范围
        self.species_range = species_range            # 智能体种类数量范围
        self.tasks_range = tasks_range                # 任务数量范围
        self.max_task_size = max_task_size            # 任务最大规模
        self.duration_scale = duration_scale          # 持续时间缩放
        self.plot_figure = plot_figure                # 是否绘制图形
        self.single_skill = single_skill              # 是否单技能智能体（SA）
        self.binary_task = binary_task                # 是否二元任务（BT）
        self.use_deadline = use_deadline              # 是否使用截止时间特征
        self.reward_config = normalize_reward_config(reward_config)
        
        # 设置随机种子
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        # 维度设置
        self.traits_dim = traits_dim      # 能力特征维度
        self.decision_dim = decision_dim  # 决策维度

        # 电量系统参数
        self.initial_battery = 100.0      # 初始电量（满电）
        self.battery_consume_idle = 0.05  # 静止/等待时的耗电速率（每分钟）
        self.battery_consume_moving = 2  # 移动时的耗电速率（每分钟）
        self.battery_min_threshold = 5.0  # 最低电量阈值（低于此值必须充电）
        self.charging_time = 0.0          # 充电时间（瞬时完成）

        # 生成环境实体
        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict, self.charging_station_dic = self.generate_env()
        
        # 生成距离矩阵和邻居矩阵
        self.species_distance_matrix, self.species_neighbor_matrix = self.generate_distance_matrix()
        
        # 环境基本信息
        self.tasks_num = len(self.task_dic)        # 任务总数
        self.agents_num = len(self.agent_dic)      # 智能体总数
        self.species_num = len(self.species_dict['number'])  # 智能体种类数
        
        # 协作矩阵：记录智能体-任务分配关系
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))

        # 时间相关参数
        self.current_time = 0           # 当前仿真时间
        self.dt = 0.1                   # 时间步长
        self.max_waiting_time = 200     # 最大等待时间
        self.depot_waiting_time = 0     # 基地等待时间
        self.finished = False           # 是否完成标志
        self.reactive_planning = False  # 是否使用反应式规划

    def random_int(self, low, high, size=None):
        """
        生成随机整数
        
        根据是否设置了随机种子，使用对应的随机数生成器
        
        Args:
            low: 最小值（包含）
            high: 最大值（不包含）
            size: 输出形状
            
        Returns:
            随机整数或整数数组
        """
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        """
        生成随机浮点数矩阵
        
        Args:
            row: 行数
            col: 列数
            
        Returns:
            [0,1)范围内的随机浮点数矩阵
        """
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        """
        从数组中随机选择元素
        
        Args:
            a: 输入数组
            size: 输出大小
            replace: 是否允许重复选择
            
        Returns:
            随机选择的元素
        """
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice

    def generate_task(self, tasks_num):
        """
        生成任务需求矩阵
        
        每个任务都有特定的能力需求，表示为一个向量。
        任务需要对应能力的智能体来完成。
        
        Args:
            tasks_num: 任务数量
            
        Returns:
            tasks_ini: 任务需求矩阵，形状为 (tasks_num, traits_dim)
        """
        if self.binary_task:
            # BT模式：二元任务，需求只能是0或1（是否需要该技能）
            # 确保每个任务至少需要某种能力
            tasks_ini = self.random_int(0, 2, (tasks_num, self.traits_dim))
            while not np.all(np.sum(tasks_ini, axis=1) != 0):
                tasks_ini = self.random_int(0, 2, (tasks_num, self.traits_dim))
        else:
            # AT模式：加性任务，需求可以是0到max_task_size
            # 生成随机任务需求，确保每个任务至少需要某种能力
            tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))
            while not np.all(np.sum(tasks_ini, axis=1) != 0):
                tasks_ini = self.random_int(0, self.max_task_size, (tasks_num, self.traits_dim))
        
        return tasks_ini

    def generate_agent(self, species_num):
        """
        生成智能体种类能力矩阵
        
        每种智能体都有特定的能力配置，不同种类的智能体具有不同的能力组合。
        
        Args:
            species_num: 智能体种类数量
            
        Returns:
            agents_ini: 智能体能力矩阵，形状为 (species_num, traits_dim)
        """
        if self.single_skill:
            # SA模式：单技能智能体，使用 one-hot 编码
            # 每个智能体只具备一种技能
            if species_num > self.traits_dim:
                # 如果种类数超过技能维度，循环使用技能
                agents_ini = np.zeros((species_num, self.traits_dim), dtype=int)
                for i in range(species_num):
                    agents_ini[i, i % self.traits_dim] = 1
            else:
                # 为每个种类分配唯一的技能
                agents_ini = np.eye(species_num, self.traits_dim, dtype=int)
                # 如果种类数少于技能维度，随机打乱以避免总是使用前几个技能
                if species_num < self.traits_dim:
                    skill_indices = self.random_choice(self.traits_dim, species_num, replace=False)
                    temp = np.zeros((species_num, self.traits_dim), dtype=int)
                    for i, skill_idx in enumerate(skill_indices):
                        temp[i, skill_idx] = 1
                    agents_ini = temp
        else:
            # MA模式：多技能智能体，生成0-1的整数能力值
            # 确保每种智能体至少有一种能力且各不相同
            agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))
            while not np.all(np.sum(agents_ini, axis=1) != 0) or np.unique(agents_ini, axis=0).shape[0] != species_num:
                agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

        return agents_ini

    def generate_env(self):
        """
        生成完整的环境实例
        
        这是环境初始化的核心方法，生成所有的任务、智能体、基地、充电站等实体。
        确保生成的环境是可解的（即智能体的总能力能够满足所有任务需求）。
        
        Returns:
            task_dic: 任务字典，包含所有任务信息
            agent_dic: 智能体字典，包含所有智能体信息
            depot_dic: 基地字典，包含所有基地信息
            species_dict: 种类字典，包含智能体种类信息
            charging_station_dic: 充电站字典，包含所有充电站信息
        """
        # 随机生成环境规模参数
        tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
        species_num = self.random_int(self.species_range[0], self.species_range[1] + 1)
        agents_species_num = [self.random_int(self.per_species_range[0], self.per_species_range[1] + 1) 
                             for _ in range(species_num)]

        # 生成智能体能力和任务需求，确保环境可解
        agents_ini = self.generate_agent(species_num)
        tasks_ini = self.generate_task(tasks_num)
        
        # 检查可解性：智能体总能力必须满足所有任务需求
        # SA-BT模式优化：由于单技能+二元任务，使用智能生成确保一次可解
        if self.single_skill and self.binary_task:
            # SA-BT模式：确保每个技能有足够的智能体数量
            # agents_ini 是 one-hot 矩阵，每行只有一个1
            # tasks_ini 是二元矩阵，每个元素是0或1
            
            # 计算每个技能的智能体总数
            skill_counts = np.matmul(agents_species_num, agents_ini)  # (traits_dim,)
            
            # 最多尝试10次生成可解的任务
            max_attempts = 10
            for attempt in range(max_attempts):
                # 生成任务需求
                tasks_ini = self.generate_task(tasks_num)
                
                # 检查每个技能的需求是否超过供给
                task_demands = np.sum(tasks_ini, axis=0)  # (traits_dim,)
                
                if np.all(skill_counts >= task_demands):
                    # 可解，退出循环
                    break
                    
                if attempt == max_attempts - 1:
                    # 最后一次尝试，强制调整任务需求使其可解
                    for skill_idx in range(self.traits_dim):
                        if task_demands[skill_idx] > skill_counts[skill_idx]:
                            # 该技能需求过多，随机减少一些任务的该技能需求
                            excess = int(task_demands[skill_idx] - skill_counts[skill_idx])
                            # 找到需要该技能的任务
                            tasks_with_skill = np.where(tasks_ini[:, skill_idx] == 1)[0]
                            # 随机选择一些任务，移除该技能需求
                            if len(tasks_with_skill) > excess:
                                remove_indices = self.random_choice(
                                    tasks_with_skill, 
                                    min(excess, len(tasks_with_skill)), 
                                    replace=False
                                )
                                for idx in remove_indices:
                                    tasks_ini[idx, skill_idx] = 0
                                    # 确保任务至少有一个技能需求
                                    if np.sum(tasks_ini[idx]) == 0:
                                        # 随机选择一个该任务不需要的技能
                                        available_skills = np.where(tasks_ini[idx] == 0)[0]
                                        if len(available_skills) > 0:
                                            tasks_ini[idx, self.random_choice(available_skills, 1)[0]] = 1
        else:
            # MA或非BT模式：使用原来的while循环
            max_attempts = 100  # 添加最大尝试次数避免死循环
            attempts = 0
            while not np.all(np.matmul(agents_species_num, agents_ini) >= tasks_ini):
                agents_ini = self.generate_agent(species_num)
                tasks_ini = self.generate_task(tasks_num)
                attempts += 1
                if attempts >= max_attempts:
                    # 超过最大尝试次数，使用最后一次生成的结果
                    # 强制调整使其可解
                    agent_totals = np.matmul(agents_species_num, agents_ini)
                    tasks_ini = np.minimum(tasks_ini, agent_totals)
                    break

        # 生成空间位置信息
        depot_loc = self.random_value(species_num, 2)      # 基地位置
        cost_ini = [self.random_value(1, 1) for _ in range(species_num)]  # 智能体成本
        tasks_loc = self.random_value(tasks_num, 2)        # 任务位置
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale  # 任务持续时间

        # ==================== 生成充电站位置 ====================
        # 为每个智能体种类生成一个唯一的充电站
        # 充电站位置在基地附近但不重合，距离基地约0.1-0.2的随机偏移
        charging_station_loc = np.zeros((species_num, 2))
        for s in range(species_num):
            # 在基地周围生成随机偏移
            offset = (self.random_value(1, 2) - 0.5) * 0.3  # [-0.15, 0.15]的偏移
            charging_station_loc[s] = depot_loc[s] + offset.flatten()
            # 确保充电站位置在[0,1]范围内
            charging_station_loc[s] = np.clip(charging_station_loc[s], 0.05, 0.95)

        # ==================== 生成任务截止时间 ====================
        max_speed = 0.2  # 机器人最大速度
        total_agents = sum(agents_species_num)  # 总智能体数量
        
        # 计算每个任务到所有基地的最大距离
        max_distances = np.zeros(tasks_num)
        for i in range(tasks_num):
            distances_to_depots = [np.linalg.norm(tasks_loc[i] - depot_loc[s]) 
                                  for s in range(species_num)]
            max_distances[i] = max(distances_to_depots)
        
        # 计算最小截止时间：基于最远距离 + 缓冲时间
        d_low = (max_distances / max_speed + 10).astype(np.int64) + 1  # 从7提高到10，放松初始任务

        # 计算最大截止时间（适度放松）
        d_high = np.full(tasks_num, 22, dtype=np.int64)  # 最长截止时间从16提高到22

        # 根据任务数量和智能体数量调整分组参数（1或2）
        group_factor = 1 if total_agents >= tasks_num else 2
        d_low = (d_low * (0.30 * group_factor)).astype(np.int64)  # 从0.20提高到0.30，放松
        # d_high 保持为22，不再进行缩放
        
        # 生成基础截止时间（均匀分布）
        deadline_base = (np.random.rand(tasks_num) * (d_high - d_low) + d_low).astype(np.int64) + 1

        # 设置紧急任务和松弛任务的比例为90%:10%（从97%:3%放松到90%:10%）
        n_urgent_tasks = int(0.90 * tasks_num)

        # 随机选择紧急任务
        urgent_indices = np.random.choice(tasks_num, n_urgent_tasks, replace=False)
        is_urgent = np.zeros(tasks_num, dtype=bool)
        is_urgent[urgent_indices] = True
        
        # 为紧急任务分配正常截止时间，为松弛任务分配较长截止时间
        tasks_deadline = np.where(is_urgent, deadline_base, d_high)
        
        # 初始化各类字典
        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()
        charging_station_dic = dict()
        
        # 智能体种类信息
        species_dict['abilities'] = agents_ini      # 各种类能力矩阵
        species_dict['number'] = agents_species_num # 各种类数量

        # 创建任务字典
        for i in range(tasks_num):
            task_dic[i] = {
                'ID': i,                                    # 任务ID
                'requirements': tasks_ini[i, :],            # 任务能力需求
                'members': [],                              # 分配到此任务的智能体列表
                'cost': [],                                 # 每个智能体的成本
                'location': tasks_loc[i, :],                # 任务位置坐标
                'deadline': float(tasks_deadline[i]),       # 任务截止时间
                'is_urgent': bool(is_urgent[i]),            # 是否为紧急任务
                'feasible_assignment': False,               # 是否有可行的智能体分配
                'finished': False,                          # 任务是否完成
                'time_start': 0,                           # 任务开始时间
                'time_finish': 0,                          # 任务结束时间
                'status': tasks_ini[i, :],                 # 当前任务状态（剩余需求）
                'time': float(tasks_time[i, 0]),           # 任务持续时间（修复：使用[i,0]而非[i,:]）
                'sum_waiting_time': 0,                     # 总等待时间
                'efficiency': 0,                           # 任务效率
                'abandoned_agent': [],                     # 被放弃的智能体列表
                'optimized_ability': None,                 # 优化后的能力分配
                'optimized_species': [],                   # 优化后的种类分配
                'appear_time': 0.0,                        # 任务出现时间（动态任务特征）
                'is_dynamic': False                        # 是否为动态添加的任务
            }

        # 创建智能体字典
        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []  # 记录该种类包含的智能体ID
            for j in range(n):
                agent_dic[i] = {
                    'ID': i,                                # 智能体ID
                    'species': s,                           # 所属种类
                    'abilities': agents_ini[s, :],          # 能力向量
                    'location': depot_loc[s, :],            # 当前位置
                    'route': [- s - 1],                     # 路径记录（负数表示基地）
                    'current_task': - s - 1,                # 当前任务（负数表示在基地）
                    'contributed': False,                   # 是否已贡献能力
                    'arrival_time': [0.],                   # 到达时间记录
                    'cost': cost_ini[s],                    # 智能体成本
                    'travel_time': 0,                       # 行驶时间
                    'velocity': 0.2,                        # 移动速度
                    'next_decision': 0,                     # 下次决策时间
                    'depot': depot_loc[s, :],               # 所属基地位置
                    'travel_dist': 0,                       # 总行驶距离
                    'sum_waiting_time': 0,                  # 总等待时间
                    'current_action_index': 0,              # 当前动作索引
                    'decision_step': 0,                     # 决策步数
                    'task_waiting_ratio': 1,                # 任务等待比例
                    'trajectory': [],                       # 轨迹记录
                    'angle': 0,                             # 朝向角度
                    'returned': False,                      # 是否已返回基地
                    'assigned': False,                      # 是否已分配任务
                    'pre_set_route': None,                  # 预设路径
                    'no_choice': False,                     # 是否无可选择的任务
                    # ==================== 电量相关属性 ====================
                    'battery': self.initial_battery,        # 当前电量
                    'battery_history': [self.initial_battery],  # 电量历史记录
                    'charging_station': charging_station_loc[s, :],  # 所属充电站位置
                    'total_charging_times': 0,              # 总充电次数
                    'is_charging': False,                   # 是否正在充电
                    'last_update_time': 0.0,                # 上次电量更新时间
                    'is_moving': False,                     # 是否正在移动
                }
                species_dict[s].append(i)
                i += 1

        # 创建基地字典
        for s in range(species_num):
            depot_dic[s] = {
                'location': depot_loc[s, :],                # 基地位置
                'members': species_dict[s],                 # 基地所属智能体
                'ID': - s - 1                              # 基地ID（负数）
            }

        # ==================== 创建充电站字典 ====================
        for s in range(species_num):
            charging_station_dic[s] = {
                'location': charging_station_loc[s, :],     # 充电站位置
                'species': s,                               # 所属智能体种类
                'ID': - species_num - s - 1,               # 充电站ID（更小的负数）
                'charging_agents': [],                      # 当前正在充电的智能体列表
            }

        return task_dic, agent_dic, depot_dic, species_dict, charging_station_dic

    def generate_distance_matrix(self):
        """
        生成距离矩阵和邻居矩阵
        
        为每种智能体计算到所有任务和基地的距离，用于路径规划和决策。
        
        Returns:
            species_distance_matrix: 各种类智能体的距离矩阵
            species_neighbor_matrix: 各种类智能体的邻居矩阵（按距离排序）
        """
        species_distance_matrix = {}
        species_neighbor_matrix = {}
        
        # 为每种智能体计算距离矩阵
        for species in range(len(self.species_dict['number'])):
            # 构建临时字典：包含基地(-1)和所有任务
            tmp_dic = {-1: self.depot_dic[species], **self.task_dic}
            distances = {}
            
            # 计算任意两点间的距离
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = self.calculate_eulidean_distance(from_node, to_node)

            # 按距离排序生成邻居矩阵
            sorted_distance_matrix = {k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()}
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix
            
        return species_distance_matrix, species_neighbor_matrix

    def reset(self, test_env=None, seed=None):
        """
        重置环境到初始状态
        
        Args:
            test_env: 可选的测试环境配置
            seed: 新的随机种子
        """
        # 设置新的随机种子
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
            
        # 使用指定环境或重新生成环境
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict, self.charging_station_dic = test_env
        else:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict, self.charging_station_dic = self.generate_env()
            
        # 更新环境基本信息
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.finished = False

    def init_state(self):
        """
        初始化环境状态
        
        将所有任务和智能体重置到初始状态，用于开始新的仿真回合。
        """
        # 重置所有任务状态
        for task in self.task_dic.values():
            task.update(
                members=[],                     # 清空分配的智能体
                cost=[],                        # 清空成本记录
                finished=False,                 # 重置完成状态
                status=task['requirements'],    # 恢复原始需求
                feasible_assignment=False,      # 重置可行分配标志
                time_start=0,                   # 重置开始时间
                time_finish=0,                  # 重置结束时间
                sum_waiting_time=0,             # 重置等待时间
                efficiency=0,                   # 重置效率
                abandoned_agent=[]              # 清空放弃的智能体
            )
            
        # 重置所有智能体状态
        for agent in self.agent_dic.values():
            agent.update(
                route=[-agent['species'] - 1],     # 重置路径，起始位置为基地
                location=agent['depot'],            # 重置位置到基地
                contributed=False,                  # 重置贡献标志
                next_decision=0,                    # 重置下次决策时间
                travel_time=0,                      # 重置行驶时间
                travel_dist=0,                      # 重置行驶距离
                arrival_time=[0.],                  # 重置到达时间
                assigned=False,                     # 重置分配状态
                sum_waiting_time=0,                 # 重置等待时间
                current_action_index=0,             # 重置动作索引
                decision_step=0,                    # 重置决策步数
                trajectory=[],                      # 清空轨迹
                angle=0,                           # 重置角度
                returned=False,                     # 重置返回状态
                pre_set_route=None,                 # 清空预设路径
                current_task=-1,                    # 重置当前任务
                task_waiting_ratio=1,               # 重置等待比例
                no_choice=False,                    # 重置无选择状态
                next_action=0,                      # 重置下一动作
                # ==================== 重置电量相关属性 ====================
                battery=self.initial_battery,       # 重置电量为满电
                battery_history=[self.initial_battery],  # 重置电量历史
                total_charging_times=0,             # 重置充电次数
                is_charging=False,                  # 重置充电状态
                last_update_time=0.0,               # 重置上次更新时间
                is_moving=False,                    # 重置移动状态
            )
            
        # 重置基地成员信息
        for depot in self.depot_dic.values():
            depot.update(members=self.species_dict[-depot['ID'] - 1])
            
        # 重置充电站状态
        for station in self.charging_station_dic.values():
            station['charging_agents'] = []
            
        # 重置全局状态
        self.current_time = 0
        self.max_waiting_time = 200
        self.finished = False

    @staticmethod
    def find_by_key(data, target):
        """
        递归查找字典中特定键的所有值
        
        Args:
            data: 要搜索的数据结构
            target: 目标键名
            
        Yields:
            找到的目标键对应的值
        """
        for key, value in data.items():
            if isinstance(value, dict):
                yield from TaskEnv.find_by_key(value, target)
            elif key == target:
                yield value

    @staticmethod
    def get_matrix(dictionary, key):
        """
        从字典中提取指定键的所有值，组成矩阵
        
        这是一个常用的工具函数，用于快速提取所有实体的某个属性。
        
        Args:
            dictionary: 输入字典（如任务字典或智能体字典）
            key: 要提取的键名
            
        Returns:
            key_matrix: 包含所有指定键值的列表
        """
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    def all_tasks_feasibly_assigned(self):
        """返回当前是否所有任务都已有可行分配。"""
        return all(task['feasible_assignment'] for task in self.task_dic.values())

    def all_tasks_finished(self):
        """返回当前是否所有任务都已完成。"""
        return all(task['finished'] for task in self.task_dic.values())

    def all_agents_returned(self):
        """返回当前是否所有智能体都已返回基地。"""
        return all(agent['returned'] for agent in self.agent_dic.values())

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        """
        计算两点间的欧几里得距离
        
        Args:
            agent: 智能体或起点，包含'location'字段
            task: 任务或终点，包含'location'字段
            
        Returns:
            两点间的欧几里得距离
        """
        return np.linalg.norm(agent['location'] - task['location'])

    def update_battery(self, agent_id, time_elapsed):
        """
        更新智能体的电量消耗
        
        根据智能体的状态（移动/静止）和经过的时间计算电量消耗。
        移动状态下的耗电速率更高。
        
        Args:
            agent_id: 智能体ID
            time_elapsed: 从上次更新到现在经过的时间
        """
        agent = self.agent_dic[agent_id]
        
        # 如果正在充电，不消耗电量
        if agent['is_charging']:
            return
        
        # 根据移动状态选择不同的耗电速率
        if agent['is_moving']:
            consume_rate = self.battery_consume_moving
        else:
            consume_rate = self.battery_consume_idle
        
        # 计算电量消耗
        battery_consumed = consume_rate * time_elapsed
        agent['battery'] = max(0.0, agent['battery'] - battery_consumed)
        
        # 记录电量历史
        agent['battery_history'].append(agent['battery'])
        agent['last_update_time'] = self.current_time

    def update_all_batteries(self, new_time):
        """
        更新所有智能体的电量到指定时间
        
        Args:
            new_time: 目标时间点
        """
        for agent_id, agent in self.agent_dic.items():
            time_elapsed = new_time - agent['last_update_time']
            if time_elapsed > 0:
                self.update_battery(agent_id, time_elapsed)

    def check_battery_critical(self, agent_id):
        """
        检查智能体电量是否低于临界值
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            True: 电量低于临界值，需要充电
            False: 电量充足
        """
        return self.agent_dic[agent_id]['battery'] <= self.battery_min_threshold

    def can_reach_with_battery(self, agent_id, destination):
        """
        检查智能体当前电量是否足够到达目的地（并返回充电站）
        
        Args:
            agent_id: 智能体ID
            destination: 目的地字典，包含'location'字段
            
        Returns:
            True: 电量足够
            False: 电量不足，需要先充电
        """
        agent = self.agent_dic[agent_id]
        
        # 计算到目的地的距离
        dist_to_dest = self.calculate_eulidean_distance(agent, destination)
        
        # 计算从目的地到充电站的距离
        charging_station = self.charging_station_dic[agent['species']]
        dist_to_charging = np.linalg.norm(destination['location'] - charging_station['location'])
        
        # 总距离
        total_dist = dist_to_dest + dist_to_charging
        
        # 计算行驶时间
        travel_time = total_dist / agent['velocity']
        
        # 计算需要的电量（移动消耗）
        battery_needed = self.battery_consume_moving * travel_time
        
        # 留出安全余量（1.2倍）
        battery_needed *= 1.2
        
        return agent['battery'] >= battery_needed

    def charge_agent(self, agent_id):
        """
        为智能体充电（瞬时完成）
        
        Args:
            agent_id: 智能体ID
        """
        agent = self.agent_dic[agent_id]
        agent['battery'] = self.initial_battery
        agent['total_charging_times'] += 1
        agent['is_charging'] = False
        agent['battery_history'].append(agent['battery'])
        
        # 从充电站的充电列表中移除
        charging_station = self.charging_station_dic[agent['species']]
        if agent_id in charging_station['charging_agents']:
            charging_station['charging_agents'].remove(agent_id)

    def calculate_optimized_ability(self):
        """
        计算每个任务的优化能力分配方案
        
        这个方法为每个任务计算最优的智能体种类组合，目标是找到能够最有效完成任务的
        智能体配置。该方法考虑了所有可能的智能体种类组合，并基于效率评分选择最佳方案。
        
        核心思想：
        1. 为每个任务枚举所有可能的智能体种类组合
        2. 计算每种组合的有效能力和效率评分
        3. 选择评分最高的组合作为优化方案
        4. 生成种类掩码，指导后续的任务分配决策
        
        Returns:
            species_mask: 种类掩码矩阵，指示每个任务应优先考虑哪些智能体种类
        """
        # 遍历所有任务，为每个任务计算优化的能力分配
        for task in self.task_dic.values():
            task_status = task['status']  # 当前任务的剩余需求
            
            # 获取环境中智能体种类的基本信息
            in_species_num = self.species_dict['number']    # 各种类智能体的数量
            species_ability = self.species_dict['abilities'] # 各种类智能体的能力矩阵
            
            # 生成所有可能的智能体数量组合
            # 为每个种类创建可能的数量选择范围：[0, 1, 2, ..., max_task_size]
            num_set = [list(range(0, self.max_task_size + 1)) for _ in in_species_num]
            
            # 生成所有种类数量的笛卡尔积组合
            # 例如：如果有2个种类，max_task_size=2，则生成：
            # [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
            group_combinations = list(product(*num_set))

            # 存储所有组合的能力和种类信息
            abilities = []      # 各组合的总能力
            contained_spe = []  # 各组合包含的种类（布尔数组）
            
            # 计算每种组合的总能力
            for sample in group_combinations:
                # 初始化该组合的总能力
                ability = np.zeros((1, self.traits_dim))
                
                # 累加各种类的贡献：数量 × 种类能力
                for spe, num in enumerate(sample):
                    ability += sample[spe] * species_ability[spe]
                    
                # 记录该组合包含哪些种类（数量>0的种类）
                contained_spe.append(np.array(sample) > 0)
                abilities.append(ability)

            # 计算有效能力：考虑任务需求的上限约束
            # effective_ability[i,j] = min(task_need[j], available_ability[i,j])
            # 即：有效能力不能超过任务实际需求
            effective_ability = np.maximum(np.minimum(task_status, np.vstack(abilities)), 0)
            
            # 计算效率评分：基于能力利用率和有效贡献
            # score = (effective_ability / total_ability) * effective_ability
            # 这个公式鼓励：1) 高利用率，2) 高有效贡献
            score = np.divide(effective_ability, np.vstack(abilities), 
                             where=np.vstack(abilities) > 0,  # 避免除零错误
                             out=np.zeros_like(np.vstack(abilities), dtype=float)) * effective_ability
            
            # 对每个组合的所有能力维度求和，得到总评分
            score = np.sum(score, axis=1)
            
            # 选择评分最高的组合作为最优方案
            action_index = np.argmax(score)
            
            # 选择评分最高的前2个组合，增加方案的多样性
            group_sort = np.argsort(score)[-2:]
            
            # 保存该任务的优化能力分配
            task['optimized_ability'] = abilities[action_index]
            
            # 生成优化的种类组合：取前2个最佳方案的并集
            optimized_species = []
            for ind in group_sort:
                optimized_species.append(contained_spe[ind])
            
            # 使用逻辑或操作合并多个方案，增加灵活性
            task['optimized_species'] = np.logical_or(*optimized_species)
        
        # 将所有任务的优化种类信息堆叠成矩阵
        # species_mask[i,j] = True 表示任务i应优先考虑种类j的智能体
        species_mask = np.vstack(self.get_matrix(self.task_dic, 'optimized_species'))
        
        return species_mask

    def get_current_agent_status(self, agent):
        """
        获取当前所有智能体的状态信息
        
        为指定的观察者智能体收集环境中所有智能体的当前状态。这个方法为强化学习
        算法提供智能体间的协作信息，帮助做出更好的决策。
        
        状态信息包括：
        1. 智能体能力向量
        2. 到达当前任务的剩余时间
        3. 剩余工作时间
        4. 当前等待时间
        5. 相对位置信息
        6. 分配状态
        
        Args:
            agent: 观察者智能体的信息字典
            
        Returns:
            current_agents: 所有智能体状态矩阵，形状为 (agents_num, status_dim)
        """
        status = []
        
        # 遍历环境中的所有智能体
        for a in self.agent_dic.values():
            # 检查智能体是否正在执行任务（task_id >= 0表示有任务，< 0表示在基地）
            if a['current_task'] >= 0:
                current_task = a['current_task']
                
                # 计算智能体到达当前任务的时间
                arrival_time = self.get_arrival_time(a['ID'], current_task)
                
                # **修复**: 处理arrival_time为inf的情况（动态规划器可能没有更新route）
                # 如果arrival_time是inf，说明route中没有记录该任务
                # 这时应该直接根据距离重新计算到达时间
                if np.isinf(arrival_time):
                    # 重新计算到达时间：当前时间 + 行驶时间
                    distance = np.linalg.norm(a['location'] - self.task_dic[current_task]['location'])
                    travel_time = distance / a['velocity']
                    arrival_time = self.current_time + travel_time
                
                # 计算剩余行驶时间（如果还未到达）
                travel_time = np.clip(arrival_time - self.current_time, a_min=0, a_max=None)
                
                # 根据任务开始时间计算等待时间和剩余工作时间
                if self.current_time <= self.task_dic[current_task]['time_start']:
                    # 任务尚未开始：计算当前等待时间和剩余工作时间
                    current_waiting_time = np.clip(self.current_time - arrival_time, a_min=0, a_max=None)
                    remaining_working_time = np.clip(
                        self.task_dic[current_task]['time_start'] + 
                        self.task_dic[current_task]['time'] - self.current_time, 
                        a_min=0, a_max=None
                    )
                else:
                    # 任务已经开始或结束：无等待时间和工作时间
                    current_waiting_time = 0
                    remaining_working_time = 0
            else:
                # 智能体在基地或无任务：所有时间相关值为0
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            
            # 构建该智能体的完整状态向量
            temp_status = np.hstack([
                np.atleast_1d(a['abilities']),            # 智能体能力向量
                travel_time,                              # 剩余行驶时间
                remaining_working_time,                   # 剩余工作时间
                current_waiting_time,                     # 当前等待时间
                agent['location'] - a['location'],       # 观察者与该智能体的相对位置
                a['assigned'],                            # 该智能体的分配状态
                a['battery'] / self.initial_battery,      # 归一化的电量 [0, 1]
            ])
            status.append(temp_status)
        
        # 将所有智能体状态堆叠成矩阵
        if len(status) == 0:
            # 如果没有智能体，返回空矩阵
            return np.array([]).reshape(0, 0)
        
        current_agents = np.vstack(status)
        return current_agents

    def get_current_task_status(self, agent):
        """
        获取当前所有任务的状态信息
        
        为指定的智能体收集环境中所有任务的当前状态。这个方法为强化学习算法
        提供任务选择的关键信息，帮助智能体做出最优的任务分配决策。
        
        任务状态信息包括：
        1. 任务当前剩余需求
        2. 任务原始需求
        3. 任务持续时间
        4. 到达任务的行驶时间
        5. 智能体与任务的相对位置
        6. 任务是否已有可行分配
        7. (可选) 任务截止时间 - 仅当use_deadline=True时包含
        
        注意：返回的状态列表第一个元素是基地状态，其余为任务状态
        
        Args:
            agent: 观察者智能体的信息字典
            
        Returns:
            current_tasks: 所有任务状态矩阵，形状为 (tasks_num+1, status_dim)
                          第一行为基地状态，后续行为各任务状态
        """
        status = []
        
        # 遍历环境中的所有任务
        for t in self.task_dic.values():
            # 计算智能体到达该任务的行驶时间
            travel_time = self.calculate_eulidean_distance(agent, t) / agent['velocity']
            
            # 根据use_deadline标志构建不同的状态向量
            if self.use_deadline:
                # 归一化截止时间到 [0, 1] 范围（任务最大截止时间为60）
                normalized_deadline = t['deadline'] / 60.0
                
                # 构建包含截止时间的完整状态向量
                temp_status = np.hstack([
                    t['status'],                              # 任务当前剩余需求
                    t['requirements'],                        # 任务原始需求
                    t['time'],                               # 任务持续时间
                    normalized_deadline,                      # 归一化的任务截止时间
                    travel_time,                             # 到达该任务的行驶时间
                    agent['location'] - t['location'],       # 智能体与任务的相对位置
                    t['feasible_assignment']                 # 任务是否已有可行分配
                ])
            else:
                # 构建不包含截止时间的状态向量（用于AttentionNet等模型）
                temp_status = np.hstack([
                    t['status'],                              # 任务当前剩余需求
                    t['requirements'],                        # 任务原始需求
                    t['time'],                               # 任务持续时间
                    travel_time,                             # 到达该任务的行驶时间
                    agent['location'] - t['location'],       # 智能体与任务的相对位置
                    t['feasible_assignment']                 # 任务是否已有可行分配
                ])
            status.append(temp_status)
        
        # 在任务列表前添加基地状态（索引0）
        if self.use_deadline:
            # 包含截止时间的基地状态
            depot_status = np.hstack([
                np.zeros(self.traits_dim),                   # 基地无能力需求
                -np.ones(self.traits_dim),                   # 负值标识这是基地
                0,                                           # 基地无持续时间
                100.0 / 60.0,                                # 基地归一化截止时间 (100/60 ≈ 1.67)
                self.calculate_eulidean_distance(            # 到达基地的行驶时间
                    agent, self.depot_dic[agent['species']]
                ) / agent['velocity'],
                agent['location'] - agent['depot'],          # 智能体与基地的相对位置
                1                                           # 基地总是可用
            ])
        else:
            # 不包含截止时间的基地状态
            depot_status = np.hstack([
                np.zeros(self.traits_dim),                   # 基地无能力需求
                -np.ones(self.traits_dim),                   # 负值标识这是基地
                0,                                           # 基地无持续时间
                self.calculate_eulidean_distance(            # 到达基地的行驶时间
                    agent, self.depot_dic[agent['species']]
                ) / agent['velocity'],
                agent['location'] - agent['depot'],          # 智能体与基地的相对位置
                1                                           # 基地总是可用
            ])
        
        # 将基地状态放在列表开头，然后是所有任务状态
        status = [depot_status] + status
        
        # 将所有状态堆叠成矩阵
        current_tasks = np.vstack(status)
        return current_tasks

    def get_unfinished_task_mask(self):
        """
        获取未完成任务的掩码
        
        生成一个布尔掩码，标识哪些任务尚未完成。这个掩码主要用于强化学习
        的动作空间限制，防止智能体选择已经完成的任务。
        
        掩码逻辑：
        - True: 任务已完成或不可选择
        - False: 任务未完成且可以选择
        
        Returns:
            mask: 布尔数组，True表示对应任务不应被选择
        """
        # 获取未完成任务的布尔列表，然后取逻辑非
        # get_unfinished_tasks()返回的是"任务是否未完成"的列表
        # 我们需要"任务是否应该被掩码"的列表，所以取逻辑非
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        """
        获取未完成任务的状态列表
        
        检查所有任务的完成状态，返回每个任务是否仍未完成的布尔列表。
        一个任务被认为是"未完成"的条件是：
        1. 尚未获得可行的智能体分配 (feasible_assignment == False)
        2. 仍有未满足的需求 (status中有大于0的元素)
        
        这个方法用于确定哪些任务仍然需要智能体的关注和分配。
        
        Returns:
            unfinished_tasks: 布尔列表，True表示对应任务尚未完成
        """
        unfinished_tasks = []
        # 遍历所有任务，检查每个任务的完成状态
        for task in self.task_dic.values():
            # 任务未完成的条件：
            # 1. 没有可行的分配方案 (feasible_assignment == False)
            # 2. 还有未满足的需求 (status中有元素 > 0)
            status = task.get('status')
            # 确保 status 是数组
            if status is None:
                is_unfinished = False
            elif np.isscalar(status):
                is_unfinished = (task['feasible_assignment'] is False and status > 0)
            else:
                is_unfinished = (task['feasible_assignment'] is False and 
                               np.any(status > 0))
            unfinished_tasks.append(is_unfinished)
            
        return unfinished_tasks

    def get_arrival_time(self, agent_id, task_id):
        """
        获取智能体到达指定任务的时间
        
        根据智能体的路径规划，计算智能体到达特定任务位置的预计时间。
        这个时间基于智能体的路径记录和对应的到达时间记录。
        
        智能体的路径规划信息存储方式：
        - route: 路径序列，记录智能体访问的任务/基地ID序列
        - arrival_time: 对应的到达时间序列，与route一一对应
        
        Args:
            agent_id: 智能体ID
            task_id: 目标任务ID
            
        Returns:
            arrival_time: 智能体到达该任务的时间（浮点数）
        """
        # 获取智能体的到达时间记录
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        
        # 在智能体的路径中查找目标任务的位置索引
        # np.where返回匹配位置的数组，[-1]取最后一次出现的位置
        # 这处理了智能体可能多次访问同一任务的情况
        route_array = np.array(self.agent_dic[agent_id]['route'])
        matching_indices = np.where(route_array == task_id)[0]
        
        if len(matching_indices) == 0:
            # 如果路径中没有找到该任务，返回一个很大的时间值或当前时间
            # 这表示智能体不会到达该任务
            return float('inf')
        
        arrival_for_task = matching_indices[-1]
        # 返回对应位置的到达时间
        return float(arrival_time[arrival_for_task])

    def get_abilities(self, members):
        """
        计算智能体团队的总能力
        
        给定一组智能体成员列表，计算他们的总能力。这个方法用于评估
        一个智能体团队是否能够满足特定任务的需求。
        
        能力计算方式：
        - AT（加性任务）：将所有成员的能力向量求和
          例如：[1,0,0] + [1,0,0] = [2,0,0]
        - BT（二元任务）：使用覆盖逻辑（逐元素取最大值）
          例如：[1,0,0] + [0,1,0] = [1,1,0]，只要有一个智能体具备某技能即满足
        
        Args:
            members: 智能体ID列表，包含参与任务的所有智能体
            
        Returns:
            total_abilities: 团队总能力向量，长度为traits_dim
        """
        # 如果没有成员，返回零能力向量
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            # 收集所有成员的能力向量
            member_abilities = np.array([self.agent_dic[member]['abilities'] for member in members])
            
            if self.binary_task:
                # BT模式：二元任务，使用覆盖逻辑（逐元素取最大值）
                # 只要联盟中存在具备某技能的智能体，该技能需求就被满足
                return np.max(member_abilities, axis=0)
            else:
                # AT模式：加性任务，能力累加
                # 多个智能体的相同能力可以累积
                return np.sum(member_abilities, axis=0)

    def get_contributable_task_mask(self, agent_id):
        """
        获取智能体可贡献任务的掩码
        
        确定指定智能体可以对哪些任务做出有效贡献。这个方法用于动作空间
        的限制，防止智能体选择它无法有效帮助的任务。
        
        贡献判断逻辑：
        1. 只考虑尚未获得可行分配的任务
        2. 计算智能体能力与任务需求的有效交集
        3. 如果有效交集大于0，则该智能体可以贡献
        
        有效能力计算：effective_ability = min(task_need, agent_ability)
        这确保智能体的贡献不会超过任务的实际需求。
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            contributable_task_mask: 布尔数组，True表示智能体无法贡献该任务
        """
        agent = self.agent_dic[agent_id]
        
        # 初始化掩码：假设所有任务都无法贡献（True表示被掩码）
        contributable_task_mask = np.ones(self.tasks_num, dtype=bool)
        
        # 遍历所有任务，检查智能体是否可以贡献
        for task in self.task_dic.values():
            # 只考虑尚未获得可行分配的任务
            if not task['feasible_assignment']:
                # 计算智能体对该任务的有效贡献能力
                # effective_ability[i] = min(task_need[i], agent_ability[i])
                # 这确保贡献不超过任务需求，避免资源浪费
                ability = np.maximum(
                    np.minimum(task['status'], agent['abilities']), 
                    0.  # 确保结果非负
                )
                
                # 如果智能体在任何维度上都有有效贡献，则该任务不被掩码
                if ability.sum() > 0:
                    contributable_task_mask[task['ID']] = False
                    
        return contributable_task_mask

    def get_waiting_tasks(self):
        """
        获取正在等待的任务和智能体信息
        
        识别当前环境中哪些任务正在等待更多智能体加入，以及哪些智能体
        正在等待其他智能体到达以开始任务执行。
        
        等待任务的判断条件：
        1. 任务尚未获得可行的分配方案 (feasible_assignment == False)
        2. 任务已经有智能体分配给它 (len(members) > 0)
        
        这种情况通常发生在：
        - 任务需要多个智能体协作
        - 部分智能体已到达，但还需要等待其他智能体
        - 智能体总能力仍不足以满足任务需求
        
        Returns:
            waiting_tasks: 布尔数组，True表示对应任务不在等待状态
            waiting_agents: 正在等待的智能体ID列表
        """
        # 初始化等待任务掩码：假设所有任务都不在等待状态
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)
        waiting_agents = []
        
        # 遍历所有任务，识别等待状态的任务
        for task in self.task_dic.values():
            # 等待任务的条件：没有可行分配但已有智能体参与
            if not task['feasible_assignment'] and len(task['members']) > 0:
                # 标记该任务为等待状态（False表示正在等待）
                waiting_tasks[task['ID']] = False
                
                # 收集该任务中正在等待的所有智能体
                waiting_agents += task['members']
                
        return waiting_tasks, waiting_agents

    def agent_update(self):
        """
        更新所有智能体的下次决策时间
        
        这个方法根据智能体的当前状态和任务情况，计算每个智能体下次需要
        做出决策的时间。决策时间的确定对于环境的时间推进和智能体调度
        至关重要。
        
        决策时间计算逻辑：
        1. 智能体在基地且所有任务已分配：决策时间设为NaN（仿真结束）
        2. 智能体在基地但仍有未分配任务：决策时间设为无穷大（等待状态）
        3. 智能体在执行任务：
           - 任务有可行分配且智能体是成员：在任务结束时决策
           - 任务有可行分配但智能体不是成员：在到达+最大等待时间后决策
           - 任务无可行分配：在到达+最大等待时间后决策
        """
        # 该条件在一次 agent_update 内不会变化，避免对全部任务做重复扫描。
        all_tasks_assigned = self.all_tasks_feasibly_assigned()

        # 遍历所有智能体，更新其决策时间
        for agent in self.agent_dic.values():
            # 充电中的智能体由外部逻辑控制决策时间，避免被这里覆盖
            if agent['current_task'] == -999:
                continue
            # 情况1：智能体当前在基地（current_task < 0）
            if agent['current_task'] < 0:
                # 检查是否所有任务都已获得可行分配
                if all_tasks_assigned:
                    # 所有任务已分配，智能体无需再做决策
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    # 仍有未分配任务，智能体进入等待状态
                    agent['next_decision'] = np.inf
                else:
                    # 保持当前状态不变
                    pass
            
            # 情况2：智能体正在前往任务或执行任务（current_task >= 0）
            else:
                current_task = self.task_dic[agent['current_task']]
                
                # 检查当前任务是否已获得可行分配
                if current_task['feasible_assignment']:
                    # 任务有可行分配方案
                    if agent['ID'] in current_task['members']:
                        # 智能体是任务成员，在任务完成时做下次决策
                        # 防止决策时间回退
                        agent['next_decision'] = max(float(current_task['time_finish']), self.current_time)
                        
                        # 如果任务已开始，标记智能体为已分配状态
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        # 智能体不是任务成员，在最大等待时间后决策
                        # 防止决策时间回退
                        agent['next_decision'] = max(
                            self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time,
                            self.current_time
                        )
                        agent['assigned'] = False
                else:
                    # 任务尚无可行分配，智能体在最大等待时间后决策
                    # 防止决策时间回退
                    agent['next_decision'] = max(
                        self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time,
                        self.current_time
                    )
                    agent['assigned'] = False

    def task_update(self):
        """
        更新所有任务的状态和分配情况
        
        这是环境动态更新的核心方法，负责：
        1. 检查每个任务的完成条件
        2. 更新任务的可行分配状态
        3. 处理智能体的等待超时情况
        4. 标记已完成的任务
        5. 处理智能体返回基地的状态
        
        Returns:
            f_task: 新获得可行分配的任务ID列表
        """
        f_task = []  # 存储新获得可行分配的任务
        
        # 检查每个任务的状态和是否完成
        for task in self.task_dic.values():
            # 处理尚未获得可行分配的任务
            if not task['feasible_assignment']:
                # 计算当前分配智能体的总能力
                abilities = self.get_abilities(task['members'])
                
                # 获取所有成员智能体的到达时间
                arrival = np.array([self.get_arrival_time(member, task['ID']) 
                                  for member in task['members']])
                
                # 更新任务状态：剩余需求 = 原始需求 - 已分配能力
                task['status'] = task['requirements'] - abilities
                
                # 检查任务需求是否已满足
                if (task['status'] <= 0).all():
                    # 能力需求已满足，检查时间约束
                    
                    # 智能体需要等待其他智能体到达
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        # 等待时间在允许范围内，任务获得可行分配
                        task['time_start'] = float(np.max(arrival, keepdims=True))
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                        task['feasible_assignment'] = True
                        f_task.append(task['ID'])
                    else:
                        # 等待时间过长，移除早到的智能体
                        task['feasible_assignment'] = False
                        
                        # 识别到达过早的智能体（需要等待超过最大等待时间）
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        
                        # 移除这些智能体并标记为被放弃
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    # 能力需求未满足，检查智能体等待超时情况
                    task['feasible_assignment'] = False
                    
                    # 移除等待时间过长的智能体
                    for member in np.array(task['members']):
                        # 如果智能体等待时间超过最大限制，移除该智能体
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                # 处理已有可行分配的任务
                # 检查任务是否已完成
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        # 任务扫描结束后再统一判断一次，避免在基地成员循环内重复做全表扫描。
        all_tasks_assigned = self.all_tasks_feasibly_assigned()

        # 检查基地状态：智能体是否可以标记为已返回
        for depot in self.depot_dic.values():
            for member in depot['members']:
                # 智能体返回基地的条件：
                # 1. 已到达基地时间
                # 2. 所有任务都已获得可行分配
                arrival_condition = self.current_time >= self.get_arrival_time(member, depot['ID'])
                
                if arrival_condition and all_tasks_assigned:
                    self.agent_dic[member]['returned'] = True
                    
        return f_task

    def next_decision(self):
        """
        确定下次决策的时间和需要做决策的智能体
        
        这个方法是环境时间推进的核心，它决定了仿真的下一个时间点和
        哪些智能体需要在该时间点做出决策。这种事件驱动的时间推进
        机制确保仿真效率和准确性。
        
        决策逻辑：
        1. 找到所有智能体中最早的决策时间
        2. 识别在该时间点需要决策的智能体
        3. 处理被阻塞的智能体（无法做出选择的智能体）
        4. 返回需要决策的智能体和决策时间
        
        Returns:
            release_agents: 元组 (finished_agents, blocked_agents)
                - finished_agents: 完成当前任务需要新决策的智能体列表
                - blocked_agents: 被释放的阻塞智能体列表
            next_decision: 下次决策的时间点
        """
        # 获取所有智能体的下次决策时间
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'), dtype=float)
        due_epsilon = 1e-9

        # 浮点误差可能让事件时间略早于 current_time。
        # 这些事件已经“到时”，需要在当前时刻释放，而不是被误判为无事件。
        stale_mask = np.isfinite(decision_time) & (decision_time < self.current_time - due_epsilon)
        if np.any(stale_mask):
            decision_time = decision_time.copy()
            decision_time[stale_mask] = self.current_time
        
        # 情况1：所有智能体的决策时间都是NaN（仿真结束）
        if np.all(np.isnan(decision_time)):
            # 更新所有智能体电量到最终时间
            max_arrival = max(map(lambda x: max(x) if x else 0, 
                                self.get_matrix(self.agent_dic, 'arrival_time')))
            # 避免时间回退导致的重复决策
            safe_time = max(max_arrival, self.current_time)
            self.update_all_batteries(safe_time)
            # 返回空的智能体列表和安全时间
            return ([], []), safe_time
        
        # 获取无选择的智能体标记
        no_choice = self.get_matrix(self.agent_dic, 'no_choice')
        
        # 将无选择的智能体的决策时间设为无穷大
        decision_time = np.where(no_choice, np.inf, decision_time)
        
        # 找到最早的有效决策时间
        next_decision = np.nanmin(decision_time)
        
        # 情况2：最早决策时间是无穷大（所有活跃智能体都被阻塞）
        if np.isinf(next_decision):
            # 使用智能体的最后到达时间作为替代
            arrival_time = np.array([agent['arrival_time'][-1]
                                   for agent in self.agent_dic.values()], dtype=float)
            stale_arrivals = np.isfinite(arrival_time) & (arrival_time < self.current_time - due_epsilon)
            if np.any(stale_arrivals):
                arrival_time = arrival_time.copy()
                arrival_time[stale_arrivals] = self.current_time
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
            if np.isinf(next_decision):
                finite_arrivals = arrival_time[np.isfinite(arrival_time)]
                max_arrival = np.max(finite_arrivals) if finite_arrivals.size > 0 else self.current_time
                safe_time = max(float(max_arrival), self.current_time)
                self.update_all_batteries(safe_time)
                return ([], []), safe_time

        # ==================== 确保决策时间不回退 ====================
        if next_decision < self.current_time:
            next_decision = self.current_time
        
        # ==================== 更新所有智能体电量到下次决策时间 ====================
        self.update_all_batteries(next_decision)
        
        # ==================== 更新智能体移动状态 ====================
        for agent in self.agent_dic.values():
            # 如果智能体已到达目的地，标记为非移动状态
            if len(agent['arrival_time']) > 0 and next_decision >= agent['arrival_time'][-1]:
                agent['is_moving'] = False
        
        # 找到在该时间点需要做决策的智能体
        finished_agents = np.where(
            np.isfinite(decision_time) & (decision_time <= next_decision + due_epsilon)
        )[0].tolist()
        
        # 找到可以被释放的阻塞智能体
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            # no_choice 的智能体不应在同一时间点被反复释放
            if no_choice[agent_id]:
                continue
            # 如果阻塞智能体的到达时间不晚于下次决策时间，则释放它
            if next_decision + due_epsilon >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)
        
        # 组合需要决策的智能体
        release_agents = (finished_agents, blocked_agents)
        
        return release_agents, next_decision

    def agent_step(self, agent_id, task_id, decision_step):
        """
        执行智能体的动作步骤
        
        这是环境交互的核心方法，处理智能体选择任务的动作。
        
        Args:
            agent_id: 智能体ID
            task_id: 选择的任务ID（0表示返回基地，其他值表示任务ID+1）
            decision_step: 决策步数
            
        Returns:
            reward: 奖励值
            doable: 动作是否可执行
            f_t: 完成的任务列表
        """
        # 在执行动作前，更新电量到当前时间
        time_elapsed = self.current_time - self.agent_dic[agent_id]['last_update_time']
        if time_elapsed > 0:
            self.update_battery(agent_id, time_elapsed)
        
        # 转换任务ID（UI中的task_id需要减1才是实际的任务索引）
        task_id = task_id - 1
        
        # 检查任务是否已完成分配
        if task_id != -1:
            agent = self.agent_dic[agent_id]
            task = self.task_dic[task_id]
            if task['feasible_assignment']:
                return -1, False, []
        else:
            # 选择返回基地
            agent = self.agent_dic[agent_id]
            task = self.depot_dic[agent['species']]
        
        # 检查电量是否足够到达目的地
        if not self.can_reach_with_battery(agent_id, task):
            # 电量不足，强制前往充电站
            # print(f"警告: 智能体 {agent_id} 电量不足 ({agent['battery']:.2f})，强制前往充电站")
            # 这里返回失败，由调用者处理充电逻辑
            return -1, False, []
            
        # 更新智能体路径和当前任务
        agent['route'].append(task['ID'])
        previous_task = agent['current_task']
        agent['current_task'] = task_id
        
        # 计算行驶时间和距离
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)
        
        # 标记智能体为移动状态
        agent['is_moving'] = True
        
        # 确定出发时间
        if previous_task >= 0:
            # 防止使用过去的完成时间导致时间回退
            prev_finish = float(self.task_dic[previous_task]['time_finish'])
            current_time = max(self.current_time, prev_finish)
        else:
            current_time = self.current_time
            
        # 更新到达时间和位置
        agent['arrival_time'] += [current_time + travel_time]
        agent['location'] = task['location']
        agent['decision_step'] = decision_step
        agent['no_choice'] = False

        # 将智能体添加到任务成员列表
        if agent_id not in task['members']:
            task['members'].append(agent_id)
            
        # 更新任务和智能体状态
        f_t = self.task_update()
        self.agent_update()
        
        return 0, True, f_t

    def agent_observe(self, agent_id, max_waiting=False):
        """
        获取智能体的观察信息
        
        为指定智能体生成当前环境的观察，包括任务信息、其他智能体信息和动作掩码。
        
        Args:
            agent_id: 观察者智能体的ID
            max_waiting: 是否考虑最大等待时间限制
            
        Returns:
            tasks_info: 任务信息矩阵
            agents_info: 智能体信息矩阵  
            mask: 动作掩码，True表示不可选择的动作
        """
        agent = self.agent_dic[agent_id]
        
        # 获取未完成任务的掩码
        mask = self.get_unfinished_task_mask()
        
        # 获取该智能体可贡献的任务掩码
        contributable_mask = self.get_contributable_task_mask(agent_id)
        mask = np.logical_or(mask, contributable_mask)
        
        # 如果启用最大等待时间，考虑等待任务的限制
        if max_waiting:
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            waiting_len = np.sum(waiting_tasks_mask == 0)
            if waiting_len > 5:
                mask = np.logical_or(mask, waiting_tasks_mask)
                
        # 在掩码前添加基地选项（索引0，总是可选择）
        mask = np.insert(mask, 0, False)
        
        # 获取当前环境状态信息
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return tasks_info, agents_info, mask

    def calculate_waiting_time(self):
        """
        计算所有智能体和任务的累计等待时间
        
        等待时间是多智能体任务分配中的重要性能指标，它反映了协作效率
        和资源利用情况。该方法计算每个智能体和任务的总等待时间，
        包括正常等待和因超时被放弃而产生的等待时间。
        
        等待时间计算规则：
        1. 任务等待时间：
           - 已分配任务：最晚到达智能体与其他智能体的时间差之和
           - 未分配任务：当前时间与各智能体到达时间的差值之和
           - 被放弃智能体：每个贡献最大等待时间
        
        2. 智能体等待时间：
           - 正常等待：等待其他智能体到达的时间
           - 超时等待：因等待超时被放弃的时间（最大等待时间）
        """
        # 初始化所有智能体的等待时间为0
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = 0
        
        # 计算每个任务及其相关智能体的等待时间
        for task in self.task_dic.values():
            # 获取该任务所有成员智能体的到达时间
            arrival = np.array([self.get_arrival_time(member, task['ID']) 
                              for member in task['members']])
            
            if len(arrival) != 0:
                if task['feasible_assignment']:
                    # 任务已获得可行分配：计算同步等待时间
                    # 等待时间 = 每个智能体等待最晚到达者的时间之和
                    task['sum_waiting_time'] = (np.sum(np.max(arrival) - arrival) + 
                                               len(task['abandoned_agent']) * self.max_waiting_time)
                else:
                    # 任务尚未获得可行分配：计算当前累计等待时间
                    # 等待时间 = 当前时间与各智能体到达时间的差值之和
                    current_waiting = np.maximum(self.current_time - arrival, 0)
                    task['sum_waiting_time'] = (np.sum(current_waiting) + 
                                               len(task['abandoned_agent']) * self.max_waiting_time)
            else:
                # 任务无成员智能体：只计算被放弃智能体的等待时间
                task['sum_waiting_time'] = len(task['abandoned_agent']) * self.max_waiting_time
            
            # 为每个参与任务的智能体累加等待时间
            for member in task['members']:
                if task['feasible_assignment']:
                    # 可行分配：智能体等待最晚到达者
                    waiting_time = np.max(arrival) - self.get_arrival_time(member, task['ID'])
                    self.agent_dic[member]['sum_waiting_time'] += waiting_time
                else:
                    # 未分配：智能体等待到当前时间
                    waiting_time = self.current_time - self.get_arrival_time(member, task['ID'])
                    if waiting_time > 0:
                        self.agent_dic[member]['sum_waiting_time'] += waiting_time
            
            # 为被放弃的智能体添加最大等待时间惩罚
            for member in task['abandoned_agent']:
                self.agent_dic[member]['sum_waiting_time'] += self.max_waiting_time

    def check_finished(self):
        """
        检查仿真是否已完成
        
        这个方法判断多智能体任务分配仿真是否已经结束。仿真完成的条件是：
        1. 所有智能体都已返回基地
        2. 所有任务都已完成
        3. 没有智能体需要做出新的决策
        
        该方法首先更新任务状态，然后检查是否还有智能体需要做决策。
        如果没有待决策的智能体，则进一步检查所有智能体和任务的完成状态。
        
        Returns:
            finished: 布尔值，True表示仿真已完成
        """
        # 首先更新所有任务的状态
        self.task_update()
        
        # 获取下次需要决策的智能体和时间
        decision_agents, current_time = self.next_decision()
        
        # 检查是否还有智能体需要做决策
        total_decision_agents = len(decision_agents[0]) + len(decision_agents[1])
        
        if total_decision_agents == 0:
            # 没有智能体需要决策，更新当前时间并检查完成条件
            self.current_time = current_time
            
            # 仿真完成的双重条件：
            # 1. 所有智能体都已返回基地
            all_agents_returned = self.all_agents_returned()
            
            # 2. 所有任务都已完成
            all_tasks_finished = self.all_tasks_finished()
            
            # 只有两个条件都满足才认为仿真完成
            finished = all_agents_returned and all_tasks_finished
        else:
            # 仍有智能体需要做决策，仿真未完成
            finished = False
            
        return finished

    def generate_traj(self):
        """
        生成所有智能体的运动轨迹
        
        这个方法为可视化和动画生成每个智能体的详细运动轨迹。轨迹包含
        智能体在整个仿真过程中每个时间步的位置和朝向信息。
        
        轨迹生成逻辑：
        1. 遍历智能体路径中的每个路段（从一个位置到下一个位置）
        2. 计算每个路段的运动参数（角度、距离、时间）
        3. 确定在每个位置的停留时间（等待或工作时间）
        4. 按时间步长插值生成连续的位置序列
        5. 处理特殊情况（基地等待、任务执行、返回基地）
        
        轨迹数据格式：
        每个时间步包含 [x坐标, y坐标, 朝向角度]
        """
        # 遍历所有智能体，为每个智能体生成轨迹
        for agent in self.agent_dic.values():
            time_step = 0  # 当前仿真时间
            angle = 0      # 智能体朝向角度
            
            # 遍历智能体路径中的每个路段
            for i in range(1, len(agent['route'])):
                # 获取当前位置和下一个位置的信息
                # 路径中负数表示基地，非负数表示任务
                if agent['route'][i - 1] >= 0:
                    current_task = self.task_dic[agent['route'][i - 1]]
                else:
                    current_task = self.depot_dic[agent['species']]
                    
                if agent['route'][i] >= 0:
                    next_task = self.task_dic[agent['route'][i]]
                else:
                    next_task = self.depot_dic[agent['species']]
                
                # 计算运动方向角度
                angle = np.arctan2(
                    next_task['location'][1] - current_task['location'][1],
                    next_task['location'][0] - current_task['location'][0]
                )
                
                # 计算距离和行驶时间
                distance = self.calculate_eulidean_distance(next_task, current_task)
                total_time = distance / agent['velocity']
                
                # 获取到达时间
                arrival_time_next = agent['arrival_time'][i]
                arrival_time_current = agent['arrival_time'][i - 1]
                
                # 确定在下一个位置的决策时间（何时离开该位置）
                if next_task['ID'] >= 0 and agent['ID'] in next_task['members'] and next_task['feasible_assignment']:
                    # 到达任务且参与执行：等待时间在限制内则执行到完成，否则等待超时离开
                    if next_task['time_start'] - arrival_time_next <= self.max_waiting_time:
                        next_decision = next_task['time_finish']
                    else:
                        next_decision = arrival_time_next + self.max_waiting_time
                elif next_task['ID'] < 0 and i != len(agent['route']) - 1:
                    # 到达基地但不是最终目的地：在基地等待一段时间
                    next_decision = arrival_time_next + self.depot_waiting_time
                else:
                    # 其他情况：等待最大等待时间后离开
                    next_decision = arrival_time_next + self.max_waiting_time
                
                # 确定在当前位置的离开时间
                if current_task['ID'] < 0 and i == 1:
                    # 从基地开始：立即出发
                    current_decision = 0
                elif current_task['ID'] < 0:
                    # 在基地：等待基地等待时间后出发
                    current_decision = arrival_time_current + self.depot_waiting_time
                else:
                    # 在任务位置：根据任务状态确定离开时间
                    task_member = agent['ID'] in current_task['members']
                    valid_wait = current_task['time_start'] - arrival_time_current <= self.max_waiting_time
                    feasible = current_task['feasible_assignment']
                    
                    if task_member and valid_wait and feasible:
                        # 参与任务执行：任务完成后离开
                        current_decision = current_task['time_finish']
                    else:
                        # 不参与或等待超时：等待最大时间后离开
                        current_decision = arrival_time_current + self.max_waiting_time
                
                # 生成该路段的轨迹点
                while time_step < next_decision:
                    time_step += self.dt
                    
                    if time_step < arrival_time_next:
                        # 智能体正在移动中：插值计算当前位置
                        fraction_of_time = (time_step - current_decision) / total_time
                        
                        if fraction_of_time <= 1:
                            # 在路径上：线性插值位置
                            x = (current_task['location'][0] + 
                                fraction_of_time * (next_task['location'][0] - current_task['location'][0]))
                            y = (current_task['location'][1] + 
                                fraction_of_time * (next_task['location'][1] - current_task['location'][1]))
                            agent['trajectory'].append(np.hstack([x, y, angle]))
                        else:
                            # 已到达目的地：保持在目的地位置
                            agent['trajectory'].append(
                                np.hstack([next_task['location'][0], next_task['location'][1], angle])
                            )
                    else:
                        # 智能体已到达并停留在目的地
                        agent['trajectory'].append(
                            np.array([next_task['location'][0], next_task['location'][1], angle])
                        )
            
            # 填充剩余时间的轨迹（智能体返回基地后的状态）
            while time_step < self.current_time:
                time_step += self.dt
                # 智能体保持在基地位置
                depot_location = self.depot_dic[agent['species']]['location']
                agent['trajectory'].append(
                    np.array([depot_location[0], depot_location[1], angle])
                )

    # def get_episode_reward(self, max_time=100, episode_number=0):
    #     # !origin reward
    #     """
    #     计算回合奖励
        
    #     评估当前回合的整体表现，考虑完成时间、效率等因素。
        
    #     Args:
    #         max_time: 最大允许时间
            
    #     Returns:
    #         reward: 回合奖励（负值，越大越好）
    #         finished_tasks: 各任务的完成状态列表
    #     """
    #     # 计算等待时间
    #     self.calculate_waiting_time()
        
    #     # 获取效率和完成状态
    #     eff = self.get_efficiency()
    #     finished_tasks = self.get_matrix(self.task_dic, 'finished')
    #     completion = np.mean(finished_tasks) if len(finished_tasks) else 0.0

    #     if self.finished:
    #         reward = - self.current_time - eff * 10 + 50.0 * completion
    #     else:
    #         reward = - max_time - eff * 10 + 50.0 * completion
        
    #     # 计算奖励：如果完成则基于实际时间，否则使用最大时间惩罚
    #     # reward = - self.current_time - eff * 10 if self.finished else - max_time - eff * 10
    #     return reward, finished_tasks



    def _route_positions(self, agent, idx):
        if idx >= 0:
            return np.array(self.task_dic[idx]['location'])
        else:
            return np.array(self.depot_dic[agent['species']]['location'])

    def agent_invalid_moves(self, agent):
        """计算单个智能体的无效移动次数（相同位置间的移动）"""
        cnt = 0
        route = agent['route']
        for i in range(1, len(route)):
            cur = self._route_positions(agent, route[i-1] if route[i-1] >= 0 else -1)
            nxt = self._route_positions(agent, route[i] if route[i] >= 0 else -1)
            if np.linalg.norm(cur - nxt) < 1e-3:
                cnt += 1
        return cnt

    def agent_total_distance(self, agent):
        """计算单个智能体的总移动距离"""
        dist = 0.0
        route = agent['route']
        for i in range(1, len(route)):
            prev_pos = self._route_positions(agent, route[i-1] if route[i-1] >= 0 else -1)
            curr_pos = self._route_positions(agent, route[i] if route[i] >= 0 else -1)
            dist += np.linalg.norm(curr_pos - prev_pos)
        return dist

    def agent_proximity_ratio(self, agent):
        """计算单个智能体的就近选择比例"""
        route = agent['route']
        score, decisions = 0, 0
        for i in range(1, len(route)):
            if route[i] >= 0:
                current_pos = self._route_positions(agent, route[i-1] if route[i-1] >= 0 else -1)
                selected_pos = self._route_positions(agent, route[i])
                actual_dist = np.linalg.norm(selected_pos - current_pos)
                future_tasks = [tid for tid in route[i:] if tid >= 0]
                if len(future_tasks) > 1:
                    dists = [np.linalg.norm(np.array(self.task_dic[tid]['location']) - current_pos)
                            for tid in future_tasks]
                    if actual_dist <= min(dists) * 1.1:
                        score += 1
                    decisions += 1
        return (score / decisions) if decisions > 0 else 0.0

    def agent_finished_tasks(self, agent):
        """计算单个智能体完成的任务数量（去重）"""
        tasks = set([tid for tid in agent['route'] if tid >= 0])
        return sum(int(self.task_dic[tid]['finished']) for tid in tasks)

    def get_reward_config(self):
        """
        获取当前环境的奖励配置。

        对旧 pickle 环境做兼容：如果缺少 reward_config，则回退到默认配置。
        """
        self.reward_config = normalize_reward_config(getattr(self, 'reward_config', None))
        return copy.deepcopy(self.reward_config)

    def set_reward_config(self, reward_config):
        """更新当前环境的奖励配置。"""
        self.reward_config = normalize_reward_config(reward_config)

    def get_episode_reward_breakdown(self, max_time=100, episode_number=0, include_credit_details=False):
        """
        计算终端奖励分解，并在需要时返回信用分配诊断信息。

        Args:
            max_time: 回合时间上限
            episode_number: 训练轮次（当前实现中仅保留接口兼容）
            include_credit_details: 是否返回逐任务/逐智能体信用细节

        Returns:
            breakdown: dict
            finished_tasks: ndarray/list
        """
        del episode_number  # 保留签名兼容，当前奖励不依赖 episode 调权

        eps = 1e-6
        reward_config = self.get_reward_config()
        zero_marginal_eps = float(reward_config.get('zero_marginal_epsilon', 1e-8))
        positive_credit_eps = float(reward_config.get('positive_credit_epsilon', 1e-8))
        T_end = float(getattr(self, "current_time", 0.0))
        T_end = min(T_end, float(max_time))

        self.calculate_waiting_time()
        eff = float(self.get_efficiency())
        finished_tasks = self.get_matrix(self.task_dic, "finished")

        scoped_tasks = []
        arrived_tasks = []
        for tid, task in self.task_dic.items():
            if isinstance(tid, (int, np.integer)) and tid < 0:
                continue
            appear_time = float(task.get("appear_time", 0.0))
            if appear_time <= T_end + 1e-9:
                arrived_tasks.append(task)
            if reward_config.get('use_revealed_task_set', True):
                if appear_time <= T_end + 1e-9:
                    scoped_tasks.append(task)
            else:
                scoped_tasks.append(task)

        N = len(scoped_tasks)
        if N == 0:
            empty_breakdown = {
                'reward_config': reward_config,
                'task_scope': 'revealed' if reward_config.get('use_revealed_task_set', True) else 'all_tasks',
                'n_tasks_in_scope': 0,
                'n_arrived_tasks': len(arrived_tasks),
                'finished_tasks_in_scope': 0,
                'unfinished_tasks_in_scope': 0,
                'global_contribution': 0.0,
                'progress_share': 0.0,
                'shared_term': 0.0,
                'local_contribution': 0.0,
                'local_term': 0.0,
                'team_reward': 0.0,
                'per_agent_service_credit': {},
                'per_agent_cost_term': {},
                'per_agent_local_net': {},
                'credit_analysis': {
                    'zero_marginal_cases': 0,
                    'zero_marginal_positive_cases': 0,
                    'zero_marginal_positive_rate': 0.0,
                    'mean_zero_marginal_credit': 0.0,
                    'max_zero_marginal_credit': 0.0,
                    'mean_nonzero_marginal_credit': 0.0,
                },
            }
            if include_credit_details:
                empty_breakdown['task_credit_details'] = []
                empty_breakdown['per_agent_credit_stats'] = {}
            return empty_breakdown, finished_tasks

        F = sum(1 for t in scoped_tasks if bool(t.get("finished", False)))
        U = N - F
        completion_rate = F / max(N, 1)
        backlog_ratio = U / max(N, 1)
        n_agents = max(len(self.agent_dic), 1)

        slack_sum = 0.0
        violation_sum = 0.0
        ontime_cnt = 0
        overdue_sum = 0.0

        for task in scoped_tasks:
            deadline = task.get("deadline", None)
            if deadline is None:
                continue
            deadline = float(deadline)
            if bool(task.get("finished", False)):
                tf = float(task.get("time_finish", 0.0))
                diff = tf - deadline
                if diff <= 0:
                    ontime_cnt += 1
                    slack_sum += (-diff)
                else:
                    violation_sum += diff
            elif T_end > deadline:
                overdue_sum += (T_end - deadline)

        ontime_rate = ontime_cnt / max(F, 1)
        slack_norm = slack_sum / (max_time * max(F, 1) + eps)
        violation_norm = violation_sum / (max_time * max(F, 1) + eps)
        overdue_norm = overdue_sum / (max_time * max(N, 1) + eps)

        agent_wait = np.array(self.get_matrix(self.agent_dic, "sum_waiting_time"), dtype=float)
        task_wait = np.array(self.get_matrix(self.task_dic, "sum_waiting_time"), dtype=float)
        agent_wait = np.nan_to_num(agent_wait, nan=0.0, posinf=0.0, neginf=0.0)
        task_wait = np.nan_to_num(task_wait, nan=0.0, posinf=0.0, neginf=0.0)
        avg_waiting_time = (float(agent_wait.sum()) + float(task_wait.sum())) / 2.0

        def _norm(value, cap):
            cap = max(float(cap), eps)
            value = max(float(value), 0.0)
            return min(value / cap, 1.0)

        waiting_cap = getattr(self, "waiting_norm_cap", max_time)
        waiting_norm = _norm(avg_waiting_time, waiting_cap)

        def _task_requirement_mass(task):
            requirements = np.asarray(task.get("requirements", []), dtype=float)
            return max(float(np.maximum(requirements, 0.0).sum()), eps)

        def _task_coverage_ratio(task, members=None):
            coalition = task.get("members", []) if members is None else members
            coalition = [aid for aid in coalition if aid in self.agent_dic]
            if not coalition:
                return 0.0
            covered_ability = np.asarray(self.get_abilities(coalition), dtype=float)
            requirements = np.asarray(task.get("requirements", []), dtype=float)
            covered_mass = np.minimum(
                np.maximum(covered_ability, 0.0),
                np.maximum(requirements, 0.0),
            ).sum()
            return float(np.clip(covered_mass / _task_requirement_mass(task), 0.0, 1.0))

        total_distance = 0.0
        if hasattr(self, "agent_total_distance"):
            for agent in self.agent_dic.values():
                total_distance += float(self.agent_total_distance(agent))
        else:
            total_distance = float(np.sum(self.get_matrix(self.agent_dic, "travel_dist")))

        optimal_distance = 0.0
        if len(self.depot_dic) > 0:
            for task in scoped_tasks:
                pos = np.array(task["location"], dtype=float)
                min_depot = min(
                    np.linalg.norm(pos - np.array(depot["location"], dtype=float))
                    for depot in self.depot_dic.values()
                )
                optimal_distance += 2.0 * float(min_depot)

        distance_cap = getattr(self, "distance_norm_cap", max(2.0 * optimal_distance, 1.0))
        distance_norm = _norm(total_distance, distance_cap)
        compactness = min(optimal_distance / max(total_distance, eps), 1.0) if total_distance > eps else 0.0

        total_charging_times = 0.0
        for agent in self.agent_dic.values():
            total_charging_times += float(agent.get("total_charging_times", 0.0))
        energy_norm = total_charging_times / (max_time * n_agents + eps)

        throughput_norm = F / (max_time + eps)
        throughput_speed = F / (max(T_end, 1.0) + eps)
        eff_norm = min(max(eff / 10.0, 0.0), 1.0)

        w_thr = 120.0
        w_speed = 25.0
        w_comp = 50.0
        w_ontime = 40.0
        w_slack = 15.0
        w_compact = 10.0
        w_early = 20.0
        w_backlog = 60.0
        w_overdue = 120.0
        w_violation = 90.0
        w_eff = 35.0
        w_wait = 10.0
        w_dist = 10.0
        w_energy = 15.0
        w_unfinished_hard = 40.0
        local_weight = float(reward_config.get('local_weight', 12.0))
        phi_beta = float(reward_config.get('phi_beta', 0.5))

        global_contribution = 0.0
        global_contribution += w_thr * throughput_norm + w_speed * throughput_speed
        global_contribution += w_comp * completion_rate
        global_contribution += w_ontime * ontime_rate + w_slack * slack_norm
        global_contribution += w_compact * compactness
        global_contribution -= w_backlog * backlog_ratio
        global_contribution -= w_overdue * overdue_norm
        global_contribution -= w_violation * violation_norm
        global_contribution -= w_eff * eff_norm
        global_contribution -= w_wait * waiting_norm
        global_contribution -= w_dist * distance_norm
        global_contribution -= w_energy * energy_norm

        if U == 0 and bool(getattr(self, "finished", False)) and T_end < max_time - 1e-6:
            global_contribution += w_early * (1.0 - T_end / (max_time + eps))

        if U > 0:
            global_contribution -= w_unfinished_hard * min(U, 5) / 5.0

        progress_share = phi_beta * float(
            1.0 * completion_rate
            - 0.8 * backlog_ratio
            - 1.0 * overdue_norm
            - 0.3 * waiting_norm
            - 0.2 * distance_norm
            - 0.2 * energy_norm
        )

        def _finished_task_credit_value(task):
            deadline = task.get("deadline", None)
            if deadline is None:
                return 1.0
            deadline = float(deadline)
            finish_time = float(task.get("time_finish", 0.0))
            diff = finish_time - deadline
            if diff <= 0:
                slack_ratio = min((-diff) / max(float(max_time), 1.0), 1.0)
                return 1.0 + 0.25 * slack_ratio
            violation_ratio = min(diff / max(float(max_time), 1.0), 1.0)
            return max(1.0 - violation_ratio, 0.0)

        def _agent_cost_ratio(agent):
            route_decisions = max(len(agent.get("route", [])) - 1, 1)
            invalid_norm = float(self.agent_invalid_moves(agent)) / route_decisions
            dist_norm = float(self.agent_total_distance(agent)) / max(agent.get("velocity", 0.2) * max_time, eps)
            wait_norm_i = _norm(agent.get("sum_waiting_time", 0.0), max_time)
            charge_norm = _norm(agent.get("total_charging_times", 0.0), 1.0)
            return float(np.clip(np.mean([
                invalid_norm,
                min(dist_norm, 1.0),
                wait_norm_i,
                charge_norm,
            ]), 0.0, 1.0))

        service_credit = {aid: 0.0 for aid in self.agent_dic.keys()}
        agent_cost_term = {}
        per_agent_credit_stats = {
            aid: {
                'zero_marginal_cases': 0,
                'zero_marginal_positive_cases': 0,
                'zero_marginal_credit_sum': 0.0,
                'max_zero_marginal_credit': 0.0,
                'nonzero_marginal_cases': 0,
                'nonzero_marginal_credit_sum': 0.0,
            }
            for aid in self.agent_dic.keys()
        }
        task_credit_details = [] if include_credit_details else None

        for task in scoped_tasks:
            if not bool(task.get("finished", False)):
                continue
            members = [aid for aid in task.get("members", []) if aid in self.agent_dic]
            if not members:
                continue

            task_utility = _finished_task_credit_value(task)
            if task_utility <= eps:
                continue

            coalition_coverage = _task_coverage_ratio(task, members=members)
            marginal_credit = {}
            for aid in members:
                reduced_members = [mid for mid in members if mid != aid]
                marginal_drop = coalition_coverage - _task_coverage_ratio(task, members=reduced_members)
                marginal_credit[aid] = max(float(marginal_drop), 0.0)

            marginal_sum = float(sum(marginal_credit.values()))
            assigned_credit = {}
            if marginal_sum <= eps:
                equal_share = task_utility / len(members)
                for aid in members:
                    assigned_credit[aid] = float(equal_share)
                    service_credit[aid] += float(equal_share)
            else:
                for aid in members:
                    assigned_credit[aid] = float(task_utility * (marginal_credit[aid] / marginal_sum))
                    service_credit[aid] += assigned_credit[aid]

            for aid in members:
                if marginal_credit[aid] <= zero_marginal_eps:
                    per_agent_credit_stats[aid]['zero_marginal_cases'] += 1
                    per_agent_credit_stats[aid]['zero_marginal_credit_sum'] += assigned_credit[aid]
                    per_agent_credit_stats[aid]['max_zero_marginal_credit'] = max(
                        per_agent_credit_stats[aid]['max_zero_marginal_credit'],
                        assigned_credit[aid],
                    )
                    if assigned_credit[aid] > positive_credit_eps:
                        per_agent_credit_stats[aid]['zero_marginal_positive_cases'] += 1
                else:
                    per_agent_credit_stats[aid]['nonzero_marginal_cases'] += 1
                    per_agent_credit_stats[aid]['nonzero_marginal_credit_sum'] += assigned_credit[aid]

            if include_credit_details:
                task_credit_details.append({
                    'task_id': int(task.get('ID', -1)),
                    'task_utility': float(task_utility),
                    'coalition_coverage': float(coalition_coverage),
                    'members': [int(aid) for aid in members],
                    'marginal_credit': {int(aid): float(marginal_credit[aid]) for aid in members},
                    'assigned_credit': {int(aid): float(assigned_credit[aid]) for aid in members},
                    'degenerate_equal_share': bool(marginal_sum <= eps),
                })

        local_contribution = 0.0
        per_agent_local_net = {}
        for aid, agent in self.agent_dic.items():
            service_term = service_credit[aid] / max(F, 1)
            cost_term = _agent_cost_ratio(agent) / n_agents
            agent_cost_term[aid] = float(cost_term)
            per_agent_local_net[aid] = float(service_term - cost_term)
            local_contribution += per_agent_local_net[aid]

        shared_term = 0.0
        if reward_config.get('enable_shared_term', True):
            shared_term = float(global_contribution + progress_share)

        local_term = 0.0
        if reward_config.get('enable_local_contribution', True):
            local_term = float(local_weight * local_contribution)

        team_reward = float(shared_term + local_term)

        zero_cases = sum(v['zero_marginal_cases'] for v in per_agent_credit_stats.values())
        zero_positive_cases = sum(v['zero_marginal_positive_cases'] for v in per_agent_credit_stats.values())
        zero_credit_sum = sum(v['zero_marginal_credit_sum'] for v in per_agent_credit_stats.values())
        nonzero_cases = sum(v['nonzero_marginal_cases'] for v in per_agent_credit_stats.values())
        nonzero_credit_sum = sum(v['nonzero_marginal_credit_sum'] for v in per_agent_credit_stats.values())
        max_zero_credit = max(
            (v['max_zero_marginal_credit'] for v in per_agent_credit_stats.values()),
            default=0.0,
        )

        breakdown = {
            'reward_config': reward_config,
            'task_scope': 'revealed' if reward_config.get('use_revealed_task_set', True) else 'all_tasks',
            'n_tasks_in_scope': int(N),
            'n_arrived_tasks': int(len(arrived_tasks)),
            'finished_tasks_in_scope': int(F),
            'unfinished_tasks_in_scope': int(U),
            'global_contribution': float(global_contribution),
            'progress_share': float(progress_share),
            'shared_term': float(shared_term),
            'local_contribution': float(local_contribution),
            'local_term': float(local_term),
            'team_reward': float(team_reward),
            'per_agent_service_credit': {int(aid): float(value) for aid, value in service_credit.items()},
            'per_agent_cost_term': {int(aid): float(value) for aid, value in agent_cost_term.items()},
            'per_agent_local_net': {int(aid): float(value) for aid, value in per_agent_local_net.items()},
            'credit_analysis': {
                'zero_marginal_cases': int(zero_cases),
                'zero_marginal_positive_cases': int(zero_positive_cases),
                'zero_marginal_positive_rate': float(zero_positive_cases / max(zero_cases, 1)),
                'mean_zero_marginal_credit': float(zero_credit_sum / max(zero_cases, 1)),
                'max_zero_marginal_credit': float(max_zero_credit),
                'mean_nonzero_marginal_credit': float(nonzero_credit_sum / max(nonzero_cases, 1)),
            },
        }
        if include_credit_details:
            breakdown['task_credit_details'] = task_credit_details
            breakdown['per_agent_credit_stats'] = {
                int(aid): {
                    key: (int(value) if 'cases' in key else float(value))
                    for key, value in stats.items()
                }
                for aid, stats in per_agent_credit_stats.items()
            }
        return breakdown, finished_tasks

    def get_episode_reward(self, max_time=100, episode_number=0):
        """
        混合版终端奖励：
        1. 保留旧版已验证有效的“强全局项”，直接优化完成率、时效性、等待、距离与能耗。
        2. 去掉旧版随 episode_number 变化的动态调权，避免训练目标非平稳。
        3. 用“已完成任务上的边际贡献”替换旧版启发式个体项，改善 credit assignment。
        4. 对未完成任务增加额外硬惩罚，强化“最后几个任务必须真正收尾”。

        返回:
            team_reward: float
            finished_tasks: ndarray/list
        """
        breakdown, finished_tasks = self.get_episode_reward_breakdown(
            max_time=max_time,
            episode_number=episode_number,
            include_credit_details=False,
        )
        return float(breakdown['team_reward']), finished_tasks


    def get_efficiency(self):
        """
        计算任务执行效率
        
        评估任务分配和执行的整体效率。
        
        Returns:
            efficiency: 平均效率值
        """
        for task in self.task_dic.values():
            if task['feasible_assignment']:
                # 已分配任务：计算需求满足程度
                task['efficiency'] = abs(np.sum(task['requirements'] - task['status'])) / task['requirements'].sum()
            else:
                # 未分配任务：给予较大惩罚
                task['efficiency'] = 10
                
        # 返回所有任务的平均效率
        efficiency = np.mean(self.get_matrix(self.task_dic, 'efficiency'))
        return efficiency

    def stack_trajectory(self):
        """
        将智能体轨迹列表转换为NumPy数组
        
        这个方法将每个智能体的轨迹从Python列表格式转换为NumPy数组格式，
        便于后续的数值计算和可视化处理。
        
        轨迹数据格式转换：
        - 转换前：agent['trajectory'] 是包含多个[x, y, angle]数组的列表
        - 转换后：agent['trajectory'] 是形状为(time_steps, 3)的NumPy数组
        
        这种转换提高了数据访问效率，特别是在绘制动画时需要频繁访问轨迹数据。
        """
        # 遍历所有智能体，将其轨迹列表转换为NumPy数组
        for agent in self.agent_dic.values():
            # 使用vstack将轨迹点列表垂直堆叠成二维数组
            # 结果形状：(时间步数, 3) - 每行包含[x坐标, y坐标, 朝向角度]
            agent['trajectory'] = np.vstack(agent['trajectory'])

    def plot_animation(self, path, n):
        """
        生成多智能体任务分配的动画可视化
        
        这个方法创建一个动态的GIF动画，展示智能体在执行任务过程中的运动轨迹、
        任务完成状态和实时统计信息。动画包含丰富的视觉元素来帮助理解
        多智能体系统的行为和性能。
        
        可视化元素：
        1. 智能体轨迹：不同种类用不同颜色表示
        2. 任务状态：多边形表示任务，颜色表示完成状态
        3. 基地位置：圆形表示各种类智能体的基地
        4. 实时信息：完成率、当前时间等统计数据
        5. 智能体聚集：在同一位置的智能体数量显示
        
        Args:
            path: 动画文件保存路径
            n: 动画文件编号或标识符
        """
        # 生成所有智能体的运动轨迹
        self.generate_traj()
        
        # 配置是否使用机器人图标（当前设为False，使用几何图形）
        plot_robot_icon = False
        if plot_robot_icon:
            # 加载无人机图标（如果启用图标模式）
            drone = plt.imread('env/drone.png')
            drone_oi = OffsetImage(drone, zoom=0.05)

        def get_cmap(n, name='Dark2'):
            """
            获取颜色映射函数
            
            为不同智能体种类生成不同的颜色，确保视觉区分度。
            
            Args:
                n: 需要的颜色数量
                name: matplotlib颜色图谱名称
                
            Returns:
                颜色映射函数
            """
            return plt.cm.get_cmap(name, n)

        # 生成智能体种类的颜色映射
        cmap = get_cmap(self.species_num)
        
        # 设置绘图画布和基本属性
        self.stack_trajectory()  # 转换轨迹数据格式
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        finished_rate = np.sum(finished_tasks) / len(finished_tasks)
        gif_len = int(self.current_time / self.dt)  # 动画帧数
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(dpi=100)
        ax.set_xlim(-0.5, 10.5)  # 设置X轴范围（坐标放大10倍显示）
        ax.set_ylim(-0.5, 10.5)  # 设置Y轴范围
        ax.set_xticks([])        # 隐藏X轴刻度
        ax.set_yticks([])        # 隐藏Y轴刻度
        ax.set_aspect('equal')   # 保持纵横比
        plt.subplots_adjust(left=0, right=0.85, top=0.87, bottom=0.02)  # 调整布局
        
        # 创建智能体轨迹线条（初始为空）
        lines = [ax.plot([], [], color=cmap(a['species']), zorder=0)[0] 
                for a in self.agent_dic.values()]
        
        # 设置动画标题
        ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                     f'\nCurrent time is {0:.2f}min')
        
        # 创建图例
        color_map = []
        for i in range(self.species_num):
            color_map.append(patches.Patch(color=cmap(i), label='Agent species ' + str(i)))
        color_map.append(patches.Patch(color='g', label='Finished task'))    # 绿色：已完成任务
        color_map.append(patches.Patch(color='b', label='Unfinished task'))  # 蓝色：未完成任务
        color_map.append(patches.Patch(color='r', label='Depot'))            # 红色：基地
        color_map.append(patches.Patch(color='gold', label='Charging Station'))  # 金色：充电站
        
        # 设置图例位置
        ax.legend(handles=color_map, bbox_to_anchor=(0.99, 0.7))
        
        # 绘制任务：使用多边形表示，边数=需求总和+3
        task_squares = [
            ax.add_patch(patches.RegularPolygon(
                xy=(task['location'][0] * 10, task['location'][1] * 10),
                numVertices=int(task['requirements'].sum()) + 3,  # 多边形边数反映任务复杂度
                radius=0.3, 
                color='b'  # 初始为蓝色（未完成）
            )) 
            for task in self.task_dic.values()
        ]
        
        # 绘制基地：使用红色圆形表示
        depot_tri = [
            ax.add_patch(patches.Circle(
                (depot['location'][0] * 10, depot['location'][1] * 10),
                0.2, 
                color='r'
            )) 
            for depot in self.depot_dic.values()
        ]
        
        # ==================== 绘制充电站：使用黄色五角星表示 ====================
        charging_stations = [
            ax.add_patch(patches.RegularPolygon(
                xy=(station['location'][0] * 10, station['location'][1] * 10),
                numVertices=5,  # 五角星
                radius=0.25,
                color='gold',
                orientation=np.pi/2  # 旋转使一个角朝上
            ))
            for station in self.charging_station_dic.values()
        ]
        
        # 创建智能体标签（显示聚集数量）
        agent_group = [
            ax.text(agent['location'][0] * 10, agent['location'][1] * 10, str(agent['ID']),
                   horizontalalignment='center', verticalalignment='center', fontsize=8) 
            for agent in self.agent_dic.values()
        ]
        
        # 创建智能体图形表示
        if plot_robot_icon:
            # 使用无人机图标（如果启用）
            agent_triangles = []
            for a in self.agent_dic.values():
                agent_triangles.append(
                    ax.add_artist(AnnotationBbox(
                        drone_oi, 
                        (self.depot_dic[a['species']]['location'][0] * 10,
                         self.depot_dic[a['species']]['location'][1] * 10),
                        frameon=False
                    ))
                )
        else:
            # 使用几何图形（三角形）表示智能体
            agent_triangles = [
                ax.add_patch(patches.RegularPolygon(
                    xy=(self.depot_dic[a['species']]['location'][0] * 10,
                        self.depot_dic[a['species']]['location'][1] * 10), 
                    numVertices=3,      # 三角形
                    radius=0.2, 
                    color=cmap(a['species'])  # 种类对应颜色
                ))
                for a in self.agent_dic.values()
            ]

        # 定义动画更新函数
        def update(frame):
            """
            动画帧更新函数
            
            在每一帧中更新智能体位置、轨迹、任务状态等视觉元素。
            
            Args:
                frame: 当前帧编号
                
            Returns:
                lines: 更新后的轨迹线条列表
            """
            # 更新标题显示当前时间
            ax.set_title(f'Agents finish {finished_rate * 100}% tasks within {self.current_time:.2f}min.'
                         f'\nCurrent time is {frame * self.dt:.2f}min')
            
            # 获取当前帧所有智能体的位置
            pos = np.round([agent['trajectory'][frame, 0:2] for agent in self.agent_dic.values()], 4)
            unq, count = np.unique(pos, axis=0, return_counts=True)  # 统计位置重复数量
            
            # 更新每个智能体的显示
            for agent in self.agent_dic.values():
                # 计算该位置的智能体数量（用于显示聚集情况）
                agent_pos = np.round(agent['trajectory'][frame, 0:2], 4)
                repeats = int(count[np.argwhere(np.all(unq == agent_pos, axis=1))])
                
                # 更新智能体位置
                agent_triangles[agent['ID']].xy = tuple(agent['trajectory'][frame, 0:2] * 10)
                agent_group[agent['ID']].set_position(tuple(agent['trajectory'][frame, 0:2] * 10))
                agent_group[agent['ID']].set_text(str(repeats))  # 显示聚集数量
                
                # 如果使用图标模式，更新图标位置
                if plot_robot_icon:
                    agent_triangles[agent['ID']].xyann = tuple(agent['trajectory'][frame, 0:2] * 10)
                    agent_triangles[agent['ID']].xybox = tuple(agent['trajectory'][frame, 0:2] * 10)
                
                # 更新智能体朝向
                agent_triangles[agent['ID']].orientation = agent['trajectory'][frame, 2] - np.pi / 2
                
                # 绘制智能体轨迹（显示最近40帧的轨迹）
                if frame > 40:
                    lines[agent['ID']].set_data(
                        agent['trajectory'][frame - 40:frame + 1, 0] * 10,
                        agent['trajectory'][frame - 40:frame + 1, 1] * 10
                    )
                else:
                    lines[agent['ID']].set_data(
                        agent['trajectory'][:frame + 1, 0] * 10,
                        agent['trajectory'][:frame + 1, 1] * 10
                    )

            # 更新任务状态显示
            for task in self.task_dic.values():
                # 如果启用反应式规划，动态显示任务
                if self.reactive_planning:
                    task_visibility_time = np.clip(frame * self.dt // 10 * 20 + 20, 20, 100)
                    if task['ID'] > task_visibility_time:
                        task_squares[task['ID']].set_color('w')  # 白色：尚未出现的任务
                        task_squares[task['ID']].set_zorder(0)
                    else:
                        task_squares[task['ID']].set_color('b')  # 蓝色：可见的未完成任务
                        task_squares[task['ID']].set_zorder(1)
                
                # 如果任务已完成，改为绿色
                if frame * self.dt >= task['time_finish'] > 0:
                    task_squares[task['ID']].set_color('g')
                    
            return lines

        # 创建并保存动画
        ani = FuncAnimation(fig, update, frames=gif_len, interval=100, blit=True)
        ani.save(f'{path}/episode_{n}_{self.current_time:.1f}.gif')

    def execute_by_route(self, path='./', method=0, plot_figure=False):
        """
        按预设路径执行智能体任务分配
        
        这个方法用于执行预先规划好的智能体路径，常用于测试预定义的
        任务分配策略或验证路径规划算法的性能。每个智能体按照其
        预设路径依次访问任务。
        
        执行流程：
        1. 获取需要决策的智能体
        2. 检查智能体是否有预设路径
        3. 按预设路径执行下一个动作
        4. 如果没有预设路径，智能体返回基地
        5. 重复直到所有任务完成或超时
        
        Args:
            path: 结果保存路径
            method: 方法标识符（用于文件命名）
            plot_figure: 是否生成动画可视化
            
        Returns:
            current_time: 仿真完成的总时间
        """
        self.plot_figure = plot_figure
        self.max_waiting_time = 200  # 设置最大等待时间
        
        # 主仿真循环：继续直到仿真完成或超时
        while not self.finished and self.current_time < 200:
            # 获取下次需要决策的智能体和时间
            decision_agents, current_time = self.next_decision()
            self.current_time = current_time
            
            # 合并所有需要决策的智能体
            decision_agents = decision_agents[0] + decision_agents[1]
            
            # 为每个需要决策的智能体执行动作
            for agent in decision_agents:
                # 检查智能体是否有预设路径
                agent_route = self.agent_dic[agent]['pre_set_route']
                
                if agent_route is None or not agent_route:
                    # 没有预设路径或路径已执行完毕：返回基地
                    self.agent_step(agent, 0, 0)  # 动作0表示返回基地
                    self.agent_dic[agent]['next_decision'] = np.nan  # 标记不再需要决策
                    continue
                
                # 执行预设路径中的下一个动作
                next_action = self.agent_dic[agent]['pre_set_route'].pop(0)
                self.agent_step(agent, next_action, 0)
            
            # 检查仿真是否完成
            self.finished = self.check_finished()
        
        # 如果启用可视化，生成动画
        if self.plot_figure:
            self.plot_animation(path, method)
        
        # 打印仿真完成时间
        print(self.current_time)
        return self.current_time

    def execute_greedy_action(self, path='./', method=0, plot_figure=False):
        """
        执行贪心策略的智能体任务分配
        
        这个方法实现了一个简单但有效的贪心任务分配策略，智能体总是
        选择距离最近的可执行任务。这种策略常用作基线算法，用于
        与强化学习等更复杂算法进行性能对比。
        
        贪心策略逻辑：
        1. 智能体观察当前环境状态
        2. 获取所有可选择的任务（未被掩码的任务）
        3. 计算到每个可选任务的欧几里得距离
        4. 选择距离最近的任务执行
        5. 如果没有可选任务，返回基地
        
        优点：计算简单、决策快速、避免长距离移动
        缺点：缺乏全局优化、可能导致次优分配
        
        Args:
            path: 结果保存路径
            method: 方法标识符（用于文件命名）
            plot_figure: 是否生成动画可视化
            
        Returns:
            current_time: 仿真完成的总时间
        """
        self.plot_figure = plot_figure
        
        # 贪心算法特殊初始化：让所有智能体在时间0就开始决策
        # 这是因为 agent_update 会将在基地的智能体的 next_decision 设为 inf
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:  # 在基地的智能体
                agent['next_decision'] = 0
        
        # 添加循环计数器，防止无限循环
        iteration_count = 0
        max_iterations = 100000  # 设置最大迭代次数
        last_time = -1  # 记录上次的时间
        stuck_count = 0  # 记录时间停滞的次数
        
        # 主仿真循环：继续直到仿真完成或超时
        while not self.finished and self.current_time < 200:
            iteration_count += 1
            
            # 检查是否超过最大迭代次数
            if iteration_count > max_iterations:
                print(f"警告：达到最大迭代次数 {max_iterations}，强制退出循环")
                print(f"当前时间: {self.current_time}, finished: {self.finished}")
                break
            
            # 检查时间是否停滞不前
            if self.current_time == last_time:
                stuck_count += 1
                if stuck_count > 100:  # 如果时间连续100次不变，强制退出
                    print(f"警告：时间停滞 {stuck_count} 次，强制退出")
                    print(f"当前时间: {self.current_time}, 迭代次数: {iteration_count}")
                    break
            else:
                stuck_count = 0  # 时间有推进，重置计数器
                last_time = self.current_time
            
            # 每1000次迭代打印一次进度
            if iteration_count % 1000 == 0:
                print(f"迭代次数: {iteration_count}, 当前时间: {self.current_time:.2f}")
            
            # 获取下次需要决策的智能体和时间
            release_agents, current_time = self.next_decision()
            self.current_time = current_time
            
            # 处理所有需要决策的智能体
            while release_agents[0] or release_agents[1]:
                # 获取下一个需要决策的智能体
                agent_id = (release_agents[0].pop(0) if release_agents[0] 
                           else release_agents[1].pop(0))
                agent = self.agent_dic[agent_id]
                # 获取智能体的观察信息（任务状态、动作掩码等）
                tasks_info, agents_info, mask = self.agent_observe(agent_id, max_waiting=True)
                
                # 贪心选择：寻找距离最近的可选任务
                min_dist = np.inf  # 初始化最小距离为无穷大
                best_action = None  # 最优动作
                
                # 遍历所有可能的动作（任务）
                for task_id, is_masked in enumerate(mask[0, :]):
                    if not is_masked:  # 如果该任务未被掩码（可选择）
                        # 如果智能体已在基地，跳过"返回基地"选项（task_id=0）
                        if task_id == 0 and agent['current_task'] < 0:
                            continue
                        
                        if task_id - 1 >= 0:
                            # 计算到任务的距离
                            dist = self.calculate_eulidean_distance(agent, self.task_dic[task_id - 1])
                        else:
                            # 计算到基地的距离（task_id = 0）
                            dist = self.calculate_eulidean_distance(agent, 
                                                                  self.depot_dic[agent['species']])
                        
                        # 更新最小距离和最优动作
                        if dist < min_dist:
                            min_dist = dist
                            best_action = task_id
                
                # 如果没有找到任何可选任务，强制返回基地
                if best_action is None:
                    best_action = 0
                
                # 执行选择的动作
                self.agent_step(agent_id, best_action, 0)
            
            # 检查仿真是否完成
            self.finished = self.check_finished()
        
        # 如果启用可视化，生成动画
        if self.plot_figure:
            self.plot_animation(path, method)
        
        # 打印仿真完成时间
        print(self.current_time)
        return self.current_time

    def pre_set_route(self, routes, agent_id):
        """
        为智能体设置预定义的任务执行路径
        
        这个方法允许为特定智能体预先规划任务访问序列，常用于：
        1. 测试特定的路径规划算法
        2. 验证理论最优解
        3. 实现确定性的任务分配策略
        4. 作为强化学习算法的初始化或参考
        
        路径格式：
        - 路径是一个包含任务ID的列表
        - 任务ID从1开始（1表示第一个任务）
        - 0表示返回基地
        - 路径将按顺序依次执行
        
        Args:
            routes: 任务ID列表，表示智能体应该访问的任务序列
            agent_id: 目标智能体的ID
            
        Example:
            # 让智能体0按顺序执行任务1、任务3、然后返回基地
            env.pre_set_route([1, 3, 0], 0)
        """
        # 检查智能体是否已有预设路径
        if not self.agent_dic[agent_id]['pre_set_route']:
            # 没有预设路径，直接设置新路径
            self.agent_dic[agent_id]['pre_set_route'] = routes
        else:
            # 已有预设路径，将新路径追加到现有路径后面
            self.agent_dic[agent_id]['pre_set_route'] += routes

    def process_map(self, path):
        """
        处理和分析任务完成情况的时间序列数据
        
        这个方法对仿真结果进行后处理分析，生成任务完成率随时间变化的
        统计数据。主要用于算法性能评估和可视化分析。
        
        分析流程：
        1. 按任务需求对任务进行分组
        2. 提取每组任务的完成时间
        3. 生成时间序列的完成率统计
        4. 保存为CSV文件供进一步分析
        
        输出数据格式：
        - 每列代表一个时间点（从0到仿真结束，步长0.1）
        - 每行代表一种任务类型的完成率
        - 值范围[0,1]，表示该时间点该类型任务的完成比例
        
        Args:
            path: 数据保存路径
        """
        import pandas as pd
        
        # 步骤1：按任务需求对任务进行分组
        grouped_tasks = dict()
        
        # 获取所有不同的任务需求类型
        task_requirements = self.get_matrix(self.task_dic, 'requirements')
        # 注意：这里假设requirements是标量，但实际可能是向量
        # 如果是向量，需要修改为适当的分组键
        groups = list(set(np.array(task_requirements).squeeze().tolist()))
        
        # 为每种需求类型初始化分组字典
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        
        # 为每组任务分配索引
        index = np.zeros_like(groups)
        
        # 将任务分配到对应的组中
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])  # 假设requirements可以转换为整数
            group_index = groups.index(requirement)
            ind = index[group_index]
            grouped_tasks[requirement].update({ind: task})
            index[group_index] += 1
        
        # 过滤掉空组
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        
        # 步骤2：提取每组任务的完成时间
        time_finished = [self.get_matrix(dic, 'time_finish') for dic in grouped_tasks.values()]
        
        # 步骤3：生成时间序列统计
        t = 0
        time_tick_stamp = dict()
        
        # 以0.1为步长遍历整个仿真时间
        while t <= self.current_time:
            # 计算每种任务类型在时间t的完成率
            completion_rates = []
            for task_group_times in time_finished:
                # 计算该组中在时间t之前完成的任务比例
                completed_count = np.sum(np.array(task_group_times) < t)
                completion_rate = completed_count / len(task_group_times)
                completion_rates.append(completion_rate)
            
            time_tick_stamp[t] = completion_rates
            t += 0.1
            t = np.round(t, 1)  # 避免浮点数精度问题
        
        # 步骤4：保存数据到CSV文件
        df = pd.DataFrame(time_tick_stamp)
        df.to_csv(f'{path}time_RL.csv')
        
        print(f"任务完成分析数据已保存到: {path}time_RL.csv")
        print(f"包含 {len(grouped_tasks)} 种任务类型的时间序列数据")

    def add_dynamic_task(self, num_tasks=1, appear_time=None, location=None, 
                        requirements=None, deadline=None, duration=None, 
                        is_urgent=None):
        """
        动态添加新任务到环境中
        
        该方法允许在仿真运行过程中动态添加新任务。新任务会被分配唯一的ID，
        并在指定的时间出现。这使得环境能够模拟真实场景中任务动态到达的情况。
        
        Args:
            num_tasks: 要添加的任务数量（默认1个）
            appear_time: 任务出现时间（默认为当前时间）
            location: 任务位置，形状为(num_tasks, 2)或(2,)，如果为None则随机生成
            requirements: 任务能力需求，形状为(num_tasks, traits_dim)，如果为None则随机生成
            deadline: 任务截止时间，可以是单个值或数组，如果为None则自动计算
            duration: 任务持续时间，可以是单个值或数组，如果为None则随机生成
            is_urgent: 是否为紧急任务，可以是单个值或数组，如果为None则随机决定
            
        Returns:
            new_task_ids: 新添加任务的ID列表
            
        Example:
            # 添加一个任务，在时间10出现，位置随机
            env.add_dynamic_task(num_tasks=1, appear_time=10.0)
            
            # 添加3个任务，指定位置和需求
            locations = np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
            reqs = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 1]])
            env.add_dynamic_task(num_tasks=3, appear_time=15.0, 
                               location=locations, requirements=reqs)
        """
        # 设置默认出现时间为当前时间
        if appear_time is None:
            appear_time = self.current_time
        
        # 获取新任务的起始ID（从现有最大ID+1开始）
        if len(self.task_dic) > 0:
            new_task_id_start = max(self.task_dic.keys()) + 1
        else:
            new_task_id_start = 0
        
        new_task_ids = []
        
        # 为每个新任务生成或使用提供的属性
        for i in range(num_tasks):
            task_id = new_task_id_start + i
            
            # ==================== 处理任务位置 ====================
            if location is None:
                # 随机生成位置
                task_location = self.random_value(1, 2).flatten()
            else:
                # 使用提供的位置
                if location.ndim == 1:  # 单个任务的位置
                    task_location = location.copy()
                else:  # 多个任务的位置
                    task_location = location[i, :].copy()
            
            # ==================== 处理任务需求 ====================
            if requirements is None:
                # 按照环境设置随机生成需求
                if self.binary_task:
                    # 二元任务模式
                    task_requirements = self.random_int(0, 2, (1, self.traits_dim)).flatten()
                    # 确保至少需要一种能力
                    while np.sum(task_requirements) == 0:
                        task_requirements = self.random_int(0, 2, (1, self.traits_dim)).flatten()
                else:
                    # 加性任务模式
                    task_requirements = self.random_int(0, self.max_task_size, 
                                                       (1, self.traits_dim)).flatten()
                    # 确保至少需要一种能力
                    while np.sum(task_requirements) == 0:
                        task_requirements = self.random_int(0, self.max_task_size, 
                                                           (1, self.traits_dim)).flatten()
            else:
                # 使用提供的需求
                if requirements.ndim == 1:  # 单个任务的需求
                    task_requirements = requirements.copy()
                else:  # 多个任务的需求
                    task_requirements = requirements[i, :].copy()
            
            # ==================== 处理任务持续时间 ====================
            if duration is None:
                # 随机生成持续时间
                task_duration = float(self.random_value(1, 1)[0, 0] * self.duration_scale)
            else:
                # 使用提供的持续时间
                if isinstance(duration, (list, np.ndarray)):
                    task_duration = float(duration[i])
                else:
                    task_duration = float(duration)
            
            # ==================== 处理截止时间 ====================
            if deadline is None:
                # 自动计算截止时间
                max_speed = 0.2  # 机器人最大速度
                
                # 计算到所有基地的最大距离
                distances_to_depots = [np.linalg.norm(task_location - self.depot_dic[s]['location']) 
                                      for s in range(self.species_num)]
                max_distance = max(distances_to_depots)
                
                # 最小截止时间：从出现时间开始，考虑距离和缓冲时间
                min_deadline = appear_time + max_distance / max_speed + 20
                
                # 最大截止时间：出现时间 + 40分钟
                max_deadline = appear_time + 40
                
                # 随机生成截止时间
                task_deadline = float(np.random.rand() * (max_deadline - min_deadline) + min_deadline)
            else:
                # 使用提供的截止时间
                if isinstance(deadline, (list, np.ndarray)):
                    task_deadline = float(deadline[i])
                else:
                    task_deadline = float(deadline)
            
            # ==================== 处理紧急程度 ====================
            if is_urgent is None:
                # 70%概率为紧急任务
                task_is_urgent = bool(np.random.rand() < 0.7)
                # 如果不紧急，延长截止时间
                if not task_is_urgent:
                    task_deadline = appear_time + 40
            else:
                # 使用提供的紧急程度
                if isinstance(is_urgent, (list, np.ndarray)):
                    task_is_urgent = bool(is_urgent[i])
                else:
                    task_is_urgent = bool(is_urgent)
            
            # ==================== 创建新任务 ====================
            new_task = {
                'ID': task_id,                              # 任务ID
                'requirements': task_requirements,          # 任务能力需求
                'members': [],                              # 分配到此任务的智能体列表
                'cost': [],                                 # 每个智能体的成本
                'location': task_location,                  # 任务位置坐标
                'deadline': task_deadline,                  # 任务截止时间
                'is_urgent': task_is_urgent,                # 是否为紧急任务
                'feasible_assignment': False,               # 是否有可行的智能体分配
                'finished': False,                          # 任务是否完成
                'time_start': 0,                           # 任务开始时间
                'time_finish': 0,                          # 任务结束时间
                'status': task_requirements.copy(),         # 当前任务状态（剩余需求）
                'time': task_duration,                      # 任务持续时间
                'sum_waiting_time': 0,                     # 总等待时间
                'efficiency': 0,                           # 任务效率
                'abandoned_agent': [],                     # 被放弃的智能体列表
                'optimized_ability': None,                 # 优化后的能力分配
                'optimized_species': [],                   # 优化后的种类分配
                'appear_time': appear_time,                # 任务出现时间（动态任务特征）
                'is_dynamic': True                         # 标记为动态添加的任务
            }
            
            # 添加到任务字典中
            self.task_dic[task_id] = new_task
            new_task_ids.append(task_id)
            
            # 更新任务总数
            self.tasks_num = len(self.task_dic)
            
            # 打印添加信息（如果详细模式开启）
            if hasattr(self, 'verbose') and self.verbose:
                print(f"[动态任务] 时间 {appear_time:.2f}: 添加任务 {task_id}")
                print(f"  位置: {task_location}")
                print(f"  需求: {task_requirements}")
                print(f"  截止时间: {task_deadline:.2f}")
                print(f"  持续时间: {task_duration:.2f}")
                print(f"  紧急程度: {'紧急' if task_is_urgent else '普通'}")
        
        return new_task_ids
    
    def get_dynamic_tasks_at_time(self, time=None):
        """
        获取在指定时间或之前出现的所有动态任务
        
        Args:
            time: 查询时间，如果为None则使用当前时间
            
        Returns:
            dynamic_task_ids: 已出现的动态任务ID列表
        """
        if time is None:
            time = self.current_time
        
        dynamic_task_ids = []
        for task_id, task in self.task_dic.items():
            if task.get('is_dynamic', False) and task['appear_time'] <= time:
                dynamic_task_ids.append(task_id)
        
        return dynamic_task_ids
    
    def get_task_statistics(self):
        """
        获取任务统计信息（包括静态和动态任务）
        
        Returns:
            stats: 包含任务统计信息的字典
        """
        total_tasks = len(self.task_dic)
        static_tasks = sum(1 for t in self.task_dic.values() if not t.get('is_dynamic', False))
        dynamic_tasks = sum(1 for t in self.task_dic.values() if t.get('is_dynamic', False))
        finished_tasks = sum(1 for t in self.task_dic.values() if t['finished'])
        unfinished_tasks = total_tasks - finished_tasks
        
        # 计算已出现的动态任务
        appeared_dynamic_tasks = sum(1 for t in self.task_dic.values() 
                                    if t.get('is_dynamic', False) and 
                                    t['appear_time'] <= self.current_time)
        
        stats = {
            'total_tasks': total_tasks,
            'static_tasks': static_tasks,
            'dynamic_tasks': dynamic_tasks,
            'appeared_dynamic_tasks': appeared_dynamic_tasks,
            'finished_tasks': finished_tasks,
            'unfinished_tasks': unfinished_tasks,
            'current_time': self.current_time
        }
        
        return stats


if __name__ == '__main__':
    """
    测试代码：生成标准测试集
    
    创建50个不同的环境实例，用于算法性能评估和对比。
    每个实例使用不同的随机种子，确保测试的多样性和可重现性。
    """
    import pickle
    
    # 设置测试集名称和参数
    testSet = 'TestSet_Deadline'
    os.makedirs(f'../{testSet}', exist_ok=True)  # 如果目录已存在则不报错
    
    # 生成50个测试环境实例
    for i in range(50):
        # 创建环境：3-3个智能体/种类，5种类，20个任务，5维能力
        env = TaskEnv((3, 3), (5, 5), (20, 20), 5, seed=i)
        
        # 保存环境实例到pickle文件
        pickle.dump(env, open(f'../{testSet}/env_{i}.pkl', 'wb'))
        
        # 打印环境信息（包括充电站）
        if i == 0:
            print(f"环境 {i} 已创建:")
            print(f"  - 智能体数量: {env.agents_num}")
            print(f"  - 任务数量: {env.tasks_num}")
            print(f"  - 种类数量: {env.species_num}")
            print(f"  - 充电站数量: {len(env.charging_station_dic)}")
            print(f"  - 初始电量: {env.initial_battery}")
            print(f"  - 移动耗电速率: {env.battery_consume_moving}/min")
            print(f"  - 静止耗电速率: {env.battery_consume_idle}/min")
        
    # 初始化最后一个环境的状态
    env.init_state()
    print(f"\n已生成 50 个测试环境，保存在 {testSet}/ 目录")
