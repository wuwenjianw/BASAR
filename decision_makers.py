#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
决策器集合 (Decision Makers)

提供不同的任务分配策略，可作为外部决策器注入到DynamicCentralizedPlanner中。

每个决策器的函数签名:
    decision_maker(agent_id, available_tasks, env) -> task_id or None
    
    Args:
        agent_id: 智能体ID
        available_tasks: 可用任务ID列表
        env: TaskEnv环境实例
        
    Returns:
        task_id: 选择的任务ID，如果没有合适的任务则返回None
"""

import numpy as np


class GreedyDecisionMaker:
    """
    贪婪决策器
    
    策略：为智能体选择距离最近的可行任务
    """
    
    def __init__(self):
        pass
    
    def __call__(self, agent_id, available_tasks, env):
        """
        为智能体选择任务
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务ID列表
            env: TaskEnv环境实例
            
        Returns:
            task_id: 选择的任务ID，如果没有合适的任务则返回None
        """
        if not available_tasks:
            return None
        
        agent = env.agent_dic[agent_id]
        agent_loc = agent['location']
        agent_abilities = agent['abilities']
        
        # 检查电量是否过低
        if env.check_battery_critical(agent_id):
            return None  # 需要充电，不分配任务
        
        best_task = None
        min_distance = float('inf')
        
        for task_id in available_tasks:
            task = env.task_dic[task_id]
            
            # 检查智能体能力是否匹配任务**当前剩余需求**
            task_status = task['status']  # 使用status而不是requirements
            if not self._can_contribute(agent_abilities, task_status):
                continue
            
            # 检查电量是否足够到达任务点并返回充电站
            if not env.can_reach_with_battery(agent_id, task):
                continue
            
            # 计算距离
            distance = np.linalg.norm(agent_loc - task['location'])
            
            # 选择最近的任务
            if distance < min_distance:
                min_distance = distance
                best_task = task_id
        
        return best_task
    
    def _can_contribute(self, agent_abilities, task_requirements):
        """
        检查智能体是否能为任务贡献能力
        
        Args:
            agent_abilities: 智能体能力向量
            task_requirements: 任务需求向量
            
        Returns:
            bool: 是否能贡献
        """
        # 智能体能力与任务需求的交集
        contribution = np.minimum(agent_abilities, task_requirements)
        # 如果有任何能力可以贡献，返回True
        return np.any(contribution > 0)


class HRLFDecisionMaker:
    """
    HRLF (Hierarchical Reinforcement Learning Framework) 决策器
    
    这是一个占位符，需要集成实际的HRLF模型
    """
    
    def __init__(self, model_path=None):
        """
        初始化HRLF决策器
        
        Args:
            model_path: 训练好的模型路径
        """
        self.model_path = model_path
        # TODO: 加载HRLF模型
        print(f"[HRLF] 初始化决策器 (model_path={model_path})")
    
    def __call__(self, agent_id, available_tasks, env):
        """
        使用HRLF模型为智能体选择任务
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务ID列表
            env: TaskEnv环境实例
            
        Returns:
            task_id: 选择的任务ID
        """
        if not available_tasks:
            return None
        
        # TODO: 实现HRLF模型推理
        # 1. 构造状态表示
        # 2. 调用模型获取动作
        # 3. 将动作映射到任务ID
        
        # 当前占位符：随机选择
        print(f"[HRLF] 为智能体 {agent_id} 选择任务 (从 {len(available_tasks)} 个可用任务中)")
        return np.random.choice(available_tasks)


class RandomDecisionMaker:
    """
    随机决策器
    
    策略：从可用任务中随机选择（用于基准测试）
    """
    
    def __init__(self, seed=None):
        """
        初始化随机决策器
        
        Args:
            seed: 随机种子
        """
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, agent_id, available_tasks, env):
        """
        随机为智能体选择任务
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务ID列表
            env: TaskEnv环境实例
            
        Returns:
            task_id: 随机选择的任务ID
        """
        if not available_tasks:
            return None
        
        agent = env.agent_dic[agent_id]
        agent_abilities = agent['abilities']
        
        # 过滤出智能体能够完成的任务
        feasible_tasks = []
        for task_id in available_tasks:
            task = env.task_dic[task_id]
            contribution = np.minimum(agent_abilities, task['requirements'])
            if np.any(contribution > 0):
                feasible_tasks.append(task_id)
        
        if not feasible_tasks:
            return None
        
        return self.rng.choice(feasible_tasks)


class NearestUnfinishedDecisionMaker:
    """
    最近未完成任务优先决策器
    
    策略：优先选择距离最近且未完成的任务，考虑任务的完成度
    """
    
    def __init__(self, completion_weight=0.5):
        """
        初始化决策器
        
        Args:
            completion_weight: 任务完成度权重 (0-1)
                             0表示只考虑距离，1表示只考虑完成度
        """
        self.completion_weight = completion_weight
    
    def __call__(self, agent_id, available_tasks, env):
        """
        为智能体选择任务
        
        Args:
            agent_id: 智能体ID
            available_tasks: 可用任务ID列表
            env: TaskEnv环境实例
            
        Returns:
            task_id: 选择的任务ID
        """
        if not available_tasks:
            return None
        
        agent = env.agent_dic[agent_id]
        agent_loc = agent['location']
        agent_abilities = agent['abilities']
        
        best_task = None
        min_score = float('inf')
        
        for task_id in available_tasks:
            task = env.task_dic[task_id]
            
            # 检查能力匹配
            contribution = np.minimum(agent_abilities, task['requirements'])
            if not np.any(contribution > 0):
                continue
            
            # 计算距离
            distance = np.linalg.norm(agent_loc - task['location'])
            
            # 计算完成度（已贡献能力 / 总需求能力）
            current_abilities = task.get('current_abilities', np.zeros_like(task['requirements']))
            completion = np.sum(current_abilities) / max(np.sum(task['requirements']), 1e-6)
            
            # 综合评分（距离越小越好，完成度越低越优先）
            # 归一化距离到 [0, 1]
            normalized_distance = distance / 2.0  # 假设地图最大距离为sqrt(2)
            score = (1 - self.completion_weight) * normalized_distance + self.completion_weight * completion
            
            if score < min_score:
                min_score = score
                best_task = task_id
        
        return best_task


# 便捷函数：创建不同类型的决策器
def create_decision_maker(maker_type='greedy', **kwargs):
    """
    创建决策器实例
    
    Args:
        maker_type: 决策器类型
            - 'greedy': 贪婪决策器
            - 'hrlf': HRLF决策器
            - 'random': 随机决策器
            - 'nearest': 最近未完成任务优先决策器
        **kwargs: 决策器特定的参数
        
    Returns:
        decision_maker: 决策器实例
    """
    if maker_type == 'greedy':
        return GreedyDecisionMaker()
    elif maker_type == 'hrlf':
        return HRLFDecisionMaker(**kwargs)
    elif maker_type == 'random':
        return RandomDecisionMaker(**kwargs)
    elif maker_type == 'nearest':
        return NearestUnfinishedDecisionMaker(**kwargs)
    else:
        raise ValueError(f"Unknown decision maker type: {maker_type}")
