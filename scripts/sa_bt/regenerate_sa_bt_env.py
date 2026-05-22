#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成环境文件 - SA-BT场景，目标：
1. 截止日期满足率约60%
2. 充电次数约5次
"""

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv


def regenerate_challenging_env():
    """
    生成更具挑战性的环境
    
    调整策略:
    1. SA-BT场景 (single_skill=True, binary_task=True)
    2. 更紧的截止时间 -> 降低deadline满足率
    3. 更多任务/更长持续时间 -> 增加充电次数
    4. 更高的任务到达密度 -> 增加压力
    """
    print("=" * 80)
    print("生成挑战性环境: SA-BT场景")
    print("目标: 截止日期满足率~60%, 充电次数~5次")
    print("=" * 80)
    
    # 创建环境实例
    print("\n[步骤1] 创建环境实例 (SA-BT)")
    env = TaskEnv(
        per_species_range=(3, 3),   # 每个种类3个智能体 (共15个)
        species_range=(5, 5),       # 5个种类
        tasks_range=(25, 25),       # 增加初始任务到25个（之前20个）
        traits_dim=5,               # 能力维度
        decision_dim=6,             # 决策维度
        max_task_size=2,            # 任务最大规模
        duration_scale=8,           # 增加持续时间缩放（之前5，现在8）
        seed=42,                    # 固定随机种子
        plot_figure=False,
        single_skill=True,          # SA: Single-skill Agent
        binary_task=True,           # BT: Binary Task
        use_deadline=True
    )
    
    # 验证参数
    print("\n[步骤2] 验证场景配置")
    print(f"  场景类型: SA-BT (Single-skill Agent, Binary Task)")
    print(f"  Single Skill: {env.single_skill}")
    print(f"  Binary Task: {env.binary_task}")
    
    print(f"\n[步骤3] 验证电量系统参数")
    print(f"  初始电量: {env.initial_battery}%")
    print(f"  移动耗电速率: {env.battery_consume_moving}% / 分钟")
    print(f"  静止耗电速率: {env.battery_consume_idle}% / 分钟")
    print(f"  最低电量阈值: {env.battery_min_threshold}%")
    
    print(f"\n[步骤4] 验证环境规模")
    print(f"  智能体总数: {env.agents_num}")
    print(f"  初始任务数: {env.tasks_num}")
    print(f"  持续时间缩放: {env.duration_scale}")
    
    # 检查deadline分布
    print(f"\n[步骤5] 验证任务截止时间分布")
    deadlines = [task['deadline'] for task in env.task_dic.values()]
    import numpy as np
    print(f"  最短deadline: {np.min(deadlines):.2f} 分钟")
    print(f"  最长deadline: {np.max(deadlines):.2f} 分钟")
    print(f"  平均deadline: {np.mean(deadlines):.2f} 分钟")
    print(f"  中位数deadline: {np.median(deadlines):.2f} 分钟")
    
    # 检查智能体能力 (SA模式下应该是单技能)
    print(f"\n[步骤6] 验证智能体能力 (SA模式)")
    for agent_id in range(min(3, env.agents_num)):
        agent = env.agent_dic[agent_id]
        print(f"  智能体 {agent_id}: 能力 {agent['abilities']}")
    
    # 检查任务需求 (BT模式下应该是二元)
    print(f"\n[步骤7] 验证任务需求 (BT模式)")
    for task_id in range(min(3, env.tasks_num)):
        task = env.task_dic[task_id]
        print(f"  任务 {task_id}: 需求 {task['status']}")
    
    # 保存环境
    print(f"\n[步骤8] 保存环境到文件")
    output_file = 'env_sa_bt_challenging.pkl'
    
    # 备份旧文件（如果存在）
    import os
    import shutil
    if os.path.exists(output_file):
        backup_file = f'{output_file}.bak'
        shutil.copy(output_file, backup_file)
        print(f"  ✓ 已备份旧文件到: {backup_file}")
    
    # 保存新文件
    with open(output_file, 'wb') as f:
        pickle.dump(env, f)
    print(f"  ✓ 新环境已保存到: {output_file}")
    
    # 验证保存的文件
    print(f"\n[步骤9] 验证保存的文件")
    with open(output_file, 'rb') as f:
        loaded_env = pickle.load(f)
    
    print(f"  ✓ 文件加载成功")
    print(f"  ✓ 智能体数量: {loaded_env.agents_num}")
    print(f"  ✓ 任务数量: {loaded_env.tasks_num}")
    print(f"  ✓ SA模式: {loaded_env.single_skill}")
    print(f"  ✓ BT模式: {loaded_env.binary_task}")
    print(f"  ✓ 移动耗电速率: {loaded_env.battery_consume_moving}% / 分钟")
    
    # 理论分析
    available_battery = loaded_env.initial_battery - loaded_env.battery_min_threshold
    max_moving_time = available_battery / loaded_env.battery_consume_moving
    print(f"\n[理论分析]")
    print(f"  可用电量: {available_battery}%")
    print(f"  理论最长连续移动时间: {max_moving_time:.2f} 分钟")
    print(f"  初始任务数: {loaded_env.tasks_num}")
    print(f"  建议动态任务数: 35-40 (总任务60-65)")
    print(f"  建议任务到达率: 0.4-0.5 任务/分钟")
    
    print("\n" + "=" * 80)
    print("环境文件生成完成！")
    print("下一步: 使用 test_sa_bt.py 进行测试")
    print("=" * 80)
    
    return env


if __name__ == '__main__':
    env = regenerate_challenging_env()
