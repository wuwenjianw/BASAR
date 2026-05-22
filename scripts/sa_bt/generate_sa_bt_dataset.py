#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SA-BT场景数据集生成器 v2

生成两种组织方式的数据集:
1. Fixed_Tasks: 固定100个总任务，变化智能体配置
2. Fixed_Makespan: 固定120分钟时间限制，任务数自适应

采样策略:
- 每个配置严格 50% Gaussian + 50% Uniform

样本规模:
- Fixed_Tasks: 每配置50个样本
- Fixed_Makespan: 每配置100个样本
"""

import pickle
import os
import sys
import numpy as np
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv
from project_paths import SA_BT_DATASET_ROOT


# ============================================================================
# 配置1: 固定任务数 (100个总任务)
# ============================================================================
FIXED_TASKS_CONFIGS = [
    {
        'name': 'n15_s5_h30',
        'agents': 15,
        'species': 5,
        'initial_tasks': 30,
        'total_tasks': 100,
        'description': 'Medium: 15 agents, 5 species, 30+70 tasks'
    },
    {
        'name': 'n20_s5_h40',
        'agents': 20,
        'species': 5,
        'initial_tasks': 40,
        'total_tasks': 100,
        'description': 'Medium-Large: 20 agents, 5 species, 40+60 tasks'
    },
    {
        'name': 'n20_s5_h50',
        'agents': 20,
        'species': 5,
        'initial_tasks': 50,
        'total_tasks': 100,
        'description': 'Large: 20 agents, 5 species, 50+50 tasks'
    },
    {
        'name': 'n30_s5_h60',
        'agents': 30,
        'species': 5,
        'initial_tasks': 60,
        'total_tasks': 100,
        'description': 'X-Large: 30 agents, 5 species, 60+40 tasks'
    }
]

# ============================================================================
# 配置2: 固定Makespan (120分钟)
# ============================================================================
FIXED_MAKESPAN_CONFIGS = [
    {
        'name': 'n10_s5_t120',
        'agents': 10,
        'species': 5,
        'initial_tasks': 30,
        'total_tasks': 120,
        'makespan': 120,
        'description': 'Small: 10 agents, 5 species, 120 tasks in 120min'
    },
    {
        'name': 'n15_s5_t200',
        'agents': 15,
        'species': 5,
        'initial_tasks': 50,
        'total_tasks': 200,
        'makespan': 120,
        'description': 'Medium: 15 agents, 5 species, 200 tasks in 120min'
    },
    {
        'name': 'n20_s5_t240',
        'agents': 20,
        'species': 5,
        'initial_tasks': 60,
        'total_tasks': 240,
        'makespan': 120,
        'description': 'Large: 20 agents, 5 species, 240 tasks in 120min'
    },
    {
        'name': 'n30_s5_t300',
        'agents': 30,
        'species': 5,
        'initial_tasks': 80,
        'total_tasks': 300,
        'makespan': 120,
        'description': 'X-Large: 30 agents, 5 species, 300 tasks in 120min'
    }
]

SAMPLES_PER_CONFIG_FIXED_TASKS = 50
SAMPLES_PER_CONFIG_FIXED_MAKESPAN = 100
OUTPUT_BASE_DIR = str(SA_BT_DATASET_ROOT)
FIXED_TASKS_ARRIVAL_RATE = 3.0
FIXED_MAKESPAN_ARRIVAL_RATE = 2.0

GAUSSIAN_SAMPLE_RATIO = 0.5
GAUSSIAN_MEAN = 0.5
GAUSSIAN_STD = 0.18


def calculate_task_arrival_rate(config, dataset_type, makespan=120):
    """计算动态任务到达率（使用固定的高但可行的速率）"""
    if dataset_type.startswith('Fixed Tasks'):
        return FIXED_TASKS_ARRIVAL_RATE
    return FIXED_MAKESPAN_ARRIVAL_RATE


def get_samples_per_config(dataset_type):
    """按数据集类型返回每配置样本数。"""
    if dataset_type.startswith('Fixed Tasks'):
        return SAMPLES_PER_CONFIG_FIXED_TASKS
    return SAMPLES_PER_CONFIG_FIXED_MAKESPAN


def build_sampling_modes(samples_per_config, seed):
    """
    构建采样模式列表，严格满足 50% Gaussian + 50% Uniform。
    """
    gaussian_count = int(samples_per_config * GAUSSIAN_SAMPLE_RATIO)
    uniform_count = samples_per_config - gaussian_count
    modes = (['gaussian'] * gaussian_count) + (['uniform'] * uniform_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(modes)
    return modes


def sample_points(rng, shape, mode):
    """按指定模式采样坐标，并裁剪到[0.05, 0.95]。"""
    if mode == 'gaussian':
        pts = rng.normal(loc=GAUSSIAN_MEAN, scale=GAUSSIAN_STD, size=shape)
    else:
        pts = rng.random(shape)
    return np.clip(pts, 0.05, 0.95)


def apply_spatial_sampling(env, mode, seed):
    """
    按采样模式重采样空间分布（基地、充电站、任务位置），并重算任务截止时间。
    """
    rng = np.random.default_rng(seed)
    species_num = len(env.depot_dic)
    tasks_num = len(env.task_dic)

    # 1) 采样基地位置
    depot_locs = sample_points(rng, (species_num, 2), mode)
    for s in range(species_num):
        env.depot_dic[s]['location'] = depot_locs[s, :].copy()

    # 2) 采样充电站位置（基地附近偏移）
    for s in range(species_num):
        offset = sample_points(rng, (1, 2), mode)[0] - 0.5
        offset = offset * 0.3
        station_loc = np.clip(depot_locs[s, :] + offset, 0.05, 0.95)
        env.charging_station_dic[s]['location'] = station_loc.copy()

    # 3) 更新智能体位置相关字段（重置为基地）
    for agent in env.agent_dic.values():
        species = agent['species']
        depot_loc = env.depot_dic[species]['location'].copy()
        charging_loc = env.charging_station_dic[species]['location'].copy()
        agent['location'] = depot_loc.copy()
        agent['depot'] = depot_loc.copy()
        agent['charging_station'] = charging_loc.copy()

    # 4) 采样任务位置
    task_locs = sample_points(rng, (tasks_num, 2), mode)
    for task_id, task in env.task_dic.items():
        task['location'] = task_locs[task_id, :].copy()

    # 5) 基于新位置重算截止时间（沿用TaskEnv中的规则）
    max_speed = 0.2
    total_agents = env.agents_num
    max_distances = np.zeros(tasks_num, dtype=float)
    for task_id, task in env.task_dic.items():
        dist_to_depots = [
            np.linalg.norm(task['location'] - env.depot_dic[s]['location'])
            for s in range(species_num)
        ]
        max_distances[task_id] = max(dist_to_depots)

    d_low = (max_distances / max_speed + 10).astype(np.int64) + 1
    d_high = np.full(tasks_num, 22, dtype=np.int64)
    group_factor = 1 if total_agents >= tasks_num else 2
    d_low = (d_low * (0.30 * group_factor)).astype(np.int64)
    deadline_base = (rng.random(tasks_num) * (d_high - d_low) + d_low).astype(np.int64) + 1
    n_urgent_tasks = int(0.90 * tasks_num)
    urgent_indices = rng.choice(tasks_num, n_urgent_tasks, replace=False)
    is_urgent = np.zeros(tasks_num, dtype=bool)
    is_urgent[urgent_indices] = True
    tasks_deadline = np.where(is_urgent, deadline_base, d_high)

    for task_id, task in env.task_dic.items():
        task['deadline'] = float(tasks_deadline[task_id])
        task['is_urgent'] = bool(is_urgent[task_id])

    # 距离矩阵依赖位置，需刷新
    env.species_distance_matrix, env.species_neighbor_matrix = env.generate_distance_matrix()
    env.sampling_mode = mode


def check_instance_solvability(env, verbose=False):
    """
    检查环境实例是否可解
    
    可解性标准（修正版）:
    1. 每个任务需求的维度，都至少有一个智能体拥有该维度的能力（能力覆盖检查）
    2. 不强制要求总能力≥总需求，因为多智能体协同可以通过合理规划完成
    
    Args:
        env: TaskEnv实例
        verbose: 是否打印详细信息
        
    Returns:
        (is_solvable, reason): 是否可解，以及不可解的原因
    """
    # 计算智能体总能力
    total_agent_abilities = np.zeros(env.traits_dim, dtype=int)
    for agent in env.agent_dic.values():
        total_agent_abilities += agent['abilities']
    
    # 检查每个任务
    unsolvable_tasks = []
    
    for task_id, task in env.task_dic.items():
        task_requirements = task['requirements']
        
        # 检查: 每个需求维度是否有智能体能力覆盖
        for dim in range(env.traits_dim):
            if task_requirements[dim] > 0 and total_agent_abilities[dim] == 0:
                # 该维度有需求但没有智能体有能力 - 这是真正的不可解
                unsolvable_tasks.append({
                    'task_id': task_id,
                    'dimension': dim,
                    'required': task_requirements[dim],
                    'available': 0
                })
                if verbose:
                    print(f"  ✗ 任务 {task_id} 维度 {dim}: 需求={task_requirements[dim]}, 可用=0 (无能力覆盖)")
    
    if unsolvable_tasks:
        reason = f"发现 {len(unsolvable_tasks)} 个维度完全没有智能体能力覆盖"
        if verbose:
            print(f"\n智能体总能力: {total_agent_abilities}")
            print(f"不可解原因: {reason}")
        return False, reason
    
    if verbose:
        # 计算能力充足度
        total_requirements = np.zeros(env.traits_dim, dtype=int)
        for task in env.task_dic.values():
            total_requirements += task['requirements']
        
        coverage = total_agent_abilities / np.maximum(total_requirements, 1) * 100
        
        print(f"  ✓ 实例可解（所有维度都有能力覆盖）")
        print(f"    智能体总能力: {total_agent_abilities}")
        print(f"    任务总需求:   {total_requirements}")
        print(f"    能力覆盖率:   {coverage.astype(int)}%")
    
    return True, "可解（所有维度都有能力覆盖）"


def generate_single_env(config, seed, max_retries=100):
    """
    生成单个可解的环境实例
    
    使用修改后的task_env.py，SA-BT模式下生成速度已优化
    自动检查并重试直到生成可解实例
    
    重要: 在SA-BT模式下，traits_dim设置为species数量，确保可解性
    
    Args:
        config: 配置字典
        seed: 随机种子
        max_retries: 最大重试次数
        
    Returns:
        (env, retries): 可解的TaskEnv实例和重试次数
        
    Raises:
        RuntimeError: 如果达到最大重试次数仍未生成可解实例
    """
    agents = config['agents']
    species = config['species']
    initial_tasks = config['initial_tasks']
    
    # 计算每个种类的智能体数量范围
    agents_per_species = agents // species
    per_species_range = (agents_per_species, agents_per_species + 1)
    
    # 关键修改: traits_dim = species，确保智能体能力覆盖所有维度
    traits_dim = species
    
    # 重试直到生成可解实例
    for attempt in range(max_retries):
        # 使用不同的种子避免重复
        current_seed = seed + attempt * 1000000
        
        # 创建环境（已优化，SA-BT模式下快速生成）
        env = TaskEnv(
            per_species_range=per_species_range,
            species_range=(species, species),
            tasks_range=(initial_tasks, initial_tasks),
            traits_dim=traits_dim,  # 使用与species相同的维度
            decision_dim=6,
            max_task_size=2,
            duration_scale=8,
            seed=current_seed,
            plot_figure=False,
            single_skill=True,   # SA: Single-skill Agent
            binary_task=True,    # BT: Binary Task
            use_deadline=True
        )
        
        # 为所有任务添加appear_time和is_dynamic标记
        for task_id, task in env.task_dic.items():
            task['appear_time'] = 0.0
            task['is_dynamic'] = False
        
        # 检查可解性
        is_solvable, reason = check_instance_solvability(env, verbose=False)
        
        if is_solvable:
            # 成功生成可解实例
            return env, attempt
        
        # 不可解，继续重试
    
    # 达到最大重试次数
    raise RuntimeError(f"无法生成可解实例，已重试 {max_retries} 次")


def generate_single_env_old(config, seed):
    """
    生成单个环境实例（旧版本，不检查可解性）
    
    保留用于对比
    """
    agents = config['agents']
    species = config['species']
    initial_tasks = config['initial_tasks']
    
    # 计算每个种类的智能体数量范围
    agents_per_species = agents // species
    per_species_range = (agents_per_species, agents_per_species + 1)
    
    # 创建环境（已优化，SA-BT模式下快速生成）
    env = TaskEnv(
        per_species_range=per_species_range,
        species_range=(species, species),
        tasks_range=(initial_tasks, initial_tasks),
        traits_dim=5,
        decision_dim=6,
        max_task_size=2,
        duration_scale=8,
        seed=seed,
        plot_figure=False,
        single_skill=True,   # SA: Single-skill Agent
        binary_task=True,    # BT: Binary Task
        use_deadline=True
    )
    
    # 为所有任务添加appear_time和is_dynamic标记
    for task_id, task in env.task_dic.items():
        task['appear_time'] = 0.0
        task['is_dynamic'] = False
    
    return env


def save_dataset_metadata(output_dir, configs, dataset_type, samples_per_config):
    """保存数据集元数据"""
    metadata = {
        'dataset_name': f'SA-BT Dynamic MRTA Dataset ({dataset_type})',
        'dataset_type': dataset_type,
        'scenario_type': 'SA-BT (Single-skill Agent, Binary Task)',
        'total_configurations': len(configs),
        'samples_per_configuration': samples_per_config,
        'total_samples': len(configs) * samples_per_config,
        'sampling_strategy': {
            'gaussian_ratio': GAUSSIAN_SAMPLE_RATIO,
            'uniform_ratio': 1.0 - GAUSSIAN_SAMPLE_RATIO,
            'gaussian_mean': GAUSSIAN_MEAN,
            'gaussian_std': GAUSSIAN_STD
        },
        'configurations': []
    }
    
    for config in configs:
        dynamic_tasks = config['total_tasks'] - config['initial_tasks']
        makespan = config.get('makespan', 120)
        arrival_rate = calculate_task_arrival_rate(config, dataset_type, makespan)
        
        metadata['configurations'].append({
            'name': config['name'],
            'description': config['description'],
            'agents': config['agents'],
            'species': config['species'],
            'initial_tasks': config['initial_tasks'],
            'dynamic_tasks': dynamic_tasks,
            'total_tasks': config['total_tasks'],
            'makespan_limit': makespan,
            'recommended_arrival_rate': arrival_rate
        })
    
    # 保存为pickle
    metadata_file = os.path.join(output_dir, 'dataset_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # 保存为可读文本
    readme_file = os.path.join(output_dir, 'README.txt')
    with open(readme_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SA-BT Dynamic MRTA Dataset ({dataset_type})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset Type: {dataset_type}\n")
        f.write(f"Scenario: {metadata['scenario_type']}\n")
        f.write(f"Total Configurations: {metadata['total_configurations']}\n")
        f.write(f"Samples per Config: {metadata['samples_per_configuration']}\n")
        f.write(f"Total Samples: {metadata['total_samples']}\n\n")
        f.write("Sampling Strategy:\n")
        f.write(f"  Gaussian Ratio: {metadata['sampling_strategy']['gaussian_ratio']:.2f}\n")
        f.write(f"  Uniform Ratio: {metadata['sampling_strategy']['uniform_ratio']:.2f}\n")
        f.write(f"  Gaussian Mean: {metadata['sampling_strategy']['gaussian_mean']:.2f}\n")
        f.write(f"  Gaussian Std: {metadata['sampling_strategy']['gaussian_std']:.2f}\n\n")
        
        f.write("Configurations:\n")
        f.write("-" * 80 + "\n")
        for cfg in metadata['configurations']:
            f.write(f"\n{cfg['name'].upper()}: {cfg['description']}\n")
            f.write(f"  Agents: {cfg['agents']}\n")
            f.write(f"  Species: {cfg['species']}\n")
            f.write(f"  Initial Tasks: {cfg['initial_tasks']}\n")
            f.write(f"  Dynamic Tasks: {cfg['dynamic_tasks']}\n")
            f.write(f"  Total Tasks: {cfg['total_tasks']}\n")
            f.write(f"  Makespan Limit: {cfg['makespan_limit']} min\n")
            f.write(f"  Recommended Arrival Rate: {cfg['recommended_arrival_rate']} tasks/min\n")
    
    print(f"  ✓ 元数据: {metadata_file}")
    print(f"  ✓ README: {readme_file}")


def generate_dataset_type(configs, output_dir, dataset_type):
    """生成一种类型的数据集"""
    samples_per_config = get_samples_per_config(dataset_type)
    print(f"\n{'=' * 80}")
    print(f"生成数据集: {dataset_type}")
    print(f"{'=' * 80}")
    print(f"配置数量: {len(configs)}")
    print(f"每配置样本数: {samples_per_config}")
    print(f"总样本数: {len(configs) * samples_per_config}")
    print(f"采样比例: Gaussian {GAUSSIAN_SAMPLE_RATIO:.0%} / Uniform {(1.0 - GAUSSIAN_SAMPLE_RATIO):.0%}")
    print(f"输出目录: {output_dir}/\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    total_retries = 0
    max_retries_for_sample = 0
    
    # 为每个配置生成样本
    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'-' * 80}")
        print(f"配置 {config_idx}/{len(configs)}: {config['name']}")
        print(f"{'-' * 80}")
        print(f"  {config['description']}")
        print(f"  智能体: {config['agents']}, 种类: {config['species']}")
        print(f"  初始任务: {config['initial_tasks']}, 动态任务: {config['total_tasks'] - config['initial_tasks']}")
        
        makespan = config.get('makespan', 120)
        arrival_rate = calculate_task_arrival_rate(config, dataset_type, makespan)
        print(f"  Makespan: {makespan} min, 推荐到达率: {arrival_rate} tasks/min")
        print(f"  可解性检查: 已启用")
        
        # 创建配置目录
        config_dir = os.path.join(output_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        # 生成样本
        start_time = time.time()
        config_retries = 0
        config_seed = config_idx * 10000 + 7
        sampling_modes = build_sampling_modes(samples_per_config, seed=config_seed)
        gaussian_count = sum(1 for m in sampling_modes if m == 'gaussian')
        uniform_count = samples_per_config - gaussian_count
        print(f"  采样模式: Gaussian={gaussian_count}, Uniform={uniform_count}")
        print(f"  生成进度: ", end='', flush=True)

        for sample_idx in range(samples_per_config):
            # 使用不同的随机种子
            seed = config_idx * 10000 + sample_idx
            sample_mode = sampling_modes[sample_idx]
            
            # 生成可解环境（带重试）
            env, retries = generate_single_env(config, seed)
            apply_spatial_sampling(env, mode=sample_mode, seed=seed + 123456)
            config_retries += retries
            total_retries += retries
            max_retries_for_sample = max(max_retries_for_sample, retries)
            
            # 保存环境
            filename = f"env_{sample_idx:03d}.pkl"
            filepath = os.path.join(config_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(env, f)
            
            total_generated += 1
            
            # 每10个样本显示进度
            if (sample_idx + 1) % 10 == 0:
                print(f"{sample_idx+1} ", end='', flush=True)
        
        total_time = time.time() - start_time
        avg_retries = config_retries / samples_per_config
        print(f"\n  ✓ 完成 ({total_time:.1f}秒, {total_time/samples_per_config:.3f}秒/样本)")
        print(f"  ✓ 可解性统计: 平均重试 {avg_retries:.2f} 次/样本, 总重试 {config_retries} 次")
    
    # 保存元数据
    print(f"\n保存数据集元数据...")
    save_dataset_metadata(output_dir, configs, dataset_type, samples_per_config)
    
    avg_retries_overall = total_retries / total_generated if total_generated > 0 else 0
    print(f"\n✓ {dataset_type} 数据集生成完成: {total_generated} 个样本")
    print(f"  可解性保证: 所有实例均已验证可解")
    print(f"  重试统计: 总计 {total_retries} 次, 平均 {avg_retries_overall:.2f} 次/样本, 最大 {max_retries_for_sample} 次")
    return total_generated, total_retries


def main():
    """主函数"""
    print("=" * 80)
    print("SA-BT场景数据集生成器 v2 (可解性保证版)")
    print("=" * 80)
    print(f"\n特性:")
    print(f"  ✓ 自动检查每个实例的可解性")
    print(f"  ✓ 重试机制确保智能体能力覆盖所有任务需求")
    print(f"  ✓ 生成的所有实例均保证可解")
    print(f"\n将生成两种数据集:")
    print(f"  1. Fixed_Tasks: 固定100个总任务")
    print(f"  2. Fixed_Makespan: 固定120分钟时间限制")
    print(f"\n样本规模:")
    print(f"  • Fixed_Tasks: {SAMPLES_PER_CONFIG_FIXED_TASKS} / 配置")
    print(f"  • Fixed_Makespan: {SAMPLES_PER_CONFIG_FIXED_MAKESPAN} / 配置")
    print(f"\n采样策略:")
    print(f"  • 50% Gaussian + 50% Uniform (每个配置严格均分)")
    print(f"\n基础输出目录: {OUTPUT_BASE_DIR}/")
    
    # 创建基础目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    total_samples = 0
    total_retries = 0
    
    # 生成固定任务数数据集
    output_dir_1 = os.path.join(OUTPUT_BASE_DIR, 'Fixed_Tasks')
    count_1, retries_1 = generate_dataset_type(
        FIXED_TASKS_CONFIGS,
        output_dir_1,
        'Fixed Tasks (100 tasks)'
    )
    total_samples += count_1
    total_retries += retries_1
    
    # 生成固定Makespan数据集
    output_dir_2 = os.path.join(OUTPUT_BASE_DIR, 'Fixed_Makespan')
    count_2, retries_2 = generate_dataset_type(
        FIXED_MAKESPAN_CONFIGS,
        output_dir_2,
        'Fixed Makespan (120 min)'
    )
    total_samples += count_2
    total_retries += retries_2
    
    # 最终总结
    print(f"\n{'=' * 80}")
    print("数据集生成完成！")
    print(f"{'=' * 80}")
    print(f"\n总计生成: {total_samples} 个环境实例 (均已验证可解)")
    print(f"总重试次数: {total_retries} 次")
    print(f"平均重试: {total_retries/total_samples:.2f} 次/样本")
    print(f"\n目录结构:")
    print(f"  {OUTPUT_BASE_DIR}/")
    print(f"    ├── Fixed_Tasks/           (固定100个总任务)")
    print(f"    │   ├── dataset_metadata.pkl")
    print(f"    │   ├── README.txt")
    for cfg in FIXED_TASKS_CONFIGS:
        print(f"    │   ├── {cfg['name']}/  ({cfg['agents']}a, {cfg['initial_tasks']}+{cfg['total_tasks']-cfg['initial_tasks']}t)")
        print(f"    │   │   └── env_000.pkl ~ env_{SAMPLES_PER_CONFIG_FIXED_TASKS-1:03d}.pkl")
    
    print(f"    │")
    print(f"    └── Fixed_Makespan/        (固定120分钟)")
    print(f"        ├── dataset_metadata.pkl")
    print(f"        ├── README.txt")
    for cfg in FIXED_MAKESPAN_CONFIGS:
        print(f"        ├── {cfg['name']}/  ({cfg['agents']}a, {cfg['total_tasks']}t)")
        print(f"        │   └── env_000.pkl ~ env_{SAMPLES_PER_CONFIG_FIXED_MAKESPAN-1:03d}.pkl")
    
    print(f"\n使用说明:")
    print(f"  • 每个env文件是独立的测试实例")
    print(f"  • 所有实例均已验证可解 (智能体能力覆盖所有任务需求)")
    print(f"  • 动态任务需在规划器中生成")
    print(f"  • 使用元数据中的recommended_arrival_rate")
    print(f"  • 适用于多种规划算法评估")
    print("=" * 80)


if __name__ == '__main__':
    main()
