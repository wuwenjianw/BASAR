#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用贪婪算法评估SA-BT数据集 - 动态评估模式

与 evaluate_hrlf_on_sa_bt_dataset.py 完全相同的评估方式，
唯一区别是使用贪婪算法而不是HRLF模型。

评估指标（与HRLF评估完全一致）:
1. Success Rate (成功率)
2. Deadline Satisfaction Rate (截止日期满足率)
3. Deadline Violation (截止日期违反)
4. Makespan (完工时间)
5. Waiting Time (等待时间)
6. Travel Distance (行驶距离)
7. Flow Time (流程时间)
8. Planning CPU Time (规划时间)
9. Charging Times (充电次数)
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_centralized_planner import DynamicCentralizedPlanner
from dynamic_eval_stats import build_summary_like_my_model, calculate_metrics_like_my_model
from project_paths import ARTIFACTS_ROOT, SA_BT_DATASET_ROOT


def calculate_metrics(env, results, initial_task_count, dataset_type='Fixed_Tasks'):
    """计算评估指标，复用主模型同口径实现。"""
    return calculate_metrics_like_my_model(
        env=env,
        results=results,
        initial_task_count=initial_task_count,
        dataset_type=dataset_type,
    )


def main():
    """主函数 - 动态评估模式（与HRLF评估完全相同的流程）"""
    print("=" * 80)
    print("SA-BT数据集贪婪算法批量评估 - 动态模式")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 输出目录
    output_base_dir = ARTIFACTS_ROOT / 'results' / 'greedy_dynamic'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件
    log_path = output_base_dir / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, 'w', encoding='utf-8')
    
    log_file.write("SA-BT数据集贪婪算法动态评估日志\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")
    
    print("开始动态评估...\n")
    log_file.write("开始动态评估...\n\n")
    
    dataset_configs = {
        'Fixed_Tasks': [
            {'name': 'n15_s5_h30', 'agents': 15, 'species': 5, 'initial_tasks': 30, 'total_tasks': 100, 'recommended_arrival_rate': 3},
            {'name': 'n20_s5_h40', 'agents': 20, 'species': 5, 'initial_tasks': 40, 'total_tasks': 100, 'recommended_arrival_rate': 3},
            {'name': 'n20_s5_h50', 'agents': 20, 'species': 5, 'initial_tasks': 50, 'total_tasks': 100, 'recommended_arrival_rate': 3},
            {'name': 'n30_s5_h60', 'agents': 30, 'species': 5, 'initial_tasks': 60, 'total_tasks': 100, 'recommended_arrival_rate': 3},
        ],
        'Fixed_Makespan': [
            {'name': 'n10_s5_t120', 'agents': 10, 'species': 5, 'initial_tasks': 30, 'total_tasks': 120, 'recommended_arrival_rate': 2},
            {'name': 'n15_s5_t200', 'agents': 15, 'species': 5, 'initial_tasks': 50, 'total_tasks': 200, 'recommended_arrival_rate': 2},
            {'name': 'n20_s5_t240', 'agents': 20, 'species': 5, 'initial_tasks': 60, 'total_tasks': 240, 'recommended_arrival_rate': 2},
            {'name': 'n30_s5_t300', 'agents': 30, 'species': 5, 'initial_tasks': 80, 'total_tasks': 300, 'recommended_arrival_rate': 2},
        ]
    }
    
    # 数据集基础路径
    base_dir = SA_BT_DATASET_ROOT
    
    # 评估两种数据集类型
    for dataset_type in ['Fixed_Tasks', 'Fixed_Makespan']:
        dataset_msg = f"\n{'#'*80}\n处理数据集: {dataset_type}\n{'#'*80}\n\n"
        print(dataset_msg, end='')
        log_file.write(dataset_msg)
        
        # 创建输出目录
        output_dir = Path(output_base_dir) / dataset_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = dataset_configs[dataset_type]
        
        # 遍历所有配置
        for i, config_info in enumerate(configs, 1):
            config_name = config_info['name']
            arrival_rate = config_info['recommended_arrival_rate']
            max_total_tasks = config_info['total_tasks']
            
            config_msg = f"\n{'='*80}\n"
            config_msg += f"配置 {i}/{len(configs)}: {dataset_type}/{config_name}\n"
            config_msg += f"{'='*80}\n"
            config_msg += f"  智能体: {config_info['agents']}, 种类: {config_info['species']}\n"
            config_msg += f"  初始任务: {config_info['initial_tasks']}, 最大总任务: {max_total_tasks}\n"
            config_msg += f"  到达率: {arrival_rate} 任务/分钟\n"
            
            # 获取环境文件列表
            config_dir = base_dir / dataset_type / config_name
            env_files = sorted(config_dir.glob('env_*.pkl'))
            if not env_files:
                warn_msg = f"⚠ {config_name}: 未找到环境文件\n"
                print(warn_msg, end='')
                log_file.write(warn_msg)
                continue
            
            config_msg += f"  样本数: {len(env_files)}\n\n"
            print(config_msg, end='')
            log_file.write(config_msg)
            
            config_results = []
            
            # 评估每个环境实例
            for env_file in env_files:
                try:
                    # 加载环境
                    with open(env_file, 'rb') as f:
                        env = pickle.load(f)
                    
                    # 确保任务有appear_time和is_dynamic属性
                    for task in env.task_dic.values():
                        if 'appear_time' not in task:
                            task['appear_time'] = 0.0
                        if 'is_dynamic' not in task:
                            task['is_dynamic'] = False
                    
                    initial_task_count = len(env.task_dic)
                    
                    # 根据数据集类型设置不同的仿真参数
                    if dataset_type == 'Fixed_Tasks':
                        # Fixed_Tasks: 固定任务数，给足够的时间完成所有任务
                        sim_time_limit = 10000  # 10000分钟足够完成100个任务
                    else:
                        # Fixed_Makespan: 固定时间限制
                        sim_time_limit = 120
                    
                    # 创建动态规划器，使用默认的贪婪算法（decision_maker=None）
                    planner = DynamicCentralizedPlanner(
                        env=env,
                        max_total_tasks=max_total_tasks,
                        dynamic_task_arrival_rate=arrival_rate,
                        simulation_time_limit=sim_time_limit,
                        random_seed=42,
                        verbose=False,
                        decision_maker=None,  # 使用默认贪婪算法
                        max_waiting_time=30  # 智能体最大等待时间30分钟，超时触发重规划
                    )
                    
                    # 运行仿真
                    results = planner.run()
                    
                    # 计算指标
                    metrics = calculate_metrics(env, results, initial_task_count, dataset_type)
                    metrics['env_file'] = env_file.name
                    metrics['config_name'] = config_name
                    metrics['dataset_type'] = dataset_type
                    metrics['arrival_rate'] = arrival_rate
                    
                    config_results.append(metrics)
                    
                    # 输出单个实例结果
                    if dataset_type == 'Fixed_Tasks':
                        log_msg = (f"  ✓ {env_file.name}: "
                                   f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                                   f"({metrics['success_rate']*100:.1f}%), "
                                   f"makespan={metrics['makespan']:.1f}min, "
                                   f"截止日期满足={metrics.get('deadline_satisfaction_rate',0)*100:.1f}%, "
                                   f"充电={metrics.get('total_charging_times',0):.0f}次\n")
                    else:
                        log_msg = (f"  ✓ {env_file.name}: "
                                   f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                                   f"({metrics['success_rate']*100:.1f}%), "
                                   f"截止日期满足={metrics.get('deadline_satisfaction_rate',0)*100:.1f}%, "
                                   f"充电={metrics.get('total_charging_times',0):.0f}次\n")
                    
                    print(log_msg, end='')
                    log_file.write(log_msg)
                    log_file.flush()
                    
                except Exception as e:
                    error_msg = f"  ✗ {env_file.name}: 失败 - {str(e)}\n"
                    print(error_msg, end='')
                    log_file.write(error_msg)
                    import traceback
                    traceback.print_exc()
            
            # 保存配置结果
            if config_results:
                results_pkl = output_dir / f"{config_name}_results.pkl"
                with open(results_pkl, 'wb') as f:
                    pickle.dump(config_results, f)
                
                results_json = output_dir / f"{config_name}_results.json"
                with open(results_json, 'w', encoding='utf-8') as f:
                    json.dump(config_results, f, indent=2)
                
                summary_msg = build_summary_like_my_model(config_name, config_results)
                print(summary_msg, end='')
                log_file.write(summary_msg)
    
    # 结束
    end_msg = f"\n{'='*80}\n动态评估完成！\n{'='*80}\n"
    end_msg += f"结果保存在: {output_base_dir}/\n"
    end_msg += f"日志文件: {str(log_path)}\n"
    end_msg += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    end_msg += "=" * 80 + "\n"
    
    print(end_msg, end='')
    log_file.write(end_msg)
    log_file.close()


if __name__ == '__main__':
    main()
