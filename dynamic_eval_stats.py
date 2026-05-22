#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态评估统计辅助函数。

这里的指标计算与配置级汇总逻辑直接对齐 evaluate_my_model_dynamic.py，
供 baseline 评估脚本复用，避免统计口径漂移。
"""

import numpy as np
import pandas as pd


def calculate_metrics_like_my_model(env, results, initial_task_count, dataset_type='Fixed_Tasks'):
    """
    计算评估指标（用于动态评估）- 对齐主模型 evaluate_my_model_dynamic.py 的逻辑

    关键对齐点：
    - Deadline Satisfaction Rate：仅基于“已完成任务”统计（避免未完成任务影响）
    - Waiting Time：仅统计已完成任务，并过滤 NaN/inf
    - dynamic_tasks：使用 env.get_task_statistics() 的统计（基于 is_dynamic）
    - initial_planning_time / avg_replan_time / replan_count 保持主模型当前语义
    """
    metrics = {}

    stats = env.get_task_statistics()
    total_tasks = stats['total_tasks']
    finished_tasks = stats['finished_tasks']
    dynamic_tasks = stats['dynamic_tasks']

    # (1) Success Rate
    metrics['success_rate'] = finished_tasks / total_tasks if total_tasks > 0 else 0

    finished_task_list = [t for t in env.task_dic.values() if t['finished']]

    # (2) Deadline Satisfaction Rate（只基于完成任务）
    on_time_count = sum(1 for t in finished_task_list if t['time_finish'] <= t['deadline'])
    metrics['deadline_satisfaction_rate'] = on_time_count / finished_tasks if finished_tasks > 0 else 0

    # (3) Deadline Violation
    violation_count = 0
    total_violation_time = 0.0
    max_violation = 0.0
    for task in finished_task_list:
        if task['time_finish'] > task['deadline']:
            violation_count += 1
            violation = task['time_finish'] - task['deadline']
            total_violation_time += violation
            max_violation = max(max_violation, violation)

    metrics['deadline_violation_count'] = violation_count
    metrics['deadline_violation_rate'] = violation_count / finished_tasks if finished_tasks > 0 else 0
    metrics['avg_deadline_violation'] = total_violation_time / finished_tasks if finished_tasks > 0 else 0
    metrics['avg_deadline_violation_scope'] = 'completed_tasks'
    metrics['max_deadline_violation'] = max_violation

    # (4) Makespan
    # 统一语义：若存在未完成任务，则按仿真时间上限记为超时
    effective_makespan = float(results.get('effective_makespan', env.current_time))
    metrics['makespan'] = effective_makespan

    # (5) Waiting Time（只统计已完成任务，过滤异常值）
    env.calculate_waiting_time()
    total_waiting_time = 0.0
    for task in finished_task_list:
        task_waiting = float(task.get('sum_waiting_time', 0.0))
        if not np.isfinite(task_waiting):
            task_waiting = 0.0
        total_waiting_time += task_waiting
    metrics['waiting_time'] = total_waiting_time

    # (6) Travel Distance
    total_distance = sum(agent.get('travel_dist', 0) for agent in env.agent_dic.values())
    metrics['total_travel_distance'] = total_distance
    metrics['avg_travel_distance'] = total_distance / env.agents_num if env.agents_num > 0 else 0

    # (7) Flow Time（出现到完成）
    flow_times = []
    for task in finished_task_list:
        appear_time = task.get('appear_time', 0.0)
        flow_times.append(task['time_finish'] - appear_time)

    if flow_times:
        metrics['avg_flow_time'] = float(np.mean(flow_times))
        metrics['max_flow_time'] = float(np.max(flow_times))
        metrics['min_flow_time'] = float(np.min(flow_times))
    else:
        metrics['avg_flow_time'] = 0.0
        metrics['max_flow_time'] = 0.0
        metrics['min_flow_time'] = 0.0

    # (8) Planning CPU Time / Inference Time
    # 兼容两类结果命名：
    # 1. 学习方法使用 total_inference_time / avg_inference_time
    # 2. 贪婪/集中式规划器使用 total_planning_time / avg_planning_time
    metrics['total_planning_time'] = results.get(
        'total_inference_time',
        results.get('total_planning_time', 0),
    )
    metrics['avg_planning_time'] = results.get(
        'avg_inference_time',
        results.get('avg_planning_time', 0),
    )
    metrics['initial_planning_time'] = results.get('initial_planning_time', 0)
    metrics['avg_replan_time'] = results.get('avg_replan_time', 0)

    # (9) Charging Times
    total_charging = sum(agent.get('total_charging_times', 0) for agent in env.agent_dic.values())
    agents_with_charging = sum(1 for agent in env.agent_dic.values() if agent.get('total_charging_times', 0) > 0)

    metrics['total_charging_times'] = total_charging
    metrics['avg_charging_times_per_agent'] = total_charging / env.agents_num if env.agents_num > 0 else 0
    metrics['agents_with_charging'] = agents_with_charging

    # Basic stats
    metrics['total_tasks'] = total_tasks
    metrics['initial_tasks'] = initial_task_count
    metrics['dynamic_tasks'] = dynamic_tasks
    metrics['finished_tasks'] = finished_tasks
    metrics['simulation_time'] = effective_makespan
    metrics['replan_count'] = results.get('replan_count', 0)
    metrics['termination_reason'] = results.get('termination_reason', 'unknown')

    return metrics


def build_summary_like_my_model(config_name, config_results):
    """生成配置级汇总字符串，直接对齐 evaluate_my_model_dynamic.py。"""
    df = pd.DataFrame(config_results)
    complete_mask = df['finished_tasks'] == df['total_tasks']
    complete_sample_count = int(complete_mask.sum())
    total_sample_count = len(df)
    metric_df = df
    metric_scope = "全部样本"
    if complete_sample_count < total_sample_count and complete_sample_count > 0:
        metric_df = df[complete_mask].copy()
        metric_scope = f"完整完成样本 {complete_sample_count}/{total_sample_count}"
    elif complete_sample_count == 0:
        metric_scope = "全部样本（无完整完成样本）"

    summary_msg = f"\n配置 {config_name} 汇总统计（平均值 ± 标准差）:\n"
    summary_msg += f"基础统计:\n"
    summary_msg += f"  总任务数: {df['total_tasks'].mean():.0f}\n"
    summary_msg += f"  初始任务: {df['initial_tasks'].mean():.0f}\n"
    summary_msg += f"  动态任务: {df['dynamic_tasks'].mean():.0f}\n"
    summary_msg += f"  完成任务: {df['finished_tasks'].mean():.1f} ± {df['finished_tasks'].std():.1f}\n"
    summary_msg += f"  仿真时间: {df['simulation_time'].mean():.1f} ± {df['simulation_time'].std():.1f} 分钟\n"
    summary_msg += f"  重规划次数: {df['replan_count'].mean():.1f} ± {df['replan_count'].std():.1f}\n\n"
    summary_msg += f"  完整完成样本: {complete_sample_count}/{total_sample_count}\n"
    summary_msg += f"  指标(2)-(9)统计口径: {metric_scope}\n\n"

    summary_msg += f"(1) 成功率 (Success Rate):\n"
    summary_msg += f"  {df['success_rate'].mean()*100:.2f}% ± {df['success_rate'].std()*100:.2f}% ({df['finished_tasks'].mean():.0f}/{df['total_tasks'].mean():.0f})\n\n"

    summary_msg += f"(2) 截止日期满足率 (Deadline Satisfaction Rate):\n"
    summary_msg += f"  {metric_df['deadline_satisfaction_rate'].mean()*100:.2f}% ± {metric_df['deadline_satisfaction_rate'].std()*100:.2f}% (基于已完成任务)\n\n"

    summary_msg += f"(3) Deadline Violation:\n"
    summary_msg += f"  违反次数: {metric_df['deadline_violation_count'].mean():.1f} ± {metric_df['deadline_violation_count'].std():.1f}\n"
    summary_msg += f"  违反比例: {metric_df['deadline_violation_rate'].mean()*100:.2f}% ± {metric_df['deadline_violation_rate'].std()*100:.2f}%\n"
    summary_msg += f"  平均违反时长: {metric_df['avg_deadline_violation'].mean():.2f} ± {metric_df['avg_deadline_violation'].std():.2f} 分钟\n"
    summary_msg += f"  最大违反时长: {metric_df['max_deadline_violation'].mean():.2f} ± {metric_df['max_deadline_violation'].std():.2f} 分钟\n\n"

    summary_msg += f"(4) Makespan:\n"
    summary_msg += f"  {metric_df['makespan'].mean():.2f} ± {metric_df['makespan'].std():.2f} 分钟\n\n"

    summary_msg += f"(5) Waiting Time:\n"
    summary_msg += f"  总等待时间: {metric_df['waiting_time'].mean():.2f} ± {metric_df['waiting_time'].std():.2f} 分钟\n\n"

    summary_msg += f"(6) Travel Distance:\n"
    summary_msg += f"  总行驶距离: {metric_df['total_travel_distance'].mean():.2f} ± {metric_df['total_travel_distance'].std():.2f}\n"
    summary_msg += f"  平均每智能体: {metric_df['avg_travel_distance'].mean():.2f} ± {metric_df['avg_travel_distance'].std():.2f}\n\n"

    summary_msg += f"(7) Flow Time (任务从出现到完成的时间):\n"
    summary_msg += f"  平均: {metric_df['avg_flow_time'].mean():.2f} ± {metric_df['avg_flow_time'].std():.2f} 分钟\n"
    summary_msg += f"  最大: {metric_df['max_flow_time'].mean():.2f} ± {metric_df['max_flow_time'].std():.2f} 分钟\n"
    summary_msg += f"  最小: {metric_df['min_flow_time'].mean():.2f} ± {metric_df['min_flow_time'].std():.2f} 分钟\n\n"

    summary_msg += f"(8) Planning CPU Time (规划算法CPU耗时):\n"
    summary_msg += f"  总规划时间: {metric_df['total_planning_time'].mean():.4f} ± {metric_df['total_planning_time'].std():.4f} 秒\n"
    summary_msg += f"  初始规划时间: {metric_df['initial_planning_time'].mean():.4f} ± {metric_df['initial_planning_time'].std():.4f} 秒\n"
    summary_msg += f"  平均每次规划: {metric_df['avg_planning_time'].mean():.4f} ± {metric_df['avg_planning_time'].std():.4f} 秒\n"
    summary_msg += f"  平均每次重规划: {metric_df['avg_replan_time'].mean():.4f} ± {metric_df['avg_replan_time'].std():.4f} 秒\n\n"

    summary_msg += f"(9) Charging Times (充电次数统计):\n"
    summary_msg += f"  总充电次数: {metric_df['total_charging_times'].mean():.2f} ± {metric_df['total_charging_times'].std():.2f}\n"
    summary_msg += f"  平均每智能体: {metric_df['avg_charging_times_per_agent'].mean():.2f} ± {metric_df['avg_charging_times_per_agent'].std():.2f} 次\n"
    summary_msg += f"  充电过的智能体数: {metric_df['agents_with_charging'].mean():.1f} ± {metric_df['agents_with_charging'].std():.1f}\n\n"
    return summary_msg
