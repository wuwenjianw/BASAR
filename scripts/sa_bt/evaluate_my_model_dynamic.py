#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用自己训练的模型评估动态SA-BT场景

python -u scripts/sa_bt/evaluate_my_model_dynamic.py

"""

import pickle
import numpy as np
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv
from dynamic_worker import create_dynamic_model, create_dynamic_planner
from worker import Worker
from parameters import EnvParams, TrainParams, SaverParams
from project_paths import ARTIFACTS_ROOT, SA_BT_DATASET_ROOT, ensure_checkpoint_exists, resolve_model_dir


def load_env(env_path):
    """加载环境，并确保动态评估所需字段齐全（与HRLF评估保持一致）。"""
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    # 新模型：需要完整任务/智能体维度信息
    # - 任务特征包含 deadline
    # - 智能体特征包含 battery（环境内置）
    if hasattr(env, 'use_deadline'):
        env.use_deadline = True

    # 确保所有任务都有 appear_time 和 is_dynamic 属性（动态评估依赖）
    for task in env.task_dic.values():
        task.setdefault('appear_time', 0.0)
        task.setdefault('is_dynamic', False)

    return env


def extract_model_state_dict(checkpoint):
    """
    从不同格式的 checkpoint 中提取模型参数。

    兼容以下几种常见格式：
    - {'model': state_dict, ...}
    - {'best_model': state_dict, ...}
    - {'state_dict': state_dict}
    - {'model_state_dict': state_dict}
    - checkpoint 本身就是 state_dict
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ('model', 'best_model', 'state_dict', 'model_state_dict'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "无法从 checkpoint 中解析模型参数，支持的键包括: "
        "'model', 'best_model', 'state_dict', 'model_state_dict'. "
        f"当前键: {list(checkpoint.keys())}"
    )




def run_model_dynamic(env, global_network, device, max_total_tasks=50, arrival_rate=0.3,
                      simulation_time_limit=200, random_seed=42, sampling=False,
                      decision_log_path=None, debug_verbose=False,
                      dynamic_task_options=None):
    """
    使用自定义模型在动态环境下运行仿真（对齐 HRLF 的评估逻辑）

    对齐点（与 evaluate_hrlf_dynamic 的 run_hrlf_dynamic 保持一致）：
    - 动态任务到达与时间推进逻辑：优先处理到达事件，避免时间跨越导致任务晚出现
    - 决策循环：使用 env.next_decision() + env.agent_step() 驱动环境状态演化（不手动改 env 内部状态）
    - 决策冻结规则：对“正在等待/执行/在途中”的智能体不重规划（protect_traveling）
    - 动作掩码：额外屏蔽未来任务（appear_time>current_time）与电量不可达任务（can_reach_with_battery=False）
    - 充电站到达处理：字段更新与 HRLF 评估一致
    """
    # 随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 新模型需要完整维度：deadline + battery
    if hasattr(env, 'use_deadline'):
        env.use_deadline = True

    # 初始化环境状态
    env.init_state()

    # Worker
    worker = Worker(0, global_network, global_network, 0, device)
    worker.env = env

    # 动态规划器：任务到达逻辑与贪婪/HRLF评估保持一致
    planner = create_dynamic_planner(
        env=env,
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
        decision_maker=None,
        verbose=False,
        dynamic_task_options=dynamic_task_options,
    )

    total_inference_time = 0.0
    inference_count = 0
    decision_step = 0
    last_decision_time = {}
    decision_history = []
    deadlock_recovery_times = set()
    rescue_replan_times = set()
    termination_reason = 'running'
    rescue_replan_count = 0
    stagnation_window = 200.0
    max_main_iterations = 100000
    main_iteration = 0

    def _finished_task_count():
        return sum(1 for task in env.task_dic.values() if task.get('finished', False))

    last_finished_count = _finished_task_count()
    last_progress_time = float(env.current_time)

    decision_log_file = None
    if decision_log_path is not None:
        decision_log_file = open(decision_log_path, 'w', encoding='utf-8')
        decision_log_file.write("time,agent_id,selected_task,available_tasks\n")

    def _write_decision_log(agent_id, selected_task, available_tasks):
        if decision_log_file is None:
            return
        decision_log_file.write(
            f"{env.current_time:.3f},{agent_id},{selected_task},{available_tasks}\n"
        )

    def _record_decision(agent_id, selected_task, available_tasks):
        decision_history.append({
            'time': float(env.current_time),
            'agent_id': int(agent_id),
            'task_id': int(selected_task),
            'available_tasks': int(available_tasks),
            'battery': float(env.agent_dic[agent_id].get('battery', 0.0))
        })

    def _dispatch_to_charge(target_agent_id):
        arrival_time = planner._dispatch_to_charge(target_agent_id, plan_time=env.current_time)
        if arrival_time is None:
            return
        _record_decision(target_agent_id, -999, 0)
        _write_decision_log(target_agent_id, -999, 0)

    def _can_reach_contributable_task(target_agent_id):
        contributable_mask = env.get_contributable_task_mask(target_agent_id)
        for task_id, task in env.task_dic.items():
            if task.get('finished', False) or task.get('feasible_assignment', False):
                continue
            if task_id >= len(contributable_mask):
                continue
            if contributable_mask[task_id]:
                continue
            if env.can_reach_with_battery(target_agent_id, task):
                return True
        return False

    def _peek_next_decision_time():
        decision_time = np.array(env.get_matrix(env.agent_dic, 'next_decision'), dtype=float)
        if decision_time.size == 0:
            return float('inf')
        if np.all(np.isnan(decision_time)):
            arrival = [
                max(a) if a else env.current_time
                for a in env.get_matrix(env.agent_dic, 'arrival_time')
            ]
            if not arrival:
                return float('inf')
            return max(max(arrival), env.current_time)
        no_choice = env.get_matrix(env.agent_dic, 'no_choice')
        decision_time = np.where(no_choice, np.inf, decision_time)
        next_decision = np.nanmin(decision_time)
        if np.isinf(next_decision):
            arrival_time = np.array([
                agent['arrival_time'][-1] if agent['arrival_time'] else env.current_time
                for agent in env.agent_dic.values()
            ])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
        if np.isnan(next_decision):
            return float('inf')
        return max(float(next_decision), env.current_time)

    def _apply_extra_mask(agent_id, mask_np):
        """对齐 HRLF wrapper：屏蔽未来任务 + 电量不可达任务。"""
        if mask_np is None or mask_np.ndim != 2:
            return mask_np
        current_time = getattr(env, 'current_time', 0.0)
        for idx, task in enumerate(env.task_dic.values()):
            if task.get('appear_time', 0.0) > current_time:
                mask_np[0, idx + 1] = True
            elif not env.can_reach_with_battery(agent_id, task):
                mask_np[0, idx + 1] = True
        return mask_np

    def _sync_agent_motion_state(target_time):
        env.update_all_batteries(target_time)
        for agent in env.agent_dic.values():
            if agent['arrival_time'] and target_time >= agent['arrival_time'][-1]:
                agent['is_moving'] = False

    def _advance_time(target_time):
        env.current_time = max(env.current_time, target_time)
        _sync_agent_motion_state(env.current_time)

    def _generate_ready_dynamic_tasks():
        while (planner.next_task_arrival_time is not None and
               env.current_time >= planner.next_task_arrival_time and
               len(env.task_dic) < max_total_tasks):
            planner._generate_dynamic_task()

    def _freeze_agent(agent_id, selected_task=-1, available_tasks=0):
        agent = env.agent_dic[agent_id]
        agent['no_choice'] = True
        agent['next_decision'] = float('inf')
        agent['is_moving'] = False
        _write_decision_log(agent_id, selected_task, available_tasks)
        _record_decision(agent_id, selected_task, available_tasks)

    def _all_tasks_completed():
        if len(env.task_dic) == 0:
            return True
        return all(task.get('finished', False) for task in env.task_dic.values())

    def _future_arrivals_pending():
        return len(env.task_dic) < max_total_tasks and planner.next_task_arrival_time is not None

    def _episode_complete():
        return _all_tasks_completed() and not _future_arrivals_pending()

    def _reactivate_idle_agents():
        """
        死锁恢复：重新激活空闲且被冻结的智能体，允许其再次决策。
        """
        reactivated = 0
        for agent in env.agent_dic.values():
            if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                agent['no_choice'] = False
                agent['next_decision'] = env.current_time
                reactivated += 1
        return reactivated

    def _rescue_with_planner_replan():
        """
        使用集中规划器做一次强制重规划，作为死锁/长期停滞恢复手段。
        """
        nonlocal rescue_replan_count
        if rescue_replan_count >= 20:
            return False
        if env.current_time in rescue_replan_times:
            return False

        prev_active_events = sum(
            1 for agent in env.agent_dic.values()
            if agent.get('next_decision', float('inf')) < float('inf')
        )
        prev_assignments = tuple(
            (aid, agent.get('current_task'), float(agent.get('next_decision', float('inf'))))
            for aid, agent in env.agent_dic.items()
        )

        try:
            planner._plan(replan=True, force_waiting=True)
            rescue_replan_count += 1
            rescue_replan_times.add(env.current_time)
        except Exception:
            return False

        # 重规划后，确保有计划事件的智能体能够被正常释放
        for agent in env.agent_dic.values():
            if agent.get('next_decision', float('inf')) < float('inf'):
                agent['no_choice'] = False

        next_decision_time_after_replan = _peek_next_decision_time()
        next_active_events = sum(
            1 for agent in env.agent_dic.values()
            if agent.get('next_decision', float('inf')) < float('inf')
        )
        next_assignments = tuple(
            (aid, agent.get('current_task'), float(agent.get('next_decision', float('inf'))))
            for aid, agent in env.agent_dic.items()
        )
        has_new_schedule = next_active_events > 0 and np.isfinite(next_decision_time_after_replan)
        has_state_change = next_assignments != prev_assignments or next_active_events != prev_active_events
        return has_new_schedule and has_state_change

    def _handle_charging_station_arrivals():
        for agent_id, agent in env.agent_dic.items():
            if agent.get('current_task') != -999:
                continue
            if agent['next_decision'] > env.current_time + 1e-6:
                continue

            old_location = agent['location'].copy()
            agent['location'] = agent['charging_station'].copy()
            distance_traveled = np.linalg.norm(old_location - agent['charging_station'])
            agent['travel_dist'] = agent.get('travel_dist', 0) + distance_traveled
            agent['battery'] = env.initial_battery
            agent['total_charging_times'] += 1
            agent['is_moving'] = False
            agent['is_charging'] = False
            agent['current_task'] = -agent['species'] - 1
            agent['assigned'] = False
            agent['next_decision'] = float('inf')

    while not env.finished and env.current_time < simulation_time_limit:
        main_iteration += 1
        if main_iteration > max_main_iterations:
            termination_reason = 'iteration_limit'
            env.current_time = simulation_time_limit
            break

        if _episode_complete():
            env.finished = True
            termination_reason = 'all_tasks_completed'
            break

        # 已达到任务上限时，停止后续到达事件，避免空转
        if len(env.task_dic) >= max_total_tasks:
            planner.next_task_arrival_time = None

        # 先处理任务到达事件，避免时间跨越导致任务晚出现
        next_decision_time = _peek_next_decision_time()
        next_arrival_time = planner.next_task_arrival_time
        if next_arrival_time is not None and next_arrival_time <= next_decision_time:
            _advance_time(next_arrival_time)
            _generate_ready_dynamic_tasks()
            if len(env.task_dic) >= max_total_tasks:
                planner.next_task_arrival_time = None
            for agent in env.agent_dic.values():
                # 新任务到达后允许所有空闲智能体重新决策
                agent['no_choice'] = False
                if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                    agent['next_decision'] = env.current_time
            continue

        if next_decision_time == float('inf') and next_arrival_time is None:
            if _episode_complete():
                env.finished = True
                termination_reason = 'all_tasks_completed'
                break

            # 优先尝试一次“强制重规划恢复”
            if _rescue_with_planner_replan():
                if debug_verbose:
                    print(f"[恢复] time={env.current_time:.3f}, 强制重规划恢复成功")
                continue

            # 有未完成任务但无事件可推进：先尝试一次死锁恢复，再不行按超时处理
            if env.current_time not in deadlock_recovery_times:
                reactivated = _reactivate_idle_agents()
                deadlock_recovery_times.add(env.current_time)
                if reactivated > 0:
                    if debug_verbose:
                        print(f"[恢复] time={env.current_time:.3f}, 重新激活空闲智能体: {reactivated}")
                    continue

            termination_reason = 'deadlock_no_events'
            env.current_time = simulation_time_limit
            break

        # 获取下一个决策时刻
        release_agents, current_time = env.next_decision()
        env.current_time = current_time

        # 若当前没有可决策智能体，且下个任务到达在未来，直接跳到到达时刻
        if not release_agents[0] and not release_agents[1]:
            if next_arrival_time is None:
                if _episode_complete():
                    env.finished = True
                    termination_reason = 'all_tasks_completed'
                    break

                if _rescue_with_planner_replan():
                    if debug_verbose:
                        print(f"[恢复] time={env.current_time:.3f}, 空释放下强制重规划成功")
                    continue

                if env.current_time not in deadlock_recovery_times:
                    reactivated = _reactivate_idle_agents()
                    deadlock_recovery_times.add(env.current_time)
                    if reactivated > 0:
                        if debug_verbose:
                            print(f"[恢复] time={env.current_time:.3f}, 空释放后重激活: {reactivated}")
                        continue

                termination_reason = 'deadlock_empty_release'
                env.current_time = simulation_time_limit
                break
            if next_arrival_time > env.current_time + 1e-9:
                _advance_time(next_arrival_time)
                continue

        # 处理到达充电站的智能体（对齐 HRLF）
        _handle_charging_station_arrivals()

        # 生成当前时间前到达的动态任务（使用规划器内部逻辑）
        _generate_ready_dynamic_tasks()

        # 打乱智能体顺序
        np.random.shuffle(release_agents[0])

        processed_agents = set()
        while release_agents[0] or release_agents[1]:
            agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
            if agent_id in processed_agents:
                continue
            processed_agents.add(agent_id)

            agent = env.agent_dic[agent_id]
            agent['no_choice'] = False

            # 若同一时刻已为该智能体做过决策，强制推迟到下一个最小时间步
            if last_decision_time.get(agent_id) == env.current_time:
                agent['next_decision'] = env.current_time + getattr(env, 'dt', 1e-3)
                continue

            # 已被分配且仍在等待/执行的智能体不允许重规划
            if agent.get('current_task', -1) >= 0:
                current_task = env.task_dic.get(agent['current_task'])
                if current_task and agent_id in current_task.get('members', []):
                    if current_task.get('feasible_assignment', False):
                        if env.current_time < float(current_task.get('time_finish', env.current_time)):
                            agent['no_choice'] = True
                            agent['next_decision'] = float(current_task.get('time_finish', env.current_time))
                            continue
                    else:
                        arrival_time = env.get_arrival_time(agent_id, current_task['ID'])
                        wait_deadline = arrival_time + float(getattr(env, 'max_waiting_time', 0.0))
                        if env.current_time < wait_deadline:
                            agent['no_choice'] = True
                            agent['next_decision'] = wait_deadline
                            continue

            # 如果智能体仍在途中，不应再次决策
            if agent['current_task'] >= 0 and agent.get('arrival_time'):
                last_arrival = agent['arrival_time'][-1]
                if last_arrival > env.current_time + 1e-9:
                    agent['next_decision'] = last_arrival
                    continue

            # 电量过低时直接派往充电站
            if env.check_battery_critical(agent_id):
                _dispatch_to_charge(agent_id)
                continue

            # 获取观察，并对掩码做 HRLF 风格额外处理
            tasks_info_np, agents_info_np, mask_np = env.agent_observe(agent_id, False)
            mask_np = _apply_extra_mask(agent_id, mask_np)
            task_info, total_agents, mask = worker.convert_torch((tasks_info_np, agents_info_np, mask_np))
            mask_bool = mask > 0.5
            all_tasks_assigned = env.all_tasks_feasibly_assigned()

            # 检查是否所有任务都被阻塞
            block_flag = mask_bool[0, 1:].all().item()
            has_contributable = not env.get_contributable_task_mask(agent_id).all()
            if block_flag and not all_tasks_assigned:
                can_reach_task = _can_reach_contributable_task(agent_id)
                if has_contributable and not can_reach_task:
                    # 无法到达可贡献任务时，优先让智能体去充电站“复位位置+满电”再尝试
                    # 仅当已在充电站且满电时，判定为暂时不可恢复并冻结
                    at_station = np.linalg.norm(agent['location'] - agent['charging_station']) <= 1e-6
                    full_battery = agent.get('battery', 0.0) >= env.initial_battery - 1e-6
                    if at_station and full_battery:
                        _freeze_agent(agent_id, -1, 0)
                        continue
                    _dispatch_to_charge(agent_id)
                    continue

                # 当前无可选任务，冻结，避免同一时刻反复释放该智能体
                _freeze_agent(agent_id, -1, 0)
                continue
            elif block_flag and all_tasks_assigned and agent['current_task'] < 0:
                _freeze_agent(agent_id, -1, 0)
                continue

            # 若存在可选任务，禁止返回基地动作
            if not mask_bool[0, 1:].all().item():
                mask_bool[0, 0] = True
            mask = mask_bool.float()

            # 网络前向
            index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(device)
            start_time = time.time()
            with torch.no_grad():
                probs, _ = global_network(task_info, total_agents, mask, index)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            inference_count += 1

            # 动作选择
            if sampling:
                action = Categorical(probs).sample()
            else:
                action = torch.argmax(probs, dim=1)

            selected_task = action.item() - 1
            available_tasks = int((~mask_bool[0, 1:]).sum().item())
            _write_decision_log(agent_id, selected_task, available_tasks)

            # 执行动作：严格走 env.agent_step（对齐 HRLF）
            _, doable, _ = env.agent_step(agent_id, action.item(), decision_step)
            if not doable:
                agent['no_choice'] = True
                agent['next_decision'] = float('inf')
                agent['is_moving'] = False
            elif selected_task == -1 and mask_bool[0, 1:].all().item():
                agent['no_choice'] = True
                agent['next_decision'] = float('inf')
                agent['is_moving'] = False
            else:
                # 防止任务耗时为0导致同一时刻反复决策
                min_step = getattr(env, 'dt', 1e-3)
                if agent['next_decision'] <= env.current_time + 1e-9:
                    agent['next_decision'] = env.current_time + min_step

            _record_decision(agent_id, selected_task, available_tasks)
            last_decision_time[agent_id] = env.current_time

        env.finished = env.check_finished()
        if env.finished and _future_arrivals_pending():
            env.finished = False

        if _episode_complete():
            env.finished = True
            termination_reason = 'all_tasks_completed'
        elif env.finished:
            termination_reason = 'all_tasks_finished'

        # 长时间无任务完成时，触发一次强制重规划，避免在局部死循环中拖到time_limit
        current_finished_count = _finished_task_count()
        if current_finished_count > last_finished_count:
            last_finished_count = current_finished_count
            last_progress_time = float(env.current_time)
        elif (not env.finished and
              env.current_time - last_progress_time >= stagnation_window and
              current_finished_count < len(env.task_dic)):
            if _rescue_with_planner_replan():
                if debug_verbose:
                    print(f"[恢复] time={env.current_time:.3f}, 长期停滞触发强制重规划")
                last_progress_time = float(env.current_time)

        decision_step += 1

    if termination_reason == 'running':
        if _episode_complete():
            termination_reason = 'all_tasks_completed'
        elif env.finished:
            termination_reason = 'all_tasks_finished'
        elif env.current_time >= simulation_time_limit:
            termination_reason = 'time_limit'
        else:
            termination_reason = 'loop_exit'

    # 退出主循环后补一次任务状态结算，避免“应完成但未刷 finished”的尾部状态残留。
    env.task_update()
    stats = env.get_task_statistics()
    raw_end_time = float(env.current_time)
    capped_end_time = min(raw_end_time, float(simulation_time_limit))
    effective_makespan = capped_end_time
    if stats['finished_tasks'] < stats['total_tasks']:
        effective_makespan = float(simulation_time_limit)

    if decision_log_file is not None:
        decision_log_file.close()

    return {
        'dynamic_tasks_generated': getattr(planner, 'dynamic_tasks_generated', 0),
        'total_inference_time': total_inference_time,
        'avg_inference_time': total_inference_time / inference_count if inference_count > 0 else 0,
        'inference_count': inference_count,
        'decision_history': decision_history,
        'termination_reason': termination_reason,
        'simulation_time_limit': simulation_time_limit,
        'raw_end_time': raw_end_time,
        'effective_makespan': effective_makespan,
        'rescue_replan_count': rescue_replan_count,
    }


def calculate_metrics(env, results, initial_task_count, dataset_type='Fixed_Tasks'):
    """
    计算评估指标（用于动态评估）- 对齐 HRLF/贪婪评估的逻辑

    关键对齐点：
    - Deadline Satisfaction Rate：仅基于“已完成任务”统计（避免未完成任务影响）
    - Waiting Time：仅统计已完成任务，并过滤 NaN/inf
    - dynamic_tasks：使用 env.get_task_statistics() 的统计（基于 is_dynamic）
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
    metrics['total_planning_time'] = results.get('total_inference_time', 0)
    metrics['avg_planning_time'] = results.get('avg_inference_time', 0)
    metrics['initial_planning_time'] = 0
    metrics['avg_replan_time'] = 0

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
    metrics['replan_count'] = 0
    metrics['termination_reason'] = results.get('termination_reason', 'unknown')

    return metrics


def evaluate_single_env(env_path, global_network, device, config_info, arrival_rate, 
                       max_total_tasks, dataset_type, log_file, sampling=False):
    try:
        env = load_env(env_path)
        initial_task_count = len(env.task_dic)
        
        if dataset_type == 'Fixed_Tasks':
            simulation_time_limit = 10000
        else:
            simulation_time_limit = 120

        results = run_model_dynamic(
            env=env,
            global_network=global_network,
            device=device,
            max_total_tasks=max_total_tasks,
            arrival_rate=arrival_rate,
            simulation_time_limit=simulation_time_limit,
            random_seed=42,
            sampling=sampling
        )
        
        metrics = calculate_metrics(env, results, initial_task_count, dataset_type)
        unfinished = metrics['total_tasks'] - metrics['finished_tasks']
        reason_suffix = ""
        if unfinished > 0:
            reason_suffix = f", 终止原因={metrics.get('termination_reason', 'unknown')}"
        rescue_suffix = f", 恢复重规划={int(results.get('rescue_replan_count', 0))}次"
        
        if dataset_type == 'Fixed_Tasks':
            log_msg = (f"  ✓ {os.path.basename(env_path)}: "
                       f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                       f"({metrics['success_rate']*100:.1f}%), "
                       f"makespan={metrics['makespan']:.1f}min, "
                       f"截止日期满足={metrics['deadline_satisfaction_rate']*100:.1f}%, "
                       f"充电={metrics['total_charging_times']:.0f}次"
                       f"{rescue_suffix}"
                       f"{reason_suffix}\n")
        else:
            log_msg = (f"  ✓ {os.path.basename(env_path)}: "
                       f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                       f"({metrics['success_rate']*100:.1f}%), "
                       f"截止日期满足={metrics['deadline_satisfaction_rate']*100:.1f}%, "
                       f"充电={metrics['total_charging_times']:.0f}次"
                       f"{rescue_suffix}"
                       f"{reason_suffix}\n")
        
        log_file.write(log_msg)
        log_file.flush()
        print(log_msg, end='')
        
        return metrics
        
    except Exception as e:
        import traceback
        error_msg = f"  ✗ {os.path.basename(env_path)}: 失败 - {str(e)}\n"
        traceback.print_exc()
        log_file.write(error_msg)
        log_file.flush()
        print(error_msg, end='')
        return None


def main(protocol_filter=None):
    print("=" * 80)
    print("自定义模型动态场景评估")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    use_gpu_global = TrainParams.USE_GPU_GLOBAL
    folder_name = SaverParams.FOLDER_NAME
    model_path = resolve_model_dir(folder_name)
    
    def infer_input_dims():
        env = TaskEnv(
            EnvParams.SPECIES_AGENTS_RANGE,
            EnvParams.SPECIES_RANGE,
            EnvParams.TASKS_RANGE,
            EnvParams.TRAIT_DIM,
            EnvParams.DECISION_DIM,
        )
        agent_id = list(env.agent_dic.keys())[0]
        tasks_info, agents_info, _ = env.agent_observe(agent_id, False)
        return int(agents_info.shape[-1]), int(tasks_info.shape[-1])
    
    agent_input_dim, task_input_dim = infer_input_dims()
    
    print(f"模型配置:")
    print(f"  MODEL_NAME: {TrainParams.MODEL_NAME}")
    print(f"  AGENT_INPUT_DIM: {agent_input_dim}")
    print(f"  TASK_INPUT_DIM: {task_input_dim}")
    print(f"  EMBEDDING_DIM: {TrainParams.EMBEDDING_DIM}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu_global else 'cpu')
    print(f"使用设备: {device}\n")
    
    print("加载模型...")
    global_network = create_dynamic_model(
        agent_input_dim=agent_input_dim,
        task_input_dim=task_input_dim,
        embedding_dim=TrainParams.EMBEDDING_DIM,
        device=device,
    )
    checkpoint_path = ensure_checkpoint_exists(folder_name, method_label=TrainParams.MODEL_NAME)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    global_network.load_state_dict(state_dict, strict=False)
    global_network.eval()
    print(f"✓ 成功加载模型: {checkpoint_path}\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = ARTIFACTS_ROOT / 'results' / f'{folder_name.lower()}_dynamic'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = output_base_dir / f'evaluation_log_{timestamp}.txt'
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    log_file.write(f"自定义模型动态场景评估日志\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"模型路径: {model_path}\n")
    log_file.write("=" * 80 + "\n\n")
    
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
    
    protocol_order = ['Fixed_Tasks', 'Fixed_Makespan']
    if protocol_filter is not None:
        selected = set(protocol_filter)
        unknown = selected.difference(dataset_configs)
        if unknown:
            raise ValueError(f"Unknown SA-BT protocol(s): {sorted(unknown)}")
        protocol_order = [protocol for protocol in protocol_order if protocol in selected]

    for dataset_type in protocol_order:
        msg = f"\n{'#'*80}\n处理数据集: {dataset_type}\n{'#'*80}\n"
        log_file.write(msg)
        print(msg, end='')
        
        configs = dataset_configs[dataset_type]
        
        for config_idx, config in enumerate(configs, 1):
            config_name = config['name']
            arrival_rate = config['recommended_arrival_rate']
            max_total_tasks = config['total_tasks']
            
            input_dir = SA_BT_DATASET_ROOT / dataset_type / config_name
            output_dir = output_base_dir / dataset_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            env_files = sorted(input_dir.glob('env_*.pkl'))
            
            if not env_files:
                msg = f"⚠ {config_name}: 未找到环境文件\n"
                log_file.write(msg)
                print(msg, end='')
                continue
            
            config_msg = f"\n{'='*80}\n配置 {config_idx}/{len(configs)}: {dataset_type}/{config_name}\n{'='*80}\n"
            config_msg += f"  智能体: {config['agents']}, 种类: {config['species']}\n"
            config_msg += f"  初始任务: {config['initial_tasks']}, 最大总任务: {max_total_tasks}\n"
            config_msg += f"  到达率: {arrival_rate} 任务/分钟\n"
            config_msg += f"  样本数: {len(env_files)}\n\n"
            log_file.write(config_msg)
            print(config_msg, end='')
            
            config_results = []
            
            for env_file in env_files:
                metrics = evaluate_single_env(
                    env_path=env_file,
                    global_network=global_network,
                    device=device,
                    config_info=config,
                    arrival_rate=arrival_rate,
                    max_total_tasks=max_total_tasks,
                    dataset_type=dataset_type,
                    log_file=log_file,
                    sampling=False
                )
                
                if metrics:
                    metrics['env_file'] = env_file.name
                    metrics['config_name'] = config_name
                    metrics['dataset_type'] = dataset_type
                    metrics['arrival_rate'] = arrival_rate
                    config_results.append(metrics)
            
            if config_results:
                results_pkl = output_dir / f"{config_name}_results.pkl"
                with open(results_pkl, 'wb') as f:
                    pickle.dump(config_results, f)
                
                results_json = output_dir / f"{config_name}_results.json"
                with open(results_json, 'w') as f:
                    json.dump(config_results, f, indent=2)
                
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
                
                log_file.write(summary_msg)
                print(summary_msg, end='')
    
    end_msg = f"\n{'='*80}\n评估完成！\n{'='*80}\n"
    end_msg += f"结果保存在: {output_base_dir}/\n"
    end_msg += f"日志文件: {log_file_path}\n"
    end_msg += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    end_msg += "=" * 80 + "\n"
    
    log_file.write(end_msg)
    print(end_msg)
    
    log_file.close()


if __name__ == '__main__':
    main()
