#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 CTAS-D 基线评估动态 SA-BT 场景（独立文件）

对齐 evaluate_my_model_dynamic.py：
- 事件驱动推进逻辑一致
- 终止条件与恢复机制一致
- 指标统计与日志格式一致
- 唯一区别：动作由 CTAS-D 静态 route 规划器产生

运行：
  python -u scripts/sa_bt/evaluate_ctasd_dynamic.py
"""

import pickle
import numpy as np
import os
import time
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctasd_static_planner import CTASDStaticRoutePlanner
from dynamic_eval_stats import (
    build_summary_like_my_model,
    calculate_metrics_like_my_model,
)
from dynamic_worker import create_dynamic_planner
from project_paths import ARTIFACTS_ROOT, SA_BT_DATASET_ROOT


def load_env(env_path):
    """加载环境，并确保动态评估所需字段齐全。"""
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    if hasattr(env, 'use_deadline'):
        env.use_deadline = True

    for task in env.task_dic.values():
        task.setdefault('appear_time', 0.0)
        task.setdefault('is_dynamic', False)

    return env


def run_ctasd_dynamic(env, route_planner, max_total_tasks=50, arrival_rate=0.3,
                      simulation_time_limit=200, random_seed=42,
                      decision_log_path=None, debug_verbose=False,
                      dynamic_task_options=None):
    """
    CTAS-D 动态仿真主循环。
    这里的动作生成不再使用在线局部打分器，而是：
    - 在每个动态决策批次，把当前已揭示任务视为一个静态子问题
    - 用 CTAS-D 风格的静态 route 构造器生成 route
    - 每个智能体执行自己 route 的首任务

    除 route 构造外，其余逻辑与 evaluate_my_model_dynamic.py 对齐。
    """
    np.random.seed(random_seed)

    if hasattr(env, 'use_deadline'):
        env.use_deadline = True

    env.init_state()

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

    total_planning_time = 0.0
    planning_count = 0
    initial_planning_time = 0.0
    total_replan_time = 0.0
    explicit_replan_count = 0
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
        decision_log_file.write('time,agent_id,selected_task,available_tasks\n')

    def _write_decision_log(agent_id, selected_task, available_tasks):
        if decision_log_file is None:
            return
        decision_log_file.write(f"{env.current_time:.3f},{agent_id},{selected_task},{available_tasks}\n")

    def _record_decision(agent_id, selected_task, available_tasks):
        decision_history.append({
            'time': float(env.current_time),
            'agent_id': int(agent_id),
            'task_id': int(selected_task),
            'available_tasks': int(available_tasks),
            'battery': float(env.agent_dic[agent_id].get('battery', 0.0)),
        })

    def _plan_batch_routes(target_agent_ids):
        nonlocal total_planning_time
        nonlocal planning_count
        nonlocal initial_planning_time
        nonlocal total_replan_time
        nonlocal explicit_replan_count

        if not target_agent_ids:
            return {}

        start_time = time.time()
        routes = route_planner.plan_routes(target_agent_ids, env)
        elapsed = time.time() - start_time

        total_planning_time += elapsed
        planning_count += 1
        if planning_count == 1:
            initial_planning_time = elapsed
        else:
            total_replan_time += elapsed
            explicit_replan_count += 1
        return routes

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
        reactivated = 0
        for agent in env.agent_dic.values():
            if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                agent['no_choice'] = False
                agent['next_decision'] = env.current_time
                reactivated += 1
        return reactivated

    def _rescue_with_planner_replan():
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

    def _agent_requires_assignment_now(agent_id):
        agent = env.agent_dic[agent_id]

        if last_decision_time.get(agent_id) == env.current_time:
            return False

        if agent.get('current_task', -1) >= 0:
            current_task = env.task_dic.get(agent['current_task'])
            if current_task and agent_id in current_task.get('members', []):
                if current_task.get('feasible_assignment', False):
                    if env.current_time < float(current_task.get('time_finish', env.current_time)):
                        return False
                else:
                    arrival_time = env.get_arrival_time(agent_id, current_task['ID'])
                    wait_deadline = arrival_time + float(getattr(env, 'max_waiting_time', 0.0))
                    if env.current_time < wait_deadline:
                        return False

        if agent['current_task'] >= 0 and agent.get('arrival_time'):
            last_arrival = agent['arrival_time'][-1]
            if last_arrival > env.current_time + 1e-9:
                return False

        if env.check_battery_critical(agent_id):
            return False

        return True

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

        if len(env.task_dic) >= max_total_tasks:
            planner.next_task_arrival_time = None

        next_decision_time = _peek_next_decision_time()
        next_arrival_time = planner.next_task_arrival_time
        if next_arrival_time is not None and next_arrival_time <= next_decision_time:
            _advance_time(next_arrival_time)
            _generate_ready_dynamic_tasks()
            if len(env.task_dic) >= max_total_tasks:
                planner.next_task_arrival_time = None
            for agent in env.agent_dic.values():
                agent['no_choice'] = False
                if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                    agent['next_decision'] = env.current_time
            continue

        if next_decision_time == float('inf') and next_arrival_time is None:
            if _episode_complete():
                env.finished = True
                termination_reason = 'all_tasks_completed'
                break

            if _rescue_with_planner_replan():
                if debug_verbose:
                    print(f"[恢复] time={env.current_time:.3f}, 强制重规划恢复成功")
                continue

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

        release_agents, current_time = env.next_decision()
        env.current_time = current_time

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

        _handle_charging_station_arrivals()
        _generate_ready_dynamic_tasks()

        np.random.shuffle(release_agents[0])
        batch_agent_ids = list(dict.fromkeys(release_agents[0] + release_agents[1]))
        planning_agent_ids = [aid for aid in batch_agent_ids if _agent_requires_assignment_now(aid)]
        batch_routes = _plan_batch_routes(planning_agent_ids)

        processed_agents = set()
        while release_agents[0] or release_agents[1]:
            agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
            if agent_id in processed_agents:
                continue
            processed_agents.add(agent_id)

            agent = env.agent_dic[agent_id]
            agent['no_choice'] = False

            if last_decision_time.get(agent_id) == env.current_time:
                agent['next_decision'] = env.current_time + getattr(env, 'dt', 1e-3)
                continue

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

            if agent['current_task'] >= 0 and agent.get('arrival_time'):
                last_arrival = agent['arrival_time'][-1]
                if last_arrival > env.current_time + 1e-9:
                    agent['next_decision'] = last_arrival
                    continue

            if env.check_battery_critical(agent_id):
                _dispatch_to_charge(agent_id)
                continue

            _, _, mask_np = env.agent_observe(agent_id, False)
            mask_np = _apply_extra_mask(agent_id, mask_np)
            mask_bool = np.asarray(mask_np > 0.5, dtype=bool)

            block_flag = mask_bool[0, 1:].all().item()
            has_contributable = not env.get_contributable_task_mask(agent_id).all()
            if block_flag and not np.all(env.get_matrix(env.task_dic, 'feasible_assignment')):
                can_reach_task = _can_reach_contributable_task(agent_id)
                if has_contributable and not can_reach_task:
                    at_station = np.linalg.norm(agent['location'] - agent['charging_station']) <= 1e-6
                    full_battery = agent.get('battery', 0.0) >= env.initial_battery - 1e-6
                    if at_station and full_battery:
                        _freeze_agent(agent_id, -1, 0)
                        continue
                    _dispatch_to_charge(agent_id)
                    continue

                _freeze_agent(agent_id, -1, 0)
                continue
            elif block_flag and np.all(env.get_matrix(env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                _freeze_agent(agent_id, -1, 0)
                continue

            if not mask_bool[0, 1:].all().item():
                mask_bool[0, 0] = True

            available_tasks = int((~mask_bool[0, 1:]).sum().item())
            selected_task = route_planner.select_task(
                agent_id=agent_id,
                env=env,
                mask_bool=mask_bool,
                planned_route=batch_routes.get(agent_id, []),
            )
            if selected_task is None:
                remaining_agents = [agent_id] + [
                    aid for aid in (release_agents[0] + release_agents[1])
                    if aid not in processed_agents and _agent_requires_assignment_now(aid)
                ]
                if remaining_agents:
                    batch_routes = _plan_batch_routes(remaining_agents)
                    selected_task = route_planner.select_task(
                        agent_id=agent_id,
                        env=env,
                        mask_bool=mask_bool,
                        planned_route=batch_routes.get(agent_id, []),
                    )

            if selected_task is None:
                action = 0
                selected_task_id = -1
            else:
                action = int(selected_task) + 1
                selected_task_id = int(selected_task)

            _write_decision_log(agent_id, selected_task_id, available_tasks)

            _, doable, _ = env.agent_step(agent_id, action, decision_step)
            if not doable:
                agent['no_choice'] = True
                agent['next_decision'] = float('inf')
                agent['is_moving'] = False
            elif selected_task_id == -1 and mask_bool[0, 1:].all().item():
                agent['no_choice'] = True
                agent['next_decision'] = float('inf')
                agent['is_moving'] = False
            else:
                min_step = getattr(env, 'dt', 1e-3)
                if agent['next_decision'] <= env.current_time + 1e-9:
                    agent['next_decision'] = env.current_time + min_step

            _record_decision(agent_id, selected_task_id, available_tasks)
            last_decision_time[agent_id] = env.current_time

        env.finished = env.check_finished()
        if env.finished and _future_arrivals_pending():
            env.finished = False

        if _episode_complete():
            env.finished = True
            termination_reason = 'all_tasks_completed'
        elif env.finished:
            termination_reason = 'all_tasks_finished'

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
        'total_inference_time': total_planning_time,
        'avg_inference_time': total_planning_time / planning_count if planning_count > 0 else 0,
        'inference_count': planning_count,
        'initial_planning_time': initial_planning_time,
        'avg_replan_time': total_replan_time / explicit_replan_count if explicit_replan_count > 0 else 0,
        'replan_count': explicit_replan_count,
        'decision_history': decision_history,
        'termination_reason': termination_reason,
        'simulation_time_limit': simulation_time_limit,
        'raw_end_time': raw_end_time,
        'effective_makespan': effective_makespan,
        'rescue_replan_count': rescue_replan_count,
    }


def calculate_metrics(env, results, initial_task_count, dataset_type='Fixed_Tasks'):
    """计算评估指标，严格复用主模型同口径实现。"""
    return calculate_metrics_like_my_model(
        env=env,
        results=results,
        initial_task_count=initial_task_count,
        dataset_type=dataset_type,
    )


def evaluate_single_env(env_path, route_planner, config_info, arrival_rate,
                        max_total_tasks, dataset_type, log_file):
    try:
        env = load_env(env_path)
        initial_task_count = len(env.task_dic)

        if dataset_type == 'Fixed_Tasks':
            simulation_time_limit = 10000
        else:
            simulation_time_limit = 120

        results = run_ctasd_dynamic(
            env=env,
            route_planner=route_planner,
            max_total_tasks=max_total_tasks,
            arrival_rate=arrival_rate,
            simulation_time_limit=simulation_time_limit,
            random_seed=42,
        )

        metrics = calculate_metrics(env, results, initial_task_count, dataset_type)
        metrics['raw_initial_planning_time'] = results.get('initial_planning_time', 0)
        metrics['raw_avg_replan_time'] = results.get('avg_replan_time', 0)
        metrics['raw_replan_count'] = results.get('replan_count', 0)
        unfinished = metrics['total_tasks'] - metrics['finished_tasks']
        reason_suffix = ''
        if unfinished > 0:
            reason_suffix = f", 终止原因={metrics.get('termination_reason', 'unknown')}"
        rescue_suffix = f", 恢复重规划={int(results.get('rescue_replan_count', 0))}次"

        if dataset_type == 'Fixed_Tasks':
            log_msg = (
                f"  ✓ {os.path.basename(env_path)}: "
                f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                f"({metrics['success_rate']*100:.1f}%), "
                f"makespan={metrics['makespan']:.1f}min, "
                f"截止日期满足={metrics['deadline_satisfaction_rate']*100:.1f}%, "
                f"充电={metrics['total_charging_times']:.0f}次"
                f"{rescue_suffix}"
                f"{reason_suffix}\n"
            )
        else:
            log_msg = (
                f"  ✓ {os.path.basename(env_path)}: "
                f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                f"({metrics['success_rate']*100:.1f}%), "
                f"截止日期满足={metrics['deadline_satisfaction_rate']*100:.1f}%, "
                f"充电={metrics['total_charging_times']:.0f}次"
                f"{rescue_suffix}"
                f"{reason_suffix}\n"
            )

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


def main():
    print('=' * 80)
    print('CTAS-D 动态场景评估')
    print('=' * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    route_planner = CTASDStaticRoutePlanner()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = ARTIFACTS_ROOT / 'results' / 'ctasd_dynamic'
    output_base_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_base_dir / f'evaluation_log_{timestamp}.txt'
    log_file = open(log_file_path, 'w', encoding='utf-8')

    log_file.write('CTAS-D 动态场景评估日志\n')
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write('=' * 80 + '\n\n')

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
        ],
    }

    for dataset_type in ['Fixed_Tasks', 'Fixed_Makespan']:
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
                    route_planner=route_planner,
                    config_info=config,
                    arrival_rate=arrival_rate,
                    max_total_tasks=max_total_tasks,
                    dataset_type=dataset_type,
                    log_file=log_file,
                )

                if metrics:
                    metrics['env_file'] = env_file.name
                    metrics['config_name'] = config_name
                    metrics['dataset_type'] = dataset_type
                    metrics['arrival_rate'] = arrival_rate
                    config_results.append(metrics)

            if config_results:
                results_pkl = output_dir / f'{config_name}_results.pkl'
                with open(results_pkl, 'wb') as f:
                    pickle.dump(config_results, f)

                results_json = output_dir / f'{config_name}_results.json'
                with open(results_json, 'w') as f:
                    json.dump(config_results, f, indent=2)

                summary_msg = build_summary_like_my_model(config_name, config_results)

                log_file.write(summary_msg)
                print(summary_msg, end='')

    end_msg = f"\n{'='*80}\n评估完成！\n{'='*80}\n"
    end_msg += f"结果保存在: {output_base_dir}/\n"
    end_msg += f"日志文件: {log_file_path}\n"
    end_msg += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    end_msg += '=' * 80 + '\n'

    log_file.write(end_msg)
    print(end_msg)
    log_file.close()


if __name__ == '__main__':
    main()
