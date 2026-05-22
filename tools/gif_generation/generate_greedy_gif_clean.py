#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成Greedy算法动态场景GIF - 清洁版本
参考 generate_greedy_gif_static.py 的渲染与导出方式
"""

import argparse
import contextlib
import io
import json
import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import sys
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from dynamic_centralized_planner import DynamicCentralizedPlanner
from project_paths import SA_BT_DATASET_ROOT

DEFAULT_OUTPUT_STEM = 'env_023_basar_clean'
BASELINE_METHODS = ['greedy', 'taco', 'ctasd', 'hrlf', 'capam']
METHOD_LABELS = {
    'model': 'BASAR',
    'greedy': 'Greedy',
    'taco': 'TACO',
    'ctasd': 'CTAS-D',
    'hrlf': 'HRLF',
    'capam': 'CAPAM',
}
METHOD_OUTPUT_SLUGS = {
    'model': 'basar',
    'greedy': 'greedy',
    'taco': 'taco',
    'ctasd': 'ctasd',
    'hrlf': 'hrlf',
    'capam': 'capam',
}

# ==================== 样式配置（参照generate_greedy_gif_detailed.py）====================
STYLE = {
    'unfinished_bg': '#F8FAFC',
    'unfinished_edge': '#94A3B8',
    'task_shadow': '#0F172A',
    'finished_early': '#2A9D8F',
    'finished_late': '#E76F51',
    'executing': '#3B82F6',
    'agents': ['#264653', '#E9C46A', '#F4A261', '#8E44AD', '#27AE60'], 
    'text_stroke': '#333333'       
}

FONT_SIZE_DELTA = 3


def bump_font(size):
    return size + FONT_SIZE_DELTA


def get_agent_color(species_idx):
    """获取智能体颜色"""
    return STYLE['agents'][species_idx % len(STYLE['agents'])]

def get_cmap(n, name='Dark2'):
    """获取颜色映射（保留以兼容旧代码）"""
    return plt.cm.get_cmap(name, n)


def finalize_gif(gif_path, fps=10, colors=96, optimize=True):
    """重新编码 GIF，去掉循环扩展，避免播放到末帧后跳回初始状态。"""
    gif_path = Path(gif_path)
    tmp_path = gif_path.with_suffix(gif_path.suffix + ".tmp.gif")
    ok = False
    try:
        with Image.open(gif_path) as im:
            frames = []
            for i in range(im.n_frames):
                im.seek(i)
                frame = im.convert("RGBA").copy()
                frame.info = {}
                frames.append(frame)

        if not frames:
            return

        frames[0].save(
            tmp_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            optimize=bool(optimize),
            duration=int(1000 / max(fps, 1)),
            disposal=2,
            loop=None,
        )

        with Image.open(tmp_path) as check_im:
            _ = check_im.size
            _ = check_im.n_frames
            check_im.seek(0)
            check_im.convert("RGBA")
            check_im.seek(max(0, check_im.n_frames - 1))
            check_im.convert("RGBA")

        ok = True
        os.replace(tmp_path, gif_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if not ok:
            pass


def choose_render_settings(makespan, interval, fps, max_frames, max_interval):
    estimated_interval = max(interval, makespan / max(max_frames, 1))
    effective_interval = estimated_interval
    if max_interval is not None and max_interval > 0:
        effective_interval = min(effective_interval, max_interval)
    effective_fps = max(1, fps)
    return effective_interval, effective_fps


def resolve_output_path(output_path):
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    return output_path


def resolve_output_paths(args):
    if args.output:
        output_path = resolve_output_path(args.output)
        if output_path.suffix.lower() == ".mp4":
            mp4_path = output_path
            gif_path = output_path.with_suffix(".gif")
        else:
            gif_path = output_path.with_suffix(".gif")
            mp4_path = output_path.with_suffix(".mp4")
        metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
        return gif_path, mp4_path, metrics_path

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_stem = args.output_stem.strip()
    return (
        output_dir / f"{output_stem}.gif",
        output_dir / f"{output_stem}.mp4",
        output_dir / f"{output_stem}_metrics.json",
    )


def infer_model_input_dims():
    from env.task_env import TaskEnv
    from parameters import EnvParams

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


def load_dynamic_checkpoint_model(folder_name, model_name, method_label):
    import torch

    from dynamic_worker import create_dynamic_model
    from parameters import TrainParams
    from project_paths import ensure_checkpoint_exists
    from scripts.sa_bt.evaluate_my_model_dynamic import extract_model_state_dict

    agent_input_dim, task_input_dim = infer_model_input_dims()
    device = torch.device("cuda" if torch.cuda.is_available() and TrainParams.USE_GPU_GLOBAL else "cpu")
    model = create_dynamic_model(
        agent_input_dim=agent_input_dim,
        task_input_dim=task_input_dim,
        embedding_dim=TrainParams.EMBEDDING_DIM,
        device=device,
        model_name=model_name,
    )
    checkpoint_path = ensure_checkpoint_exists(folder_name, method_label=method_label)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.eval()
    return {
        "model": model,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "agent_input_dim": agent_input_dim,
        "task_input_dim": task_input_dim,
    }


def load_basar_model(folder_name):
    return load_dynamic_checkpoint_model(folder_name, model_name='myself', method_label='BASAR')


def load_capam_model(folder_name):
    return load_dynamic_checkpoint_model(folder_name, model_name='capam', method_label='CAPAM')


def load_hrlf_model(folder_name):
    import torch

    from attention import AttentionNet
    from parameters import EnvParams, TrainParams
    from project_paths import ensure_checkpoint_exists

    device = torch.device('cuda' if torch.cuda.is_available() and TrainParams.USE_GPU_GLOBAL else 'cpu')
    model = AttentionNet(
        agent_input_dim=6 + EnvParams.TRAIT_DIM,
        task_input_dim=5 + 2 * EnvParams.TRAIT_DIM,
        embedding_dim=TrainParams.EMBEDDING_DIM,
    ).to(device)
    checkpoint_path = ensure_checkpoint_exists(folder_name, method_label='HRLF')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return {
        "model": model,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }


def run_greedy_dynamic(
    env,
    arrival_rate=0.4,
    max_total_tasks=50,
    simulation_time_limit=200,
    random_seed=42,
    verbose=True,
    max_waiting_time=30,
):
    """运行动态场景"""
    planner = DynamicCentralizedPlanner(
        env=env,
        max_total_tasks=max_total_tasks,
        dynamic_task_arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
        verbose=verbose,
        decision_maker=None,
        max_waiting_time=max_waiting_time,
    )
    
    print("开始运行动态场景模拟...")
    results = planner.run()
    
    stats = env.get_task_statistics()
    print(f"\n运行结果:")
    print(f"  完成任务: {stats['finished_tasks']}/{stats['total_tasks']}")
    print(f"  完成率: {stats['finished_tasks']/stats['total_tasks']*100:.1f}%")
    print(f"  完工时间: {env.current_time:.2f} 分钟")
    
    return results


def run_learning_model_dynamic(
    env,
    model_bundle,
    display_label,
    arrival_rate=3.0,
    max_total_tasks=50,
    simulation_time_limit=200,
    random_seed=42,
):
    from scripts.sa_bt.evaluate_my_model_dynamic import run_model_dynamic

    print(f"开始运行 {display_label} 动态场景评估...")
    results = run_model_dynamic(
        env=env,
        global_network=model_bundle["model"],
        device=model_bundle["device"],
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
        sampling=False,
    )

    stats = env.get_task_statistics()
    print(f"\n运行结果:")
    print(f"  完成任务: {stats['finished_tasks']}/{stats['total_tasks']}")
    print(f"  完成率: {stats['finished_tasks']/stats['total_tasks']*100:.1f}%")
    print(f"  终止原因: {results.get('termination_reason', 'unknown')}")
    print(f"  完工时间: {results.get('effective_makespan', env.current_time):.2f} 分钟")
    print(f"  推理总时长: {results.get('total_inference_time', 0.0):.4f} 秒")
    return results


def run_basar_dynamic(
    env,
    model_bundle,
    arrival_rate=3.0,
    max_total_tasks=50,
    simulation_time_limit=200,
    random_seed=42,
):
    return run_learning_model_dynamic(
        env,
        model_bundle,
        display_label='BASAR',
        arrival_rate=arrival_rate,
        max_total_tasks=max_total_tasks,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
    )


def run_hrlf_visual_dynamic(
    env,
    model_bundle,
    arrival_rate=3.0,
    max_total_tasks=50,
    simulation_time_limit=200,
    random_seed=42,
):
    from scripts.sa_bt.evaluate_hrlf_dynamic import run_hrlf_dynamic

    print("开始运行 HRLF 动态场景评估...")
    results = run_hrlf_dynamic(
        env=env,
        global_network=model_bundle["model"],
        device=model_bundle["device"],
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
        sampling=False,
    )

    stats = env.get_task_statistics()
    print(f"\n运行结果:")
    print(f"  完成任务: {stats['finished_tasks']}/{stats['total_tasks']}")
    print(f"  完成率: {stats['finished_tasks']/stats['total_tasks']*100:.1f}%")
    print(f"  终止原因: {results.get('termination_reason', 'unknown')}")
    print(f"  完工时间: {results.get('effective_makespan', env.current_time):.2f} 分钟")
    print(f"  推理总时长: {results.get('total_inference_time', 0.0):.4f} 秒")
    return results


def run_static_route_baseline_dynamic(
    env,
    route_planner,
    display_label,
    arrival_rate=3.0,
    max_total_tasks=50,
    simulation_time_limit=200,
    random_seed=42,
):
    from scripts.sa_bt.evaluate_ctasd_dynamic import run_ctasd_dynamic

    print(f"开始运行 {display_label} 动态场景评估...")
    results = run_ctasd_dynamic(
        env=env,
        route_planner=route_planner,
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=random_seed,
    )

    stats = env.get_task_statistics()
    print(f"\n运行结果:")
    print(f"  完成任务: {stats['finished_tasks']}/{stats['total_tasks']}")
    print(f"  完成率: {stats['finished_tasks']/stats['total_tasks']*100:.1f}%")
    print(f"  终止原因: {results.get('termination_reason', 'unknown')}")
    print(f"  完工时间: {results.get('effective_makespan', env.current_time):.2f} 分钟")
    print(f"  规划总时长: {results.get('total_inference_time', 0.0):.4f} 秒")
    return results


def _event_target_location(env, agent_id, task_id):
    if task_id == -999:
        return env.agent_dic[agent_id]['charging_station']
    if task_id >= 0 and task_id in env.task_dic:
        return env.task_dic[task_id]['location']
    return env.agent_dic[agent_id]['depot']


def build_agent_paths_from_routes(env):
    """从模型评估后的 route/arrival_time 重建动画路径。"""
    agent_task_paths = {}
    for agent_id, agent in env.agent_dic.items():
        route = list(agent.get('route', []))
        arrival_times = list(agent.get('arrival_time', []))
        agent_task_paths[agent_id] = []
        if len(route) < 2 or len(arrival_times) < 2:
            continue

        prev_pos = agent['depot'].copy()
        prev_finish = 0.0
        pair_count = min(len(route), len(arrival_times))
        for idx in range(1, pair_count):
            task_id = int(route[idx])
            target_pos = _event_target_location(env, agent_id, task_id)
            if isinstance(target_pos, (int, float)):
                continue

            arrival_time = float(arrival_times[idx])
            distance = float(np.linalg.norm(target_pos - prev_pos))
            speed = max(float(agent.get('velocity', 0.2)), 1e-12)
            travel_time = distance / speed
            start_time = max(prev_finish, arrival_time - travel_time)
            if start_time > arrival_time:
                start_time = arrival_time

            if task_id >= 0 and task_id in env.task_dic:
                task = env.task_dic[task_id]
                finish_time = float(task.get('time_finish', 0.0))
                if finish_time <= 0:
                    finish_time = arrival_time
            else:
                finish_time = arrival_time

            completion_time = max(arrival_time, finish_time)
            agent_task_paths[agent_id].append(
                (task_id, start_time, arrival_time, completion_time, prev_pos.copy())
            )
            prev_pos = target_pos.copy()
            prev_finish = completion_time

    return agent_task_paths


def create_gif_from_planning(
    env,
    results,
    output_path,
    output_mp4_path=None,
    interval=0.2,
    fps=10,
    dpi=120,
    figsize=10,
    max_frames=700,
    max_interval=1.0,
    warmup_frames=12,
    metric_overlay_frames=20,
    colors=96,
    optimize=False,
    blit=False,
    model_name="greedy",
    agent_task_paths_override=None,
    metrics_overlay_lines=None,
):
    """
    从规划历史创建GIF动画
    参考task_env.py的plot_animation实现
    """
    print(f"\n正在创建GIF动画...")
    
    output_path = Path(output_path)

    planning_history = results.get('planning_history', [])
    if agent_task_paths_override is not None:
        print("  正在使用 route/arrival_time 重建智能体任务路径...")
        agent_task_paths = agent_task_paths_override
    else:
        # 获取规划历史
        if not planning_history:
            print("错误: 规划历史为空！")
            return

        # 提前提取每个智能体的完整任务路径（包括充电路径）
        print(f"  正在提取智能体任务路径...")
        agent_task_paths = {}  # {agent_id: [(task_id, start_time, completion_time, start_pos), ...]}
        agent_charging_events = {}  # {agent_id: [(start_time, arrival_time, completion_time, start_pos), ...]}

        for agent_id in env.agent_dic.keys():
            agent_task_paths[agent_id] = []
            agent_charging_events[agent_id] = []

        # 检查智能体的充电次数
        total_charging_count = 0
        for agent_id, agent in env.agent_dic.items():
            charging_times = agent.get('total_charging_times', 0)
            if charging_times > 0:
                total_charging_count += charging_times
                print(f"  智能体 {agent_id}: 充电 {charging_times} 次")

        if total_charging_count > 0:
            print(f"  ✓ 总计 {total_charging_count} 次充电事件（智能体电量统计）")
        else:
            print(f"  ℹ 未检测到充电事件（电量消耗可能不足）")

        # 只提取真正完成的任务（有完成时间的）和充电事件
        completed_tasks = {}  # {task_id: (agent_id, completion_time)}
        charging_events = {}  # {(agent_id, plan_time): (arrival_time, charging_station_loc, start_pos)}

        for task_id, task in env.task_dic.items():
            completion_time = task.get('time_finish', 0)
            if completion_time > 0:
                # 找到完成这个任务的智能体
                for record in planning_history:
                    assignments = record.get('assignments', {})
                    for agent_id, assign_info in assignments.items():
                        task_queue = []
                        if isinstance(assign_info, dict) and 'task_queue' in assign_info:
                            task_queue = assign_info['task_queue']
                        elif isinstance(assign_info, list):
                            task_queue = assign_info

                        if task_id in task_queue:
                            completed_tasks[task_id] = (agent_id, completion_time)
                            break
                    if task_id in completed_tasks:
                        break

        # 提取充电事件
        for record in planning_history:
            plan_time = record.get('time', 0)
            assignments = record.get('assignments', {})
            for agent_id, assign_info in assignments.items():
                if isinstance(assign_info, dict):
                    task_id = assign_info.get('task_id', None)
                    if task_id == -999 and assign_info.get('is_charging', False):
                        # 这是一个充电事件
                        arrival_time = assign_info.get('arrival_time', 0)
                        charging_station_loc = assign_info.get('charging_station', None)
                        if charging_station_loc is not None:
                            # 获取智能体当前位置作为起始位置
                            start_pos = env.agent_dic[agent_id]['location'].copy()
                            charging_events[(agent_id, plan_time)] = (arrival_time, charging_station_loc, start_pos)

        if charging_events:
            print(f"  ✓ 提取到 {len(charging_events)} 个充电事件")
            for (aid, ptime), (atime, _, _) in list(charging_events.items())[:3]:
                print(f"    智能体 {aid}: 在 {ptime:.2f} 分派往充电站，到达 {atime:.2f}")

        # 按智能体组织已完成的任务
        for task_id, (agent_id, completion_time) in completed_tasks.items():
            agent_task_paths[agent_id].append((task_id, completion_time))

        # 按完成时间排序并计算正确的起始位置和起始时间（包括充电事件）
        for agent_id in agent_task_paths:
            # 先按完成时间排序任务
            agent_task_paths[agent_id].sort(key=lambda x: x[1])

            # 收集该智能体的所有事件（任务+充电），并按时间排序
            all_events = []  # [(event_type, time, data), ...]

            # 添加任务事件
            for task_id, completion_time in agent_task_paths[agent_id]:
                all_events.append(('task', completion_time, task_id))

            # 添加充电事件
            for (aid, plan_time), (arrival_time, cs_loc, start_pos) in charging_events.items():
                if aid == agent_id:
                    # 充电完成时间 = 到达时间（瞬时充电）
                    all_events.append(('charging', arrival_time, (cs_loc, start_pos, plan_time)))

            # 按事件完成时间排序
            all_events.sort(key=lambda x: x[1])

            # 重新构建路径，计算正确的起始位置和起始时间
            corrected_path = []
            prev_pos = env.agent_dic[agent_id]['depot'].copy()
            prev_time = 0.0

            for event_type, event_time, event_data in all_events:
                if event_type == 'task':
                    task_id = event_data
                    completion_time = event_time

                    if task_id in env.task_dic:
                        task = env.task_dic[task_id]
                        task_loc = task['location']

                        # 获取任务的执行时间
                        execution_time = task.get('execution_time', 0)
                        if execution_time <= 0:
                            # 如果没有执行时间，根据requirements估算
                            execution_time = task['requirements'].sum() * 0.5

                        # 计算从prev_pos到task_loc的距离和旅行时间
                        distance = np.linalg.norm(task_loc - prev_pos)
                        agent_speed = max(float(env.agent_dic[agent_id].get('velocity', 0.2)), 1e-12)
                        travel_time = distance / agent_speed

                        # 任务到达时间 = 完成时间 - 执行时间
                        arrival_time = completion_time - execution_time

                        # 任务开始移动时间 = 到达时间 - 旅行时间
                        start_time = arrival_time - travel_time

                        # 确保开始时间不早于上一个事件的完成时间
                        if start_time < prev_time:
                            start_time = prev_time
                            arrival_time = start_time + travel_time

                        # 保存：(task_id, start_time, arrival_time, completion_time, start_pos)
                        corrected_path.append((task_id, start_time, arrival_time, completion_time, prev_pos.copy()))

                        # 更新位置和时间
                        prev_pos = task_loc.copy()
                        prev_time = completion_time

                elif event_type == 'charging':
                    cs_loc, recorded_start_pos, plan_time = event_data
                    arrival_time = event_time

                    # 计算从prev_pos到充电站的距离和旅行时间
                    distance = np.linalg.norm(cs_loc - prev_pos)
                    agent_speed = max(float(env.agent_dic[agent_id].get('velocity', 0.2)), 1e-12)
                    travel_time = distance / agent_speed

                    # 开始移动时间
                    start_time = arrival_time - travel_time

                    # 确保开始时间不早于上一个事件的完成时间
                    if start_time < prev_time:
                        start_time = prev_time
                        arrival_time = start_time + travel_time

                    # 充电瞬时完成，完成时间 = 到达时间
                    completion_time = arrival_time

                    # 保存：(-999, start_time, arrival_time, completion_time, start_pos)
                    corrected_path.append((-999, start_time, arrival_time, completion_time, prev_pos.copy()))

                    # 更新位置和时间
                    prev_pos = cs_loc.copy()
                    prev_time = completion_time

            agent_task_paths[agent_id] = corrected_path
    
    mission_end_time = float(env.current_time)
    return_end_time = mission_end_time

    def resolve_path_target(agent_id, task_id):
        if task_id == -999:
            return env.agent_dic[agent_id]['charging_station']
        if task_id in env.task_dic:
            return env.task_dic[task_id]['location']
        if task_id < 0:
            return env.agent_dic[agent_id]['depot']
        return None

    for agent_id, agent in env.agent_dic.items():
        path = agent_task_paths.setdefault(agent_id, [])
        final_pos = agent['depot'].copy()
        for task_id, _, _, completion_time, _ in path:
            if completion_time > mission_end_time + 1e-6:
                continue
            target_loc = resolve_path_target(agent_id, task_id)
            if target_loc is not None and not isinstance(target_loc, (int, float)):
                final_pos = target_loc.copy()

        depot = agent['depot'].copy()
        distance_to_depot = np.linalg.norm(final_pos - depot)
        if distance_to_depot > 1e-6:
            speed = max(float(agent.get('velocity', 0.2)), 1e-12)
            return_arrival = mission_end_time + distance_to_depot / speed
            path.append((-100000 - int(agent_id), mission_end_time, return_arrival, return_arrival, final_pos.copy()))
            return_end_time = max(return_end_time, return_arrival)

    # 打印提取的路径信息（调试用，包括充电事件）
    for agent_id, path in agent_task_paths.items():
        if path:
            task_count = sum(1 for task_id, _, _, _, _ in path if task_id >= 0)
            charging_count = sum(1 for task_id, _, _, _, _ in path if task_id == -999)
            return_count = sum(1 for task_id, _, _, _, _ in path if task_id < 0 and task_id != -999)
            print(f"    智能体 {agent_id}: {task_count} 个任务, {charging_count} 次充电, {return_count} 次返仓")
            for idx, (task_id, start_time, arrival_time, completion_time, start_pos) in enumerate(path[:5]):  # 只打印前5个
                if task_id == -999:
                    print(f"      充电{idx+1}: ⚡ [移动:{start_time:.2f}→到达:{arrival_time:.2f}→完成:{completion_time:.2f}]")
                elif task_id < 0:
                    print(f"      返仓{idx+1}: depot [移动:{start_time:.2f}→到达:{arrival_time:.2f}]")
                else:
                    print(f"      任务{idx+1}: T{task_id} [移动:{start_time:.2f}→到达:{arrival_time:.2f}→完成:{completion_time:.2f}]")
    
    # 确定时间范围和帧数：与静态版一致，拆成 warmup / simulation / final metrics 三段。
    max_time = mission_end_time
    animation_end_time = return_end_time
    effective_interval, effective_fps = choose_render_settings(
        makespan=animation_end_time,
        interval=interval,
        fps=fps,
        max_frames=max_frames,
        max_interval=max_interval,
    )
    sim_frames = int(np.ceil(animation_end_time / max(effective_interval, 1e-12))) + 1
    num_frames = int(warmup_frames) + sim_frames + int(metric_overlay_frames) + 1
    
    print(f"  任务完成时间: {max_time:.2f} 分钟")
    print(f"  返仓结束时间: {animation_end_time:.2f} 分钟")
    print(f"  时间间隔: {effective_interval:.3f} 分钟")
    print(f"  总帧数: {num_frames}")
    print(f"  帧率: {effective_fps} fps")
    
    # 创建图形
    fig, ax = plt.subplots(dpi=dpi, figsize=(figsize, figsize))
    fig.patch.set_facecolor('white')
    ax.set_xlim(-0.25, 10.25)
    ax.set_ylim(-0.25, 10.25)
    ax.set_aspect('equal')
    ax.grid(True, zorder=0, alpha=0.35)
    
    ax.set_xlabel('X Position', fontweight='bold', fontsize=bump_font(10), labelpad=1)
    ax.set_ylabel('Y Position', fontweight='bold', fontsize=bump_font(10), labelpad=1)
    ax.tick_params(axis='both', labelsize=bump_font(9), pad=1)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.105)
    
    # 获取任务完成统计
    finished_tasks = sum(1 for t in env.task_dic.values() if t.get('time_finish', 0) > 0)
    total_tasks = len(env.task_dic)
    finished_rate = finished_tasks / total_tasks if total_tasks > 0 else 0
    
    # 创建图例（使用新样式，添加"正在执行"状态和充电路径）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=STYLE['unfinished_bg'], markeredgecolor=STYLE['unfinished_edge'],
               markersize=10, markeredgewidth=2, label='Pending Task'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=STYLE['executing'], markersize=10, label='Executing Task'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=STYLE['finished_early'], markersize=10, label='Done (On Time)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=STYLE['finished_late'], markersize=10, label='Done (Late)'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='gray', markersize=14,
               markeredgecolor='white', markeredgewidth=1.5, label='Agent'),
        Line2D([0], [0], color='gray', linewidth=2,
               linestyle='-', label='Current Route'),
        Line2D([0], [0], color='orange', linewidth=2.5,
               linestyle='--', label='Charging Route'),  # 新增：充电路径
        Line2D([0], [0], color='gray', linewidth=2,
               linestyle='-', alpha=0.4, label='History Trail'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper center',
             bbox_to_anchor=(0.5, -0.045), ncol=4,
             fontsize=bump_font(9), framealpha=0.92, edgecolor='#34495E',
             fancybox=False, shadow=False, borderaxespad=0.0,
             columnspacing=1.0, handletextpad=0.6)
    
    # 绘制任务（多边形）
    task_shadows = {}
    task_polygons = {}
    task_labels = {}  # 新增：存储任务标签文本对象
    for tid, task in env.task_dic.items():
        loc = task['location']
        if not isinstance(loc, (int, float)):
            x, y = loc[0] * 10, loc[1] * 10
            num_vertices = int(np.clip(task['requirements'].sum() + 3, 5, 8))
            shadow = ax.add_patch(patches.RegularPolygon(
                xy=(x + 0.035, y - 0.035),
                numVertices=num_vertices,
                radius=0.43,
                facecolor=STYLE['task_shadow'],
                edgecolor='none',
                alpha=0.08,
                zorder=4
            ))
            poly = ax.add_patch(patches.RegularPolygon(
                xy=(x, y),
                numVertices=num_vertices,
                radius=0.40,
                facecolor=STYLE['unfinished_bg'],
                edgecolor=STYLE['unfinished_edge'],
                linewidth=1.3,
                zorder=5
            ))
            task_shadows[tid] = shadow
            task_polygons[tid] = poly
            
            # 添加任务标签（初始隐藏，动态更新时显示）
            is_dynamic = task.get('is_dynamic', False)
            # 计算动态任务的索引（从1开始）
            if is_dynamic:
                # 收集所有动态任务ID并排序，找到当前任务的索引
                all_dynamic_tids = sorted([t_id for t_id, t in env.task_dic.items() if t.get('is_dynamic', False)])
                dynamic_index = all_dynamic_tids.index(tid) + 1  # 从1开始
                label = f"D{dynamic_index}"
            else:
                label = str(tid)
            
            text_obj = ax.text(x, y, label, ha='center', va='center',
                              fontsize=bump_font(11), fontweight='bold', color='#475569',
                              zorder=6, alpha=1.0, visible=False)
            text_obj.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground='white', alpha=0.85),
            ])
            task_labels[tid] = text_obj
    
    # 绘制基地（矩形虚线框，使用新样式）
    depots_drawn = set()
    for agent in env.agent_dic.values():
        sp = agent['species']
        if sp not in depots_drawn:
            depot_loc = agent['depot']
            if not isinstance(depot_loc, (int, float)):
                dx, dy = depot_loc[0] * 10, depot_loc[1] * 10
                c = get_agent_color(sp)
                
                # 绘制虚线矩形
                rect = patches.Rectangle(
                    (dx-0.3, dy-0.3), 0.6, 0.6, lw=1.5, edgecolor=c,
                    facecolor='none', ls='--', alpha=0.6, zorder=4
                )
                ax.add_patch(rect)
                
                # 添加基地标签
                ax.text(dx, dy-0.65, f'B-{sp}', ha='center', va='top', 
                       fontsize=bump_font(11), color=c, fontweight='bold', alpha=0.8, zorder=4)
                depots_drawn.add(sp)
    
    # 绘制充电站（星形标记）
    charging_stations_drawn = set()
    if hasattr(env, 'charging_station_dic') and env.charging_station_dic:
        for cs_id, cs_info in env.charging_station_dic.items():
            if cs_id not in charging_stations_drawn:
                cs_loc = cs_info['location']
                if not isinstance(cs_loc, (int, float)):
                    cx, cy = cs_loc[0] * 10, cs_loc[1] * 10
                    # 使用星形表示充电站
                    ax.add_patch(patches.RegularPolygon(
                        (cx, cy), numVertices=5, radius=0.25,
                        facecolor='gold', edgecolor='orange', linewidth=2, zorder=3
                    ))
                    # 添加充电站标签
                    ax.text(cx, cy + 0.4, '⚡', ha='center', va='bottom',
                           fontsize=bump_font(12), color='orange', fontweight='bold')
                    charging_stations_drawn.add(cs_id)
    
    # 创建智能体标记（使用plot绘制星形，与参考文件一致）
    agent_markers = {}
    for aid, agent in env.agent_dic.items():
        depot_loc = agent['depot']
        if not isinstance(depot_loc, (int, float)):
            x, y = depot_loc[0] * 10, depot_loc[1] * 10
            c = get_agent_color(agent['species'])
            
            # 使用plot绘制星形标记
            marker, = ax.plot(x, y, marker='*', markersize=18, color=c, 
                            markeredgecolor='white', markeredgewidth=1.2, zorder=10)
            agent_markers[aid] = marker
    
    # 创建箭头（用于显示智能体到任务的路径）
    # 每个智能体只需要一个箭头显示当前任务
    from matplotlib.patches import FancyArrowPatch
    agent_arrows = {}
    for aid in env.agent_dic.keys():
        arrow = FancyArrowPatch((0, 0), (0, 0),
                               arrowstyle='-|>',
                               mutation_scale=16,
                               linewidth=0,
                               color='gray',
                               alpha=0,
                               connectionstyle='arc3,rad=0.04',
                               zorder=5)
        arrow.set_path_effects([
            pe.Stroke(linewidth=4.0, foreground='white', alpha=0.65),
            pe.Normal(),
        ])
        ax.add_patch(arrow)
        agent_arrows[aid] = arrow
    
    # 每个智能体一条完整历史轨迹线，避免只显示最近几帧导致末帧信息不完整。
    history_lines = {}
    for aid, agent in env.agent_dic.items():
        line, = ax.plot(
            [], [], '-', color=get_agent_color(agent['species']),
            linewidth=1.1, alpha=0.32, zorder=4,
            solid_capstyle='round', solid_joinstyle='round'
        )
        line.set_visible(False)
        history_lines[aid] = line
    history_arrows = []

    def get_event_target_loc(aid, task_id):
        if task_id == -999:
            return env.agent_dic[aid]['charging_station']
        if task_id in env.task_dic:
            return env.task_dic[task_id]['location']
        if task_id < 0:
            return env.agent_dic[aid]['depot']
        return None

    def get_agent_history_path(aid, target_time):
        path = agent_task_paths.get(aid, [])
        depot = env.agent_dic[aid]['depot'].copy()
        points = [depot]
        last = depot

        for task_id, start_time, arrival_time, completion_time, start_pos in path:
            target_loc = get_event_target_loc(aid, task_id)
            if target_loc is None or isinstance(target_loc, (int, float)):
                continue
            appear_time = 0.0 if task_id < 0 else env.task_dic[task_id].get('appear_time', 0.0)
            if appear_time > target_time or target_time < start_time:
                break
            if start_time <= target_time < arrival_time:
                total = max(arrival_time - start_time, 1e-12)
                ratio = (target_time - start_time) / total
                current = start_pos + (target_loc - start_pos) * ratio
                if np.linalg.norm(current - last) > 1e-6:
                    points.append(current.copy())
                break
            if target_time >= arrival_time:
                if np.linalg.norm(target_loc - last) > 1e-6:
                    points.append(target_loc.copy())
                    last = target_loc.copy()

        return points
    
    def get_agent_current_task(aid, target_time):
        """获取智能体在指定时间的当前位置和当前任务
        
        返回: (current_pos, current_task_id, task_start_pos, task_end_pos, is_executing)
        - current_pos: 当前位置（插值计算）
        - current_task_id: 当前正在执行的任务ID（如果有），-999表示充电
        - task_start_pos: 当前任务的起始位置
        - task_end_pos: 当前任务的目标位置
        - is_executing: 是否正在执行任务或充电（已到达但未完成）
        """
        path = agent_task_paths.get(aid, [])
        if not path:
            depot = env.agent_dic[aid]['depot'].copy()
            return depot, None, None, None, False
        
        # 找到智能体在当前时刻的状态
        current_pos = env.agent_dic[aid]['depot'].copy()
        last_completed_time = 0.0
        
        for task_id, start_time, arrival_time, completion_time, start_pos in path:
            # 确定目标位置
            if task_id == -999:
                # 充电站任务
                target_loc = env.agent_dic[aid]['charging_station']
            elif task_id in env.task_dic:
                task = env.task_dic[task_id]
                target_loc = task['location']
                
                # *** 关键修复：检查任务是否已经出现 ***
                appear_time = task.get('appear_time', 0.0)
                if appear_time > target_time:
                    # 任务还未出现，智能体不应该知道这个任务
                    # 停留在当前位置
                    break
            elif task_id < 0:
                target_loc = env.agent_dic[aid]['depot']
            else:
                continue
            
            # 如果事件已完成且完成时间在目标时间之前
            if completion_time <= target_time:
                current_pos = target_loc.copy()
                last_completed_time = completion_time
            # 如果智能体正在执行任务/充电（已到达但未完成）
            elif arrival_time <= target_time < completion_time:
                # 智能体在目标点执行任务或充电
                current_pos = target_loc.copy()
                return current_pos, task_id, target_loc, target_loc, True  # is_executing=True
            # 如果智能体正在前往目标（开始移动但未到达）
            elif start_time <= target_time < arrival_time:
                # 计算当前位置（线性插值）
                if not isinstance(target_loc, (int, float)) and not isinstance(start_pos, (int, float)):
                    direction = target_loc - start_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0.001:
                        time_elapsed = target_time - start_time
                        speed = max(float(env.agent_dic[aid].get('velocity', 0.2)), 1e-12)
                        traveled = min(time_elapsed * speed, distance)
                        progress = traveled / distance
                        current_pos = start_pos + direction * progress
                        current_pos = np.clip(current_pos, 0.0, 1.0)
                    else:
                        current_pos = target_loc.copy()
                    
                    return current_pos, task_id, start_pos, target_loc, False  # is_executing=False
            # 如果事件还未开始
            else:
                # 智能体应该在上一个事件完成的位置等待
                # current_pos 已经在前面设置好了
                break
        
        # 没有正在执行的任务/充电，智能体在等待或已完成所有事件
        return current_pos, None, None, None, False
    
    title_text = ax.text(
        0.5, 1.005, '', ha='center', va='bottom',
        fontsize=bump_font(13), fontweight='bold', color='#2C3E50', transform=ax.transAxes
    )
    metrics_text = ax.text(
        0.02, 0.03, '',
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=bump_font(10),
        color='#1F2D3A',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#9FB3C8', alpha=0.9),
        zorder=20,
    )
    
    def update_frame(frame):
        """更新动画帧"""
        in_warmup = frame <= warmup_frames
        sim_end_frame = int(warmup_frames) + int(sim_frames)
        in_final_overlay = frame > sim_end_frame

        if in_warmup:
            current_time = 0.0
        elif frame <= sim_end_frame:
            current_time = min((frame - warmup_frames) * effective_interval, animation_end_time)
        else:
            current_time = animation_end_time + 1e-6

        display_time = min(current_time, max_time)
        if current_time > animation_end_time:
            phase_label = 'returned'
        elif current_time > max_time:
            phase_label = 'returning to depots'
        else:
            phase_label = ''
        current_finished = sum(
            1 for task in env.task_dic.values()
            if task.get('time_finish', 0) > 0 and task.get('time_finish', 0) <= current_time
        )
        title = (
            f'{model_name} | t={display_time:.2f} | '
            f'finished={current_finished}/{total_tasks} | makespan={max_time:.2f}'
        )
        if phase_label:
            title = f'{title} | {phase_label}'
        title_text.set_text(title)

        if in_final_overlay:
            if metrics_overlay_lines is not None:
                metrics_text.set_text("\n".join(metrics_overlay_lines))
            else:
                metrics_text.set_text(
                    f'model: {model_name}\n'
                    f'tasks_done: {finished_tasks}/{total_tasks}\n'
                    f'completion_rate: {finished_rate * 100:.1f}%\n'
                    f'makespan: {max_time:.4f}\n'
                    f'replans: {len(planning_history)}'
                )
            metrics_text.set_visible(True)
        else:
            metrics_text.set_visible(False)
        
        # 任务标签随任务到达一起显示，尚未到达的动态任务保持隐藏。
        executing_tasks = set()
        if not in_warmup:
            for aid in agent_task_paths:
                path = agent_task_paths[aid]
                for task_id, start_time, arrival_time, completion_time, start_pos in path:
                    if task_id < 0 or task_id not in env.task_dic:
                        continue
                    if arrival_time <= current_time < completion_time:
                        executing_tasks.add(task_id)
        
        # 更新任务颜色和标签（使用新样式，添加"正在执行"状态）
        for tid, poly in task_polygons.items():
            task = env.task_dic[tid]
            appear_time = task.get('appear_time', 0.0)
            time_finish = task.get('time_finish', 0)
            shadow = task_shadows.get(tid)
            
            if appear_time > current_time:
                # 任务还未出现 - 完全隐藏（包括标签）
                poly.set_alpha(0.0)
                if shadow is not None:
                    shadow.set_alpha(0.0)
                if tid in task_labels:
                    task_labels[tid].set_visible(False)
            elif time_finish > 0 and time_finish <= current_time:
                # 任务已完成
                deadline = task.get('deadline', float('inf'))
                if time_finish <= deadline:
                    poly.set_facecolor(STYLE['finished_early'])  # 绿色 - 按时
                else:
                    poly.set_facecolor(STYLE['finished_late'])   # 红色 - 迟到
                poly.set_edgecolor('white')
                poly.set_linewidth(1.0)
                poly.set_alpha(1.0)
                if shadow is not None:
                    shadow.set_alpha(0.09)
                if tid in task_labels:
                    task_labels[tid].set_visible(True)
            elif tid in executing_tasks:
                # 任务正在执行（已到达但未完成）
                poly.set_facecolor(STYLE['executing'])  # 蓝色 - 正在执行
                poly.set_edgecolor('white')
                poly.set_linewidth(1.1)
                poly.set_alpha(1.0)
                if shadow is not None:
                    shadow.set_alpha(0.12)
                if tid in task_labels:
                    task_labels[tid].set_visible(True)
            else:
                # 任务未完成但已出现（等待中）
                poly.set_facecolor(STYLE['unfinished_bg'])
                poly.set_edgecolor(STYLE['unfinished_edge'])
                poly.set_linewidth(1.3)
                poly.set_alpha(1.0)
                if shadow is not None:
                    shadow.set_alpha(0.07)
                if tid in task_labels:
                    task_labels[tid].set_visible(True)
        
        # 绘制完整历史轨迹
        for aid, line in history_lines.items():
            if in_warmup:
                line.set_visible(False)
                continue
            hist_points = get_agent_history_path(aid, current_time)
            if len(hist_points) >= 2:
                hist = np.vstack(hist_points)
                line.set_data(hist[:, 0] * 10, hist[:, 1] * 10)
                line.set_visible(True)
            else:
                line.set_visible(False)
        
        # 隐藏所有历史箭头
        for arrow in history_arrows:
            arrow.set_visible(False)
        
        # 更新智能体位置和当前任务箭头
        for aid, marker in agent_markers.items():
            agent = env.agent_dic[aid]
            
            # 获取当前位置和当前任务（包括is_executing状态）
            if in_warmup:
                current_pos = agent['depot'].copy()
                current_task_id, task_start_pos, task_end_pos, is_executing = None, None, None, False
            else:
                current_pos, current_task_id, task_start_pos, task_end_pos, is_executing = get_agent_current_task(aid, current_time)
            
            # 更新智能体位置
            if not isinstance(current_pos, (int, float)):
                ax_pos, ay_pos = current_pos[0] * 10, current_pos[1] * 10
                marker.set_data([ax_pos], [ay_pos])  # 使用set_data更新plot对象
            
            # 绘制当前任务/充电箭头（只在移动时显示，执行时不显示）
            arrow = agent_arrows[aid]
            if current_task_id is not None and not is_executing and task_start_pos is not None and task_end_pos is not None:
                if not isinstance(task_end_pos, (int, float)) and not isinstance(current_pos, (int, float)):
                    cx, cy = current_pos[0] * 10, current_pos[1] * 10
                    tx, ty = task_end_pos[0] * 10, task_end_pos[1] * 10
                    dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
                    
                    if dist > 0.2:  # 距离足够远才显示箭头
                        arrow.set_positions((cx, cy), (tx, ty))
                        
                        # 充电箭头使用橙色，任务箭头使用智能体颜色
                        if current_task_id == -999:
                            arrow.set_color('orange')  # 充电箭头
                            arrow.set_linewidth(2.2)
                            arrow.set_linestyle('--')  # 虚线表示充电
                            arrow.set_alpha(0.78)
                        else:
                            arrow.set_color(get_agent_color(agent['species']))  # 任务箭头
                            arrow.set_linewidth(2.0)
                            arrow.set_linestyle('-')
                            arrow.set_alpha(0.76)
                        
                        arrow.set_visible(True)
                    else:
                        arrow.set_visible(False)
                else:
                    arrow.set_visible(False)
            else:
                arrow.set_visible(False)
        
        return [title_text, metrics_text] + list(task_shadows.values()) + \
               list(task_polygons.values()) + list(task_labels.values()) + \
               list(agent_markers.values()) + list(agent_arrows.values()) + \
               list(history_lines.values()) + history_arrows
    
    # 创建动画
    print(f"\n  正在生成动画...")
    anim = animation.FuncAnimation(
        fig, update_frame,
        frames=num_frames,
        interval=1000 / effective_fps,
        blit=bool(blit),
        repeat=False
    )
    saved_paths = {}
    
    # 保存GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  正在保存GIF到 {output_path}...")
    writer = animation.PillowWriter(fps=effective_fps)
    anim.save(output_path, writer=writer)
    finalize_gif(output_path, fps=effective_fps, colors=colors, optimize=optimize)
    saved_paths["gif"] = output_path

    if output_mp4_path is not None:
        output_mp4_path = Path(output_mp4_path)
        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
        if not animation.writers.is_available("ffmpeg") and shutil.which("ffmpeg") is None:
            raise RuntimeError("未检测到 ffmpeg，无法导出 MP4。请安装 ffmpeg 后重试。")
        print(f"  正在保存MP4到 {output_mp4_path}...")
        mp4_writer = animation.FFMpegWriter(
            fps=effective_fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        anim.save(output_mp4_path, writer=mp4_writer)
        saved_paths["mp4"] = output_mp4_path
    
    plt.close(fig)
    
    # 检查文件大小
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ GIF保存成功！文件大小: {file_size:.2f} MB")
    else:
        print(f"  ✗ GIF保存失败！")
    if output_mp4_path is not None and output_mp4_path.exists():
        file_size = output_mp4_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ MP4保存成功！文件大小: {file_size:.2f} MB")
    return saved_paths


def build_metrics_payload(
    env,
    results,
    args,
    env_file,
    gif_path,
    mp4_path,
    checkpoint_path=None,
    baseline_metrics=None,
    display_label=None,
):
    stats = env.get_task_statistics()
    total_tasks = int(stats.get('total_tasks', len(env.task_dic)))
    finished_tasks = int(stats.get('finished_tasks', 0))
    success_rate = finished_tasks / total_tasks if total_tasks > 0 else 0.0
    effective_makespan = float(results.get('effective_makespan', env.current_time))
    payload = {
        "method": args.method,
        "model_label": display_label or METHOD_LABELS.get(args.method, args.method),
        "folder_name": args.folder_name if args.method == "model" else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "env_path": str(env_file),
        "arrival_rate": float(args.arrival_rate),
        "max_total_tasks": int(args.max_total_tasks),
        "simulation_time_limit": float(args.simulation_time_limit),
        "random_seed": int(args.random_seed),
        "total_tasks": total_tasks,
        "finished_tasks": finished_tasks,
        "success_rate": float(success_rate),
        "dynamic_tasks": int(stats.get('dynamic_tasks', 0)),
        "makespan": effective_makespan,
        "raw_end_time": float(results.get('raw_end_time', env.current_time)),
        "termination_reason": results.get('termination_reason', 'unknown'),
        "total_inference_time": float(results.get('total_inference_time', 0.0)),
        "avg_inference_time": float(results.get('avg_inference_time', 0.0)),
        "inference_count": int(results.get('inference_count', 0)),
        "rescue_replan_count": int(results.get('rescue_replan_count', 0)),
        "planning_count": int(results.get('planning_count', results.get('replan_count', 0))),
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path),
    }
    if baseline_metrics is not None:
        payload["baseline"] = baseline_metrics
    return payload


def evaluate_greedy_baseline(env_file, args):
    with open(env_file, 'rb') as f:
        baseline_env = pickle.load(f)

    if args.quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            baseline_results = run_greedy_dynamic(
                baseline_env,
                arrival_rate=args.arrival_rate,
                max_total_tasks=args.max_total_tasks,
                simulation_time_limit=args.simulation_time_limit,
                random_seed=args.random_seed,
                verbose=False,
                max_waiting_time=args.max_waiting_time,
            )
    else:
        print("运行 Greedy baseline 对照...")
        baseline_results = run_greedy_dynamic(
            baseline_env,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
            verbose=False,
            max_waiting_time=args.max_waiting_time,
        )

    stats = baseline_env.get_task_statistics()
    total_tasks = int(stats.get('total_tasks', len(baseline_env.task_dic)))
    finished_tasks = int(stats.get('finished_tasks', 0))
    makespan = float(baseline_results.get('effective_makespan', baseline_env.current_time))
    return {
        "label": "Greedy",
        "finished_tasks": finished_tasks,
        "total_tasks": total_tasks,
        "success_rate": finished_tasks / total_tasks if total_tasks > 0 else 0.0,
        "makespan": makespan,
        "termination_reason": baseline_results.get('termination_reason', 'unknown'),
        "planning_count": int(baseline_results.get('planning_count', 0)),
    }


def metrics_overlay_lines(metrics):
    label = metrics["model_label"]
    lines = [
        f"model: {label}",
        f"tasks_done: {metrics['finished_tasks']}/{metrics['total_tasks']}",
        f"completion_rate: {metrics['success_rate'] * 100:.1f}%",
        f"makespan: {metrics['makespan']:.4f}",
    ]
    if metrics["method"] in {"model", "hrlf", "capam"}:
        lines.extend([
            f"inference: {metrics['total_inference_time']:.4f}s",
            f"termination: {metrics['termination_reason']}",
        ])
    else:
        lines.extend([
            f"planning: {metrics['total_inference_time']:.4f}s",
            f"replans: {metrics['planning_count']}",
            f"termination: {metrics['termination_reason']}",
        ])
    baseline = metrics.get("baseline")
    if baseline is not None:
        gain = baseline["makespan"] - metrics["makespan"]
        if gain >= 0:
            comparison = f"vs {baseline['label']}: {gain:.2f} shorter makespan"
        else:
            comparison = f"vs {baseline['label']}: {-gain:.2f} longer makespan"
        lines.extend([
            f"baseline: {baseline['label']} {baseline['finished_tasks']}/{baseline['total_tasks']}, {baseline['makespan']:.2f}",
            comparison,
        ])
    return lines


def write_metrics_json(metrics_path, payload):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✓ metrics 已保存: {metrics_path}")


def default_output_stem_for_method(method):
    return f"env_023_{METHOD_OUTPUT_SLUGS[method]}_clean"


def args_for_method(args, method):
    method_args = argparse.Namespace(**vars(args))
    method_args.method = method
    if method_args.output:
        return method_args
    if args.method == 'all-baselines' or args.output_stem == DEFAULT_OUTPUT_STEM:
        method_args.output_stem = default_output_stem_for_method(method)
    else:
        method_args.output_stem = f"{args.output_stem}_{METHOD_OUTPUT_SLUGS[method]}"
    return method_args


def run_visualization(args):
    env_file = Path(args.env).expanduser()
    gif_path, mp4_path, metrics_path = resolve_output_paths(args)
    display_label = args.model_label if args.method == 'model' else METHOD_LABELS[args.method]
    
    print(f"\n{'='*70}")
    print(f"动态场景评估动画生成（清洁版本）")
    print(f"  方法: {args.method}")
    print(f"  显示名: {display_label}")
    print(f"  环境: {env_file}")
    print(f"  任务到达率: {args.arrival_rate:g} 任务/分钟")
    print(f"  最大任务数: {args.max_total_tasks}")
    print(f"  时间间隔: {args.interval:g} 分钟")
    print(f"  帧率: {args.fps} fps")
    print(f"  GIF输出: {gif_path}")
    print(f"  MP4输出: {mp4_path}")
    print(f"  metrics输出: {metrics_path}")
    print(f"{'='*70}\n")
    
    checkpoint_path = None
    agent_task_paths = None
    baseline_metrics = None

    if args.method == 'model':
        from scripts.sa_bt.evaluate_my_model_dynamic import load_env as load_model_env

        env = load_model_env(env_file)
        print("加载 BASAR 模型...")
        model_bundle = load_basar_model(args.folder_name)
        checkpoint_path = model_bundle["checkpoint_path"]
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  device: {model_bundle['device']}")
        results = run_learning_model_dynamic(
            env,
            model_bundle,
            display_label=display_label,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
        )
        agent_task_paths = build_agent_paths_from_routes(env)
        model_name = args.model_label
        baseline_metrics = evaluate_greedy_baseline(env_file, args)
        print(
            f"  Greedy baseline: {baseline_metrics['finished_tasks']}/{baseline_metrics['total_tasks']}, "
            f"makespan={baseline_metrics['makespan']:.2f}"
        )
    elif args.method == 'capam':
        from scripts.sa_bt.evaluate_my_model_dynamic import load_env as load_model_env

        env = load_model_env(env_file)
        print("加载 CAPAM 模型...")
        model_bundle = load_capam_model(args.capam_folder_name)
        checkpoint_path = model_bundle["checkpoint_path"]
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  device: {model_bundle['device']}")
        results = run_learning_model_dynamic(
            env,
            model_bundle,
            display_label=display_label,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
        )
        agent_task_paths = build_agent_paths_from_routes(env)
    elif args.method == 'hrlf':
        from scripts.sa_bt.evaluate_hrlf_dynamic import load_env as load_hrlf_env

        env = load_hrlf_env(env_file)
        print("加载 HRLF 模型...")
        model_bundle = load_hrlf_model(args.hrlf_folder_name)
        checkpoint_path = model_bundle["checkpoint_path"]
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  device: {model_bundle['device']}")
        results = run_hrlf_visual_dynamic(
            env,
            model_bundle,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
        )
        agent_task_paths = build_agent_paths_from_routes(env)
    elif args.method == 'ctasd':
        from ctasd_static_planner import CTASDStaticRoutePlanner
        from scripts.sa_bt.evaluate_ctasd_dynamic import load_env as load_route_env

        env = load_route_env(env_file)
        results = run_static_route_baseline_dynamic(
            env,
            route_planner=CTASDStaticRoutePlanner(),
            display_label=display_label,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
        )
        agent_task_paths = build_agent_paths_from_routes(env)
    elif args.method == 'taco':
        from scripts.sa_bt.evaluate_ctasd_dynamic import load_env as load_route_env
        from taco_static_planner import TACOStaticRoutePlanner

        env = load_route_env(env_file)
        results = run_static_route_baseline_dynamic(
            env,
            route_planner=TACOStaticRoutePlanner(random_seed=args.random_seed),
            display_label=display_label,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
        )
        agent_task_paths = build_agent_paths_from_routes(env)
    else:
        # 加载环境
        with open(env_file, 'rb') as f:
            env = pickle.load(f)

        results = run_greedy_dynamic(
            env,
            arrival_rate=args.arrival_rate,
            max_total_tasks=args.max_total_tasks,
            simulation_time_limit=args.simulation_time_limit,
            random_seed=args.random_seed,
            verbose=not args.quiet,
            max_waiting_time=args.max_waiting_time,
        )
        model_name = display_label

    if args.method != 'greedy':
        model_name = display_label

    metrics = build_metrics_payload(
        env,
        results,
        args,
        env_file,
        gif_path,
        mp4_path,
        checkpoint_path=checkpoint_path,
        baseline_metrics=baseline_metrics,
        display_label=display_label,
    )
    
    # 创建 GIF + MP4
    create_gif_from_planning(
        env,
        results,
        gif_path,
        output_mp4_path=mp4_path,
        interval=args.interval,
        fps=args.fps,
        dpi=args.dpi,
        figsize=args.figsize,
        max_frames=args.max_frames,
        max_interval=args.max_interval,
        warmup_frames=args.warmup_frames,
        metric_overlay_frames=args.metric_overlay_frames,
        colors=args.colors,
        optimize=(args.optimize and not args.no_optimize),
        blit=args.blit,
        model_name=model_name,
        agent_task_paths_override=agent_task_paths,
        metrics_overlay_lines=metrics_overlay_lines(metrics),
    )
    write_metrics_json(metrics_path, metrics)
    
    print(f"\n{'='*70}")
    print("完成。")
    print(f"{'='*70}")
    return metrics


def main():
    """主函数"""
    default_env = SA_BT_DATASET_ROOT / 'Fixed_Tasks' / 'n10_s5_h20' / 'env_023.pkl'
    parser = argparse.ArgumentParser(description='动态场景评估动画生成（clean 版）')
    parser.add_argument(
        '--method',
        choices=['model', 'greedy', 'taco', 'ctasd', 'hrlf', 'capam', 'all-baselines'],
        default='model',
        help='评估与可视化方法',
    )
    parser.add_argument('--folder-name', type=str, default='SAVE_5', help='BASAR checkpoint 文件夹')
    parser.add_argument('--hrlf-folder-name', type=str, default='save_baseline', help='HRLF checkpoint 文件夹')
    parser.add_argument('--capam-folder-name', type=str, default='CAPAM_DYNAMIC', help='CAPAM checkpoint 文件夹')
    parser.add_argument('--model-label', type=str, default='BASAR', help='动画和 metrics 中显示的主模型名')
    parser.add_argument('--env', type=str, default=str(default_env), help='输入环境 pkl 文件')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='兼容旧参数：输出路径；会从 stem 派生 .gif/.mp4/_metrics.json',
    )
    parser.add_argument('--output-dir', type=str, default=str(SCRIPT_DIR), help='默认输出目录')
    parser.add_argument('--output-stem', type=str, default=DEFAULT_OUTPUT_STEM, help='默认输出文件名前缀')
    parser.add_argument('--arrival-rate', type=float, default=1.5, help='动态任务到达率')
    parser.add_argument('--max-total-tasks', type=int, default=50, help='最大任务数')
    parser.add_argument('--simulation-time-limit', type=float, default=200.0, help='仿真时间上限')
    parser.add_argument('--random-seed', type=int, default=45, help='随机种子')
    parser.add_argument('--max-waiting-time', type=float, default=30.0, help='最大等待时间')
    parser.add_argument('--interval', type=float, default=0.2, help='基础动画时间步长')
    parser.add_argument('--fps', type=int, default=10, help='GIF 帧率')
    parser.add_argument('--dpi', type=int, default=120, help='渲染 DPI')
    parser.add_argument('--figsize', type=float, default=10.0, help='画布边长（英寸）')
    parser.add_argument('--max-frames', type=int, default=700, help='目标最大帧数（自动调整 interval）')
    parser.add_argument('--max-interval', type=float, default=1.0, help='自动抽帧允许的最大时间步长')
    parser.add_argument('--warmup-frames', type=int, default=12, help='开场固定显示初始位置的帧数')
    parser.add_argument('--metric-overlay-frames', type=int, default=20, help='末尾指标展示帧数')
    parser.add_argument('--colors', type=int, default=96, help='GIF 调色板颜色数')
    parser.add_argument('--optimize', action='store_true', help='启用 GIF 压缩优化')
    parser.add_argument('--no-optimize', action='store_true', help='关闭 GIF 压缩优化')
    parser.add_argument('--blit', action='store_true', help='启用 blit（默认关闭，避免残影连线）')
    parser.add_argument('--quiet', action='store_true', help='减少规划器日志输出')
    args = parser.parse_args()

    if args.method == 'all-baselines':
        if args.output:
            raise ValueError('--method all-baselines 不能与 --output 同时使用；请使用 --output-dir。')
        for method in BASELINE_METHODS:
            run_visualization(args_for_method(args, method))
        return

    run_visualization(args_for_method(args, args.method))

if __name__ == '__main__':
    main()
