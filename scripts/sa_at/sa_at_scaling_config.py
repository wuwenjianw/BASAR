#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SA-AT 大规模 scaling benchmark 的共享配置。

场景定义对应用户当前论文口径：
- 第 1 个数：智能体数 n
- 第 2 个数：仓库数 w
- 第 3 个数：初始任务数 h
- 第 4 个数：总任务数 t

注意：
- 当前 TaskEnv 没有单独的“仓库”维度，因此这里将 w 映射为 species/depot 数。
- 技能维度 traits_dim 仍固定为 5，这意味着 30 个 depot 会复用 5 个 one-hot 技能。
- 数据集文件只保存初始任务 h；其余动态任务在评估时按 arrival_rate 在线生成，直到总任务数 t。
"""

from pathlib import Path

from project_paths import ARTIFACTS_ROOT, DOCS_ROOT, SA_AT_SCALING_DATASET_ROOT


AGENTS = 300
WAREHOUSES = 30
INITIAL_TASKS = 200
TOTAL_TASK_OPTIONS = (300, 500, 800, 1500, 2000)

TRAITS_DIM = 5
MAX_TASK_SIZE = 2
DURATION_SCALE = 5

DEFAULT_SAMPLES_PER_CONFIG = 50
DEFAULT_RANDOM_SEED = 42

# SA-AT scaling 的基础目标是只改变 total_tasks，所有配置共享同一组初始环境。
# 两个最大动态规模采用显式 relaxed profile，避免动态到达过密和截止时间过紧
# 掩盖方法在可扩展在线规划上的比较。
DEFAULT_RECOMMENDED_ARRIVAL_RATE = 2.0
DEFAULT_SIMULATION_TIME_LIMIT = 10000.0
SHARED_INITIAL_ENV_DIRNAME = "_shared_initial_envs"

DEFAULT_DYNAMIC_TASK_OPTIONS = {
    "dynamic_deadline_buffer": 8.0,
    "dynamic_deadline_horizon": 17.0,
    "dynamic_urgent_probability": 0.88,
}

RELAXED_DYNAMIC_TASK_PROFILES = {
    1500: {
        "recommended_arrival_rate": 1.3,
        "dynamic_deadline_buffer": 20.0,
        "dynamic_deadline_horizon": 60.0,
        "dynamic_urgent_probability": 0.65,
    },
    2000: {
        "recommended_arrival_rate": 1.0,
        "dynamic_deadline_buffer": 24.0,
        "dynamic_deadline_horizon": 75.0,
        "dynamic_urgent_probability": 0.60,
    },
}

DATASET_ROOT = SA_AT_SCALING_DATASET_ROOT
RESULTS_ROOT = ARTIFACTS_ROOT / "results" / "sa_at_scaling"
DEFAULT_FIGURE_PATH = DOCS_ROOT / "figures" / "sa_at_scaling_comparison.pdf"

METHOD_SPECS = {
    "greedy": {"label": "Greedy", "color": "#4c4c4c", "marker": "o"},
    "hrlf": {"label": "HRLF", "color": "#d95f02", "marker": "s"},
    "capam": {"label": "CAPAM", "color": "#1b9e77", "marker": "^"},
    "ours": {"label": "Ours", "color": "#1f78b4", "marker": "D"},
}

PLOT_METRIC_SPECS = {
    "success_rate": {"label": "Success Rate (%)", "scale": 100.0},
    "finished_tasks": {"label": "Finished Tasks", "scale": 1.0},
    "total_planning_time": {"label": "Planning Time (s)", "scale": 1.0},
    "avg_planning_time": {"label": "Avg Planning Time (s)", "scale": 1.0},
    "makespan": {"label": "Makespan (min)", "scale": 1.0},
}


if AGENTS % WAREHOUSES != 0:
    raise ValueError(
        f"AGENTS={AGENTS} 必须能被 WAREHOUSES={WAREHOUSES} 整除，"
        "否则当前生成脚本无法固定每个 depot 的智能体数。"
    )


def build_scaling_config(total_tasks):
    """构造单个 SA-AT scaling 配置。"""
    total_tasks = int(total_tasks)
    if total_tasks < INITIAL_TASKS:
        raise ValueError(
            f"total_tasks={total_tasks} 不能小于 initial_tasks={INITIAL_TASKS}。"
        )

    dynamic_tasks = total_tasks - INITIAL_TASKS
    relaxed_profile = RELAXED_DYNAMIC_TASK_PROFILES.get(total_tasks, {})
    arrival_rate = relaxed_profile.get(
        "recommended_arrival_rate",
        DEFAULT_RECOMMENDED_ARRIVAL_RATE,
    ) if dynamic_tasks > 0 else 0.0
    dynamic_task_options = dict(DEFAULT_DYNAMIC_TASK_OPTIONS)
    dynamic_task_options.update(
        {
            key: value
            for key, value in relaxed_profile.items()
            if key != "recommended_arrival_rate"
        }
    )

    return {
        "name": f"n{AGENTS}_w{WAREHOUSES}_h{INITIAL_TASKS}_t{total_tasks}",
        "agents": AGENTS,
        "warehouses": WAREHOUSES,
        "species": WAREHOUSES,
        "traits_dim": TRAITS_DIM,
        "per_species_agents": AGENTS // WAREHOUSES,
        "initial_tasks": INITIAL_TASKS,
        "total_tasks": total_tasks,
        "dynamic_tasks": dynamic_tasks,
        "recommended_arrival_rate": float(round(arrival_rate, 3)),
        "dynamic_task_options": dynamic_task_options,
        "dynamic_task_profile": "relaxed" if relaxed_profile else "default",
        "simulation_time_limit": float(DEFAULT_SIMULATION_TIME_LIMIT),
    }


SCALING_CONFIGS = [build_scaling_config(total_tasks) for total_tasks in TOTAL_TASK_OPTIONS]


def iter_scaling_configs():
    """返回 benchmark 配置列表。"""
    return list(SCALING_CONFIGS)


def get_dataset_dir(config, dataset_root=DATASET_ROOT):
    """返回某个配置对应的数据集目录。"""
    return Path(dataset_root) / config["name"]


def get_shared_initial_env_dir(dataset_root=DATASET_ROOT):
    """返回所有 SA-AT 配置共享的初始环境目录。"""
    base_name = f"n{AGENTS}_w{WAREHOUSES}_h{INITIAL_TASKS}"
    return Path(dataset_root) / SHARED_INITIAL_ENV_DIRNAME / base_name


def get_method_output_dir(method_tag, output_root=RESULTS_ROOT):
    """返回某个方法的结果目录。"""
    return Path(output_root) / method_tag


def get_metric_spec(metric_name):
    """返回绘图指标的显示配置。"""
    return PLOT_METRIC_SPECS.get(metric_name, {"label": metric_name, "scale": 1.0})
