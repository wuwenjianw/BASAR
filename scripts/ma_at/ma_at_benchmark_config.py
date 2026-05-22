#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MA-AT dynamic benchmark 的共享配置。

设计假设：
- benchmark 复用历史论文中的 5 个经典 MA-AT 场景；
- 场景名中的 `km` 解释为总任务数；
- 为了构造动态环境，只在环境文件中写入一部分初始任务，其余任务在评估时在线到达；
- benchmark 的对比重点是“跨场景方法表现”，不是 SA-AT 那种单轴 scaling。
"""

from pathlib import Path

from project_paths import ARTIFACTS_ROOT, DOCS_ROOT, MA_AT_DYNAMIC_DATASET_ROOT


TRAITS_DIM = 5
MAX_TASK_SIZE = 2
DURATION_SCALE = 5

DEFAULT_SAMPLES_PER_CONFIG = 50
DEFAULT_RANDOM_SEED = 42

# 固定使用 40% 初始任务，剩余任务在线到达。
INITIAL_TASK_RATIO = 0.4
DEFAULT_RECOMMENDED_ARRIVAL_RATE = 2.0

METHOD_SPECS = {
    "greedy": {"label": "Greedy", "color": "#4c4c4c", "marker": "o"},
    "hrlf": {"label": "HRLF", "color": "#d95f02", "marker": "s"},
    "capam": {"label": "CAPAM", "color": "#1b9e77", "marker": "^"},
    "ours": {"label": "Ours", "color": "#1f78b4", "marker": "D"},
}

PLOT_METRIC_SPECS = {
    "makespan": {"label": "Makespan (min)", "scale": 1.0},
    "waiting_time": {"label": "Waiting Time (min)", "scale": 1.0},
    "deadline_satisfaction_rate": {"label": "Deadline Satisfaction Rate (%)", "scale": 100.0},
    "success_rate": {"label": "Success Rate (%)", "scale": 100.0},
    "total_planning_time": {"label": "Planning Time (s)", "scale": 1.0},
}

DATASET_ROOT = MA_AT_DYNAMIC_DATASET_ROOT
RESULTS_ROOT = ARTIFACTS_ROOT / "results" / "ma_at_dynamic"
DEFAULT_FIGURE_PATH = DOCS_ROOT / "figures" / "ma_at_dynamic_benchmark.pdf"


def _build_config(agents, species, total_tasks, *, arrival_rate, simulation_time_limit):
    if agents % species != 0:
        raise ValueError(
            f"agents={agents} 必须能被 species={species} 整除，"
            "否则当前生成脚本无法固定每种智能体数量。"
        )

    initial_tasks = max(1, int(round(total_tasks * INITIAL_TASK_RATIO)))
    if initial_tasks >= total_tasks:
        raise ValueError("initial_tasks 必须小于 total_tasks，当前动态 benchmark 假设需要动态任务。")

    return {
        "name": f"kn{agents}_ks{species}_km{total_tasks}",
        "agents": int(agents),
        "species": int(species),
        "traits_dim": int(TRAITS_DIM),
        "per_species_agents": int(agents // species),
        "initial_tasks": int(initial_tasks),
        "total_tasks": int(total_tasks),
        "dynamic_tasks": int(total_tasks - initial_tasks),
        "recommended_arrival_rate": float(arrival_rate),
        "simulation_time_limit": float(simulation_time_limit),
        "plot_label": f"{agents}/{species}/{total_tasks}",
    }


BENCHMARK_CONFIGS = [
    _build_config(25, 5, 100, arrival_rate=DEFAULT_RECOMMENDED_ARRIVAL_RATE, simulation_time_limit=1000.0),
    _build_config(50, 10, 200, arrival_rate=DEFAULT_RECOMMENDED_ARRIVAL_RATE, simulation_time_limit=1500.0),
    _build_config(150, 5, 500, arrival_rate=DEFAULT_RECOMMENDED_ARRIVAL_RATE, simulation_time_limit=4000.0),
    _build_config(150, 10, 500, arrival_rate=DEFAULT_RECOMMENDED_ARRIVAL_RATE, simulation_time_limit=4000.0),
    _build_config(200, 10, 500, arrival_rate=DEFAULT_RECOMMENDED_ARRIVAL_RATE, simulation_time_limit=4000.0),
]


def iter_benchmark_configs():
    return list(BENCHMARK_CONFIGS)


def get_dataset_dir(config, dataset_root=DATASET_ROOT):
    return Path(dataset_root) / config["name"]


def get_method_output_dir(method_tag, output_root=RESULTS_ROOT):
    return Path(output_root) / method_tag


def get_metric_spec(metric_name):
    return PLOT_METRIC_SPECS.get(metric_name, {"label": metric_name, "scale": 1.0})
