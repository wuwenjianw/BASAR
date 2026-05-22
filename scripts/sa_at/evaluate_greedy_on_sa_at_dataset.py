#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 SA-AT scaling benchmark 上评估 Greedy 基线。

实现策略：
- 复用 evaluate_ctasd_dynamic.py 的统一动态事件循环
- 仅替换 route planner 为 GreedyStaticRoutePlanner
- 指标统计继续走 dynamic_eval_stats.py，避免和主模型口径漂移

运行示例：
  python -u scripts/sa_at/evaluate_greedy_on_sa_at_dataset.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_eval_stats import build_summary_like_my_model, calculate_metrics_like_my_model
from scripts.sa_at.eval_resume_utils import (
    build_resume_signature,
    load_matching_results,
    save_results_snapshot,
    select_results_for_env_files,
)
from scripts.sa_at.greedy_static_planner import GreedyStaticRoutePlanner
from scripts.sa_at.sa_at_scaling_config import (
    DATASET_ROOT,
    RESULTS_ROOT,
    get_dataset_dir,
    get_method_output_dir,
    iter_scaling_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Greedy on SA-AT scaling benchmark.")
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
        help="SA-AT scaling dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_ROOT),
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Only evaluate one SA-AT config, e.g. n300_w30_h200_t1500.",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Override the recommended arrival rate for all configs.",
    )
    parser.add_argument(
        "--simulation-time-limit",
        type=float,
        default=None,
        help="Override the default simulation time limit for all configs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dynamic task generation during evaluation.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Only evaluate one environment file. Use with --config-name, e.g. env_15.pkl.",
    )
    parser.add_argument(
        "--dynamic-deadline-buffer",
        type=float,
        default=None,
        help="Override dynamic task deadline buffer.",
    )
    parser.add_argument(
        "--dynamic-deadline-horizon",
        type=float,
        default=None,
        help="Override dynamic task deadline horizon.",
    )
    parser.add_argument(
        "--dynamic-urgent-probability",
        type=float,
        default=None,
        help="Override dynamic task urgent probability.",
    )
    parser.add_argument(
        "--dynamic-task-profile",
        default=None,
        help="Override profile label written to result rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-env results when available.",
    )
    args = parser.parse_args()
    if args.env_file and not args.config_name:
        parser.error("--env-file 需要和 --config-name 一起使用。")
    return args


def resolve_env_files(input_dir, env_file=None):
    if env_file is None:
        return sorted(input_dir.glob("env_*.pkl"))

    selected = Path(env_file)
    if not selected.is_absolute():
        selected = input_dir / env_file
    if not selected.exists():
        return []
    return [selected]


def evaluate_single_env(
    env_path,
    route_planner,
    arrival_rate,
    max_total_tasks,
    simulation_time_limit,
    seed,
    dynamic_task_options=None,
):
    from scripts.sa_bt.evaluate_ctasd_dynamic import load_env, run_ctasd_dynamic

    env = load_env(env_path)
    initial_task_count = len(env.task_dic)

    results = run_ctasd_dynamic(
        env=env,
        route_planner=route_planner,
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=seed,
        dynamic_task_options=dynamic_task_options,
    )
    metrics = calculate_metrics_like_my_model(
        env=env,
        results=results,
        initial_task_count=initial_task_count,
        dataset_type="Fixed_Tasks",
    )
    return metrics, results


def main():
    args = parse_args()

    print("=" * 80)
    print("Greedy on SA-AT scaling benchmark")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    route_planner = GreedyStaticRoutePlanner()
    output_dir = get_method_output_dir("greedy", args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"evaluation_log_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    log_file.write("Greedy on SA-AT scaling benchmark\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")

    configs = iter_scaling_configs()
    if args.config_name is not None:
        configs = [config for config in configs if config["name"] == args.config_name]
        if not configs:
            raise ValueError(f"未知 SA-AT 配置: {args.config_name}")

    for config_idx, config in enumerate(configs, 1):
        config_name = config["name"]
        arrival_rate = args.arrival_rate
        if arrival_rate is None:
            arrival_rate = config["recommended_arrival_rate"]
        simulation_time_limit = args.simulation_time_limit
        if simulation_time_limit is None:
            simulation_time_limit = config["simulation_time_limit"]
        dynamic_task_options = dict(config.get("dynamic_task_options", {}))
        if args.dynamic_deadline_buffer is not None:
            dynamic_task_options["dynamic_deadline_buffer"] = args.dynamic_deadline_buffer
        if args.dynamic_deadline_horizon is not None:
            dynamic_task_options["dynamic_deadline_horizon"] = args.dynamic_deadline_horizon
        if args.dynamic_urgent_probability is not None:
            dynamic_task_options["dynamic_urgent_probability"] = args.dynamic_urgent_probability
        dynamic_task_profile = args.dynamic_task_profile or config.get("dynamic_task_profile", "default")

        input_dir = get_dataset_dir(config, args.dataset_root)
        env_files = resolve_env_files(input_dir, args.env_file)

        if not env_files:
            msg = f"⚠ {config_name}: 未找到环境文件 ({input_dir})\n"
            log_file.write(msg)
            print(msg, end="")
            continue

        header = f"\n{'=' * 80}\n配置 {config_idx}/{len(configs)}: {config_name}\n{'=' * 80}\n"
        header += f"  智能体: {config['agents']}\n"
        header += f"  仓库: {config['warehouses']}\n"
        header += f"  初始任务: {config['initial_tasks']}\n"
        header += f"  总任务: {config['total_tasks']}\n"
        header += f"  到达率: {arrival_rate} 任务/分钟\n"
        header += f"  动态任务profile: {dynamic_task_profile}\n"
        header += f"  动态任务deadline参数: {dynamic_task_options}\n"
        header += f"  仿真上限: {simulation_time_limit} 分钟\n"
        header += f"  样本数: {len(env_files)}\n\n"
        log_file.write(header)
        print(header, end="")

        results_json_path = output_dir / f"{config_name}_results.json"
        resume_signature = build_resume_signature(
            config_name=config_name,
            arrival_rate=arrival_rate,
            simulation_time_limit=simulation_time_limit,
            method_tag="greedy",
            dynamic_task_options=dynamic_task_options,
        )
        results_by_env = {}
        if args.resume:
            results_by_env, resume_warning = load_matching_results(results_json_path, resume_signature)
            if resume_warning is not None:
                log_file.write(resume_warning + "\n")
                log_file.flush()
                print(resume_warning)

        for env_file in env_files:
            if args.resume and env_file.name in results_by_env:
                line = f"  ↷ {env_file.name}: 已有结果，跳过\n"
                log_file.write(line)
                log_file.flush()
                print(line, end="")
                continue

            try:
                metrics, results = evaluate_single_env(
                    env_path=env_file,
                    route_planner=route_planner,
                    arrival_rate=arrival_rate,
                    max_total_tasks=config["total_tasks"],
                    simulation_time_limit=simulation_time_limit,
                    seed=args.seed,
                    dynamic_task_options=dynamic_task_options,
                )
                metrics["env_file"] = env_file.name
                metrics["config_name"] = config_name
                metrics["dataset_type"] = "SA_AT_Scaling"
                metrics["arrival_rate"] = arrival_rate
                metrics["simulation_time_limit"] = simulation_time_limit
                metrics["dynamic_task_options"] = dynamic_task_options
                metrics["dynamic_task_profile"] = dynamic_task_profile
                metrics["method"] = "greedy"
                metrics["rescue_replan_count"] = int(results.get("rescue_replan_count", 0))
                results_by_env[env_file.name] = metrics
                save_results_snapshot(output_dir, config_name, results_by_env)

                line = (
                    f"  ✓ {env_file.name}: "
                    f"完成={metrics['finished_tasks']}/{metrics['total_tasks']} "
                    f"({metrics['success_rate'] * 100:.1f}%), "
                    f"makespan={metrics['makespan']:.1f}min, "
                    f"规划耗时={metrics['total_planning_time']:.3f}s\n"
                )
                log_file.write(line)
                log_file.flush()
                print(line, end="")
            except Exception as exc:
                line = f"  ✗ {env_file.name}: 失败 - {exc}\n"
                log_file.write(line)
                log_file.flush()
                print(line, end="")

        config_results = select_results_for_env_files(env_files, results_by_env)
        if not config_results:
            continue

        summary = build_summary_like_my_model(config_name, config_results)
        log_file.write(summary)
        print(summary, end="")

    tail = f"\n{'=' * 80}\n评估完成\n{'=' * 80}\n"
    tail += f"结果目录: {output_dir}\n"
    tail += f"日志文件: {log_path}\n"
    tail += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    tail += "=" * 80 + "\n"
    log_file.write(tail)
    print(tail, end="")
    log_file.close()


if __name__ == "__main__":
    main()
