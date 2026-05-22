#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 MA-AT dynamic benchmark 上评估 Greedy 基线。"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_eval_stats import build_summary_like_my_model, calculate_metrics_like_my_model
from scripts.ma_at.ma_at_benchmark_config import (
    DATASET_ROOT,
    RESULTS_ROOT,
    get_dataset_dir,
    get_method_output_dir,
    iter_benchmark_configs,
)
from scripts.sa_at.eval_resume_utils import (
    build_resume_signature,
    load_matching_results,
    save_results_snapshot,
    select_results_for_env_files,
)
from scripts.sa_at.greedy_static_planner import GreedyStaticRoutePlanner


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Greedy on the MA-AT dynamic benchmark.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="MA-AT dynamic dataset root directory.")
    parser.add_argument("--output-root", default=str(RESULTS_ROOT), help="Root directory for evaluation outputs.")
    parser.add_argument("--arrival-rate", type=float, default=None, help="Override the recommended arrival rate.")
    parser.add_argument(
        "--simulation-time-limit",
        type=float,
        default=None,
        help="Override the default simulation time limit.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dynamic task generation during evaluation.")
    parser.add_argument("--config-name", default=None, help="Only evaluate one MA-AT config, e.g. kn25_ks5_km100.")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Only evaluate one environment file. Use with --config-name, e.g. env_15.pkl.",
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


def evaluate_single_env(env_path, route_planner, arrival_rate, max_total_tasks, simulation_time_limit, seed):
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
    )
    metrics = calculate_metrics_like_my_model(
        env=env,
        results=results,
        initial_task_count=initial_task_count,
        dataset_type="MA_AT_Dynamic",
    )
    return metrics, results


def main():
    args = parse_args()

    print("=" * 80)
    print("Greedy on MA-AT dynamic benchmark")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    route_planner = GreedyStaticRoutePlanner()
    output_dir = get_method_output_dir("greedy", args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"evaluation_log_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("Greedy on MA-AT dynamic benchmark\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")

    configs = iter_benchmark_configs()
    if args.config_name is not None:
        configs = [config for config in configs if config["name"] == args.config_name]
        if not configs:
            raise ValueError(f"未知 MA-AT 配置: {args.config_name}")

    for config_idx, config in enumerate(configs, 1):
        config_name = config["name"]
        arrival_rate = args.arrival_rate if args.arrival_rate is not None else config["recommended_arrival_rate"]
        simulation_time_limit = (
            args.simulation_time_limit if args.simulation_time_limit is not None else config["simulation_time_limit"]
        )

        input_dir = get_dataset_dir(config, args.dataset_root)
        env_files = resolve_env_files(input_dir, args.env_file)
        if not env_files:
            msg = f"⚠ {config_name}: 未找到环境文件 ({input_dir})\n"
            log_file.write(msg)
            print(msg, end="")
            continue

        header = f"\n{'=' * 80}\n配置 {config_idx}/{len(configs)}: {config_name}\n{'=' * 80}\n"
        header += f"  智能体: {config['agents']}\n"
        header += f"  种类: {config['species']}\n"
        header += f"  初始任务: {config['initial_tasks']}\n"
        header += f"  总任务: {config['total_tasks']}\n"
        header += f"  到达率: {arrival_rate} 任务/分钟\n"
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
                )
                metrics["env_file"] = env_file.name
                metrics["config_name"] = config_name
                metrics["dataset_type"] = "MA_AT_Dynamic"
                metrics["arrival_rate"] = arrival_rate
                metrics["simulation_time_limit"] = simulation_time_limit
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
                if len(env_files) == 1:
                    line = line.rstrip("\n")
                    line += (
                        f", 终止={results.get('termination_reason', 'unknown')}"
                        f", 动态生成={results.get('dynamic_tasks_generated', 0)}"
                        f", raw_end={results.get('raw_end_time', 0.0):.1f}min\n"
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
