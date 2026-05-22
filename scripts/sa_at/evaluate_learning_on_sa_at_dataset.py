#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在 SA-AT scaling benchmark 上评估学习方法。

支持方法：
- ours  : 当前自定义动态模型
- hrlf  : AttentionNet / save_baseline
- capam : CAPAM 动态模型

运行示例：
  python -u scripts/sa_at/evaluate_learning_on_sa_at_dataset.py --method ours
  python -u scripts/sa_at/evaluate_learning_on_sa_at_dataset.py --method hrlf
  python -u scripts/sa_at/evaluate_learning_on_sa_at_dataset.py --method capam --folder-name CAPAM_DYNAMIC
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_eval_stats import build_summary_like_my_model, calculate_metrics_like_my_model
from parameters import EnvParams, SaverParams, TrainParams
from project_paths import ensure_checkpoint_exists, set_saver_paths
from scripts.sa_at.eval_resume_utils import (
    build_resume_signature,
    load_matching_results,
    save_results_snapshot,
    select_results_for_env_files,
)
from scripts.sa_at.sa_at_scaling_config import (
    DATASET_ROOT,
    RESULTS_ROOT,
    get_dataset_dir,
    get_method_output_dir,
    iter_scaling_configs,
)


METHOD_DEFAULT_FOLDERS = {
    "ours": SaverParams.FOLDER_NAME,
    "hrlf": "save_baseline",
    "capam": "CAPAM_DYNAMIC",
}


def parse_args(default_method=None):
    parser = argparse.ArgumentParser(description="Evaluate learning baselines on SA-AT scaling benchmark.")
    if default_method is None:
        parser.add_argument(
            "--method",
            choices=sorted(METHOD_DEFAULT_FOLDERS.keys()),
            required=True,
            help="Learning method to evaluate.",
        )
    parser.add_argument(
        "--folder-name",
        default=None,
        help="Checkpoint folder under artifacts/models/<folder-name>. Uses a method-specific default when omitted.",
    )
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
        "--config-name",
        default=None,
        help="Only evaluate one SA-AT config, e.g. n300_w30_h200_t300.",
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
    if default_method is not None:
        args.method = default_method
    if args.folder_name is None:
        args.folder_name = METHOD_DEFAULT_FOLDERS[args.method]
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


def infer_dynamic_input_dims():
    """按 evaluate_my_model_dynamic.py 的方式推断输入维度。"""
    from env.task_env import TaskEnv

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


def load_learning_method(method, folder_name, device):
    """根据方法类型加载模型和对应的动态 runner。"""
    import torch

    from attention import AttentionNet
    from dynamic_worker import create_dynamic_model
    from scripts.sa_bt.evaluate_hrlf_dynamic import load_env as load_hrlf_env
    from scripts.sa_bt.evaluate_hrlf_dynamic import run_hrlf_dynamic
    from scripts.sa_bt.evaluate_my_model_dynamic import (
        extract_model_state_dict,
        load_env as load_dynamic_env,
        run_model_dynamic,
    )

    checkpoint_path = ensure_checkpoint_exists(folder_name, method_label=method.upper())
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if method == "hrlf":
        agent_input_dim = 6 + EnvParams.TRAIT_DIM
        task_input_dim = 5 + 2 * EnvParams.TRAIT_DIM
        network = AttentionNet(
            agent_input_dim=agent_input_dim,
            task_input_dim=task_input_dim,
            embedding_dim=TrainParams.EMBEDDING_DIM,
        ).to(device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        network.load_state_dict(state_dict, strict=False)
        network.eval()
        return {
            "method_tag": "hrlf",
            "display_name": "HRLF",
            "checkpoint_path": str(checkpoint_path),
            "network": network,
            "env_loader": load_hrlf_env,
            "runner": run_hrlf_dynamic,
        }

    model_name = "myself" if method == "ours" else "capam"
    TrainParams.MODEL_NAME = model_name
    if method == "capam":
        set_saver_paths(SaverParams, folder_name)

    agent_input_dim, task_input_dim = infer_dynamic_input_dims()
    network = create_dynamic_model(
        agent_input_dim=agent_input_dim,
        task_input_dim=task_input_dim,
        embedding_dim=TrainParams.EMBEDDING_DIM,
        device=device,
        model_name=model_name,
    )
    network.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    network.eval()

    return {
        "method_tag": method,
        "display_name": "Ours" if method == "ours" else "CAPAM",
        "checkpoint_path": str(checkpoint_path),
        "network": network,
        "env_loader": load_dynamic_env,
        "runner": run_model_dynamic,
    }


def evaluate_single_env(
    env_path,
    method_bundle,
    device,
    arrival_rate,
    max_total_tasks,
    simulation_time_limit,
    seed,
    dynamic_task_options=None,
):
    env = method_bundle["env_loader"](env_path)
    initial_task_count = len(env.task_dic)

    results = method_bundle["runner"](
        env=env,
        global_network=method_bundle["network"],
        device=device,
        max_total_tasks=max_total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        random_seed=seed,
        sampling=False,
        dynamic_task_options=dynamic_task_options,
    )
    metrics = calculate_metrics_like_my_model(
        env=env,
        results=results,
        initial_task_count=initial_task_count,
        dataset_type="Fixed_Tasks",
    )
    return metrics, results


def main(default_method=None):
    args = parse_args(default_method=default_method)

    import torch

    device = torch.device(
        "cuda" if torch.cuda.is_available() and TrainParams.USE_GPU_GLOBAL else "cpu"
    )
    method_bundle = load_learning_method(args.method, args.folder_name, device)

    output_dir = get_method_output_dir(method_bundle["method_tag"], args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{method_bundle['display_name']} on SA-AT scaling benchmark")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"checkpoint: {method_bundle['checkpoint_path']}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"evaluation_log_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"{method_bundle['display_name']} on SA-AT scaling benchmark\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"设备: {device}\n")
    log_file.write(f"checkpoint: {method_bundle['checkpoint_path']}\n")
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
            method_tag=method_bundle["method_tag"],
            model_folder=args.folder_name,
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
                    method_bundle=method_bundle,
                    device=device,
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
                metrics["method"] = method_bundle["method_tag"]
                metrics["model_folder"] = args.folder_name
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
