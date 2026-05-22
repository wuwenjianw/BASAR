#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 TACO 基线评估动态 SA-BT 场景（独立文件）

对齐 evaluate_my_model_dynamic.py：
- 事件驱动推进逻辑一致
- 终止条件与恢复机制一致
- 指标统计与日志格式一致
- 唯一区别：动作由 TACO 风格的静态 route 规划器产生

运行：
  python -u scripts/sa_bt/evaluate_taco_dynamic.py
"""

from datetime import datetime
from pathlib import Path
import json
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_eval_stats import build_summary_like_my_model
from project_paths import ARTIFACTS_ROOT, SA_BT_DATASET_ROOT
from scripts.sa_bt.evaluate_ctasd_dynamic import evaluate_single_env
from taco_static_planner import TACOStaticRoutePlanner


def main():
    print("=" * 80)
    print("TACO 动态场景评估")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    route_planner = TACOStaticRoutePlanner()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = ARTIFACTS_ROOT / "results" / "taco_dynamic"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_base_dir / f"evaluation_log_{timestamp}.txt"
    log_file = open(log_file_path, "w", encoding="utf-8")

    log_file.write("TACO 动态场景评估日志\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n\n")

    dataset_configs = {
        "Fixed_Tasks": [
            {"name": "n15_s5_h30", "agents": 15, "species": 5, "initial_tasks": 30, "total_tasks": 100, "recommended_arrival_rate": 3},
            {"name": "n20_s5_h40", "agents": 20, "species": 5, "initial_tasks": 40, "total_tasks": 100, "recommended_arrival_rate": 3},
            {"name": "n20_s5_h50", "agents": 20, "species": 5, "initial_tasks": 50, "total_tasks": 100, "recommended_arrival_rate": 3},
            {"name": "n30_s5_h60", "agents": 30, "species": 5, "initial_tasks": 60, "total_tasks": 100, "recommended_arrival_rate": 3},
        ],
        "Fixed_Makespan": [
            {"name": "n10_s5_t120", "agents": 10, "species": 5, "initial_tasks": 30, "total_tasks": 120, "recommended_arrival_rate": 2},
            {"name": "n15_s5_t200", "agents": 15, "species": 5, "initial_tasks": 50, "total_tasks": 200, "recommended_arrival_rate": 2},
            {"name": "n20_s5_t240", "agents": 20, "species": 5, "initial_tasks": 60, "total_tasks": 240, "recommended_arrival_rate": 2},
            {"name": "n30_s5_t300", "agents": 30, "species": 5, "initial_tasks": 80, "total_tasks": 300, "recommended_arrival_rate": 2},
        ],
    }

    for dataset_type in ["Fixed_Tasks", "Fixed_Makespan"]:
        msg = f"\n{'#'*80}\n处理数据集: {dataset_type}\n{'#'*80}\n"
        log_file.write(msg)
        print(msg, end="")

        configs = dataset_configs[dataset_type]
        for config_idx, config in enumerate(configs, 1):
            config_name = config["name"]
            arrival_rate = config["recommended_arrival_rate"]
            max_total_tasks = config["total_tasks"]

            input_dir = SA_BT_DATASET_ROOT / dataset_type / config_name
            output_dir = output_base_dir / dataset_type
            output_dir.mkdir(parents=True, exist_ok=True)

            env_files = sorted(input_dir.glob("env_*.pkl"))
            if not env_files:
                msg = f"⚠ {config_name}: 未找到环境文件\n"
                log_file.write(msg)
                print(msg, end="")
                continue

            config_msg = f"\n{'='*80}\n配置 {config_idx}/{len(configs)}: {dataset_type}/{config_name}\n{'='*80}\n"
            config_msg += f"  智能体: {config['agents']}, 种类: {config['species']}\n"
            config_msg += f"  初始任务: {config['initial_tasks']}, 最大总任务: {max_total_tasks}\n"
            config_msg += f"  到达率: {arrival_rate} 任务/分钟\n"
            config_msg += f"  样本数: {len(env_files)}\n\n"
            log_file.write(config_msg)
            print(config_msg, end="")

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
                    metrics["env_file"] = env_file.name
                    metrics["config_name"] = config_name
                    metrics["dataset_type"] = dataset_type
                    metrics["arrival_rate"] = arrival_rate
                    config_results.append(metrics)

            if not config_results:
                continue

            results_pkl = output_dir / f"{config_name}_results.pkl"
            with open(results_pkl, "wb") as f:
                pickle.dump(config_results, f)

            results_json = output_dir / f"{config_name}_results.json"
            with open(results_json, "w", encoding="utf-8") as f:
                json.dump(config_results, f, indent=2)

            summary_msg = build_summary_like_my_model(config_name, config_results)

            log_file.write(summary_msg)
            print(summary_msg, end="")

    end_msg = f"\n{'='*80}\n评估完成！\n{'='*80}\n"
    end_msg += f"结果保存在: {output_base_dir}/\n"
    end_msg += f"日志文件: {log_file_path}\n"
    end_msg += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    end_msg += "=" * 80 + "\n"
    log_file.write(end_msg)
    print(end_msg)
    log_file.close()


if __name__ == "__main__":
    main()
