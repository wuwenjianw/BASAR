#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 MA-AT 动态 benchmark 的初始环境。"""

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv
from scripts.ma_at.ma_at_benchmark_config import (
    DATASET_ROOT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLES_PER_CONFIG,
    DURATION_SCALE,
    MAX_TASK_SIZE,
    TRAITS_DIM,
    get_dataset_dir,
    iter_benchmark_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MA-AT dynamic benchmark datasets.")
    parser.add_argument(
        "--output-root",
        default=str(DATASET_ROOT),
        help="Directory for generated MA-AT dynamic environments.",
    )
    parser.add_argument(
        "--samples-per-config",
        type=int,
        default=DEFAULT_SAMPLES_PER_CONFIG,
        help="Number of environments to generate for each benchmark config.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base seed; each environment uses seed_base + env_index.",
    )
    return parser.parse_args()


def build_env(config, seed):
    return TaskEnv(
        per_species_range=(config["per_species_agents"], config["per_species_agents"]),
        species_range=(config["species"], config["species"]),
        tasks_range=(config["initial_tasks"], config["initial_tasks"]),
        traits_dim=TRAITS_DIM,
        max_task_size=MAX_TASK_SIZE,
        duration_scale=DURATION_SCALE,
        seed=seed,
        plot_figure=False,
        single_skill=False,
        binary_task=False,
    )


def remove_existing_env_files(target_dir):
    for env_path in Path(target_dir).glob("env_*.pkl"):
        env_path.unlink()


def write_manifest(output_root, samples_per_config):
    manifest = {
        "benchmark": "MA-AT dynamic benchmark",
        "description": (
            "Each classic MA-AT scenario stores only the initial task set. "
            "Remaining tasks are injected online during evaluation."
        ),
        "samples_per_config": int(samples_per_config),
        "configs": iter_benchmark_configs(),
    }
    with open(Path(output_root) / "benchmark_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MA-AT dynamic benchmark 数据集生成")
    print("=" * 80)
    print(f"输出目录: {output_root}")
    print(f"每个配置样本数: {args.samples_per_config}")
    print(f"seed_base: {args.seed_base}\n")

    write_manifest(output_root, args.samples_per_config)

    for config_idx, config in enumerate(iter_benchmark_configs(), 1):
        config_dir = get_dataset_dir(config, output_root)
        config_dir.mkdir(parents=True, exist_ok=True)
        remove_existing_env_files(config_dir)

        with open(config_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"{'=' * 80}")
        print(f"配置 {config_idx}/{len(iter_benchmark_configs())}: {config['name']}")
        print(f"  智能体数: {config['agents']}")
        print(f"  种类数: {config['species']}")
        print(f"  初始任务数: {config['initial_tasks']}")
        print(f"  总任务数: {config['total_tasks']} (动态任务 {config['dynamic_tasks']})")
        print(f"  推荐到达率: {config['recommended_arrival_rate']} 任务/分钟")
        print(f"  仿真上限: {config['simulation_time_limit']} 分钟")
        print(f"  输出目录: {config_dir}")

        for env_idx in range(args.samples_per_config):
            seed = args.seed_base + env_idx
            env = build_env(config, seed)
            with open(config_dir / f"env_{env_idx}.pkl", "wb") as f:
                pickle.dump(env, f)

            if (env_idx + 1) % 10 == 0 or env_idx + 1 == args.samples_per_config:
                print(f"  已生成: {env_idx + 1}/{args.samples_per_config}")

        print(f"  ✓ 完成 {config['name']}\n")

    print("=" * 80)
    print("生成完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
