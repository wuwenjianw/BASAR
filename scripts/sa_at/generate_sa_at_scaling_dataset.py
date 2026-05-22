#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 SA-AT 大规模 scaling benchmark 的初始环境。

重要说明：
- 本脚本只生成初始 h=200 个任务的环境文件。
- 评估时再根据配置把其余动态任务在线补到总任务数 t。
- 脚本只负责“写代码可运行”，不会自动替你执行生成。

运行示例：
  python -u scripts/sa_at/generate_sa_at_scaling_dataset.py
  python -u scripts/sa_at/generate_sa_at_scaling_dataset.py --samples-per-config 20
  python -u scripts/sa_at/generate_sa_at_scaling_dataset.py --output-root data/testsets/sa_at_scaling
"""

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.task_env import TaskEnv
from scripts.sa_at.sa_at_scaling_config import (
    DATASET_ROOT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLES_PER_CONFIG,
    DURATION_SCALE,
    MAX_TASK_SIZE,
    TRAITS_DIM,
    get_dataset_dir,
    get_shared_initial_env_dir,
    iter_scaling_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SA-AT scaling benchmark datasets.")
    parser.add_argument(
        "--output-root",
        default=str(DATASET_ROOT),
        help="Directory for generated SA-AT scaling environments.",
    )
    parser.add_argument(
        "--samples-per-config",
        type=int,
        default=DEFAULT_SAMPLES_PER_CONFIG,
        help="Number of environments to generate for each scaling config.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base seed; each environment uses seed_base + env_index.",
    )
    return parser.parse_args()


def build_env(config, seed):
    """构造单个 SA-AT scaling 环境。"""
    env = TaskEnv(
        per_species_range=(config["per_species_agents"], config["per_species_agents"]),
        species_range=(config["species"], config["species"]),
        tasks_range=(config["initial_tasks"], config["initial_tasks"]),
        traits_dim=config["traits_dim"],
        max_task_size=MAX_TASK_SIZE,
        duration_scale=DURATION_SCALE,
        seed=seed,
        plot_figure=False,
        single_skill=True,
        binary_task=False,
    )
    return env


def remove_existing_env_files(target_dir):
    for env_path in Path(target_dir).glob("env_*.pkl"):
        env_path.unlink()


def link_or_copy(src_path, dst_path):
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def write_manifest(output_root, samples_per_config):
    """写出 benchmark 元数据，方便后续生成和评估对齐。"""
    manifest_path = Path(output_root) / "benchmark_manifest.json"
    manifest = {
        "benchmark": "SA-AT scaling",
        "description": (
            "All configs share exactly the same initial environments. "
            "Only total_tasks changes across configs; dynamic tasks are injected online during evaluation."
        ),
        "samples_per_config": int(samples_per_config),
        "shared_initial_env_dir": str(get_shared_initial_env_dir(output_root)),
        "configs": iter_scaling_configs(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def generate_shared_initial_envs(shared_dir, reference_config, samples_per_config, seed_base):
    """只生成一次共享初始环境，供所有 total_tasks 配置复用。"""
    shared_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_env_files(shared_dir)

    shared_meta = {
        "name": shared_dir.name,
        "agents": reference_config["agents"],
        "warehouses": reference_config["warehouses"],
        "species": reference_config["species"],
        "initial_tasks": reference_config["initial_tasks"],
        "samples_per_config": int(samples_per_config),
        "seed_base": int(seed_base),
        "description": "Shared initial environments used by every SA-AT total_tasks config.",
    }
    with open(shared_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(shared_meta, f, indent=2)

    print(f"{'=' * 80}")
    print("生成共享初始环境")
    print(f"{'=' * 80}")
    print(f"目录: {shared_dir}")
    print(f"智能体数: {reference_config['agents']}")
    print(f"仓库数: {reference_config['warehouses']}")
    print(f"初始任务数: {reference_config['initial_tasks']}")

    for env_idx in range(samples_per_config):
        seed = seed_base + env_idx
        env = build_env(reference_config, seed)
        output_file = shared_dir / f"env_{env_idx}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(env, f)

        if (env_idx + 1) % 10 == 0 or env_idx + 1 == samples_per_config:
            print(f"  已生成: {env_idx + 1}/{samples_per_config}")

    print("  ✓ 共享初始环境生成完成\n")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SA-AT scaling 数据集生成")
    print("=" * 80)
    print(f"输出目录: {output_root}")
    print(f"每个配置样本数: {args.samples_per_config}")
    print(f"seed_base: {args.seed_base}\n")

    write_manifest(output_root, args.samples_per_config)
    configs = iter_scaling_configs()
    reference_config = configs[0]
    shared_dir = get_shared_initial_env_dir(output_root)
    generate_shared_initial_envs(
        shared_dir=shared_dir,
        reference_config=reference_config,
        samples_per_config=args.samples_per_config,
        seed_base=args.seed_base,
    )

    for config in configs:
        config_dir = get_dataset_dir(config, output_root)
        config_dir.mkdir(parents=True, exist_ok=True)
        remove_existing_env_files(config_dir)

        print(f"{'=' * 80}")
        print(f"配置: {config['name']}")
        print(f"  智能体数: {config['agents']}")
        print(f"  仓库数: {config['warehouses']} (映射为 species/depot 数)")
        print(f"  初始任务数: {config['initial_tasks']}")
        print(f"  总任务数: {config['total_tasks']} (其中动态任务 {config['dynamic_tasks']})")
        print(f"  推荐到达率: {config['recommended_arrival_rate']} 任务/分钟")
        print(f"  输出目录: {config_dir}")

        config_metadata = dict(config)
        config_metadata["shared_initial_env_source"] = str(shared_dir)
        metadata_path = config_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(config_metadata, f, indent=2)

        for env_idx in range(args.samples_per_config):
            src_file = shared_dir / f"env_{env_idx}.pkl"
            output_file = config_dir / f"env_{env_idx}.pkl"
            link_or_copy(src_file, output_file)

            if (env_idx + 1) % 10 == 0 or env_idx + 1 == args.samples_per_config:
                print(f"  已同步: {env_idx + 1}/{args.samples_per_config}")

        print(f"  ✓ 完成 {config['name']}\n")

    print("=" * 80)
    print("生成完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
