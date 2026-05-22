#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""绘制 MA-AT dynamic benchmark 的单张对比图。"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ma_at.ma_at_benchmark_config import (
    DEFAULT_FIGURE_PATH,
    METHOD_SPECS,
    RESULTS_ROOT,
    get_metric_spec,
    iter_benchmark_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a single MA-AT dynamic benchmark figure.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Root directory containing per-method MA-AT results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["greedy", "hrlf", "capam", "ours"],
        help="Method tags to include in the figure.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["makespan", "waiting_time", "deadline_satisfaction_rate"],
        help="Metrics to plot as subplots in the single figure.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_FIGURE_PATH),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_metric_values(results_root, method_tag, config_name, metric_name):
    result_path = Path(results_root) / method_tag / f"{config_name}_results.json"
    if not result_path.exists():
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return None
    values = np.asarray([float(row.get(metric_name, 0.0)) for row in rows], dtype=float)
    spec = get_metric_spec(metric_name)
    return values * float(spec.get("scale", 1.0))


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configs = iter_benchmark_configs()
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(6.0 * len(args.metrics), 5.0))
    if len(args.metrics) == 1:
        axes = [axes]

    method_tags = [tag for tag in args.methods if tag in METHOD_SPECS]
    if not method_tags:
        raise ValueError("没有可绘图的方法。")

    x = np.arange(len(configs), dtype=float)
    bar_width = 0.8 / max(1, len(method_tags))
    plotted_any = False

    for ax, metric_name in zip(axes, args.metrics):
        metric_spec = get_metric_spec(metric_name)

        for idx, method_tag in enumerate(method_tags):
            means = []
            stds = []
            available = False

            for config in configs:
                values = load_metric_values(args.results_root, method_tag, config["name"], metric_name)
                if values is None:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                available = True
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=0)))

            if not available:
                continue

            plotted_any = True
            offset = (idx - (len(method_tags) - 1) / 2.0) * bar_width
            method_spec = METHOD_SPECS[method_tag]
            ax.bar(
                x + offset,
                means,
                width=bar_width,
                yerr=stds,
                label=method_spec["label"],
                color=method_spec["color"],
                alpha=0.88,
                capsize=3.0,
            )

        ax.set_xlabel("Scenario (agents/species/total_tasks)")
        ax.set_ylabel(metric_spec["label"])
        ax.set_xticks(x)
        ax.set_xticklabels([config["plot_label"] for config in configs], rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    if not plotted_any:
        raise FileNotFoundError(f"在 {args.results_root} 下没有找到可绘图的 MA-AT 结果。")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
