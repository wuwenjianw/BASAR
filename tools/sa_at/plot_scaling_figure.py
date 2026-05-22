#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 SA-AT scaling benchmark 的单张对比图。

默认输出是一张包含两个子图的 figure：
- 左图：Success Rate vs Total Tasks
- 右图：Planning Time vs Total Tasks

运行示例：
  python -u tools/sa_at/plot_scaling_figure.py
  python -u tools/sa_at/plot_scaling_figure.py --metrics success_rate total_planning_time
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sa_at.sa_at_scaling_config import (
    DEFAULT_FIGURE_PATH,
    METHOD_SPECS,
    RESULTS_ROOT,
    get_metric_spec,
    iter_scaling_configs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a single SA-AT scaling comparison figure.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Root directory containing per-method SA-AT scaling results.",
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
        default=["success_rate", "total_planning_time"],
        help="Metrics to plot as subplots in the single figure.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_FIGURE_PATH),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_metric_series(results_root, method_tag, metric_name):
    method_dir = Path(results_root) / method_tag
    xs = []
    means = []
    stds = []

    for config in iter_scaling_configs():
        result_path = method_dir / f"{config['name']}_results.json"
        if not result_path.exists():
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not rows:
            continue

        values = np.asarray([float(row.get(metric_name, 0.0)) for row in rows], dtype=float)
        spec = get_metric_spec(metric_name)
        values = values * float(spec.get("scale", 1.0))

        xs.append(config["total_tasks"])
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=0)))

    return np.asarray(xs, dtype=float), np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(args.metrics), figsize=(6.4 * len(args.metrics), 4.6))
    if len(args.metrics) == 1:
        axes = [axes]

    plotted_any = False

    for ax, metric_name in zip(axes, args.metrics):
        metric_spec = get_metric_spec(metric_name)
        for method_tag in args.methods:
            if method_tag not in METHOD_SPECS:
                continue

            xs, means, stds = load_metric_series(args.results_root, method_tag, metric_name)
            if xs.size == 0:
                continue

            plotted_any = True
            order = np.argsort(xs)
            xs = xs[order]
            means = means[order]
            stds = stds[order]

            method_spec = METHOD_SPECS[method_tag]
            ax.plot(
                xs,
                means,
                label=method_spec["label"],
                color=method_spec["color"],
                marker=method_spec["marker"],
                linewidth=2.0,
                markersize=6.5,
            )
            ax.fill_between(
                xs,
                means - stds,
                means + stds,
                color=method_spec["color"],
                alpha=0.12,
            )

        ax.set_xlabel("Total Tasks")
        ax.set_ylabel(metric_spec["label"])
        ax.set_xticks([config["total_tasks"] for config in iter_scaling_configs()])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    if not plotted_any:
        raise FileNotFoundError(
            f"在 {args.results_root} 下没有找到可绘图的 SA-AT scaling 结果。"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
