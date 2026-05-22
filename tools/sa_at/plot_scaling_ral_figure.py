#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot an RAL-style SA-AT scaling figure and export PNG/PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sa_at.sa_at_scaling_config import (  # noqa: E402
    RESULTS_ROOT,
    iter_scaling_configs,
)


FONT_SIZE = 16
DEFAULT_OUTPUT_STEM = REPO_ROOT / "docs" / "figures" / "sa_at_scaling_ral"

METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "greedy": {"label": "Greedy", "color": "#5B8FF9", "marker": "o", "zorder": 3},
    "hrlf": {"label": "HRLF", "color": "#F6BD16", "marker": "s", "zorder": 3},
    "capam": {"label": "CAPAM", "color": "#61DDAA", "marker": "^", "zorder": 3},
    "ours": {"label": "Ours", "color": "#E8684A", "marker": "D", "zorder": 5},
}

PANELS: List[Tuple[str, str]] = [
    ("total_planning_time", "Total Planning Time (s)"),
    ("avg_planning_time_ms", "Avg. Planning Time (ms)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an RAL-style SA-AT scaling figure.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Root directory containing SA-AT scaling results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["greedy", "hrlf", "capam", "ours"],
        help="Methods to include.",
    )
    parser.add_argument(
        "--output-stem",
        default=str(DEFAULT_OUTPUT_STEM),
        help="Output path stem without suffix; both .png and .pdf will be written.",
    )
    return parser.parse_args()


def load_metric_values(results_root: Path, method_tag: str, config_name: str, metric_name: str) -> np.ndarray | None:
    result_path = results_root / method_tag / f"{config_name}_results.json"
    if not result_path.exists():
        return None

    rows = json.loads(result_path.read_text(encoding="utf-8"))
    if not rows:
        return None

    if metric_name == "avg_planning_time_ms":
        values = np.asarray(
            [1000.0 * float(row.get("avg_planning_time", 0.0)) for row in rows],
            dtype=float,
        )
    else:
        values = np.asarray([float(row.get(metric_name, 0.0)) for row in rows], dtype=float)
    return values


def build_series(results_root: Path, method_tag: str, metric_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    means: List[float] = []
    stds: List[float] = []

    for config in iter_scaling_configs():
        values = load_metric_values(results_root, method_tag, config["name"], metric_name)
        if values is None:
            continue
        xs.append(float(config["total_tasks"]))
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=0)))

    return np.asarray(xs), np.asarray(means), np.asarray(stds)


def plot_panel(ax: plt.Axes, results_root: Path, metric_name: str, ylabel: str, methods: List[str]) -> None:
    plotted_any = False
    for method_tag in methods:
        if method_tag not in METHOD_STYLES:
            continue
        xs, means, stds = build_series(results_root, method_tag, metric_name)
        if xs.size == 0:
            continue

        plotted_any = True
        order = np.argsort(xs)
        xs = xs[order]
        means = means[order]
        stds = stds[order]
        style = METHOD_STYLES[method_tag]

        ax.plot(
            xs,
            means,
            color=style["color"],
            marker=style["marker"],
            linewidth=3.0 if method_tag == "ours" else 2.0,
            markersize=8.5 if method_tag == "ours" else 7.0,
            alpha=0.95 if method_tag == "ours" else 0.85,
            zorder=style["zorder"],
            label=style["label"],
        )
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            color=style["color"],
            alpha=0.16 if method_tag == "ours" else 0.10,
            zorder=style["zorder"] - 1,
        )

    if not plotted_any:
        raise FileNotFoundError(f"No plottable SA-AT results found under {results_root}.")

    ax.set_xlabel("Total Tasks", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_xticks([config["total_tasks"] for config in iter_scaling_configs()])
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)


def build_legend(fig: plt.Figure, methods: List[str]) -> None:
    handles = []
    labels = []
    for method_tag in methods:
        if method_tag not in METHOD_STYLES:
            continue
        style = METHOD_STYLES[method_tag]
        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=3.0 if method_tag == "ours" else 2.0,
            markersize=8.5 if method_tag == "ours" else 7.0,
            markerfacecolor=style["color"],
            markeredgecolor="#202020" if method_tag == "ours" else "white",
        )
        handles.append(handle)
        labels.append(style["label"])

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
    )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    methods = [method for method in args.methods if method in METHOD_STYLES]
    if not methods:
        raise ValueError("No supported methods were provided.")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )

    fig, axes = plt.subplots(1, len(PANELS), figsize=(12.8, 5.8))
    if len(PANELS) == 1:
        axes = [axes]

    for ax, (metric_name, ylabel) in zip(axes, PANELS):
        plot_panel(ax, results_root, metric_name, ylabel, methods)

    fig.tight_layout(rect=(0, 0.18, 1, 1))
    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    fig.text(
        (left_box.x0 + left_box.x1) / 2.0,
        0.04,
        "(a) Total Planning Time",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    fig.text(
        (right_box.x0 + right_box.x1) / 2.0,
        0.04,
        "(b) Avg. Planning Time",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    build_legend(fig, methods)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
