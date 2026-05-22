#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot a two-panel reward ablation figure for SA-BT and export PNG/PDF."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_STEM = REPO_ROOT / "docs" / "figures" / "sa_bt_reward_ablation_dual_panel"
FONT_SIZE = 16

CONFIGS: List[Tuple[str, str]] = [
    ("n15_s5_h30", "30"),
    ("n20_s5_h40", "40"),
    ("n20_s5_h50", "50"),
    ("n30_s5_h60", "60"),
]

VARIANTS: Dict[str, Dict[str, str]] = {
    "Full reward": {
        "folder": "save_5_dynamic",
        "color": "#E8684A",
        "marker": "D",
    },
    "w/o shared": {
        "folder": "abl_no_shared_dynamic",
        "color": "#5B8FF9",
        "marker": "o",
    },
    "w/o local": {
        "folder": "abl_no_local_dynamic",
        "color": "#F6BD16",
        "marker": "s",
    },
    "static penalty": {
        "folder": "abl_static_full_penalty_dynamic",
        "color": "#61DDAA",
        "marker": "^",
    },
}

METRIC_LABELS = {
    "deadline_satisfaction_rate": "Deadline sat.",
    "makespan": "Makespan",
    "waiting_time": "Waiting",
    "avg_flow_time": "Flow time",
}

HEATMAP_X_LABELS = {
    "deadline_satisfaction_rate": "Deadline\nSat.",
    "makespan": "Make-\nspan",
    "waiting_time": "Waiting\nTime",
    "avg_flow_time": "Flow\nTime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a RAL-style two-panel figure for reward ablation."
    )
    parser.add_argument(
        "--output-stem",
        default=str(DEFAULT_OUTPUT_STEM),
        help="Output path stem without suffix. Both .png and .pdf are written.",
    )
    return parser.parse_args()


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def load_result_rows(folder: str, config_name: str) -> List[dict]:
    path = (
        REPO_ROOT
        / "artifacts"
        / "results"
        / folder
        / "Fixed_Tasks"
        / f"{config_name}_results.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing reward ablation result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of instance results in {path}")
    return rows


def collect_stats() -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    stats: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    metrics = list(METRIC_LABELS)
    for variant, meta in VARIANTS.items():
        stats[variant] = {}
        for config_name, _label in CONFIGS:
            rows = load_result_rows(meta["folder"], config_name)
            stats[variant][config_name] = {}
            for metric in metrics:
                values = [float(row[metric]) for row in rows if metric in row]
                stats[variant][config_name][metric] = mean_std(values)
    return stats


def collect_overall_means(
    stats: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]
) -> Dict[str, Dict[str, float]]:
    overall: Dict[str, Dict[str, float]] = {}
    for variant, meta in VARIANTS.items():
        overall[variant] = {}
        for metric in METRIC_LABELS:
            rows: List[float] = []
            for config_name, _label in CONFIGS:
                result_rows = load_result_rows(meta["folder"], config_name)
                rows.extend(float(row[metric]) for row in result_rows if metric in row)
            overall[variant][metric] = statistics.fmean(rows)
    return overall


def draw_deadline_panel(
    ax: plt.Axes,
    stats: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
) -> None:
    x = np.arange(len(CONFIGS))
    xlabels = [label for _cfg, label in CONFIGS]
    for variant, meta in VARIANTS.items():
        means = np.array(
            [
                stats[variant][config_name]["deadline_satisfaction_rate"][0] * 100.0
                for config_name, _label in CONFIGS
            ]
        )
        stds = np.array(
            [
                stats[variant][config_name]["deadline_satisfaction_rate"][1] * 100.0
                for config_name, _label in CONFIGS
            ]
        )
        linewidth = 3.0 if variant == "Full reward" else 2.0
        alpha = 0.16 if variant == "Full reward" else 0.10
        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=meta["color"],
            alpha=alpha,
            linewidth=0,
        )
        ax.plot(
            x,
            means,
            marker=meta["marker"],
            color=meta["color"],
            linewidth=linewidth,
            markersize=8.5 if variant == "Full reward" else 7.0,
            markeredgecolor="#202020" if variant == "Full reward" else "white",
            markeredgewidth=1.0 if variant == "Full reward" else 0.8,
            label=variant,
            zorder=5 if variant == "Full reward" else 4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Initial Task Count", fontsize=FONT_SIZE)
    ax.set_ylabel("Deadline Sat. (%)", fontsize=FONT_SIZE)
    ax.set_ylim(15, 76)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)


def degradation_matrix(overall: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    full = overall["Full reward"]
    row_labels = ["w/o shared", "w/o local", "static penalty"]
    matrix: List[List[float]] = []
    for variant in row_labels:
        row = []
        for metric in METRIC_LABELS:
            if metric == "deadline_satisfaction_rate":
                row.append((full[metric] - overall[variant][metric]) / full[metric] * 100.0)
            else:
                row.append((overall[variant][metric] - full[metric]) / full[metric] * 100.0)
        matrix.append(row)
    return np.array(matrix), row_labels


def format_degradation(metric: str, value: float) -> str:
    return f"{value:+.0f}%"


def draw_heatmap_panel(ax: plt.Axes, overall: Dict[str, Dict[str, float]]) -> None:
    matrix, row_labels = degradation_matrix(overall)
    capped = np.clip(matrix, 0.0, 150.0)
    image = ax.imshow(capped, cmap="Reds", vmin=0.0, vmax=150.0, aspect="auto")

    metrics = list(METRIC_LABELS)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([HEATMAP_X_LABELS[m] for m in metrics])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i, variant in enumerate(row_labels):
        for j, metric in enumerate(metrics):
            value = matrix[i, j]
            text_color = "white" if capped[i, j] >= 80 else "#222222"
            fontweight = "bold" if value > 40 else "normal"
            ax.text(
                j,
                i,
                format_degradation(metric, value),
                ha="center",
                va="center",
                fontsize=FONT_SIZE - 2,
                color=text_color,
                fontweight=fontweight,
            )

    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", length=0, labelsize=FONT_SIZE - 1)
    ax.tick_params(axis="y", length=0, labelsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.030)
    cbar.set_ticks([0, 50, 100, 150])
    cbar.set_ticklabels(["0", "50", "100", ">=150"])
    cbar.ax.tick_params(labelsize=FONT_SIZE)


def build_legend(fig: plt.Figure) -> None:
    handles = []
    labels = []
    for variant, style in VARIANTS.items():
        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=3.0 if variant == "Full reward" else 2.0,
            markersize=8.5 if variant == "Full reward" else 7.0,
            markerfacecolor=style["color"],
            markeredgecolor="#202020" if variant == "Full reward" else "white",
        )
        handles.append(handle)
        labels.append(variant)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
    )


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    apply_style()
    stats = collect_stats()
    overall = collect_overall_means(stats)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        gridspec_kw={"width_ratios": [0.90, 1.45], "wspace": 0.34},
    )
    draw_deadline_panel(axes[0], stats)
    draw_heatmap_panel(axes[1], overall)

    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.28, top=0.96, wspace=0.36)
    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    fig.text(
        (left_box.x0 + left_box.x1) / 2.0,
        0.04,
        "(a) Deadline Satisfaction",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    fig.text(
        (right_box.x0 + right_box.x1) / 2.0,
        0.04,
        "(b) Relative Change",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    build_legend(fig)

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
