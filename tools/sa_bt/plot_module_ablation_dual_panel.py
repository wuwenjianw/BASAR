#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot a grouped-bar module ablation figure for SA-BT Fixed_Makespan."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_STEM = REPO_ROOT / "docs" / "figures" / "sa_bt_module_ablation_dual_panel"
FONT_SIZE = 16

CONFIGS: List[Tuple[str, str]] = [
    ("n10_s5_t120", "120"),
    ("n15_s5_t200", "200"),
    ("n20_s5_t240", "240"),
    ("n30_s5_t300", "300"),
]

VARIANTS: Dict[str, Dict[str, str]] = {
    "Full model": {
        "folder": "save_5_dynamic",
        "color": "#E8684A",
        "hatch": "",
    },
    "w/o L1 loss": {
        "folder": "save_2_dynamic",
        "color": "#9A60B4",
        "hatch": "..",
    },
    "w/o dual cross-attn": {
        "folder": "modabl_single_self_attention_dynamic",
        "color": "#5B8FF9",
        "hatch": "//",
    },
    "w/o attn decoder": {
        "folder": "modabl_global_mlp_dynamic",
        "color": "#F6BD16",
        "hatch": "\\\\",
    },
    "both removed": {
        "folder": "modabl_single_self_attention_global_mlp_dynamic",
        "color": "#61DDAA",
        "hatch": "xx",
    },
}

QUALITY_METRICS = [
    ("success_rate", "Success\nRate", True, True),
    ("deadline_satisfaction_rate", "Deadline\nSat.", True, True),
]

TEMPORAL_METRICS = [
    ("avg_deadline_violation", "Deadline\nViol.", False, False),
    ("avg_flow_time", "Flow\nTime", False, False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a RAL-style grouped-bar figure for SA-BT module ablation."
    )
    parser.add_argument(
        "--output-stem",
        default=str(DEFAULT_OUTPUT_STEM),
        help="Output path stem without suffix. Both .png and .pdf are written.",
    )
    return parser.parse_args()


def load_result_rows(folder: str, config_name: str) -> List[dict]:
    path = (
        REPO_ROOT
        / "artifacts"
        / "results"
        / folder
        / "Fixed_Makespan"
        / f"{config_name}_results.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing module ablation result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of instance results in {path}")
    return rows


def metric_values(folder: str, metric: str, percent: bool) -> List[float]:
    values: List[float] = []
    for config_name, _label in CONFIGS:
        rows = load_result_rows(folder, config_name)
        scale = 100.0 if percent else 1.0
        values.extend(scale * float(row[metric]) for row in rows if metric in row)
    return values


def config_means(folder: str, metric: str, percent: bool) -> List[float]:
    means: List[float] = []
    for config_name, _label in CONFIGS:
        rows = load_result_rows(folder, config_name)
        scale = 100.0 if percent else 1.0
        values = [scale * float(row[metric]) for row in rows if metric in row]
        means.append(statistics.fmean(values))
    return means


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def collect_overall() -> Dict[str, Dict[str, float]]:
    overall: Dict[str, Dict[str, float]] = {}
    for variant, meta in VARIANTS.items():
        overall[variant] = {}
        for metric, _label, _higher_is_better, percent in QUALITY_METRICS + TEMPORAL_METRICS:
            overall[variant][metric] = mean(metric_values(meta["folder"], metric, percent))
    return overall


def add_bar_labels(ax: plt.Axes, bars, *, suffix: str = "", decimals: int = 1) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.9,
            f"{height:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 4,
            rotation=0,
        )


def draw_grouped_panel(
    ax: plt.Axes,
    metrics: List[Tuple[str, str, bool, bool]],
    *,
    ylabel: str,
    ylim: Tuple[float, float],
    value_suffix: str,
    panel_note: str,
) -> None:
    x = np.arange(len(metrics))
    variant_names = list(VARIANTS)
    width = 0.72 / len(variant_names)
    offsets = (np.arange(len(variant_names)) - (len(variant_names) - 1) / 2.0) * width

    for variant_idx, variant in enumerate(variant_names):
        meta = VARIANTS[variant]
        heights = [
            mean(metric_values(meta["folder"], metric, percent))
            for metric, _label, _higher_is_better, percent in metrics
        ]
        bars = ax.bar(
            x + offsets[variant_idx],
            heights,
            width=width,
            label=variant,
            color=meta["color"],
            edgecolor="#202020" if variant == "Full model" else "white",
            linewidth=1.2 if variant == "Full model" else 0.8,
            hatch=meta["hatch"],
            alpha=0.96,
            zorder=3,
        )
        add_bar_labels(ax, bars, suffix=value_suffix, decimals=1)

        for metric_idx, (metric, _label, _higher_is_better, percent) in enumerate(metrics):
            dots = config_means(meta["folder"], metric, percent)
            jitter = np.linspace(-0.035, 0.035, len(dots))
            ax.scatter(
                np.full(len(dots), x[metric_idx] + offsets[variant_idx]) + jitter,
                dots,
                s=20,
                color="#202020",
                alpha=0.50,
                linewidths=0,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label, _higher, _percent in metrics])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.text(
        0.02,
        0.96,
        panel_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE - 2,
        color="#333333",
    )


def build_legend(fig: plt.Figure) -> None:
    handles = []
    labels = []
    for variant, style in VARIANTS.items():
        handle = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=style["color"],
            edgecolor="#202020" if variant == "Full model" else "white",
            linewidth=1.2 if variant == "Full model" else 0.8,
            hatch=style["hatch"],
        )
        handles.append(handle)
        labels.append(variant)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.080),
        ncol=len(handles),
        frameon=False,
        fontsize=FONT_SIZE - 0,
        handlelength=1.2,
        columnspacing=0.8,
        handletextpad=0.4,
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
            "hatch.linewidth": 0.8,
        }
    )


def main() -> None:
    args = parse_args()
    apply_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.26},
    )
    draw_grouped_panel(
        axes[0],
        QUALITY_METRICS,
        ylabel="Rate (%)",
        ylim=(30, 78),
        value_suffix="",
        panel_note="higher is better",
    )
    draw_grouped_panel(
        axes[1],
        TEMPORAL_METRICS,
        ylabel="Simulation Time",
        ylim=(0, 34),
        value_suffix="",
        panel_note="lower is better",
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.36, top=0.95, wspace=0.28)
    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    fig.text(
        (left_box.x0 + left_box.x1) / 2.0,
        0.055,
        "(a) Completion Quality",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    fig.text(
        (right_box.x0 + right_box.x1) / 2.0,
        0.055,
        "(b) Temporal Efficiency",
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

    overall = collect_overall()
    print("Overall means across four Fixed_Makespan configurations:")
    for variant in VARIANTS:
        values = overall[variant]
        print(
            f"{variant}: "
            f"success={values['success_rate']:.2f}, "
            f"deadline={values['deadline_satisfaction_rate']:.2f}, "
            f"violation={values['avg_deadline_violation']:.2f}, "
            f"flow={values['avg_flow_time']:.2f}"
        )


if __name__ == "__main__":
    main()
