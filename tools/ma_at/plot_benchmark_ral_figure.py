#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot a publication-style MA-AT benchmark figure and export PNG/PDF."""

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

from scripts.ma_at.ma_at_benchmark_config import (  # noqa: E402
    RESULTS_ROOT,
    get_metric_spec,
    iter_benchmark_configs,
)


FONT_SIZE = 16
DEFAULT_OUTPUT_STEM = REPO_ROOT / "docs" / "figures" / "ma_at_benchmark_ral"

METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "greedy": {"label": "Greedy", "color": "#F7A6A3", "hatch": ""},
    "hrlf": {"label": "HRLF", "color": "#67ADDE", "hatch": "///"},
    "capam": {"label": "CAPAM", "color": "#D3E2E5", "hatch": "\\\\\\"},
    "ours": {"label": "Ours", "color": "#F2D1CA", "hatch": "xx"},
}

PANELS: List[Tuple[str, str]] = [
    ("makespan", "Makespan"),
    ("waiting_time", "Waiting Time"),
    ("deadline_satisfaction_rate", "Deadline Satisfaction Rate (%)"),
    ("total_planning_time", "Planning Time (s)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an RAL-style MA-AT benchmark figure.")
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT),
        help="Root directory containing MA-AT benchmark results.",
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
    values = np.asarray([float(row.get(metric_name, 0.0)) for row in rows], dtype=float)
    spec = get_metric_spec(metric_name)
    return values * float(spec.get("scale", 1.0))


def plot_panel(ax: plt.Axes, results_root: Path, metric_name: str, ylabel: str, methods: List[str]) -> None:
    configs = iter_benchmark_configs()
    x = np.arange(len(configs), dtype=float)
    bar_width = 0.76 / max(1, len(methods))
    plotted_any = False

    for idx, method_tag in enumerate(methods):
        if method_tag not in METHOD_STYLES:
            continue
        means = []
        stds = []
        available = False
        style = METHOD_STYLES[method_tag]

        for config in configs:
            values = load_metric_values(results_root, method_tag, config["name"], metric_name)
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
        offset = (idx - (len(methods) - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds,
            label=style["label"],
            color=style["color"],
            alpha=0.92 if method_tag == "ours" else 0.84,
            edgecolor="black",
            linewidth=1.2 if method_tag == "ours" else 0.8,
            capsize=3.0,
            hatch=style["hatch"],
            error_kw={"elinewidth": 1.1, "ecolor": "black"},
            zorder=3,
        )

    if not plotted_any:
        raise FileNotFoundError(f"No plottable MA-AT results found under {results_root}.")

    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(config["total_tasks"]) for config in configs], rotation=0, ha="center")
    ax.set_xlabel("Total Tasks", fontsize=FONT_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.0)


def build_legend(fig: plt.Figure, methods: List[str]) -> None:
    handles = []
    labels = []
    for method_tag in methods:
        if method_tag not in METHOD_STYLES:
            continue
        style = METHOD_STYLES[method_tag]
        handle = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=style["color"],
            edgecolor="black",
            linewidth=1.2 if method_tag == "ours" else 0.8,
            hatch=style["hatch"],
        )
        handles.append(handle)
        labels.append(style["label"])

    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.6,
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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    axes = axes.flatten()

    for ax, (metric_name, ylabel) in zip(axes, PANELS):
        plot_panel(ax, results_root, metric_name, ylabel, methods)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    build_legend(fig, methods)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
