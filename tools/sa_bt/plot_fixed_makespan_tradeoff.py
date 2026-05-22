#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot a Fixed-Makespan trade-off figure for SA-BT and export PNG/PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sa_bt.generate_ieee_tables import (  # noqa: E402
    DISPLAY_METHODS,
    PROTOCOL_CONFIGS,
    get_metric_stat,
    load_stat_map,
)


DEFAULT_OUTPUT_STEM = REPO_ROOT / "docs" / "figures" / "sa_bt_fixed_makespan_tradeoff"
FONT_SIZE = 16

METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "Greedy": {"color": "#5B8FF9", "zorder": 3},
    "TACO": {"color": "#61DDAA", "zorder": 3},
    "CTAS-D": {"color": "#65789B", "zorder": 3},
    "HRLF": {"color": "#F6BD16", "zorder": 3},
    "Ours": {"color": "#E8684A", "zorder": 5},
}

CASE_STYLES: Dict[str, Dict[str, object]] = {
    "n10_s5_t120": {"marker": "o", "label": "10"},
    "n15_s5_t200": {"marker": "s", "label": "15"},
    "n20_s5_t240": {"marker": "^", "label": "20"},
    "n30_s5_t300": {"marker": "D", "label": "30"},
}

PANELS: List[Tuple[str, str]] = [
    ("waiting_time", "Waiting Time"),
    ("avg_deadline_violation", "Avg. Deadline Violation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a trade-off figure for the SA-BT Fixed-Makespan protocol."
    )
    parser.add_argument(
        "--output-stem",
        default=str(DEFAULT_OUTPUT_STEM),
        help="Output path stem without suffix; both .png and .pdf will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DISPLAY_METHODS,
        help="Methods to include.",
    )
    return parser.parse_args()


def collect_tradeoff_points(methods: List[str]) -> Dict[str, List[dict]]:
    stat_map = load_stat_map()
    rows: Dict[str, List[dict]] = {panel_key: [] for panel_key, _ in PANELS}

    for config_name, _n_agents, _n_species, _horizon in PROTOCOL_CONFIGS["Fixed_Makespan"]:
        for method in methods:
            success_stat = get_metric_stat(
                stat_map, method, "Fixed_Makespan", config_name, "success_rate"
            )
            if success_stat is None:
                continue
            success_mean = success_stat[0] * 100.0

            for panel_key, _panel_ylabel in PANELS:
                metric_stat = get_metric_stat(
                    stat_map, method, "Fixed_Makespan", config_name, panel_key
                )
                if metric_stat is None:
                    continue
                rows[panel_key].append(
                    {
                        "method": method,
                        "config_name": config_name,
                        "case_label": CASE_STYLES[config_name]["label"],
                        "success_mean": success_mean,
                        "success_std": success_stat[1] * 100.0,
                        "metric_mean": metric_stat[0],
                        "metric_std": metric_stat[1],
                    }
                )
    return rows


def plot_panel(ax: plt.Axes, panel_rows: List[dict], ylabel: str, methods: List[str]) -> None:
    for method in methods:
        series = [row for row in panel_rows if row["method"] == method]
        if not series:
            continue
        series.sort(key=lambda row: PROTOCOL_CONFIGS["Fixed_Makespan"].index(
            next(cfg for cfg in PROTOCOL_CONFIGS["Fixed_Makespan"] if cfg[0] == row["config_name"])
        ))
        style = METHOD_STYLES[method]
        xs = [row["success_mean"] for row in series]
        ys = [row["metric_mean"] for row in series]
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=2.0 if method == "Ours" else 1.5,
            alpha=0.9 if method == "Ours" else 0.6,
            zorder=style["zorder"] - 1,
        )
        for row in series:
            case_style = CASE_STYLES[row["config_name"]]
            ax.errorbar(
                row["success_mean"],
                row["metric_mean"],
                xerr=row["success_std"],
                yerr=row["metric_std"],
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.0,
                alpha=0.18 if method == "Ours" else 0.12,
                capsize=0,
                zorder=style["zorder"] - 2,
            )
            ax.scatter(
                row["success_mean"],
                row["metric_mean"],
                s=130 if method == "Ours" else 88,
                marker=case_style["marker"],
                color=style["color"],
                edgecolors="#202020" if method == "Ours" else "white",
                linewidths=1.4 if method == "Ours" else 0.8,
                zorder=style["zorder"],
            )
            if method == "Ours":
                ax.annotate(
                    row["case_label"],
                    (row["success_mean"], row["metric_mean"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=FONT_SIZE - 4,
                    color=style["color"],
                    fontweight="bold",
                )

    ax.set_xlabel("Success Rate (%)", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    x_values = [row["success_mean"] for row in panel_rows]
    y_values = [row["metric_mean"] for row in panel_rows]
    x_margin = max(2.5, 0.08 * (max(x_values) - min(x_values)))
    y_margin = max(1.0, 0.10 * (max(y_values) - min(y_values)))
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)


def build_legends(fig: plt.Figure, methods: List[str]) -> None:
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[method]["color"],
            marker="o",
            linestyle="-",
            linewidth=2.0 if method == "Ours" else 1.5,
            markersize=7.5,
            markerfacecolor=METHOD_STYLES[method]["color"],
            markeredgecolor="#202020" if method == "Ours" else "white",
            label=method,
        )
        for method in methods
    ]
    case_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=style["marker"],
            linestyle="None",
            markersize=7.5,
            markerfacecolor="white",
            markeredgecolor="#555555",
            label=style["label"],
        )
        for _config_name, style in CASE_STYLES.items()
    ]

    spacer = [
        Line2D(
            [0],
            [0],
            color="none",
            marker=None,
            linestyle="None",
            label="   ",
        )
    ]
    combined_handles = method_handles + spacer + case_handles
    combined_labels = [handle.get_label() for handle in combined_handles]
    fig.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=len(combined_handles),
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.5,
    )


def main() -> None:
    args = parse_args()
    methods = [method for method in args.methods if method in METHOD_STYLES]
    if not methods:
        raise ValueError("No supported methods were provided.")

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    panel_rows = collect_tradeoff_points(methods)

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    })

    fig, axes = plt.subplots(1, len(PANELS), figsize=(12.8, 5.8))
    for ax, (panel_key, ylabel) in zip(axes, PANELS):
        plot_panel(ax, panel_rows[panel_key], ylabel, methods)

    fig.tight_layout(rect=(0, 0.18, 1, 1))
    left_box = axes[0].get_position()
    right_box = axes[1].get_position()
    fig.text(
        (left_box.x0 + left_box.x1) / 2.0,
        0.11,
        "(a) Completion vs Congestion",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    fig.text(
        (right_box.x0 + right_box.x1) / 2.0,
        0.11,
        "(b) Completion vs Tardiness",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    build_legends(fig, methods)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
