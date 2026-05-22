#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the compact SA-AT / MA-AT complementary metrics table."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ma_at.ma_at_benchmark_config import (  # noqa: E402
    RESULTS_ROOT as MA_AT_RESULTS_ROOT,
    iter_benchmark_configs,
)
from scripts.sa_at.sa_at_scaling_config import (  # noqa: E402
    RESULTS_ROOT as SA_AT_RESULTS_ROOT,
    iter_scaling_configs,
)


OUTPUT_DIR = ROOT / "docs" / "tables"
OUTPUT_TEX = OUTPUT_DIR / "sa_at_ma_at_complementary_generated.tex"
PREVIEW_TEX = OUTPUT_DIR / "sa_at_ma_at_complementary_preview.tex"
SUMMARY_CSV = OUTPUT_DIR / "sa_at_ma_at_complementary_summary.csv"
SA_AT_ARRIVAL4_DEADLINE1P3X_ROOT = ROOT / "artifacts" / "results" / "sa_at_scaling_arrival4_deadline1p3x"
SA_AT_RESULT_ROOT_OVERRIDES = {
    "n300_w30_h200_t1500": SA_AT_ARRIVAL4_DEADLINE1P3X_ROOT,
    "n300_w30_h200_t2000": SA_AT_ARRIVAL4_DEADLINE1P3X_ROOT,
}

METHODS: List[Tuple[str, str]] = [
    ("greedy", "Greedy"),
    ("hrlf", "HRLF"),
    ("capam", "CAPAM"),
    ("ours", "BASAR"),
]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    sa_header: str
    ma_header: str
    scale: float = 1.0
    higher_is_better: bool = False


METRICS: List[MetricSpec] = [
    MetricSpec("deadline_satisfaction_rate", "\\makecell{DSR (\\%)}", "\\makecell{DSA (\\%)}", 100.0, True),
    MetricSpec("avg_deadline_violation", "\\makecell{ADV}", "\\makecell{ADV}", 1.0, False),
    MetricSpec("makespan", "\\makecell{Makespan}", "\\makecell{Makespan}", 1.0, False),
    MetricSpec("avg_flow_time", "\\makecell{FT}", "\\makecell{FT}", 1.0, False),
]


def load_rows(results_root: Path, method_tag: str, config_name: str) -> List[dict]:
    result_path = results_root / method_tag / f"{config_name}_results.json"
    if not result_path.exists():
        return []
    with result_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else []


def load_sa_at_rows(method_tag: str, config_name: str) -> List[dict]:
    override_root = SA_AT_RESULT_ROOT_OVERRIDES.get(config_name)
    if override_root is not None:
        rows = load_rows(override_root, method_tag, config_name)
        if rows:
            return rows
    return load_rows(SA_AT_RESULTS_ROOT, method_tag, config_name)


def sa_at_source_path(method_tag: str, config_name: str) -> Path:
    override_root = SA_AT_RESULT_ROOT_OVERRIDES.get(config_name)
    if override_root is not None:
        result_path = override_root / method_tag / f"{config_name}_results.json"
        if result_path.exists():
            return result_path
    return SA_AT_RESULTS_ROOT / method_tag / f"{config_name}_results.json"


def metric_values(rows: List[dict], spec: MetricSpec) -> np.ndarray:
    values = []
    for row in rows:
        if spec.key not in row or not isinstance(row[spec.key], (int, float)):
            continue
        value = float(row[spec.key])
        if (
            spec.key == "avg_deadline_violation"
            and row.get("avg_deadline_violation_scope") != "completed_tasks"
            and isinstance(row.get("deadline_violation_rate"), (int, float))
        ):
            value *= float(row["deadline_violation_rate"])
        values.append(value * spec.scale)
    return np.asarray(values, dtype=float)


def summarize_metric(rows: List[dict], spec: MetricSpec) -> Tuple[float, float, int] | None:
    values = metric_values(rows, spec)
    if values.size == 0:
        return None
    return float(values.mean()), float(values.std(ddof=0)), int(values.size)


def config_label(config: dict, *, include_total: bool) -> str:
    agents = int(config["agents"])
    species = int(config.get("species", config.get("warehouses")))
    initial_tasks = int(config["initial_tasks"])
    dynamic_tasks = int(config["dynamic_tasks"])
    lines = [f"$A={agents}$", f"$S={species}$", f"$I={initial_tasks}$", f"$D={dynamic_tasks}$"]
    if include_total:
        lines.append(f"$T={int(config['total_tasks'])}$")
    return "\\makecell{" + "\\\\".join(lines) + "}"


def metric_rank(value: float | None, candidates: Iterable[float], spec: MetricSpec) -> int | None:
    finite = [candidate for candidate in candidates if candidate is not None and np.isfinite(candidate)]
    if value is None or not finite:
        return None
    ordered = sorted(finite, reverse=spec.higher_is_better)
    distinct: List[float] = []
    for candidate in ordered:
        if not any(abs(candidate - existing) <= 1e-9 for existing in distinct):
            distinct.append(candidate)
    for idx, candidate in enumerate(distinct[:2], start=1):
        if abs(value - candidate) <= 1e-9:
            return idx
    return None


def format_number(value: float, *, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}"


def render_metric_cell(summary: Tuple[float, float, int] | None, rank: int | None) -> str:
    if summary is None:
        return "--"
    mean, std, _count = summary
    cell = f"{format_number(mean)}$\\pm${format_number(std)}"
    if rank == 1:
        return f"\\textbf{{{cell}}}"
    if rank == 2:
        return f"\\underline{{{cell}}}"
    return cell


def collect_summary_rows() -> List[dict]:
    summary_rows: List[dict] = []
    for benchmark, results_root, configs in [
        ("SA-AT", SA_AT_RESULTS_ROOT, iter_scaling_configs()),
        ("MA-AT", MA_AT_RESULTS_ROOT, iter_benchmark_configs()),
    ]:
        for config in configs:
            config_name = config["name"]
            for method_tag, method_label in METHODS:
                if benchmark == "SA-AT":
                    rows = load_sa_at_rows(method_tag, config_name)
                    source_path = sa_at_source_path(method_tag, config_name)
                else:
                    rows = load_rows(results_root, method_tag, config_name)
                    source_path = results_root / method_tag / f"{config_name}_results.json"
                row = {
                    "benchmark": benchmark,
                    "config_name": config_name,
                    "method_tag": method_tag,
                    "method": method_label,
                    "source_path": str(source_path.relative_to(ROOT)),
                    "instances": len(rows),
                }
                for spec in METRICS:
                    summary = summarize_metric(rows, spec)
                    if summary is None:
                        row[f"{spec.key}_mean"] = ""
                        row[f"{spec.key}_std"] = ""
                    else:
                        mean, std, count = summary
                        row[f"{spec.key}_mean"] = mean
                        row[f"{spec.key}_std"] = std
                        row[f"{spec.key}_count"] = count
                summary_rows.append(row)
    return summary_rows


def render_table() -> str:
    sa_configs = iter_scaling_configs()
    ma_configs = iter_benchmark_configs()

    lines = [
        "% Auto-generated by tools/generate_sa_at_ma_at_complementary_table.py",
        "% The table reports introduced metrics that complement the main figures.",
        "% All metrics are aggregated over available local result instances and reported as mean$\\pm$standard deviation.",
        "",
        "\\begin{table*}[t]",
        "\\caption{Complementary performance on the SA-AT and MA-AT benchmarks. "
        "The table reports deadline compliance and execution efficiency using the metrics defined in the experimental protocol. "
        "All metrics are reported as mean$\\pm$standard deviation over 50 test instances. Higher DSR/DSA is better, while lower ADV, makespan, and FT are better.}",
        "\\label{tab:saat_maat_complementary}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\renewcommand{\\arraystretch}{1.04}",
        "\\begin{tabular}{llcccc@{\\hspace{5mm}}lcccc}",
        "\\toprule",
        "\\multicolumn{6}{c}{SA-AT Scaling} & \\multicolumn{5}{c}{MA-AT Dynamic Scenarios} \\\\",
        "\\cmidrule(lr){1-6}\\cmidrule(lr){7-11}",
        "\\makecell{Instance\\\\Config.} & Model & "
        + " & ".join(spec.sa_header for spec in METRICS)
        + " & \\makecell{Instance\\\\Config.} & "
        + " & ".join(spec.ma_header for spec in METRICS)
        + " \\\\",
        "\\midrule",
    ]

    for pair_idx, (sa_config, ma_config) in enumerate(zip(sa_configs, ma_configs)):
        sa_name = sa_config["name"]
        ma_name = ma_config["name"]

        sa_method_rows = {method: load_sa_at_rows(method, sa_name) for method, _label in METHODS}
        ma_method_rows = {method: load_rows(MA_AT_RESULTS_ROOT, method, ma_name) for method, _label in METHODS}

        sa_summaries: Dict[Tuple[str, str], Tuple[float, float, int] | None] = {}
        ma_summaries: Dict[Tuple[str, str], Tuple[float, float, int] | None] = {}
        for method_tag, _method_label in METHODS:
            for spec in METRICS:
                sa_summaries[(method_tag, spec.key)] = summarize_metric(sa_method_rows[method_tag], spec)
                ma_summaries[(method_tag, spec.key)] = summarize_metric(ma_method_rows[method_tag], spec)

        for row_idx, (method_tag, method_label) in enumerate(METHODS):
            sa_label = f"\\multirow{{{len(METHODS)}}}{{*}}{{{config_label(sa_config, include_total=False)}}}" if row_idx == 0 else ""
            ma_label = f"\\multirow{{{len(METHODS)}}}{{*}}{{{config_label(ma_config, include_total=False)}}}" if row_idx == 0 else ""

            sa_cells = []
            ma_cells = []
            for spec in METRICS:
                sa_summary = sa_summaries[(method_tag, spec.key)]
                ma_summary = ma_summaries[(method_tag, spec.key)]
                sa_rank = metric_rank(
                    None if sa_summary is None else sa_summary[0],
                    [None if sa_summaries[(m, spec.key)] is None else sa_summaries[(m, spec.key)][0] for m, _ in METHODS],
                    spec,
                )
                ma_rank = metric_rank(
                    None if ma_summary is None else ma_summary[0],
                    [None if ma_summaries[(m, spec.key)] is None else ma_summaries[(m, spec.key)][0] for m, _ in METHODS],
                    spec,
                )
                sa_cells.append(render_metric_cell(sa_summary, sa_rank))
                ma_cells.append(render_metric_cell(ma_summary, ma_rank))

            lines.append(
                f"{sa_label} & {method_label} & "
                + " & ".join(sa_cells)
                + f" & {ma_label} & "
                + " & ".join(ma_cells)
                + " \\\\"
            )

        if pair_idx < len(sa_configs) - 1:
            lines.append("\\addlinespace[1.5pt]")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    return "\n".join(lines)


def build_preview_document() -> str:
    return "\n".join([
        "\\documentclass[journal]{IEEEtran}",
        "\\usepackage{booktabs}",
        "\\usepackage{makecell}",
        "\\usepackage{multirow}",
        "\\usepackage{graphicx}",
        "\\usepackage[margin=0.7in]{geometry}",
        "\\begin{document}",
        "\\input{sa_at_ma_at_complementary_generated.tex}",
        "\\end{document}",
        "",
    ])


def write_summary_csv(rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(render_table(), encoding="utf-8")
    PREVIEW_TEX.write_text(build_preview_document(), encoding="utf-8")
    write_summary_csv(collect_summary_rows())
    print(f"Generated table: {OUTPUT_TEX.relative_to(ROOT)}")
    print(f"Generated preview: {PREVIEW_TEX.relative_to(ROOT)}")
    print(f"Generated summary: {SUMMARY_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
