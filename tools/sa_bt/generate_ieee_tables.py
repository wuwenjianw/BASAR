#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已有 SA-BT 动态评估结果生成论文版紧凑大表。

设计目标：
1. 使用单个 paper-friendly 大表，把 Fixed Tasks 和 Fixed Makespan 放在左右两大列
2. SA-BT 当前所有配置的 species 都固定为 5，直接写入表头
3. 指标按协议的判别力选择，而不是强行共用一套：
   - Fixed Tasks：Success / Deadline Satisfaction / Avg Deadline Violation / Flow / Planning Time
   - Fixed Makespan (H=120)：Success / Deadline Satisfaction / Avg Deadline Violation / Waiting Time / Planning Time
4. 论文表格每个方法/配置统一使用 50 个 env-indexed 样本，并把来源写入 csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "tables"
OUTPUT_TEX = OUTPUT_DIR / "sa_bt_baseline_ieee_generated.tex"
PREVIEW_TEX = OUTPUT_DIR / "sa_bt_baseline_ieee_preview.tex"
SUMMARY_CSV = OUTPUT_DIR / "sa_bt_baseline_summary.csv"
SOURCE_CSV = OUTPUT_DIR / "sa_bt_baseline_sources.csv"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    header: str
    higher_is_better: bool
    decimals: int
    percent: bool = False
    show_std: bool = True


@dataclass(frozen=True)
class ResolvedResult:
    path: Path
    mtime: float
    df: pd.DataFrame
    source_instances: int
    evaluated_instances: int


METHOD_SOURCE_CANDIDATES = {
    "Greedy": [
        ROOT / "artifacts" / "result_comparison" / "results_greedy_dynamic",
        ROOT / "artifacts" / "results" / "greedy_dynamic",
        ROOT / "results_greedy_dynamic",
    ],
    "TACO": [
        ROOT / "artifacts" / "results" / "taco_dynamic",
        ROOT / "results_taco_dynamic",
    ],
    "CTAS-D": [
        ROOT / "artifacts" / "results" / "ctasd_dynamic",
        ROOT / "results_ctasd_dynamic",
    ],
    "HRLF": [
        ROOT / "artifacts" / "results" / "save_baseline_dynamic",
        ROOT / "artifacts" / "results" / "hrlf_dynamic",
        ROOT / "artifacts" / "result_comparison" / "results_hrlf_dynamic",
        ROOT / "results_save_baseline_dynamic",
        ROOT / "results_hrlf_dynamic",
    ],
    "BASAR": [
        ROOT / "artifacts" / "results" / "save_5_dynamic",
        ROOT / "artifacts" / "result_comparison" / "results_save_5_dynamic",
        ROOT / "results_save_5_dynamic",
    ],
    "CAPAM": [
        ROOT / "artifacts" / "results" / "capam_dynamic",
        ROOT / "artifacts" / "results" / "capam_dynamic_dynamic",
        ROOT / "results_capam_dynamic",
    ],
}

METHOD_ORDER = ["Greedy", "TACO", "CTAS-D", "HRLF", "CAPAM", "BASAR"]
DISPLAY_METHODS = ["Greedy", "TACO", "CTAS-D", "HRLF", "CAPAM", "BASAR"]
TABLE_SAMPLE_LIMITS = {
    "Fixed_Tasks": 50,
    "Fixed_Makespan": 50,
}

PROTOCOL_CONFIGS = {
    "Fixed_Tasks": [
        ("n15_s5_h30", 15, 5, 30),
        ("n20_s5_h40", 20, 5, 40),
        ("n20_s5_h50", 20, 5, 50),
        ("n30_s5_h60", 30, 5, 60),
    ],
    "Fixed_Makespan": [
        ("n10_s5_t120", 10, 5, 120),
        ("n15_s5_t200", 15, 5, 200),
        ("n20_s5_t240", 20, 5, 240),
        ("n30_s5_t300", 30, 5, 300),
    ],
}

CONFIG_DETAILS = {
    "n15_s5_h30": {"agents": 15, "species": 5, "initial_tasks": 30, "total_tasks": 100},
    "n20_s5_h40": {"agents": 20, "species": 5, "initial_tasks": 40, "total_tasks": 100},
    "n20_s5_h50": {"agents": 20, "species": 5, "initial_tasks": 50, "total_tasks": 100},
    "n30_s5_h60": {"agents": 30, "species": 5, "initial_tasks": 60, "total_tasks": 100},
    "n10_s5_t120": {"agents": 10, "species": 5, "initial_tasks": 30, "total_tasks": 120},
    "n15_s5_t200": {"agents": 15, "species": 5, "initial_tasks": 50, "total_tasks": 200},
    "n20_s5_t240": {"agents": 20, "species": 5, "initial_tasks": 60, "total_tasks": 240},
    "n30_s5_t300": {"agents": 30, "species": 5, "initial_tasks": 80, "total_tasks": 300},
}

PROTOCOL_TITLES = {
    "Fixed_Tasks": "Fixed Task Count",
    "Fixed_Makespan": "Fixed Makespan (H=120)",
}

SUMMARY_METRICS: List[MetricSpec] = [
    MetricSpec("success_rate", "Success Rate (\\%)", True, 1, percent=True, show_std=False),
    MetricSpec("deadline_satisfaction_rate", "Deadline Satisfaction Rate (\\%)", True, 1, percent=True),
    MetricSpec("avg_deadline_violation", "Avg. Deadline Violation", False, 1),
    MetricSpec("makespan", "Makespan", False, 1),
    MetricSpec("finished_tasks", "Finished", True, 1),
    MetricSpec("on_time_finished_tasks", "On-Time Finished Tasks", True, 1),
    MetricSpec("task_throughput", "Task Throughput", True, 1),
    MetricSpec("wait_per_finished_task", "Waiting Time per Finished Task", False, 1),
    MetricSpec("travel_per_finished_task", "Travel Distance per Finished Task", False, 1),
    MetricSpec("avg_flow_time", "Flow Time", False, 1),
    MetricSpec("planning_time_per_finished_task_ms", "Planning Time per Finished Task (ms)", False, 3),
    MetricSpec("total_planning_time", "Total Planning Time (s)", False, 2),
]

FIXED_TASK_TABLE_METRICS: List[MetricSpec] = [
    MetricSpec("success_rate", "Success Rate (\\%)", True, 1, percent=True, show_std=False),
    MetricSpec("deadline_satisfaction_rate", "Deadline Satisfaction Rate (\\%)", True, 1, percent=True),
    MetricSpec("avg_deadline_violation", "Avg. Deadline Violation", False, 1),
    MetricSpec("avg_flow_time", "Flow Time", False, 1),
    MetricSpec("total_planning_time", "Total Planning Time (s)", False, 2),
]

FIXED_MAKESPAN_TABLE_METRICS: List[MetricSpec] = [
    MetricSpec("success_rate", "Success Rate (\\%)", True, 1, percent=True, show_std=False),
    MetricSpec("deadline_satisfaction_rate", "Deadline Satisfaction Rate (\\%)", True, 1, percent=True),
    MetricSpec("avg_deadline_violation", "Avg. Deadline Violation", False, 1),
    MetricSpec("waiting_time", "Waiting Time", False, 1),
    MetricSpec("total_planning_time", "Total Planning Time (s)", False, 2),
]

StatMap = Dict[str, Dict[str, Dict[str, Optional[ResolvedResult]]]]


def load_results(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return None
    return pd.DataFrame(data)


def env_file_order(df: pd.DataFrame) -> pd.Series:
    if "env_file" not in df.columns:
        return pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    env_numbers = df["env_file"].astype(str).str.extract(r"env[_-]?(\d+)")[0]
    return pd.to_numeric(env_numbers, errors="coerce").fillna(np.inf)


def select_table_samples(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    sample_limit = TABLE_SAMPLE_LIMITS.get(protocol)
    if sample_limit is None or len(df) <= sample_limit:
        return df.copy().reset_index(drop=True)

    ordered = df.copy()
    ordered["_env_order"] = env_file_order(ordered)
    sort_columns = ["_env_order"]
    if "env_file" in ordered.columns:
        sort_columns.append("env_file")
    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    return ordered.head(sample_limit).drop(columns=["_env_order"]).reset_index(drop=True)


def resolve_result_file(method: str, protocol: str, config_name: str) -> Optional[Path]:
    filename = f"{config_name}_results.json"
    candidates: List[Path] = []
    for base_dir in METHOD_SOURCE_CANDIDATES.get(method, []):
        path = base_dir / protocol / filename
        if path.exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(df[column], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan)


def compute_metric_series(df: pd.DataFrame, metric_key: str) -> pd.Series:
    finished = numeric_series(df, "finished_tasks")
    makespan = numeric_series(df, "makespan").replace(0, np.nan)

    if metric_key == "success_rate":
        series = numeric_series(df, "success_rate")
    elif metric_key == "deadline_satisfaction_rate":
        series = numeric_series(df, "deadline_satisfaction_rate")
    elif metric_key == "avg_deadline_violation":
        series = numeric_series(df, "avg_deadline_violation")
    elif metric_key == "makespan":
        series = numeric_series(df, "makespan")
    elif metric_key == "finished_tasks":
        series = finished
    elif metric_key == "on_time_finished_tasks":
        series = finished * numeric_series(df, "deadline_satisfaction_rate")
    elif metric_key == "task_throughput":
        series = finished / makespan
    elif metric_key == "wait_per_finished_task":
        series = numeric_series(df, "waiting_time") / finished.replace(0, np.nan)
    elif metric_key == "waiting_time":
        series = numeric_series(df, "waiting_time")
    elif metric_key == "travel_per_finished_task":
        series = numeric_series(df, "total_travel_distance") / finished.replace(0, np.nan)
    elif metric_key == "total_travel_distance":
        series = numeric_series(df, "total_travel_distance")
    elif metric_key == "avg_flow_time":
        series = numeric_series(df, "avg_flow_time")
    elif metric_key == "planning_time_per_finished_task_ms":
        series = 1000.0 * numeric_series(df, "total_planning_time") / finished.replace(0, np.nan)
    elif metric_key == "total_planning_time":
        series = numeric_series(df, "total_planning_time")
    else:
        series = pd.Series(dtype=float)

    return series.replace([np.inf, -np.inf], np.nan).dropna()


def aggregate_metric(df: pd.DataFrame, metric_key: str) -> Optional[Tuple[float, float]]:
    series = compute_metric_series(df, metric_key)
    if series.empty:
        return None
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return float(series.mean()), std


def format_stat(stat: Optional[Tuple[float, float]], spec: MetricSpec) -> str:
    if stat is None:
        return "--"
    mean, std = stat
    if spec.percent:
        mean *= 100.0
        std *= 100.0
    if not spec.show_std:
        return f"{mean:.{spec.decimals}f}"
    return f"{mean:.{spec.decimals}f}$\\pm${std:.{spec.decimals}f}"


def case_label(protocol: str, n_agents: int, task_value: int) -> str:
    prefix = "FT" if protocol == "Fixed_Tasks" else "FM"
    return f"{prefix} {n_agents}+{task_value}"


def compact_case_label(protocol: str, n_agents: int, task_value: int) -> str:
    return f"{n_agents}+{task_value}"


def dynamic_tasks(config_name: str) -> int:
    cfg = CONFIG_DETAILS[config_name]
    return cfg["total_tasks"] - cfg["initial_tasks"]


def instance_case_cell(config_name: str) -> str:
    cfg = CONFIG_DETAILS[config_name]
    return (
        "\\multirow{6}{*}{\\makecell{"
        f"$A={cfg['agents']}$\\\\"
        f"$S={cfg['species']}$\\\\"
        f"$I={cfg['initial_tasks']}$\\\\"
        f"$D={dynamic_tasks(config_name)}$"
        "}}"
    )


def load_stat_map() -> StatMap:
    stat_map: StatMap = {}
    for method in METHOD_ORDER:
        if method not in METHOD_SOURCE_CANDIDATES:
            continue
        stat_map[method] = {}
        for protocol, configs in PROTOCOL_CONFIGS.items():
            stat_map[method][protocol] = {}
            for config_name, *_ in configs:
                path = resolve_result_file(method, protocol, config_name)
                if path is None:
                    stat_map[method][protocol][config_name] = None
                    continue
                raw_df = load_results(path)
                if raw_df is None:
                    stat_map[method][protocol][config_name] = None
                    continue
                df = select_table_samples(raw_df, protocol)
                stat_map[method][protocol][config_name] = ResolvedResult(
                    path=path,
                    mtime=path.stat().st_mtime,
                    df=df,
                    source_instances=len(raw_df),
                    evaluated_instances=len(df),
                )
    return stat_map


def get_metric_stat(
    stat_map: StatMap,
    method: str,
    protocol: str,
    config_name: str,
    metric_key: str,
) -> Optional[Tuple[float, float]]:
    if method not in stat_map:
        return None
    resolved = stat_map[method][protocol][config_name]
    if resolved is None:
        return None
    return aggregate_metric(resolved.df, metric_key)


def best_methods_for_case(
    stat_map: StatMap,
    protocol: str,
    config_name: str,
    spec: MetricSpec,
) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for method in METHOD_ORDER:
        if method not in stat_map:
            continue
        stat = get_metric_stat(stat_map, method, protocol, config_name, spec.key)
        if stat is None:
            continue
        scored.append((method, stat[0]))
    if not scored:
        return []
    if spec.higher_is_better:
        best_value = max(value for _, value in scored)
    else:
        best_value = min(value for _, value in scored)
    tol = 1e-12
    return [method for method, value in scored if abs(value - best_value) <= tol]


def render_metric_cell(
    stat_map: StatMap,
    method: str,
    protocol: str,
    config_name: str,
    spec: MetricSpec,
    best_methods: List[str],
) -> str:
    stat = get_metric_stat(stat_map, method, protocol, config_name, spec.key)
    text = format_stat(stat, spec)
    if method in best_methods and text != "--":
        return f"\\textbf{{{text}}}"
    return text


def build_summary_rows(stat_map: StatMap) -> List[dict]:
    rows: List[dict] = []
    for protocol, configs in PROTOCOL_CONFIGS.items():
        for config_name, n_agents, n_species, task_value in configs:
            case = case_label(protocol, n_agents, task_value)
            for method in METHOD_ORDER:
                resolved = None
                if method in stat_map:
                    resolved = stat_map[method][protocol][config_name]

                row = {
                    "protocol": protocol,
                    "protocol_title": PROTOCOL_TITLES[protocol],
                    "config_name": config_name,
                    "case_label": case,
                    "agents": n_agents,
                    "species": n_species,
                    "task_value": task_value,
                    "initial_tasks": CONFIG_DETAILS[config_name]["initial_tasks"],
                    "total_tasks": CONFIG_DETAILS[config_name]["total_tasks"],
                    "dynamic_tasks": dynamic_tasks(config_name),
                    "method": method,
                    "source_path": "" if resolved is None else str(resolved.path.relative_to(ROOT)),
                    "source_mtime": "" if resolved is None else datetime.fromtimestamp(resolved.mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "source_instances": "" if resolved is None else resolved.source_instances,
                    "evaluated_instances": "" if resolved is None else resolved.evaluated_instances,
                }
                for spec in SUMMARY_METRICS:
                    stat = get_metric_stat(stat_map, method, protocol, config_name, spec.key)
                    if stat is None:
                        row[f"{spec.key}_mean"] = None
                        row[f"{spec.key}_std"] = None
                        row[f"{spec.key}_text"] = "--"
                    else:
                        mean, std = stat
                        if spec.percent:
                            mean *= 100.0
                            std *= 100.0
                        row[f"{spec.key}_mean"] = mean
                        row[f"{spec.key}_std"] = std
                        row[f"{spec.key}_text"] = format_stat(stat, spec)
                rows.append(row)
    return rows


def build_source_rows(stat_map: StatMap) -> List[dict]:
    rows: List[dict] = []
    for method in METHOD_ORDER:
        if method not in stat_map:
            continue
        for protocol, configs in PROTOCOL_CONFIGS.items():
            for config_name, n_agents, n_species, task_value in configs:
                resolved = stat_map[method][protocol][config_name]
                rows.append({
                    "method": method,
                    "protocol": protocol,
                    "protocol_title": PROTOCOL_TITLES[protocol],
                    "config_name": config_name,
                    "case_label": case_label(protocol, n_agents, task_value),
                    "agents": n_agents,
                    "species": n_species,
                    "task_value": task_value,
                    "initial_tasks": CONFIG_DETAILS[config_name]["initial_tasks"],
                    "total_tasks": CONFIG_DETAILS[config_name]["total_tasks"],
                    "dynamic_tasks": dynamic_tasks(config_name),
                    "source_path": "" if resolved is None else str(resolved.path.relative_to(ROOT)),
                    "source_mtime": "" if resolved is None else datetime.fromtimestamp(resolved.mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "source_instances": "" if resolved is None else resolved.source_instances,
                    "evaluated_instances": "" if resolved is None else resolved.evaluated_instances,
                })
    return rows


def render_ral_compact_table(stat_map: StatMap) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Performance on the dynamic SA-BT benchmark under fixed-task and fixed-horizon protocols. "
        "The fixed-horizon setting uses a shared execution horizon $H=120$. "
        "Execution-time metrics are reported in simulation-time units, whereas planning time is reported in wall-clock seconds. "
        "SR is averaged over 50 test instances, and the remaining metrics are reported as mean$\\pm$standard deviation.}",
        "\\label{tab:sabt_compact_ral}",
        "\\centering",
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{0.95pt}",
        "\\renewcommand{\\arraystretch}{1.02}",
        r"\begin{tabular}{@{}llccccc@{\hspace{0.8mm}}lccccc@{}}",
        "\\toprule",
        "\\multicolumn{7}{c}{Fixed Task Count} & \\multicolumn{6}{c}{Fixed Makespan ($H=120$)} \\\\",
        "\\cmidrule(lr){1-7}\\cmidrule(lr){8-13}",
        "\\makecell{Instance\\\\Config.} & Model & \\makecell{Success\\\\Rate (\\%)} & "
        "\\makecell{Deadline\\\\Satisfaction\\\\Rate (\\%)} & "
        "\\makecell{Avg. Deadline\\\\Violation} & \\makecell{Flow\\\\Time} & "
        "\\makecell{Total\\\\Planning\\\\Time (s)} & "
        "\\makecell{Instance\\\\Config.} & \\makecell{Success\\\\Rate (\\%)} & "
        "\\makecell{Deadline\\\\Satisfaction\\\\Rate (\\%)} & "
        "\\makecell{Avg. Deadline\\\\Violation} & \\makecell{Waiting\\\\Time} & "
        "\\makecell{Total\\\\Planning\\\\Time (s)} \\\\",
        "\\midrule",
    ]

    left_rows = PROTOCOL_CONFIGS["Fixed_Tasks"]
    right_rows = PROTOCOL_CONFIGS["Fixed_Makespan"]

    for pair_idx, (left_cfg, right_cfg) in enumerate(zip(left_rows, right_rows)):
        left_name, *_left_unused = left_cfg
        right_name, *_right_unused = right_cfg

        left_case = instance_case_cell(left_name)
        right_case = instance_case_cell(right_name)

        left_best = {
            spec.key: best_methods_for_case(stat_map, "Fixed_Tasks", left_name, spec)
            for spec in FIXED_TASK_TABLE_METRICS
        }
        right_best = {
            spec.key: best_methods_for_case(stat_map, "Fixed_Makespan", right_name, spec)
            for spec in FIXED_MAKESPAN_TABLE_METRICS
        }

        for row_idx, method in enumerate(DISPLAY_METHODS):
            left_label = left_case if row_idx == 0 else ""
            right_label = right_case if row_idx == 0 else ""
            left_cells = [
                render_metric_cell(
                    stat_map=stat_map,
                    method=method,
                    protocol="Fixed_Tasks",
                    config_name=left_name,
                    spec=spec,
                    best_methods=left_best[spec.key],
                )
                for spec in FIXED_TASK_TABLE_METRICS
            ]
            right_cells = [
                render_metric_cell(
                    stat_map=stat_map,
                    method=method,
                    protocol="Fixed_Makespan",
                    config_name=right_name,
                    spec=spec,
                    best_methods=right_best[spec.key],
                )
                for spec in FIXED_MAKESPAN_TABLE_METRICS
            ]
            lines.append(
                f"{left_label} & {method} & "
                + " & ".join(left_cells)
                + f" & {right_label} & "
                + " & ".join(right_cells)
                + " \\\\"
            )

        if pair_idx < len(left_rows) - 1:
            lines.append("\\addlinespace[1.5pt]")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\endgroup",
        "\\end{table*}",
        "",
    ])
    return "\n".join(lines)


def build_tex_document(stat_map: StatMap) -> str:
    return "\n".join([
        "% Auto-generated by tools/generate_sa_bt_ieee_tables.py",
        "% Notes:",
        "% 1. This file renders a compact paper table instead of the exhaustive summary table.",
        "% 2. The instance columns report A/S/I/D: agents/species/initial tasks/dynamic tasks.",
        "% 3. Fixed Makespan rows correspond to a shared execution horizon H=120.",
        "% 4. The two protocol blocks use different metrics because their discriminative signals differ.",
        "% 5. For each method/config, the newest available local result file is used.",
        "% 6. Execution-time metrics use simulation-time units; planning time uses wall-clock seconds.",
        "% 7. Each method/config is aggregated over the first 50 env-indexed instances.",
        "",
        render_ral_compact_table(stat_map),
    ])


def build_preview_document() -> str:
    return "\n".join([
        "\\documentclass[journal]{IEEEtran}",
        "\\usepackage{booktabs}",
        "\\usepackage{makecell}",
        "\\usepackage{multirow}",
        "\\usepackage[margin=0.7in]{geometry}",
        "\\begin{document}",
        "\\input{sa_bt_baseline_ieee_generated.tex}",
        "\\end{document}",
        "",
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stat_map = load_stat_map()
    OUTPUT_TEX.write_text(build_tex_document(stat_map), encoding="utf-8")
    PREVIEW_TEX.write_text(build_preview_document(), encoding="utf-8")
    pd.DataFrame(build_summary_rows(stat_map)).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(build_source_rows(stat_map)).to_csv(SOURCE_CSV, index=False)
    print(f"Generated table body: {OUTPUT_TEX}")
    print(f"Generated preview doc: {PREVIEW_TEX}")
    print(f"Generated summary csv: {SUMMARY_CSV}")
    print(f"Generated source manifest: {SOURCE_CSV}")


if __name__ == "__main__":
    main()
