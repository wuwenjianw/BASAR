#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SA-AT 评估脚本的断点续跑辅助函数。"""

import json
import pickle


def build_resume_signature(
    config_name,
    arrival_rate,
    simulation_time_limit,
    method_tag,
    model_folder=None,
    dynamic_task_options=None,
):
    signature = {
        "config_name": config_name,
        "arrival_rate": arrival_rate,
        "simulation_time_limit": simulation_time_limit,
        "method": method_tag,
        "dynamic_task_options": dynamic_task_options or {},
    }
    if model_folder is not None:
        signature["model_folder"] = model_folder
    return signature


def load_matching_results(results_json_path, signature):
    if not results_json_path.exists():
        return {}, None

    try:
        with open(results_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"⚠ 无法读取已有结果文件，将重新计算: {results_json_path} ({exc})"

    if not isinstance(payload, list):
        return {}, f"⚠ 结果文件格式异常，将重新计算: {results_json_path}"

    results_by_env = {}
    mismatched_count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        env_file = item.get("env_file")
        if not env_file:
            continue
        if all(item.get(key) == value for key, value in signature.items()):
            results_by_env[env_file] = item
        else:
            mismatched_count += 1

    warning = None
    if mismatched_count:
        warning = (
            f"⚠ {results_json_path.name}: {mismatched_count} 条已有结果与当前参数不匹配，"
            "本次不会复用。"
        )
    return results_by_env, warning


def select_results_for_env_files(env_files, results_by_env):
    selected_results = []
    for env_file in env_files:
        result = results_by_env.get(env_file.name)
        if result is not None:
            selected_results.append(result)
    return selected_results


def save_results_snapshot(output_dir, config_name, results_by_env):
    ordered_results = [results_by_env[env_name] for env_name in sorted(results_by_env)]

    with open(output_dir / f"{config_name}_results.pkl", "wb") as f:
        pickle.dump(ordered_results, f)
    with open(output_dir / f"{config_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2)
