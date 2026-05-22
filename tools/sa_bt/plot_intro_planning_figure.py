#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot an introduction figure highlighting planning-time contrast on SA-BT."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
import torch

from ctasd_static_planner import CTASDStaticRoutePlanner
from dynamic_worker import create_dynamic_model
from env.task_env import TaskEnv
from parameters import EnvParams, SaverParams, TrainParams
from project_paths import DOCS_ROOT, REPO_ROOT, ensure_checkpoint_exists
from scripts.sa_bt.evaluate_ctasd_dynamic import (
    load_env as load_ctasd_env,
    run_ctasd_dynamic,
)
from scripts.sa_bt.evaluate_my_model_dynamic import (
    extract_model_state_dict,
    load_env as load_ours_env,
    run_model_dynamic,
)


FONT_SIZE = 16
LEGEND_FONT_SIZE = 13
OURS_COLOR = "#EE6C4D"
CTASD_COLOR = "#3D5A80"
STATIC_COLOR = "#D9F0FF"
DYNAMIC_COLOR = "#FFBE5C"
DEPOT_COLOR = "#1F2937"
PANEL_BG_TOP = np.array([250, 244, 236]) / 255.0
PANEL_BG_BOTTOM = np.array([234, 244, 255]) / 255.0
REPRESENTATIVE_ROUTE_COUNT = 3

OUTPUT_STEM = DOCS_ROOT / "figures" / "sa_bt_intro_planning"
PANEL_CONFIGS = [
    {
        "config_name": "n15_s5_h30",
        "panel_title": "30 Static + 20 Dynamic Tasks",
        "total_tasks": 50,
        "arrival_rate": 3,
    },
    {
        "config_name": "n15_s5_h30",
        "panel_title": "30 Static + 70 Dynamic Tasks",
        "total_tasks": 100,
        "arrival_rate": 3,
    },
]


def infer_input_dims() -> tuple[int, int]:
    env = TaskEnv(
        EnvParams.SPECIES_AGENTS_RANGE,
        EnvParams.SPECIES_RANGE,
        EnvParams.TASKS_RANGE,
        EnvParams.TRAIT_DIM,
        EnvParams.DECISION_DIM,
    )
    agent_id = next(iter(env.agent_dic.keys()))
    tasks_info, agents_info, _ = env.agent_observe(agent_id, False)
    return int(agents_info.shape[-1]), int(tasks_info.shape[-1])


def load_model(device: torch.device):
    agent_input_dim, task_input_dim = infer_input_dims()
    model = create_dynamic_model(
        agent_input_dim=agent_input_dim,
        task_input_dim=task_input_dim,
        embedding_dim=TrainParams.EMBEDDING_DIM,
        device=device,
    )
    checkpoint = torch.load(
        ensure_checkpoint_exists(SaverParams.FOLDER_NAME, method_label=TrainParams.MODEL_NAME),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.eval()
    return model


def build_route_points(agent: Dict, task_dic: Dict[int, Dict], route: Iterable[int]) -> List[List[float]]:
    depot = agent["depot"]
    points: List[List[float]] = [[float(depot[0]), float(depot[1])]]
    for task_id in route:
        if task_id < 0 or task_id not in task_dic:
            continue
        loc = task_dic[task_id]["location"]
        points.append([float(loc[0]), float(loc[1])])
    points.append([float(depot[0]), float(depot[1])])
    return points


def route_score(agent: Dict, task_dic: Dict[int, Dict], route: List[int]) -> tuple[int, float]:
    if not route:
        return (0, 0.0)
    points = build_route_points(agent, task_dic, route)
    total_distance = 0.0
    for start, end in zip(points[:-1], points[1:]):
        total_distance += float(np.hypot(end[0] - start[0], end[1] - start[1]))
    return (len(route), total_distance)


def select_representative_agents(env, routes: Dict[int, List[int]], top_k: int = REPRESENTATIVE_ROUTE_COUNT) -> List[int]:
    scored = []
    for agent_id, route in routes.items():
        if not route:
            continue
        scored.append((route_score(env.agent_dic[agent_id], env.task_dic, route), agent_id))
    scored.sort(reverse=True)
    return [agent_id for _, agent_id in scored[:top_k]]


def select_shared_representative_agents(
    env,
    ours_routes: Dict[int, List[int]],
    ctasd_routes: Dict[int, List[int]],
    top_k: int = REPRESENTATIVE_ROUTE_COUNT,
) -> List[int]:
    scored = []
    for agent_id in env.agent_dic.keys():
        ours_score = route_score(env.agent_dic[agent_id], env.task_dic, ours_routes.get(agent_id, []))
        ctasd_score = route_score(env.agent_dic[agent_id], env.task_dic, ctasd_routes.get(agent_id, []))
        max_tasks = max(ours_score[0], ctasd_score[0])
        max_distance = max(ours_score[1], ctasd_score[1])
        if max_tasks == 0:
            continue
        scored.append(((max_tasks, max_distance), agent_id))
    scored.sort(reverse=True)
    return [agent_id for _, agent_id in scored[:top_k]]


def collect_panel_payload(model, device: torch.device, config_name: str, total_tasks: int, arrival_rate: float):
    env_path = REPO_ROOT / "data" / "testsets" / "sa_bt" / "Fixed_Tasks" / config_name / "env_000.pkl"
    ours_env = load_ours_env(env_path)
    ours_results = run_model_dynamic(
        ours_env,
        model,
        device,
        max_total_tasks=total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=10000,
        random_seed=42,
    )
    ctasd_env = load_ctasd_env(env_path)
    ctasd_results = run_ctasd_dynamic(
        ctasd_env,
        CTASDStaticRoutePlanner(),
        max_total_tasks=total_tasks,
        arrival_rate=arrival_rate,
        simulation_time_limit=10000,
        random_seed=42,
    )

    ours_routes = {
        agent_id: [task_id for task_id in agent.get("route", []) if isinstance(task_id, int) and task_id >= 0]
        for agent_id, agent in ours_env.agent_dic.items()
    }
    ctasd_routes = {
        agent_id: [task_id for task_id in agent.get("route", []) if isinstance(task_id, int) and task_id >= 0]
        for agent_id, agent in ctasd_env.agent_dic.items()
    }

    static_points = []
    dynamic_points = []
    for task in ours_env.task_dic.values():
        point = [float(task["location"][0]), float(task["location"][1])]
        if task.get("is_dynamic", False):
            dynamic_points.append(point)
        else:
            static_points.append(point)

    depots = [
        [float(agent["depot"][0]), float(agent["depot"][1])]
        for agent in ours_env.agent_dic.values()
    ]

    return {
        "env": ours_env,
        "ours_routes": ours_routes,
        "ctasd_routes": ctasd_routes,
        "ours_time": float(ours_results["total_inference_time"]),
        "ctasd_time": float(ctasd_results["total_inference_time"]),
        "static_points": static_points,
        "dynamic_points": dynamic_points,
        "depots": depots,
    }


def add_panel_background(ax) -> None:
    x = np.linspace(0.0, 1.0, 256)
    y = np.linspace(0.0, 1.0, 256)
    mix = (np.outer(np.ones_like(y), x) + np.outer(y, np.ones_like(x))) / 2.0
    gradient = PANEL_BG_TOP[None, None, :] * (1.0 - mix[:, :, None]) + PANEL_BG_BOTTOM[None, None, :] * mix[:, :, None]
    ax.imshow(
        gradient,
        extent=(-0.02, 1.02, -0.02, 1.02),
        origin="lower",
        aspect="auto",
        zorder=0,
        alpha=1.0,
    )
    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        linewidth=1.5,
        edgecolor="#243042",
        facecolor="none",
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(card)


def plot_method_routes(
    ax,
    env,
    routes: Dict[int, List[int]],
    color: str,
    linestyle: str,
    representative_agents: List[int],
    base_alpha: float,
    highlight_alpha: float,
) -> None:
    for agent_id, route in routes.items():
        if not route:
            continue
        agent = env.agent_dic[agent_id]
        points = build_route_points(agent, env.task_dic, route)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        is_representative = agent_id in representative_agents
        line = ax.plot(
            xs,
            ys,
            color=color,
            linestyle=linestyle,
            linewidth=2.8 if is_representative else 1.0,
            alpha=highlight_alpha if is_representative else base_alpha,
            zorder=4 if is_representative else 1,
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
        )[0]
        if is_representative:
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=5.4, foreground="white", alpha=0.72),
                    pe.Normal(),
                ]
            )


def style_axis(ax) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_annotations(ax, title: str, ours_time: float, ctasd_time: float) -> None:
    speedup = ctasd_time / max(ours_time, 1e-9)
    ax.text(
        0.03,
        0.965,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=8,
    )
    ax.text(
        0.97,
        0.965,
        f"{speedup:.0f}x Faster",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE - 1,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=OURS_COLOR, edgecolor="none", alpha=0.96),
        zorder=8,
    )
    ax.text(
        0.03,
        0.085,
        f"Ours: {ours_time:.2f} s\nCTAS-D: {ctasd_time:.2f} s",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE - 2,
        color="#243042",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.9),
        zorder=8,
    )


def build_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=STATIC_COLOR,
               markeredgecolor="#355070", markeredgewidth=1.0, markersize=9, label="Static Task"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor=DYNAMIC_COLOR,
               markeredgecolor="#7C4F00", markeredgewidth=1.0, markersize=8, label="Dynamic Task"),
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor=DEPOT_COLOR,
               markeredgecolor=DEPOT_COLOR, markersize=12, label="Depot"),
        Line2D([0], [0], color=OURS_COLOR, linewidth=3.0, label="Ours"),
        Line2D([0], [0], color=CTASD_COLOR, linewidth=3.0, linestyle="--", label="CTAS-D"),
    ]
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.9,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def main() -> None:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": 11,
    })

    device = torch.device("cpu")
    model = load_model(device)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))

    for ax, panel in zip(axes, PANEL_CONFIGS):
        payload = collect_panel_payload(model, device, panel["config_name"], panel["total_tasks"], panel["arrival_rate"])
        ours_time = payload["ours_time"]
        ctasd_time = payload["ctasd_time"]
        add_panel_background(ax)

        static_points = payload["static_points"]
        dynamic_points = payload["dynamic_points"]
        depots = payload["depots"]
        shared_rep_agents = select_shared_representative_agents(
            payload["env"],
            payload["ours_routes"],
            payload["ctasd_routes"],
        )

        if static_points:
            ax.scatter(
                [p[0] for p in static_points],
                [p[1] for p in static_points],
                s=120,
                marker="o",
                facecolors=STATIC_COLOR,
                edgecolors="none",
                alpha=0.28,
                zorder=2,
            )
            ax.scatter(
                [p[0] for p in static_points],
                [p[1] for p in static_points],
                s=60,
                marker="o",
                facecolors=STATIC_COLOR,
                edgecolors="#4F6D7A",
                linewidths=1.0,
                zorder=3,
            )
        if dynamic_points:
            ax.scatter(
                [p[0] for p in dynamic_points],
                [p[1] for p in dynamic_points],
                s=135,
                marker="D",
                facecolors=DYNAMIC_COLOR,
                edgecolors="none",
                alpha=0.20,
                zorder=2,
            )
            ax.scatter(
                [p[0] for p in dynamic_points],
                [p[1] for p in dynamic_points],
                s=64,
                marker="D",
                facecolors=DYNAMIC_COLOR,
                edgecolors="#7C4F00",
                linewidths=1.0,
                zorder=3,
            )
        if depots:
            ax.scatter(
                [p[0] for p in depots],
                [p[1] for p in depots],
                s=220,
                marker="*",
                facecolors="#94A3B8",
                edgecolors="none",
                alpha=0.24,
                zorder=3,
            )
            ax.scatter(
                [p[0] for p in depots],
                [p[1] for p in depots],
                s=115,
                marker="*",
                facecolors=DEPOT_COLOR,
                edgecolors=DEPOT_COLOR,
                linewidths=0.8,
                zorder=4,
            )

        plot_method_routes(
            ax,
            payload["env"],
            payload["ctasd_routes"],
            CTASD_COLOR,
            "--",
            representative_agents=shared_rep_agents,
            base_alpha=0.025,
            highlight_alpha=0.76,
        )
        plot_method_routes(
            ax,
            payload["env"],
            payload["ours_routes"],
            OURS_COLOR,
            "-",
            representative_agents=shared_rep_agents,
            base_alpha=0.03,
            highlight_alpha=0.96,
        )

        style_axis(ax)
        add_panel_annotations(ax, title=panel["panel_title"], ours_time=ours_time, ctasd_time=ctasd_time)

    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), w_pad=2.0)
    build_legend(fig)

    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
