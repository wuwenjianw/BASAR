#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于动态评估框架的简单双阶段 Greedy route planner。

目标：
- 与 evaluate_ctasd_dynamic.py 的 route-planner 接口兼容
- 在统一事件循环里提供纯 Greedy 任务选择
- 不依赖额外求解器，便于大规模 SA-AT benchmark 复用
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class GreedyStaticRoutePlanner:
    """按最短 travel + service cost 排序的 Greedy 静态 route 规划器。"""

    @staticmethod
    def _visible_open_task_ids(env) -> List[int]:
        task_ids = []
        current_time = float(env.current_time)
        for task_id, task in env.task_dic.items():
            if task.get("finished", False):
                continue
            if task.get("feasible_assignment", False):
                continue
            if float(task.get("appear_time", 0.0)) > current_time:
                continue
            remaining = np.asarray(task.get("status", task.get("requirements", [])), dtype=float)
            if remaining.size == 0 or not np.any(remaining > 1e-9):
                continue
            task_ids.append(int(task_id))
        return task_ids

    @staticmethod
    def _agent_can_contribute(agent, remaining_status) -> bool:
        abilities = np.asarray(agent.get("abilities", []), dtype=float)
        remaining = np.asarray(remaining_status, dtype=float)
        if abilities.size == 0 or remaining.size == 0:
            return False
        return bool(np.any(np.minimum(abilities, remaining) > 1e-9))

    @staticmethod
    def _task_cost(agent, task, from_loc) -> float:
        velocity = max(float(agent.get("velocity", 0.2)), 1e-8)
        travel_time = float(
            np.linalg.norm(np.asarray(from_loc, dtype=float) - np.asarray(task["location"], dtype=float))
        ) / velocity
        service_time = float(task.get("time", task.get("duration", 0.0)))
        return travel_time + service_time

    def _rank_tasks(self, agent_id: int, env, candidate_task_ids: Sequence[int]) -> List[int]:
        agent = env.agent_dic[agent_id]
        current_loc = np.asarray(agent["location"], dtype=float)
        ranked = []

        for task_id in candidate_task_ids:
            task = env.task_dic[task_id]
            remaining = task.get("status", task.get("requirements", []))

            if not self._agent_can_contribute(agent, remaining):
                continue
            if not env.can_reach_with_battery(agent_id, task):
                continue

            score = self._task_cost(agent, task, current_loc)
            deadline = float(task.get("deadline", np.inf))
            ranked.append((score, deadline, int(task_id)))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [task_id for _, _, task_id in ranked]

    def plan_routes(self, agent_ids: Sequence[int], env) -> Dict[int, List[int]]:
        """对每个 agent 返回一条按 Greedy 顺序排序的候选 route。"""
        visible_tasks = self._visible_open_task_ids(env)
        return {
            int(agent_id): self._rank_tasks(int(agent_id), env, visible_tasks)
            for agent_id in dict.fromkeys(int(aid) for aid in agent_ids)
        }

    def select_task(
        self,
        agent_id: int,
        env,
        mask_bool,
        planned_route: Sequence[int],
    ) -> Optional[int]:
        """优先按 route 取第一个可执行任务，失败时回退到在线 greedy 排序。"""
        task_index_map = {task_id: idx + 1 for idx, task_id in enumerate(env.task_dic.keys())}

        for task_id in planned_route:
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if not bool(mask_bool[0, mask_idx]):
                return int(task_id)

        fallback_route = self._rank_tasks(agent_id, env, self._visible_open_task_ids(env))
        for task_id in fallback_route:
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if not bool(mask_bool[0, mask_idx]):
                return int(task_id)

        return None

