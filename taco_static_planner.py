#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TACO 静态 route 逻辑的本地动态适配版。

说明：
- 原论文中的 TACO 是离线、集中式、ant-as-robot 的 ACO 方法。
- 本文件把它适配到当前仓库的动态 SA-BT 评估框架中：
  在每个动态重规划时刻，把当前已揭示任务视为一个静态子问题，
  用 TACO 风格的构解 + 信息素更新生成多智能体 route。
- 由于动态评估会频繁重规划，这里使用了缩减后的 ACO 预算，并直接优化
  论文中用于信息素更新的标量目标 `f1 + R * f2`，而不是显式维护完整 Pareto 集。
  这样更适合在线反复调用，同时保留了论文核心机制：
  greedy seeding、argmin(path_cost) 轮转、tau^alpha * eta^beta 选点、
  willingness、deadlock reversal、pheromone evaporation / reinforcement。
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class TACOStaticRoutePlanner:
    START_NODE = "__start__"

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        rho: float = 0.01,
        p0: float = 0.9,
        willingness: float = 0.9,
        unresolved_penalty: float = 50.0,
        n_iterations: Optional[int] = None,
        n_solutions: Optional[int] = None,
        random_seed: int = 42,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.p0 = float(p0)
        self.willingness = float(willingness)
        self.unresolved_penalty = float(unresolved_penalty)
        self.n_iterations = n_iterations
        self.n_solutions = n_solutions
        self.rng = np.random.default_rng(random_seed)

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
            status = np.asarray(task.get("status", task.get("requirements", [])), dtype=float)
            if status.size == 0 or not np.any(status > 1e-9):
                continue
            task_ids.append(int(task_id))
        return task_ids

    @staticmethod
    def _travel_time(from_loc, to_loc, velocity: float) -> float:
        velocity = max(float(velocity), 1e-8)
        distance = float(np.linalg.norm(np.asarray(from_loc, dtype=float) - np.asarray(to_loc, dtype=float)))
        return distance / velocity

    @staticmethod
    def _effective_overlap(agent_abilities, remaining_status) -> float:
        abilities = np.asarray(agent_abilities, dtype=float)
        remaining = np.asarray(remaining_status, dtype=float)
        if abilities.size == 0 or remaining.size == 0:
            return 0.0
        return float(np.minimum(np.maximum(abilities, 0.0), np.maximum(remaining, 0.0)).sum())

    @staticmethod
    def _task_missing_after_waiters(env, task_remaining, waiting_agents: Sequence[int]) -> np.ndarray:
        if not waiting_agents:
            return np.asarray(task_remaining, dtype=float).copy()
        waiting_ability = np.asarray(env.get_abilities(list(waiting_agents)), dtype=float)
        return np.maximum(np.asarray(task_remaining, dtype=float) - waiting_ability, 0.0)

    def _get_fixed_member_arrivals(self, env, task_id: int, planning_agent_ids: Sequence[int]) -> List[float]:
        task = env.task_dic[task_id]
        planning_agent_ids = set(int(aid) for aid in planning_agent_ids)
        arrivals: List[float] = []
        for member in task.get("members", []):
            if member in planning_agent_ids:
                continue
            if member not in env.agent_dic:
                continue
            arrival = float(env.get_arrival_time(member, task_id))
            if not np.isfinite(arrival):
                agent = env.agent_dic[member]
                arrival = float(env.current_time) + self._travel_time(
                    agent.get("location", task["location"]),
                    task["location"],
                    agent.get("velocity", 0.2),
                )
            arrivals.append(arrival)
        return arrivals

    def _adaptive_budget(self, n_agents: int, n_tasks: int) -> Tuple[int, int]:
        if self.n_iterations is not None and self.n_solutions is not None:
            return int(self.n_iterations), int(self.n_solutions)

        scale = max(int(n_tasks), int(n_agents))
        if scale <= 12:
            return 20, 8
        if scale <= 24:
            return 12, 6
        if scale <= 40:
            return 8, 4
        if scale <= 64:
            return 6, 3
        return 4, 2

    def _initialize_route_state(self, agent_ids: Sequence[int], env):
        route_state = {}
        for aid in agent_ids:
            agent = env.agent_dic[aid]
            route_state[aid] = {
                "route": [],
                "current_loc": np.asarray(agent["location"], dtype=float).copy(),
                "current_cost": 0.0,
                "battery": float(agent.get("battery", 0.0)),
                "last_node": self.START_NODE,
                "pending_task": None,
                "pending_arrival": None,
            }
        return route_state

    def _candidate_tasks_for_agent(
        self,
        agent_id: int,
        env,
        task_ids: Sequence[int],
        completed_tasks: set,
        task_remaining: Dict[int, np.ndarray],
        waiting_lists: Dict[int, List[int]],
    ) -> List[int]:
        agent = env.agent_dic[agent_id]
        candidates = []
        for task_id in task_ids:
            if task_id in completed_tasks:
                continue
            missing = self._task_missing_after_waiters(env, task_remaining[task_id], waiting_lists[task_id])
            if not np.any(missing > 1e-9):
                continue
            if self._effective_overlap(agent.get("abilities", []), missing) <= 1e-9:
                continue
            candidates.append(task_id)
        return candidates

    def _edge_cost(self, agent_state, agent, task) -> float:
        travel_time = self._travel_time(
            agent_state["current_loc"],
            task["location"],
            agent.get("velocity", 0.2),
        )
        return float(travel_time + float(task.get("time", 0.0)))

    def _select_next_task(
        self,
        agent_id: int,
        agent_index: Dict[int, int],
        env,
        candidates: Sequence[int],
        route_state,
        pheromone: Optional[np.ndarray],
        from_index: Dict[object, int],
        to_index: Dict[int, int],
        greedy: bool,
    ) -> Optional[int]:
        if not candidates:
            return None

        agent = env.agent_dic[agent_id]
        state = route_state[agent_id]

        if greedy:
            best_task = None
            best_cost = float("inf")
            for task_id in candidates:
                task = env.task_dic[task_id]
                cost = self._edge_cost(state, agent, task)
                if cost < best_cost:
                    best_cost = cost
                    best_task = task_id
            return best_task

        weights = []
        for task_id in candidates:
            task = env.task_dic[task_id]
            edge_cost = max(self._edge_cost(state, agent, task), 1e-8)
            heuristic = 1.0 / edge_cost
            from_node = state["last_node"]
            tau = float(pheromone[from_index[from_node], to_index[task_id], agent_index[agent_id]])
            weight = (tau ** self.alpha) * (heuristic ** self.beta)
            weights.append(max(weight, 1e-12))

        weights = np.asarray(weights, dtype=float)
        if weights.size == 0:
            return None
        if self.rng.random() < self.p0:
            return int(candidates[int(np.argmax(weights))])

        prob = weights / max(float(weights.sum()), 1e-12)
        selected_idx = int(self.rng.choice(len(candidates), p=prob))
        return int(candidates[selected_idx])

    def _claim_task(
        self,
        agent_id: int,
        task_id: int,
        env,
        route_state,
        waiting_lists,
        waiting_arrivals,
    ) -> None:
        agent = env.agent_dic[agent_id]
        task = env.task_dic[task_id]
        state = route_state[agent_id]

        travel_time = self._travel_time(
            state["current_loc"],
            task["location"],
            agent.get("velocity", 0.2),
        )
        state["current_cost"] += travel_time
        state["current_loc"] = np.asarray(task["location"], dtype=float).copy()
        state["battery"] = max(
            0.0,
            float(state["battery"]) - float(env.battery_consume_moving) * float(travel_time),
        )
        state["pending_task"] = int(task_id)
        state["pending_arrival"] = float(env.current_time + state["current_cost"])

        if agent_id not in waiting_lists[task_id]:
            waiting_lists[task_id].append(agent_id)
        waiting_arrivals[task_id][agent_id] = float(state["pending_arrival"])

    def _task_is_coverable(self, task_id: int, env, task_remaining, waiting_lists) -> bool:
        missing = self._task_missing_after_waiters(env, task_remaining[task_id], waiting_lists[task_id])
        return bool(not np.any(missing > 1e-9))

    def _finalize_task(
        self,
        task_id: int,
        env,
        route_state,
        waiting_lists,
        waiting_arrivals,
        fixed_arrivals,
        completed_tasks,
        active_agents,
    ) -> bool:
        participants = list(waiting_lists[task_id])
        if not participants:
            return False

        task = env.task_dic[task_id]
        participant_arrivals = [float(waiting_arrivals[task_id][aid]) for aid in participants]
        all_arrivals = participant_arrivals + list(fixed_arrivals.get(task_id, []))
        if not all_arrivals:
            return False

        start_time = float(max(all_arrivals))
        finish_time = float(start_time + float(task.get("time", 0.0)))
        idle_consume = float(env.battery_consume_idle)

        for aid in participants:
            state = route_state[aid]
            wait_time = max(start_time - float(state["pending_arrival"]), 0.0)
            state["battery"] = max(
                0.0,
                float(state["battery"]) - idle_consume * float(wait_time + task.get("time", 0.0)),
            )
            state["current_cost"] = float(finish_time - float(env.current_time))
            state["route"].append(int(task_id))
            state["last_node"] = int(task_id)
            state["pending_task"] = None
            state["pending_arrival"] = None
            active_agents.add(aid)

        waiting_lists[task_id].clear()
        waiting_arrivals[task_id].clear()
        completed_tasks.add(int(task_id))
        return True

    def _choose_deadlock_target(self, env, task_ids, task_remaining, waiting_lists) -> Optional[int]:
        best_task = None
        best_key = None
        for task_id in task_ids:
            waiters = waiting_lists[task_id]
            if not waiters:
                continue
            missing = self._task_missing_after_waiters(env, task_remaining[task_id], waiters)
            total_required = float(np.maximum(task_remaining[task_id], 0.0).sum())
            missing_mass = float(np.maximum(missing, 0.0).sum())
            coverage_ratio = 1.0 if total_required <= 1e-9 else 1.0 - missing_mass / total_required
            deadline = float(env.task_dic[task_id].get("deadline", float("inf")))
            key = (-coverage_ratio, missing_mass, deadline, len(waiters), task_id)
            if best_key is None or key < best_key:
                best_key = key
                best_task = int(task_id)
        return best_task

    def _best_deadlock_donor(
        self,
        target_task_id: int,
        env,
        route_state,
        task_ids,
        task_remaining,
        waiting_lists,
    ) -> Optional[int]:
        target_loc = np.asarray(env.task_dic[target_task_id]["location"], dtype=float)
        missing = self._task_missing_after_waiters(env, task_remaining[target_task_id], waiting_lists[target_task_id])

        best_agent = None
        best_score = -float("inf")
        best_tie = float("inf")

        for task_id in task_ids:
            if task_id == target_task_id:
                continue
            for aid in waiting_lists[task_id]:
                state = route_state[aid]
                agent = env.agent_dic[aid]
                contribution = self._effective_overlap(agent.get("abilities", []), missing)
                if contribution <= 1e-9:
                    continue
                extra_travel = self._travel_time(
                    state["current_loc"],
                    target_loc,
                    agent.get("velocity", 0.2),
                )
                score = contribution / max(extra_travel, 1e-6)
                if score > best_score or (abs(score - best_score) <= 1e-9 and extra_travel < best_tie):
                    best_score = score
                    best_tie = extra_travel
                    best_agent = int(aid)

        return best_agent

    def _deadlock_reversal(
        self,
        env,
        task_ids,
        route_state,
        task_remaining,
        waiting_lists,
        waiting_arrivals,
        fixed_arrivals,
        completed_tasks,
        active_agents,
    ) -> bool:
        target_task_id = self._choose_deadlock_target(env, task_ids, task_remaining, waiting_lists)
        if target_task_id is None:
            return False

        target_task = env.task_dic[target_task_id]
        target_loc = np.asarray(target_task["location"], dtype=float)

        while not self._task_is_coverable(target_task_id, env, task_remaining, waiting_lists):
            donor = self._best_deadlock_donor(
                target_task_id=target_task_id,
                env=env,
                route_state=route_state,
                task_ids=task_ids,
                task_remaining=task_remaining,
                waiting_lists=waiting_lists,
            )
            if donor is None:
                break

            donor_state = route_state[donor]
            old_task_id = donor_state["pending_task"]
            if old_task_id is None or old_task_id == target_task_id:
                break

            if donor in waiting_lists[old_task_id]:
                waiting_lists[old_task_id].remove(donor)
            waiting_arrivals[old_task_id].pop(donor, None)

            agent = env.agent_dic[donor]
            extra_travel = self._travel_time(
                donor_state["current_loc"],
                target_loc,
                agent.get("velocity", 0.2),
            )
            donor_state["current_cost"] += extra_travel
            donor_state["current_loc"] = target_loc.copy()
            donor_state["battery"] = max(
                0.0,
                float(donor_state["battery"]) - float(env.battery_consume_moving) * float(extra_travel),
            )
            donor_state["pending_task"] = int(target_task_id)
            donor_state["pending_arrival"] = float(env.current_time + donor_state["current_cost"])
            waiting_lists[target_task_id].append(donor)
            waiting_arrivals[target_task_id][donor] = float(donor_state["pending_arrival"])

        if self._task_is_coverable(target_task_id, env, task_remaining, waiting_lists):
            return self._finalize_task(
                task_id=target_task_id,
                env=env,
                route_state=route_state,
                waiting_lists=waiting_lists,
                waiting_arrivals=waiting_arrivals,
                fixed_arrivals=fixed_arrivals,
                completed_tasks=completed_tasks,
                active_agents=active_agents,
            )

        return False

    def _solution_stats(self, env, task_ids, route_state, task_remaining, waiting_lists) -> Dict[str, object]:
        path_costs = {aid: float(state["current_cost"]) for aid, state in route_state.items()}
        f1 = float(sum(path_costs.values()))
        f2 = float(max(path_costs.values(), default=0.0))

        unresolved_mass = 0.0
        completed_count = 0
        for task_id in task_ids:
            missing = self._task_missing_after_waiters(env, task_remaining[task_id], waiting_lists[task_id])
            missing_mass = float(np.maximum(missing, 0.0).sum())
            if missing_mass <= 1e-9:
                completed_count += 1
            unresolved_mass += missing_mass

        scalar = float(f1 + len(route_state) * f2 + self.unresolved_penalty * unresolved_mass)
        routes = {}
        for aid, state in route_state.items():
            route = list(state["route"])
            if state["pending_task"] is not None:
                pending_task = int(state["pending_task"])
                missing = self._task_missing_after_waiters(
                    env,
                    task_remaining[pending_task],
                    waiting_lists[pending_task],
                )
                if not np.any(missing > 1e-9) and (not route or route[-1] != pending_task):
                    route.append(int(state["pending_task"]))
            routes[int(aid)] = route

        return {
            "routes": routes,
            "path_costs": path_costs,
            "f1": f1,
            "f2": f2,
            "scalar": scalar,
            "completed_count": int(completed_count),
        }

    def _construct_solution(
        self,
        agent_ids: Sequence[int],
        task_ids: Sequence[int],
        env,
        pheromone: Optional[np.ndarray],
        from_index: Dict[object, int],
        to_index: Dict[int, int],
        agent_index: Dict[int, int],
        greedy: bool,
    ) -> Dict[str, object]:
        route_state = self._initialize_route_state(agent_ids, env)
        task_remaining = {
            task_id: np.asarray(
                env.task_dic[task_id].get("status", env.task_dic[task_id].get("requirements", [])),
                dtype=float,
            ).copy()
            for task_id in task_ids
        }
        fixed_arrivals = {
            task_id: self._get_fixed_member_arrivals(env, task_id, agent_ids)
            for task_id in task_ids
        }
        waiting_lists = {task_id: [] for task_id in task_ids}
        waiting_arrivals = {task_id: {} for task_id in task_ids}
        completed_tasks = set()
        active_agents = set(int(aid) for aid in agent_ids)

        max_steps = max(4 * len(task_ids) + 4 * len(agent_ids), 1)
        no_progress_steps = 0
        progress_value = 0

        for _ in range(max_steps):
            if len(completed_tasks) >= len(task_ids):
                break

            if not active_agents:
                recovered = self._deadlock_reversal(
                    env=env,
                    task_ids=task_ids,
                    route_state=route_state,
                    task_remaining=task_remaining,
                    waiting_lists=waiting_lists,
                    waiting_arrivals=waiting_arrivals,
                    fixed_arrivals=fixed_arrivals,
                    completed_tasks=completed_tasks,
                    active_agents=active_agents,
                )
                if not recovered:
                    break
                continue

            agent_id = min(active_agents, key=lambda aid: (route_state[aid]["current_cost"], aid))
            active_agents.remove(agent_id)

            candidates = self._candidate_tasks_for_agent(
                agent_id=agent_id,
                env=env,
                task_ids=task_ids,
                completed_tasks=completed_tasks,
                task_remaining=task_remaining,
                waiting_lists=waiting_lists,
            )

            if not candidates:
                continue

            selected_task = self._select_next_task(
                agent_id=agent_id,
                agent_index=agent_index,
                env=env,
                candidates=candidates,
                route_state=route_state,
                pheromone=pheromone,
                from_index=from_index,
                to_index=to_index,
                greedy=greedy,
            )
            if selected_task is None:
                continue

            if not greedy and self.rng.random() >= self.willingness:
                active_agents.add(agent_id)
                no_progress_steps += 1
                if no_progress_steps > max(len(agent_ids), 1) * 3:
                    break
                continue

            no_progress_steps = 0
            self._claim_task(
                agent_id=agent_id,
                task_id=selected_task,
                env=env,
                route_state=route_state,
                waiting_lists=waiting_lists,
                waiting_arrivals=waiting_arrivals,
            )

            if self._task_is_coverable(selected_task, env, task_remaining, waiting_lists):
                self._finalize_task(
                    task_id=selected_task,
                    env=env,
                    route_state=route_state,
                    waiting_lists=waiting_lists,
                    waiting_arrivals=waiting_arrivals,
                    fixed_arrivals=fixed_arrivals,
                    completed_tasks=completed_tasks,
                    active_agents=active_agents,
                )

            current_progress = len(completed_tasks) + sum(1 for state in route_state.values() if state["pending_task"] is not None)
            if current_progress <= progress_value:
                no_progress_steps += 1
                if no_progress_steps > max(len(agent_ids), 1) * 3:
                    break
            else:
                progress_value = current_progress
                no_progress_steps = 0

        return self._solution_stats(
            env=env,
            task_ids=task_ids,
            route_state=route_state,
            task_remaining=task_remaining,
            waiting_lists=waiting_lists,
        )

    def _reinforce_solution(self, solution, pheromone, from_index, to_index, agent_ids) -> None:
        deposit = 1.0 / max(float(solution["scalar"]), 1e-8)
        for agent_pos, agent_id in enumerate(agent_ids):
            route = solution["routes"].get(agent_id, [])
            last_node = self.START_NODE
            for task_id in route:
                pheromone[from_index[last_node], to_index[task_id], agent_pos] = (
                    0.5 * pheromone[from_index[last_node], to_index[task_id], agent_pos] + 0.5 * deposit
                )
                last_node = task_id

    def plan_routes(self, agent_ids: Sequence[int], env) -> Dict[int, List[int]]:
        agent_ids = list(dict.fromkeys(int(aid) for aid in agent_ids))
        if not agent_ids:
            return {}

        task_ids = self._visible_open_task_ids(env)
        if not task_ids:
            return {aid: [] for aid in agent_ids}

        iterations, n_solutions = self._adaptive_budget(len(agent_ids), len(task_ids))
        from_nodes = [self.START_NODE] + list(task_ids)
        from_index = {node: idx for idx, node in enumerate(from_nodes)}
        to_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
        agent_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}

        greedy_solution = self._construct_solution(
            agent_ids=agent_ids,
            task_ids=task_ids,
            env=env,
            pheromone=None,
            from_index=from_index,
            to_index=to_index,
            agent_index=agent_index,
            greedy=True,
        )
        tau0 = 1.0 / max(float(greedy_solution["scalar"]), 1e-8)
        pheromone = np.full((len(from_nodes), len(task_ids), len(agent_ids)), tau0, dtype=float)

        best_solution = copy.deepcopy(greedy_solution)
        for _ in range(iterations):
            iteration_best = None
            for _ in range(n_solutions):
                solution = self._construct_solution(
                    agent_ids=agent_ids,
                    task_ids=task_ids,
                    env=env,
                    pheromone=pheromone,
                    from_index=from_index,
                    to_index=to_index,
                    agent_index=agent_index,
                    greedy=False,
                )
                if iteration_best is None or solution["scalar"] < iteration_best["scalar"]:
                    iteration_best = solution

            if iteration_best is None:
                continue

            if iteration_best["scalar"] < best_solution["scalar"]:
                best_solution = copy.deepcopy(iteration_best)

            pheromone *= max(1.0 - self.rho, 0.0)
            self._reinforce_solution(
                solution=iteration_best,
                pheromone=pheromone,
                from_index=from_index,
                to_index=to_index,
                agent_ids=agent_ids,
            )
            self._reinforce_solution(
                solution=best_solution,
                pheromone=pheromone,
                from_index=from_index,
                to_index=to_index,
                agent_ids=agent_ids,
            )

        return {
            aid: list(best_solution["routes"].get(aid, []))
            for aid in agent_ids
        }

    def select_task(self, agent_id: int, env, mask_bool, planned_route: Sequence[int]) -> Optional[int]:
        task_index_map = {task_id: idx + 1 for idx, task_id in enumerate(env.task_dic.keys())}

        for task_id in planned_route:
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if not bool(mask_bool[0, mask_idx]):
                return int(task_id)

        best_task_id = None
        best_cost = float("inf")
        agent = env.agent_dic[agent_id]
        current_loc = np.asarray(agent["location"], dtype=float)
        for task_id, task in env.task_dic.items():
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if bool(mask_bool[0, mask_idx]):
                continue
            remaining = np.asarray(task.get("status", task.get("requirements", [])), dtype=float)
            if self._effective_overlap(agent.get("abilities", []), remaining) <= 1e-9:
                continue
            travel_time = self._travel_time(
                current_loc,
                task["location"],
                agent.get("velocity", 0.2),
            )
            cost = float(travel_time + float(task.get("time", 0.0)))
            if cost < best_cost:
                best_cost = cost
                best_task_id = task_id

        return None if best_task_id is None else int(best_task_id)
