#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTAS-D 动态适配中的静态 route 规划器。

说明：
- 历史 CTAS-D 静态流程并不是在 Python 内直接实时求解，
  而是读取外部 `TEAMPLANNER_DET` 导出的 `results.yaml`。
- 这里提供一个“可在仓库内直接运行”的 solver-backed 近似版本：
  对当前已揭示任务构造静态子问题，并使用 OR-Tools 的 routing solver
  为各物种的可用智能体求解 route。
- 若 OR-Tools 不可用或该批次求解失败，则回退到旧的启发式构造器。

适配假设：
- 当前项目重点是动态 SA-BT；该版本按“缺失技能对应物种分别建 route”
  的方式近似外部 CTAS-D/TEAMPLANNER 的协作规划语义。
- 在 SA-BT 场景下，每个 species 基本对应单技能，因此这个映射是合理的。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2

    ORTOOLS_AVAILABLE = True
except Exception:
    pywrapcp = None
    routing_enums_pb2 = None
    ORTOOLS_AVAILABLE = False


class CTASDStaticRoutePlanner:
    def __init__(
        self,
        load_balance_span_cost: int = 20,
        non_preferred_species_penalty: float = 0.25,
        route_time_limit_ms: int = 100,
        cost_scale: int = 1000,
        drop_penalty: int = 10_000_000,
        use_ortools: bool = True,
    ):
        self.load_balance_span_cost = int(load_balance_span_cost)
        self.non_preferred_species_penalty = float(non_preferred_species_penalty)
        self.route_time_limit_ms = int(route_time_limit_ms)
        self.cost_scale = int(cost_scale)
        self.drop_penalty = int(drop_penalty)
        self.use_ortools = bool(use_ortools and ORTOOLS_AVAILABLE)

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
    def _agent_can_contribute(agent, remaining_status) -> bool:
        abilities = np.asarray(agent.get("abilities", []), dtype=float)
        remaining_status = np.asarray(remaining_status, dtype=float)
        if abilities.size == 0 or remaining_status.size == 0:
            return False
        return bool(np.any(np.minimum(abilities, remaining_status) > 0))

    @staticmethod
    def _travel_service_cost(from_loc, task_loc, velocity, service_time):
        velocity = max(float(velocity), 1e-8)
        distance = float(np.linalg.norm(np.asarray(from_loc, dtype=float) - np.asarray(task_loc, dtype=float)))
        travel_time = distance / velocity
        service_time = float(service_time)
        return travel_time + service_time, travel_time

    @staticmethod
    def _preferred_species_mask(task, env):
        optimized_species = np.asarray(task.get("optimized_species", []))
        if optimized_species.size == env.species_num:
            return optimized_species.astype(bool)
        return np.ones(env.species_num, dtype=bool)

    def _score_assignment(self, route_state, agent, task, preferred_species_mask):
        score, travel_time = self._travel_service_cost(
            from_loc=route_state["cursor_loc"],
            task_loc=task["location"],
            velocity=agent.get("velocity", 0.2),
            service_time=task.get("time", 0.0),
        )
        score += 0.05 * len(route_state["route"])
        if agent.get("species") >= len(preferred_species_mask) or not preferred_species_mask[agent["species"]]:
            score += self.non_preferred_species_penalty
        return float(score), float(travel_time)

    def _heuristic_plan_routes(self, agent_ids, env):
        agent_ids = list(dict.fromkeys(int(aid) for aid in agent_ids))
        if not agent_ids:
            return {}

        if hasattr(env, "calculate_optimized_ability"):
            env.calculate_optimized_ability()

        task_ids = self._visible_open_task_ids(env)
        if not task_ids:
            return {aid: [] for aid in agent_ids}

        route_state = {
            aid: {
                "route": [],
                "cursor_loc": np.asarray(env.agent_dic[aid]["location"], dtype=float).copy(),
            }
            for aid in agent_ids
        }
        task_members = {task_id: set() for task_id in task_ids}
        remaining_requirements = {
            task_id: np.asarray(
                env.task_dic[task_id].get("status", env.task_dic[task_id].get("requirements", [])),
                dtype=float,
            ).copy()
            for task_id in task_ids
        }

        while True:
            best = None

            for task_id in task_ids:
                remaining = remaining_requirements[task_id]
                if not np.any(remaining > 1e-9):
                    continue

                task = env.task_dic[task_id]
                preferred_species_mask = self._preferred_species_mask(task, env)

                for agent_id in agent_ids:
                    if agent_id in task_members[task_id]:
                        continue
                    agent = env.agent_dic[agent_id]
                    if not self._agent_can_contribute(agent, remaining):
                        continue
                    if not env.can_reach_with_battery(agent_id, task):
                        continue

                    score, travel_time = self._score_assignment(
                        route_state=route_state[agent_id],
                        agent=agent,
                        task=task,
                        preferred_species_mask=preferred_species_mask,
                    )
                    if best is None or score < best["score"]:
                        best = {
                            "agent_id": agent_id,
                            "task_id": task_id,
                            "score": score,
                            "travel_time": travel_time,
                        }

            if best is None:
                break

            agent_id = best["agent_id"]
            task_id = best["task_id"]
            agent = env.agent_dic[agent_id]
            task = env.task_dic[task_id]

            route_state[agent_id]["route"].append(task_id)
            route_state[agent_id]["cursor_loc"] = np.asarray(task["location"], dtype=float).copy()
            task_members[task_id].add(agent_id)

            remaining = remaining_requirements[task_id]
            contribution = np.minimum(np.asarray(agent.get("abilities", []), dtype=float), remaining)
            remaining_requirements[task_id] = np.maximum(remaining - contribution, 0.0)

        return {aid: route_state[aid]["route"] for aid in agent_ids}

    def _species_relevant_tasks(self, env, task_ids: Sequence[int], species: int, agent_ids: Sequence[int]) -> List[int]:
        if species < len(env.species_dict.get("abilities", [])):
            species_ability = np.asarray(env.species_dict["abilities"][species], dtype=float)
        else:
            species_ability = np.asarray(env.agent_dic[agent_ids[0]].get("abilities", []), dtype=float)

        relevant = []
        for task_id in task_ids:
            task = env.task_dic[task_id]
            remaining = np.asarray(task.get("status", task.get("requirements", [])), dtype=float)
            if not np.any(np.minimum(species_ability, remaining) > 1e-9):
                continue
            reachable = any(env.can_reach_with_battery(agent_id, task) for agent_id in agent_ids)
            if reachable:
                relevant.append(task_id)
        return relevant

    def _make_search_parameters(self, n_tasks: int, n_agents: int):
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        time_limit_ms = max(self.route_time_limit_ms, 20 + 4 * int(n_tasks) + 2 * int(n_agents))
        search_parameters.time_limit.seconds = int(time_limit_ms // 1000)
        search_parameters.time_limit.nanos = int(time_limit_ms % 1000) * 1_000_000
        search_parameters.log_search = False
        return search_parameters

    def _solve_species_routes(self, env, species: int, agent_ids: Sequence[int], task_ids: Sequence[int]) -> Dict[int, List[int]]:
        if not agent_ids or not task_ids:
            return {int(aid): [] for aid in agent_ids}

        task_ids = list(dict.fromkeys(int(tid) for tid in task_ids))
        agent_ids = list(dict.fromkeys(int(aid) for aid in agent_ids))

        task_nodes = []
        for task_id in task_ids:
            task = env.task_dic[task_id]
            task_nodes.append(
                {
                    "kind": "task",
                    "task_id": int(task_id),
                    "location": np.asarray(task["location"], dtype=float),
                    "service_time": float(task.get("time", 0.0)),
                }
            )

        starts = []
        ends = []
        node_meta = list(task_nodes)
        for agent_id in agent_ids:
            agent = env.agent_dic[agent_id]
            starts.append(len(node_meta))
            node_meta.append(
                {
                    "kind": "start",
                    "task_id": None,
                    "location": np.asarray(agent["location"], dtype=float),
                    "service_time": 0.0,
                    "agent_id": int(agent_id),
                }
            )
        for agent_id in agent_ids:
            ends.append(len(node_meta))
            node_meta.append(
                {
                    "kind": "end",
                    "task_id": None,
                    "location": np.asarray(env.agent_dic[agent_id]["location"], dtype=float),
                    "service_time": 0.0,
                    "agent_id": int(agent_id),
                }
            )

        manager = pywrapcp.RoutingIndexManager(len(node_meta), len(agent_ids), starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        def transit_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            to_meta = node_meta[to_node]
            if to_meta["kind"] == "end":
                return 0

            from_loc = node_meta[from_node]["location"]
            to_loc = to_meta["location"]
            travel = float(np.linalg.norm(from_loc - to_loc))
            service = float(to_meta["service_time"])
            return int(round((travel / 0.2 + service) * self.cost_scale))

        transit_callback_index = routing.RegisterTransitCallback(transit_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        routing.AddDimension(
            transit_callback_index,
            0,
            int(1e9),
            True,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")
        time_dimension.SetGlobalSpanCostCoefficient(max(self.load_balance_span_cost, 0))

        task_id_to_local = {task_id: idx for idx, task_id in enumerate(task_ids)}
        for local_idx, task_id in enumerate(task_ids):
            node_index = manager.NodeToIndex(local_idx)
            allowed_vehicles = []
            task = env.task_dic[task_id]
            for vehicle_idx, agent_id in enumerate(agent_ids):
                if env.can_reach_with_battery(agent_id, task):
                    allowed_vehicles.append(vehicle_idx)
            if not allowed_vehicles:
                continue
            routing.SetAllowedVehiclesForIndex(allowed_vehicles, node_index)
            routing.AddDisjunction([node_index], self.drop_penalty)

        solution = routing.SolveWithParameters(self._make_search_parameters(len(task_ids), len(agent_ids)))
        if solution is None:
            raise RuntimeError(f"ORTools routing returned no solution for species {species}")

        routes = {int(agent_id): [] for agent_id in agent_ids}
        for vehicle_idx, agent_id in enumerate(agent_ids):
            index = routing.Start(vehicle_idx)
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                if routing.IsEnd(index):
                    break
                node = manager.IndexToNode(index)
                meta = node_meta[node]
                if meta["kind"] == "task":
                    routes[int(agent_id)].append(int(meta["task_id"]))
        return routes

    def _solver_plan_routes(self, agent_ids: Sequence[int], env) -> Dict[int, List[int]]:
        agent_ids = list(dict.fromkeys(int(aid) for aid in agent_ids))
        if not agent_ids:
            return {}

        task_ids = self._visible_open_task_ids(env)
        if not task_ids:
            return {aid: [] for aid in agent_ids}

        routes = {aid: [] for aid in agent_ids}
        species_groups: Dict[int, List[int]] = {}
        for agent_id in agent_ids:
            species_groups.setdefault(int(env.agent_dic[agent_id]["species"]), []).append(int(agent_id))

        for species, species_agent_ids in species_groups.items():
            species_task_ids = self._species_relevant_tasks(
                env=env,
                task_ids=task_ids,
                species=species,
                agent_ids=species_agent_ids,
            )
            if not species_task_ids:
                continue
            species_routes = self._solve_species_routes(
                env=env,
                species=species,
                agent_ids=species_agent_ids,
                task_ids=species_task_ids,
            )
            for agent_id, route in species_routes.items():
                routes[agent_id] = route
        return routes

    def plan_routes(self, agent_ids, env):
        agent_ids = list(dict.fromkeys(int(aid) for aid in agent_ids))
        if not agent_ids:
            return {}

        if self.use_ortools:
            try:
                return self._solver_plan_routes(agent_ids, env)
            except Exception:
                pass

        return self._heuristic_plan_routes(agent_ids, env)

    def select_task(self, agent_id, env, mask_bool, planned_route):
        task_index_map = {task_id: idx + 1 for idx, task_id in enumerate(env.task_dic.keys())}

        for task_id in planned_route:
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if not bool(mask_bool[0, mask_idx]):
                return int(task_id)

        best_task_id = None
        best_score = float("inf")
        agent = env.agent_dic[agent_id]
        current_loc = np.asarray(agent["location"], dtype=float)
        for task_id, task in env.task_dic.items():
            mask_idx = task_index_map.get(task_id)
            if mask_idx is None or mask_idx >= mask_bool.shape[1]:
                continue
            if bool(mask_bool[0, mask_idx]):
                continue
            if not env.can_reach_with_battery(agent_id, task):
                continue
            if not self._agent_can_contribute(agent, task.get("status", task.get("requirements", []))):
                continue
            score, _ = self._travel_service_cost(
                from_loc=current_loc,
                task_loc=task["location"],
                velocity=agent.get("velocity", 0.2),
                service_time=task.get("time", 0.0),
            )
            if score < best_score:
                best_score = score
                best_task_id = task_id
        return None if best_task_id is None else int(best_task_id)
