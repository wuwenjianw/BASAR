import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from attention import AttentionNet, ImproMetaNet
from dynamic_centralized_planner import DynamicCentralizedPlanner
from model_capam import CapamMetaAggregator
from env.task_env import TaskEnv
from module_ablation import export_current_module_ablation_config
from parameters import EnvParams, TrainParams, SaverParams
from worker import Worker


def create_dynamic_model(agent_input_dim, task_input_dim, embedding_dim, device, model_name=None):
    """
    统一构建动态任务训练/评估模型，模型类型由参数配置控制
    """
    selected_model = (model_name or TrainParams.MODEL_NAME).lower()
    module_config = export_current_module_ablation_config()
    if selected_model == 'attention':
        return AttentionNet(
            agent_input_dim,
            task_input_dim,
            embedding_dim,
            cross_attention_mode=module_config['cross_attention_mode'],
            global_decoder_mode=module_config['global_decoder_mode'],
        ).to(device)
    if selected_model == 'myself':
        return ImproMetaNet(
            agent_input_dim,
            task_input_dim,
            embedding_dim,
            cross_attention_mode=module_config['cross_attention_mode'],
            global_decoder_mode=module_config['global_decoder_mode'],
        ).to(device)
    if selected_model == 'capam':
        return CapamMetaAggregator(agent_input_dim, task_input_dim, embedding_dim, device).to(device)
    raise ValueError(
        f"Unsupported TrainParams.MODEL_NAME='{selected_model}'. "
        "Expected one of: attention, myself, capam."
    )


def create_dynamic_planner(env, max_total_tasks, arrival_rate, simulation_time_limit,
                           random_seed=None, decision_maker=None, verbose=False,
                           dynamic_task_options=None):
    """
    统一构建训练/评估共用的动态集中规划器，避免 train-test planner 漂移。
    """
    planner = DynamicCentralizedPlanner(
        env=env,
        max_total_tasks=max_total_tasks,
        dynamic_task_arrival_rate=arrival_rate,
        simulation_time_limit=simulation_time_limit,
        verbose=verbose,
        random_seed=random_seed,
        decision_maker=decision_maker,
        max_waiting_time=30,
        protect_traveling=True,
        **(dynamic_task_options or {}),
    )
    env.max_waiting_time = planner.max_waiting_time
    planner._schedule_next_task_arrival()
    return planner


class DynamicWorker(Worker):
    """
    动态任务分配训练工作器

    在静态环境训练流程基础上，加入动态任务到达逻辑，保证训练过程在动态场景下进行。
    """

    def __init__(self, mete_agent_id, local_network, local_baseline, global_step,
                 device='cuda', save_image=False, seed=None, env_params=None,
                 arrival_rate_choices=None, max_total_tasks=100, max_action_index=None,
                 reward_config=None):
        super().__init__(mete_agent_id, local_network, local_baseline, global_step,
                         device=device, save_image=save_image, seed=seed, env_params=env_params,
                         reward_config=reward_config)
        # 动态任务到达率候选集合（单位：任务/分钟）
        self.arrival_rate_choices = arrival_rate_choices or [0.5, 1.0, 1.5, 2.0]
        # 动态+静态的任务上限
        self.max_total_tasks = max_total_tasks
        # 动作步数上限，用于防止死循环
        self.max_action_index = max_action_index or max(300, self.max_total_tasks * 5)
        # 保存一份基础环境快照，避免动态任务跨回合残留
        self._base_env_snapshot = self._snapshot_env()
        # 动态任务的随机数生成器（每个回合初始化）
        self._dynamic_rng = None

    def _snapshot_env(self):
        """
        记录环境的初始快照，用于回合重置
        """
        return copy.deepcopy((
            self.env.task_dic,
            self.env.agent_dic,
            self.env.depot_dic,
            self.env.species_dict,
            self.env.charging_station_dic
        ))

    def reset_env(self, seed=None, use_seed_env=False):
        """
        重置环境到基础任务状态，确保动态任务不会跨回合累积
        """
        if use_seed_env and seed is not None:
            # 使用种子驱动生成初始环境，保证评估时基线与当前模型环境一致
            random.seed(seed)
            np.random.seed(seed)
            self.env.reset(test_env=None, seed=seed)
        else:
            # 训练阶段默认复用基础快照，避免动态任务跨回合残留
            self.env.reset(test_env=copy.deepcopy(self._base_env_snapshot), seed=seed)
        self.env.init_state()

    def _reset_dynamic_rng(self, seed=None, env=None):
        """
        初始化动态任务的随机数生成器，保证同一回合内的动态任务一致
        """
        if seed is None:
            self._dynamic_rng = np.random.default_rng()
        else:
            self._dynamic_rng = np.random.default_rng(seed)
        # 可选地与环境随机数对齐，便于统一控制随机性
        if env is not None:
            env.rng = self._dynamic_rng

    def _sample_arrival_rate(self):
        """
        采样动态任务到达率
        """
        if self._dynamic_rng is None:
            return random.choice(self.arrival_rate_choices)
        return float(self._dynamic_rng.choice(self.arrival_rate_choices))

    def _sample_next_arrival_time(self, base_time, arrival_rate):
        """
        采样下一次动态任务的到达时间
        """
        if arrival_rate <= 0:
            return np.inf
        if self._dynamic_rng is None:
            interval = np.random.exponential(1.0 / arrival_rate)
        else:
            interval = self._dynamic_rng.exponential(1.0 / arrival_rate)
        return float(base_time + interval)

    def _advance_time_to(self, new_time, env=None):
        """
        将环境时间推进到指定时刻，并更新电量和移动状态
        """
        target_env = env or self.env
        if new_time < target_env.current_time:
            new_time = target_env.current_time
        target_env.current_time = new_time
        target_env.update_all_batteries(new_time)
        for agent in target_env.agent_dic.values():
            if len(agent['arrival_time']) > 0 and new_time >= agent['arrival_time'][-1]:
                agent['is_moving'] = False

    def _build_dynamic_task_spec(self, appear_time, env=None):
        """
        使用统一随机数生成器构造动态任务属性，避免全局随机导致不可复现
        """
        target_env = env or self.env
        rng = self._dynamic_rng or np.random.default_rng()
        # 任务位置
        task_location = rng.random(2)
        # 任务需求
        if target_env.binary_task:
            task_requirements = rng.integers(0, 2, size=(target_env.traits_dim,))
            while np.sum(task_requirements) == 0:
                task_requirements = rng.integers(0, 2, size=(target_env.traits_dim,))
        else:
            task_requirements = rng.integers(0, target_env.max_task_size, size=(target_env.traits_dim,))
            while np.sum(task_requirements) == 0:
                task_requirements = rng.integers(0, target_env.max_task_size, size=(target_env.traits_dim,))
        # 任务持续时间
        task_duration = float(rng.random() * target_env.duration_scale)
        # 任务截止时间
        max_speed = 0.2
        distances_to_depots = [
            np.linalg.norm(task_location - target_env.depot_dic[s]['location'])
            for s in range(target_env.species_num)
        ]
        max_distance = max(distances_to_depots) if distances_to_depots else 0.0
        min_deadline = appear_time + max_distance / max_speed + 20
        max_deadline = appear_time + 40
        task_deadline = float(rng.random() * (max_deadline - min_deadline) + min_deadline)
        # 任务紧急程度
        task_is_urgent = bool(rng.random() < 0.7)
        if not task_is_urgent:
            task_deadline = appear_time + 40

        return {
            "location": np.array(task_location, dtype=float),
            "requirements": np.array(task_requirements, dtype=float),
            "duration": task_duration,
            "deadline": task_deadline,
            "is_urgent": task_is_urgent,
        }

    def _add_one_dynamic_task(self, env=None):
        """
        在当前时刻添加一个动态任务，并更新调度状态
        """
        target_env = env or self.env
        spec = self._build_dynamic_task_spec(target_env.current_time, env=target_env)
        target_env.add_dynamic_task(
            num_tasks=1,
            appear_time=target_env.current_time,
            location=spec["location"],
            requirements=spec["requirements"],
            duration=spec["duration"],
            deadline=spec["deadline"],
            is_urgent=spec["is_urgent"],
        )
        # 动态任务出现后清理 no_choice，允许智能体重新决策
        for agent in target_env.agent_dic.values():
            agent['no_choice'] = False
            if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                agent['next_decision'] = target_env.current_time

    def _dispatch_to_charge(self, agent_id, env=None):
        """
        将智能体派发到充电站，避免因电量不足导致死循环
        """
        target_env = env or self.env
        agent = target_env.agent_dic[agent_id]
        if agent.get('current_task') == -999:
            return None
        charging_station_loc = agent['charging_station']
        agent['current_task'] = -999
        agent['assigned'] = True
        agent['is_moving'] = True
        agent['is_charging'] = False
        distance = np.linalg.norm(agent['location'] - charging_station_loc)
        travel_time = distance / agent['velocity']
        arrival_time = target_env.current_time + travel_time
        agent['next_decision'] = arrival_time
        agent['route'].append(-target_env.species_num - agent['species'] - 1)
        agent['arrival_time'].append(arrival_time)
        agent['travel_dist'] += distance
        return arrival_time

    def _peek_next_decision_time(self, env=None):
        """
        仅推断下一次决策时间，不修改环境状态
        """
        target_env = env or self.env
        decision_time = np.array(target_env.get_matrix(target_env.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            max_arrival = max(
                map(lambda x: max(x) if x else 0, target_env.get_matrix(target_env.agent_dic, 'arrival_time'))
            )
            safe_time = max(max_arrival, target_env.current_time)
            return ([], []), safe_time

        no_choice = target_env.get_matrix(target_env.agent_dic, 'no_choice')
        decision_time = np.where(no_choice, np.inf, decision_time)
        next_decision = np.nanmin(decision_time)
        if np.isinf(next_decision):
            arrival_time = np.array([agent['arrival_time'][-1] for agent in target_env.agent_dic.values()])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
        if next_decision < target_env.current_time:
            next_decision = target_env.current_time

        finished_agents = np.where(decision_time == next_decision)[0].tolist()
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            if no_choice[agent_id]:
                continue
            if next_decision >= target_env.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)

        return (finished_agents, blocked_agents), float(next_decision)

    def run_episode(self, episode_number=0, training=True, sample=False, max_waiting=False,
                    arrival_rate=None, dynamic_seed=None, debug_every=None):
        """
        运行动态环境的一个回合，支持动态任务到达
        """
        return self._run_episode_core(
            env=self.env,
            network=self.local_net,
            episode_number=episode_number,
            training=training,
            sample=sample,
            max_waiting=max_waiting,
            arrival_rate=arrival_rate,
            dynamic_seed=dynamic_seed,
            debug_every=debug_every,
            collect_buffer=training,
        )

    def _run_episode_core(self, env, network, episode_number, training, sample, max_waiting,
                          arrival_rate, dynamic_seed, debug_every, collect_buffer):
        """
        与 evaluate_my_model_dynamic.py 保持一致的事件驱动逻辑。
        """
        buffer_dict = {idx: [] for idx in range(7)} if collect_buffer else None
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0
        last_decision_time = {}
        deadlock_recovery_times = set()
        rescue_replan_times = set()
        termination_reason = 'running'
        rescue_replan_count = 0
        stagnation_window = 200.0
        max_main_iterations = 100000
        main_iteration = 0

        self._reset_dynamic_rng(dynamic_seed, env=env)
        if arrival_rate is None:
            arrival_rate = self._sample_arrival_rate()

        planner = create_dynamic_planner(
            env=env,
            max_total_tasks=self.max_total_tasks,
            arrival_rate=arrival_rate,
            simulation_time_limit=self.max_time,
            random_seed=dynamic_seed,
            decision_maker=None,
            verbose=False,
        )

        def _can_reach_contributable_task(agent_id):
            contributable_mask = env.get_contributable_task_mask(agent_id)
            for task_id, task in env.task_dic.items():
                if task.get('finished', False) or task.get('feasible_assignment', False):
                    continue
                if task_id >= len(contributable_mask):
                    continue
                if contributable_mask[task_id]:
                    continue
                if env.can_reach_with_battery(agent_id, task):
                    return True
            return False

        def _finished_task_count():
            return sum(1 for task in env.task_dic.values() if task.get('finished', False))

        def _peek_next_decision_time():
            decision_time = np.array(env.get_matrix(env.agent_dic, 'next_decision'), dtype=float)
            if decision_time.size == 0:
                return float('inf')
            if np.all(np.isnan(decision_time)):
                arrival = [
                    max(a) if a else env.current_time
                    for a in env.get_matrix(env.agent_dic, 'arrival_time')
                ]
                if not arrival:
                    return float('inf')
                return max(max(arrival), env.current_time)

            no_choice = env.get_matrix(env.agent_dic, 'no_choice')
            decision_time = np.where(no_choice, np.inf, decision_time)
            next_decision = np.nanmin(decision_time)
            if np.isinf(next_decision):
                arrival_time = np.array([
                    agent['arrival_time'][-1] if agent['arrival_time'] else env.current_time
                    for agent in env.agent_dic.values()
                ])
                decision_time = np.where(no_choice, np.inf, arrival_time)
                next_decision = np.nanmin(decision_time)
            if np.isnan(next_decision):
                return float('inf')
            return max(float(next_decision), env.current_time)

        def _apply_extra_mask(agent_id, mask_np):
            if mask_np is None or mask_np.ndim != 2:
                return mask_np
            current_time = getattr(env, 'current_time', 0.0)
            for idx, task in enumerate(env.task_dic.values()):
                if task.get('appear_time', 0.0) > current_time:
                    mask_np[0, idx + 1] = True
                elif not env.can_reach_with_battery(agent_id, task):
                    mask_np[0, idx + 1] = True
            return mask_np

        def _sync_agent_motion_state(target_time):
            env.update_all_batteries(target_time)
            for agent in env.agent_dic.values():
                if agent['arrival_time'] and target_time >= agent['arrival_time'][-1]:
                    agent['is_moving'] = False

        def _advance_time(target_time):
            env.current_time = max(env.current_time, target_time)
            _sync_agent_motion_state(env.current_time)

        def _generate_ready_dynamic_tasks():
            while (planner.next_task_arrival_time is not None and
                   env.current_time >= planner.next_task_arrival_time and
                   len(env.task_dic) < self.max_total_tasks):
                planner._generate_dynamic_task()

        def _freeze_agent(agent_id):
            agent = env.agent_dic[agent_id]
            agent['no_choice'] = True
            agent['next_decision'] = float('inf')
            agent['is_moving'] = False

        def _all_tasks_completed():
            if len(env.task_dic) == 0:
                return True
            return all(task.get('finished', False) for task in env.task_dic.values())

        def _reactivate_idle_agents():
            reactivated = 0
            for agent in env.agent_dic.values():
                if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                    agent['no_choice'] = False
                    agent['next_decision'] = env.current_time
                    reactivated += 1
            return reactivated

        def _rescue_with_planner_replan():
            nonlocal rescue_replan_count
            if rescue_replan_count >= 20:
                return False
            if env.current_time in rescue_replan_times:
                return False

            prev_active_events = sum(
                1 for agent in env.agent_dic.values()
                if agent.get('next_decision', float('inf')) < float('inf')
            )
            prev_assignments = tuple(
                (aid, agent.get('current_task'), float(agent.get('next_decision', float('inf'))))
                for aid, agent in env.agent_dic.items()
            )

            try:
                planner._plan(replan=True, force_waiting=True)
                rescue_replan_count += 1
                rescue_replan_times.add(env.current_time)
            except Exception:
                return False

            for agent in env.agent_dic.values():
                if agent.get('next_decision', float('inf')) < float('inf'):
                    agent['no_choice'] = False

            next_decision_time_after_replan = _peek_next_decision_time()
            next_active_events = sum(
                1 for agent in env.agent_dic.values()
                if agent.get('next_decision', float('inf')) < float('inf')
            )
            next_assignments = tuple(
                (aid, agent.get('current_task'), float(agent.get('next_decision', float('inf'))))
                for aid, agent in env.agent_dic.items()
            )
            has_new_schedule = next_active_events > 0 and np.isfinite(next_decision_time_after_replan)
            has_state_change = next_assignments != prev_assignments or next_active_events != prev_active_events
            return has_new_schedule and has_state_change

        def _handle_charging_station_arrivals():
            for agent in env.agent_dic.values():
                if agent.get('current_task') != -999:
                    continue
                if agent['next_decision'] > env.current_time + 1e-6:
                    continue

                old_location = agent['location'].copy()
                agent['location'] = agent['charging_station'].copy()
                distance_traveled = np.linalg.norm(old_location - agent['charging_station'])
                agent['travel_dist'] = agent.get('travel_dist', 0) + distance_traveled
                agent['battery'] = env.initial_battery
                agent['total_charging_times'] += 1
                agent['is_moving'] = False
                agent['is_charging'] = False
                agent['current_task'] = -agent['species'] - 1
                agent['assigned'] = False
                agent['next_decision'] = float('inf')

        min_action_limit = self.max_total_tasks * max(1, len(env.agent_dic)) * 5
        action_limit = max(self.max_action_index, min_action_limit)

        last_finished_count = _finished_task_count()
        last_progress_time = float(env.current_time)
        while not env.finished and env.current_time < self.max_time and current_action_index < action_limit:
            main_iteration += 1
            if main_iteration > max_main_iterations:
                termination_reason = 'iteration_limit'
                env.current_time = self.max_time
                break

            if _all_tasks_completed():
                env.finished = True
                termination_reason = 'all_tasks_completed'
                break

            if len(env.task_dic) >= self.max_total_tasks:
                planner.next_task_arrival_time = None

            next_decision_time = _peek_next_decision_time()
            next_arrival_time = planner.next_task_arrival_time
            if debug_every is not None and decision_step % debug_every == 0:
                remaining_dynamic = max(0, self.max_total_tasks - len(env.task_dic))
                no_choice_count = int(np.sum(env.get_matrix(env.agent_dic, 'no_choice')))
                next_decisions = np.array(env.get_matrix(env.agent_dic, 'next_decision'), dtype=float)
                finite_decisions = next_decisions[np.isfinite(next_decisions)]
                min_next = float(finite_decisions.min()) if finite_decisions.size > 0 else float('inf')
                print(
                    f"[调试] step={decision_step} time={env.current_time:.3f} "
                    f"peek_time={next_decision_time:.3f} 动态剩余={remaining_dynamic} "
                    f"下次到达={float('inf') if next_arrival_time is None else next_arrival_time:.3f} "
                    f"动作数={current_action_index} "
                    f"no_choice={no_choice_count} min_next={min_next:.3f}"
                )

            if next_arrival_time is not None and next_arrival_time <= next_decision_time:
                _advance_time(next_arrival_time)
                _generate_ready_dynamic_tasks()
                if len(env.task_dic) >= self.max_total_tasks:
                    planner.next_task_arrival_time = None
                for agent in env.agent_dic.values():
                    agent['no_choice'] = False
                    if agent['current_task'] < 0 and agent['next_decision'] == float('inf'):
                        agent['next_decision'] = env.current_time
                continue

            if next_decision_time == float('inf') and next_arrival_time is None:
                if _all_tasks_completed():
                    env.finished = True
                    termination_reason = 'all_tasks_completed'
                    break
                if _rescue_with_planner_replan():
                    continue
                if env.current_time not in deadlock_recovery_times:
                    reactivated = _reactivate_idle_agents()
                    deadlock_recovery_times.add(env.current_time)
                    if reactivated > 0:
                        continue
                termination_reason = 'deadlock_no_events'
                env.current_time = self.max_time
                break

            release_agents, current_time = env.next_decision()
            env.current_time = current_time
            if not release_agents[0] and not release_agents[1]:
                if next_arrival_time is None:
                    if _all_tasks_completed():
                        env.finished = True
                        termination_reason = 'all_tasks_completed'
                        break
                    if _rescue_with_planner_replan():
                        continue
                    if env.current_time not in deadlock_recovery_times:
                        reactivated = _reactivate_idle_agents()
                        deadlock_recovery_times.add(env.current_time)
                        if reactivated > 0:
                            continue
                    termination_reason = 'deadlock_empty_release'
                    env.current_time = self.max_time
                    break
                if next_arrival_time > env.current_time + 1e-9:
                    _advance_time(next_arrival_time)
                    continue

            _handle_charging_station_arrivals()
            _generate_ready_dynamic_tasks()

            np.random.shuffle(release_agents[0])
            processed_agents = set()
            if debug_every is not None and decision_step % debug_every == 0:
                print(
                    f"[调试] 决策 time={current_time:.3f} "
                    f"完成队列=({len(release_agents[0])},{len(release_agents[1])})"
                )

            while release_agents[0] or release_agents[1]:
                agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                if agent_id in processed_agents:
                    continue
                processed_agents.add(agent_id)
                agent = env.agent_dic[agent_id]
                agent['no_choice'] = False

                if last_decision_time.get(agent_id) == env.current_time:
                    agent['next_decision'] = env.current_time + getattr(env, 'dt', 1e-3)
                    continue

                if agent.get('current_task', -1) >= 0:
                    current_task = env.task_dic.get(agent['current_task'])
                    if current_task and agent_id in current_task.get('members', []):
                        if current_task.get('feasible_assignment', False):
                            if env.current_time < float(current_task.get('time_finish', env.current_time)):
                                agent['no_choice'] = True
                                agent['next_decision'] = float(current_task.get('time_finish', env.current_time))
                                continue
                        else:
                            arrival_time = env.get_arrival_time(agent_id, current_task['ID'])
                            wait_deadline = arrival_time + float(getattr(env, 'max_waiting_time', 0.0))
                            if env.current_time < wait_deadline:
                                agent['no_choice'] = True
                                agent['next_decision'] = wait_deadline
                                continue

                if agent['current_task'] >= 0 and agent.get('arrival_time'):
                    last_arrival = agent['arrival_time'][-1]
                    if last_arrival > env.current_time + 1e-9:
                        agent['next_decision'] = last_arrival
                        continue

                if env.check_battery_critical(agent_id):
                    planner._dispatch_to_charge(agent_id, plan_time=env.current_time)
                    continue

                tasks_info_np, agents_info_np, mask_np = env.agent_observe(agent_id, max_waiting)
                mask_np = _apply_extra_mask(agent_id, mask_np)
                task_info, total_agents, mask = self.convert_torch((tasks_info_np, agents_info_np, mask_np))
                mask_bool = mask > 0.5

                block_flag = mask_bool[0, 1:].all().item()
                has_contributable = not env.get_contributable_task_mask(agent_id).all()
                if block_flag and not np.all(env.get_matrix(env.task_dic, 'feasible_assignment')):
                    can_reach_task = _can_reach_contributable_task(agent_id)
                    if has_contributable and not can_reach_task:
                        at_station = np.linalg.norm(agent['location'] - agent['charging_station']) <= 1e-6
                        full_battery = agent.get('battery', 0.0) >= env.initial_battery - 1e-6
                        if at_station and full_battery:
                            _freeze_agent(agent_id)
                            continue
                        planner._dispatch_to_charge(agent_id, plan_time=env.current_time)
                        continue
                    _freeze_agent(agent_id)
                    continue
                elif block_flag and np.all(env.get_matrix(env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                    _freeze_agent(agent_id)
                    continue

                if not mask_bool[0, 1:].all().item():
                    mask_bool[0, 0] = True
                mask_float = mask_bool.float()

                index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                with torch.no_grad():
                    probs, _ = network(task_info, total_agents, mask_float, index)

                if training:
                    action = Categorical(probs).sample()
                    while action.item() > env.tasks_num:
                        action = Categorical(probs).sample()
                else:
                    action = Categorical(probs).sample() if sample else torch.argmax(probs, dim=1)

                selected_task = action.item() - 1
                _, doable, _ = env.agent_step(agent_id, action.item(), decision_step)
                if not doable:
                    _freeze_agent(agent_id)
                elif selected_task == -1 and mask_bool[0, 1:].all().item():
                    _freeze_agent(agent_id)
                else:
                    min_step = getattr(env, 'dt', 1e-3)
                    if agent['next_decision'] <= env.current_time + 1e-9:
                        agent['next_decision'] = env.current_time + min_step

                if collect_buffer and doable:
                    padded_task_info, padded_agents, padded_mask = self.obs_padding(
                        task_info.clone(),
                        total_agents.clone(),
                        mask_float.clone(),
                    )
                    buffer_dict[0] += padded_agents
                    buffer_dict[1] += padded_task_info
                    buffer_dict[2] += action.unsqueeze(0)
                    buffer_dict[3] += padded_mask
                    buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)
                    buffer_dict[5] += index
                    buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                    current_action_index += 1

                last_decision_time[agent_id] = env.current_time

            env.finished = env.check_finished()
            if _all_tasks_completed():
                env.finished = True
                termination_reason = 'all_tasks_completed'
            elif env.finished:
                termination_reason = 'all_tasks_finished'

            current_finished_count = _finished_task_count()
            if current_finished_count > last_finished_count:
                last_finished_count = current_finished_count
                last_progress_time = float(env.current_time)
            elif (not env.finished and
                  env.current_time - last_progress_time >= stagnation_window and
                  current_finished_count < len(env.task_dic)):
                if _rescue_with_planner_replan():
                    last_progress_time = float(env.current_time)

            decision_step += 1

        if termination_reason == 'running':
            if _all_tasks_completed():
                termination_reason = 'all_tasks_completed'
            elif env.finished:
                termination_reason = 'all_tasks_finished'
            elif env.current_time >= self.max_time:
                termination_reason = 'time_limit'
            else:
                termination_reason = 'loop_exit'

        # 训练/评估回合退出后补一次任务状态结算，和评估脚本保持一致。
        env.task_update()
        terminal_reward, finished_tasks = env.get_episode_reward(self.max_time, episode_number)
        stats = env.get_task_statistics()
        total_tasks = stats['total_tasks']
        finished_task_count = stats['finished_tasks']
        raw_end_time = float(env.current_time)
        effective_makespan = min(raw_end_time, float(self.max_time))
        if finished_task_count < total_tasks:
            effective_makespan = float(self.max_time)

        perf_metrics['success_rate'] = [finished_task_count / total_tasks if total_tasks > 0 else 0.0]
        perf_metrics['makespan'] = [effective_makespan]
        perf_metrics['time_cost'] = [np.nanmean(env.get_matrix(env.task_dic, 'time_start'))]
        env.calculate_waiting_time()
        finished_task_list = [task for task in env.task_dic.values() if task.get('finished', False)]
        total_waiting_time = 0.0
        for task in finished_task_list:
            task_waiting = float(task.get('sum_waiting_time', 0.0))
            if not np.isfinite(task_waiting):
                task_waiting = 0.0
            total_waiting_time += task_waiting
        perf_metrics['waiting_time'] = [total_waiting_time]
        perf_metrics['travel_dist'] = [np.sum(env.get_matrix(env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [env.get_efficiency()]

        deadline_violations = []
        on_time_count = 0
        finished_count = 0
        for task in env.task_dic.values():
            time_finish = task['time_finish']
            deadline = task['deadline']
            if time_finish > 0:
                finished_count += 1
                if time_finish <= deadline:
                    on_time_count += 1
                else:
                    deadline_violations.append(time_finish - deadline)

        if finished_count > 0:
            perf_metrics['deadline_success_rate'] = [on_time_count / finished_count]
        else:
            perf_metrics['deadline_success_rate'] = [0.0]

        if deadline_violations:
            perf_metrics['total_deadline_violation'] = [np.sum(deadline_violations)]
            perf_metrics['avg_deadline_violation'] = [np.mean(deadline_violations)]
            perf_metrics['max_deadline_violation'] = [np.max(deadline_violations)]
            perf_metrics['violation_count'] = [len(deadline_violations)]
        else:
            perf_metrics['total_deadline_violation'] = [0.0]
            perf_metrics['avg_deadline_violation'] = [0.0]
            perf_metrics['max_deadline_violation'] = [0.0]
            perf_metrics['violation_count'] = [0]

        return terminal_reward, buffer_dict, perf_metrics

    def baseline_test(self):
        """
        动态环境下的基线测试
        """
        baseline_env = TaskEnv(
            EnvParams.SPECIES_AGENTS_RANGE,
            EnvParams.SPECIES_RANGE,
            EnvParams.TASKS_RANGE,
            EnvParams.TRAIT_DIM,
            EnvParams.DECISION_DIM,
            max_task_size=self.env.max_task_size,
            duration_scale=self.env.duration_scale,
            single_skill=self.env.single_skill,
            binary_task=self.env.binary_task,
            use_deadline=self.env.use_deadline,
            reward_config=self.reward_config,
        )
        baseline_env.reset(test_env=copy.deepcopy(self._base_env_snapshot))
        baseline_env.init_state()

        self._reset_dynamic_rng(0, env=baseline_env)
        arrival_rate = self._sample_arrival_rate()
        reward, _, _ = self._run_episode_core(
            env=baseline_env,
            network=self.local_baseline,
            episode_number=0,
            training=False,
            sample=False,
            max_waiting=False,
            arrival_rate=arrival_rate,
            dynamic_seed=0,
            debug_every=None,
            collect_buffer=False,
        )
        return reward

    def evaluate_episode(self, seed=None, use_baseline=True, sample=False):
        """
        使用与训练一致的动态逻辑进行评估
        """
        self.reset_env(seed=seed, use_seed_env=True)
        network = self.local_baseline if use_baseline else self.local_net
        reward, _, _ = self._run_episode_core(
            env=self.env,
            network=network,
            episode_number=0,
            training=False,
            sample=sample,
            max_waiting=False,
            arrival_rate=None,
            dynamic_seed=seed,
            debug_every=None,
            collect_buffer=False,
        )
        return reward

    def work(self, episode_number):
        """
        动态环境下的工作流程，与静态训练流程保持一致
        """
        baseline_rewards = []
        buffers = []
        metrics = []
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK

        dynamic_seed = random.randint(0, 10_000_000)
        self._reset_dynamic_rng(dynamic_seed, env=self.env)
        arrival_rate = self._sample_arrival_rate()

        for _ in range(TrainParams.POMO_SIZE):
            self.reset_env(seed=dynamic_seed)
            terminal_reward, buffer, perf_metrics = self.run_episode(
                episode_number, True, False, max_waiting,
                arrival_rate=arrival_rate, dynamic_seed=dynamic_seed
            )
            if terminal_reward is np.nan:
                max_waiting = True
                continue
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)

        baseline_reward = np.nanmean(baseline_rewards)
        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key == 4:
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx]   # 现在是 0 + terminal_reward
                if key == 6:
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except Exception:
                pass

        self.episode_number = episode_number

    def obs_padding(self, task_info, agents, mask):
        """
        动态任务数量下的观测填充
        """
        target_task_len = self.max_total_tasks + 1
        task_padding = target_task_len - task_info.shape[1]
        if task_padding < 0:
            raise ValueError("任务数量超过最大上限，请检查 max_total_tasks 配置")

        task_info = F.pad(task_info, (0, 0, 0, task_padding), 'constant', 0)
        agents = F.pad(
            agents,
            (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]),
            'constant',
            0
        )
        mask = F.pad(mask, (0, task_padding), 'constant', 1)
        return task_info, agents, mask
