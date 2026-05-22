import copy
import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import ray
from scipy.stats import ttest_rel

from dynamic_worker import create_dynamic_model
from dynamic_runner import DynamicRLRunner
from module_ablation import export_current_module_ablation_config, normalize_module_ablation_config
from parameters import EnvParams, TrainParams, SaverParams
from project_paths import set_saver_paths
from env.task_env import TaskEnv
from reward_ablation import normalize_reward_config


class Logger(object):
    """
    训练日志记录器，负责模型保存加载与TensorBoard记录
    """
    def __init__(self):
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)

        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)

    def set(self, global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay

    def write_to_board(self, tensorboard_data, curr_episode):
        """
        写入TensorBoard数据
        """
        tensorboard_data = np.array(tensorboard_data)
        reward_std = float(np.nanstd(tensorboard_data[:, 0])) if tensorboard_data.size > 0 else 0.0
        reward_var = float(np.nanvar(tensorboard_data[:, 0])) if tensorboard_data.size > 0 else 0.0
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
        reward, p_l, entropy, grad_norm, success_rate, time, time_cost, waiting, distance, effi, _, _, _, _, _ = tensorboard_data
        metrics = {
            'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],
            'Loss/Policy Loss': p_l,
            'Loss/Entropy': entropy,
            'Loss/Grad Norm': grad_norm,
            'Loss/Reward': reward,
            'Loss/Reward Std': reward_std,
            'Loss/Reward Var': reward_var,
            'Perf/Makespan': time,
            'Perf/Success rate': success_rate,
            'Perf/Time cost': time_cost,
            'Perf/Waiting time': waiting,
            'Perf/Traveling distance': distance,
            'Perf/Waiting Efficiency': effi,
        }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

    def load_saved_model(self):
        """
        加载模型检查点
        """
        print('Loading Model...')
        checkpoint = torch.load(os.path.join(SaverParams.MODEL_PATH, 'checkpoint.pth'), weights_only=False)
        if SaverParams.LOAD_FROM == 'best':
            model = 'best_model'
        else:
            model = 'model'
        self.global_net.load_state_dict(checkpoint[model], strict=False)
        self.baseline_net.load_state_dict(checkpoint[model], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        best_perf = checkpoint['best_perf']
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
        return curr_episode, curr_level, best_perf

    def save_model(self, curr_episode, curr_level, best_perf):
        """
        保存模型检查点
        """
        print('Saving model', end='\n')
        checkpoint = {
            "model": self.global_net.state_dict(),
            "best_model": self.baseline_net.state_dict(),
            "best_optimizer": self.optimizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": curr_episode,
            "lr_decay": self.lr_decay.state_dict(),
            "level": curr_level,
            "best_perf": best_perf
        }
        path_checkpoint = os.path.join(SaverParams.MODEL_PATH, 'checkpoint.pth')
        torch.save(checkpoint, path_checkpoint)
        print('Saved model', end='\n')

    @staticmethod
    def generate_env_params(curr_level=None):
        """
        生成环境参数
        """
        per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
        species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
        tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
        params = [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num)]
        return params

    @staticmethod
    def generate_test_set_seed():
        """
        生成测试集随机种子
        """
        test_seed = np.random.randint(low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES).tolist()
        return test_seed


def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    """
    合并两个字典，用于拼接训练数据
    """
    if ini_dictionary2 is not None:
        merged_dict = {**ini_dictionary1, **ini_dictionary2}
        final_dict = {}
        for k, v in merged_dict.items():
            final_dict[k] = ini_dictionary1[k] + v
        return final_dict
    return ini_dictionary1


def infer_input_dims():
    """
    根据TaskEnv实际观测推断网络输入维度
    """
    env = TaskEnv(
        EnvParams.SPECIES_AGENTS_RANGE,
        EnvParams.SPECIES_RANGE,
        EnvParams.TASKS_RANGE,
        EnvParams.TRAIT_DIM,
        EnvParams.DECISION_DIM,
    )
    agent_id = list(env.agent_dic.keys())[0]
    tasks_info, agents_info, _ = env.agent_observe(agent_id, False)
    return int(agents_info.shape[-1]), int(tasks_info.shape[-1])


def configure_saver_paths(folder_name):
    """
    配置动态训练专用的保存路径
    """
    set_saver_paths(SaverParams, folder_name)


def main(folder_name=None, reward_config=None, module_ablation_config=None):
    reward_config = normalize_reward_config(reward_config)
    if module_ablation_config is None:
        module_ablation_config = export_current_module_ablation_config()
    else:
        module_ablation_config = normalize_module_ablation_config(module_ablation_config)
    if folder_name is None:
        folder_name = SaverParams.FOLDER_NAME
    configure_saver_paths(folder_name)
    logger = Logger()
    ray.init()

    os.makedirs(SaverParams.TRAIN_PATH, exist_ok=True)
    with open(os.path.join(SaverParams.TRAIN_PATH, 'reward_config.json'), 'w', encoding='utf-8') as f:
        json.dump(reward_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SaverParams.TRAIN_PATH, 'module_ablation_config.json'), 'w', encoding='utf-8') as f:
        json.dump(module_ablation_config, f, indent=2, ensure_ascii=False)
    print('Reward config:')
    print(json.dumps(reward_config, indent=2, ensure_ascii=False))
    print('Module ablation config:')
    print(json.dumps(module_ablation_config, indent=2, ensure_ascii=False))

    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')

    agent_input_dim, task_input_dim = infer_input_dims()
    global_network = create_dynamic_model(agent_input_dim, task_input_dim, TrainParams.EMBEDDING_DIM, device)
    baseline_network = create_dynamic_model(agent_input_dim, task_input_dim, TrainParams.EMBEDDING_DIM, device)

    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode = 0
    curr_level = 0
    best_perf = -200
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()

    arrival_rate_choices = [1, 2, 3, 4]
    max_total_tasks = 100
    max_action_index = None

    meta_agents = [
        DynamicRLRunner.remote(
            i,
            agent_input_dim,
            task_input_dim,
            TrainParams.EMBEDDING_DIM,
            arrival_rate_choices=arrival_rate_choices,
            max_total_tasks=max_total_tasks,
            max_action_index=max_action_index,
            reward_config=reward_config,
        )
        for i in range(TrainParams.NUM_META_AGENT)
    ]

    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)

    jobs = []
    env_params = logger.generate_env_params(curr_level)
    for i, meta_agent in enumerate(meta_agents):
        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
        curr_episode += 1

    test_set = logger.generate_test_set_seed()
    baseline_value = None
    experience_buffer = {idx: [] for idx in range(7)}
    perf_metrics = {
        'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [],
        'travel_dist': [], 'efficiency': [], 'deadline_success_rate': [],
        'avg_deadline_violation': [], 'max_deadline_violation': [],
        'violation_count': [], 'total_deadline_violation': []
    }
    training_data = []

    try:
        while True:
            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
            buffer, metrics, info = done_job

            experience_buffer = fuse_two_dicts(experience_buffer, buffer)
            perf_metrics = fuse_two_dicts(perf_metrics, metrics)

            update_done = False
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]

                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)

                    probs, _ = global_network(task_inputs, agent_inputs, global_mask_batch, index)
                    dist = Categorical(probs)
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
                    policy_loss = -logp * advantage_batch.flatten().detach()
                    policy_loss = policy_loss.mean()

                    l1_loss = torch.tensor(0.0, device=device)
                    if hasattr(global_network, 'get_l1_regularization_loss'):
                        l1_loss = global_network.get_l1_regularization_loss()
                    loss = policy_loss + l1_loss
                    global_optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=100, norm_type=2)
                    global_optimizer.step()

                    train_metrics.append([reward_batch.mean().item(), policy_loss.item(), entropy.item(), grad_norm.item()])
                lr_decay.step()

                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
                train_metrics = np.nanmean(train_metrics, axis=0)
                for v in perf_metrics.values():
                    del v[:]
                data = [*train_metrics, *perf_data]
                training_data.append(data)

            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)

            env_params = logger.generate_env_params(curr_level)
            jobs.append(meta_agents[info['id']].training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
            curr_episode += 1

            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)

            if TrainParams.EVALUATE and curr_episode % 256 == 0:
                # 停止当前训练任务，避免等待所有任务完成导致卡死
                if jobs:
                    for job in jobs:
                        try:
                            ray.cancel(job, force=True)
                        except Exception:
                            pass
                    jobs = []
                for a in meta_agents:
                    ray.kill(a)
                print('Evaluate baseline model at episode', curr_episode)

                if baseline_value is None:
                    test_agent_list = [
                        DynamicRLRunner.remote(
                            metaAgentID=i,
                            agent_input_dim=agent_input_dim,
                            task_input_dim=task_input_dim,
                            embedding_dim=TrainParams.EMBEDDING_DIM,
                            arrival_rate_choices=arrival_rate_choices,
                            max_total_tasks=max_total_tasks,
                            max_action_index=max_action_index,
                            reward_config=reward_config,
                        )
                        for i in range(TrainParams.NUM_META_AGENT)
                    ]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(baseline_weights_memory))
                    rewards = dict()
                    seed_list = copy.deepcopy(test_set)
                    evaluate_jobs = [test_agent_list[i].testing.remote(seed=seed_list.pop(), use_baseline=True)
                                     for i in range(TrainParams.NUM_META_AGENT)]
                    while True:
                        test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                        reward, seed, meta_id = ray.get(test_done_id)[0]
                        rewards[seed] = reward
                        if seed_list:
                            evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop(), use_baseline=True))
                        if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                            break
                    rewards = dict(sorted(rewards.items()))
                    baseline_value = np.stack(list(rewards.values()))
                    for a in test_agent_list:
                        ray.kill(a)

                test_agent_list = [
                    DynamicRLRunner.remote(
                        metaAgentID=i,
                        agent_input_dim=agent_input_dim,
                        task_input_dim=task_input_dim,
                        embedding_dim=TrainParams.EMBEDDING_DIM,
                        arrival_rate_choices=arrival_rate_choices,
                        max_total_tasks=max_total_tasks,
                        max_action_index=max_action_index,
                        reward_config=reward_config,
                    )
                    for i in range(TrainParams.NUM_META_AGENT)
                ]
                for _, test_agent in enumerate(test_agent_list):
                    ray.get(test_agent.set_weights.remote(weights_memory))
                rewards = dict()
                seed_list = copy.deepcopy(test_set)
                evaluate_jobs = [test_agent_list[i].testing.remote(seed=seed_list.pop(), use_baseline=False)
                                 for i in range(TrainParams.NUM_META_AGENT)]
                while True:
                    test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                    reward, seed, meta_id = ray.get(test_done_id)[0]
                    rewards[seed] = reward
                    if seed_list:
                        evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop(), use_baseline=False))
                    if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                        break
                rewards = dict(sorted(rewards.items()))
                test_value = np.stack(list(rewards.values()))
                for a in test_agent_list:
                    ray.kill(a)

                meta_agents = [
                    DynamicRLRunner.remote(
                        i,
                        agent_input_dim,
                        task_input_dim,
                        TrainParams.EMBEDDING_DIM,
                        arrival_rate_choices=arrival_rate_choices,
                        max_total_tasks=max_total_tasks,
                        max_action_index=max_action_index,
                        reward_config=reward_config,
                    )
                    for i in range(TrainParams.NUM_META_AGENT)
                ]

                print('Current model test value:', test_value.mean())
                print('Baseline model value:', baseline_value.mean())
                if test_value.mean() > baseline_value.mean():
                    _, p = ttest_rel(test_value, baseline_value)
                    print('p value:', p)
                    if p < 0.1:
                        print('✅ Current model is significantly better! Updating baseline and saving model at episode', curr_episode)
                        if device != local_device:
                            weights = global_network.to(local_device).state_dict()
                            global_network.to(device)
                        else:
                            weights = global_network.state_dict()
                        baseline_weights = copy.deepcopy(weights)
                        baseline_network.load_state_dict(baseline_weights, strict=False)
                        weights_memory = ray.put(weights)
                        baseline_weights_memory = ray.put(baseline_weights)
                        test_set = logger.generate_test_set_seed()
                        print('Updated test set')
                        baseline_value = None
                        best_perf = test_value.mean()
                        logger.save_model(curr_episode, curr_level, best_perf)
                    else:
                        print('⚠️ Current model is better but not statistically significant (p >= 0.1). Not saving.')
                else:
                    print('❌ Current model is not better than baseline. Not saving.')

                jobs = []
                for i, meta_agent in enumerate(meta_agents):
                    jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
                    curr_episode += 1

            if curr_episode >= 10250:
                break

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
