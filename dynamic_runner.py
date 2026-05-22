import torch
import ray
import copy

from dynamic_worker import DynamicWorker, create_dynamic_model
from parameters import TrainParams, SaverParams


def configure_worker_torch_threads():
    if TrainParams.USE_GPU:
        return
    threads = max(1, int(getattr(TrainParams, 'WORKER_TORCH_NUM_THREADS', 1)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


class DynamicRunner(object):
    """
    动态任务分配的训练运行器，负责权重同步与Worker调度
    """

    def __init__(self, metaAgentID, agent_input_dim, task_input_dim, embedding_dim,
                 arrival_rate_choices=None, max_total_tasks=100, max_action_index=None,
                 reward_config=None):
        configure_worker_torch_threads()
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')
        self.localNetwork = create_dynamic_model(agent_input_dim, task_input_dim, embedding_dim, self.device)
        self.localBaseline = create_dynamic_model(agent_input_dim, task_input_dim, embedding_dim, self.device)
        # 动态任务配置
        self.arrival_rate_choices = arrival_rate_choices or [0.5, 1.0, 1.5, 2.0]
        self.max_total_tasks = max_total_tasks
        self.max_action_index = max_action_index
        self.reward_config = copy.deepcopy(reward_config)

    def set_weights(self, weights):
        """
        同步主网络权重
        """
        self.localNetwork.load_state_dict(weights, strict=False)

    def set_baseline_weights(self, weights):
        """
        同步基线网络权重
        """
        self.localBaseline.load_state_dict(weights, strict=False)

    def training(self, global_weights, baseline_weights, curr_episode, env_params):
        """
        执行一轮动态任务训练
        """
        print("starting episode {} on metaAgent {}".format(curr_episode, self.metaAgentID))
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)

        save_img = False
        if SaverParams.SAVE_IMG and curr_episode % SaverParams.SAVE_IMG_GAP == 0:
            save_img = True

        worker = DynamicWorker(
            self.metaAgentID,
            self.localNetwork,
            self.localBaseline,
            curr_episode,
            self.device,
            save_img,
            None,
            env_params,
            arrival_rate_choices=self.arrival_rate_choices,
            max_total_tasks=self.max_total_tasks,
            max_action_index=self.max_action_index,
            reward_config=self.reward_config,
        )

        worker.work(curr_episode)

        buffer = worker.experience
        perf_metrics = worker.perf_metrics
        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }
        return buffer, perf_metrics, info

    def testing(self, seed=None, use_baseline=True, sample=False):
        """
        动态任务环境下的测试
        """
        worker = DynamicWorker(
            self.metaAgentID,
            self.localNetwork,
            self.localBaseline,
            0,
            self.device,
            False,
            seed,
            None,
            arrival_rate_choices=self.arrival_rate_choices,
            max_total_tasks=self.max_total_tasks,
            max_action_index=self.max_action_index,
            reward_config=self.reward_config,
        )
        reward = worker.evaluate_episode(seed=seed, use_baseline=use_baseline, sample=sample)
        return reward, seed, self.metaAgentID


@ray.remote(num_cpus=1, num_gpus=(TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT) if TrainParams.USE_GPU else 0)
class DynamicRLRunner(DynamicRunner):
    """
    Ray版本的动态任务运行器
    """
    def __init__(self, metaAgentID, agent_input_dim, task_input_dim, embedding_dim,
                 arrival_rate_choices=None, max_total_tasks=100, max_action_index=None,
                 reward_config=None):
        super().__init__(
            metaAgentID,
            agent_input_dim,
            task_input_dim,
            embedding_dim,
            arrival_rate_choices=arrival_rate_choices,
            max_total_tasks=max_total_tasks,
            max_action_index=max_action_index,
            reward_config=reward_config,
        )
