import pickle
import time
import torch
import numpy as np
import random
from env.task_env import TaskEnv
from attention import AttentionNet
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical


def discount(x, gamma):
    """
    计算折扣奖励
    
    使用指定的折扣因子对奖励序列进行折扣处理，
    用于强化学习中的回报计算。
    
    Args:
        x: 奖励序列
        gamma: 折扣因子
        
    Returns:
        折扣后的奖励序列
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def zero_padding(x, padding_size, length):
    """
    对张量进行零填充
    
    Args:
        x: 输入张量
        padding_size: 目标填充大小
        length: 当前长度
        
    Returns:
        填充后的张量
    """
    pad = torch.nn.ZeroPad2d((0, 0, 0, padding_size - length))
    x = pad(x)
    return x


class Worker:
    """
    工作器类，负责与环境交互并收集训练数据
    
    Worker是强化学习系统中的核心组件，主要职责包括：
    1. 管理任务环境的交互
    2. 执行智能体的决策过程
    3. 收集经验数据用于训练
    4. 计算性能指标
    5. 处理基线测试
    
    在分布式训练中，每个Worker实例独立运行，
    与各自的环境实例交互，收集数据后统一汇总到主进程。
    """
    
    def __init__(self, mete_agent_id, local_network, local_baseline, global_step,
                device='cuda', save_image=False, seed=None, env_params=None, reward_config=None):
        """
        初始化Worker实例
        
        Args:
            mete_agent_id: 元智能体ID
            local_network: 本地主网络模型
            local_baseline: 本地基线网络模型
            global_step: 全局训练步数
            device: 计算设备（'cuda' 或 'cpu'）
            save_image: 是否保存可视化图像
            seed: 随机种子
            env_params: 环境参数配置
        """
        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.reward_config = copy.deepcopy(reward_config)
        
        # 设置环境参数，如果未提供则使用默认值
        if env_params is None:
            env_params = [EnvParams.SPECIES_AGENTS_RANGE, EnvParams.SPECIES_RANGE, EnvParams.TASKS_RANGE]
            
        # 创建任务环境实例
        self.env = TaskEnv(
            *env_params,
            EnvParams.TRAIT_DIM,
            EnvParams.DECISION_DIM,
            seed=seed,
            plot_figure=save_image,
            reward_config=self.reward_config,
        )
        
        # 创建基线测试用的环境副本
        self.baseline_env = copy.deepcopy(self.env)
        
        # 设置网络模型
        self.local_baseline = local_baseline  # 基线网络
        self.local_net = local_network        # 主网络
        
        # 初始化数据结构
        self.experience = {idx:[] for idx in range(7)}  # 经验数据缓冲区
        self.episode_number = None                       # 当前轮次编号
        self.perf_metrics = {}                          # 性能指标字典
        self.p_rnn_state = {}                           # RNN状态字典
        self.max_time = EnvParams.MAX_TIME              # 最大时间限制

    def run_episode(self, episode_number=0, training=True, sample=False, max_waiting=False):
        """
        运行一个完整的环境交互回合
        
        这是Worker的核心方法，执行智能体与环境的完整交互过程：
        1. 环境决策循环
        2. 智能体观察和动作选择
        3. 环境状态更新
        4. 经验数据收集
        5. 性能指标计算
        
        Args:
            episode_number: 当前训练轮数（用于V13自适应奖励）
            training: 是否为训练模式
            sample: 是否使用采样策略选择动作
            max_waiting: 是否强制最大等待时间
            
        Returns:
            terminal_reward: 终端奖励
            buffer_dict: 经验数据缓冲区
            perf_metrics: 性能指标字典
        """
        # 初始化数据缓冲区和性能指标
        buffer_dict = {idx:[] for idx in range(7)}
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0
        
        # 主要的环境交互循环
        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME and current_action_index < 300:
            with torch.no_grad():
                # 获取下一个决策时刻的智能体列表
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                
                # 随机打乱智能体顺序，增加探索性
                random.shuffle(release_agents[0])
                finished_task = []
                
                # 处理所有需要做决策的智能体
                while release_agents[0] or release_agents[1]:
                    # 选择下一个要处理的智能体
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.env.agent_dic[agent_id]
                    
                    # 获取智能体的观察信息
                    task_info, total_agents, mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting))
                    
                    # 检查是否所有任务都被阻塞
                    block_flag = mask[0, 1:].all().item()
                    
                    # 处理任务阻塞情况
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                        
                    # 在训练模式下对观察进行填充
                    if training:
                        task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                        
                    # 准备网络输入
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    
                    # 通过神经网络获取动作概率
                    probs, _ = self.local_net(task_info, total_agents, mask, index)
                    
                    # 根据模式选择动作
                    if training:
                        # 训练模式：使用采样策略
                        action = Categorical(probs).sample()
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        # 测试模式：根据sample参数决定策略
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                            
                    # 执行动作并获取奖励
                    r, doable, f_t = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    finished_task.append(f_t)
                    
                    # 在训练模式下收集经验数据
                    if training and doable:
                        buffer_dict[0] += total_agents      # 智能体信息
                        buffer_dict[1] += task_info         # 任务信息
                        buffer_dict[2] += action.unsqueeze(0)  # 动作
                        buffer_dict[3] += mask              # 掩码
                        buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # 奖励占位符
                        buffer_dict[5] += index             # 智能体索引
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)  # 优势值占位符
                        current_action_index += 1
                        
                # 检查环境是否结束
                self.env.finished = self.env.check_finished()
                decision_step += 1

        # 计算终端奖励和任务完成情况（V13：传递episode_number用于自适应奖励）
        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time, episode_number)

        # 计算性能指标
        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]  # 成功率
        perf_metrics['makespan'] = [self.env.current_time]                          # 完成时间跨度
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]  # 平均任务开始时间
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]  # 平均等待时间
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]  # 总行驶距离
        perf_metrics['efficiency'] = [self.env.get_efficiency()]                    # 效率指标
        
        # 计算截止日期相关指标
        deadline_violations = []  # 存储截止时间违约信息
        on_time_count = 0         # 在截止日期前完成的任务数
        finished_count = 0        # 已完成任务总数
        
        for task_id, task in self.env.task_dic.items():
            time_finish = task['time_finish']
            deadline = task['deadline']
            
            # 通过 time_finish > 0 判断任务是否完成
            if time_finish > 0:
                finished_count += 1
                
                if time_finish <= deadline:
                    on_time_count += 1
                else:
                    # 记录违约任务及其超时量
                    deadline_violations.append(time_finish - deadline)
        
        # 计算截止日期前完成的成功率
        if finished_count > 0:
            perf_metrics['deadline_success_rate'] = [on_time_count / finished_count]
        else:
            perf_metrics['deadline_success_rate'] = [0.0]
        
        # 计算平均截止时间违约量
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
        
        # print(perf_metrics)

        return terminal_reward, buffer_dict, perf_metrics

    def baseline_test(self):
        """
        执行基线模型测试
        
        使用基线网络在测试环境中运行完整回合，
        用于评估当前模型性能，不收集训练数据。
        
        Returns:
            reward: 测试回合获得的总奖励
        """
        # 关闭图像保存以加快测试速度
        self.baseline_env.plot_figure = False
        perf_metrics = {}
        current_action_index = 0
        start = time.time()
        
        # 测试交互循环
        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            with torch.no_grad():
                # 获取决策智能体
                release_agents, current_time = self.baseline_env.next_decision()
                random.shuffle(release_agents[0])
                self.baseline_env.current_time = current_time
                
                # 超时保护机制
                if time.time() - start > 30:
                    break
                    
                # 处理所有需要决策的智能体
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.baseline_env.agent_dic[agent_id]
                    
                    # 获取智能体观察
                    task_info, total_agents, mask = self.convert_torch(self.baseline_env.agent_observe(agent_id, False))
                    
                    # 检查返回标志
                    return_flag = mask[0, 1:].all().item()
                    
                    # 处理阻塞情况
                    if return_flag and not np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')):
                        self.baseline_env.agent_dic[agent_id]['no_choice'] = return_flag
                        continue
                    elif return_flag and np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                        
                    # 观察填充
                    task_info, total_agents, mask = self.obs_padding(task_info, total_agents, mask)
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    
                    # 使用基线网络进行决策
                    probs, _ = self.local_baseline(task_info, total_agents, mask, index)
                    action = torch.argmax(probs, 1)  # 选择概率最大的动作
                    
                    # 执行动作
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    current_action_index += 1
                    
                # 检查是否完成
                self.baseline_env.finished = self.baseline_env.check_finished()

        # 获取测试奖励（baseline测试不需要传递episode_number）
        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def work(self, episode_number):
        """
        执行一轮完整的工作流程
        
        这是Worker的主要工作方法，负责：
        1. 运行多个POMO（Parallel Optimization using Multiple Objectives）实例
        2. 收集所有实例的经验数据和性能指标
        3. 计算基线奖励和优势值
        4. 合并数据到主缓冲区
        5. 可选地保存可视化动画
        
        Args:
            episode_number: 当前回合编号
        """
        baseline_rewards = []  # 存储所有实例的基线奖励
        buffers = []          # 存储所有实例的经验缓冲区
        metrics = []          # 存储所有实例的性能指标
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK
        
        # 运行多个POMO实例
        for _ in range(TrainParams.POMO_SIZE):
            # 重新初始化环境状态
            self.env.init_state()
            
            # 运行一个完整回合
            terminal_reward, buffer, perf_metrics = self.run_episode(episode_number, True, max_waiting)
            
            # 检查奖励是否有效
            if terminal_reward is np.nan:
                max_waiting = True
                continue
            # 收集有效的训练数据
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)
            
        # 计算基线奖励（所有实例的平均奖励）
        baseline_reward = np.nanmean(baseline_rewards)

        # 处理每个缓冲区的数据
        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                # 更新优势值（索引6对应advantage）
                if key == 6:
                    for i in range(len(buffer[key])):
                        # 优势值 = 当前实例奖励 - 平均基线奖励
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                        
                # 将缓冲区数据合并到主经验缓冲区
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        # 合并所有性能指标
        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        # 如果需要保存图像，则生成动画
        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except:
                pass
                
        self.episode_number = episode_number

    def convert_torch(self, args):
        """
        将输入数据转换为PyTorch张量
        
        将环境返回的numpy数组或其他格式的数据转换为PyTorch张量，
        并移动到指定的计算设备上。
        
        Args:
            args: 包含多个数据项的列表或元组
            
        Returns:
            data: 转换后的PyTorch张量列表
        """
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data

    @staticmethod
    def obs_padding(task_info, agents, mask):
        """
        对观察数据进行填充以匹配网络输入要求
        
        由于不同环境实例可能有不同数量的任务和智能体，
        需要将它们填充到统一的维度以便批量处理。
        
        Args:
            task_info: 任务信息张量
            agents: 智能体信息张量
            mask: 掩码张量
            
        Returns:
            填充后的任务信息、智能体信息和掩码张量
        """
        # 填充任务信息到最大任务数量+1
        task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
        
        # 填充智能体信息到最大智能体数量
        agents = F.pad(agents, (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]), 'constant', 0)
        
        # 填充掩码，填充部分设为1（表示不可选择）
        mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
        
        return task_info, agents, mask


if __name__ == '__main__':
    """
    测试代码示例
    
    演示如何使用Worker类进行独立测试：
    1. 创建计算设备和神经网络
    2. 循环创建Worker实例
    3. 执行工作流程并记录结果
    
    这个测试代码可用于调试Worker的功能或进行性能基准测试。
    """
    device = 'cpu' # torch.device('cuda')
    
    # 可选：设置随机种子和加载预训练模型
    # torch.manual_seed(9)
    # checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
    
    # 创建神经网络模型
    localNetwork = AttentionNet(11, 15, 128).to(device)
    # localNetwork.load_state_dict(checkpoint['best_model'])
    
    # 运行100个测试回合
    for i in range(100):
        # 创建Worker实例，使用不同的随机种子
        worker = Worker(1, localNetwork, localNetwork, 0, device=device, seed=i, save_image=False)
        
        # 执行工作流程
        worker.work(i)
        print(i)
