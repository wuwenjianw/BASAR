#!/usr/bin/env python3
"""
详细分析HRLF和Greedy的每一步决策差异
"""
import pickle
import numpy as np
import torch
from torch.distributions import Categorical
from dynamic_centralized_planner import DynamicCentralizedPlanner
from attention import AttentionNet
from parameters import EnvParams, TrainParams
from project_paths import SA_BT_DATASET_ROOT, ensure_checkpoint_exists, model_dir
import copy

# 兼容性包装器
class CompatibilityWrapper:
    def __init__(self, original_env):
        self.original_env = original_env
        self.has_battery = hasattr(original_env, 'initial_battery')
        self.has_deadline = getattr(original_env, 'use_deadline', True)
        self.original_env.use_deadline = False
    
    def agent_observe(self, agent_id, max_waiting=False):
        tasks_info, agents_info, mask = self.original_env.agent_observe(agent_id, max_waiting)
        agents_info = agents_info[:, :, :-1]
        return tasks_info, agents_info, mask
    
    def __getattr__(self, name):
        return getattr(self.original_env, name)
    
    def __setattr__(self, name, value):
        if name in ['original_env', 'has_battery', 'has_deadline']:
            object.__setattr__(self, name, value)
        else:
            setattr(self.original_env, name, value)


def create_hrlf_decision_maker(global_network, device):
    """创建HRLF决策器"""
    def hrlf_decision_maker(agent_id, available_tasks, env):
        if not available_tasks:
            return None
        
        original_env = env.original_env if isinstance(env, CompatibilityWrapper) else env
        agent = original_env.agent_dic[agent_id]
        
        if original_env.check_battery_critical(agent_id):
            return None
        
        with torch.no_grad():
            task_info, total_agents, mask = env.agent_observe(agent_id, max_waiting=False)
            
            # 添加额外mask
            additional_mask = np.ones(len(original_env.task_dic) + 1, dtype=bool)
            additional_mask[0] = False
            for task_id in available_tasks:
                additional_mask[task_id + 1] = False
            mask = np.logical_or(mask, additional_mask.reshape(1, -1))
            
            task_info = torch.from_numpy(task_info).float().to(device)
            total_agents = torch.from_numpy(total_agents).float().to(device)
            mask = torch.from_numpy(mask).to(device)
            
            if mask[0, 1:].all().item():
                return None
            
            index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(device)
            probs, _ = global_network(task_info, total_agents, mask, index)
            
            action = torch.argmax(probs, dim=1)
            task_idx = action.item()
            
            if task_idx == 0 or task_idx > len(original_env.task_dic):
                return None
            
            task_id = task_idx - 1
            
            if task_id in available_tasks:
                task = original_env.task_dic[task_id]
                can_contribute = np.any((agent['abilities'] > 0) & (task['status'] > 0))
                if not can_contribute:
                    return None
                if not original_env.can_reach_with_battery(agent_id, task):
                    return None
                return task_id
        
        return None
    
    return hrlf_decision_maker


class DecisionTracker(DynamicCentralizedPlanner):
    """追踪每一步决策的规划器"""
    def __init__(self, *args, max_tracked_plans=5, track_details=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_plans = []
        self.max_tracked_plans = max_tracked_plans
        self.track_details = track_details
        self.plan_counter = 0
    
    def _plan(self, replan=False):
        """重写_plan以追踪决策细节"""
        plan_time = self.env.current_time
        
        # 获取当前状态
        replanning_agents = self._get_idle_agents()
        available_tasks = self._get_available_tasks()
        
        # 如果需要追踪且未达到上限
        if self.track_details and self.plan_counter < self.max_tracked_plans:
            # 深拷贝当前环境状态
            tracked_info = {
                'plan_id': self.plan_counter,
                'plan_time': plan_time,
                'is_replan': replan,
                'replanning_agents': replanning_agents.copy(),
                'available_tasks': available_tasks.copy(),
                'agent_states': {},
                'task_states': {},
                'actual_decisions': []  # 记录实际的决策
            }
            
            # 记录智能体状态
            for agent_id in replanning_agents:
                agent = self.env.agent_dic[agent_id]
                tracked_info['agent_states'][agent_id] = {
                    'location': agent['location'].copy(),
                    'abilities': agent['abilities'].copy(),
                    'battery': agent.get('battery', 100),
                    'current_task': agent['current_task']
                }
            
            # 记录任务状态
            for task_id in available_tasks:
                task = self.env.task_dic[task_id]
                tracked_info['task_states'][task_id] = {
                    'location': task['location'].copy(),
                    'status': task['status'].copy(),
                    'requirements': task['requirements'].copy(),
                    'members': task['members'].copy(),
                    'appear_time': task.get('appear_time', 0)
                }
            
            # 模拟决策过程
            remaining_tasks = set(available_tasks)
            for agent_id in replanning_agents:
                if not remaining_tasks:
                    break
                
                current_available = list(remaining_tasks)
                
                # 调用决策器
                if self.decision_maker is not None:
                    selected_task = self.decision_maker(agent_id, current_available, self.env)
                else:
                    # 贪婪算法
                    agent = self.env.agent_dic[agent_id]
                    best_task = None
                    min_distance = float('inf')
                    
                    for task_id in current_available:
                        task = self.env.task_dic[task_id]
                        
                        # 检查能力
                        can_contribute = np.any((agent['abilities'] > 0) & (task['status'] > 0))
                        if not can_contribute:
                            continue
                        
                        # 检查电量
                        if not self.env.can_reach_with_battery(agent_id, task):
                            continue
                        
                        distance = np.linalg.norm(agent['location'] - task['location'])
                        if distance < min_distance:
                            min_distance = distance
                            best_task = task_id
                    
                    selected_task = best_task
                
                # 记录决策
                if selected_task is not None:
                    task = self.env.task_dic[selected_task]
                    tracked_info['actual_decisions'].append({
                        'agent_id': agent_id,
                        'task_id': selected_task,
                        'distance': np.linalg.norm(agent['location'] - task['location'])
                    })
                    remaining_tasks.remove(selected_task)
                else:
                    tracked_info['actual_decisions'].append({
                        'agent_id': agent_id,
                        'task_id': None,
                        'distance': 0
                    })
            
            self.tracked_plans.append(tracked_info)
        
        self.plan_counter += 1
        
        # 调用父类的_plan
        return super()._plan(replan)


def analyze_single_planning(tracked_info, decision_maker_name):
    """分析单次规划的详细决策 - 使用实际记录的决策"""
    print(f"\n{'='*100}")
    print(f"第 {tracked_info['plan_id']+1} 次规划 - {decision_maker_name}")
    print(f"{'='*100}")
    print(f"时间: {tracked_info['plan_time']:.2f} 分钟 | 类型: {'重规划' if tracked_info['is_replan'] else '初始规划'}")
    print(f"待规划智能体: {len(tracked_info['replanning_agents'])} 个 - {tracked_info['replanning_agents']}")
    print(f"可用任务: {len(tracked_info['available_tasks'])} 个 - {tracked_info['available_tasks'][:10]}{'...' if len(tracked_info['available_tasks']) > 10 else ''}")
    
    decisions = tracked_info['actual_decisions']
    
    print(f"\n{'智能体':<8} {'位置':<20} {'能力':<15} {'选择任务':<10} {'任务位置':<20} {'距离':<10} {'任务需求':<15}")
    print("-" * 100)
    
    for decision in decisions:
        agent_id = decision['agent_id']
        agent_state = tracked_info['agent_states'][agent_id]
        task_id = decision['task_id']
        
        if task_id is not None and task_id in tracked_info['task_states']:
            task_state = tracked_info['task_states'][task_id]
            distance = decision['distance']
            
            print(f"{agent_id:<8} {str(agent_state['location']):<20} {str(agent_state['abilities']):<15} "
                  f"{task_id:<10} {str(task_state['location']):<20} {distance:<10.2f} {str(task_state['requirements']):<15}")
        else:
            print(f"{agent_id:<8} {str(agent_state['location']):<20} {str(agent_state['abilities']):<15} "
                  f"{'无分配':<10} {'-':<20} {'-':<10} {'-':<15}")
    
    # 统计剩余任务
    assigned_tasks = set(d['task_id'] for d in decisions if d['task_id'] is not None)
    remaining_tasks = set(tracked_info['available_tasks']) - assigned_tasks
    
    print(f"\n剩余未分配任务: {len(remaining_tasks)} 个")
    if len(remaining_tasks) > 0 and len(remaining_tasks) <= 10:
        print(f"  任务ID: {sorted(remaining_tasks)}")
    
    return decisions


# ============================================================================
# 主程序
# ============================================================================

print("=" * 100)
print("详细决策步骤分析 - HRLF vs Greedy")
print("=" * 100)

# 设置参数
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

# 加载HRLF模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = model_dir('save_baseline')

print(f"\n加载HRLF模型: {model_path}/checkpoint.pth")

global_network = AttentionNet(
    TrainParams.AGENT_INPUT_DIM,
    TrainParams.TASK_INPUT_DIM,
    TrainParams.EMBEDDING_DIM
).to(device)

checkpoint = torch.load(ensure_checkpoint_exists('save_baseline', method_label='HRLF'), map_location=device, weights_only=False)
global_network.load_state_dict(checkpoint['best_model'], strict=False)
global_network.eval()

print("✓ 模型加载成功\n")

hrlf_decision_maker = create_hrlf_decision_maker(global_network, device)

# 测试环境
env_file = SA_BT_DATASET_ROOT / 'Fixed_Tasks' / 'n10_s5_h20' / 'env_005.pkl'

# ============================================================================
# 追踪Greedy决策
# ============================================================================
print("=" * 100)
print("运行 Greedy 算法并追踪前20次规划...")
print("=" * 100)

with open(env_file, 'rb') as f:
    env_greedy = pickle.load(f)

tracker_greedy = DecisionTracker(
    env=env_greedy,
    max_total_tasks=100,
    dynamic_task_arrival_rate=0.4,
    simulation_time_limit=400,
    random_seed=42,
    verbose=False,
    decision_maker=None,
    max_waiting_time=30,
    max_tracked_plans=20,  # 追踪前20次规划
    track_details=True
)

results_greedy = tracker_greedy.run()

print(f"\nGreedy完成: {results_greedy['finished_tasks']}/{results_greedy['total_tasks']} 任务")
print(f"Makespan: {results_greedy['simulation_time']:.2f} 分钟")

# ============================================================================
# 追踪HRLF决策
# ============================================================================
print("\n" + "=" * 100)
print("运行 HRLF 并追踪前20次规划...")
print("=" * 100)

with open(env_file, 'rb') as f:
    env_hrlf = pickle.load(f)

wrapped_env = CompatibilityWrapper(env_hrlf)

tracker_hrlf = DecisionTracker(
    env=wrapped_env,
    max_total_tasks=100,
    dynamic_task_arrival_rate=0.4,
    simulation_time_limit=400,
    random_seed=42,
    verbose=False,
    decision_maker=hrlf_decision_maker,
    max_waiting_time=30,
    max_tracked_plans=20,  # 追踪前20次规划
    track_details=True
)

results_hrlf = tracker_hrlf.run()

print(f"\nHRLF完成: {results_hrlf['finished_tasks']}/{results_hrlf['total_tasks']} 任务")
print(f"Makespan: {results_hrlf['simulation_time']:.2f} 分钟")

# ============================================================================
# 详细对比每次规划
# ============================================================================
print("\n" + "=" * 100)
print("详细决策对比")
print("=" * 100)

for i in range(min(len(tracker_greedy.tracked_plans), len(tracker_hrlf.tracked_plans))):
    greedy_plan = tracker_greedy.tracked_plans[i]
    hrlf_plan = tracker_hrlf.tracked_plans[i]
    
    print(f"\n\n{'#' * 100}")
    print(f"规划 #{i+1} 对比")
    print(f"{'#' * 100}")
    
    # 分析Greedy决策（使用保存的状态）
    greedy_decisions = analyze_single_planning(
        greedy_plan,
        "Greedy算法"
    )
    
    # 分析HRLF决策（使用保存的状态）
    hrlf_decisions = analyze_single_planning(
        hrlf_plan,
        "HRLF"
    )
    
    # 对比决策差异
    print(f"\n{'='*100}")
    print(f"决策差异分析")
    print(f"{'='*100}")
    
    greedy_dict = {d['agent_id']: d for d in greedy_decisions}
    hrlf_dict = {d['agent_id']: d for d in hrlf_decisions}
    
    all_agents = set(greedy_dict.keys()) | set(hrlf_dict.keys())
    
    print(f"\n{'智能体':<10} {'Greedy任务':<15} {'Greedy距离':<15} {'HRLF任务':<15} {'HRLF距离':<15} {'差异':<10}")
    print("-" * 100)
    
    diff_count = 0
    for agent_id in sorted(all_agents):
        greedy_task = greedy_dict.get(agent_id, {}).get('task_id', '无')
        greedy_dist = greedy_dict.get(agent_id, {}).get('distance', 0)
        hrlf_task = hrlf_dict.get(agent_id, {}).get('task_id', '无')
        hrlf_dist = hrlf_dict.get(agent_id, {}).get('distance', 0)
        
        is_diff = greedy_task != hrlf_task
        diff_count += 1 if is_diff else 0
        diff_marker = "⚠️ 不同" if is_diff else "✓ 相同"
        
        print(f"{agent_id:<10} {str(greedy_task):<15} {greedy_dist:<15.2f} "
              f"{str(hrlf_task):<15} {hrlf_dist:<15.2f} {diff_marker:<10}")
    
    print(f"\n差异统计: {diff_count}/{len(all_agents)} 个智能体的决策不同")

# ============================================================================
# 总体统计分析
# ============================================================================
print("\n\n" + "=" * 100)
print("总体统计分析")
print("=" * 100)

total_plans = min(len(tracker_greedy.tracked_plans), len(tracker_hrlf.tracked_plans))
total_agents_compared = 0
total_different_decisions = 0
distance_comparisons = []

for i in range(total_plans):
    greedy_plan = tracker_greedy.tracked_plans[i]
    hrlf_plan = tracker_hrlf.tracked_plans[i]
    
    # 获取实际决策
    greedy_decisions = greedy_plan['actual_decisions']
    hrlf_decisions = hrlf_plan['actual_decisions']
    
    greedy_dict = {d['agent_id']: d for d in greedy_decisions}
    hrlf_dict = {d['agent_id']: d for d in hrlf_decisions}
    
    all_agents = set(greedy_dict.keys()) | set(hrlf_dict.keys())
    
    for agent_id in all_agents:
        total_agents_compared += 1
        greedy_task = greedy_dict.get(agent_id, {}).get('task_id')
        hrlf_task = hrlf_dict.get(agent_id, {}).get('task_id')
        
        if greedy_task != hrlf_task:
            total_different_decisions += 1
        
        # 比较距离
        if greedy_task is not None and hrlf_task is not None:
            greedy_dist = greedy_dict[agent_id]['distance']
            hrlf_dist = hrlf_dict[agent_id]['distance']
            distance_comparisons.append({
                'plan': i+1,
                'agent': agent_id,
                'greedy_dist': greedy_dist,
                'hrlf_dist': hrlf_dist,
                'diff': hrlf_dist - greedy_dist
            })

print(f"\n分析规划次数: {total_plans}")
print(f"总决策次数: {total_agents_compared}")
print(f"决策不同次数: {total_different_decisions}")
print(f"决策差异率: {total_different_decisions/total_agents_compared*100:.1f}%")

if distance_comparisons:
    import numpy as np
    diffs = [d['diff'] for d in distance_comparisons]
    greedy_dists = [d['greedy_dist'] for d in distance_comparisons]
    hrlf_dists = [d['hrlf_dist'] for d in distance_comparisons]
    
    print(f"\n距离统计（仅统计有效任务分配）:")
    print(f"  有效决策对比数: {len(distance_comparisons)}")
    print(f"  Greedy平均距离: {np.mean(greedy_dists):.3f}")
    print(f"  HRLF平均距离: {np.mean(hrlf_dists):.3f}")
    print(f"  平均距离差异: {np.mean(diffs):.3f} ({'+'if np.mean(diffs)>0 else ''}{np.mean(diffs)/np.mean(greedy_dists)*100:.1f}%)")
    print(f"  距离差异标准差: {np.std(diffs):.3f}")
    print(f"  HRLF选择更远任务次数: {sum(1 for d in diffs if d > 0.01)} ({sum(1 for d in diffs if d > 0.01)/len(diffs)*100:.1f}%)")
    print(f"  HRLF选择更近任务次数: {sum(1 for d in diffs if d < -0.01)} ({sum(1 for d in diffs if d < -0.01)/len(diffs)*100:.1f}%)")

print("\n" + "=" * 100)
print("分析完成")
print("=" * 100)
