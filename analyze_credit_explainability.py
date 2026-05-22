#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
信用可解释性分析：
检查零边际贡献 agent 是否拿到了接近零的正向信用。
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from env.task_env import TaskEnv
from dynamic_worker import create_dynamic_model
from scripts.sa_bt.evaluate_my_model_dynamic import load_env, run_model_dynamic
from module_ablation import apply_module_ablation_config, load_module_ablation_config, module_ablation_config_to_json
from parameters import EnvParams, TrainParams
from project_paths import REPO_ROOT, SA_BT_DATASET_ROOT, ensure_checkpoint_exists, resolve_train_dir
from reward_ablation import build_reward_config, normalize_reward_config


DATASET_CONFIGS = {
    'Fixed_Tasks': {
        'n15_s5_h30': {'arrival_rate': 3, 'max_total_tasks': 100, 'simulation_time_limit': 10000},
        'n20_s5_h40': {'arrival_rate': 3, 'max_total_tasks': 100, 'simulation_time_limit': 10000},
        'n20_s5_h50': {'arrival_rate': 3, 'max_total_tasks': 100, 'simulation_time_limit': 10000},
        'n30_s5_h60': {'arrival_rate': 3, 'max_total_tasks': 100, 'simulation_time_limit': 10000},
    },
    'Fixed_Makespan': {
        'n10_s5_t120': {'arrival_rate': 2, 'max_total_tasks': 120, 'simulation_time_limit': 120},
        'n15_s5_t200': {'arrival_rate': 2, 'max_total_tasks': 200, 'simulation_time_limit': 120},
        'n20_s5_t240': {'arrival_rate': 2, 'max_total_tasks': 240, 'simulation_time_limit': 120},
        'n30_s5_t300': {'arrival_rate': 2, 'max_total_tasks': 300, 'simulation_time_limit': 120},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze reward credit explainability on dynamic SA-BT evaluation.')
    parser.add_argument('--folder-name', required=True, help='Model folder under artifacts/models/<folder-name>.')
    parser.add_argument('--dataset-type', required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument('--config-name', required=True, help='Dataset config name, e.g. n15_s5_h30.')
    parser.add_argument('--limit', type=int, default=None, help='Optional max number of env files to analyze.')
    parser.add_argument(
        '--load-from',
        default='best',
        choices=['best', 'current'],
        help='Checkpoint branch to load: best_model or model.',
    )
    parser.add_argument(
        '--model-name',
        default=None,
        choices=['attention', 'myself', 'capam'],
        help='Override TrainParams.MODEL_NAME during model creation.',
    )
    parser.add_argument(
        '--reward-preset',
        default=None,
        choices=['baseline', 'no_shared', 'no_local', 'static_full_penalty'],
        help='Override reward config for the credit analysis. Defaults to artifacts/training/<folder>/reward_config.json if present.',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Override output directory. Defaults to credit_explainability/<folder>/<dataset>/<config>.',
    )
    parser.add_argument(
        '--save-task-details',
        action='store_true',
        help='Persist task-level marginal-credit details as JSONL.',
    )
    return parser.parse_args()


def infer_input_dims():
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


def load_reward_config(folder_name, override_preset=None):
    if override_preset:
        return build_reward_config(override_preset)
    config_path = resolve_train_dir(folder_name) / 'reward_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return normalize_reward_config(json.load(f))
    return build_reward_config('baseline')


def load_model(folder_name, device, load_from='best', model_name=None):
    agent_input_dim, task_input_dim = infer_input_dims()
    network = create_dynamic_model(
        agent_input_dim=agent_input_dim,
        task_input_dim=task_input_dim,
        embedding_dim=TrainParams.EMBEDDING_DIM,
        device=device,
        model_name=model_name,
    )
    checkpoint_path = ensure_checkpoint_exists(folder_name, method_label='Credit analysis')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = 'best_model' if load_from == 'best' else 'model'
    network.load_state_dict(checkpoint[state_key], strict=False)
    network.eval()
    return network


def summarize_agent_credit_stats(stats):
    zero_cases = int(stats.get('zero_marginal_cases', 0))
    zero_positive = int(stats.get('zero_marginal_positive_cases', 0))
    zero_credit_sum = float(stats.get('zero_marginal_credit_sum', 0.0))
    nonzero_cases = int(stats.get('nonzero_marginal_cases', 0))
    nonzero_credit_sum = float(stats.get('nonzero_marginal_credit_sum', 0.0))
    return {
        'zero_marginal_cases': zero_cases,
        'zero_marginal_positive_cases': zero_positive,
        'zero_marginal_positive_rate': zero_positive / max(zero_cases, 1),
        'mean_zero_marginal_credit': zero_credit_sum / max(zero_cases, 1),
        'max_zero_marginal_credit': float(stats.get('max_zero_marginal_credit', 0.0)),
        'nonzero_marginal_cases': nonzero_cases,
        'mean_nonzero_marginal_credit': nonzero_credit_sum / max(nonzero_cases, 1),
    }


def main():
    args = parse_args()
    dataset_info = DATASET_CONFIGS[args.dataset_type].get(args.config_name)
    if dataset_info is None:
        raise ValueError(f'Unknown config {args.dataset_type}/{args.config_name}')

    reward_config = load_reward_config(args.folder_name, override_preset=args.reward_preset)
    module_ablation_config = load_module_ablation_config(args.folder_name)
    apply_module_ablation_config(module_ablation_config)
    default_output_dir = REPO_ROOT / 'credit_explainability' / args.folder_name / args.dataset_type / args.config_name
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and TrainParams.USE_GPU_GLOBAL else 'cpu')
    network = load_model(
        folder_name=args.folder_name,
        device=device,
        load_from=args.load_from,
        model_name=args.model_name,
    )

    env_files = sorted((SA_BT_DATASET_ROOT / args.dataset_type / args.config_name).glob('env_*.pkl'))
    if args.limit is not None:
        env_files = env_files[:args.limit]
    if not env_files:
        raise FileNotFoundError(f'No env files found for {args.dataset_type}/{args.config_name}')

    print('Credit explainability analysis')
    print(f'  model folder: {args.folder_name}')
    print(f'  dataset: {args.dataset_type}/{args.config_name}')
    print(f'  env count: {len(env_files)}')
    print(f'  reward preset: {reward_config["preset"]}')
    print('  module ablation config:')
    print(module_ablation_config_to_json(module_ablation_config))

    per_env_rows = []
    per_agent_rows = []
    task_detail_path = output_dir / 'task_credit_details.jsonl'
    task_detail_file = open(task_detail_path, 'w', encoding='utf-8') if args.save_task_details else None

    for env_file in env_files:
        env = load_env(env_file)
        env.set_reward_config(reward_config)
        initial_task_count = len(env.task_dic)

        results = run_model_dynamic(
            env=env,
            global_network=network,
            device=device,
            max_total_tasks=dataset_info['max_total_tasks'],
            arrival_rate=dataset_info['arrival_rate'],
            simulation_time_limit=dataset_info['simulation_time_limit'],
            random_seed=42,
            sampling=False,
        )

        breakdown, _ = env.get_episode_reward_breakdown(
            max_time=dataset_info['simulation_time_limit'],
            include_credit_details=True,
        )
        env_row = {
            'env_file': env_file.name,
            'initial_task_count': initial_task_count,
            'team_reward': breakdown['team_reward'],
            'shared_term': breakdown['shared_term'],
            'local_term': breakdown['local_term'],
            'global_contribution': breakdown['global_contribution'],
            'progress_share': breakdown['progress_share'],
            'local_contribution': breakdown['local_contribution'],
            'task_scope': breakdown['task_scope'],
            'n_tasks_in_scope': breakdown['n_tasks_in_scope'],
            'n_arrived_tasks': breakdown['n_arrived_tasks'],
            'finished_tasks_in_scope': breakdown['finished_tasks_in_scope'],
            'unfinished_tasks_in_scope': breakdown['unfinished_tasks_in_scope'],
            'termination_reason': results.get('termination_reason', 'unknown'),
            'effective_makespan': results.get('effective_makespan', 0.0),
            'zero_marginal_cases': breakdown['credit_analysis']['zero_marginal_cases'],
            'zero_marginal_positive_cases': breakdown['credit_analysis']['zero_marginal_positive_cases'],
            'zero_marginal_positive_rate': breakdown['credit_analysis']['zero_marginal_positive_rate'],
            'mean_zero_marginal_credit': breakdown['credit_analysis']['mean_zero_marginal_credit'],
            'max_zero_marginal_credit': breakdown['credit_analysis']['max_zero_marginal_credit'],
            'mean_nonzero_marginal_credit': breakdown['credit_analysis']['mean_nonzero_marginal_credit'],
        }
        per_env_rows.append(env_row)

        for agent_id, stats in breakdown.get('per_agent_credit_stats', {}).items():
            agent_summary = summarize_agent_credit_stats(stats)
            agent_summary.update({
                'env_file': env_file.name,
                'agent_id': int(agent_id),
            })
            per_agent_rows.append(agent_summary)

        if task_detail_file is not None:
            for task_detail in breakdown.get('task_credit_details', []):
                task_detail_file.write(json.dumps({
                    'env_file': env_file.name,
                    **task_detail,
                }, ensure_ascii=False) + '\n')

        print(
            f"[{env_file.name}] zero_marginal={env_row['zero_marginal_cases']}, "
            f"positive={env_row['zero_marginal_positive_cases']}, "
            f"rate={env_row['zero_marginal_positive_rate']:.4f}, "
            f"mean_credit={env_row['mean_zero_marginal_credit']:.6f}"
        )

    if task_detail_file is not None:
        task_detail_file.close()

    per_env_df = pd.DataFrame(per_env_rows)
    per_agent_df = pd.DataFrame(per_agent_rows)
    per_env_df.to_csv(output_dir / 'per_env_credit_summary.csv', index=False)
    per_agent_df.to_csv(output_dir / 'per_agent_credit_summary.csv', index=False)

    aggregate = {
        'model_folder': args.folder_name,
        'dataset_type': args.dataset_type,
        'config_name': args.config_name,
        'reward_config': reward_config,
        'n_envs': int(len(per_env_df)),
        'zero_marginal_cases': int(per_env_df['zero_marginal_cases'].sum()),
        'zero_marginal_positive_cases': int(per_env_df['zero_marginal_positive_cases'].sum()),
        'zero_marginal_positive_rate': float(
            per_env_df['zero_marginal_positive_cases'].sum() / max(per_env_df['zero_marginal_cases'].sum(), 1)
        ),
        'mean_zero_marginal_credit': float(
            per_env_df['mean_zero_marginal_credit'].mean() if not per_env_df.empty else 0.0
        ),
        'max_zero_marginal_credit': float(
            per_env_df['max_zero_marginal_credit'].max() if not per_env_df.empty else 0.0
        ),
        'mean_nonzero_marginal_credit': float(
            per_env_df['mean_nonzero_marginal_credit'].mean() if not per_env_df.empty else 0.0
        ),
    }

    with open(output_dir / 'credit_explainability_summary.json', 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print('\nAggregate summary')
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f'\nSaved results to: {output_dir}')


if __name__ == '__main__':
    main()
