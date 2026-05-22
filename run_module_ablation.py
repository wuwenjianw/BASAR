#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态训练入口：按模块消融 preset 启动训练。

示例:
  python -u run_module_ablation.py --preset baseline --folder-name MODABL_BASELINE --model-name myself
  python -u run_module_ablation.py --preset single_self_attention --folder-name MODABL_NO_CROSS --model-name myself
  python -u run_module_ablation.py --preset global_mlp --folder-name MODABL_GLOBAL_MLP --model-name myself
"""

import argparse

import dynamic_driver
from module_ablation import (
    apply_module_ablation_config,
    build_module_ablation_config,
    list_module_ablation_presets,
    module_ablation_config_to_json,
)
from parameters import SaverParams, TrainParams
from reward_ablation import build_reward_config


def parse_args():
    parser = argparse.ArgumentParser(description='Run dynamic training with a module ablation preset.')
    parser.add_argument(
        '--preset',
        required=True,
        choices=list_module_ablation_presets(),
        help='Module ablation preset to use during training.',
    )
    parser.add_argument(
        '--folder-name',
        default=None,
        help='Override SaverParams.FOLDER_NAME for this run.',
    )
    parser.add_argument(
        '--model-name',
        default=TrainParams.MODEL_NAME,
        choices=['attention', 'myself', 'capam'],
        help='Model family to train. Use myself for your current model.',
    )
    parser.add_argument(
        '--reward-preset',
        default='baseline',
        choices=['baseline', 'no_shared', 'no_local', 'static_full_penalty'],
        help='Reward preset used during training. Module ablation experiments normally keep this at baseline.',
    )
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='Resume from the checkpoint under the target folder.',
    )
    parser.add_argument(
        '--load-from',
        default='current',
        choices=['current', 'best'],
        help='Checkpoint branch to restore when --load-model is used.',
    )
    return parser.parse_args()


def default_folder_name(preset):
    return f'MODABL_{preset.upper()}'


def main():
    args = parse_args()
    reward_config = build_reward_config(args.reward_preset)
    module_ablation_config = build_module_ablation_config(args.preset)
    module_ablation_config = apply_module_ablation_config(module_ablation_config)
    folder_name = args.folder_name or default_folder_name(args.preset)

    TrainParams.MODEL_NAME = args.model_name
    SaverParams.LOAD_MODEL = bool(args.load_model)
    SaverParams.LOAD_FROM = args.load_from

    print('Launching dynamic training with module ablation preset:')
    print(module_ablation_config_to_json(module_ablation_config))
    print(f'Model name: {TrainParams.MODEL_NAME}')
    print(f'Reward preset: {args.reward_preset}')
    print(f'Target folder: {folder_name}')
    print(f'Load model: {SaverParams.LOAD_MODEL} ({SaverParams.LOAD_FROM})')

    dynamic_driver.main(
        folder_name=folder_name,
        reward_config=reward_config,
        module_ablation_config=module_ablation_config,
    )


if __name__ == '__main__':
    main()
