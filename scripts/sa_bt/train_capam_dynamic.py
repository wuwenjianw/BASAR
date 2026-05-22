#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态训练入口：重新训练 CAPAM 模型。

示例:
  python -u scripts/sa_bt/train_capam_dynamic.py
  python -u scripts/sa_bt/train_capam_dynamic.py --folder-name CAPAM_DYNAMIC
  python -u scripts/sa_bt/train_capam_dynamic.py --folder-name CAPAM_DYNAMIC --load-model --load-from best
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module_ablation import build_module_ablation_config, module_ablation_config_to_json
from parameters import SaverParams, TrainParams
from reward_ablation import build_reward_config, reward_config_to_json


DEFAULT_CAPAM_FOLDER = 'CAPAM_DYNAMIC'


def parse_args():
    parser = argparse.ArgumentParser(description='Retrain the CAPAM model on the dynamic environment.')
    parser.add_argument(
        '--folder-name',
        default=DEFAULT_CAPAM_FOLDER,
        help='Checkpoint/log folder under artifacts/models, artifacts/training, artifacts/gifs.',
    )
    parser.add_argument(
        '--reward-preset',
        default='baseline',
        choices=['baseline', 'no_shared', 'no_local', 'static_full_penalty'],
        help='Reward preset used during training.',
    )
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='Resume from checkpoint under the target folder.',
    )
    parser.add_argument(
        '--load-from',
        default='current',
        choices=['current', 'best'],
        help='Checkpoint branch to restore when --load-model is used.',
    )
    return parser.parse_args()


def configure_capam_training(args):
    TrainParams.MODEL_NAME = 'capam'
    SaverParams.LOAD_MODEL = bool(args.load_model)
    SaverParams.LOAD_FROM = args.load_from

    reward_config = build_reward_config(args.reward_preset)
    module_ablation_config = build_module_ablation_config('baseline')
    return reward_config, module_ablation_config


def main():
    args = parse_args()
    reward_config, module_ablation_config = configure_capam_training(args)

    print('Launching dynamic CAPAM training...')
    print(f'Model name: {TrainParams.MODEL_NAME}')
    print(f'Target folder: {args.folder_name}')
    print(f'Load model: {SaverParams.LOAD_MODEL} ({SaverParams.LOAD_FROM})')
    print('Reward config:')
    print(reward_config_to_json(reward_config))
    print('Module ablation config:')
    print(module_ablation_config_to_json(module_ablation_config))

    import dynamic_driver

    dynamic_driver.main(
        folder_name=args.folder_name,
        reward_config=reward_config,
        module_ablation_config=module_ablation_config,
    )


if __name__ == '__main__':
    main()
