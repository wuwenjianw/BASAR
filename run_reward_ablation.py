#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态训练入口：按 reward ablation preset 启动训练。

示例:
  python -u run_reward_ablation.py --preset baseline --folder-name ABL_BASELINE
  python -u run_reward_ablation.py --preset no_shared --folder-name ABL_NO_SHARED
"""

import argparse

from parameters import SaverParams
from reward_ablation import build_reward_config, list_reward_presets, reward_config_to_json
import dynamic_driver


def parse_args():
    parser = argparse.ArgumentParser(description='Run dynamic training with a reward ablation preset.')
    parser.add_argument(
        '--preset',
        required=True,
        choices=list_reward_presets(),
        help='Reward ablation preset to use during training.',
    )
    parser.add_argument(
        '--folder-name',
        default=None,
        help='Override SaverParams.FOLDER_NAME for this run.',
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
    return f'ABL_{preset.upper()}'


def main():
    args = parse_args()
    reward_config = build_reward_config(args.preset)
    folder_name = args.folder_name or default_folder_name(args.preset)

    SaverParams.LOAD_MODEL = bool(args.load_model)
    SaverParams.LOAD_FROM = args.load_from

    print('Launching dynamic training with reward preset:')
    print(reward_config_to_json(reward_config))
    print(f'Target folder: {folder_name}')
    print(f'Load model: {SaverParams.LOAD_MODEL} ({SaverParams.LOAD_FROM})')

    dynamic_driver.main(folder_name=folder_name, reward_config=reward_config)


if __name__ == '__main__':
    main()
