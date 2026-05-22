#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为指定模型文件夹运行动态基准评估。
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module_ablation import apply_module_ablation_config, load_module_ablation_config, module_ablation_config_to_json
from parameters import SaverParams, TrainParams
from project_paths import set_saver_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluate_my_model_dynamic.py for a specific model folder.')
    parser.add_argument('--folder-name', required=True, help='Model folder under artifacts/models/<folder-name>.')
    parser.add_argument(
        '--model-name',
        default=None,
        choices=['attention', 'myself', 'capam'],
        help='Optional override for TrainParams.MODEL_NAME.',
    )
    parser.add_argument(
        '--protocol',
        choices=['Fixed_Tasks', 'Fixed_Makespan'],
        action='append',
        help='Optional SA-BT protocol to evaluate. Repeat to evaluate multiple protocols. Defaults to all protocols.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from scripts.sa_bt import evaluate_my_model_dynamic

    set_saver_paths(SaverParams, args.folder_name)
    module_ablation_config = load_module_ablation_config(args.folder_name)
    apply_module_ablation_config(module_ablation_config)
    if args.model_name:
        TrainParams.MODEL_NAME = args.model_name
    print('Module ablation config for evaluation:')
    print(module_ablation_config_to_json(module_ablation_config))
    print(f'Model name for evaluation: {TrainParams.MODEL_NAME}')
    evaluate_my_model_dynamic.main(protocol_filter=args.protocol)


if __name__ == '__main__':
    main()
