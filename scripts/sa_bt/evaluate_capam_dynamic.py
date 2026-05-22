#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 CAPAM 模型评估动态 SA-BT 场景。

该入口直接复用 evaluate_my_model_dynamic.py 的动态事件循环、
终止条件、恢复策略、指标统计和日志格式，只切换：
- 模型类型为 CAPAM
- 默认模型目录为 CAPAM 的 checkpoint 目录

运行：
  python -u scripts/sa_bt/evaluate_capam_dynamic.py
  python -u scripts/sa_bt/evaluate_capam_dynamic.py --folder-name CAPAM_DYNAMIC

  断点运行：
  python -u scripts/sa_bt/train_capam_dynamic.py --folder-name CAPAM_DYNAMIC --load-model --load-from current


"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parameters import SaverParams, TrainParams
from project_paths import set_saver_paths


DEFAULT_CAPAM_FOLDER = 'CAPAM_DYNAMIC'


def parse_args():
    parser = argparse.ArgumentParser(description='Run dynamic evaluation for a CAPAM checkpoint.')
    parser.add_argument(
        '--folder-name',
        default=DEFAULT_CAPAM_FOLDER,
        help='CAPAM model folder under artifacts/models/<folder-name>.',
    )
    return parser.parse_args()


def configure_capam_eval(folder_name):
    TrainParams.MODEL_NAME = 'capam'
    set_saver_paths(SaverParams, folder_name)


def main():
    args = parse_args()
    configure_capam_eval(args.folder_name)
    from scripts.sa_bt import evaluate_my_model_dynamic

    evaluate_my_model_dynamic.main()


if __name__ == '__main__':
    main()
