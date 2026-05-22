#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""项目内统一路径辅助。"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
MODEL_ROOT = ARTIFACTS_ROOT / "models"
TRAIN_ROOT = ARTIFACTS_ROOT / "training"
GIFS_ROOT = ARTIFACTS_ROOT / "gifs"
DATA_ROOT = REPO_ROOT / "data"
DOCS_ROOT = REPO_ROOT / "docs"

SA_BT_DATASET_ROOT = DATA_ROOT / "testsets" / "sa_bt"
SA_AT_SCALING_DATASET_ROOT = DATA_ROOT / "testsets" / "sa_at_scaling"
MA_AT_DYNAMIC_DATASET_ROOT = DATA_ROOT / "testsets" / "ma_at_dynamic"


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)


def model_dir(folder_name):
    return MODEL_ROOT / folder_name


def train_dir(folder_name):
    return TRAIN_ROOT / folder_name


def gifs_dir(folder_name):
    return GIFS_ROOT / folder_name


def resolve_model_dir(folder_name):
    return model_dir(folder_name)


def resolve_train_dir(folder_name):
    return train_dir(folder_name)


def resolve_gifs_dir(folder_name):
    return gifs_dir(folder_name)


def checkpoint_path(folder_name):
    return resolve_model_dir(folder_name) / "checkpoint.pth"


def available_model_folders():
    if not MODEL_ROOT.exists():
        return []
    return sorted(path.name for path in MODEL_ROOT.iterdir() if path.is_dir())


def format_available_model_folders():
    folders = available_model_folders()
    if not folders:
        return "(none)"
    return ", ".join(folders)


def ensure_checkpoint_exists(folder_name, *, method_label=None):
    ckpt_path = checkpoint_path(folder_name)
    if ckpt_path.exists():
        return ckpt_path

    method_prefix = f"{method_label} " if method_label else ""
    raise FileNotFoundError(
        f"{method_prefix}checkpoint 不存在: {ckpt_path}\n"
        f"请求的模型文件夹: {folder_name}\n"
        f"当前可用的 model 文件夹: {format_available_model_folders()}\n"
        "请确认 checkpoint 已放到正确目录，或通过 --folder-name 显式指定。"
    )


def set_saver_paths(saver_params, folder_name):
    saver_params.FOLDER_NAME = folder_name
    saver_params.MODEL_PATH = str(model_dir(folder_name))
    saver_params.TRAIN_PATH = str(train_dir(folder_name))
    saver_params.GIFS_PATH = str(gifs_dir(folder_name))
