#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块消融配置：
- 双交叉注意力 -> 单一共享自注意力
- globalDecoder -> MLP
"""

import copy
import json

from parameters import TrainParams
from project_paths import resolve_train_dir


DEFAULT_MODULE_ABLATION_CONFIG = {
    'preset': 'baseline',
    'cross_attention_mode': 'dual_cross',
    'global_decoder_mode': 'attention',
}


MODULE_ABLATION_PRESETS = {
    'baseline': {
        'preset': 'baseline',
        'cross_attention_mode': 'dual_cross',
        'global_decoder_mode': 'attention',
    },
    'single_self_attention': {
        'preset': 'single_self_attention',
        'cross_attention_mode': 'shared_self',
        'global_decoder_mode': 'attention',
    },
    'global_mlp': {
        'preset': 'global_mlp',
        'cross_attention_mode': 'dual_cross',
        'global_decoder_mode': 'mlp',
    },
    'single_self_attention_global_mlp': {
        'preset': 'single_self_attention_global_mlp',
        'cross_attention_mode': 'shared_self',
        'global_decoder_mode': 'mlp',
    },
}


def list_module_ablation_presets():
    return sorted(MODULE_ABLATION_PRESETS.keys())


def normalize_module_ablation_config(config=None):
    normalized = copy.deepcopy(DEFAULT_MODULE_ABLATION_CONFIG)
    if config:
        normalized.update(config)
    normalized['cross_attention_mode'] = str(normalized['cross_attention_mode']).lower()
    normalized['global_decoder_mode'] = str(normalized['global_decoder_mode']).lower()
    if normalized['cross_attention_mode'] not in {'dual_cross', 'shared_self'}:
        raise ValueError(f"Unsupported cross_attention_mode: {normalized['cross_attention_mode']}")
    if normalized['global_decoder_mode'] not in {'attention', 'mlp'}:
        raise ValueError(f"Unsupported global_decoder_mode: {normalized['global_decoder_mode']}")
    return normalized


def build_module_ablation_config(preset):
    if preset not in MODULE_ABLATION_PRESETS:
        raise KeyError(f'Unknown module ablation preset: {preset}')
    return normalize_module_ablation_config(MODULE_ABLATION_PRESETS[preset])


def module_ablation_config_to_json(config):
    return json.dumps(normalize_module_ablation_config(config), indent=2, ensure_ascii=False)


def apply_module_ablation_config(config):
    normalized = normalize_module_ablation_config(config)
    TrainParams.CROSS_ATTENTION_MODE = normalized['cross_attention_mode']
    TrainParams.GLOBAL_DECODER_MODE = normalized['global_decoder_mode']
    return normalized


def export_current_module_ablation_config():
    return normalize_module_ablation_config({
        'cross_attention_mode': getattr(TrainParams, 'CROSS_ATTENTION_MODE', 'dual_cross'),
        'global_decoder_mode': getattr(TrainParams, 'GLOBAL_DECODER_MODE', 'attention'),
    })


def load_module_ablation_config(folder_name, default_preset='baseline'):
    config_path = resolve_train_dir(folder_name) / 'module_ablation_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return normalize_module_ablation_config(json.load(f))
    return build_module_ablation_config(default_preset)
