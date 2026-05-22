import copy
import json


DEFAULT_REWARD_CONFIG = {
    'preset': 'baseline',
    'description': 'Full reward with shared term, local contribution, and revealed-set conditioning.',
    'enable_shared_term': True,
    'enable_local_contribution': True,
    'use_revealed_task_set': True,
    'local_weight': 12.0,
    'phi_beta': 0.5,
    'zero_marginal_epsilon': 1e-8,
    'positive_credit_epsilon': 1e-8,
}


REWARD_ABLATION_PRESETS = {
    'baseline': {
        'description': 'Full reward with shared term, local contribution, and revealed-set conditioning.',
        'enable_shared_term': True,
        'enable_local_contribution': True,
        'use_revealed_task_set': True,
    },
    'no_shared': {
        'description': 'Remove the shared/global reward term and keep only local contribution.',
        'enable_shared_term': False,
        'enable_local_contribution': True,
        'use_revealed_task_set': True,
    },
    'no_local': {
        'description': 'Remove the local contribution term and keep only the shared/global reward.',
        'enable_shared_term': True,
        'enable_local_contribution': False,
        'use_revealed_task_set': True,
    },
    'static_full_penalty': {
        'description': 'Replace revealed-set conditioning with static full-set penalties in dynamic scenes.',
        'enable_shared_term': True,
        'enable_local_contribution': True,
        'use_revealed_task_set': False,
    },
}


def list_reward_presets():
    return sorted(REWARD_ABLATION_PRESETS.keys())


def build_reward_config(preset='baseline', overrides=None):
    if preset not in REWARD_ABLATION_PRESETS:
        raise ValueError(
            f"Unknown reward preset '{preset}'. Expected one of: {', '.join(list_reward_presets())}"
        )
    config = copy.deepcopy(DEFAULT_REWARD_CONFIG)
    config.update(copy.deepcopy(REWARD_ABLATION_PRESETS[preset]))
    config['preset'] = preset
    if overrides:
        config.update(copy.deepcopy(overrides))
    return config


def normalize_reward_config(config=None):
    if config is None:
        return build_reward_config('baseline')
    preset = config.get('preset', 'baseline')
    normalized = build_reward_config(preset)
    normalized.update(copy.deepcopy(config))
    normalized['preset'] = preset
    return normalized


def reward_config_to_json(config):
    return json.dumps(normalize_reward_config(config), indent=2, sort_keys=True)
