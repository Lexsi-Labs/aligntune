"""
Config translator for veRL - maps AlignTune dataclass → OmegaConf.

This module translates AlignTune's unified_config to veRL's Hydra-based OmegaConf format.
"""

import logging
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

from aligntune.core.rl.config import UnifiedConfig

logger = logging.getLogger(__name__)


def _build_verl_config_dict(unified_config: UnifiedConfig) -> Dict[str, Any]:
    """
    Build plain dict from AlignTune UnifiedConfig following veRL schema.

    Handles: model path, data path, training args, reward config, FSDP config, ray config.

    Args:
        unified_config: AlignTune UnifiedConfig dataclass

    Returns:
        Dict with veRL-compatible configuration
    """
    config_dict = {}

    # Model Configuration
    if hasattr(unified_config, 'model') and unified_config.model:
        model_cfg = unified_config.model
        config_dict['model'] = {
            'type': 'default',
            'model_path': getattr(model_cfg, 'name_or_path', 'gpt2'),
        }

        if hasattr(model_cfg, 'trust_remote_code'):
            config_dict['model']['trust_remote_code'] = model_cfg.trust_remote_code
        if hasattr(model_cfg, 'use_peft') and model_cfg.use_peft:
            config_dict['model']['use_peft'] = True

    # Data Configuration
    if hasattr(unified_config, 'dataset') and unified_config.dataset:
        dataset_cfg = unified_config.dataset
        config_dict['data'] = {
            'dataset_name': dataset_cfg.name,
            'split': getattr(dataset_cfg, 'split', 'train'),
        }

        if hasattr(dataset_cfg, 'max_samples') and dataset_cfg.max_samples:
            config_dict['data']['max_samples'] = dataset_cfg.max_samples

    # Training Configuration
    if hasattr(unified_config, 'train') and unified_config.train:
        train_cfg = unified_config.train
        config_dict['train'] = {
            'output_dir': train_cfg.output_dir,
            'num_episodes': getattr(train_cfg, 'epochs', 3),
            'batch_size': getattr(train_cfg, 'per_device_batch_size', 32),
            'micro_batch_size': getattr(train_cfg, 'verl_micro_batch_size', max(1, getattr(train_cfg, 'per_device_batch_size', 32) // 2)),
            'learning_rate': getattr(train_cfg, 'learning_rate', 2e-4),
            'max_steps': getattr(train_cfg, 'max_steps', None),
            'gradient_accumulation_steps': getattr(train_cfg, 'gradient_accumulation_steps', 1),
            'warmup_ratio': getattr(train_cfg, 'warmup_ratio', 0.0),
            'weight_decay': getattr(train_cfg, 'weight_decay', 0.0),
            'seed': getattr(train_cfg, 'seed', 42),
            'save_interval': getattr(train_cfg, 'save_interval', 100),
            'eval_interval': getattr(train_cfg, 'eval_interval', 100),
        }

        if hasattr(train_cfg, 'max_seq_length'):
            config_dict['train']['max_seq_length'] = train_cfg.max_seq_length
        if hasattr(train_cfg, 'bf16') and train_cfg.bf16:
            config_dict['train']['dtype'] = 'bf16'
        else:
            config_dict['train']['dtype'] = 'fp32'

    # PPO/GRPO Specific Configuration
    if hasattr(unified_config, 'algo'):
        algo = unified_config.algo
        algo_name = getattr(algo, 'value', 'ppo') if hasattr(algo, 'value') else 'ppo'

        if algo_name in ['ppo', 'grpo']:
            config_dict['algorithm'] = {
                'type': algo_name,
            }

            if hasattr(unified_config, 'train'):
                train_cfg = unified_config.train
                ppo_config = {
                    'kl_coef': getattr(train_cfg, 'kl_coef', 0.05),
                    'gamma': getattr(train_cfg, 'gamma', 0.99),
                    'gae_lambda': getattr(train_cfg, 'lam', 0.95),
                    'cliprange': getattr(train_cfg, 'cliprange', 0.2),
                    'vf_coef': getattr(train_cfg, 'vf_coef', 0.1),
                }

                if algo_name == 'grpo':
                    ppo_config['adv_estimator'] = 'grpo'
                    ppo_config['num_rollouts'] = getattr(train_cfg, 'verl_rollout_n', 4)

                config_dict['algorithm'].update(ppo_config)

    # Reward Configuration
    if hasattr(unified_config, 'reward') and unified_config.reward:
        reward_cfg = unified_config.reward
        config_dict['reward'] = {
            'type': getattr(reward_cfg, 'type', 'composite'),
        }

        if hasattr(reward_cfg, 'model_name'):
            config_dict['reward']['model_name'] = reward_cfg.model_name

    # Distributed/FSDP Configuration
    if hasattr(unified_config, 'train'):
        train_cfg = unified_config.train

        n_gpus_per_node = getattr(train_cfg, 'verl_n_gpus_per_node', 8)
        nnodes = getattr(train_cfg, 'verl_nnodes', 1)

        config_dict['ray'] = {
            'num_gpus_per_node': n_gpus_per_node,
            'nnodes': nnodes,
            'object_store_memory': 64 * 1024 * 1024 * 1024,
        }

        fsdp_config = getattr(train_cfg, 'verl_fsdp_config', {})
        if fsdp_config:
            config_dict['fsdp'] = fsdp_config

    return config_dict


def translate_to_verl_config(unified_config: UnifiedConfig) -> 'OmegaConf':
    """
    Translate AlignTune UnifiedConfig to veRL OmegaConf format.

    Args:
        unified_config: AlignTune UnifiedConfig dataclass

    Returns:
        OmegaConf configuration object for veRL
    """
    config_dict = _build_verl_config_dict(unified_config)

    verl_config = OmegaConf.create(config_dict)

    logger.info(f"Translated UnifiedConfig to veRL OmegaConf with algorithm: "
                f"{config_dict.get('algorithm', {}).get('type', 'unknown')}")

    return verl_config
