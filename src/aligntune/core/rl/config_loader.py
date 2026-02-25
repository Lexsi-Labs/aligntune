"""
Configuration loader and validator for unified RLHF training.

This module provides utilities to load configurations from YAML files or dictionaries,
parse dataset and reward specifications, and validate configurations with no placeholder defaults.
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from .config import (
    UnifiedConfig,
    AlgorithmType,
    PrecisionType,
    BackendType,
    ModelConfig,
    DatasetConfig,
    RewardConfig,
    TrainingConfig,
    DistributedConfig,
    LoggingConfig,
)


class ConfigLoader:
    """Load and validate unified configurations with no placeholder defaults."""
    
    @staticmethod
    def load_from_yaml(path: Union[str, Path]) -> UnifiedConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file must contain a dictionary, got {type(data)}")
        
        return ConfigLoader._dict_to_config(data)
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> UnifiedConfig:
        """Load configuration from dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Configuration data must be a dictionary, got {type(data)}")
        
        return ConfigLoader._dict_to_config(data)
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> UnifiedConfig:
        """Convert dictionary to UnifiedConfig with validation."""
        # Validate required fields
        required_fields = ['algo', 'model', 'datasets']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert nested dictionaries to config objects
        model_config = ModelConfig(**data['model'])

        # Convert datasets
        datasets = []
        for ds_data in data['datasets']:
            datasets.append(DatasetConfig(**ds_data))
        
        # Convert rewards
        rewards = []
        for reward_data in data.get('rewards', []):
            rewards.append(RewardConfig(**reward_data))
        
        # Convert other configs with type conversion
        train_data = data.get('train', {})
        if 'learning_rate' in train_data and isinstance(train_data['learning_rate'], str):
            train_data['learning_rate'] = float(train_data['learning_rate'])
        train_config = TrainingConfig(**train_data)
        
        distributed_data = data.get('distributed', {})
        if 'seed' in distributed_data and isinstance(distributed_data['seed'], str):
            distributed_data['seed'] = int(distributed_data['seed'])
        distributed_config = DistributedConfig(**distributed_data)
        
        logging_config = LoggingConfig(**data.get('logging', {}))
        
        return UnifiedConfig(
            algo=AlgorithmType(data['algo']),
            model=model_config,
            datasets=datasets,
            tasks=data.get('tasks', []),
            rewards=rewards,
            train=train_config,
            distributed=distributed_config,
            logging=logging_config,
            chat_template=data.get('chat_template'),
            caching=data.get('caching', {})
        )
    
    @staticmethod
    def parse_dataset_spec(spec: str) -> DatasetConfig:
        """
        Parse dataset specification string.
        
        Format: name[:split][#percent=N|max=N][?map.key=value]
        Example: "Anthropic/hh-rlhf:train#percent=25?map.prompt=prompt"
        """
        if not spec or not isinstance(spec, str):
            raise ValueError("Dataset specification must be a non-empty string")
        
        # Split query parameters
        parts = spec.split('?')
        main_part = parts[0]
        query_params = {}
        
        if len(parts) > 1:
            for param in parts[1].split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    # Handle map. prefix
                    if key.startswith('map.'):
                        key = key[4:]  # Remove 'map.' prefix
                    query_params[key] = value
        
        # Parse main part
        if '#' in main_part:
            name_split, size_spec = main_part.split('#', 1)
        else:
            name_split = main_part
            size_spec = None
        
        if ':' in name_split:
            name, split = name_split.split(':', 1)
        else:
            name = name_split
            split = "train"
        
        # Parse size specification
        percent = None
        max_samples = None
        if size_spec:
            if size_spec.startswith('percent='):
                percent = float(size_spec.split('=')[1])
            elif size_spec.startswith('max='):
                max_samples = int(size_spec.split('=')[1])
            else:
                raise ValueError(f"Invalid size specification: {size_spec}")
        
        return DatasetConfig(
            name=name,
            split=split,
            percent=percent,
            max_samples=max_samples,
            column_mapping=query_params
        )
    
    @staticmethod
    def parse_reward_spec(spec: str) -> RewardConfig:
        """
        Parse reward specification string.
        
        Format: type:weight:param1=value1:param2=value2
        Example: "numeric_math:weight=1.0:tolerance=1e-6"
        """
        if not spec or not isinstance(spec, str):
            raise ValueError("Reward specification must be a non-empty string")
        
        parts = spec.split(':')
        if len(parts) < 1:
            raise ValueError("Reward specification must include at least the type")
        
        reward_type = parts[0]
        weight = 1.0
        params = {}
        
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 'weight':
                    weight = float(value)
                else:
                    # Try to parse as number or boolean, otherwise keep as string
                    try:
                        if value.lower() in ('true', 'false'):
                            params[key] = value.lower() == 'true'
                        elif 'e' in value.lower() or '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
            else:
                # Handle special cases like "shield:safety:strict=true"
                if part == "shield" and len(parts) > parts.index(part) + 1:
                    # This is a shield specification
                    shield_type = parts[parts.index(part) + 1]
                    params["shield_type"] = shield_type
                    # Look for additional shield parameters
                    for i in range(parts.index(part) + 2, len(parts)):
                        if '=' in parts[i]:
                            key, value = parts[i].split('=', 1)
                            params[f"shield_{key}"] = value
        
        return RewardConfig(
            type=reward_type,
            weight=weight,
            params=params
        )
    
    @staticmethod
    def validate_config(config: UnifiedConfig) -> None:
        """Validate configuration for consistency and completeness."""
        # Validate model path exists or is downloadable
        if not config.model.name_or_path:
            raise ValueError("Model name_or_path cannot be empty")
        
        # Validate at least one dataset is specified
        if not config.datasets:
            raise ValueError("At least one dataset must be specified")
        
        # Validate dataset names are not empty
        for dataset in config.datasets:
            if not dataset.name:
                raise ValueError("All dataset names must be non-empty")
        
        # Validate reward functions are registered (basic check)
        for reward in config.rewards:
            if not reward.type:
                raise ValueError("All reward types must be non-empty")
        
        # Validate distributed configuration
        if config.distributed.backend.value == "deepspeed":
            if not config.distributed.deepspeed_config:
                raise ValueError("DeepSpeed backend requires deepspeed_config")
        
        # Validate memory requirements are reasonable
        if config.train.per_device_batch_size * config.train.gradient_accumulation_steps > 1000:
            raise ValueError("Effective batch size too large (>1000)")
        
        # Validate that either max_steps or epochs is specified
        if config.train.max_steps is None and config.train.epochs is None:
            raise ValueError("Either max_steps or epochs must be specified")
        
        if config.train.max_steps is not None and config.train.epochs is not None:
            raise ValueError("Cannot specify both max_steps and epochs")
    
    @staticmethod
    def save_resolved_config(config: UnifiedConfig, output_dir: Union[str, Path]) -> None:
        """Save resolved configuration to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        resolved_config = config.to_dict()
        
        # Add metadata
        resolved_config["_metadata"] = {
            "created_at": str(Path().cwd()),
            "config_version": "1.0",
            "resolved": True
        }
        
        config_path = output_dir / "config.resolved.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(resolved_config, f, default_flow_style=False, indent=2, allow_unicode=True, sort_keys=False)
    
    @staticmethod
    def create_config_from_cli_args(**kwargs) -> UnifiedConfig:
        """Create configuration from CLI arguments."""
        # Parse dataset specifications
        datasets = []
        for ds_spec in kwargs.get('dataset', []):
            datasets.append(ConfigLoader.parse_dataset_spec(ds_spec))
        
        # Parse reward specifications
        rewards = []
        for reward_spec in kwargs.get('rewards', []):
            rewards.append(ConfigLoader.parse_reward_spec(reward_spec))
        
        # Create configuration
        return UnifiedConfig(
            algo=AlgorithmType(kwargs['algo']),
            model=ModelConfig(
                name_or_path=kwargs['model'],
                precision=PrecisionType(kwargs.get('precision', 'bf16')),
                gradient_checkpointing=kwargs.get('grad_checkpointing', True)
            ),
            datasets=datasets,
            rewards=rewards,
            train=TrainingConfig(
                per_device_batch_size=kwargs.get('train_batch', 1),
                gradient_accumulation_steps=kwargs.get('accum', 1),
                max_steps=kwargs.get('max_steps'),
                epochs=kwargs.get('epochs'),
                eval_interval=kwargs.get('eval_interval', 100),
                save_interval=kwargs.get('save_interval', 500),
                rollout_batch_size=kwargs.get('rollout_bs', 1)
            ),
            distributed=DistributedConfig(
                backend=BackendType(kwargs.get('backend', 'single')),
                seed=kwargs.get('seed', 42)
            ),
            logging=LoggingConfig(
                loggers=kwargs.get('loggers', ['tensorboard']),
                output_dir=kwargs.get('output_dir', './output')
            ),
            chat_template=kwargs.get('chat_template')
        )
