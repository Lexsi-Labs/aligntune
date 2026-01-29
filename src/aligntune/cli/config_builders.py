"""
CLI Configuration Builders for AlignTune.


This module provides configuration builders for different training types,
extracted from the main.py file for better organization.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.backend_factory import (
    BackendType,
    RLAlgorithm,
    create_sft_trainer,
    create_rl_trainer,
)
from ..core.rl.config import UnifiedConfig
from ..core.sft.config import SFTConfig

logger = logging.getLogger(__name__)


def build_sft_config(
    model_name: str,
    dataset_name: str,
    backend: BackendType = BackendType.UNSLOTH,
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    eval_steps: int = 500,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: Optional[str] = None,
    **kwargs
) -> SFTConfig:
    """Build SFT configuration from arguments."""
    
    try:
        # Create SFT configuration
        config = SFTConfig(
            model=SFTConfig.ModelConfig(
                name_or_path=model_name,
                max_seq_length=max_seq_length,
                quantization={
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                }
            ),
            dataset=SFTConfig.DatasetConfig(
                name=dataset_name,
                split="train",
                max_samples=max_samples,
                auto_detect_fields=True,
            ),
            training=SFTConfig.TrainingConfig(
                num_epochs=num_epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                save_steps=save_steps,
                eval_steps=eval_steps,
            ),
            logging=SFTConfig.LoggingConfig(
                output_dir=output_dir,
                wandb_project=wandb_project,
            ),
        )
        
        logger.info(f"Built SFT configuration for model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Backend: {backend.value}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to build SFT configuration: {e}")
        raise


def build_dpo_config(
    model_name: str,
    dataset_name: str,
    backend: BackendType = BackendType.UNSLOTH,
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    eval_steps: int = 500,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: Optional[str] = None,
    beta: float = 0.1,
    **kwargs
) -> UnifiedConfig:
    """Build DPO configuration from arguments."""
    
    try:
        # Create unified configuration for DPO
        config = UnifiedConfig(
            model=UnifiedConfig.ModelConfig(
                name_or_path=model_name,
                max_seq_length=max_seq_length,
                quantization={
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                }
            ),
            datasets=[
                UnifiedConfig.DatasetConfig(
                    name=dataset_name,
                    split="train",
                    max_samples=max_samples,
                    auto_detect_fields=True,
                )
            ],
            training=UnifiedConfig.TrainingConfig(
                num_epochs=num_epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                save_steps=save_steps,
                eval_steps=eval_steps,
            ),
            logging=UnifiedConfig.LoggingConfig(
                output_dir=output_dir,
                wandb_project=wandb_project,
            ),
            reward=UnifiedConfig.RewardConfig(
                beta=beta,
                functions=[
                    {"type": "length", "weight": 0.2},
                    {"type": "coherence", "weight": 0.3},
                    {"type": "sentiment", "weight": 0.2},
                    {"type": "safety", "weight": 0.3},
                ]
            ),
        )
        
        logger.info(f"Built DPO configuration for model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Backend: {backend.value}")
        logger.info(f"Beta: {beta}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to build DPO configuration: {e}")
        raise


def build_ppo_config(
    model_name: str,
    dataset_name: str,
    backend: BackendType = BackendType.UNSLOTH,
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    eval_steps: int = 500,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: Optional[str] = None,
    ppo_epochs: int = 4,
    kl_coef: float = 0.1,
    vf_coef: float = 0.1,
    cliprange: float = 0.2,
    **kwargs
) -> UnifiedConfig:
    """Build PPO configuration from arguments."""
    
    try:
        # Create unified configuration for PPO
        config = UnifiedConfig(
            model=UnifiedConfig.ModelConfig(
                name_or_path=model_name,
                max_seq_length=max_seq_length,
                quantization={
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                }
            ),
            datasets=[
                UnifiedConfig.DatasetConfig(
                    name=dataset_name,
                    split="train",
                    max_samples=max_samples,
                    auto_detect_fields=True,
                )
            ],
            training=UnifiedConfig.TrainingConfig(
                num_epochs=num_epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                save_steps=save_steps,
                eval_steps=eval_steps,
            ),
            logging=UnifiedConfig.LoggingConfig(
                output_dir=output_dir,
                wandb_project=wandb_project,
            ),
            reward=UnifiedConfig.RewardConfig(
                functions=[
                    {"type": "length", "weight": 0.2},
                    {"type": "coherence", "weight": 0.3},
                    {"type": "sentiment", "weight": 0.2},
                    {"type": "safety", "weight": 0.3},
                ]
            ),
        )
        
        logger.info(f"Built PPO configuration for model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Backend: {backend.value}")
        logger.info(f"PPO epochs: {ppo_epochs}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to build PPO configuration: {e}")
        raise


def build_grpo_config(
    model_name: str,
    dataset_name: str,
    backend: BackendType = BackendType.UNSLOTH,
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    eval_steps: int = 500,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: Optional[str] = None,
    grpo_epochs: int = 4,
    kl_coef: float = 0.1,
    vf_coef: float = 0.1,
    cliprange: float = 0.2,
    **kwargs
) -> UnifiedConfig:
    """Build GRPO configuration from arguments."""
    
    try:
        # Create unified configuration for GRPO
        config = UnifiedConfig(
            model=UnifiedConfig.ModelConfig(
                name_or_path=model_name,
                max_seq_length=max_seq_length,
                quantization={
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                }
            ),
            datasets=[
                UnifiedConfig.DatasetConfig(
                    name=dataset_name,
                    split="train",
                    max_samples=max_samples,
                    auto_detect_fields=True,
                    weight=1.0,
                )
            ],
            training=UnifiedConfig.TrainingConfig(
                num_epochs=num_epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                save_steps=save_steps,
                eval_steps=eval_steps,
            ),
            logging=UnifiedConfig.LoggingConfig(
                output_dir=output_dir,
                wandb_project=wandb_project,
            ),
            reward=UnifiedConfig.RewardConfig(
                functions=[
                    {"type": "length", "weight": 0.2},
                    {"type": "coherence", "weight": 0.3},
                    {"type": "sentiment", "weight": 0.2},
                    {"type": "safety", "weight": 0.3},
                ]
            ),
        )
        
        logger.info(f"Built GRPO configuration for model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Backend: {backend.value}")
        logger.info(f"GRPO epochs: {grpo_epochs}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to build GRPO configuration: {e}")
        raise


def build_gspo_config(
    model_name: str,
    dataset_name: str,
    backend: BackendType = BackendType.UNSLOTH,
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    save_steps: int = 500,
    eval_steps: int = 500,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: Optional[str] = None,
    gspo_epochs: int = 4,
    kl_coef: float = 0.1,
    vf_coef: float = 0.1,
    cliprange: float = 0.2,
    **kwargs
) -> UnifiedConfig:
    """Build GSPO configuration from arguments."""
    
    try:
        # Create unified configuration for GSPO
        config = UnifiedConfig(
            model=UnifiedConfig.ModelConfig(
                name_or_path=model_name,
                max_seq_length=max_seq_length,
                quantization={
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                }
            ),
            datasets=[
                UnifiedConfig.DatasetConfig(
                    name=dataset_name,
                    split="train",
                    max_samples=max_samples,
                    auto_detect_fields=True,
                )
            ],
            training=UnifiedConfig.TrainingConfig(
                num_epochs=num_epochs,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                save_steps=save_steps,
                eval_steps=eval_steps,
            ),
            logging=UnifiedConfig.LoggingConfig(
                output_dir=output_dir,
                wandb_project=wandb_project,
            ),
            reward=UnifiedConfig.RewardConfig(
                functions=[
                    {"type": "length", "weight": 0.2},
                    {"type": "coherence", "weight": 0.3},
                    {"type": "sentiment", "weight": 0.2},
                    {"type": "safety", "weight": 0.3},
                ]
            ),
        )
        
        logger.info(f"Built GSPO configuration for model: {model_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Backend: {backend.value}")
        logger.info(f"GSPO epochs: {gspo_epochs}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to build GSPO configuration: {e}")
        raise


def create_trainer_from_config(
    training_type: str,
    config: Any,
    backend: BackendType = BackendType.UNSLOTH,
) -> Any:
    """Create trainer from configuration based on training type."""
    
    try:
        if training_type.lower() == "sft":
            trainer = create_sft_trainer(
                model_name=config.model.name_or_path,
                dataset_name=config.datasets[0].name if hasattr(config, 'datasets') else config.dataset.name,
                backend=backend,
                output_dir=config.logging.output_dir,
                num_epochs=config.training.num_epochs,
                batch_size=config.training.per_device_batch_size,
                learning_rate=config.training.learning_rate,
                max_seq_length=config.model.max_seq_length,
                max_samples=config.datasets[0].max_samples if hasattr(config, 'datasets') else config.dataset.max_samples,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                warmup_ratio=config.training.warmup_ratio,
                save_steps=config.training.save_steps,
                eval_steps=config.training.eval_steps,
                load_in_4bit=config.model.quantization.get("load_in_4bit", False),
                load_in_8bit=config.model.quantization.get("load_in_8bit", False),
                wandb_project=config.logging.wandb_project,
            )
        else:
            # RL training
            algorithm_map = {
                "dpo": RLAlgorithm.DPO,
                "ppo": RLAlgorithm.PPO,
                "grpo": RLAlgorithm.GRPO,
                "gspo": RLAlgorithm.GSPO,
            }
            
            algorithm = algorithm_map.get(training_type.lower())
            if not algorithm:
                raise ValueError(f"Invalid RL algorithm: {training_type}")
            
            trainer = create_rl_trainer(
                model_name=config.model.name_or_path,
                dataset_name=config.datasets[0].name,
                algorithm=algorithm,
                backend=backend,
                output_dir=config.logging.output_dir,
                num_epochs=config.training.num_epochs,
                batch_size=config.training.per_device_batch_size,
                learning_rate=config.training.learning_rate,
                max_seq_length=config.model.max_seq_length,
                max_samples=config.datasets[0].max_samples,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                warmup_ratio=config.training.warmup_ratio,
                save_steps=config.training.save_steps,
                eval_steps=config.training.eval_steps,
                load_in_4bit=config.model.quantization.get("load_in_4bit", False),
                load_in_8bit=config.model.quantization.get("load_in_8bit", False),
                wandb_project=config.logging.wandb_project,
            )
        
        logger.info(f"Created {training_type.upper()} trainer with {backend.value} backend")
        return trainer
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        raise
