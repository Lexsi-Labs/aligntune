"""
Factory for creating SFT trainers.

This module provides a factory that delegates to Backend Factory for creating
SFT trainers. This maintains backward compatibility while using the primary
Backend Factory system.
"""

import logging
from typing import Any
from .config import SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainerFactory:
    """Factory for creating SFT trainers - delegates to Backend Factory."""
    
    @classmethod
    def create_trainer(cls, config: SFTConfig) -> Any:
        """
        Create a trainer by delegating to Backend Factory.
        
        Args:
            config: SFT configuration
            
        Returns:
            Trainer instance from Backend Factory
        """
        # Import here to avoid circular imports
        from ..backend_factory import create_sft_trainer
        
        # Extract backend from config, default to automatic selection
        backend = "auto"
        if hasattr(config, 'backend'):
            backend = config.backend
        elif hasattr(config.model, 'backend'):
            backend = config.model.backend
        
        # If use_unsloth is explicitly False, use TRL backend
        if hasattr(config.model, 'use_unsloth') and config.model.use_unsloth is False:
            backend = "trl"
        elif hasattr(config.model, 'use_unsloth') and config.model.use_unsloth is True:
            backend = "unsloth"
        
        backend_value = backend.value if hasattr(backend, 'value') else backend
        
        # Delegate to Backend Factory
        # Extract optional parameters with defaults
        logging_steps = getattr(config.train, 'logging_steps', getattr(config.logging, 'log_interval', 10))
        save_steps = getattr(config.train, 'save_steps', config.train.save_interval)
        max_steps = getattr(config.train, 'max_steps', None)
        
        return create_sft_trainer(
            model_name=config.model.name_or_path,
            dataset_name=config.dataset.name,
            backend=backend_value,
            output_dir=config.logging.output_dir,
            num_epochs=config.train.epochs,
            batch_size=config.train.per_device_batch_size,
            learning_rate=config.train.learning_rate,
            max_seq_length=config.model.max_seq_length,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            warmup_steps=config.train.warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            max_steps=max_steps,
            task_type=config.dataset.task_type.value if hasattr(config.dataset.task_type, 'value') else str(config.dataset.task_type),
            column_mapping=config.dataset.column_mapping,
            max_samples=config.dataset.max_samples,
            use_peft=config.model.peft_enabled,
            lora_r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            lora_target_modules=getattr(config.model, 'target_modules', None),
            bf16=config.train.bf16,
            gradient_checkpointing=config.model.gradient_checkpointing,
            # Pass seed
            seed=config.train.seed,
            data_seed=config.train.data_seed,
        )
