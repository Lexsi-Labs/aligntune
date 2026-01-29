"""
Configuration loader for unified SFT training.

This module provides utilities for loading and parsing SFT configurations
from YAML files and CLI arguments.
"""

import yaml
import logging
from typing import Dict, Any, Union
from pathlib import Path

from .config import SFTConfig, ModelConfig, DatasetConfig, TrainingConfig, LoggingConfig, EvaluationConfig, TaskType, PrecisionType

logger = logging.getLogger(__name__)


class SFTConfigLoader:
    """Loader for SFT configurations."""
    
    @staticmethod
    def load_from_yaml(path: Union[str, Path]) -> SFTConfig:
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = SFTConfigLoader._dict_to_config(data)
        SFTConfigLoader.validate_config(config)
        return config

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> SFTConfig:
        """Load configuration from a dictionary."""
        config = SFTConfigLoader._dict_to_config(data)
        SFTConfigLoader.validate_config(config)
        return config

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> SFTConfig:
        """Convert dictionary to SFTConfig object."""
        # Convert model config
        model_data = data.get("model", {})
        # Store backend separately since ModelConfig doesn't have it
        backend = model_data.get("backend", "auto")
        
        model_config = ModelConfig(
            name_or_path=model_data["name_or_path"],
            precision=PrecisionType(model_data.get("precision", "bf16")),
            quantization=model_data.get("quantization", {}),
            attn_implementation=model_data.get("attn_implementation", "auto"),
            gradient_checkpointing=model_data.get("gradient_checkpointing", True),
            max_memory=model_data.get("max_memory"),
            use_unsloth=model_data.get("use_unsloth", False),
            max_seq_length=model_data.get("max_seq_length", 2048),
            peft_enabled=model_data.get("peft_enabled", False),
            lora_rank=model_data.get("lora_rank", 16),
            lora_alpha=model_data.get("lora_alpha", 32),
            lora_dropout=model_data.get("lora_dropout", 0.1),
            target_modules=model_data.get("target_modules")
        )
        
        # Attach backend to model_config for trainer factory to access
        model_config.backend = backend
        
        # Convert dataset config
        dataset_data = data.get("dataset", {})
        dataset_config = DatasetConfig(
            name=dataset_data["name"],
            split=dataset_data.get("split", "train"),
            subset=dataset_data.get("subset"),
            config=dataset_data.get("config"),
            percent=dataset_data.get("percent"),
            max_samples=dataset_data.get("max_samples"),
            column_mapping=dataset_data.get("column_mapping", {}),
            task_type=TaskType(dataset_data.get("task_type", "supervised_fine_tuning"))
        )
        
        # Convert training config
        train_data = data.get("train", {})
        training_config = TrainingConfig(
            per_device_batch_size=train_data.get("per_device_batch_size", 1),
            gradient_accumulation_steps=train_data.get("gradient_accumulation_steps", 1),
            max_steps=train_data.get("max_steps"),
            epochs=train_data.get("epochs"),
            learning_rate=float(train_data.get("learning_rate", 1e-5)),
            weight_decay=float(train_data.get("weight_decay", 0.01)),
            warmup_steps=train_data.get("warmup_steps", 0),
            eval_interval=train_data.get("eval_interval", 100),
            save_interval=train_data.get("save_interval", 500),
            max_grad_norm=float(train_data.get("max_grad_norm", 1.0)),
            fp16=train_data.get("fp16", False),
            bf16=train_data.get("bf16", False),
            dataloader_num_workers=train_data.get("dataloader_num_workers", 0),
            remove_unused_columns=train_data.get("remove_unused_columns", False),
            dataset_num_proc=train_data.get("dataset_num_proc"),
            dataset_kwargs=train_data.get("dataset_kwargs", {}),
            packing=train_data.get("packing", False),
            packing_strategy=train_data.get("packing_strategy", "bfd"),
            eval_packing=train_data.get("eval_packing"),
            padding_free=train_data.get("padding_free", False),
            pad_to_multiple_of=train_data.get("pad_to_multiple_of"),
            completion_only_loss=train_data.get("completion_only_loss"),
            assistant_only_loss=train_data.get("assistant_only_loss", False),
            loss_type=train_data.get("loss_type", "nll"),
            activation_offloading=train_data.get("activation_offloading", False),
            use_flash_attention_2=train_data.get("use_flash_attention_2")
        )
        
        # Convert logging config
        logging_data = data.get("logging", {})
        logging_config = LoggingConfig(
            output_dir=logging_data.get("output_dir", "./output"),
            run_name=logging_data.get("run_name"),
            loggers=logging_data.get("loggers", ["tensorboard"]),
            log_level=logging_data.get("log_level", "INFO"),
            log_interval=logging_data.get("log_interval", 10),
            save_strategy=logging_data.get("save_strategy", "steps"),
            eval_strategy=logging_data.get("eval_strategy", "steps")
        )
        
        # Convert evaluation config
        eval_data = data.get("evaluation", {})
        evaluation_config = EvaluationConfig(
            compute_perplexity=eval_data.get("compute_perplexity", True),
            compute_rouge=eval_data.get("compute_rouge", True),
            compute_bleu=eval_data.get("compute_bleu", True),
            compute_meteor=eval_data.get("compute_meteor", False),
            compute_bertscore=eval_data.get("compute_bertscore", False),
            compute_semantic_similarity=eval_data.get("compute_semantic_similarity", False),
            compute_codebleu=eval_data.get("compute_codebleu", False),
            max_samples_for_quality_metrics=eval_data.get("max_samples_for_quality_metrics", 50),
            bertscore_model=eval_data.get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
            semantic_similarity_model=eval_data.get("semantic_similarity_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        # Store custom evaluation fields as attributes (not in EvaluationConfig dataclass)
        evaluation_config.enabled = eval_data.get("enabled", False)
        evaluation_config.pre_training = eval_data.get("pre_training", False)
        evaluation_config.post_training = eval_data.get("post_training", False)
        evaluation_config.eval_dataset = eval_data.get("eval_dataset", {})
        evaluation_config.metrics = eval_data.get("metrics", [])
        
        # Create main config
        return SFTConfig(
            model=model_config,
            dataset=dataset_config,
            train=training_config,
            logging=logging_config,
            evaluation=evaluation_config
        )
    
    @staticmethod
    def validate_config(config: SFTConfig) -> None:
        """Validate configuration."""
        # Validate model name
        if not config.model.name_or_path:
            raise ValueError("Model name_or_path is required and cannot be empty")
        
        # Validate dataset name
        if not config.dataset.name:
            raise ValueError("Dataset name is required and cannot be empty")
        
        # Validate training parameters
        if config.train.max_steps is None and config.train.epochs is None:
            raise ValueError("Either max_steps or epochs must be specified")
        
        logger.info("Configuration validation passed")
    
    @staticmethod
    def save_config(config: SFTConfig, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
