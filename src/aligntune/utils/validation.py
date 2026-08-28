"""
Configuration validation utilities for AlignTune.


This module provides comprehensive validation for training configurations,
including memory estimates, compatibility checks, and dataset validation.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import is_dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Import config classes
try:
    from ..core.sft.config import SFTConfig
    from ..core.rl.config import UnifiedConfig
except ImportError:
    # Handle circular import issues
    SFTConfig = None
    UnifiedConfig = None


class ConfigValidator:
    """Comprehensive configuration validator."""

    @staticmethod
    def validate_sft_config(config) -> List[str]:
        """
        Validate SFT configuration.

        Args:
            config: SFTConfig object

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Basic field validation
        if not config.model.name_or_path:
            errors.append("Model name_or_path is required")

        if not config.dataset.name:
            errors.append("Dataset name is required")

        if config.train.max_steps is None and config.train.epochs is None:
            errors.append("Either max_steps or epochs must be specified")

        # Model-specific validations
        errors.extend(ConfigValidator._validate_model_config(config.model))

        # Dataset-specific validations
        errors.extend(ConfigValidator._validate_dataset_config(config.dataset))

        # Training-specific validations
        errors.extend(ConfigValidator._validate_training_config(config.train))

        # Cross-field validations
        errors.extend(ConfigValidator._validate_sft_cross_fields(config))

        return errors

    @staticmethod
    def validate_rl_config(config) -> List[str]:
        """
        Validate RL configuration.

        Args:
            config: UnifiedConfig object

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Basic field validation
        if not config.model.name_or_path:
            errors.append("Model name_or_path is required")

        if not config.datasets:
            errors.append("At least one dataset must be specified")

        for i, dataset in enumerate(config.datasets):
            if not dataset.name:
                errors.append(f"Dataset {i}: name is required")

        # Algorithm-specific validations
        if config.algo.value == "dpo":
            errors.extend(ConfigValidator._validate_dpo_config(config))
        elif config.algo.value == "ppo":
            errors.extend(ConfigValidator._validate_ppo_config(config))
        elif config.algo.value == "grpo":
            errors.extend(ConfigValidator._validate_grpo_config(config))

        # Model-specific validations
        errors.extend(ConfigValidator._validate_rl_model_config(config.model))

        # Training-specific validations
        errors.extend(ConfigValidator._validate_rl_training_config(config.train))

        return errors

    @staticmethod
    def _validate_model_config(model_config) -> List[str]:
        """Validate model configuration."""
        errors = []

        # Precision validation
        valid_precisions = {"fp32", "fp16", "bf16"}
        if hasattr(model_config, 'precision') and model_config.precision not in valid_precisions:
            errors.append(f"Invalid precision: {model_config.precision}. Must be one of {valid_precisions}")

        # LoRA validation
        if hasattr(model_config, 'peft_enabled') and model_config.peft_enabled:
            if hasattr(model_config, 'lora_rank') and model_config.lora_rank <= 0:
                errors.append("LoRA rank must be positive")
            if hasattr(model_config, 'lora_alpha') and model_config.lora_alpha <= 0:
                errors.append("LoRA alpha must be positive")

        return errors

    @staticmethod
    def _validate_dataset_config(dataset_config) -> List[str]:
        """Validate dataset configuration."""
        errors = []

        # Dataset size validation
        if hasattr(dataset_config, 'percent') and dataset_config.percent is not None:
            if not (0 < dataset_config.percent <= 100):
                errors.append(f"Dataset percent must be between 0 and 100, got {dataset_config.percent}")

        if hasattr(dataset_config, 'max_samples') and dataset_config.max_samples is not None:
            if dataset_config.max_samples <= 0:
                errors.append("max_samples must be positive")

        # Dataset processing validation
        if hasattr(dataset_config, 'dataset_num_proc') and dataset_config.dataset_num_proc is not None:
            if dataset_config.dataset_num_proc <= 0:
                errors.append("dataset_num_proc must be positive")

        return errors

    @staticmethod
    def _validate_training_config(train_config) -> List[str]:
        """Validate training configuration."""
        errors = []

        # Batch size validation
        if train_config.per_device_batch_size <= 0:
            errors.append("per_device_batch_size must be positive")

        if train_config.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")

        # Learning rate validation
        if train_config.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        # Optimizer/scheduler validation
        if hasattr(train_config, 'optimizer'):
            from ..core.optimization import validate_optimizer_availability
            if not validate_optimizer_availability(train_config.optimizer):
                errors.append(f"Optimizer '{train_config.optimizer}' is not available")

        if hasattr(train_config, 'lr_scheduler'):
            from ..core.optimization import validate_scheduler_availability
            if not validate_scheduler_availability(train_config.lr_scheduler):
                errors.append(f"Scheduler '{train_config.lr_scheduler}' is not available")

        return errors

    @staticmethod
    def _validate_sft_cross_fields(config) -> List[str]:
        """Validate cross-field relationships for SFT."""
        errors = []

        # Task-specific validations
        task_type = config.dataset.task_type.value if hasattr(config.dataset.task_type, 'value') else config.dataset.task_type

        if task_type == "text_classification":
            if not hasattr(config.model, 'num_labels') or config.model.num_labels is None:
                errors.append("num_labels is required for text classification")
            elif config.model.num_labels <= 1:
                errors.append("num_labels must be > 1 for classification")

        return errors

    @staticmethod
    def _validate_rl_model_config(model_config) -> List[str]:
        """Validate RL model configuration."""
        errors = []

        # Reward model validation for PPO
        if hasattr(model_config, 'reward_value_model'):
            if not model_config.reward_value_model:
                errors.append("reward_value_model is required for RL training")

        return errors

    @staticmethod
    def _validate_rl_training_config(train_config) -> List[str]:
        """Validate RL training configuration."""
        errors = []

        # PPO-specific validations
        if hasattr(train_config, 'kl_coef') and train_config.kl_coef < 0:
            errors.append("kl_coef must be non-negative")

        if hasattr(train_config, 'cliprange') and train_config.cliprange <= 0:
            errors.append("cliprange must be positive")

        if hasattr(train_config, 'vf_coef') and train_config.vf_coef < 0:
            errors.append("vf_coef must be non-negative")

        # DPO-specific validations
        if hasattr(train_config, 'beta') and train_config.beta <= 0:
            errors.append("DPO beta must be positive")

        return errors

    @staticmethod
    def _validate_dpo_config(config) -> List[str]:
        """Validate DPO-specific configuration."""
        errors = []

        # Check for required reward functions
        if not config.rewards:
            errors.append("DPO requires at least one reward function")

        return errors

    @staticmethod
    def _validate_ppo_config(config) -> List[str]:
        """Validate PPO-specific configuration."""
        errors = []

        # Check for reward model
        if not hasattr(config.model, 'reward_model_name') and not hasattr(config.model, 'reward_model_path'):
            if not hasattr(config.model, 'reward_model_source') or not config.model.reward_model_source:
                errors.append("PPO requires a reward model (reward_model_name, reward_model_path, or reward_model_source)")

        return errors

    @staticmethod
    def _validate_grpo_config(config) -> List[str]:
        """Validate GRPO-specific configuration."""
        errors = []

        # GRPO requires reward functions
        if not config.rewards:
            errors.append("GRPO requires reward functions")

        return errors

    @staticmethod
    def estimate_memory_usage(config, num_gpus: int = 1) -> Dict[str, Any]:
        """
        Estimate memory usage for a configuration.

        Args:
            config: Training configuration
            num_gpus: Number of GPUs

        Returns:
            Dictionary with memory estimates
        """
        # Rough memory estimation based on model size and batch size
        model_params = ConfigValidator._estimate_model_params(config)
        batch_size = getattr(config.train, 'per_device_batch_size', 1) * getattr(config.train, 'gradient_accumulation_steps', 1)

        # Base memory per sample (rough estimate)
        memory_per_sample_gb = model_params / 1e9 * 2  # 2 bytes per param for fp16/bf16

        # Total memory estimate
        total_memory_gb = memory_per_sample_gb * batch_size * 4  # 4x for gradients, optimizer states, etc.

        return {
            "estimated_model_params": model_params,
            "memory_per_sample_gb": memory_per_sample_gb,
            "total_estimated_memory_gb": total_memory_gb,
            "recommended_min_memory_gb": total_memory_gb * 1.5,  # 50% buffer
            "num_gpus": num_gpus,
            "memory_per_gpu_gb": total_memory_gb / num_gpus if num_gpus > 0 else total_memory_gb
        }

    @staticmethod
    def _estimate_model_params(config) -> int:
        """Roughly estimate number of model parameters."""
        model_name = config.model.name_or_path.lower()

        # Rough parameter counts for common models
        param_estimates = {
            "llama-3": 8e9,
            "llama-2": 7e9,
            "qwen": 7e9,
            "mistral": 7e9,
            "gemma": 7e9,
            "phi": 3e9,
            "falcon": 7e9,
            "opt": 6e9,
            "gpt": 6e9,
        }

        for key, params in param_estimates.items():
            if key in model_name:
                return int(params)

        # Default estimate
        return 7e9  # 7B parameters

    @staticmethod
    def check_environment_compatibility(config) -> Dict[str, Any]:
        """
        Check if the current environment is compatible with the configuration.

        Args:
            config: Training configuration

        Returns:
            Dictionary with compatibility results
        """
        results = {
            "cuda_available": False,
            "torch_cuda_available": False,
            "bitsandbytes_available": False,
            "unsloth_available": False,
            "flash_attention_available": False,
            "compatible": True,
            "issues": []
        }

        try:
            import torch
            results["torch_cuda_available"] = torch.cuda.is_available()
            results["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            results["issues"].append("PyTorch not available")
            results["compatible"] = False

        # Check bitsandbytes
        if hasattr(config.model, 'quantization') and 'load_in_4bit' in config.model.quantization:
            try:
                import bitsandbytes
                results["bitsandbytes_available"] = True
            except ImportError:
                results["issues"].append("bitsandbytes required for 4-bit quantization")
                results["compatible"] = False

        # Check unsloth
        if hasattr(config.model, 'use_unsloth') and config.model.use_unsloth:
            try:
                import unsloth
                results["unsloth_available"] = True
            except ImportError:
                results["issues"].append("unsloth required when use_unsloth=True")
                results["compatible"] = False

        # Check flash attention
        if hasattr(config.model, 'attn_implementation') and config.model.attn_implementation == "flash_attention_2":
            try:
                from transformers import AutoModelForCausalLM
                # Flash attention 2 is available in newer transformers
                results["flash_attention_available"] = True
            except ImportError:
                results["issues"].append("Flash attention requires newer transformers version")
                results["compatible"] = False

        return results

    @staticmethod
    def validate_dataset_accessibility(dataset_name: str, config: Optional[Dict] = None) -> bool:
        """
        Check if a dataset is accessible.

        Args:
            dataset_name: Name of the dataset
            config: Optional dataset configuration

        Returns:
            True if accessible, False otherwise
        """
        try:
            from datasets import load_dataset
            # Try to load just the dataset info (not the full dataset)
            load_dataset(dataset_name, streaming=True, split="train[:1]")
            return True
        except Exception as e:
            logger.warning(f"Dataset {dataset_name} may not be accessible: {e}")
            return False

    @staticmethod
    def validate_model_accessibility(model_name: str, requires_auth: bool = False) -> bool:
        """
        Check if a model is accessible.

        Args:
            model_name: Name of the model
            requires_auth: Whether the model requires authentication

        Returns:
            True if accessible, False otherwise
        """
        try:
            from transformers import AutoConfig
            # Try to load just the config
            config = AutoConfig.from_pretrained(
                model_name,
                token=True if requires_auth else None
            )
            return True
        except Exception as e:
            logger.warning(f"Model {model_name} may not be accessible: {e}")
            return False


def estimate_memory_usage(config: Union[SFTConfig, UnifiedConfig]) -> float:
    """Estimate memory usage for a configuration."""
    validator = ConfigValidator()
    return validator.estimate_memory_usage(config)


def check_hf_authentication() -> bool:
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        return True
    except:
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    validator = ConfigValidator()
    return validator.system_info


def validate_model_access(model_name: str) -> Tuple[bool, str]:
    """
    Validate model accessibility.

    Returns:
        Tuple of (is_accessible, message)
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        return True, f"Model {model_name} is accessible locally"
    except Exception as e:
        # Try remote access
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, local_files_only=False)
            return True, f"Model {model_name} is accessible remotely"
        except Exception as e2:
            if "authentication" in str(e2).lower():
                return False, f"Model {model_name} requires authentication. Run 'huggingface-cli login'"
            else:
                return False, f"Model {model_name} is not accessible: {str(e2)}"


def validate_dataset_access(dataset_name: str) -> Tuple[bool, str]:
    """
    Validate dataset accessibility.

    Returns:
        Tuple of (is_accessible, message)
    """
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
        return True, f"Dataset {dataset_name} is accessible ({len(configs)} configs)"
    except Exception as e:
        if "authentication" in str(e).lower() or "403" in str(e):
            return False, f"Dataset {dataset_name} requires authentication. Run 'huggingface-cli login'"
        else:
            return False, f"Dataset {dataset_name} is not accessible: {str(e)}"


def validate_config(config, config_type: str = "auto") -> Tuple[bool, List[str]]:
    """
    Validate a configuration and return results.

    Args:
        config: Configuration object
        config_type: Type of config ("sft", "rl", or "auto")

    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ConfigValidator()

    if config_type == "auto":
        # Try to detect config type
        if hasattr(config, 'dataset') and hasattr(config, 'train') and hasattr(config, 'logging'):
            config_type = "sft"
        elif hasattr(config, 'algo') and hasattr(config, 'datasets'):
            config_type = "rl"
        else:
            return False, ["Unable to determine configuration type"]

    if config_type == "sft":
        errors = validator.validate_sft_config(config)
    elif config_type == "rl":
        errors = validator.validate_rl_config(config)
    else:
        return False, [f"Unknown configuration type: {config_type}"]

    return len(errors) == 0, errors
