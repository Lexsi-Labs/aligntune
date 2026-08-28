"""
TRL GRPO Backend Implementation with Math, Code, and Enhanced Rewards

This module provides a pure TRL backend for Group Relative Policy Optimization (GRPO),
with integrated support for mathematical reasoning, code quality/correctness, and
additional reward functions for diversity, fluency, relevance, and brevity.

UPDATED: Added beta/epsilon parameter support (backwards compatible with kl_coef/cliprange)
"""

import logging
import time
import yaml
import re
import ast
import math
import asyncio
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import torch
from torch.utils.data import DataLoader
import os
from aligntune.core.rl import TrainerBase
from aligntune.core.registry import TaskType
from aligntune.core.model_loader import build_model
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.reward_handler import (
    resolve_trl_reward_weights,
)
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import extract_extra_and_missing_params
logger = logging.getLogger(__name__)


# ============================================================================
# TRL GRPO TRAINER (ENHANCED WITH NEW REWARDS)
# ============================================================================

class TRLGRPOTrainer(TrainerBase):
    """GRPO trainer using pure TRL GRPOTrainer with math, code, and enhanced rewards."""

    TASK_TYPE = "grpo"  # Used by base class setup_data()
    KEEP_COLUMNS = True

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.dataset = None
        self.eval_dataset = None  # Add eval dataset support
        self.reward_functions = []
        self.training_history = []
        self.logging_manager = None
        # self.evaluator = None
        self.custom_evaluator = None  # For BaseEvaluator/RLEvaluator
        self.dataset_dict = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL is available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup policy and reference models natively via model_loader."""
        logger.info("=" * 80)
        logger.info("Setting up TRL GRPO models via model_loader")
        logger.info("=" * 80)
        
        from aligntune.core.registry import TaskType
        from aligntune.core.model_loader import build_model

        peft_enabled = getattr(self.config.model, 'use_peft', False) or \
                       getattr(self.config.model, 'load_in_4bit', False) or \
                       getattr(self.config.model, 'load_in_8bit', False)
        
        # 1. Policy Model
        self.model, self.tokenizer = build_model(
            self.config, task_type=TaskType.SFT, apply_peft=peft_enabled
        )
        
        # 2. Reference Model
        if not peft_enabled:
            logger.info("Full fine-tuning: Loading frozen reference model")
            self.ref_model, _ = build_model(
                self.config, task_type=TaskType.SFT, is_reference=True
            )
            self.reference_model = self.ref_model
        else:
            logger.info("PEFT enabled: TRL handles reference model via adapter toggle.")
            self.ref_model = None
            self.reference_model = None

    # setup_data() inherited from TrainerBase - uses TASK_TYPE="grpo"
        
        # Log sample if dataset is loaded
        if hasattr(self, 'train_dataset') and self.train_dataset is not None and len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            prompt_col = "prompt" if "prompt" in sample else "query"
            logger.info(f"Sample prompt (first 100 chars): {sample[prompt_col][:100]}...")
            if hasattr(self.train_dataset, 'column_names'):
                logger.info(f"Dataset columns: {self.train_dataset.column_names}")


    def setup_trainer(self) -> None:
        """
        Set up the GRPO trainer with all configurations.
        """
        logger.info("Setting up TRL GRPO trainer...")

        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision_args = PrecisionHandler.get_training_args_precision(
            precision)

        # Get training parameters
        num_epochs = self._get_config_value(
            self.config.train,
            'epochs',
            'num_epochs',
            'num_train_epochs',
            default=1)
        learning_rate = self._get_config_value(
            self.config.train, 'learning_rate', 'lr', default=1e-6)
        per_device_batch_size = self._get_config_value(
            self.config.train, 'per_device_batch_size', 'batch_size', default=1)
        gradient_accumulation_steps = self._get_config_value(
            self.config.train, 'gradient_accumulation_steps', default=32)
        max_grad_norm = self._get_config_value(
            self.config.train, 'max_grad_norm', default=1.0)
        weight_decay = self._get_config_value(
            self.config.train, 'weight_decay', default=0.0)
        warmup_steps = self._get_config_value(
            self.config.train, 'warmup_steps', default=10)
        seed = self._get_config_value(self.config.train, 'seed', default=42)
        precision = self._get_config_value(
            self.config.model, 'precision', default='fp32')
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/grpo_trl')

        # GRPO-specific parameters (beta = KL coefficient, epsilon = clip
        # range)
        beta = self._get_config_value(
            self.config.train, 'beta', 'kl_coef', default=0.1)
        epsilon = self._get_config_value(
            self.config.train, 'epsilon', 'cliprange', default=0.2)

        # GRPO variants: loss_type and importance sampling
        loss_type = self._get_config_value(
            self.config.train, 'loss_type', default='grpo')
        # Force a valid GRPO-family loss type (override any DPO loss types like
        # 'sigmoid' that could leak in from factory defaults), but allow all of
        # TRL's actual supported GRPOConfig loss_type values (used by GRPO
        # variants like DAPO and Dr. GRPO).
        if loss_type not in ('grpo', 'dapo', 'bnpo', 'dr_grpo', 'cispo', 'sapo', 'luspo', 'vespo'):
            loss_type = 'grpo'
        importance_sampling_level = self._get_config_value(
            self.config.train, 'importance_sampling_level', default='token')

        # Generation parameters
        num_generations = self._get_config_value(
            self.config.train, 'num_generations', default=4)
        max_completion_length = self._get_config_value(
            self.config.train, 'max_completion_length', 'max_new_tokens', default=256)
        max_prompt_length = self._get_config_value(
            self.config.train, 'max_prompt_length', default=512)
        temperature = self._get_config_value(
            self.config.train, 'temperature', default=0.7)
        top_p = self._get_config_value(
            self.config.train, 'top_p', default=0.95)

        max_steps = self._get_config_value(self.config.train, "max_steps", 500)

        # Rollout backend configuration
        rollout_backend = self._get_config_value(
            self.config.train, 'rollout_backend', default='hf')
        vllm_gpu_memory_utilization = self._get_config_value(
            self.config.train, 'vllm_gpu_memory_utilization', default=0.7)
        vllm_tensor_parallel_size = self._get_config_value(
            self.config.train, 'vllm_tensor_parallel_size', default=1)
            

        # Evaluation parameters
        eval_strategy = self._get_config_value(
            self.config.train, 'eval_strategy', default='epoch')
        eval_steps = self._get_config_value(
            self.config.train, 'eval_steps', default=100)
        per_device_eval_batch_size = self._get_config_value(
            self.config.train, 'per_device_eval_batch_size', default=per_device_batch_size)

        # Save parameters
        save_steps = self._get_config_value(
            self.config.train, 'save_steps', default=100)
        save_strategy = self._get_config_value(
            self.config.train, 'save_strategy', default='steps')
        save_total_limit = self._get_config_value(
            self.config.train, 'save_total_limit', default=None)
        load_best_model_at_end = self._get_config_value(
            self.config.train,
            'load_best_model_at_end',
            default=True if self.eval_dataset else False)
        metric_for_best_model = self._get_config_value(
            self.config.train,
            'metric_for_best_model',
            default='eval_loss' if self.eval_dataset else None)
        greater_is_better = self._get_config_value(
            self.config.train, 'greater_is_better', default=False)

        # Logging parameters
        logging_steps = self._get_config_value(
            self.config.train, 'logging_steps', default=10)
        logging_strategy = self._get_config_value(
            self.config.train, 'logging_strategy', default='steps')
        report_to = self.config.logging.loggers if self.config.logging.loggers else []
        # Use eval_dataset-aware defaults
        if not self.eval_dataset:
            eval_strategy = 'no'
            eval_steps = None

        logger.info("=" * 80)
        logger.info("TRL GRPO Training Configuration")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Batch size: {per_device_batch_size}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"Beta (KL coefficient): {beta}")
        logger.info(f"Epsilon (clip range): {epsilon}")
        logger.info(f"Num generations per prompt: {num_generations}")
        logger.info(f"Max completion length: {max_completion_length}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Rollout backend: {rollout_backend}")
        if rollout_backend == "vllm":
            logger.info(f"  vLLM GPU memory utilization: {vllm_gpu_memory_utilization}")
            logger.info(f"  vLLM tensor parallel size: {vllm_tensor_parallel_size}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup GRPO trainer
        from trl import GRPOTrainer, GRPOConfig

        if not hasattr(self, 'prepared_rewards'):
            self.setup_rewards()
        prepared_rewards = self.prepared_rewards
        reward_functions = prepared_rewards.functions
        if not reward_functions:
            raise ValueError("GRPO requires at least one configured reward function.")
        reward_weights = resolve_trl_reward_weights(
            prepared_rewards,
            self._get_config_value(self.config.train, 'reward_weights', default=None),
        )

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,

            # Evaluation parameters
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,

            # Logging parameters
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            report_to=report_to if report_to else [],

            # Save parameters
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,

            # Training parameters
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            seed=seed,
            remove_unused_columns=False,

            # Generation parameters
            # NOTE: max_prompt_length was removed from trl's GRPOConfig in the
            # installed trl version (1.7.1) - only max_completion_length remains.
            # Passing it raises "GRPOConfig.__init__() got an unexpected keyword
            # argument 'max_prompt_length'". We still read it from config above
            # for logging, but don't forward it to GRPOConfig.
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            temperature=temperature,
            top_p=top_p,
            max_steps=max_steps,

            # GRPO-specific
            beta=beta,
            loss_type=loss_type,
            importance_sampling_level=importance_sampling_level,
            reward_weights=reward_weights,

            # vLLM Rollout Backend
            use_vllm=(rollout_backend == "vllm"),
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,

            # Precision
            **precision_args,
        )

        
        missing = extract_extra_and_missing_params(
            backend_config=grpo_config,
            config=self.config,
            algorithm='grpo'
        )

        for key, value in missing.items():
            setattr(grpo_config, key, value)


        # The central bridge converts registry rewards to TRL's batch contract.
        self.reward_functions = reward_functions

        # Create GRPO trainer
        logger.info(
            "Creating GRPOTrainer with %s reward functions and weights %s",
            len(reward_functions),
            reward_weights,
        )
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
        )

        logger.info("GRPO trainer setup completed successfully!")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute a single training step.

        Note: TRL GRPO uses TRL's internal training loop, so this method
        is not called during normal training. Kept for interface compatibility.
        """
        return {"loss": 0.0}

    def train(self) -> Dict[str, Any]:
        """Execute GRPO training."""
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_rewards()
        self.setup_trainer()

        # Get output directory for saving
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/grpo_trl')

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting TRL GRPO Training")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        # Add the callbacks to the trainer
        for cb in self.get_hf_callbacks():
            self.trainer.add_callback(cb)

        # Start training
        train_result = self.trainer.train()

        # Log samples after training
        try:
            # reward_functions is now a list of callables (simplified)
            reward_callables = getattr(self, "reward_functions", None) or None

            if reward_callables:
                generate_and_log_samples(
                    self.config.logging.sample_logging,
                    self.model,
                    self.tokenizer,
                    reward_callables,
                    stage="post-train",
                    log=logger,
                )
        except Exception as sample_error:
            logger.warning(
                f"Unable to log qualitative samples: {sample_error}")

        # Record training end
        end_time = time.time()
        training_duration = end_time - start_time

        logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Extract metrics
        metrics = {}
        if hasattr(train_result, 'metrics'):
            metrics = train_result.metrics

        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Compile results
        results = {
            "training_time": training_duration,
            "final_loss": train_result.training_loss if hasattr(
                train_result,
                'training_loss') else metrics.get(
                'train_loss',
                0.0),
            "total_steps": train_result.global_step if hasattr(
                train_result,
                'global_step') else 0,
            "model_path": output_dir,
            "num_reward_functions": len(
                    self.reward_functions),
            "num_datasets": 1,
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("TRL GRPO Training Completed Successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)

        return results

    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     metric_key_prefix: str = "eval",
    #     use_custom_evaluator: bool = True,
    #     **kwargs
    # ) -> Dict[str, float]:
    #     """GRPO-specific evaluation - auto-setup evaluators and delegate to parent."""

    #     # Auto-setup evaluators on first call
    #     if self.base_evaluator is None and self.rl_evaluator is None:
    #         logger.info("Auto-initializing evaluators for first evaluation...")
    #         self.setup_custom_evaluator(evaluator_type="auto")

    #     # Call parent's unified evaluate method
    #     return super().evaluate(
    #         eval_dataset=eval_dataset,
    #         metric_key_prefix=metric_key_prefix,
    #         use_custom_evaluator=use_custom_evaluator,
    #         **kwargs
    #     )

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        num_epochs = self._get_config_value(
            self.config.train,
            'epochs',
            'num_epochs',
            'num_train_epochs',
            default=1)

        stats = {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': 'group_relative_policy_optimization',
                'dataset_name': self.config.dataset.name,
                'epochs': num_epochs,
                'learning_rate': self._get_config_value(
                    self.config.train,
                    'learning_rate',
                    'lr',
                    default=2e-4),
                'batch_size': self._get_config_value(
                    self.config.train,
                    'per_device_batch_size',
                    'batch_size',
                    default=4),
                'use_peft': self._get_config_value(
                    self.config.model,
                    'use_peft',
                    default=False),
                'precision': self._get_config_value(
                    self.config.model,
                    'precision',
                    default='fp32'),
                'num_reward_functions': len(
                    self.reward_functions),
            },
            'dataset_info': {
                'train_size': len(
                    self.train_dataset) if hasattr(
                    self,
                    'dataset') and self.train_dataset else 0,
                'val_size': 0,
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(
                    next(
                        self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(
                    self.tokenizer) if self.tokenizer else 0,
                'has_peft': hasattr(
                    self.model,
                    'peft_config') if self.model else False,
            },
            'training_history': self.training_history,
        }

        return stats

    def save_config(self, path: str):
        """Save configuration to YAML file."""
        num_epochs = self._get_config_value(
            self.config.train,
            'epochs',
            'num_epochs',
            'num_train_epochs',
            default=1)

        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': 'group_relative_policy_optimization',
            'max_seq_length': self._get_config_value(
                self.config.model,
                'max_seq_length',
                default=512),
            'learning_rate': self._get_config_value(
                self.config.train,
                'learning_rate',
                'lr',
                default=2e-4),
            'epochs': num_epochs,
            'batch_size': self._get_config_value(
                self.config.train,
                'per_device_batch_size',
                'batch_size',
                default=4),
            'dataset_name': self.config.dataset.name,
            'use_peft': self._get_config_value(
                self.config.model,
                'use_peft',
                default=False),
            'precision': self._get_config_value(
                self.config.model,
                'precision',
                default='fp32'),
            'num_reward_functions': len(
                self.reward_functions),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"TRL GRPO configuration saved to {path}")
