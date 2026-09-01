"""
Unsloth GRPO Backend Implementation - COMPLETE WORKING VERSION

This module provides an Unsloth-optimized backend for Group Relative Policy Optimization (GRPO),
using Unsloth's FastLanguageModel for optimized model loading and TRL's GRPOTrainer
for the training loop. This provides 2-5x speed improvements over standard GRPO training.

Key difference from custom GRPO: This delegates the actual training loop to TRL's GRPOTrainer,
while the custom GRPO implements the training loop manually.
"""

import logging
import time
import yaml
import asyncio
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import torch
from torch.utils.data import DataLoader

from aligntune.core.rl import TrainerBase
from aligntune.core.registry import TaskType
from aligntune.core.model_loader import build_model
from aligntune.core.rl.config import UnifiedConfig
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class UnslothGRPOTrainer(TrainerBase):
    """GRPO trainer using Unsloth's FastLanguageModel for optimized training."""

    TASK_TYPE = "grpo"  # Used by base class setup_data()
    KEEP_COLUMNS = True

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.training_history = []
        self.logging_manager = None
        self.reward_functions = []
        self.train_dataset = None
        self.eval_dataset = None
        self.reward_configs = []
        self.custom_evaluator = None  #
        self.dataset_dict = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth and TRL are available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import GRPOTrainer, GRPOConfig
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup models natively via model_loader."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info(f"Setting up Unsloth models via model_loader")
        logger.info("=" * 80)
        
        from aligntune.core.registry import TaskType
        from aligntune.core.model_loader import build_model
        import copy

        peft_enabled = getattr(self.config.model, 'use_peft', True)
        
        # Policy Model
        self.model, self.tokenizer = build_model(
            self.config, task_type=TaskType.SFT, apply_peft=peft_enabled, use_unsloth=True
        )
        
        # Reference Model
        if getattr(self, "needs_ref_model", False):
            if not peft_enabled:
                logger.info("Full fine-tuning: Loading frozen reference model")
                self.ref_model, _ = build_model(
                    self.config, task_type=TaskType.SFT, is_reference=True, use_unsloth=True
                )
                self.reference_model = self.ref_model
            else:
                logger.info("PEFT enabled: Unsloth handles reference model natively.")
                self.ref_model = None
                self.reference_model = None
                
        # Reward / Value Models (Unsloth doesn't perfectly support SequenceClassification yet, 
        # so fallback to TRL loading for them, as they are just classification heads)
        if getattr(self, "needs_reward_model", False):
            rew_config = copy.deepcopy(self.config)
            reward_path = getattr(self.config.model, 'reward_model_path', None)
            if reward_path: rew_config.model.name_or_path = reward_path
            self.reward_model, _ = build_model(rew_config, task_type=TaskType.TEXT_CLASSIFICATION, use_unsloth=False)
            
        if getattr(self, "needs_value_model", False):
            val_config = copy.deepcopy(self.config)
            val_path = getattr(self.config.model, 'value_model_path', None)
            if val_path: val_config.model.name_or_path = val_path
            self.value_model, _ = build_model(val_config, task_type=TaskType.TEXT_CLASSIFICATION, use_unsloth=False)

    # setup_data() inherited from TrainerBase - uses TASK_TYPE="grpo"

    def setup_trainer(self) -> None:
        """Setup TRL GRPOTrainer with Unsloth model."""
        try:
            from trl import GRPOTrainer, GRPOConfig

            logger.info("Setting up TRL GRPOTrainer with Unsloth model")

            # Handle both dict and object config
            if isinstance(self.config.logging, dict):
                output_dir = self.config.logging.get(
                    'output_dir', './output/grpo')
            else:
                output_dir = getattr(self.config.logging,
                                     'output_dir',
                                     './output/grpo') if hasattr(self.config,
                                                                 'logging') else './output/grpo'

            if isinstance(self.config.train, dict):
                num_epochs = self.config.train.get('epochs', 1)
                per_device_batch_size = self.config.train.get(
                    'per_device_batch_size', 4)
                gradient_accumulation_steps = self.config.train.get(
                    'gradient_accumulation_steps', 1)
                learning_rate = self.config.train.get('learning_rate', 2e-4)
                save_interval = self.config.train.get('save_interval', 500)
                eval_interval = self.config.train.get('eval_interval', 500)
                kl_coef = self.config.train.get('kl_coef', 0.1)
                cliprange = self.config.train.get('cliprange', 0.2)
            else:
                num_epochs = getattr(self.config.train, 'epochs', 1)
                per_device_batch_size = getattr(
                    self.config.train, 'per_device_batch_size', 4)
                gradient_accumulation_steps = getattr(
                    self.config.train, 'gradient_accumulation_steps', 1)
                learning_rate = getattr(
                    self.config.train, 'learning_rate', 2e-4)
                save_interval = getattr(
                    self.config.train, 'save_interval', 500)
                eval_interval = getattr(
                    self.config.train, 'eval_interval', 500)
                kl_coef = getattr(self.config.train, 'kl_coef', 0.1)
                cliprange = getattr(self.config.train, 'cliprange', 0.2)

            # Get max sequence length from model config
            if isinstance(self.config.model, dict):
                max_seq_length = self.config.model.get('max_seq_length', 512)
            else:
                max_seq_length = getattr(
                    self.config.model, 'max_seq_length', 512)

            # CRITICAL: Calculate max_prompt_length and max_completion_length
            max_prompt_length = int(max_seq_length * 0.6)  # 60% for prompt
            max_completion_length = int(
                max_seq_length * 0.4)  # 40% for completion

            # Determine evaluation and save strategy based on eval_dataset
            has_eval = self.eval_dataset is not None and len(
                self.eval_dataset) > 0

            # Set save strategy
            save_strategy = "steps"

            from aligntune.core.precision_handler import PrecisionHandler
            precision = PrecisionHandler.get_precision_from_config(
                self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "GRPO (Unsloth)")
            precision_args = PrecisionHandler.get_training_args_precision(
                precision)
            # Evaluation parameters
            eval_strategy = self._get_config_value(
                self.config.train, 'eval_strategy', default='steps')
            eval_steps = self._get_config_value(
                self.config.train, 'eval_steps', default=None)
            per_device_eval_batch_size = self._get_config_value(
                self.config.train, 'per_device_eval_batch_size', default=per_device_batch_size)
            metric_for_best_model = self._get_config_value(
                self.config.train, 'metric_for_best_model', default=None)
            greater_is_better = self._get_config_value(
                self.config.train, 'greater_is_better', default=False)
            load_best_model_at_end = self._get_config_value(
                self.config.train, 'load_best_model_at_end', default=False)

            # Adjust eval strategy based on eval_dataset availability
            if not self.eval_dataset:
                eval_strategy = 'no'
                eval_steps = None

            # Logging parameters
            logging_steps = self._get_config_value(
                self.config.train, 'logging_steps', default=10)
            logging_strategy = self._get_config_value(
                self.config.train, 'logging_strategy', default='steps')
            save_total_limit = self._get_config_value(
                self.config.train, 'save_total_limit', default=None)

            report_to = self.config.logging.loggers if self.config.logging.loggers else []

            # Run name
            run_name = self._get_config_value(
                self.config.logging, 'run_name', default='unsloth_grpo')

            # Optimizer parameters
            optimizer = self._get_config_value(
                self.config.train, 'optimizer', default='adamw_torch')
            lr_scheduler_type = self._get_config_value(
                self.config.train, 'lr_scheduler', default='cosine')
            warmup_ratio = self._get_config_value(
                self.config.train, 'warmup_ratio', default=0.1)
            warmup_steps = self._get_config_value(
                self.config.train, 'warmup_steps', default=0)
            weight_decay = self._get_config_value(
                self.config.train, 'weight_decay', default=0.0)

            # Additional training parameters
            gradient_checkpointing = self._get_config_value(
                self.config.train,
                'use_gradient_checkpointing',
                'gradient_checkpointing',
                default=True)
            group_by_length = self._get_config_value(
                self.config.train, 'group_by_length', default=True)
            seed = self._get_config_value(
                self.config.train, 'seed', default=42)
            data_seed = self._get_config_value(
                self.config.train, 'data_seed', default=47)
            max_steps = self._get_config_value(self.config.train, "max_steps", 500)

            # GRPO specific parameters
            beta = self._get_config_value(
                self.config.train, 'beta', 'kl_coef', default=kl_coef)
            epsilon = self._get_config_value(
                self.config.train, 'epsilon', 'cliprange', default=cliprange)
            loss_type = self._get_config_value(
                self.config.train, 'loss_type', default='grpo')

            # Force GRPO loss type (override any DPO loss types like 'sigmoid')
            if loss_type not in ('grpo', 'dapo', 'bnpo', 'dr_grpo', 'cispo', 'sapo', 'luspo', 'vespo'):
                loss_type = 'grpo'
            scale_rewards = self._get_config_value(
                self.config.train, 'scale_rewards', default='group')
            mask_truncated_completions = self._get_config_value(
                self.config.train, 'mask_truncated_completions', default=True)
            temperature = self._get_config_value(
                self.config.train, 'temperature', default=0.7)
            top_p = self._get_config_value(
                self.config.train, 'top_p', default=0.9)
            num_generations = self._get_config_value(
                self.config.train, 'num_generations', default=per_device_batch_size)
            if not hasattr(self, 'prepared_rewards'):
                self.setup_rewards()
            prepared_rewards = self.prepared_rewards
            if not prepared_rewards.functions:
                raise ValueError("GRPO requires at least one configured reward function.")

            # GSPO variant: sequence-level importance sampling
            importance_sampling_level = self._get_config_value(
                self.config.train, 'importance_sampling_level', default='token')

            # Adjust save strategy to save on epoch
            save_strategy = "epoch" if self.eval_dataset else "steps"

            # Create GRPO configuration
            grpo_config = GRPOConfig(
                # Output and logging
                output_dir=output_dir,
                run_name=run_name,
                logging_steps=logging_steps,
                logging_strategy=logging_strategy,
                report_to=report_to,
                max_steps=max_steps,
                # Evaluation
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                load_best_model_at_end=load_best_model_at_end,

                # Checkpointing
                save_steps=save_interval,
                save_strategy=save_strategy,
                save_total_limit=save_total_limit,

                # Training parameters
                num_train_epochs=num_epochs,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                max_grad_norm=0.5,

                # Optimizer and scheduler
                optim=optimizer,
                lr_scheduler_type=lr_scheduler_type,

                # GRPO specific parameters
                max_completion_length=max_completion_length,
                num_generations=num_generations,
                temperature=temperature,
                top_p=top_p,
                loss_type=loss_type,
                importance_sampling_level=importance_sampling_level,
                beta=beta,
                epsilon=epsilon,
                scale_rewards=scale_rewards,
                mask_truncated_completions=mask_truncated_completions,
                reward_weights=prepared_rewards.weights,

                # Seeds
                seed=seed,
                data_seed=data_seed,

                # Performance
                gradient_checkpointing=gradient_checkpointing,
                dataloader_pin_memory=False,

                # Precision
                **precision_args,

                # Other settings
                remove_unused_columns=False,
            )
            
            missing = extract_extra_and_missing_params(
                backend_config=grpo_config,
                config=self.config,
                algorithm='grpo'
            )

            for key, value in missing.items():
                setattr(grpo_config, key, value)
            self.reward_functions = prepared_rewards.functions

            self.trainer = GRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                reward_funcs=self.reward_functions,
            )

            logger.info("✓ GRPO trainer setup completed")

        except Exception as e:
            logger.error(f"Failed to setup GRPO trainer: {e}")
            raise

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute a single training step.

        NOTE: For Unsloth GRPO, the actual training is handled by TRL's GRPOTrainer,
        so this method is not used during training. It's implemented to satisfy
        the TrainerBase abstract method requirement.
        """
        # This method is required by TrainerBase but not used in Unsloth GRPO
        # because TRL's GRPOTrainer handles the training loop internally
        logger.debug(
            "train_step() called but Unsloth GRPO uses TRL's internal training loop")
        return {"loss": 0.0}

    def create_data_loader(self) -> Optional[DataLoader]:
        """
        Create data loader for training.

        NOTE: For Unsloth GRPO, data loading is handled by TRL's GRPOTrainer,
        so this method returns None. It's implemented to satisfy the TrainerBase
        abstract method requirement if it exists.
        """
        # This method might be required by TrainerBase but not used in Unsloth GRPO
        # because TRL's GRPOTrainer handles data loading internally
        logger.debug(
            "create_data_loader() called but Unsloth GRPO uses TRL's internal data loading")
        return None

    def train(self) -> Dict[str, Any]:
        """Execute GRPO training with Unsloth optimizations."""
        try:
            logger.info("Starting Unsloth GRPO training")
            start_time = time.time()

            # Setup components
            self.setup_model()
            self.setup_data()
            self.setup_rewards()
            self.setup_trainer()
            
            # Add the callbacks to the trainer
            for cb in self.get_hf_callbacks():
                self.trainer.add_callback(cb)

            # Start training
            training_result = self.trainer.train()

            # Get output directory
            if isinstance(self.config.logging, dict):
                output_dir = self.config.logging.get(
                    'output_dir', './output/grpo')
            else:
                output_dir = getattr(self.config.logging,
                                     'output_dir',
                                     './output/grpo') if hasattr(self.config,
                                                                 'logging') else './output/grpo'

            # Save model
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            training_time = time.time() - start_time

            # Compile results
            results = {
                "training_time": training_time,
                "final_loss": training_result.training_loss if hasattr(
                    training_result,
                    'training_loss') else 0.0,
                "total_steps": training_result.global_step if hasattr(
                    training_result,
                    'global_step') else 0,
                "model_path": output_dir,
                "training_history": self.training_history,
                "num_reward_functions": len(
                    self.reward_functions),
                "num_datasets": len(
                    self.config.datasets) if hasattr(
                        self.config,
                        'datasets') else 0,
            }

            logger.info(
                f"Unsloth GRPO training completed in {training_time:.2f} seconds"
            )
            if hasattr(training_result, 'training_loss'):
                logger.info(f"Final loss: {training_result.training_loss:.4f}")

            return results

        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise

    # def evaluate(self) -> Dict[str, Any]:
    #     """Evaluate the trained GRPO model."""
    #     try:
    #         if not self.eval_dataset:
    #             logger.warning("No evaluation dataset available")
    #             return {}

    #         logger.info("Evaluating Unsloth GRPO model")

    #         # Run evaluation
    #         eval_results = self.trainer.evaluate()

    #         logger.info(f"GRPO evaluation results: {eval_results}")

    #         return eval_results

    #     except Exception as e:
    #         logger.error(f"GRPO evaluation failed: {e}")
    #         raise
    def evaluate(
        self,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
        use_custom_evaluator: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """GRPO-specific evaluation - auto-setup evaluators and delegate to parent."""

        # Auto-setup evaluators on first call
        if self.base_evaluator is None and self.rl_evaluator is None:
            logger.info("Auto-initializing evaluators for first evaluation...")
            self.setup_custom_evaluator(evaluator_type="auto")

        # Call parent's unified evaluate method
        return super().evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=metric_key_prefix,
            use_custom_evaluator=use_custom_evaluator,
            **kwargs
        )
