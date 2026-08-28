"""
Unsloth DPO Trainer.

This module provides Unsloth-optimized Direct Preference Optimization training
using TRL's DPOTrainer with Unsloth's performance optimizations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.registry import TaskType
from aligntune.core.model_loader import build_model
from aligntune.core.rl.config import UnifiedConfig
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class UnslothDPOTrainer(TrainerBase):
    """Unsloth-optimized DPO trainer using TRL's DPOTrainer."""

    TASK_TYPE = "dpo"  # DPO uses the shared preference schema
    KEEP_COLUMNS = False
    needs_ref_model: bool = True

    def __init__(self, config: UnifiedConfig):
        """Initialize Unsloth DPO trainer."""
        super().__init__(config)
        self.model = None
        self.reference_model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.custom_evaluator = None
        self.dataset_dict = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth DPO trainer is available."""
        try:
            import unsloth
            from trl import DPOTrainer, DPOConfig, ModelConfig
            return True
        except ImportError:
            return False

    def setup_rewards(self) -> None:
        """Setup reward functions (handled by TRL trainer for DPO)."""
        pass

    def train_step(self, batch):
        """Train step (handled by TRL trainer for DPO)."""
        pass

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

    # setup_data() inherited from TrainerBase - uses TASK_TYPE="dpo"
    # DataFilter in DataManager handles all validation (empty strings, None values, length checks)

    def setup_trainer(self) -> None:
        """Setup TRL DPOTrainer with Unsloth model."""
        try:
            from trl import DPOTrainer, DPOConfig, ModelConfig

            logger.info("Setting up TRL DPOTrainer with Unsloth model")

            # Get optimizer, scheduler, and training params from base class
            optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

            # Extract values from returned dict
            max_steps = optim_scheduler['max_steps']
            num_epochs = optim_scheduler['num_epochs']
            eval_strategy = 'steps' if max_steps > 0 else 'epoch'
            if not self.eval_dataset:
                eval_strategy = 'no'

            # Adjust save strategy to save on epoch if eval_dataset exists
            save_strategy = 'epoch' if self.eval_dataset else 'steps'

            # Create DPO configuration
            dpo_config = DPOConfig(
                output_dir=self._get_config_value(self.config.logging, 'output_dir', default='./output'),
                run_name=self._get_config_value(self.config.logging, 'run_name', default='unsloth_dpo'),
                num_train_epochs=num_epochs if max_steps == -1 else 1,
                max_steps=max_steps,
                per_device_train_batch_size=self._get_config_value(self.config.train, 'per_device_batch_size', default=4),
                per_device_eval_batch_size=self._get_config_value(self.config.train, 'per_device_eval_batch_size', default=4),
                gradient_accumulation_steps=self._get_config_value(self.config.train, 'gradient_accumulation_steps', default=1),
                learning_rate=optim_scheduler['learning_rate'],
                lr_scheduler_type=optim_scheduler['lr_scheduler_type'],
                warmup_steps=optim_scheduler['warmup_steps'],
                warmup_ratio=optim_scheduler['warmup_ratio'],
                optim=optim_scheduler['optimizer_type'],
                weight_decay=self._get_config_value(self.config.train, 'weight_decay', default=0.0),
                max_grad_norm=self._get_config_value(self.config.train, 'max_grad_norm', default=1.0),
                logging_steps=self._get_config_value(self.config.train, 'logging_steps', default=10),
                logging_strategy=self._get_config_value(self.config.train, 'logging_strategy', default='steps'),
                eval_strategy=eval_strategy,
                eval_steps=self._get_config_value(self.config.train, 'eval_steps', 'eval_interval', default=100) if eval_strategy == 'steps' else None,
                save_steps=self._get_config_value(self.config.train, 'save_steps', 'save_interval', default=100),
                save_strategy=save_strategy,
                save_total_limit=self._get_config_value(self.config.train, 'save_total_limit', default=None),
                seed=self._get_config_value(self.config.train, 'seed', default=42),
                data_seed=self._get_config_value(self.config.train, 'data_seed', default=47),
                gradient_checkpointing=self._get_config_value(self.config.train, 'use_gradient_checkpointing', 'gradient_checkpointing', default=True),
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=self.config.logging.loggers if self.config.logging.loggers else [],
                metric_for_best_model=self._get_config_value(self.config.train, 'metric_for_best_model', default='eval_loss') if eval_strategy != 'no' else None,
                greater_is_better=self._get_config_value(self.config.train, 'greater_is_better', default=False),
                load_best_model_at_end=self._get_config_value(self.config.train, 'load_best_model_at_end', default=True) if eval_strategy != 'no' else False,
                # DPO-specific parameters
                beta=self._get_config_value(self.config.train, 'beta', default=0.1),
                loss_type=self._get_config_value(self.config.train, 'loss_type', default='sigmoid'),
                label_smoothing=self._get_config_value(self.config.train, 'label_smoothing', default=0.0),
                max_length=self._get_config_value(self.config.train, 'max_length', default=self._get_config_value(self.config.model, 'max_seq_length', default=512)),
                truncation_mode=self._get_config_value(self.config.train, 'truncation_mode', default='keep_end'),
            )
            
            missing = extract_extra_and_missing_params(
                backend_config=dpo_config,
                config=self.config,
                algorithm='dpo'
            )

            for key, value in missing.items():
                setattr(dpo_config, key, value)

            # An unset generic RL value must not erase TRL's DPO default.
            if not dpo_config.loss_type:
                dpo_config.loss_type = 'sigmoid'

            # Create trainer
            self.trainer = DPOTrainer(
                model=self.model,
                ref_model=self.reference_model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=dpo_config,
            )

            logger.info("TRL DPOTrainer setup completed")

        except Exception as e:
            logger.error(f"Failed to setup DPO trainer: {e}")
            raise

    def train(self) -> Dict[str, Any]:
        """Run DPO training."""
        try:
            logger.info("Starting Unsloth DPO training")

            # Setup components
            self.setup_model()
            self.setup_data()
            self.setup_trainer()

            # Run training
            training_result = self.trainer.train()

            # Save model
            model_path = self.save_model()

            logger.info("Unsloth DPO training completed successfully")

            return {
                "training_time": training_result.metrics.get(
                    "train_runtime",
                    0),
                "final_loss": training_result.metrics.get(
                    "train_loss",
                    0),
                "model_path": model_path,
                "total_steps": training_result.metrics.get(
                    "train_steps",
                    0)}

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise

    # def evaluate(self) -> Dict[str, Any]:
    #     """Evaluate the trained model."""
    #     try:
    #         if not self.trainer or not self.eval_dataset:
    #             logger.warning("No trainer or evaluation dataset available")
    #             return {}

    #         logger.info("Running DPO evaluation")

    #         # Run evaluation
    #         eval_result = self.trainer.evaluate()

    #         logger.info("DPO evaluation completed")

    #         return {
    #             "eval_loss": eval_result.get("eval_loss", 0),
    #             "eval_metrics": eval_result
    #         }

    #     except Exception as e:
    #         logger.error(f"DPO evaluation failed: {e}")
    #         return {}
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

    def generate_preference_samples(
            self, num_samples: int = 5) -> List[Dict[str, str]]:
        """Generate sample preference data for testing."""
        try:
            if not self.tokenizer:
                logger.warning("No tokenizer available for generation")
                return []

            # Sample prompts
            prompts = [
                "Explain the concept of machine learning",
                "Write a short story about a robot",
                "Describe the benefits of renewable energy",
                "What are the key principles of good software design?",
                "Explain quantum computing in simple terms"
            ]

            samples = []
            for i, prompt in enumerate(prompts[:num_samples]):
                samples.append({
                    "prompt": prompt,
                    "chosen": f"Sample chosen response {i + 1}",
                    "rejected": f"Sample rejected response {i + 1}"
                })

            return samples

        except Exception as e:
            logger.error(f"Failed to generate preference samples: {e}")
            return []
