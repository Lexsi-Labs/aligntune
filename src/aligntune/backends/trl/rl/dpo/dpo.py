"""
TRL DPO Trainer - Complete Implementation

This module provides a complete TRL-based Direct Preference Optimization trainer
that works with the backend factory system.
"""

import logging
from typing import Dict, Any

from aligntune.core.rl import TrainerBase
from aligntune.core.registry import TaskType
from aligntune.core.model_loader import build_model
from aligntune.core.rl.config import UnifiedConfig
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class TRLDPOTrainer(TrainerBase):
    """TRL-based DPO trainer using TRL's DPOTrainer."""
    TASK_TYPE = "dpo"
    KEEP_COLUMNS = False
    
    def __init__(self, config: UnifiedConfig):
        """Initialize TRL DPO trainer."""
        super().__init__(config)
        self.model = None
        self.reference_model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset_cache = None
        self.dataset_dict = None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL DPO trainer is available."""
        try:
            from trl import DPOTrainer, DPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False
    # Required abstract methods
    def setup_model(self) -> None:
        """Setup policy and reference models natively via model_loader."""
        logger.info("=" * 80)
        logger.info("Setting up TRL DPO models via model_loader")
        logger.info("=" * 80)
        
        peft_enabled = getattr(self.config.model, 'use_peft', False) or \
                       getattr(self.config.model, 'load_in_4bit', False) or \
                       getattr(self.config.model, 'load_in_8bit', False)
        
        self.model, self.tokenizer = build_model(
            self.config, task_type=TaskType.SFT, apply_peft=peft_enabled
        )
        
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

    def setup_rewards(self) -> None:
        """DPO uses preference pairs, not explicit reward functions."""
        logger.info("DPO uses preference pairs instead of explicit rewards")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Handled internally by TRL DPOTrainer."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        return {}

    def setup_trainer(self) -> None:
        """Setup TRL DPOTrainer with DPO-specific config."""
        try:
            from trl import DPOTrainer, DPOConfig

            logger.info("Setting up TRL DPOTrainer")

            # Get optimizer, scheduler, and training params from base class
            optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

            # Extract values from returned dict
            max_steps = optim_scheduler['max_steps']
            num_epochs = optim_scheduler['num_epochs']

            # Build DPOConfig with common and algorithm-specific params
            explicit_params = getattr(self.config.train, 'extra_params', {}) or {}
            eval_strategy = explicit_params.get('eval_strategy')
            save_strategy = explicit_params.get('save_strategy')

            dpo_kwargs = dict(
                output_dir=self._get_config_value(self.config.logging, 'output_dir', default='./output'),
                per_device_train_batch_size=self._get_config_value(self.config.train, 'per_device_batch_size', default=4),
                per_device_eval_batch_size=self._get_config_value(self.config.train, 'per_device_eval_batch_size', default=4),
                num_train_epochs=num_epochs if max_steps == -1 else 1,
                max_steps=max_steps,
                learning_rate=optim_scheduler['learning_rate'],
                lr_scheduler_type=optim_scheduler['lr_scheduler_type'],
                warmup_steps=optim_scheduler['warmup_steps'],
                warmup_ratio=optim_scheduler['warmup_ratio'],
                optim=optim_scheduler['optimizer_type'],
                logging_steps=self._get_config_value(self.config.train, 'logging_steps', default=10),
                eval_steps=self._get_config_value(self.config.train, 'eval_steps', 'eval_interval', default=100),
                save_steps=self._get_config_value(self.config.train, 'save_steps', 'save_interval', default=100),
                seed=self._get_config_value(self.config, 'seed', default=42),
                # DPO-specific parameters
                beta=self._get_config_value(self.config.train, 'beta', default=0.1),
                # trl's DPOConfig.loss_type is a list[str] (supports combining
                # multiple weighted loss terms) - wrap a plain string config value.
                loss_type=(lambda lt: lt if isinstance(lt, list) else [lt])(
                    self._get_config_value(self.config.train, 'loss_type', default='sigmoid')
                ),
                label_smoothing=self._get_config_value(self.config.train, 'label_smoothing', default=0.0),
            )
            if eval_strategy is not None:
                dpo_kwargs['eval_strategy'] = eval_strategy
            if save_strategy is not None:
                dpo_kwargs['save_strategy'] = save_strategy
            dpo_config = DPOConfig(**dpo_kwargs)

            # Backfill anything else the caller set (e.g. gradient_accumulation_steps,
            # weight_decay, max_grad_norm, seed, precompute_ref_log_probs,
            # f_divergence_type, sync_ref_model, ...) that the explicit list above
            # doesn't already cover, without hand-listing every DPOConfig field here.
            missing = extract_extra_and_missing_params(
                backend_config=dpo_config, config=self.config, algorithm='dpo'
            )
            for key, value in missing.items():
                setattr(dpo_config, key, value)

            # The backfill above may re-flatten loss_type back to a bare string
            # (DPOConfig's own default_factory also yields ['sigmoid'], so the
            # "already explicitly set" heuristic misses it) - re-wrap it.
            if not isinstance(dpo_config.loss_type, list):
                dpo_config.loss_type = [dpo_config.loss_type]

            # Create trainer
            self.trainer = DPOTrainer(
                model=self.model,
                ref_model=self.reference_model,
                processing_class=self.tokenizer,
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
            logger.info("Starting TRL DPO training")
            
            # Setup components
            self.setup_model()
            self.setup_data()
            self.setup_trainer()
            
            # Add the callbacks to the trainer
            for cb in self.get_hf_callbacks():
                self.trainer.add_callback(cb)
            
            # Run training
            training_result = self.trainer.train()
            
            # Save model
            model_path = self.save_model()
            
            logger.info("TRL DPO training completed successfully")
            
            return {
                "training_time": training_result.metrics.get("train_runtime", 0),
                "final_loss": training_result.metrics.get("train_loss", 0),
                "model_path": model_path,
                "total_steps": training_result.metrics.get("train_steps", 0)
            }
            
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise
