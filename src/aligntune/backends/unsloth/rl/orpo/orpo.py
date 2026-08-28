"""
Unsloth ORPO Trainer - Complete Implementation

This module provides a complete Unsloth-based ORPO (Odds Ratio Preference Optimization) trainer
that works with the backend factory system.
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
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class UnslothORPOTrainer(TrainerBase):
    """Unsloth-based ORPO trainer using TRL's ORPOTrainer."""

    TASK_TYPE = "dpo"  # ORPO uses the shared preference schema
    KEEP_COLUMNS = False

    def __init__(self, config: UnifiedConfig):
        """Initialize Unsloth ORPO trainer."""
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset_cache = None
        self.dataset_dict = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth ORPO trainer is available."""
        try:
            from trl.experimental.orpo import ORPOTrainer, ORPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def setup_rewards(self) -> None:
        """Setup reward functions (handled by TRL trainer)."""
        pass

    def train_step(self, batch):
        """Train step (handled by TRL trainer)."""
        pass

    def setup_model(self) -> None:
        """Setup models natively via model_loader."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info(f"Setting up models via model_loader")
        logger.info("=" * 80)
        
        from aligntune.core.registry import TaskType
        from aligntune.core.model_loader import build_model
        import copy

        peft_enabled = getattr(self.config.model, 'use_peft', False) or \
                       getattr(self.config.model, 'load_in_4bit', False) or \
                       getattr(self.config.model, 'load_in_8bit', False)
        
        self.model, self.tokenizer = build_model(
            self.config, task_type=TaskType.SFT, apply_peft=peft_enabled, use_unsloth=True
        )
        
        if getattr(self, "needs_ref_model", False):
            if not peft_enabled:
                logger.info("Full fine-tuning: Loading frozen reference model")
                self.ref_model, _ = build_model(
                    self.config, task_type=TaskType.SFT, is_reference=True, use_unsloth=True
                )
                self.reference_model = self.ref_model
            else:
                logger.info("PEFT enabled: TRL handles reference model via adapter toggle.")
                self.ref_model = None
                self.reference_model = None

    # setup_data() inherited from TrainerBase - uses TASK_TYPE="orpo"
    # DataFilter in DataManager handles all validation (empty strings, None values, length checks)

    def setup_trainer(self) -> None:
        """Setup Unsloth ORPOTrainer."""
        try:
            from trl.experimental.orpo import ORPOTrainer, ORPOConfig

            logger.info("Setting up Unsloth ORPOTrainer")

            # Setup optimizer and scheduler using base class method
            optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

            max_steps = optim_scheduler['max_steps']
            num_epochs = optim_scheduler['num_epochs']
            warmup_steps = optim_scheduler['warmup_steps']
            warmup_ratio = optim_scheduler['warmup_ratio']
            optimizer_type = optim_scheduler['optimizer_type']
            lr_scheduler_type = optim_scheduler['lr_scheduler_type']
            learning_rate = optim_scheduler['learning_rate']

            precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
            precision_args = PrecisionHandler.get_training_args_precision(precision)

            # Evaluation parameters
            explicit_params = getattr(self.config.train, 'extra_params', {}) or {}
            eval_strategy = explicit_params.get('eval_strategy')
            eval_steps = self._get_config_value(self.config.train, 'eval_steps', default=100)
            save_steps = self._get_config_value(self.config.train, 'save_steps', default=100)
            save_strategy = explicit_params.get('save_strategy')
            save_total_limit = self._get_config_value(self.config.train, 'save_total_limit', default=None)
            load_best_model_at_end = self._get_config_value(self.config.train, 'load_best_model_at_end', default=True if self.eval_dataset else False)
            metric_for_best_model = self._get_config_value(self.config.train, 'metric_for_best_model', default='eval_loss' if self.eval_dataset else None)
            greater_is_better = self._get_config_value(self.config.train, 'greater_is_better', default=False)

            logging_steps = self._get_config_value(self.config.train, 'logging_steps', default=10)
            logging_strategy = self._get_config_value(self.config.train, 'logging_strategy', default='steps')
            report_to = self._get_config_value(self.config.logging, 'report_to', default=None)

            if not self.eval_dataset:
                eval_strategy = 'no'
                eval_steps = None
                save_strategy = 'no'
            else:
                eval_strategy = eval_strategy or 'no'
                save_strategy = save_strategy or ('steps' if eval_strategy != 'no' else 'no')

            orpo_config = ORPOConfig(
                output_dir=self.config.logging.output_dir,
                num_train_epochs=num_epochs if max_steps == -1 else 1,
                max_steps=max_steps,
                per_device_train_batch_size=self.config.train.per_device_batch_size,
                per_device_eval_batch_size=self._get_config_value(
                    self.config.train, 'per_device_eval_batch_size',
                    default=self.config.train.per_device_batch_size,
                ),
                gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,

                eval_strategy=eval_strategy,
                eval_steps=eval_steps,

                logging_strategy=logging_strategy,
                logging_steps=logging_steps,

                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,

                lr_scheduler_type=lr_scheduler_type,
                optim=optimizer_type,
                weight_decay=self._get_config_value(self.config.train, 'weight_decay', default=0.01),

                **precision_args,

                dataloader_pin_memory=False,
                remove_unused_columns=False,

                # ORPO-specific parameters
                max_length=self.config.model.max_seq_length,

                report_to=report_to if report_to else (self.config.logging.loggers if self.config.logging.loggers else []),
            )

            missing = extract_extra_and_missing_params(
                backend_config=orpo_config,
                config=self.config,
                algorithm='orpo'
            )

            for key, value in missing.items():
                setattr(orpo_config, key, value)

            # Unsloth silently truncates a forward pass's returned logits to
            # its configured max_seq_length ("We shall truncate it ourselves")
            # but ORPOTrainer.concatenated_forward builds `labels` directly
            # from the untruncated concatenated_input_ids, so the two only
            # mismatch once a chosen+rejected concatenated batch exceeds
            # max_seq_length - which happens far more often here than in
            # single-sequence trainers, since ORPO/CPO concatenate both
            # sequences into one forward pass. Patch to truncate labels to
            # whatever length Unsloth actually returned.
            if not getattr(ORPOTrainer, "_aligntune_length_patched", False):
                _orig_concatenated_forward = ORPOTrainer.concatenated_forward

                def _patched_concatenated_forward(self, model, batch):
                    concatenated_batch = self.concatenated_inputs(
                        batch,
                        is_encoder_decoder=self.is_encoder_decoder,
                        padding_value=self.padding_value,
                        device=self.accelerator.device,
                    )
                    len_chosen = batch["chosen_labels"].shape[0]

                    model_kwargs = (
                        {"decoder_input_ids": self._shift_right(concatenated_batch["concatenated_labels"])}
                        if self.is_encoder_decoder else {}
                    )
                    if self.aux_loss_enabled:
                        model_kwargs["output_router_logits"] = True

                    outputs = model(
                        concatenated_batch["concatenated_input_ids"],
                        attention_mask=concatenated_batch["concatenated_attention_mask"],
                        use_cache=False,
                        **model_kwargs,
                    )
                    all_logits = outputs.logits

                    if self.is_encoder_decoder:
                        labels = concatenated_batch["concatenated_labels"].clone()
                    else:
                        labels = concatenated_batch["concatenated_input_ids"].clone()
                        attention_mask = concatenated_batch["concatenated_attention_mask"]
                        labels = torch.where(attention_mask == 1, labels, -100)

                    # Realign labels to Unsloth's actually-returned logits
                    # length whenever it silently truncated the forward pass.
                    if all_logits.shape[1] != labels.shape[1]:
                        labels = labels[:, : all_logits.shape[1]]
                        concatenated_labels_for_logps = concatenated_batch["concatenated_labels"][:, : all_logits.shape[1]]
                    else:
                        concatenated_labels_for_logps = concatenated_batch["concatenated_labels"]

                    def cross_entropy_loss(logits, lbls):
                        if not self.is_encoder_decoder:
                            logits = logits[..., :-1, :].contiguous()
                            lbls = lbls[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss()
                        logits = logits.reshape(-1, logits.shape[-1])
                        lbls = lbls.reshape(-1).to(logits.device)
                        return loss_fct(logits, lbls)

                    chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

                    all_logps = self.get_batch_logps(
                        all_logits,
                        concatenated_labels_for_logps,
                        average_log_prob=True,
                        is_encoder_decoder=self.is_encoder_decoder,
                    )

                    chosen_logps = all_logps[:len_chosen]
                    rejected_logps = all_logps[len_chosen:]

                    if not self.is_encoder_decoder:
                        chosen_logits = all_logits[:len_chosen, :-1, :]
                        rejected_logits = all_logits[len_chosen:, :-1, :]
                    else:
                        chosen_logits = all_logits[:len_chosen]
                        rejected_logits = all_logits[len_chosen:]

                    if self.aux_loss_enabled:
                        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss, outputs.aux_loss)
                    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)

                ORPOTrainer.concatenated_forward = _patched_concatenated_forward
                ORPOTrainer._aligntune_length_patched = True

            # Create trainer
            self.trainer = ORPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=orpo_config,
            )

            logger.info("Unsloth ORPOTrainer setup completed")

        except Exception as e:
            logger.error(f"Failed to setup ORPO trainer: {e}")
            raise

    def train(self) -> Dict[str, Any]:
        """Run ORPO training."""
        try:
            logger.info("Starting Unsloth ORPO training")

            self.setup_model()
            self.setup_data()
            self.setup_trainer()

            training_result = self.trainer.train()

            model_path = self.save_model()

            logger.info("Unsloth ORPO training completed successfully")

            return {
                "training_time": training_result.metrics.get("train_runtime", 0),
                "final_loss": training_result.metrics.get("train_loss", 0),
                "model_path": model_path,
                "total_steps": training_result.metrics.get("train_steps", 0)
            }

        except Exception as e:
            logger.error(f"ORPO training failed: {e}")
            raise
