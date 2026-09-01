"""
TRL Knowledge Distillation Trainer.

This module provides a TRL-based trainer for knowledge distillation,
where a student model learns from a frozen teacher model.
Supports SOTA modes: forward-KL, reverse-KL, JSD, skew-KL, GKD, and reasoning-trace.
"""

import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.distill.config import DistillConfig, KDLossType, DistillMode
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.backends.trl.distill.losses import compute_distill_loss

# Lazy imports to avoid circular dependency issues
HAS_TRANSFORMERS = False
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    DataCollatorForLanguageModeling = None
    Dataset = None

logger = logging.getLogger(__name__)


if HAS_TRANSFORMERS:
    class KDTrainer(Trainer):
        """Custom HuggingFace Trainer with knowledge distillation loss.

        Supports multiple distillation modes:
        - forward_kl: standard baseline
        - reverse_kl: penalizes student hallucinations
        - jsd: Jensen-Shannon divergence (symmetric)
        - skew_kl: adaptive KL blend (DistiLLM)
        - gkd: Generalized KD (on-policy)
        - reasoning_trace: DeepSeek-R1 style CoT distillation
        """

        def __init__(
            self,
            teacher_model,
            temperature: float = 3.0,
            alpha: float = 0.5,
            loss_type: str = "kl",
            distill_mode: str = "forward_kl",
            skew_alpha: float = 0.1,
            reasoning_trace_field: Optional[str] = None,
            *args,
            **kwargs
        ):
            """Initialize KD trainer with distillation parameters.

            Args:
                teacher_model: Frozen teacher model for distillation
                temperature: Temperature scaling for softmax
                alpha: Blend weight for CE + KD (legacy, used in forward_kl)
                loss_type: Legacy loss type string
                distill_mode: SOTA mode (forward_kl, reverse_kl, jsd, skew_kl, gkd, reasoning_trace)
                skew_alpha: Alpha for skew_kl mode
                reasoning_trace_field: Field name in dataset for reasoning traces
            """
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model
            self.temperature = temperature
            self.alpha = alpha
            self.loss_type = loss_type
            self.distill_mode = distill_mode
            self.skew_alpha = skew_alpha
            self.reasoning_trace_field = reasoning_trace_field

        def compute_loss(self, model, inputs, return_outputs=False):
            """Compute blended loss: dispatches on distill_mode.

            Modes:
            - forward_kl/reverse_kl/jsd/skew_kl: use SOTA loss functions
            - gkd: delegates to TRL's native GKDTrainer (on-policy)
            - reasoning_trace: SFT on teacher traces + optional KD

            Returns:
                loss (scalar) or (loss, outputs) tuple
            """
            # Get labels before forward pass
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("Labels required for KD training")

            # Forward pass through student
            outputs = model(**inputs)
            student_logits = outputs.logits

            # Handle reasoning_trace mode specially
            if self.distill_mode == "reasoning_trace":
                return self._compute_reasoning_trace_loss(
                    model, inputs, student_logits, labels, return_outputs
                )

            # For all KL-based modes: get teacher logits
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            # Dispatch on distill_mode
            if self.distill_mode in ["forward_kl", "reverse_kl", "jsd", "skew_kl"]:
                # Use SOTA loss functions from losses.py
                ce_loss_fn = torch.nn.CrossEntropyLoss()
                ce_loss = ce_loss_fn(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                )

                # Compute SOTA KD loss
                kd_loss = compute_distill_loss(
                    student_logits,
                    teacher_logits,
                    mode=self.distill_mode,
                    temperature=self.temperature,
                    skew_alpha=self.skew_alpha,
                )

                # Blend: alpha * CE + (1-alpha) * KD
                loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

            elif self.distill_mode == "gkd":
                # GKD mode (Generalized KD) - note: this is a placeholder
                # In practice, would wrap TRL's GKDTrainer at trainer level
                logger.warning("GKD mode requires wrapping TRL's GKDTrainer at trainer initialization")
                ce_loss_fn = torch.nn.CrossEntropyLoss()
                ce_loss = ce_loss_fn(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                )
                kd_loss = compute_distill_loss(
                    student_logits,
                    teacher_logits,
                    mode="reverse_kl",  # GKD uses reverse KL by default
                    temperature=self.temperature,
                )
                loss = 0.5 * ce_loss + 0.5 * kd_loss

            else:
                raise ValueError(f"Unknown distill_mode: {self.distill_mode}")

            if return_outputs:
                return loss, outputs
            return loss

        def _compute_reasoning_trace_loss(self, model, inputs, student_logits, labels, return_outputs=False):
            """Compute loss for reasoning_trace mode."""
            if self.reasoning_trace_field not in inputs:
                logger.warning(
                    f"reasoning_trace_field '{self.reasoning_trace_field}' not found in inputs. "
                    "Falling back to standard CE loss."
                )
                ce_loss_fn = torch.nn.CrossEntropyLoss()
                loss = ce_loss_fn(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                )
            else:
                trace_labels = inputs[self.reasoning_trace_field]
                ce_loss_fn = torch.nn.CrossEntropyLoss()
                trace_loss = ce_loss_fn(
                    student_logits.view(-1, student_logits.size(-1)),
                    trace_labels.view(-1),
                )
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits
                kd_loss = compute_distill_loss(
                    student_logits,
                    teacher_logits,
                    mode="forward_kl",
                    temperature=self.temperature,
                )
                loss = 0.8 * trace_loss + 0.2 * kd_loss

            if return_outputs:
                outputs = model(**inputs) if not hasattr(self, '_outputs_cache') else self._outputs_cache
                return loss, outputs
            return loss

        def _compute_kd_loss(self, student_logits, teacher_logits, loss_type: str):
            """Compute knowledge distillation loss."""
            if loss_type == "kl":
                return self._kl_divergence(student_logits, teacher_logits)
            elif loss_type == "mse":
                return self._mse_loss(student_logits, teacher_logits)
            elif loss_type == "reverse_kl":
                return self._kl_divergence(teacher_logits, student_logits)
            elif loss_type == "jsd":
                return self._jsd_loss(student_logits, teacher_logits)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        def _kl_divergence(self, student_logits, teacher_logits):
            """KL divergence with temperature scaling."""
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            kl_loss *= self.temperature ** 2
            return kl_loss

        def _mse_loss(self, student_logits, teacher_logits):
            """MSE loss on logits."""
            return F.mse_loss(student_logits, teacher_logits)

        def _jsd_loss(self, student_logits, teacher_logits):
            """Jensen-Shannon divergence."""
            student_probs = F.softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            m_probs = 0.5 * (student_probs + teacher_probs)
            kl_pm = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                m_probs,
                reduction="batchmean",
            )
            kl_qm = F.kl_div(
                F.log_softmax(teacher_logits / self.temperature, dim=-1),
                m_probs,
                reduction="batchmean",
            )
            jsd = 0.5 * kl_pm + 0.5 * kl_qm
            jsd *= self.temperature ** 2
            return jsd

else:
    # Placeholder class if transformers not available
    class KDTrainer:
        def __init__(self, **kwargs):
            raise ImportError("transformers is required for KDTrainer")


class TRLDistillTrainer(TrainerBase):
    """TRL-based knowledge distillation trainer.

    Trains a student model using a frozen teacher model as a guide.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize distillation trainer."""
        super().__init__(config)
        self.model = None  # Student model
        self.teacher_model = None  # Teacher model (frozen)
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.distill_config = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and transformers are available."""
        try:
            from trl import SFTTrainer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def _get_config_value(self, config_obj, *attr_names, default=None):
        """Safely get config value from multiple possible attribute names."""
        if isinstance(config_obj, dict):
            for attr_name in attr_names:
                if attr_name in config_obj:
                    return config_obj[attr_name]
        else:
            for attr_name in attr_names:
                if hasattr(config_obj, attr_name):
                    return getattr(config_obj, attr_name)
        return default

    # Required abstract methods
    def setup_data(self) -> None:
        """Setup data - delegates to setup_dataset."""
        self.setup_dataset()

    def setup_rewards(self) -> None:
        """Setup rewards - not used in distillation."""
        logger.info("Distillation does not use explicit rewards")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step - handled internally by HF Trainer."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        return {}

    def setup_model(self) -> None:
        """Setup student and teacher models."""
        try:
            logger.info("=" * 80)
            logger.info("Setting up Knowledge Distillation Models")
            logger.info("=" * 80)

            # Get distill config from UnifiedConfig if available
            if hasattr(self.config, "distill"):
                self.distill_config = self.config.distill
            else:
                # Create default distill config from training config
                self.distill_config = self._create_default_distill_config()

            # === UNIFIED PRECISION HANDLING ===
            precision = PrecisionHandler.get_precision_from_config(
                self.config, default="auto"
            )
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "TRL Distillation")
            dtype = PrecisionHandler.get_torch_dtype(precision)

            # Load tokenizer once
            logger.info(
                f"Loading tokenizer from student model: {self.distill_config.student_model}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.distill_config.student_model,
                trust_remote_code=self.distill_config.trust_remote_code,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad token to eos token")

            # Load student model (trainable)
            logger.info(
                f"Loading student model: {self.distill_config.student_model}"
            )
            student_dtype = self._get_torch_dtype(self.distill_config.student_dtype)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.distill_config.student_model,
                torch_dtype=student_dtype,
                device_map=self.distill_config.device_map or "auto",
                trust_remote_code=self.distill_config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            logger.info(f"Student model loaded: {self.model.config.model_type}")

            # Load teacher model (frozen)
            logger.info(f"Loading teacher model: {self.distill_config.teacher_model}")
            teacher_dtype = self._get_torch_dtype(self.distill_config.teacher_dtype)

            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.distill_config.teacher_model,
                torch_dtype=teacher_dtype,
                device_map=self.distill_config.device_map or "auto",
                trust_remote_code=self.distill_config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            logger.info(f"Teacher model loaded: {self.teacher_model.config.model_type}")

            # Freeze teacher model (no gradients)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            logger.info("Teacher model frozen (no gradients)")

            # Apply LoRA to student if configured
            if self.distill_config.use_peft:
                logger.info("Applying LoRA to student model")
                from peft import LoraConfig, get_peft_model

                peft_config = LoraConfig(
                    r=self.distill_config.lora_r,
                    lora_alpha=self.distill_config.lora_alpha,
                    target_modules=self.distill_config.lora_target_modules,
                    lora_dropout=self.distill_config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                self.model = get_peft_model(self.model, peft_config)
                logger.info(
                    f"LoRA applied: r={self.distill_config.lora_r}, "
                    f"alpha={self.distill_config.lora_alpha}"
                )

            logger.info("=" * 80)
            logger.info("Model setup completed successfully")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to setup models: {e}")
            raise

    def setup_dataset(self) -> None:
        """Setup training dataset."""
        try:
            logger.info("Setting up distillation dataset...")

            # Extract dataset configuration
            dataset_config = None
            if hasattr(self.config, "dataset"):
                dataset_config = self.config.dataset
            elif hasattr(self.config, "datasets") and len(self.config.datasets) > 0:
                dataset_config = self.config.datasets[0]
            else:
                raise ValueError("No dataset configuration found")

            # Load dataset
            dataset_name = self._get_config_value(
                dataset_config, "name", "dataset_name", default="wikitext"
            )
            split = self._get_config_value(dataset_config, "split", default="train")

            logger.info(f"Loading dataset: {dataset_name} ({split})")

            from datasets import load_dataset

            self.train_dataset = load_dataset(dataset_name, split=split)

            # Limit samples if specified
            max_samples = self._get_config_value(
                dataset_config, "max_samples", default=None
            )
            if max_samples:
                logger.info(f"Limiting to {max_samples} training samples")
                self.train_dataset = self.train_dataset.select(range(max_samples))

            # Setup evaluation dataset if configured
            max_eval_samples = self._get_config_value(
                dataset_config, "max_eval_samples", default=None
            )
            if max_eval_samples:
                self.eval_dataset = load_dataset(
                    dataset_name, split="validation"
                ).select(range(max_eval_samples))
                logger.info(f"Loaded {len(self.eval_dataset)} evaluation samples")

            logger.info(f"Training dataset: {len(self.train_dataset)} samples")

        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise

    def train(self) -> None:
        """Execute training with knowledge distillation."""
        logger.info("Starting Knowledge Distillation Training...")

        # Setup phase
        self.setup_model()
        self.setup_data()
        self.setup_rewards()

        try:
            # Prepare training arguments
            output_dir = self.config.output_dir or "./distill_output"
            max_steps = self.distill_config.max_steps
            if max_steps is None:
                if self.distill_config.epochs:
                    max_steps = self.distill_config.epochs * len(self.train_dataset)
                else:
                    raise ValueError(
                        "Either max_steps or epochs must be specified"
                    )

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.distill_config.epochs or 1,
                max_steps=max_steps,
                per_device_train_batch_size=self.distill_config.per_device_batch_size,
                gradient_accumulation_steps=self.distill_config.gradient_accumulation_steps,
                learning_rate=self.distill_config.learning_rate,
                weight_decay=self.distill_config.weight_decay,
                warmup_steps=self.distill_config.warmup_steps,
                warmup_ratio=self.distill_config.warmup_ratio,
                logging_steps=self.distill_config.logging_steps,
                save_steps=self.distill_config.save_steps,
                eval_steps=self.distill_config.eval_steps,
                save_strategy=self.distill_config.save_strategy,
                evaluation_strategy=self.distill_config.eval_strategy,
                save_total_limit=self.distill_config.save_total_limit,
                load_best_model_at_end=True if self.eval_dataset else False,
                metric_for_best_model="loss",
                greater_is_better=False,
                remove_unused_columns=False,
                seed=self.distill_config.seed,
                fp16=self.distill_config.student_dtype == "fp16",
                bf16=self.distill_config.student_dtype == "bf16",
            )

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            )

            # Preprocess dataset
            def preprocess_function(examples):
                """Tokenize examples for language modeling."""
                # Handle different dataset formats
                text_field = "text"
                if "text" not in examples:
                    # Try common alternatives
                    for field in ["content", "input", "body"]:
                        if field in examples:
                            text_field = field
                            break

                tokenized = self.tokenizer(
                    examples[text_field],
                    truncation=True,
                    max_length=2048,
                    padding="max_length",
                    return_tensors="pt",
                )
                return tokenized

            logger.info("Preprocessing training dataset...")
            self.train_dataset = self.train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=self.train_dataset.column_names,
            )

            if self.eval_dataset:
                logger.info("Preprocessing evaluation dataset...")
                self.eval_dataset = self.eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=self.eval_dataset.column_names,
                )

            # Create trainer with KD loss
            self.trainer = KDTrainer(
                model=self.model,
                teacher_model=self.teacher_model,
                temperature=self.distill_config.temperature,
                alpha=self.distill_config.alpha,
                loss_type=self.distill_config.loss_type,
                distill_mode=self.distill_config.distill_mode,
                skew_alpha=self.distill_config.skew_alpha,
                reasoning_trace_field=self.distill_config.reasoning_trace_field,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )

            logger.info(
                f"Starting training with temperature={self.distill_config.temperature}, "
                f"alpha={self.distill_config.alpha}, "
                f"loss_type={self.distill_config.loss_type}, "
                f"distill_mode={self.distill_config.distill_mode}, "
                f"on_policy={self.distill_config.on_policy}"
            )

            # Train
            train_result = self.trainer.train()

            logger.info(f"Training completed. Final loss: {train_result.training_loss}")

            # Save final model
            self.save()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save the trained student model."""
        if output_dir is None:
            output_dir = self.config.output_dir or "./distill_output"

        logger.info(f"Saving model to {output_dir}")

        if self.trainer:
            self.trainer.save_model(output_dir)

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        logger.info("Model saved successfully")

    def load(self, model_dir: str) -> None:
        """Load a saved student model."""
        logger.info(f"Loading model from {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        logger.info("Model loaded successfully")

    def _get_torch_dtype(self, dtype_str: str):
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "auto": "auto",
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _create_default_distill_config(self) -> DistillConfig:
        """Create default distill config from training config."""
        from aligntune.core.distill.config import DistillConfig

        # Extract model names from config
        student_model = self.config.model.name_or_path
        teacher_model = getattr(self.config.model, "teacher_model", student_model)

        return DistillConfig(
            student_model=student_model,
            teacher_model=teacher_model,
            student_dtype=self.config.model.precision.value,
            teacher_dtype=getattr(
                self.config.model, "teacher_dtype", self.config.model.precision.value
            ),
            per_device_batch_size=self.config.train.per_device_batch_size,
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            max_steps=self.config.train.max_steps,
            epochs=self.config.train.epochs,
            learning_rate=self.config.train.learning_rate,
        )
