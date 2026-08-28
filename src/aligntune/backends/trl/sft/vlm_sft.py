"""
TRL VLM SFT Backend Implementation - Vision Language Model Supervised Fine-Tuning.

This module provides VLM SFT training using TRL's native VLM support (>=0.12).
Supports three vision tower training modes:
- freeze: Vision tower parameters are frozen (LoRA only on LM)
- lora: LoRA adapters on both vision tower and language model
- full: Full fine-tuning of vision tower and language model
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

from ....core.sft.trainer_base import SFTTrainerBase
from ....core.sft.config import TaskType
from ....core.precision_handler import PrecisionHandler

logger = logging.getLogger(__name__)


class TRLVLMSFTTrainer(SFTTrainerBase):
    """VLM SFT trainer using TRL's native VLM support."""

    SUPPORTED_TASKS = [TaskType.VLM_SFT]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_type = TaskType.VLM_SFT
        self.model = None
        self.processor = None
        self.trainer = None
        self.dataset = None
        self.eval_dataset = None
        self.training_history = []

        logger.info("Initialized TRLVLMSFTTrainer for VLM SFT")

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL VLM support is available."""
        try:
            from trl import SFTTrainer
            from transformers import AutoProcessor, AutoModelForVision2Seq

            return True
        except ImportError:
            return False

    def setup_processor(self) -> None:
        """Setup image-text processor for VLM."""
        logger.info(f"Loading processor for: {self.config.model.name_or_path}")

        try:
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError("Transformers not available.") from e

        self.processor = AutoProcessor.from_pretrained(
            self.config.model.name_or_path,
            trust_remote_code=True,
        )

        logger.info(f"Processor loaded: {type(self.processor).__name__}")

    def setup_model(self) -> None:
        """Setup VLM model with vision tower training mode."""
        logger.info("=" * 80)
        logger.info(f"Setting up VLM SFT model: {self.config.model.name_or_path}")
        logger.info(f"Vision Tower Mode: {self.config.model.vision_tower_mode}")
        logger.info("=" * 80)

        # Precision handling
        precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "VLM SFT")
        dtype = PrecisionHandler.get_torch_dtype(precision)

        try:
            from transformers import AutoModelForVision2Seq
        except ImportError as e:
            raise ImportError("AutoModelForVision2Seq not available. Ensure transformers>=4.36.0") from e

        # Load VLM model
        logger.info("Loading Vision Language Model...")

        quantization_config = None
        quantization_dict = getattr(self.config.model, "quantization", {})

        if quantization_dict.get("load_in_4bit", False) or quantization_dict.get("load_in_8bit", False):
            try:
                from transformers import BitsAndBytesConfig

                load_in_4bit = quantization_dict.get("load_in_4bit", False)
                load_in_8bit = quantization_dict.get("load_in_8bit", False)

                if load_in_4bit:
                    compute_dtype_str = quantization_dict.get("bnb_4bit_compute_dtype", "bfloat16")
                    compute_dtype = torch.bfloat16 if compute_dtype_str == "bfloat16" else torch.float16

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=quantization_dict.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_use_double_quant=quantization_dict.get("bnb_4bit_use_double_quant", False),
                    )
                    logger.info("✅ 4-bit quantization enabled")
                elif load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("✅ 8-bit quantization enabled")
            except ImportError:
                logger.warning("⚠️  BitsAndBytes not available")
                quantization_config = None

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model.name_or_path,
            **model_kwargs
        )

        # Setup vision tower training mode
        self._setup_vision_tower_mode()

        logger.info("=" * 80)
        logger.info("VLM SFT model setup completed successfully")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info("=" * 80)

    def _setup_vision_tower_mode(self) -> None:
        """Configure vision tower training based on mode setting."""
        vision_tower_mode = getattr(self.config.model, "vision_tower_mode", "freeze")

        logger.info(f"Configuring vision tower: {vision_tower_mode}")

        # Get vision tower (model-specific, but commonly 'vision_model')
        vision_tower = None
        if hasattr(self.model, "vision_model"):
            vision_tower = self.model.vision_model
        elif hasattr(self.model, "vision_tower"):
            vision_tower = self.model.vision_tower
        else:
            logger.warning("Could not locate vision tower in model. Assuming defaults.")
            return

        if vision_tower_mode == "freeze":
            # Freeze vision tower completely
            for param in vision_tower.parameters():
                param.requires_grad = False
            logger.info("✅ Vision tower frozen (no gradients)")

        elif vision_tower_mode == "lora":
            # Apply LoRA to vision tower
            try:
                from peft import LoraConfig, get_peft_model

                lora_r = getattr(self.config.model, "lora_rank", 16)
                lora_alpha = getattr(self.config.model, "lora_alpha", 32)
                lora_dropout = getattr(self.config.model, "lora_dropout", 0.1)

                # For vision transformer, typical target modules
                vision_lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj"],  # Common in vision transformers
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                # Apply LoRA to vision tower
                vision_tower = get_peft_model(vision_tower, vision_lora_config)
                self.model.vision_model = vision_tower if hasattr(self.model, "vision_model") else self.model.vision_tower

                logger.info(f"✅ Vision tower LoRA enabled (r={lora_r}, alpha={lora_alpha})")
            except ImportError:
                logger.warning("PEFT not available. Falling back to freeze mode.")
                for param in vision_tower.parameters():
                    param.requires_grad = False

        elif vision_tower_mode == "full":
            # Full fine-tuning (all parameters trainable)
            for param in vision_tower.parameters():
                param.requires_grad = True
            logger.info("✅ Vision tower fully trainable")

        else:
            raise ValueError(f"Invalid vision_tower_mode: {vision_tower_mode}")

    def setup_dataset(self) -> None:
        """Load and prepare VLM dataset."""
        logger.info("=" * 80)
        logger.info("Setting up VLM dataset")
        logger.info("=" * 80)

        try:
            from datasets import load_dataset
            from ....data.loaders.vlm_loader import VLMLoader
        except ImportError as e:
            raise ImportError("Dataset loading failed.") from e

        # Use VLMLoader for proper image-text pair handling
        loader = VLMLoader(
            name=self.config.dataset.name,
            config_name=self.config.dataset.subset or self.config.dataset.config,
            split=self.config.dataset.split,
            image_column=getattr(self.config.model, "image_column", "image"),
            text_column=self.config.dataset.text_column,
            max_samples=self.config.dataset.max_samples,
        )

        self.dataset = loader.load()
        logger.info(f"Loaded {len(self.dataset)} training samples")

        # Setup eval dataset if specified
        if hasattr(self.config.dataset, "eval_split") and self.config.dataset.eval_split:
            eval_loader = VLMLoader(
                name=self.config.dataset.name,
                config_name=self.config.dataset.subset or self.config.dataset.config,
                split=self.config.dataset.eval_split,
                image_column=getattr(self.config.model, "image_column", "image"),
                text_column=self.config.dataset.text_column,
                max_samples=getattr(self.config.dataset, "max_eval_samples", None),
            )
            self.eval_dataset = eval_loader.load()
            logger.info(f"Loaded {len(self.eval_dataset)} eval samples")

    def setup_trainer(self) -> None:
        """Setup TRL SFTTrainer for VLM."""
        logger.info("=" * 80)
        logger.info("Setting up TRL VLM SFT Trainer")
        logger.info("=" * 80)

        try:
            from trl import SFTTrainer, SFTConfig as TRLSFTConfig
            from transformers import TrainingArguments
        except ImportError as e:
            raise ImportError("TRL not available.") from e

        # Prepare training arguments
        training_kwargs = {
            "output_dir": self.config.logging.output_dir,
            "num_train_epochs": self.config.train.epochs or 3,
            "per_device_train_batch_size": self.config.train.per_device_batch_size,
            "per_device_eval_batch_size": getattr(self.config.train, "per_device_eval_batch_size", 4),
            "gradient_accumulation_steps": self.config.train.gradient_accumulation_steps,
            "learning_rate": self.config.train.learning_rate,
            "weight_decay": self.config.train.weight_decay,
            "warmup_steps": self.config.train.warmup_steps,
            "max_grad_norm": self.config.train.max_grad_norm,
            "logging_steps": self.config.train.dataset_num_proc or 10,
            "save_steps": getattr(self.config.train, "save_interval", 500),
            "eval_steps": getattr(self.config.train, "eval_interval", 500),
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "seed": self.config.train.seed,
        }

        # Precision settings
        precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
        if precision == "bf16":
            training_kwargs["bf16"] = True
        elif precision == "fp16":
            training_kwargs["fp16"] = True

        training_args = TrainingArguments(**training_kwargs)

        # Setup PEFT for language model if enabled
        peft_config = None
        if getattr(self.config.model, "peft_enabled", False):
            try:
                from peft import LoraConfig

                lora_r = getattr(self.config.model, "lora_rank", 16)
                lora_alpha = getattr(self.config.model, "lora_alpha", 32)
                lora_dropout = getattr(self.config.model, "lora_dropout", 0.1)
                target_modules = getattr(self.config.model, "target_modules", None)

                if target_modules is None:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                logger.info(f"✅ PEFT LoRA enabled for language model (r={lora_r})")
            except ImportError:
                logger.warning("⚠️  PEFT not available")
                peft_config = None

        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processor,
            peft_config=peft_config,
        )

        logger.info("TRL VLM SFT Trainer created successfully")

    def train(self) -> Dict[str, Any]:
        """Execute VLM SFT training."""
        logger.info("=" * 80)
        logger.info("Starting VLM SFT Training")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            train_result = self.trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        end_time = time.time()
        training_duration = end_time - start_time

        logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Save final model
        self.trainer.save_model()

        self.training_history.append(
            {
                "timestamp": time.time(),
                "duration": training_duration,
                "steps": getattr(train_result, "global_step", 0),
                "task_type": self.task_type.value,
                "model_path": self.config.logging.output_dir,
            }
        )

        logger.info("=" * 80)
        logger.info("VLM SFT training completed successfully!")
        logger.info("=" * 80)

        return {
            "training_time": training_duration,
            "final_loss": getattr(train_result, "train_loss", 0.0),
            "model_path": self.config.logging.output_dir,
            "steps": getattr(train_result, "global_step", 0),
            "task_type": self.task_type.value,
        }

    def save(self, output_dir: str = None) -> str:
        """Save trained model."""
        save_dir = output_dir or self.config.logging.output_dir
        logger.info(f"Saving model to {save_dir}")

        if self.trainer:
            self.trainer.save_model(save_dir)
        else:
            self.model.save_pretrained(save_dir)
            self.processor.save_pretrained(save_dir)

        logger.info(f"Model saved successfully")
        return save_dir

    def load(self, model_dir: str) -> None:
        """Load trained model."""
        logger.info(f"Loading model from {model_dir}")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


__all__ = ["TRLVLMSFTTrainer"]
