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
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.registries import DatasetRegistry, RewardRegistry
from aligntune.core.rl.caching import DatasetCache
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.utils.config_extractor import  extract_extra_and_missing_params
logger = logging.getLogger(__name__)


class UnslothDPOTrainer(TrainerBase):
    """Unsloth-optimized DPO trainer using TRL's DPOTrainer."""

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
        # Required abstract methods implementation

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

    def setup_data(self) -> None:
        """Setup data - delegates to setup_dataset."""
        self.setup_dataset()

    def setup_rewards(self) -> None:
        """Setup rewards - not used in DPO."""
        pass

    def train_step(self) -> Dict[str, Any]:
        """Single training step - handled by TRL internally."""
        raise NotImplementedError("Use train() method instead")

    def setup_model(self) -> None:
        """Setup Unsloth-optimized model and tokenizer for DPO."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import DPOConfig, ModelConfig
            from transformers import AutoTokenizer

            logger.info(
                f"Setting up Unsloth DPO model: {
                    self.config.model.name_or_path}")

            device_map = self.config.model.device_map or 'auto'

            # Configure Unsloth model parameters
            model_kwargs = {
                "max_seq_length": self.config.model.max_seq_length,
                "dtype": None,  # Auto-detect
                "load_in_4bit": self.config.model.quantization.get("load_in_4bit", True),
                "device_map": device_map
            }

            # Add quantization parameters if specified
            # FIXED: Only add parameters that are valid for
            # FastLanguageModel.from_pretrained
            if self.config.model.quantization:
                # These are the parameters that Unsloth's FastLanguageModel
                # accepts
                valid_quant_params = {
                    'load_in_4bit',
                    'use_gradient_checkpointing',
                    'rope_scaling',
                    'fix_tokenizer',
                    'trust_remote_code',
                    'use_cache',
                    'token'
                }

                for key, value in self.config.model.quantization.items():
                    if key in valid_quant_params and key != "load_in_4bit":
                        model_kwargs[key] = value
                    # Silently skip unsupported parameters like bnb_4bit_compute_dtype
                    # These are handled internally by Unsloth

            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )

            # Configure model for DPO training
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            # Create reference model for DPO
            self.reference_model, _ = FastLanguageModel.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )

            logger.info("Unsloth DPO model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth DPO model: {e}")
            raise

    def setup_dataset(self) -> None:
        """Setup and prepare preference dataset for DPO training using unified DataManager."""
        try:
            logger.info("Setting up DPO dataset with DataManager...")

            # Extract dataset configuration
            if not hasattr(
                    self.config, 'datasets') or len(
                    self.config.datasets) == 0:
                raise ValueError("No dataset configuration found")

            dataset_config = self.config.datasets[0]

            # Extract parameters
            dataset_name = self._get_config_value(
                dataset_config, 'name', 'dataset_name')
            split = self._get_config_value(
                dataset_config, 'split', default=None)
            config_name = self._get_config_value(
                dataset_config, 'config_name', default=None)
            system_prompt = self._get_config_value(
                dataset_config, 'system_prompt', default=None)

            # Advanced DataManager features
            column_mapping = self._get_config_value(
                dataset_config, 'column_mapping', default=None)
            processing_fn = self._get_config_value(
                dataset_config, 'processing_fn', default=None)
            processing_batched = self._get_config_value(
                dataset_config, 'processing_batched', default=False)
            max_samples = self._get_config_value(
                dataset_config, 'max_samples', default=None)

            max_eval_samples =  self._get_config_value(dataset_config, 'max_eval_samples', default=None)

            logger.info(
                f"Loading DPO dataset: {dataset_name} (split: {split}, config: {config_name})")

            # Initialize DataManager for DPO task
            from aligntune.data.manager import DataManager
            enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)

            manager = DataManager(
                task_type="dpo",
                system_prompt=system_prompt,
                column_mapping=column_mapping,
                processing_fn=processing_fn,
                processing_batched=processing_batched,
                max_samples = max_samples, 
                tokenizer=self.tokenizer,              # ✅ ADD THIS - Pass tokenizer for chat template
                enable_thinking=enable_thinking, 
            )

            # Load dataset
            dataset_dict = manager.load_dataset(
                dataset_name,
                config_name=config_name,
                split=split,
            )

            # Extract train and validation splits
            self.train_dataset = dataset_dict.get("train", None)

            self.eval_dataset = dataset_dict.get("validation", None)
            self.dataset_dict = dataset_dict 

            # === ROBUST COLUMN DETECTION ===
            # Check what columns actually exist
            available_columns = set(self.train_dataset.column_names)
            logger.info(f"Dataset columns: {available_columns}")

            # Detect the column structure
            has_prompt = "prompt" in available_columns
            has_chosen = "chosen" in available_columns
            has_rejected = "rejected" in available_columns

            # Log first sample to understand the structure
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                logger.info(f"Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, str):
                        logger.info(f"  {key}: {value[:100]}..." if len(
                            value) > 100 else f"  {key}: {value}")

            # Define a robust filtering function that handles different column
            # structures
            def filter_valid_examples(example):
                """Filter out invalid examples with flexible column detection."""
                try:
                    # If we have the standard DPO format
                    if has_prompt and has_chosen and has_rejected:
                        return (
                            len(example.get("prompt", "")) > 0 and
                            len(example.get("chosen", "")) > 0 and
                            len(example.get("rejected", "")) > 0
                        )

                    # Alternative: dataset might have 'chosen' and 'rejected' only
                    # (TRL can handle this format too)
                    elif has_chosen and has_rejected and not has_prompt:
                        return (
                            len(example.get("chosen", "")) > 0 and
                            len(example.get("rejected", "")) > 0
                        )

                    # If columns don't match expected format, log and skip
                    else:
                        logger.warning(
                            f"Unexpected dataset format. Available columns: {available_columns}")
                        return False

                except Exception as e:
                    logger.warning(f"Error filtering example: {e}")
                    return False

            # Filter empty examples
            if len(self.train_dataset) > 0:
                initial_size = len(self.train_dataset)
                self.train_dataset = self.train_dataset.filter(
                    filter_valid_examples,
                    desc="Filtering invalid examples"
                )
                filtered_count = initial_size - len(self.train_dataset)
                if filtered_count > 0:
                    logger.info(
                        f"Filtered out {filtered_count} invalid examples")

            # Filter by approximate token count
            max_length = self.config.model.max_seq_length

            def is_valid_length(example):
                """Check if example fits within max sequence length."""
                try:
                    # Rough approximation: 1 token ≈ 4 characters

                    # Handle different column structures
                    if has_prompt:
                        prompt_tokens = len(example.get("prompt", "")) // 4
                    else:
                        # If no prompt, we'll check chosen/rejected only
                        prompt_tokens = 0

                    chosen_tokens = len(example.get("chosen", "")) // 4
                    rejected_tokens = len(example.get("rejected", "")) // 4

                    # Each response is concatenated with prompt, so check both
                    return (
                        prompt_tokens + chosen_tokens < max_length and
                        prompt_tokens + rejected_tokens < max_length
                    )
                except Exception as e:
                    logger.warning(f"Error checking length: {e}")
                    return False

            original_size = len(self.train_dataset)
            self.train_dataset = self.train_dataset.filter(
                is_valid_length,
                desc="Filtering long sequences"
            )
            filtered_size = len(self.train_dataset)

            if original_size - filtered_size > 0:
                logger.info(
                    f"Filtered {
                        original_size -
                        filtered_size} samples that were too long")

            # Apply same filtering to eval dataset if it exists
            if self.eval_dataset:
                self.eval_dataset = self.eval_dataset.filter(
                    filter_valid_examples,
                    desc="Filtering invalid eval examples"
                )
                self.eval_dataset = self.eval_dataset.filter(
                    is_valid_length,
                    desc="Filtering long eval sequences"
                )

            logger.info(
                f"DPO dataset prepared: {len(self.train_dataset)} train samples")
            if self.eval_dataset:
                logger.info(
                    f"Evaluation dataset: {len(self.eval_dataset)} samples")

            # Log final sample for debugging
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                logger.info("Final dataset structure:")
                for key in sample.keys():
                    value = sample[key]
                    if isinstance(value, str):
                        preview = value[:100] + \
                            "..." if len(value) > 100 else value
                        logger.info(f"  {key}: {preview}")

        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise

    def setup_trainer(self) -> None:
        """Setup TRL DPOTrainer with Unsloth model."""
        try:
            from trl import DPOTrainer, DPOConfig, ModelConfig

            logger.info("Setting up TRL DPOTrainer with Unsloth model")

            from aligntune.core.precision_handler import PrecisionHandler
            precision = PrecisionHandler.get_precision_from_config(
                self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "GRPO (Unsloth)")
            precision_args = PrecisionHandler.get_training_args_precision(
                precision)
            # Get training parameters
            num_epochs = self._get_config_value(
                self.config.train, 'epochs', 'num_epochs', default=1)
            if num_epochs is None:
                num_epochs = 1
            per_device_batch_size = self._get_config_value(
                self.config.train, 'per_device_batch_size', default=4)
            gradient_accumulation_steps = self._get_config_value(
                self.config.train, 'gradient_accumulation_steps', default=1)
            learning_rate = self._get_config_value(
                self.config.train, 'learning_rate', default=5e-5)
            max_steps = self._get_config_value(
                self.config.train, 'max_steps', default=-1)
            if max_steps is None:
                max_steps = -1

            # Get output and logging parameters
            output_dir = self._get_config_value(
                self.config.logging, 'output_dir', default='./output/unsloth_dpo')
            run_name = self._get_config_value(
                self.config.logging, 'run_name', default='unsloth_dpo')

            # Evaluation parameters
            eval_strategy = self._get_config_value(
                self.config.train, 'eval_strategy', default='steps')
            eval_steps = self._get_config_value(
                self.config.train, 'eval_steps', default=100)
            per_device_eval_batch_size = self._get_config_value(
                self.config.train, 'per_device_eval_batch_size', default=per_device_batch_size)
            metric_for_best_model = self._get_config_value(
                self.config.train, 'metric_for_best_model', default='eval_loss')
            greater_is_better = self._get_config_value(
                self.config.train, 'greater_is_better', default=False)
            load_best_model_at_end = self._get_config_value(
                self.config.train, 'load_best_model_at_end', default=True)

            # Adjust eval strategy based on eval_dataset availability
            if self.eval_dataset:
                eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
            else:
                eval_strategy = 'no'
                eval_steps = None
                load_best_model_at_end = False
                metric_for_best_model = None

            # Logging parameters
            logging_steps = self._get_config_value(
                self.config.train, 'logging_steps', default=10)
            logging_strategy = self._get_config_value(
                self.config.train, 'logging_strategy', default='steps')
            save_steps = self._get_config_value(
                self.config.train, 'save_steps', default=100)
            save_strategy = self._get_config_value(
                self.config.train, 'save_strategy', default='steps')
            save_total_limit = self._get_config_value(
                self.config.train, 'save_total_limit', default=None)

            # Report to
            report_to = self._get_config_value(
                self.config.logging, 'report_to', default='none')
            if isinstance(self.config.logging, dict):
                loggers = self.config.logging.get('loggers', [])
            else:
                loggers = getattr(self.config.logging, 'loggers', [])
            if loggers and report_to == 'none':
                report_to = loggers

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
            max_grad_norm = self._get_config_value(
                self.config.train, 'max_grad_norm', default=1.0)

            # Additional training parameters
            gradient_checkpointing = self._get_config_value(
                self.config.train,
                'use_gradient_checkpointing',
                'gradient_checkpointing',
                default=True)
            group_by_length = self._get_config_value(
                self.config.train, 'group_by_length', default=False)
            seed = self._get_config_value(
                self.config.train, 'seed', default=42)
            data_seed = self._get_config_value(
                self.config.train, 'data_seed', default=47)

            # DPO specific parameters
            beta = self._get_config_value(
                self.config.train, 'beta', default=0.1)
            loss_type = self._get_config_value(
                self.config.train, 'loss_type', default='sigmoid')
            label_smoothing = self._get_config_value(
                self.config.train, 'label_smoothing', default=0.0)
            truncation_mode = self._get_config_value(
                self.config.train, 'truncation_mode', default='keep_end')

            # Sequence lengths
            max_seq_length = self._get_config_value(
                self.config.model, 'max_seq_length', default=512)
            max_length = self._get_config_value(
                self.config.train, 'max_length', default=max_seq_length)
            max_prompt_length = self._get_config_value(
                self.config.train, 'max_prompt_length', default=max_seq_length // 2)
            # Adjust save strategy to save on epoch
            if self.eval_dataset:
                save_strategy = "epoch"

            # Create DPO configuration
            dpo_config = DPOConfig(
                # Output and logging
                output_dir=output_dir,
                run_name=run_name,
                logging_steps=logging_steps,
                logging_strategy=logging_strategy,
                report_to=report_to,

                # Evaluation
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                load_best_model_at_end=load_best_model_at_end,

                # Checkpointing
                save_steps=save_steps,
                save_strategy=save_strategy,
                save_total_limit=save_total_limit,

                # Training parameters
                num_train_epochs=num_epochs,
                max_steps=max_steps,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,

                # Optimizer and scheduler
                optim=optimizer,
                lr_scheduler_type=lr_scheduler_type,

                # DPO specific parameters
                beta=beta,
                loss_type=loss_type,
                label_smoothing=label_smoothing,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                truncation_mode=truncation_mode,

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
                backend_config=dpo_config,
                config=self.config,
                algorithm='dpo'
            )

            for key, value in missing.items():
                setattr(dpo_config, key, value)

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
            self.setup_dataset()
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

    def save_model(self, path: Optional[str] = None) -> str:
        """Save model."""
        try:
            default_path = self._get_config_value(self.config.logging, 'output_dir', './output/dpo')
            save_path = path or default_path
            
            logger.info(f"Saving to: {save_path}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            config_path = Path(save_path) / "training_config.yaml"
            import yaml
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info("Saved successfully")
            return save_path
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise
    

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        try:
            logger.info(f"Loading model from: {path}")

            # Load model and tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
