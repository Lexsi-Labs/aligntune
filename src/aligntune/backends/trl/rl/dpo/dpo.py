"""
TRL DPO Trainer - Complete Implementation

This module provides a complete TRL-based Direct Preference Optimization trainer
that works with the backend factory system.
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
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import  extract_extra_and_missing_params
logger = logging.getLogger(__name__)


class TRLDPOTrainer(TrainerBase):
    """TRL-based DPO trainer using TRL's DPOTrainer."""
    
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
        """Setup rewards - not used in DPO (uses preferences)."""
        logger.info("DPO uses preference pairs instead of explicit rewards")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step - handled internally by TRL DPOTrainer."""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call train() first.")
        # The actual training loop is handled by self.trainer.train()
        return {}
    
    def setup_model(self) -> None:
        """Setup model and tokenizer for DPO."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            logger.info(f"Setting up TRL DPO model: {self.config.model.name_or_path}")
            

            # === UNIFIED PRECISION HANDLING ===
            precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "TRL DPO")
            dtype = PrecisionHandler.get_torch_dtype(precision)
        
            # Determine dtype
            dtype = None
            if self.config.model.precision.value == "bf16":
                dtype = torch.bfloat16
            elif self.config.model.precision.value == "fp16":
                dtype = torch.float16
            elif self.config.model.precision.value == "fp32":
                dtype = torch.float32
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name_or_path,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Setup quantization config
            quantization_config = None
            use_quantization = False
            if self.config.model.quantization:
                if self.config.model.quantization.get("load_in_4bit", False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype or torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    use_quantization = True
                    logger.info("Using 4-bit quantization with BitsAndBytesConfig")
                elif self.config.model.quantization.get("load_in_8bit", False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    use_quantization = True
                    logger.info("Using 8-bit quantization with BitsAndBytesConfig")
            
            # Load main model
            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": self.config.model.device_map or "auto",
                "trust_remote_code": True,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name_or_path,
                **model_kwargs
            )
            # ---- Extract LoRA config safely ----
            lora_r = self._get_config_value(
                self.config.model, 'lora_r', 'r', default=16
            )
            lora_alpha = self._get_config_value(
                self.config.model, 'lora_alpha', 'alpha', default=32
            )
            lora_target_modules = self._get_config_value(
                self.config.model,
                'lora_target_modules',
                'target_modules',
                default=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            lora_dropout = self._get_config_value(
                self.config.model, 'lora_dropout', 'dropout', default=0.05
            )

            logger.info(
                f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"
            )
            logger.info(f"Target modules: {lora_target_modules}")

            # ---- Apply LoRA ----
            from peft import LoraConfig, get_peft_model

            if use_quantization:
                from peft import prepare_model_for_kbit_training

                self.model = prepare_model_for_kbit_training(self.model)

                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                self.model = get_peft_model(self.model, peft_config)
                logger.info("Applied LoRA adapters to quantized model")

            elif getattr(self.config.model, "use_peft", False):
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                self.model = get_peft_model(self.model, peft_config)
                logger.info("Applied LoRA adapters to model")

            
            # Create reference model (frozen copy)
            # For quantized models, reference model uses same quantization but no LoRA
            ref_model_kwargs = {
                "torch_dtype": dtype,
                "device_map": self.config.model.device_map or "auto",
                "trust_remote_code": True,
            }
            
            if quantization_config is not None:
                ref_model_kwargs["quantization_config"] = quantization_config
            
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name_or_path,
                **ref_model_kwargs
            )
            
            logger.info("TRL DPO model setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup TRL DPO model: {e}")
            raise
    
    def setup_dataset(self) -> None:
        """Setup and prepare preference dataset for DPO training using unified DataManager."""
        try:
            """Setup datasets for GRPO training using unified DataManager."""
            logger.info("Setting up GRPO datasets with DataManager...")
            
            # Extract dataset configuration
            dataset_config = None
            if hasattr(self.config, 'dataset'):
                dataset_config = self.config.dataset
            elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
                dataset_config = self.config.datasets[0]
            else:
                raise ValueError("No dataset configuration found")
            
            # Extract parameters
            dataset_name = self._get_config_value(dataset_config, 'name', 'dataset_name', default='imdb')
            split = self._get_config_value(dataset_config, 'split', default=None)
            config_name = self._get_config_value(dataset_config, 'config_name', default=None)
            system_prompt = self._get_config_value(dataset_config, 'system_prompt', default=None)
            enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
            
            # Advanced DataManager features
            column_mapping = self._get_config_value(dataset_config, 'column_mapping', default=None)
            processing_fn = self._get_config_value(dataset_config, 'processing_fn', default=None)
            processing_batched = self._get_config_value(dataset_config, 'processing_batched', default=False)
            max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
            max_eval_samples =  self._get_config_value(dataset_config, 'max_eval_samples', default=None)
            logger.info(f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")
            
            # Initialize DataManager for GRPO task
            from aligntune.data.manager import DataManager
            
            manager = DataManager(
                task_type="dpo",
                system_prompt=system_prompt,           # System prompt
                tokenizer=self.tokenizer,              # ✅ ADD THIS - Pass tokenizer for chat template
                enable_thinking=enable_thinking,       # ✅ ADD THIS - Enable thinking mode
                column_mapping=column_mapping,
                processing_fn=processing_fn,
                max_samples = max_samples, 
                processing_batched=processing_batched,
            )
            
            # Load dataset - DataManager handles everything including chat template
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
                        logger.info(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            
            # Define a robust filtering function that handles different column structures
            def filter_valid_examples(example):
                """Filter out invalid examples with flexible column detection."""
                try:
                    # If we have the standard DPO format
                    if has_prompt and has_chosen and has_rejected:
                        return (
                            len(example.get("prompt", "")) > 0 and 
                            len(example.get("chosen", "")) > 0 and 
                            len(example.get("rejected", ""))> 0
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
                        logger.warning(f"Unexpected dataset format. Available columns: {available_columns}")
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
                    logger.info(f"Filtered out {filtered_count} invalid examples")
            
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
                logger.info(f"Filtered {original_size - filtered_size} samples that were too long")
            
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
            
            logger.info(f"DPO dataset prepared: {len(self.train_dataset)} train samples")
            if self.eval_dataset:
                logger.info(f"Evaluation dataset: {len(self.eval_dataset)} samples")
            
            # Log final sample for debugging
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                logger.info("Final dataset structure:")
                for key in sample.keys():
                    value = sample[key]
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        logger.info(f"  {key}: {preview}")
                
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise
    def setup_trainer(self) -> None:
        """Setup TRL DPOTrainer."""
        try:
            from trl import DPOTrainer, DPOConfig
            
            logger.info("Setting up TRL DPOTrainer")
            
            # Create DPO configuration
            # Get optimizer and scheduler configurations
            from aligntune.core.optimization import get_optimizer_for_config, get_scheduler_for_config

            # Use config-specified optimizer or default to adamw_torch
            optimizer_name = getattr(self.config.train, 'optimizer', 'adamw_torch')
            scheduler_name = getattr(self.config.train, 'lr_scheduler', 'cosine')

            # Create optimizer configuration
            optimizer_config = get_optimizer_for_config(
                optimizer_name,
                self.config.train.learning_rate,
                self.config.train.weight_decay or 0.01
            )

            # Calculate warmup steps
            num_train_epochs = self.config.train.epochs or 1
            # Estimate max_steps for scheduler if not provided
            max_steps = getattr(self.config.train, 'max_steps', None)
            if max_steps is None and hasattr(self.train_dataset, '__len__'):
                # Rough estimation: steps = epochs * (dataset_size / batch_size / accumulation)
                dataset_size = len(self.train_dataset)
                batch_size = self.config.train.per_device_batch_size or 1
                accumulation = self.config.train.gradient_accumulation_steps or 1
                effective_batch_size = batch_size * accumulation
                max_steps = int(num_train_epochs * dataset_size / effective_batch_size)
            else:
                max_steps = 1000  # fallback

            warmup_steps = getattr(self.config.train, 'warmup_steps', 0)
            if hasattr(self.config.train, 'warmup_ratio') and self.config.train.warmup_ratio:
                warmup_steps = int(max_steps * self.config.train.warmup_ratio)

            # Create scheduler configuration
            scheduler_config = get_scheduler_for_config(
                scheduler_name,
                max_steps,
                warmup_steps
            )
            # print( scheduler_config)
           
            def kwargs_to_str(kwargs_dict):
                """Convert optimizer/scheduler kwargs dict to string format."""
                return ",".join(f"{k}={f'({','.join(map(str, v))})' if isinstance(v, tuple) else v}" for k, v in kwargs_dict.items()) if kwargs_dict else None
        
            optim_args_str=kwargs_to_str(optimizer_config['optimizer_kwargs'])
            precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
            precision_args = PrecisionHandler.get_training_args_precision(precision)

            # Evaluation parameters
            eval_strategy = self._get_config_value(self.config.train, 'eval_strategy', default='epoch')
            eval_steps = self._get_config_value(self.config.train, 'eval_steps', default=100)
            save_steps = self._get_config_value(self.config.train, 'save_steps', default=100)
            save_strategy = self._get_config_value(self.config.train, 'save_strategy', default='steps')
            save_total_limit = self._get_config_value(self.config.train, 'save_total_limit', default=None)
            load_best_model_at_end = self._get_config_value(self.config.train, 'load_best_model_at_end', default=True if self.eval_dataset else False)
            metric_for_best_model = self._get_config_value(self.config.train, 'metric_for_best_model', default='eval_loss' if self.eval_dataset else None)
            greater_is_better = self._get_config_value(self.config.train, 'greater_is_better', default=False)

            # Logging parameters
            logging_steps = self._get_config_value(self.config.train, 'logging_steps', default=10)
            logging_strategy = self._get_config_value(self.config.train, 'logging_strategy', default='steps')
            report_to = self._get_config_value(self.config.logging, 'report_to', default=None)

            # Use eval_dataset-aware defaults
            if self.eval_dataset:
                eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
            else:
                eval_strategy = 'no'
                eval_steps = None


            # Get max_steps from config (if set, it overrides num_train_epochs)
            config_max_steps = self._get_config_value(self.config.train, 'max_steps', default=None)
            
            dpo_config = DPOConfig(
                output_dir=self.config.logging.output_dir,
                num_train_epochs=num_train_epochs,
                max_steps=config_max_steps if config_max_steps else -1,  # -1 means use num_train_epochs
                per_device_train_batch_size=self.config.train.per_device_batch_size,
                per_device_eval_batch_size=self.config.train.per_device_batch_size,  # ADD THIS
                gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
                learning_rate=self.config.train.learning_rate,
                warmup_steps=warmup_steps,
                
                # Evaluation parameters
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                
                # Logging parameters
                logging_strategy=logging_strategy,
                logging_steps=logging_steps,
                
                # Save parameters
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                
                # Optimizer and scheduler
                lr_scheduler_type=scheduler_name,
                optim=optimizer_name,
                weight_decay=optimizer_config['optimizer_kwargs'].get('weight_decay', 0.01),
                optim_args=optim_args_str,
                
                # Precision
                **precision_args,
                
                # Data handling
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                
                # DPO-specific parameters
                beta=self.config.train.beta,
                loss_type=getattr(self.config.train, 'loss_type', 'sigmoid'),
                max_length=self.config.model.max_seq_length,
                max_prompt_length=getattr(self.config.train, 'max_prompt_length', self.config.model.max_seq_length // 2),
                truncation_mode=getattr(self.config.train, 'truncation_mode', 'keep_end'),
                precompute_ref_log_probs=getattr(self.config.train, 'precompute_ref_log_probs', False),
                
                # Logging configuration
                report_to=report_to if report_to else (self.config.logging.loggers if self.config.logging.loggers else []),
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
            self.setup_dataset()
            self.setup_trainer()
            
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
    # def evaluate(self, eval_dataset=None, metric_key_prefix: str = "eval",use_custom_evaluator=False, **kwargs) -> Dict[str, float]:
    #     # Setup evaluators on first call
    #     if self.base_evaluator is None and self.rl_evaluator is None:
    #         self.setup_custom_evaluator(evaluator_type="auto")

    #     # Call parent's evaluate method
    #     return super().evaluate(
    #         eval_dataset=eval_dataset,
    #         metric_key_prefix=metric_key_prefix,
    #         use_custom_evaluator=use_custom_evaluator, 
    #         **kwargs
    #     )
        

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