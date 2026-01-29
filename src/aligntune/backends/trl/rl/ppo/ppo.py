"""
NEW TRL PPO Backend Implementation with Math, Code, and Enhanced Rewards

This module provides a pure TRL backend for Proximal Policy Optimization (PPO),
with integrated support for mathematical reasoning, code quality/correctness, and
additional reward functions for diversity, fluency, relevance, and brevity.

Based on the GRPO implementation pattern with PPO-specific adaptations.
"""

import logging
import time
import yaml
import re
import ast
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch
from torch.utils.data import DataLoader
import os
from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.rl.registries import DatasetRegistry, RewardRegistry
from aligntune.core.rl.caching import DatasetCache
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.core.sft.evaluator import EnhancedEvaluator
import gc
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import  extract_extra_and_missing_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# ============================================================================
# TRL PPO TRAINER (ENHANCED WITH NEW REWARDS)
# ============================================================================

class TRLPPOTrainer(TrainerBase):
    """PPO trainer using pure TRL PPOTrainer with math, code, and enhanced rewards."""

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.policy_model = None
        self.ref_model = None
        self.reward_model = None
        self.value_model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.train_dataset = None
        self.eval_dataset = None
        self.reward_functions = []
        self.training_history = []
        self.logging_manager = None
        self.custom_evaluator = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL is available."""
        try:
            from trl import PPOTrainer, PPOConfig
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup models (policy, reference, reward, value) using standard Transformers."""
        # Extract model config values safely
        policy_model_name = self._get_config_value(
            self.config.model,
            'name_or_path',
            'model_name',
            default='gpt2')
        
        
       # Check if integrated reward training is configured
        if (hasattr(self.config, 'reward_training') and 
            self.config.reward_training is not None):
            
            logger.info("🏋️ Training custom reward model before PPO...")
            reward_model_path = self._train_custom_reward_model()
            
            # Override reward_model_name with trained model path
            reward_model_name = reward_model_path
            logger.info(f"✅ Custom reward model trained and saved to: {reward_model_path}")
        else:
            # Extract from config as normal
            reward_model_name = self._get_config_value(
                self.config.model,
                'reward_model_name',
                'reward_path',
                default=None)
        
        value_model_name = self._get_config_value(
            self.config.model,
            'reward_value_model',
            default=None)
        
        sft_model_path = self._get_config_value(
            self.config.model,
            'sft_path',
            default=None)
        
        trust_remote_code = self._get_config_value(
            self.config.model, 'trust_remote_code', default=False)
        
        device_map = self._get_config_value(
            self.config.model, 'device_map', default='auto')
        
        is_ddp = (
            device_map in ["DDP", "ddp"] or
            os.environ.get('ACCELERATE_USE_DDP') == 'true'
        )

        if is_ddp:
            from accelerate import PartialState
            device_string = PartialState().process_index
            device_map = {'': device_string}
            logger.info(
                f"DDP mode detected: device_map={{'': {device_string}}}")

        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "TRL PPO")
        dtype = PrecisionHandler.get_torch_dtype(precision)

        # Quantization settings
        load_in_4bit = self._get_config_value(
            self.config.model, 'load_in_4bit', default=False)
        load_in_8bit = self._get_config_value(
            self.config.model, 'load_in_8bit', default=False)
        use_peft = self._get_config_value(
            self.config.model, 'use_peft', default=False)

        # Auto-detect quantization from model name
        if not load_in_4bit and not load_in_8bit:
            model_name_lower = policy_model_name.lower()
            if 'bnb-4bit' in model_name_lower or '4bit' in model_name_lower or 'awq' in model_name_lower:
                logger.info(
                    f"Auto-detected 4-bit quantization from model name: {policy_model_name}")
                load_in_4bit = True
            elif 'bnb-8bit' in model_name_lower or '8bit' in model_name_lower:
                logger.info(
                    f"Auto-detected 8-bit quantization from model name: {policy_model_name}")
                load_in_8bit = True

        # Auto-enable PEFT if using quantization
        if (load_in_4bit or load_in_8bit) and not use_peft:
            logger.info(
                "Quantization detected - auto-enabling PEFT/LoRA adapters (required for training)")
            use_peft = True

        logger.info("=" * 80)
        logger.info(f"Setting up TRL PPO models")
        logger.info(f"Policy model: {policy_model_name}")
        logger.info(f"SFT path: {sft_model_path or 'Same as policy'}")
        logger.info(f"Reward model: {reward_model_name or 'Same as value model'}")
        logger.info(f"Value model: {value_model_name}")
        logger.info("=" * 80)

        try:
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers") from e

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            policy_model_name,
            trust_remote_code=trust_remote_code,
        )

        
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")

        # Common model kwargs
        base_model_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        # Add quantization if specified for policy model
        policy_model_kwargs = base_model_kwargs.copy()
        if load_in_4bit:
            logger.info("Loading policy model with 4-bit quantization...")
            policy_model_kwargs["load_in_4bit"] = True
            policy_model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            policy_model_kwargs["bnb_4bit_quant_type"] = "nf4"
            policy_model_kwargs["bnb_4bit_use_double_quant"] = True
        elif load_in_8bit:
            logger.info("Loading policy model with 8-bit quantization...")
            policy_model_kwargs["load_in_8bit"] = True

        # Load policy model (from SFT path if provided, else base model)
        logger.info("Loading policy model...")
        policy_path = sft_model_path or policy_model_name
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            policy_path,
            **policy_model_kwargs
        )

        # Apply PEFT if specified
        if use_peft:
            logger.info("Applying PEFT (LoRA) configuration to policy model...")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Prepare model for k-bit training if using quantization
            if load_in_4bit or load_in_8bit:
                logger.info("Preparing policy model for k-bit training...")
                self.policy_model = prepare_model_for_kbit_training(self.policy_model)

            # Extract PEFT config values
            lora_r = self._get_config_value(
                self.config.model, 'lora_r', 'r', default=16)
            lora_alpha = self._get_config_value(
                self.config.model, 'lora_alpha', 'alpha', default=32)
            lora_target_modules = self._get_config_value(
                self.config.model,
                'lora_target_modules',
                'target_modules',
                default=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            lora_dropout = self._get_config_value(
                self.config.model, 'lora_dropout', 'dropout', default=0.05)

            logger.info(
                f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Target modules: {lora_target_modules}")

            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.policy_model = get_peft_model(self.policy_model, peft_config)
            self.policy_model.print_trainable_parameters()
            logger.info("PEFT adapters applied to policy model successfully")
            
            # Reference model is None when using PEFT (TRL handles this)
            self.ref_model = None
            logger.info("Reference model: None (PEFT mode - TRL will handle reference)")
        else:
            # Load reference model (frozen copy of SFT model)
            logger.info("Loading reference model (frozen copy)...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                policy_path,
                **base_model_kwargs
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("Reference model loaded and frozen")

        # Load reward and value models
        logger.info("Loading reward and value models...")
        
        # Reward model quantization settings
        reward_quant = self._get_config_value(
            self.config.model, 'reward_model_quantization', default={})
        reward_load_4bit = reward_quant.get('load_in_4bit', False)
        reward_load_8bit = reward_quant.get('load_in_8bit', False)
        
        reward_model_kwargs = base_model_kwargs.copy()
        if reward_load_4bit:
            logger.info("Loading reward model with 4-bit quantization...")
            reward_model_kwargs["load_in_4bit"] = True
            reward_model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            reward_model_kwargs["bnb_4bit_quant_type"] = "nf4"
            reward_model_kwargs["bnb_4bit_use_double_quant"] = True
        elif reward_load_8bit:
            logger.info("Loading reward model with 8-bit quantization...")
            reward_model_kwargs["load_in_8bit"] = True

        # Value model quantization settings
        value_quant = self._get_config_value(
            self.config.model, 'value_model_quantization', default={})
        value_load_4bit = value_quant.get('load_in_4bit', False)
        value_load_8bit = value_quant.get('load_in_8bit', False)
        
        value_model_kwargs = base_model_kwargs.copy()
        if value_load_4bit:
            logger.info("Loading value model with 4-bit quantization...")
            value_model_kwargs["load_in_4bit"] = True
            value_model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            value_model_kwargs["bnb_4bit_quant_type"] = "nf4"
            value_model_kwargs["bnb_4bit_use_double_quant"] = True
        elif value_load_8bit:
            logger.info("Loading value model with 8-bit quantization...")
            value_model_kwargs["load_in_8bit"] = True

        # Load reward model (or use value model if not specified)
        try:
            reward_path = reward_model_name or value_model_name
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_path,
                num_labels=1,
                **reward_model_kwargs
            )
            logger.info(f"Reward model loaded from: {reward_path}")
        except:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                policy_model_name,
                num_labels=1,
                **reward_model_kwargs
            )

        # Load value model
        try:
            
            self.value_model = AutoModelForSequenceClassification.from_pretrained(
                value_model_name,
                num_labels=1,
                **value_model_kwargs
            )
            logger.info(f"Value model loaded from: {value_model_name}")
        except:
            self.value_model = AutoModelForSequenceClassification.from_pretrained(
                policy_model_name,
                num_labels=1,
                **value_model_kwargs
            )
    def _train_custom_reward_model(self) -> str:
        """Train a custom reward model using the reward_training config."""
        from aligntune.rewards import RewardModelTrainer
        from aligntune.rewards.factory import create_reward_functions
        
        reward_config = self.config.reward_training
        
        # Validate required fields
        if not hasattr(reward_config, 'base_model_name') or not reward_config.base_model_name:
            raise ValueError("reward_training.base_model_name is required")
        if not hasattr(reward_config, 'output_dir') or not reward_config.output_dir:
            raise ValueError("reward_training.output_dir is required")
        if not hasattr(reward_config, 'reward_functions') or not reward_config.reward_functions:
            raise ValueError("reward_training.reward_functions is required")
        if not hasattr(reward_config, 'training_texts') or not reward_config.training_texts:
            raise ValueError("reward_training.training_texts is required (minimum 10 samples)")
        
        logger.info(f"Training reward model with base: {reward_config.base_model_name}")
        logger.info(f"Output directory: {reward_config.output_dir}")
        logger.info(f"Training samples: {len(reward_config.training_texts)}")
        logger.info(f"Reward functions: {reward_config.reward_functions}")
        
        # Create reward functions from config
        # reward_functions is a List[str] according to config
        from aligntune.rewards.registry import RewardRegistry
        reward_func_list = []
        for rf_name in reward_config.reward_functions:
            try:
                rf = RewardRegistry.get_reward_function(rf_name)
                reward_func_list.append(rf)
            except Exception as e:
                logger.warning(f"Could not load reward function '{rf_name}': {e}")
        
        if not reward_func_list:
            raise ValueError("No valid reward functions could be loaded")
        
        # Get composite weights if provided
        composite_weights = None
        if hasattr(reward_config, 'reward_weights') and reward_config.reward_weights:
            composite_weights = reward_config.reward_weights
        else:
            composite_weights = [1.0] * len(reward_func_list)
        
        # Create reward model trainer
        trainer = RewardModelTrainer(
            base_model_name=reward_config.base_model_name,
            reward_functions=reward_func_list,
            composite_weights=composite_weights
        )
        
        # Generate training dataset from training_texts
        logger.info("Generating training data from reward functions...")
        training_data = trainer.generate_training_data(
            texts=reward_config.training_texts,
            references=reward_config.reference_texts if hasattr(reward_config, 'reference_texts') else None,
            batch_size=reward_config.batch_size
        )
        
        logger.info(f"Generated {len(training_data)} training examples")
        
        # Train the reward model
        logger.info("Starting reward model training...")
        model_path = trainer.train_reward_model(
            training_data=training_data,
            output_dir=reward_config.output_dir,
            num_epochs=reward_config.num_epochs,
            learning_rate=reward_config.learning_rate,
            batch_size=reward_config.batch_size,
            gradient_accumulation_steps=reward_config.gradient_accumulation_steps
        )
        
        logger.info(f"✅ Custom reward model training completed: {model_path}")
        return model_path

        

    def setup_data(self) -> None:
        """Setup datasets for PPO training using unified DataManager."""
        logger.info("Setting up PPO datasets with DataManager...")
        
        # Extract dataset configuration
        dataset_config = None
        if hasattr(self.config, 'dataset') and self.config.dataset is not None:
            dataset_config = self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            dataset_config = self.config.datasets[0]
            if len(self.config.datasets) > 1:
                logger.warning(f"Multiple datasets provided, using first one")
        else:
            raise ValueError("No dataset configuration found")
        
        # Extract parameters
        dataset_name = self._get_config_value(dataset_config, 'name', default='imdb')
        split = self._get_config_value(dataset_config, 'split', default='train')
        config_name = self._get_config_value(dataset_config, 'config_name', default=None)
        system_prompt = self._get_config_value(dataset_config, 'system_prompt', default=None)
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
        
        # Advanced DataManager features
        column_mapping = self._get_config_value(dataset_config, 'column_mapping', default=None)
        processing_fn = self._get_config_value(dataset_config, 'processing_fn', default=None)
        processing_batched = self._get_config_value(dataset_config, 'processing_batched', default=False)
        max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
        percent = self._get_config_value(dataset_config, 'percent', default=None)
        
        logger.info(f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")
        
        # Initialize DataManager for PPO task (prompt-only format)
        from aligntune.data.manager import DataManager
        
        manager = DataManager(
            task_type="sft",
            system_prompt=system_prompt,
            tokenizer=self.tokenizer,
            enable_thinking=enable_thinking,
            column_mapping=column_mapping,
            processing_fn=processing_fn,
            processing_batched=processing_batched
        )
        
        # Load dataset - DataManager handles everything including chat template
        dataset_dict = manager.load_dataset(
            dataset_name,
            config_name=config_name,
            split=split,
        )
        
        # Extract train and validation splits
        self.train_dataset = dataset_dict["train"]
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict
        
        # Apply sampling if specified
        if max_samples:
            logger.info(f"Limiting train dataset to {max_samples} samples")
            self.train_dataset = self.train_dataset.select(range(min(max_samples, len(self.train_dataset))))
        elif percent and percent < 100:
            num_samples = int(len(self.train_dataset) * percent / 100)
            logger.info(f"Using {percent}% of dataset ({num_samples} samples)")
            self.train_dataset = self.train_dataset.select(range(num_samples))
        
        if self.eval_dataset and max_samples:
            eval_max = max_samples // 10  # Use 10% for eval
            self.eval_dataset = self.eval_dataset.select(range(min(eval_max, len(self.eval_dataset))))
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train examples")
        if self.eval_dataset:
            logger.info(f"Evaluation dataset: {len(self.eval_dataset)} examples")
        
        # Tokenize dataset for PPO
        print("Tokenizing dataset...")
        max_prompt_length = self._get_config_value(
            self.config.train, 'max_prompt_length', default=512)
        
        def tokenize_function(examples):
            """Tokenize prompts for PPO training."""
            # Get prompt column (DataManager ensures consistent naming)
            prompts = examples.get("prompt", examples.get("query", []))
            
            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                padding=False,  # Keep False like TRL example
                truncation=True,
                max_length=max_prompt_length,
            )
            input_ids = tokenized["input_ids"]
            
            # Calculate lengths for each sequence in the batch
            lengths = [len(ids) for ids in input_ids]
            
            return {"input_ids": input_ids, "lengths": lengths}
            
        
        # Tokenize datasets
        
        self.train_dataset = self.train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.train_dataset.column_names,
                desc="Tokenizing train dataset",
            )
            
        if self.eval_dataset:
                self.eval_dataset = self.eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=self.eval_dataset.column_names,
                    desc="Tokenizing eval dataset",
                )
            
            # Filter by length (like TRL example)
        logger.info(f"Filtering sequences longer than {max_prompt_length} tokens...")
        self.train_dataset = self.train_dataset.filter(
                lambda x: x["lengths"] <= max_prompt_length,
                desc="Filtering train dataset",
            )
            
        if self.eval_dataset:
                self.eval_dataset = self.eval_dataset.filter(
                    lambda x: x["lengths"] <= max_prompt_length,
                    desc="Filtering eval dataset",
                )
        
        logger.info(f"Final dataset sizes - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset) if self.eval_dataset else 0}")
        
        # Verify dataset structure (like TRL example checks)
        assert self.train_dataset[0]["input_ids"][-1] != self.tokenizer.eos_token_id, \
            "The last token should not be an EOS token"
        
        # Log sample
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            decoded = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
            logger.info(f"Sample tokenized prompt (first 100 chars): {decoded[:100]}...")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")

    def setup_rewards(self) -> None:
        """Setup reward functions using the centralized registry system."""
        logger.info("Setting up reward functions for PPO...")

        # Get reward configurations
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(
                self.config.rewards, list) else []

        if not rewards_config:
            logger.warning(
                "No reward configurations found, using default rewards")
            rewards_config = [
                {
                    "type": "length", "weight": 0.2, "params": {
                        "min_length": 20, "max_length": 200}},
                {
                    "type": "sentiment", "weight": 0.2, "params": {
                        "positive_weight": 1.0}},
                {
                    "type": "safety", "weight": 0.2, "params": {
                        "strict": True}},
                {
                    "type": "diversity", "weight": 0.2, "params": {}},
                {
                    "type": "fluency", "weight": 0.2, "params": {}},
            ]

        # Load reward functions from the registry
        for reward_config in rewards_config:
            # Handle both dict and RewardConfig object
            if isinstance(reward_config, dict):
                reward_type = reward_config.get('type', 'length')
                weight = reward_config.get('weight', 1.0)
                params = reward_config.get('params', {})
            else:
                # RewardConfig object
                reward_type = getattr(reward_config, 'type', 'length')
                weight = getattr(reward_config, 'weight', 1.0)
                params = getattr(reward_config, 'params', {})

            try:
                # Special case: custom reward function passed directly
                if reward_type == 'custom' and 'reward_function' in params:
                    reward_func = params['reward_function']
                    logger.info(
                        f"Loaded custom reward function (weight: {weight})")
                else:
                    # Use the rewards registry to get reward functions
                    from aligntune.rewards.registry import RewardRegistry as RewardsRegistry
                    from aligntune.rewards.core import RewardConfig, RewardType

                    # Map common variations to standard names
                    reward_type_mapping = {
                        'math': 'math_reasoning',
                        'code': 'code_quality',
                    }
                    reward_type = reward_type_mapping.get(
                        reward_type, reward_type)

                    try:
                        # Convert reward type string to enum
                        reward_type_enum = RewardType[reward_type.upper()]

                        # Create RewardConfig with weight and params
                        reward_cfg = RewardConfig(
                            reward_type=reward_type_enum,
                            weight=1.0,  # Weight will be applied separately
                            params=params
                        )

                        # Get reward function from registry
                        reward_func_obj = RewardsRegistry.get_reward_function(
                            reward_type, reward_cfg)
                        # Extract the callable compute method
                        if hasattr(reward_func_obj, 'compute'):
                            reward_func = reward_func_obj.compute
                        elif callable(reward_func_obj):
                            reward_func = reward_func_obj
                        else:
                            logger.warning(
                                f"Reward function '{reward_type}' is not callable, skipping")
                            continue

                        logger.info(
                            f"Loaded {reward_type} reward from registry (weight: {weight})")

                    except KeyError:
                        # Reward type not in enum, try registry by name
                        logger.warning(
                            f"Reward type '{reward_type}' not in RewardType enum, trying registry by name")
                        reward_func_obj = RewardsRegistry.get_reward_function(
                            reward_type)

                        if hasattr(reward_func_obj, 'compute'):
                            reward_func = reward_func_obj.compute
                        elif callable(reward_func_obj):
                            reward_func = reward_func_obj
                        else:
                            logger.warning(
                                f"Reward function '{reward_type}' is not callable, skipping")
                            continue

                        logger.info(
                            f"Loaded {reward_type} reward by name (weight: {weight})")

                # Store reward function with metadata
                self.reward_functions.append({
                    "function": reward_func,
                    "weight": weight,
                    "name": reward_type
                })

            except Exception as e:
                logger.warning(
                    f"Failed to load reward function '{reward_type}': {e}")
                logger.debug(f"Error details:", exc_info=True)
                continue

        if not self.reward_functions:
            logger.error(
                "No reward functions were loaded! Adding a simple default length reward.")

            def default_length_reward(text, reference=None, **kwargs):
                length = len(text.split())
                if length < 20:
                    return length / 20.0 * 0.5
                elif length > 200:
                    return max(0.0, 1.0 - (length - 200) / 200.0)
                else:
                    return 1.0

            self.reward_functions.append({
                "function": default_length_reward,
                "weight": 1.0,
                "name": "default_length"
            })
            logger.info("Added default length reward as fallback")

        logger.info(
            f"✓ Configured {len(self.reward_functions)} reward functions successfully")

        # Log summary of loaded rewards
        reward_summary = ", ".join(
            [f"{rf['name']} ({rf['weight']:.2f})" for rf in self.reward_functions])
        logger.info(f"Reward functions: {reward_summary}")

    def _combined_reward_function(
            self,
            completions: List[str],
            **kwargs) -> List[float]:
        """Combined reward function that applies all registered rewards.

        Args:
            completions: List of generated completions
            **kwargs: Additional arguments including prompts, references, etc.
        """
        if not completions:
            return []

        batch_rewards = []

        # Extract batch data from kwargs
        test_lists = kwargs.get('test_list', [None] * len(completions))
        
        # Check multiple possible reference column names
        references = None
        for ref_key in ['answer', 'solution', 'reference', 'ground_truth', 'response', 'target']:
            if ref_key in kwargs:
                references = kwargs[ref_key]
                break

        if references is None:
            references = [None] * len(completions)

        # Ensure lists match completion length
        if not isinstance(test_lists, list):
            test_lists = [test_lists] * len(completions)
        if not isinstance(references, list):
            references = [references] * len(completions)

        for idx, completion in enumerate(completions):
            total_reward = 0.0

            # Get per-sample data
            test_cases = test_lists[idx] if idx < len(test_lists) else None
            reference = references[idx] if idx < len(references) else None

            for rf in self.reward_functions:
                try:
                    reward_func = rf["function"]
                    weight = rf["weight"]

                    # Handle different reward function signatures
                    if callable(reward_func):
                        try:
                            reward = reward_func(completion, test_cases=test_cases)
                        except TypeError:
                            try:
                                reward = reward_func(completion)
                            except TypeError:
                                try:
                                    reward = reward_func(completion, reference=reference)
                                except TypeError:
                                    try:
                                        reward = reward_func(
                                            completion, reference=reference, context={})
                                    except BaseException:
                                        logger.debug(
                                            f"Could not call reward function {rf['name']}, returning 0")
                                        reward = 0.0
                    else:
                        logger.warning(f"Reward function {rf['name']} is not callable")
                        reward = 0.0

                    # Apply weight
                    weighted_reward = reward * weight
                    total_reward += weighted_reward

                    logger.debug(
                        f"Reward {rf['name']}: {reward:.4f} (weighted: {weighted_reward:.4f})")
                except Exception as e:
                    logger.warning(f"Error computing reward {rf['name']}: {e}")
                    logger.debug(f"Error details:", exc_info=True)

            batch_rewards.append(total_reward)

        # Log batch statistics
        if batch_rewards:
            successful = sum(1 for r in batch_rewards if r > 0.5)
            partial = sum(1 for r in batch_rewards if 0 < r <= 0.5)
            failed = sum(1 for r in batch_rewards if r <= 0)

            print(f"\n{'=' * 60}")
            print(
                f"BATCH REWARDS: {successful} passed | {partial} partial | {failed} failed | total={len(batch_rewards)}")
            print(
                f"Reward stats: min={min(batch_rewards):.2f}, max={max(batch_rewards):.2f}, mean={sum(batch_rewards) / len(batch_rewards):.2f}")
            print(f"{'=' * 60}\n")

        return batch_rewards

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

    def setup_trainer(self) -> None:
        """Set up the PPO trainer with all configurations."""
        logger.info("Setting up TRL PPO trainer...")

        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision_args = PrecisionHandler.get_training_args_precision(precision)

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
            self.config.train, 'gradient_accumulation_steps', default=1)
        max_grad_norm = self._get_config_value(
            self.config.train, 'max_grad_norm', default=1.0)
        weight_decay = self._get_config_value(
            self.config.train, 'weight_decay', default=0.01)
        warmup_steps = self._get_config_value(
            self.config.train, 'warmup_steps', default=10)
        seed = self._get_config_value(self.config.train, 'seed', default=42)
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/ppo_trl')

        # PPO-specific parameters
        kl_coef = self._get_config_value(
            self.config.train, 'kl_coef', default=0.1)
        cliprange = self._get_config_value(
            self.config.train, 'cliprange', default=0.2)
        cliprange_value = self._get_config_value(
            self.config.train, 'cliprange_value', default=0.2)
        vf_coef = self._get_config_value(
            self.config.train, 'vf_coef', default=0.1)
        gamma = self._get_config_value(
            self.config.train, 'gamma', default=1.0)
        lam = self._get_config_value(
            self.config.train, 'lam', default=0.95)

        # Generation parameters
        response_length = self._get_config_value(
            self.config.train, 'response_length', 'max_completion_length', default=128)
        temperature = self._get_config_value(
            self.config.train, 'temperature', default=0.7)
        stop_token = self._get_config_value(
            self.config.train, 'stop_token', default='eos')
        missing_eos_penalty = self._get_config_value(
            self.config.train, 'missing_eos_penalty', default=1.0)

        # Evaluation and checkpointing
        max_steps = self._get_config_value(
            self.config.train, 'max_steps', default=1000)
        eval_strategy = self._get_config_value(
            self.config.train, 'eval_strategy', default='steps')
        eval_steps = self._get_config_value(
            self.config.train, 'eval_steps', default=100)
        save_steps = self._get_config_value(
            self.config.train, 'save_steps', default=100)
        save_strategy = self._get_config_value(
            self.config.train, 'save_strategy', default='steps')

        # Logging
        logging_steps = self._get_config_value(
            self.config.train, 'logging_steps', default=10)
        report_to = self.config.logging.loggers if self.config.logging.loggers else []

        logger.info("=" * 80)
        logger.info("TRL PPO Training Configuration")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Batch size: {per_device_batch_size}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"KL coefficient: {kl_coef}")
        logger.info(f"Clip range: {cliprange}")
        logger.info(f"Value function coef: {vf_coef}")
        logger.info(f"Response length: {response_length}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup PPO trainer
        from trl import PPOTrainer, PPOConfig

        ppo_config = PPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            seed=seed,

            # PPO-specific
            kl_coef=kl_coef,
            cliprange=cliprange,
            cliprange_value=cliprange_value,
            vf_coef=vf_coef,
            gamma=gamma,
            lam=lam,

            # Generation
            response_length=response_length,
            temperature=temperature,
            stop_token=stop_token,
            missing_eos_penalty=missing_eos_penalty,

            # Evaluation
            max_steps=max_steps,
            eval_strategy=eval_strategy if self.eval_dataset else "no",
            eval_steps=eval_steps if self.eval_dataset else None,

            # Checkpointing
            save_strategy=save_strategy,
            save_steps=save_steps,

            # Logging
            logging_steps=logging_steps,
            report_to=report_to if report_to else [],

            # Precision
            **precision_args,

            # Other
            remove_unused_columns=False,
        )

        
        missing = extract_extra_and_missing_params(
            backend_config=ppo_config,
            config=self.config,
            algorithm='ppo'
        )

        for key, value in missing.items():
            setattr(ppo_config, key, value)

        # Get PEFT config if using PEFT
        peft_config = None
        use_peft = self._get_config_value(self.config.model, 'use_peft', default=False)
        if use_peft:
            from peft import LoraConfig
            lora_r = self._get_config_value(
                self.config.model, 'lora_r', 'r', default=16)
            lora_alpha = self._get_config_value(
                self.config.model, 'lora_alpha', 'alpha', default=32)
            lora_target_modules = self._get_config_value(
                self.config.model,
                'lora_target_modules',
                'target_modules',
                default=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            lora_dropout = self._get_config_value(
                self.config.model, 'lora_dropout', 'dropout', default=0.05)

            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        
        
        
        # Create PPO trainer
        self.trainer = PPOTrainer(
            args=ppo_config,
            processing_class=self.tokenizer,
            model=self.policy_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
        )

        logger.info("PPO trainer setup completed successfully!")

    def train(self) -> Dict[str, Any]:
        """Execute PPO training."""
        # Setup components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()
        self.setup_trainer()

        # Get output directory
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/ppo_trl')

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting TRL PPO Training")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info("=" * 80)



        
        train_result = self.trainer.train()

        # Log samples after training
        try:
            raw_reward_funcs = getattr(self, "reward_functions", None)
            reward_callables = None
            if isinstance(raw_reward_funcs, list):
                try:
                    reward_callables = [
                        rf["function"]
                        for rf in raw_reward_funcs
                        if isinstance(rf, dict) and callable(rf.get("function"))
                    ]
                except Exception:
                    reward_callables = None

            generate_and_log_samples(
                self.config.logging.sample_logging,
                self.policy_model,
                self.tokenizer,
                reward_callables,
                stage="post-train",
                log=logger,
            )
        except Exception as sample_error:
            logger.warning(f"Unable to log qualitative samples: {sample_error}")

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
                train_result, 'training_loss') else metrics.get('train_loss', 0.0),
            "total_steps": train_result.global_step if hasattr(
                train_result, 'global_step') else 0,
            "model_path": output_dir,
            "num_reward_functions": len(self.reward_functions),
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("TRL PPO Training Completed Successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)

        return results

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step - handled by TRL internally."""
        logger.debug("train_step() called but TRL PPO uses TRL's internal training loop")
        return {"loss": 0.0}

    def create_data_loader(self) -> Optional[DataLoader]:
        """Create data loader - handled by TRL internally."""
        logger.debug("create_data_loader() called but TRL PPO uses TRL's internal data loading")
        return None

    # def evaluate(self) -> Dict[str, float]:
    #     try:
    #         logger.info("Running evaluation...")
    #         return self.trainer.evaluate()
    #     except Exception as e:
    #         logger.error(f"Evaluation failed: {e}")
    #         return {}

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
                'task_type': 'proximal_policy_optimization',
                'dataset_name': self.config.datasets[0].name if self.config.datasets else 'unknown',
                'epochs': num_epochs,
                'learning_rate': self._get_config_value(
                    self.config.train, 'learning_rate', 'lr', default=1e-6),
                'batch_size': self._get_config_value(
                    self.config.train, 'per_device_batch_size', 'batch_size', default=1),
                'use_peft': self._get_config_value(
                    self.config.model, 'use_peft', default=False),
                'precision': self._get_config_value(
                    self.config.model, 'precision', default='fp32'),
                'num_reward_functions': len(self.reward_functions),
            },
            'dataset_info': {
                'train_size': len(self.train_dataset) if self.train_dataset else 0,
                'val_size': len(self.eval_dataset) if self.eval_dataset else 0,
            },
            'model_info': {
                'loaded': self.policy_model is not None,
                'device': str(next(self.policy_model.parameters()).device) if self.policy_model else 'unknown',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
                'has_peft': hasattr(self.policy_model, 'peft_config') if self.policy_model else False,
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
            'task_type': 'proximal_policy_optimization',
            'max_seq_length': self._get_config_value(
                self.config.model, 'max_seq_length', default=512),
            'learning_rate': self._get_config_value(
                self.config.train, 'learning_rate', 'lr', default=1e-6),
            'epochs': num_epochs,
            'batch_size': self._get_config_value(
                self.config.train, 'per_device_batch_size', 'batch_size', default=1),
            'dataset_name': self.config.datasets[0].name if self.config.datasets else 'unknown',
            'use_peft': self._get_config_value(
                self.config.model, 'use_peft', default=False),
            'precision': self._get_config_value(
                self.config.model, 'precision', default='fp32'),
            'num_reward_functions': len(self.reward_functions),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"TRL PPO configuration saved to {path}")

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained PPO model."""
        try:
            if isinstance(self.config.logging, dict):
                default_path = self.config.logging.get('output_dir', './output/ppo')
            else:
                default_path = getattr(
                    self.config.logging, 'output_dir', './output/ppo')

            save_path = path or default_path

            logger.info(f"Saving PPO model to: {save_path}")

            # Save using trainer
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training configuration
            config_path = Path(save_path) / "ppo_training_config.yaml"
            self.save_config(str(config_path))

            logger.info(f"PPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save PPO model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained PPO model."""
        try:
            logger.info(f"Loading PPO model from: {path}")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)

            # Load model
            max_seq_length = self._get_config_value(
                self.config.model, 'max_seq_length', default=2048)

            self.policy_model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
            )

            logger.info("PPO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            raise