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

from aligntune.core.precision_handler import PrecisionHandler
from aligntune.utils.config_extractor import  extract_extra_and_missing_params
logger = logging.getLogger(__name__)


# ============================================================================
# TRL GRPO TRAINER (ENHANCED WITH NEW REWARDS)
# ============================================================================

class TRLGRPOTrainer(TrainerBase):
    """GRPO trainer using pure TRL GRPOTrainer with math, code, and enhanced rewards."""

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
        """Setup model using standard Transformers."""
        # Extract model config values safely
        model_name = self._get_config_value(
            self.config.model,
            'name_or_path',
            'model_name',
            default='gpt2')
        trust_remote_code = self._get_config_value(
            self.config.model, 'trust_remote_code', default=False)
        precision = self._get_config_value(
            self.config.model, 'precision', default='fp32')
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
        load_in_4bit = self._get_config_value(
            self.config.model, 'load_in_4bit', default=False)
        load_in_8bit = self._get_config_value(
            self.config.model, 'load_in_8bit', default=False)
        use_peft = self._get_config_value(
            self.config.model, 'use_peft', default=False)
        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "TRL GRPO")
        dtype = PrecisionHandler.get_torch_dtype(precision)

        # Auto-detect quantization from model name if not explicitly set
        if not load_in_4bit and not load_in_8bit:
            model_name_lower = model_name.lower()
            if 'bnb-4bit' in model_name_lower or '4bit' in model_name_lower or 'awq' in model_name_lower:
                logger.info(
                    f"Auto-detected 4-bit quantization from model name: {model_name}")
                load_in_4bit = True
            elif 'bnb-8bit' in model_name_lower or '8bit' in model_name_lower:
                logger.info(
                    f"Auto-detected 8-bit quantization from model name: {model_name}")
                load_in_8bit = True

        # Auto-enable PEFT if using quantization (required for training
        # quantized models)
        if (load_in_4bit or load_in_8bit) and not use_peft:
            logger.info(
                "Quantization detected - auto-enabling PEFT/LoRA adapters (required for training)")
            use_peft = True

        logger.info("=" * 80)
        logger.info(f"Setting up TRL GRPO model: {model_name}")
        logger.info("=" * 80)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers") from e

        # Load tokenizer first
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")

        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        # Add quantization if specified
        if load_in_4bit:
            logger.info("Loading model with 4-bit quantization...")
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
            model_kwargs["bnb_4bit_use_double_quant"] = True
        elif load_in_8bit:
            logger.info("Loading model with 8-bit quantization...")
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Apply PEFT if specified or if quantization is used
        if use_peft:
            logger.info("Applying PEFT (LoRA) configuration...")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Prepare model for k-bit training if using quantization
            if load_in_4bit or load_in_8bit:
                logger.info("Preparing model for k-bit training...")
                self.model = prepare_model_for_kbit_training(self.model)

            # Extract PEFT config values safely
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

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            logger.info("PEFT adapters applied successfully")

        logger.info("=" * 80)
        logger.info("TRL GRPO model setup completed successfully")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info("=" * 80)

    def setup_data(self) -> None:
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
        
        logger.info(f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")
        
        # Initialize DataManager for GRPO task
        from aligntune.data.manager import DataManager
        
        manager = DataManager(
            task_type="grpo",
            system_prompt=system_prompt,           # System prompt
            tokenizer=self.tokenizer,              # ✅ ADD THIS - Pass tokenizer for chat template
            enable_thinking=enable_thinking,       # ✅ ADD THIS - Enable thinking mode
            column_mapping=column_mapping,
            processing_fn=processing_fn,
            processing_batched=processing_batched,
            max_samples = max_samples, 
        )
        
        # Load dataset - DataManager handles everything including chat template
        dataset_dict = manager.load_dataset(
            dataset_name,
            config_name=config_name,
            split=split,
        )
    
        self.train_dataset = dataset_dict.get("train", None)
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict
      
        
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train examples")
        if self.eval_dataset:
            logger.info(f"Evaluation dataset: {len(self.eval_dataset)} examples")
        
        # Log sample
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            prompt_col = "prompt" if "prompt" in sample else "query"
            logger.info(f"Sample prompt (first 100 chars): {sample[prompt_col][:100]}...")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")            


    def setup_rewards(self) -> None:
        """Setup reward functions using the centralized registry system."""
        logger.info("Setting up reward functions for GRPO...")

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
                        "min_length": 20, "max_length": 200}}, {
                    "type": "sentiment", "weight": 0.2, "params": {
                        "positive_weight": 1.0}}, {
                    "type": "safety", "weight": 0.2, "params": {
                            "strict": True}}, {
                                "type": "diversity", "weight": 0.2, "params": {}}, {
                                    "type": "fluency", "weight": 0.2, "params": {}}, ]

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

                # Fallback: try to continue with other rewards rather than
                # failing
                continue

        if not self.reward_functions:
            logger.error(
                "No reward functions were loaded! Adding a simple default length reward.")
            # Add a simple fallback reward so training doesn't fail

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
            **kwargs: Additional arguments from TRL including:
                - prompts: List of prompts
                - test_list: List of test cases (for code datasets like MBPP)
                - answer/solution/reference: Reference answers (for math datasets)
        """
        if not completions:
            return []

        # Debug: log what TRL is passing (once)
        if not hasattr(self, '_logged_kwargs'):
            logger.info(
                f"[DEBUG] _combined_reward_function kwargs keys: {
                    list(
                        kwargs.keys())}")
            if 'test_list' in kwargs:
                sample = kwargs['test_list'][:1] if kwargs['test_list'] else 'empty'
                logger.info(f"[DEBUG] test_list found, sample: {sample}")
            else:
                logger.info("[DEBUG] test_list NOT in kwargs!")
            self._logged_kwargs = True

        batch_rewards = []

        # Extract batch data from kwargs (TRL passes these as lists)
        test_lists = kwargs.get('test_list', [None] * len(completions))
        # CHECK MULTIPLE POSSIBLE REFERENCE COLUMN NAMES
        # Try different column names in order of preference
        references = None
        for ref_key in [
            'answer',
            'solution',
            'reference',
            'ground_truth',
            'response',
                'target']:
            if ref_key in kwargs:
                references = kwargs[ref_key]
                # print(f"[INFO] Found reference data in column: '{ref_key}'")
                break

        # If no reference column found, use None for all completions
        if references is None:
            references = [None] * len(completions)
            print(
                f"[WARNING] No reference column found. Checked: answer, solution, reference, ground_truth, target")

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
                        # Try different calling patterns
                        try:
                            # Pattern 1: text + test_cases (for code execution)
                            reward = reward_func(
                                completion, test_cases=test_cases)
                        except TypeError:
                            try:
                                # Pattern 2: Just text
                                reward = reward_func(completion)
                            except TypeError:
                                try:
                                    # Pattern 3: text + reference
                                    reward = reward_func(
                                        completion, reference=reference)
                                except TypeError:
                                    try:
                                        # Pattern 4: text + reference + context
                                        reward = reward_func(
                                            completion, reference=reference, context={})
                                    except BaseException:
                                        logger.debug(
                                            f"Could not call reward function {
                                                rf['name']}, returning 0")
                                        reward = 0.0
                    else:
                        logger.warning(
                            f"Reward function {
                                rf['name']} is not callable")
                        reward = 0.0

                    # Apply weight
                    weighted_reward = reward * weight
                    total_reward += weighted_reward

                    logger.debug(
                        f"Reward {
                            rf['name']}: {
                            reward:.4f} (weighted: {
                            weighted_reward:.4f})")

                except Exception as e:
                    logger.warning(f"Error computing reward {rf['name']}: {e}")
                    logger.debug(f"Error details:", exc_info=True)

            batch_rewards.append(total_reward)

        # Log batch statistics with completions summary
        if batch_rewards:
            successful = sum(1 for r in batch_rewards if r > 0.5)
            partial = sum(1 for r in batch_rewards if 0 < r <= 0.5)
            failed = sum(1 for r in batch_rewards if r <= 0)

            print(f"\n{'=' * 60}")
            print(
                f"BATCH REWARDS: {successful} passed | {partial} partial | {failed} failed | total={
                    len(batch_rewards)}")
            print(
                f"Reward stats: min={
                    min(batch_rewards):.2f}, max={
                    max(batch_rewards):.2f}, mean={
                    sum(batch_rewards) /
                    len(batch_rewards):.2f}")
            print(f"{'=' * 60}\n")

        return batch_rewards

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step."""
        logger.debug(
            "train_step() called but TRL GRPO uses TRL's internal training loop")
        return {"loss": 0.0}

    def create_data_loader(self) -> Optional[DataLoader]:
        """Create data loader for training."""
        logger.debug(
            "create_data_loader() called but TRL GRPO uses TRL's internal data loading")
        return None

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
        if self.eval_dataset:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
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
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup GRPO trainer
        from trl import GRPOTrainer, GRPOConfig

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
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            max_steps=max_steps,

            # GRPO-specific
            beta=beta,

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


        # Create GRPO trainer
        import os
        should_use_pure_trl = os.environ.get('PURE_TRL_MODE', '0') == '1'

        if should_use_pure_trl:
            logger.info("PURE_TRL_MODE enabled - using pure TRL API")
            self.trainer = GRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                reward_funcs=self._combined_reward_function,
            )
        else:
            try:
                import unsloth
                logger.info(
                    "Detected Unsloth environment, using reward_funcs parameter")

                def unsloth_reward_wrapper(
                        prompts=None, completions=None, **kwargs):
                    if completions is None:
                        return [0.0] * (len(prompts) if prompts else 1)
                    return self._combined_reward_function(
                        completions, **kwargs)

                self.trainer = GRPOTrainer(
                    model=self.model,
                    args=grpo_config,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    processing_class=self.tokenizer,
                    reward_funcs=[unsloth_reward_wrapper],
                )
            except ImportError:
                logger.info("Using pure TRL, using reward_function parameter")
                self.trainer = GRPOTrainer(
                    model=self.model,
                    args=grpo_config,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    processing_class=self.tokenizer,
                    reward_funcs=self._combined_reward_function,
                )

        logger.info("GRPO trainer setup completed successfully!")

    def train(self) -> Dict[str, Any]:
        """Execute GRPO training."""
        # Setup components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()
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
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info("=" * 80)

        # Start training
        train_result = self.trainer.train()

        # Log samples after training
        try:
            # For qualitative samples we only need callable reward functions,
            # not the full metadata dicts stored in self.reward_functions.
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
                    # Fallback: disable reward logging rather than failing
                    # samples
                    reward_callables = None

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

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained GRPO model."""
        try:
            if isinstance(self.config.logging, dict):
                default_path = self.config.logging.get(
                    'output_dir', './output/grpo')
            else:
                default_path = getattr(self.config.logging,
                                       'output_dir',
                                       './output/grpo') if hasattr(self.config,
                                                                   'logging') else './output/grpo'

            save_path = path or default_path

            logger.info(f"Saving Unsloth GRPO model to: {save_path}")

            # Save using Unsloth's optimized saving
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training configuration
            config_path = Path(save_path) / "grpo_training_config.yaml"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(
                    self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GRPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GRPO model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained GRPO model."""
        try:
            logger.info(f"Loading Unsloth GRPO model from: {path}")

            import unsloth
            from unsloth import FastLanguageModel

            # Handle both dict and object config
            if isinstance(self.config.model, dict):
                max_seq_length = self.config.model.get('max_seq_length', 2048)
            else:
                max_seq_length = getattr(
                    self.config.model, 'max_seq_length', 2048)

            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            logger.info("GRPO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GRPO model: {e}")
            raise

