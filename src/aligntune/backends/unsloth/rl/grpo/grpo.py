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


class UnslothGRPOTrainer(TrainerBase):
    """GRPO trainer using Unsloth's FastLanguageModel for optimized training."""

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

    def setup_model(self) -> None:
        """Setup Unsloth-optimized model and tokenizer for GRPO."""
        try:
            import unsloth
            from unsloth import FastLanguageModel

            # Handle both dict and object config
            if isinstance(self.config.model, dict):
                model_name = self.config.model.get('name_or_path')
                max_seq_length = self.config.model.get('max_seq_length', 2048)
                quantization = self.config.model.get('quantization', {})
            else:
                model_name = self.config.model.name_or_path
                max_seq_length = self.config.model.max_seq_length
                quantization = getattr(self.config.model, 'quantization', {})

            logger.info(f"Setting up Unsloth GRPO model: {model_name}")

            # Configure Unsloth model parameters
            # Unsloth only accepts: max_seq_length, dtype, and load_in_4bit
            # Other quantization params are NOT passed to from_pretrained
            device_map = self.config.model.device_map or 'auto'
            model_kwargs = {
                "max_seq_length": max_seq_length,
                # Auto-detect (Unsloth will use bfloat16 automatically)
                "dtype": None,
                "load_in_4bit": quantization.get("load_in_4bit", True) if isinstance(quantization, dict) else True,
                "device_map": device_map,
            }

            logger.info(f"Loading model with kwargs: {model_kwargs}")

            # Load model with Unsloth optimizations
            # NOTE: Unsloth handles all quantization internally - don't pass
            # bnb_4bit params
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                **model_kwargs
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Configure model for GRPO training with LoRA (use config values)
            # NOTE: AlignTune configs may be dataclasses or dicts depending on how the trainer was created.
            if isinstance(self.config.model, dict):
                use_peft = self.config.model.get("use_peft", True)
                lora_r = self.config.model.get("lora_r", 16)
                lora_alpha = self.config.model.get("lora_alpha", max(16, int(lora_r)))
                lora_dropout = self.config.model.get("lora_dropout", 0.0)
                target_modules = self.config.model.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                gradient_checkpointing = self.config.model.get("gradient_checkpointing", True)
            else:
                use_peft = getattr(self.config.model, "use_peft", True)
                lora_r = getattr(self.config.model, "lora_r", 16)
                lora_alpha = getattr(self.config.model, "lora_alpha", max(16, int(lora_r)))
                lora_dropout = getattr(self.config.model, "lora_dropout", 0.0)
                target_modules = getattr(
                    self.config.model,
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                gradient_checkpointing = getattr(self.config.model, "gradient_checkpointing", True)

            if use_peft:
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=int(lora_r),
                    target_modules=target_modules,
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    bias="none",
                    use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )

            logger.info("Unsloth GRPO model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth GRPO model: {e}")
            raise

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
            reward_type = reward_config.get('type', 'length')
            weight = reward_config.get('weight', 1.0)
            params = reward_config.get('params', {})

            try:
                # ✨ SPECIAL CASE: custom reward function passed directly
                if reward_type == 'custom' and 'reward_function' in params:
                    reward_func = params['reward_function']
                    logger.info(
                        f"✓ Loaded custom reward function (weight: {weight})")

                    # Store directly - no registry needed
                    self.reward_functions.append({
                        "function": reward_func,
                        "weight": weight,
                        "name": "custom"
                    })
                    continue  # Skip registry lookup

                # Use the rewards registry to get reward functions
                from aligntune.rewards.registry import RewardRegistry as RewardsRegistry
                from aligntune.rewards.core import RewardConfig, RewardType

                # Map common variations to standard names
                reward_type_mapping = {
                    'math': 'math_reasoning',
                    'code': 'code_quality',
                }
                reward_type = reward_type_mapping.get(reward_type, reward_type)

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
                        f"✓ Loaded {reward_type} reward from registry (weight: {weight})")

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
                        f"✓ Loaded {reward_type} reward by name (weight: {weight})")

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
            'response',
            'ground_truth',
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

    def setup_data(self) -> None:
        """Setup datasets - compatible with base trainer interface."""
        self.setup_dataset()

    def setup_dataset(self) -> None:
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
        dataset_name = self._get_config_value(
            dataset_config, 'name', 'dataset_name', default='imdb')
        split = self._get_config_value(dataset_config, 'split', default=None)
        config_name = self._get_config_value(
            dataset_config, 'config_name', default=None)
        system_prompt = self._get_config_value(
            self.config.train, 'system_prompt', default=None)
        enable_thinking = self._get_config_value(
            self.config.train, 'enable_thinking', default=False)

        # Advanced DataManager features
        column_mapping = self._get_config_value(
            dataset_config, 'column_mapping', default=None)
        processing_fn = self._get_config_value(
            dataset_config, 'processing_fn', default=None)
        processing_batched = self._get_config_value(
            dataset_config, 'processing_batched', default=False)
        max_samples = self._get_config_value(
            dataset_config, 'max_samples', default=None)

        logger.info(
            f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")

        # Initialize DataManager for GRPO task
        from aligntune.data.manager import DataManager

        # Columns to preserve for reward computation (test cases, solutions,
        # etc.)
        preserve_columns = {
            'test_list',
            'test',
            'answer',
            'solution',
            'code',
            'canonical_solution'}

        manager = DataManager(
            task_type="grpo",
            system_prompt=system_prompt,
            column_mapping=column_mapping,
            processing_fn=processing_fn,
            processing_batched=processing_batched,
            max_samples = max_samples, 
            tokenizer=self.tokenizer,              # ✅ ADD THIS - Pass tokenizer for chat template
            enable_thinking=enable_thinking, 
        )

        # Load dataset - DataManager handles all the complexity
        dataset_dict = manager.load_dataset(
            dataset_name,
            config_name=config_name,
            split=split,
            max_samples=max_samples
        )

        # Extract train and validation splits
        self.train_dataset = dataset_dict.get("train", None)
       
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict 
        

        logger.info(
            f"Dataset loaded: {len(self.train_dataset)} train examples")
        if self.eval_dataset:
            logger.info(
                f"Evaluation dataset: {len(self.eval_dataset)} examples")

        # Log sample
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            prompt_col = "prompt" if "prompt" in sample else "query"
            logger.info(
                f"Sample prompt (first 100 chars): {sample[prompt_col][:100]}...")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")

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
            if self.eval_dataset:
                eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
            else:
                eval_strategy = 'no'
                eval_steps = None

            # Logging parameters
            logging_steps = self._get_config_value(
                self.config.train, 'logging_steps', default=10)
            logging_strategy = self._get_config_value(
                self.config.train, 'logging_strategy', default='steps')
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
                self.config.train, 'loss_type', default='sigmoid')
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
            # TRL GRPOConfig supports GRPO variants: "grpo", "dapo", "dr_grpo"
            VALID_GRPO_LOSS_TYPES = {"grpo", "dapo", "dr_grpo"}
            
            if loss_type is None:
                loss_type = "grpo"
            else:
                loss_type = str(loss_type).lower().strip()

            # "sigmoid" is a DPO loss type, not valid for GRPO - raise error
            if loss_type == "sigmoid":
                raise ValueError(
                    f"loss_type='sigmoid' is a DPO/IPO loss type and is not valid for GRPO training. "
                    f"Valid GRPO loss types are: {VALID_GRPO_LOSS_TYPES} use grpo"
                )

            if loss_type not in VALID_GRPO_LOSS_TYPES:
                raise ValueError(
                    f"Unknown loss_type='{loss_type}' for GRPO. "
                    f"Valid GRPO loss types are: {VALID_GRPO_LOSS_TYPES}"
                )
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
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                num_generations=num_generations,
                temperature=temperature,
                top_p=top_p,
                loss_type=loss_type,
                beta=beta,
                epsilon=epsilon,
                scale_rewards=scale_rewards,
                mask_truncated_completions=mask_truncated_completions,

                # Seeds
                seed=seed,
                data_seed=data_seed,

                # Performance
                gradient_checkpointing=gradient_checkpointing,
                group_by_length=group_by_length,
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
            # Rest of your code stays the same...

            # Get max_seq_length for text truncation in reward functions
            # Use conservative limit: most reward models (e.g., toxic-bert) have max_length=512
            # Truncate to min(512, model_max_seq_length) to prevent tensor
            # shape mismatches
            model_max_seq_length = 512  # Default
            if hasattr(self.config, 'model'):
                if isinstance(self.config.model, dict):
                    model_max_seq_length = self.config.model.get(
                        'max_seq_length', 512)
                else:
                    model_max_seq_length = getattr(
                        self.config.model, 'max_seq_length', 512)
            # Use conservative 512 limit for reward functions (most reward
            # models expect this)
            max_seq_length = min(512, model_max_seq_length)

            # # Create trainer with reward functions
            # self.trainer = GRPOTrainer(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     train_dataset=self.train_dataset,
            #     eval_dataset=self.eval_dataset,
            #     args=grpo_config,
            #     reward_funcs=self._combined_reward_function if self.reward_functions else None,
            # )
            # # Create GRPO trainer

            use_rewards_directly = self._get_config_value(
                self.config.train, 
                'use_rewards_directly', 
                default=None
            )

            if use_rewards_directly:
                # Use functions directly (no wrapper)
                reward_funcs = use_rewards_directly
                logger.info(f"✓ Using {len(reward_funcs)} reward functions DIRECTLY")
            else:
                # Use combined wrapper (default)
                reward_funcs = self._combined_reward_function
                logger.info("✓ Using _combined_reward_function wrapper")
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
                    reward_funcs=reward_funcs if isinstance(reward_funcs, list) else [reward_funcs],
                )
            else:
                try:
                    import unsloth
                    logger.info("Detected Unsloth environment")

                    # Unsloth expects a list of functions
                    if isinstance(reward_funcs, list):
                        # Already a list (direct mode)
                        logger.info(f"Passing {len(reward_funcs)} functions directly")
                        self.trainer = GRPOTrainer(
                            model=self.model,
                            args=grpo_config,
                            train_dataset=self.train_dataset,
                            eval_dataset=self.eval_dataset,
                            processing_class=self.tokenizer,
                            reward_funcs=reward_funcs,  # List of functions
                        )
                    else:
                        # Single function (wrapper mode) - wrap it for Unsloth
                        # logger.info("Wrapping _combined_reward_function for Unsloth")
                        # def unsloth_reward_wrapper(prompts=None, completions=None, **kwargs):
                        #     if completions is None:
                        #         return [0.0] * (len(prompts) if prompts else 1)
                        #     return reward_funcs(completions, **kwargs)

                        self.trainer = GRPOTrainer(
                            model=self.model,
                            args=grpo_config,
                            train_dataset=self.train_dataset,
                            eval_dataset=self.eval_dataset,
                            processing_class=self.tokenizer,
                            reward_funcs=reward_funcs,  # List with wrapper
                        )
                        
                except ImportError:
                    logger.info("Using pure TRL (no Unsloth)")
                    
                    # Pure TRL can accept either format
                    if isinstance(reward_funcs, list):
                        # Direct mode: pass list as-is
                        self.trainer = GRPOTrainer(
                            model=self.model,
                            args=grpo_config,
                            train_dataset=self.train_dataset,
                            eval_dataset=self.eval_dataset,
                            processing_class=self.tokenizer,
                            reward_funcs=reward_funcs,
                        )
                    else:
                        # Wrapper mode: wrap in list
                        self.trainer = GRPOTrainer(
                            model=self.model,
                            args=grpo_config,
                            train_dataset=self.train_dataset,
                            eval_dataset=self.eval_dataset,
                            processing_class=self.tokenizer,
                            reward_funcs=[reward_funcs],  # Wrap single function in list
                        )

            logger.info("GRPO trainer setup completed successfully!")

            logger.info("TRL GRPOTrainer setup completed")

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
            use_rewards_directly = self._get_config_value(
                self.config.train, 'use_rewards_directly', default=None
            )
            if not use_rewards_directly:
                self.setup_rewards()
            else:
                logger.warning("Skipping setup_rewards() - using reward functions directly from config")
            self.setup_trainer()

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
                f"Unsloth GRPO training completed in {
                    training_time:.2f} seconds")
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
