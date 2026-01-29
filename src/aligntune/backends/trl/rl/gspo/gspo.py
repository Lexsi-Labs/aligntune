"""
TRL GSPO Backend Implementation.

This module provides a TRL backend for Generalized Scoring Proximal Objective (GSPO).

Note: TRL doesn't have a native GSPOTrainer. GSPO uses sequence-level importance sampling
with length normalization: w^GSPO_i = [π_θ(y_i|x) / π_θ_old(y_i|x)]^(1/|y_i|)

This implementation uses GRPOTrainer as an approximation, since true GSPO would require
custom sequence-level importance sampling which is not natively supported by TRL.
GRPOTrainer uses token-level importance sampling, which is the closest available option.

Includes all logging, debugging, and miscellaneous features from legacy implementation.
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
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.core.sft.evaluator import EnhancedEvaluator
from aligntune.utils.config_extractor import  extract_extra_and_missing_params

logger = logging.getLogger(__name__)


@staticmethod
def _extract_prompt_text(prompt):
    """Extract text from prompt (handles both string and chat format)."""
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and len(prompt) > 0:
        last = prompt[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return str(prompt)


class TRLGSPOTrainer(TrainerBase):
    """GSPO trainer using custom implementation based on TRL patterns."""

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.train_dataset = None
        self.eval_dataset = None  # Add eval dataset support
        self.reward_functions = []
        self.training_history = []
        self.logging_manager = None
        self.dataset_dict = None
        # self.evaluator = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL is available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
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

    def setup_model(self) -> None:
        """Setup model using standard Transformers."""
        logger.info("=" * 80)
        logger.info(
            f"Setting up TRL GSPO model: {
                self.config.model.name_or_path}")
        logger.info("=" * 80)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers") from e

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model.name_or_path,
        #     torch_dtype=torch.float16 if self.config.model.precision == "fp16" else torch.float32,
        #     device_map=self.config.model.device_map or "auto",
        #     trust_remote_code=self.config.model.trust_remote_code,
        #     load_in_4bit=self.config.model.load_in_4bit,
        #     load_in_8bit=self.config.model.load_in_8bit,
        # )

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.config.model.name_or_path,
        #     trust_remote_code=self.config.model.trust_remote_code,
        # )

        # Safely get attributes with defaults
        model_name = getattr(self.config.model, "name_or_path", "gpt2")
        precision = getattr(self.config.model, "precision", "fp16")
        device_map = getattr(self.config.model, "device_map", "auto")
        import os
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
        trust_remote_code = getattr(
            self.config.model, "trust_remote_code", False)
        load_in_4bit = getattr(self.config.model, "load_in_4bit", False)
        load_in_8bit = getattr(self.config.model, "load_in_8bit", False)

        # Load model with fallbacks
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if precision == "fp16" else torch.float32,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        # Load tokenizer with fallback
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if "olmo" in model_name.lower():
            self.tokenizer.return_token_type_ids = False
            if hasattr(self.tokenizer, 'model_input_names'):
                self.tokenizer.model_input_names = [
                    name for name in self.tokenizer.model_input_names
                    if name != 'token_type_ids'
                ]
            self.model.config.use_cache = False

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")

        # Apply PEFT if specified
        if self.config.model.use_peft:
            logger.info("Applying PEFT (LoRA) configuration...")
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=self.config.model.lora_r or 16,
                lora_alpha=self.config.model.lora_alpha or 16,
                target_modules=self.config.model.lora_target_modules or [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj"],
                lora_dropout=self.config.model.lora_dropout or 0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, peft_config)
            logger.info("PEFT adapters applied successfully")

        logger.info("=" * 80)
        logger.info("TRL GSPO model setup completed successfully")
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
        
        max_eval_samples =  self._get_config_value(dataset_config, 'max_eval_samples', default=None)
        # Extract train and validation splits
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
        """Setup reward functions for GSPO training."""
        logger.info("Setting up TRL GSPO reward functions...")

        self.reward_functions = []

        # Get rewards config
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(
                self.config.rewards, list) else []

        if not rewards_config:
            logger.warning(
                "No reward configurations found, using default length reward")
            rewards_config = [{"type": "length", "weight": 1.0, "params": {
                "min_length": 20, "max_length": 200}}]

        for reward_config in rewards_config:
            # Handle dict vs object attribute access
            if isinstance(reward_config, dict):
                reward_type = reward_config.get('type', 'length')
                weight = reward_config.get('weight', 1.0)
                params = reward_config.get('params', {})
            else:
                reward_type = getattr(reward_config, 'type', 'length')
                weight = getattr(reward_config, 'weight', 1.0)
                params = getattr(reward_config, 'params', {})

            try:
                # **CUSTOM REWARD HANDLING** ← Add this block
                if reward_type == 'custom' and 'reward_function' in params:
                    reward_func = params['reward_function']
                    self.reward_functions.append({
                        "function": reward_func,
                        "weight": weight,
                        "name": "custom"
                    })
                    logger.info(
                        f"Loaded custom reward function (weight: {weight})")
                    continue

                # Get reward function from registry
                reward_func = RewardRegistry.get_reward(
                    reward_type,
                    **params
                )

                # Apply weight wrapper
                if weight != 1.0:
                    def weighted_reward(
                            text, weight=weight, func=reward_func, **kwargs):
                        return weight * func(text, **kwargs)
                    reward_func = weighted_reward

                self.reward_functions.append({
                    "function": reward_func,
                    "weight": weight,
                    "name": reward_type
                })

                logger.info(f"Loaded {reward_type} reward (weight: {weight})")

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

    def setup_evaluator(self) -> None:
        """Setup EnhancedEvaluator for ROUGE/BLEU metrics."""
        try:
            from aligntune.core.sft.evaluator import EnhancedEvaluator
            from aligntune.core.sft.config import SFTConfig, EvaluationConfig, ModelConfig, DatasetConfig, TaskType

            # Create evaluation config with ROUGE/BLEU enabled
            eval_config = EvaluationConfig(
                compute_perplexity=True,
                compute_rouge=True,
                compute_bleu=True,
                compute_meteor=False,
                compute_bertscore=False,
                max_samples_for_quality_metrics=50
            )

            # Create SFT config wrapper for evaluator with proper dataclass
            # instances
            from aligntune.core.sft.config import ModelConfig, DatasetConfig

            model_config = ModelConfig(
                name_or_path=getattr(
                    self.config.model, 'name_or_path', ''), max_seq_length=getattr(
                    self.config.model, 'max_seq_length', 512))

            dataset_config = DatasetConfig(
                name='eval_dataset',  # Required field
                task_type=TaskType.TEXT_GENERATION
            )

            sft_config = SFTConfig(
                model=model_config,
                dataset=dataset_config,
                evaluation=eval_config
            )

            self.evaluator = EnhancedEvaluator(sft_config)
            logger.info("EnhancedEvaluator initialized for ROUGE/BLEU metrics")
        except Exception as e:
            logger.warning(f"Could not initialize EnhancedEvaluator: {e}")
            self.evaluator = None

    def setup_trainer(self) -> None:
        """
        Set up the GRPO trainer for GSPO training.

        Note: TRL doesn't have a native GSPOTrainer. GSPO (Group Sequential Policy Optimization)
        uses sequence-level importance sampling: w^GSPO_i = [π_θ(y_i|x) / π_θ_old(y_i|x)]^(1/|y_i|)

        We use GRPOTrainer here as the closest available implementation. True GSPO would require
        custom sequence-level importance sampling with length normalization, which is not natively
        supported by TRL's GRPOTrainer (which uses token-level importance sampling).
        """
        logger.info("Setting up TRL GSPO trainer...")
        logger.warning(
            "Note: TRL doesn't have native GSPOTrainer. Using GRPOTrainer as approximation. "
            "True GSPO requires sequence-level importance sampling with length normalization.")

        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError as e:
            raise ImportError(
                "TRL not available. Install with: pip install trl") from e
        per_device_batch_size = self._get_config_value(
            self.config.train, 'per_device_batch_size', 'batch_size', default=1)
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
        use_cache = self._get_config_value(
            self.config.train, 'sue_cache', default=True)
        # Logging parameters
        logging_steps = self._get_config_value(
            self.config.train, 'logging_steps', default=10)
        logging_strategy = self._get_config_value(
            self.config.train, 'logging_strategy', default='steps')
        # report_to = self._get_config_value(self.config.logging, 'report_to', default='none')
        report_to = self.config.logging.loggers if self.config.logging.loggers else []

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

        # Additional training parameters
        max_grad_norm = self._get_config_value(
            self.config.train, 'max_grad_norm', default=1.0)
        seed = self._get_config_value(self.config.train, 'seed', default=42)

        # Precision handling
        from aligntune.core.precision_handler import PrecisionHandler
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision_args = PrecisionHandler.get_training_args_precision(
            precision)

        # Use eval_dataset-aware defaults
        if self.eval_dataset:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
            eval_strategy = 'no'
            eval_steps = None
        if self.config.train.max_steps is None or self.config.train.max_steps <= 0:
            # Calculate max_steps from dataset size
            steps_per_epoch = len(self.train_dataset) // (
                per_device_batch_size *
                getattr(self.config.train, 'gradient_accumulation_steps', 1)
            )
            max_steps = steps_per_epoch * \
                getattr(self.config.train, 'epochs', 3)
        else:
            max_steps = self.config.train.max_steps
        # Configure GRPO trainer (used as GSPO approximation)
        grpo_config = GRPOConfig(
            output_dir=getattr(self.config.logging, 'output_dir', './outputs'),
            num_train_epochs=getattr(self.config.train, 'epochs', 3),
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=getattr(
                self.config.train, 'gradient_accumulation_steps', 1),
            learning_rate=getattr(self.config.train, 'learning_rate', 5e-5),
            warmup_steps=getattr(self.config.train, 'warmup_steps', 0),

            # Evaluation parameters
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,

            # Logging parameters
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            report_to=report_to,

            # Save parameters
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,

            # Training parameters
            weight_decay=getattr(self.config.train, 'weight_decay', 0.01),
            max_grad_norm=max_grad_norm,
            seed=seed,
            remove_unused_columns=False,

            # Generation parameters
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            importance_sampling_level = "sequence",

            # GRPO-specific (used as GSPO approximation)
            beta=getattr(self.config.train, 'kl_coef', 0.1),

            # Precision
            **precision_args,
        )

        
        missing = extract_extra_and_missing_params(
            backend_config=grpo_config,
            config=self.config,
            algorithm='grpo'
        )

        gradient_checkpointing_kwargs = missing.pop('gradient_checkpointing_kwargs', None)

        for key, value in missing.items():
            setattr(grpo_config, key, value)

        # Create combined reward function wrapper

        def combined_reward_func(prompts, completions, **kwargs):
            """
            Combine all reward functions for GSPO training.

            Args:
                prompts: List of prompt strings
                completions: List of completion strings
                **kwargs: Additional arguments from GRPOTrainer

            Returns:
                List of reward values (one per completion)
            """
            # Handle both single strings and lists
            if isinstance(completions, str):
                completions = [completions]
            if isinstance(prompts, str):
                prompts = [prompts]

            # Initialize rewards list
            num_completions = len(completions)
            total_rewards = [0.0] * num_completions
            total_weights = [0.0] * num_completions

            # Process each reward function
            for reward_item in self.reward_functions:
                if isinstance(reward_item, dict):
                    reward_func = reward_item.get('function')
                    weight = reward_item.get('weight', 1.0)
                else:
                    reward_func = reward_item
                    weight = 1.0

                if reward_func:
                    try:
                        # Compute rewards for each completion
                        for i, completion in enumerate(completions):
                            # Handle both callable and RewardFunction objects
                            if hasattr(reward_func, 'compute'):
                                reward = reward_func.compute(completion)
                            else:
                                reward = reward_func(completion)

                            # Ensure reward is a float
                            if isinstance(reward, (list, tuple)):
                                reward = sum(reward) / \
                                    len(reward) if reward else 0.0
                            elif not isinstance(reward, (int, float)):
                                reward = float(reward) if reward else 0.0

                            total_rewards[i] += reward * weight
                            total_weights[i] += weight

                    except Exception as e:
                        logger.warning(f"Reward function failed: {e}")
                        continue

            # Calculate final rewards
            final_rewards = [
                total_rewards[i] / total_weights[i] if total_weights[i] > 0 else 0.0
                for i in range(num_completions)
            ]

            return final_rewards

        
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

    def train(self) -> None:
        """
        Train using GRPOTrainer for GSPO.
        Handles all setup and training execution.
        """
        logger.info("Starting TRL GSPO training...")

        # Run all setup steps
        self.setup_model()
        self.setup_data()
        self.setup_rewards()
        self.setup_evaluator()  # Initialize EnhancedEvaluator for ROUGE/BLEU metrics
        self.setup_trainer()

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(
            "Starting TRL GSPO Training (using GRPOTrainer as approximation)")
        logger.info("=" * 80)

        # Start training
        train_result = self.trainer.train()

        # Log samples after training
        try:
            generate_and_log_samples(
                self.config.logging.sample_logging,
                self.model,
                self.tokenizer,
                getattr(self, 'reward_functions', None),
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

        # Log training metrics
        if self.logging_manager and hasattr(train_result, 'metrics'):
            self.logging_manager.log_metrics(train_result.metrics)

        # Save model
        self.trainer.save_model()

        # Add to training history
        self.training_history.append({
            'timestamp': time.time(),
            'duration': training_duration,
            'steps': getattr(train_result, 'global_step', 0),
            'task_type': 'generalized_scoring_proximal_objective',
            'model_path': self.config.logging.output_dir,
        })

        logger.info("=" * 80)
        logger.info("TRL GSPO training completed successfully!")
        logger.info("=" * 80)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute training step (handled by TRL trainer)."""
        # This is handled by TRL's GRPOTrainer internally
        return {"gspo_loss": 0.0}

    def create_data_loader(self) -> DataLoader:
        """Create data loader (handled by TRL trainer)."""
        # This is handled by TRL's GRPOTrainer internally
        return None

    def _ensure_chat_format_batched(self, examples):
        """
        BATCHED version - converts prompts to chat format for entire batch at once.
        Much faster than processing one example at a time.
        """
        prompts = examples.get('prompt', examples.get('query', []))
        system_prompts = examples.get('system_prompt', [None] * len(prompts))

        formatted_prompts = []

        for prompt, system_prompt in zip(prompts, system_prompts):
            # Already in valid chat format?
            if isinstance(prompt, list) and len(prompt) > 0:
                if isinstance(
                        prompt[0],
                        dict) and 'role' in prompt[0] and 'content' in prompt[0]:
                    formatted_prompts.append(prompt)
                    continue

            # Convert string to chat format
            messages = []
            if system_prompt:
                messages.append(
                    {"role": "system", "content": str(system_prompt)})
            messages.append({"role": "user", "content": str(prompt)})

            formatted_prompts.append(messages)

        # Update the batch
        examples['prompt'] = formatted_prompts
        return examples

    def get_sample_outputs(self) -> list:
        """Get sample outputs for logging."""
        if not hasattr(self, 'dataset') or self.train_dataset is None:
            return []

        sample_outputs = []
        for i in range(min(3, len(self.train_dataset))):
            sample = self.train_dataset[i]

            # Extract text for generation
            if "text" in sample:
                text = sample["text"]
            elif "messages" in sample:
                # Convert chat format to text
                messages = sample["messages"]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False)
            else:
                text = str(sample.get("input", ""))

            # Generate response
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            # Compute rewards for this sample
            rewards = {}
            for j, reward_func in enumerate(self.reward_functions):
                reward_value = reward_func(response)
                rewards[f"reward_{j}"] = reward_value

            sample_outputs.append({
                "input": text[:100] + "..." if len(text) > 100 else text,
                "output": response,
                "rewards": rewards,
                "model": "trl_gspo"
            })

        return sample_outputs

    def cleanup(self) -> None:
        """Cleanup TRL resources."""
        super().cleanup()

        if self.model is not None:
            # Clear model from memory
            del self.model
            torch.cuda.empty_cache()

        logger.info("TRL GSPO trainer cleanup completed")

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

    #     # SAFETY: Initialize callback handler if not exists (for standalone
    #     # eval)
    #     if self.callback_handler is None:
    #         from aligntune.core.callbacks import CallbackHandler
    #         self.callback_handler = CallbackHandler(
    #             callbacks=self.callbacks,
    #             model=self.model,
    #             tokenizer=self.tokenizer
    #         )
    #         logger.debug(
    #             "Initialized minimal callback handler for standalone evaluation")

    #     # Call parent's unified evaluate method
    #     return super().evaluate(
    #         eval_dataset=eval_dataset,
    #         metric_key_prefix=metric_key_prefix,
    #         use_custom_evaluator=use_custom_evaluator,
    #         **kwargs
    #     )

    def _generate_samples(self) -> List[Dict[str, str]]:
        """Generate qualitative samples for manual inspection."""
        samples = []

        # Prepare model for inference
        self.model.eval()

        # Sample prompts for GSPO
        sample_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important skill for the 21st century is",
            "Climate change is a challenge that requires",
            "The key to successful teamwork is"
        ]

        for i, prompt in enumerate(sample_prompts[:3]):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device)
                              for k, v in inputs.items()}

                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                generated_text = self.tokenizer.decode(
                    generated[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()

                # Compute rewards for this sample
                rewards = {}
                for j, reward_func in enumerate(self.reward_functions):
                    try:
                        reward_value = reward_func(response)
                        rewards[f"reward_{j}"] = reward_value
                    except Exception as e:
                        logger.warning(f"Reward function {j} failed: {e}")
                        rewards[f"reward_{j}"] = 0.0

                samples.append({
                    'prompt': prompt,
                    'generated_response': response,
                    'rewards': rewards
                })

                logger.info(f"Generated sample {i + 1}/3")

            except Exception as e:
                logger.warning(f"Error generating sample {i}: {str(e)}")
                samples.append({
                    'prompt': prompt,
                    'generated_response': f"Error: {str(e)}",
                    'rewards': {}
                })

        return samples

    def run_zero_shot_evaluation(self, test_prompts=None) -> Dict[str, Any]:
        """Run zero-shot evaluation before training with enhanced metrics."""
        logger.info("Running zero-shot evaluation...")

        if test_prompts is None:
            test_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning briefly.",
                "How does a computer work?",
                "What is the capital of France?",
                "Write a simple Python function."
            ]

        results = []
        self.model.eval()

        for prompt in test_prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device)
                              for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=0.7,
                        top_p=0.9
                    )

                response = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                # Compute rewards for this response
                rewards = {}
                for j, reward_func in enumerate(self.reward_functions):
                    try:
                        reward_value = reward_func(response)
                        rewards[f"reward_{j}"] = reward_value
                    except Exception as e:
                        logger.warning(f"Reward function {j} failed: {e}")
                        rewards[f"reward_{j}"] = 0.0

                results.append({
                    'prompt': prompt,
                    'response': response,
                    'response_length': len(response.split()),
                    'coherence_score': self._calculate_coherence_score(response),
                    'rewards': rewards
                })

            except Exception as e:
                logger.warning(f"Zero-shot evaluation failed for prompt: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'response_length': 0,
                    'coherence_score': 0.0,
                    'rewards': {}
                })

        # Calculate aggregate metrics
        avg_length = sum(r['response_length']
                         for r in results) / len(results) if results else 0
        avg_coherence = sum(r['coherence_score']
                            for r in results) / len(results) if results else 0
        successful_responses = len(
            [r for r in results if not r['response'].startswith('Error')])

        # Calculate average rewards
        avg_rewards = {}
        if self.reward_functions:
            for j in range(len(self.reward_functions)):
                reward_values = [
                    r['rewards'].get(
                        f'reward_{j}',
                        0.0) for r in results if not r['response'].startswith('Error')]
                avg_rewards[f'reward_{j}'] = sum(
                    reward_values) / len(reward_values) if reward_values else 0.0

        zero_shot_metrics = {
            'zero_shot_results': results,
            'avg_response_length': avg_length,
            'avg_coherence_score': avg_coherence,
            'successful_responses': successful_responses,
            'success_rate': successful_responses / len(results) if results else 0,
            'avg_rewards': avg_rewards}

        # Log zero-shot metrics
        if self.logging_manager:
            self.logging_manager.log_metrics({
                'zero_shot_avg_length': avg_length,
                'zero_shot_coherence': avg_coherence,
                'zero_shot_success_rate': successful_responses / len(results) if results else 0,
                **avg_rewards
            })

        logger.info(
            f"Zero-shot evaluation completed. Success rate: {successful_responses}/{len(results)}")
        return zero_shot_metrics

    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate a simple coherence score for generated text."""
        if not text or len(text.strip()) == 0:
            return 0.0

        # Simple heuristics for coherence
        words = text.split()
        if len(words) == 0:
            return 0.0

        # Check for repetition (lower score for high repetition)
        unique_words = len(set(words))
        repetition_penalty = unique_words / len(words)

        # Check for reasonable length (not too short or too long)
        length_score = min(
            1.0,
            len(words) /
            20.0) if len(words) < 20 else max(
            0.5,
            40.0 /
            len(words))

        # Simple readability check (penalize very short or very long sentences)
        sentences = text.split('.')
        avg_sentence_length = sum(
            len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        readability_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7

        # Combine scores
        coherence_score = (
            repetition_penalty + length_score + readability_score) / 3.0
        return min(1.0, max(0.0, coherence_score))

    def get_training_stats(self) -> Dict[str, Any]:
        """Get enhanced training statistics and information."""
        stats = {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': 'generalized_scoring_proximal_objective',
                'dataset_name': self.config.dataset.name,
                'epochs': self.config.train.num_epochs,
                'learning_rate': self.config.train.learning_rate,
                'batch_size': self.config.train.per_device_batch_size,
                'use_peft': self.config.model.use_peft,
                'precision': self.config.model.precision,
                'num_reward_functions': len(self.reward_functions),
            },
            'dataset_info': {
                'train_size': len(self.train_dataset) if hasattr(self, 'dataset') and self.train_dataset else 0,
                'val_size': 0,  # No validation dataset for now
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(next(self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
                'has_peft': hasattr(self.model, 'peft_config') if self.model else False,
            },
            'training_history': self.training_history,
        }

        return stats

    def save_config(self, path: str):
        """Save enhanced configuration to YAML file."""
        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': 'generalized_scoring_proximal_objective',
            'max_seq_length': self.config.model.max_seq_length,
            'learning_rate': self.config.train.learning_rate,
            'epochs': self.config.train.num_epochs,
            'batch_size': self.config.train.per_device_batch_size,
            'dataset_name': self.config.dataset.name,
            'use_peft': self.config.model.use_peft,
            'precision': self.config.model.precision,
            'num_reward_functions': len(self.reward_functions),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"TRL GSPO configuration saved to {path}")

    def run_experiment(self):
        """Run a complete experiment with enhanced features."""
        logger.info(
            f"Starting TRL GSPO experiment: {
                self.config.logging.run_name}")

        try:
            # Setup model, data, and rewards
            self.setup_model()
            self.setup_data()
            self.setup_rewards()

            # Run zero-shot evaluation if configured
            zero_shot_results = None
            if hasattr(
                    self.config,
                    'evaluation') and getattr(
                    self.config.evaluation,
                    'run_zero_shot',
                    False):
                logger.info("Running pre-training zero-shot evaluation...")
                zero_shot_results = self.run_zero_shot_evaluation()

            # Start training
            logger.info("Starting training...")
            start_time = time.time()
            self.train()
            end_time = time.time()
            training_duration = end_time - start_time

            # Add to training history
            self.training_history.append({
                'timestamp': time.time(),
                'duration': training_duration,
                'task_type': 'generalized_scoring_proximal_objective',
                'model_path': self.config.logging.output_dir,
            })

            # Evaluate after training
            logger.info("Running post-training evaluation...")
            eval_results = self.evaluate()

            # Combine results
            results = {
                'mode': 'train_and_eval',
                'task_type': 'generalized_scoring_proximal_objective',
                'train_results': {
                    'training_duration': training_duration},
                'eval_results': eval_results,
                'training_stats': self.get_training_stats(),
                'config': self.config.__dict__ if hasattr(
                    self.config,
                    '__dict__') else {}}

            # Add zero-shot results if available
            if zero_shot_results:
                results['zero_shot_evaluation'] = zero_shot_results

            # Save results
            output_dir = Path(self.config.logging.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "experiment_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)

            logger.info(
                f"TRL GSPO experiment completed. Results saved to {results_path}")
            return results

        except Exception as e:
            logger.error(f"TRL GSPO experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up logging resources
            if self.logging_manager:
                self.logging_manager.finish()
    

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
            config_path = Path(save_path) / "gspo_training_config"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(
                    self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GRPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GRPO model: {e}")
            raise

    