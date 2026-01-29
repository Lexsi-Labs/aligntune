"""
GBMPO TRL Trainer Interface - Unified API for Generalized Bregman Mirror Descent Policy Optimization

This module provides a clean interface for all GBMPO variants:
- L2-GRPO: L2 norm regularization (log-space)
- L2KL-GRPO: Dual L2 + KL divergence
- ProbL2-GRPO: L2 norm in probability space
- ProbL2KL-GRPO: Dual probability-space L2 + KL

Usage matches TRL's GRPOTrainer pattern for easy integration.
"""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from aligntune.utils.config_extractor import  extract_extra_and_missing_params
# Import GBMPO trainers from gbmpo_utils
from .gbmpo_trainer import (
    L2GRPOConfig, L2GRPOTrainer,
    L2KLGRPOConfig, L2KLGRPOTrainer,
    ProbL2GRPOConfig, ProbL2GRPOTrainer,
    ProbL2KLGRPOConfig, ProbL2KLGRPOTrainer,
)

# Import reward functions if available
try:
    from aligntune.core.rl.trainer_base import TrainerBase
    from aligntune.core.rl.config import UnifiedConfig
    from aligntune.rewards.registry import RewardRegistry
    from aligntune.rewards.core import RewardConfig, RewardType
    FINETUNEHUB_AVAILABLE = True
except ImportError:
    FINETUNEHUB_AVAILABLE = False
    TrainerBase = object
    UnifiedConfig = dict

from aligntune.core.precision_handler import PrecisionHandler

logger = logging.getLogger(__name__)


# ============================================================================
# GBMPO Trainer Wrapper
# ============================================================================

class TRLGBMPOTrainer(TrainerBase):
    """Unified GBMPO trainer that wraps all GBMPO variants."""
    # Map divergence types to their trainer classes
    TRAINER_MAP = {
        'l2': (L2GRPOConfig, L2GRPOTrainer),
        'l2kl': (L2KLGRPOConfig, L2KLGRPOTrainer),
        'prob_l2': (ProbL2GRPOConfig, ProbL2GRPOTrainer),
        'prob_l2kl': (ProbL2KLGRPOConfig, ProbL2KLGRPOTrainer),
    }

    def __init__(
        self,
        config: Optional[Any] = None,
        divergence_type: Optional[str] = None,  # Make optional
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        reward_funcs: Optional[Any] = None,
        **kwargs
    ):
        """Initialize GBMPO trainer with specified divergence type."""
        # Handle config
        super().__init__(config)
        if isinstance(config, dict):
            self.config = type('Config', (), config)()
        else:
            self.config = config or type('Config', (), {})()

         # Extract divergence_type from config with priority order
        divergence_type = None

        # Priority 1: Explicit kwarg
        if 'divergence_type' in kwargs:
            divergence_type = kwargs.pop('divergence_type')

        # Priority 2: From config.train.gbmpo_divergence_type
        elif hasattr(self.config, 'train'):
            divergence_type = getattr(
                self.config.train, 'gbmpo_divergence_type', None)

        # Priority 3: From kwargs gbmpo_divergence_type
        elif 'gbmpo_divergence_type' in kwargs:
            divergence_type = kwargs.pop('gbmpo_divergence_type')

        # Priority 4: Infer from algo name if present
        if divergence_type is None and hasattr(self.config, 'algo'):
            algo = str(self.config.algo).lower()
            if algo.startswith('gbmpo_'):
                suffix = algo.replace('gbmpo_', '')
                divergence_map = {
                    'l2': 'l2',
                    'l2kl': 'l2kl',
                    'probl2': 'prob_l2',
                    'probl2kl': 'prob_l2kl'
                }
                divergence_type = divergence_map.get(suffix)

        # Default: Use l2kl (best general performance)
        if divergence_type is None:
            divergence_type = 'l2kl'
            logger.info("No divergence_type specified, defaulting to 'l2kl'")

        # Validate divergence type
        if divergence_type not in self.TRAINER_MAP:
            raise ValueError(
                f"Unknown divergence_type: '{divergence_type}'. "
                f"Must be one of: {list(self.TRAINER_MAP.keys())}"
            )

        self.divergence_type = divergence_type

        self.model_input = model
        self.tokenizer_input = tokenizer
        self.dataset_input = train_dataset
        self.reward_funcs_input = reward_funcs
        self.extra_kwargs = kwargs

        # Initialize components
        self.model = None
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.reward_functions = []
        self.trainer = None
        self.training_history = []
        self.custom_evaluator = None  # For BaseEvaluator/RLEvaluator
        self.dataset_dict = None
        # Validate divergence type
        if divergence_type not in self.TRAINER_MAP:
            raise ValueError(
                f"Unknown divergence_type: {divergence_type}. "
                f"Must be one of: {list(self.TRAINER_MAP.keys())}"
            )

        logger.info(
            f"Initializing GBMPO Trainer with divergence_type: {divergence_type}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if GBMPO trainers are available."""
        try:
            # Check for required imports
            from .gbmpo_trainer import (
                L2GRPOTrainer, L2KLGRPOTrainer,
                ProbL2GRPOTrainer, ProbL2KLGRPOTrainer
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError as e:
            logger.warning(f"GBMPO dependencies not available: {e}")
            return False

    def setup_model(self) -> None:
        """Setup model and tokenizer."""
        logger.info("=" * 80)
        logger.info("Setting up GBMPO model and tokenizer")
        logger.info("=" * 80)

        # Extract model name - FIX: Ensure we get a string, not the config
        # object
        model_name = None

        # If model already provided, use it
        if self.model_input is not None:
            if isinstance(self.model_input, str):
                model_name = self.model_input
            else:
                self.model = self.model_input
                logger.info(
                    f"Using provided model: {
                        type(
                            self.model).__name__}")
        else:
            # Extract from config - FIX: Handle config object properly
            if hasattr(self.config, 'model'):
                model_config = self.config.model
                # If it's a config object, extract the name_or_path attribute
                if hasattr(model_config, 'name_or_path'):
                    model_name = model_config.name_or_path
                elif isinstance(model_config, dict):
                    model_name = model_config.get(
                        'name_or_path') or model_config.get('model_name', 'gpt2')
                elif isinstance(model_config, str):
                    model_name = model_config
                else:
                    # Fallback: try _get_config_value
                    model_name = self._get_config_value(
                        self.config.model, 'name_or_path', 'model_name', default='gpt2')
            else:
                model_name = 'gpt2'

        # Ensure model_name is a string
        if model_name is not None and not isinstance(model_name, str):
            logger.error(
                f"model_name is not a string: {
                    type(model_name)} - {model_name}")
            raise ValueError(
                f"model_name must be a string, got {
                    type(model_name)}")

        logger.info(f"Model name extracted: {model_name}")

        # Load tokenizer
        if self.tokenizer_input is not None:
            self.tokenizer = self.tokenizer_input
            logger.info(f"Using provided tokenizer")
        elif model_name is not None:
            logger.info(f"Loading tokenizer from {model_name}")
            trust_remote_code = self._get_config_value(
                self.config, 'trust_remote_code', default=False)
            if hasattr(self.config, 'model'):
                trust_remote_code = self._get_config_value(
                    self.config.model, 'trust_remote_code', default=trust_remote_code)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad token to eos token")

        # Load model if not already loaded
        if self.model is None and model_name is not None:
            logger.info(f"Loading model from {model_name}")

            # === UNIFIED PRECISION HANDLING ===
            precision = PrecisionHandler.get_precision_from_config(
                self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "GBMPO")
            dtype = PrecisionHandler.get_torch_dtype(precision)

            # Check for quantization
            load_in_4bit = self._get_config_value(
                self.config, 'load_in_4bit', default=False)
            load_in_8bit = self._get_config_value(
                self.config, 'load_in_8bit', default=False)

            # Also check model config if exists
            if hasattr(self.config, 'model'):
                load_in_4bit = self._get_config_value(
                    self.config.model, 'load_in_4bit', default=load_in_4bit)
                load_in_8bit = self._get_config_value(
                    self.config.model, 'load_in_8bit', default=load_in_8bit)

            device_map = self._get_config_value(
                self.config, 'device_map', default='auto')

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

            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": self._get_config_value(
                    self.config,
                    'trust_remote_code',
                    default=False),
            }

            # Override with model-specific config if exists
            if hasattr(self.config, 'model'):
                model_kwargs["device_map"] = self._get_config_value(
                    self.config.model, 'device_map', default=model_kwargs["device_map"])
                model_kwargs["trust_remote_code"] = self._get_config_value(
                    self.config.model, 'trust_remote_code', default=model_kwargs["trust_remote_code"])

            if load_in_4bit:
                logger.info("Loading with 4-bit quantization")
                model_kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                logger.info("Loading with 8-bit quantization")
                model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs)

            # Apply PEFT if specified
            use_peft = self._get_config_value(
                self.config, 'use_peft', default=False)
            if hasattr(self.config, 'model'):
                use_peft = self._get_config_value(
                    self.config.model, 'use_peft', default=use_peft)

            if use_peft or load_in_4bit or load_in_8bit:
                self._apply_peft()

        # Enable training mode features
        if self.model is not None:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        logger.info("=" * 80)
        logger.info("Model and tokenizer setup completed")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        if self.model is not None:
            logger.info(
                f"Model device: {
                    next(
                        self.model.parameters()).device}")
        logger.info("=" * 80)

    def _apply_peft(self):
        """Apply PEFT/LoRA to the model."""
        logger.info("Applying PEFT (LoRA) configuration...")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Prepare for k-bit training if quantized
        load_in_4bit = self._get_config_value(
            self.config, 'load_in_4bit', default=False)
        load_in_8bit = self._get_config_value(
            self.config, 'load_in_8bit', default=False)

        if load_in_4bit or load_in_8bit:
            logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self._get_config_value(
                self.config,
                'lora_r',
                'r',
                default=16),
            lora_alpha=self._get_config_value(
                self.config,
                'lora_alpha',
                'alpha',
                default=32),
            target_modules=self._get_config_value(
                self.config,
                'lora_target_modules',
                'target_modules',
                default=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj"]),
            lora_dropout=self._get_config_value(
                self.config,
                'lora_dropout',
                'dropout',
                default=0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("PEFT adapters applied successfully")

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
            max_samples = max_samples, 
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
        """Setup reward functions."""
        logger.info("Setting up reward functions for GBMPO...")

        # If reward functions already provided, use them
        if self.reward_funcs_input is not None:
            if callable(self.reward_funcs_input):
                self.reward_functions = [self.reward_funcs_input]
            elif isinstance(self.reward_funcs_input, list):
                self.reward_functions = self.reward_funcs_input
            else:
                raise ValueError(
                    f"reward_funcs must be callable or list, got {
                        type(
                            self.reward_funcs_input)}")
            logger.info(
                f"Using {len(self.reward_functions)} provided reward function(s)")
            return

        # Load from config
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(
                self.config.rewards, list) else []

        if not rewards_config:
            logger.warning(
                "No reward configurations found, using default length reward")
            rewards_config = [{"type": "length", "weight": 1.0, "params": {
                "min_length": 20, "max_length": 200}}]

        # Load rewards from registry if available
        if FINETUNEHUB_AVAILABLE:
            self._load_rewards_from_registry(rewards_config)
        else:
            self._create_simple_reward_function()

        logger.info(
            f"✓ Configured {len(self.reward_functions)} reward function(s)")

    def _load_rewards_from_registry(self, rewards_config):
        """Load reward functions from AlignTune registry."""
        for reward_config in rewards_config:
            reward_type = reward_config.get('type', 'length')
            weight = reward_config.get('weight', 1.0)
            params = reward_config.get('params', {})

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

                # Map common variations to standard names
                reward_type_mapping = {
                    'math': 'math_reasoning',
                    'code': 'code_quality',
                }
                reward_type = reward_type_mapping.get(reward_type, reward_type)

                try:
                    reward_type_enum = RewardType[reward_type.upper()]
                    reward_cfg = RewardConfig(
                        reward_type=reward_type_enum,
                        weight=1.0,
                        params=params
                    )
                    reward_func_obj = RewardRegistry.get_reward_function(
                        reward_type, reward_cfg)
                except KeyError:
                    reward_func_obj = RewardRegistry.get_reward_function(
                        reward_type)

                if hasattr(reward_func_obj, 'compute'):
                    reward_func = reward_func_obj.compute
                elif callable(reward_func_obj):
                    reward_func = reward_func_obj
                else:
                    logger.warning(
                        f"Reward function '{reward_type}' is not callable, skipping")
                    continue

                self.reward_functions.append({
                    "function": reward_func,
                    "weight": weight,
                    "name": reward_type
                })

                logger.info(f"Loaded {reward_type} reward (weight: {weight})")

            except Exception as e:
                logger.warning(
                    f"Failed to load reward function '{reward_type}': {e}")
                continue

    def _create_simple_reward_function(self):
        """Create a simple default reward function."""
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
        logger.info("Created default length reward function")

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

    def setup_trainer(self) -> None:
        """Setup the GBMPO trainer with all configurations."""
        from aligntune.core.optimization import get_optimizer_for_config, get_scheduler_for_config

        # Get trainer config and class for divergence type
        config_class, trainer_class = self.TRAINER_MAP[self.divergence_type]

        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision_args = PrecisionHandler.get_training_args_precision(
            precision)

        # Extract training parameters
        output_dir = self._get_config_value(
            self.config.logging, 'output_dir', default='./output/gbmpo')
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
        num_generations = self._get_config_value(
            self.config.train, 'num_generations', default=4)
        # Optimizer and scheduler configurations
        optimizer_name = self._get_config_value(
            self.config.train, 'optimizer', default='adamw_torch')
        scheduler_name = self._get_config_value(
            self.config.train, 'lr_scheduler', default='cosine')
        weight_decay = self._get_config_value(
            self.config.train, 'weight_decay', default=0.01)

        # Get optimizer config
        optimizer_config = get_optimizer_for_config(
            optimizer_name,
            learning_rate,
            weight_decay
        )

        # Calculate max_steps for scheduler
        max_steps = self._get_config_value(self.config.train, 'max_steps')
        if max_steps is None and self.train_dataset:
            dataset_size = len(self.train_dataset)
            effective_batch_size = per_device_batch_size * gradient_accumulation_steps
            max_steps = int(num_epochs * dataset_size / effective_batch_size)
        else:
            max_steps = max_steps or 1000

        # Get scheduler config
        warmup_steps = self._get_config_value(
            self.config.train, 'warmup_steps', default=0)
        warmup_ratio = self._get_config_value(
            self.config.train, 'warmup_ratio')
        if warmup_ratio:
            warmup_steps = int(max_steps * warmup_ratio)

        scheduler_config = get_scheduler_for_config(
            scheduler_name,
            max_steps,
            warmup_steps
        )

        # Helper for converting kwargs to string
        def kwargs_to_str(kwargs_dict):
            if not kwargs_dict:
                return None
            return ",".join(
                f"{k}={f'({','.join(map(str, v))})' if isinstance(v, tuple) else v}"
                for k, v in kwargs_dict.items()
            )

        optim_args_str = kwargs_to_str(optimizer_config['optimizer_kwargs'])

        # Logging, eval, and save parameters
        logging_steps = self._get_config_value(
            self.config.train, 'logging_steps', default=10)
        eval_steps = self._get_config_value(
            self.config.train, 'eval_steps', default=100)
        eval_strategy = self._get_config_value(
            self.config.train, 'eval_strategy', default='no')
        save_steps = self._get_config_value(
            self.config.train, 'save_steps', default=100)
        save_strategy = self._get_config_value(
            self.config.train, 'save_strategy', default='steps')

        # GBMPO-specific parameters
        train_config = getattr(self.config, 'train', None)
        if train_config:
            l2_coefficient = getattr(
                train_config, 'gbmpo_l2_coefficient', 0.0001)
            epsilon = getattr(train_config, 'gbmpo_epsilon', 0.2)
            beta = getattr(train_config, 'beta', 0.01)
        else:
            l2_coefficient = 0.0001
            epsilon = 0.2
            beta = 0.01

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
        # report_to = self._get_config_value(self.config.logging, 'report_to', default=None)
        report_to = self.config.logging.loggers if self.config.logging.loggers else []

        # Generation parameters
        max_completion_length = self._get_config_value(
            self.config.train, 'max_completion_length', 'max_new_tokens', default=256)
        max_prompt_length = self._get_config_value(
            self.config.train, 'max_prompt_length', default=512)
        temperature = self._get_config_value(
            self.config.train, 'temperature', default=0.7)
        top_p = self._get_config_value(
            self.config.train, 'top_p', default=0.95)

        # Seed
        seed = self._get_config_value(self.config.train, 'seed', default=42)

        # Max grad norm
        max_grad_norm = self._get_config_value(
            self.config.train, 'max_grad_norm', default=1.0)

        # Use eval_dataset-aware defaults
        if self.eval_dataset:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
            eval_strategy = 'no'
            eval_steps = None

        loss_type = self._get_config_value(
            self.config.train, 'loss_type', default='dapo')
        if loss_type == 'sigmoid':
            print('Sigmoid not supported')
            loss_type = 'dapo'

        logger.info("=" * 80)
        logger.info(
            f"GBMPO Training Configuration ({
                self.divergence_type.upper()})")
        logger.info(f"Divergence type: {self.divergence_type}")
        logger.info(f"L2 coefficient: {l2_coefficient}")
        logger.info(f"Beta (KL): {beta}")
        logger.info(f"Epsilon (clip): {epsilon}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Optimizer: {optimizer_name}")
        logger.info(f"Scheduler: {scheduler_name}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {per_device_batch_size}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create trainer config
        trainer_config = config_class(
            output_dir=output_dir,
            divergence_type=self.divergence_type,
            l2_coefficient=l2_coefficient,
            beta=beta,
            epsilon=epsilon,

            # Training parameters
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            seed=seed,

            # Optimizer and scheduler
            optim=optimizer_name,
            optim_args=optim_args_str,
            lr_scheduler_type=scheduler_name,
            weight_decay=weight_decay,

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

            # Generation parameters
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            loss_type=loss_type,

            # Other
            remove_unused_columns=False,
            **precision_args,
        )

        
        missing = extract_extra_and_missing_params(
            backend_config=trainer_config,
            config=self.config,
            algorithm='grpo'
        )

        for key, value in missing.items():
            setattr(trainer_config, key, value)

        # Create trainer instance
        logger.info(f"Creating {trainer_class.__name__}...")
        import os
        should_use_pure_trl = os.environ.get('PURE_TRL_MODE', '0') == '1'

        if should_use_pure_trl:
            logger.info(
                "PURE_TRL_MODE enabled - using pure TRL API with CounterfactualGRPOTrainer")
            self.trainer = trainer_class(
                model=self.model,
                args=trainer_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                reward_funcs=self._combined_reward_function if self.reward_functions else None,
            )

        # Store configuration for later use
        self._training_config = {
            'output_dir': output_dir,
            'divergence_type': self.
            divergence_type,
            'l2_coefficient': l2_coefficient,
            'beta': beta,
        }

    def train(self) -> Dict[str, Any]:
        """Execute GBMPO training."""
        # Setup all components
        if self.model is None:
            self.setup_model()
        if not self.reward_functions:
            self.setup_rewards()
        if self.train_dataset is None:
            self.setup_data()
        self.setup_trainer()

        # Train
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"Starting {self.divergence_type.upper()} Training")
        logger.info("=" * 80)

        train_result = self.trainer.train()

        end_time = time.time()
        training_duration = end_time - start_time

        logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Save model
        output_dir = self._training_config['output_dir']
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Compile results
        results = {
            "training_time": training_duration,
            "final_loss": train_result.training_loss if hasattr(
                train_result,
                'training_loss') else 0.0,
            "total_steps": train_result.global_step if hasattr(
                train_result,
                'global_step') else 0,
            "model_path": output_dir,
            "divergence_type": self._training_config['divergence_type'],
            "l2_coefficient": self._training_config['l2_coefficient'],
            "beta": self._training_config['beta'],
        }

        logger.info("=" * 80)
        logger.info(f"{self.divergence_type.upper()} Training Completed!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)

        return results

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step."""
        logger.debug(
            "train_step() called but Counterfactual GRPO uses TRL's internal training loop")
        return {"loss": 0.0}

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

    # def evaluate(self) -> Dict[str, Any]:
    #     """Evaluate the trained model."""
    #     if self.trainer and hasattr(self.trainer, 'evaluate'):
    #         return self.trainer.evaluate()
    #     return {}

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
        return {
            'divergence_type': self.divergence_type,
            'model_loaded': self.model is not None,
            'dataset_size': len(self.train_dataset) if self.train_dataset else 0,
            'num_reward_functions': len(self.reward_functions),
        }
        
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
            config_path = Path(save_path) / "gmbpo_training_config"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(
                    self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GRPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GRPO model: {e}")
            raise
