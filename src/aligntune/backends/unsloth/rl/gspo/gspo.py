"""
Unsloth GSPO Backend Implementation.

This module provides an Unsloth-optimized backend for Generalized Scoring Proximal Objective (GSPO),
using Unsloth's FastLanguageModel for optimized model loading and TRL's GRPOTrainer
for the training loop (as GSPO approximation). This provides 2-5x speed improvements.

Note: TRL doesn't have a native GSPOTrainer. GSPO uses sequence-level importance sampling
with length normalization: w^GSPO_i = [π_θ(y_i|x) / π_θ_old(y_i|x)]^(1/|y_i|)

This implementation uses GRPOTrainer as an approximation, since true GSPO would require
custom sequence-level importance sampling which is not natively supported by TRL.
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
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.utils.config_extractor import  extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class UnslothGSPOTrainer(TrainerBase):
    """GSPO trainer using Unsloth's FastLanguageModel for optimized training."""

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.unsloth_model = None
        self.reward_functions = []
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
                'device_map': device_map,
                "trust_remote_code": self._get_config_value(self.config.model, 'trust_remote_code', default=False)
            }

            logger.info(f"Loading model with kwargs: {model_kwargs}")

            # Load model with Unsloth optimizations
            # NOTE: Unsloth handles all quantization internally - don't pass bnb_4bit params
            # if "olmo" in model_name.lower():
            #     logger.info("Applying OLMo compatibility patches before loading...")
            #     try:
            #         from transformers import AutoModelForCausalLM, PreTrainedModel

            #         # Import OLMo model class and add missing push_to_hub method
            #         olmo_module = __import__('transformers.models.olmo.modeling_olmo', fromlist=['OLMo'])
            #         OLMo = olmo_module.OLMo

            #         # Add push_to_hub to the OLMo class itself (not instance)
            #         if not hasattr(OLMo, 'push_to_hub'):
            #             OLMo.push_to_hub = PreTrainedModel.push_to_hub
            #             logger.debug("Added push_to_hub to OLMo class")
            #     except Exception as e:
            #         logger.warning(f"Could not pre-patch OLMo: {e}")

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                **model_kwargs
            )

            # OLMo tokenizer fix (must be after loading)
            if "olmo" in model_name.lower():
                self.tokenizer.return_token_type_ids = False
                logger.debug("Disabled token_type_ids for OLMo compatibility")
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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
            # Configure model for GRPO training with LoRA
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_r,  # LoRA rank
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            logger.info("Unsloth GRPO model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth GRPO model: {e}")
            raise

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
                # **CUSTOM REWARD HANDLING**
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

    def setup_data(self) -> None:
        """Setup datasets for GSPO training using unified DataManager."""
        logger.info("Setting up GSPO datasets with DataManager...")

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

        logger.info(
            f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")
        
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
        # Initialize DataManager for GRPO task (GSPO uses similar format)
        from aligntune.data.manager import DataManager

        # Columns to preserve for reward computation or other GSPO-specific
        # needs
        preserve_columns = {
            'test_list',
            'test',
            'answer',
            'solution',
            'code',
            'canonical_solution',
            'response'}

        manager = DataManager(
            task_type="grpo",  # GSPO uses GRPO format (query/prompt column)
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
        self.train_dataset = dataset_dict["train"]
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict

        logger.info(f"GSPO dataset loaded: {len(self.train_dataset)} train examples")
        if self.eval_dataset:
            logger.info(
                f"Evaluation dataset: {len(self.eval_dataset)} examples")

        # Log sample for debugging
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            # GRPO format standardizes to 'prompt' column
            prompt_col = "prompt" if "prompt" in sample else "query"
            logger.info(
                f"Sample prompt (first 100 chars): {sample[prompt_col][:100]}...")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")

            # Verify GSPO-compatible format
            if prompt_col not in sample:
                logger.warning(
                    "Dataset may not be properly formatted for GSPO (missing 'prompt'/'query' column)")

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
        # if batch_rewards:
        #     successful = sum(1 for r in batch_rewards if r > 0.5)
        #     partial = sum(1 for r in batch_rewards if 0 < r <= 0.5)
        #     failed = sum(1 for r in batch_rewards if r <= 0)

        #     print(f"\n{'='*60}")
        #     print(f"BATCH REWARDS: {successful} passed | {partial} partial | {failed} failed | total={len(batch_rewards)}")
        #     print(f"Reward stats: min={min(batch_rewards):.2f}, max={max(batch_rewards):.2f}, mean={sum(batch_rewards)/len(batch_rewards):.2f}")
        #     print(f"{'='*60}\n")

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
        Setup TRL GRPOTrainer with Unsloth model (GSPO approximation).

        Note: TRL doesn't have native GSPOTrainer. Using GRPOTrainer as the closest approximation.
        """
        try:
            from trl import GRPOTrainer, GRPOConfig

            logger.info(
                "Setting up TRL GRPOTrainer with Unsloth model (GSPO approximation)")
            from aligntune.core.precision_handler import PrecisionHandler
            precision = PrecisionHandler.get_precision_from_config(
                self.config, default="auto")
            precision = PrecisionHandler.validate_precision(precision)
            PrecisionHandler.log_precision_info(precision, "GSPO (Unsloth)")
            precision_args = PrecisionHandler.get_training_args_precision(
                precision)

            # Get output directory
            output_dir = self._get_config_value(
                self.config.logging, 'output_dir', default='./output/unsloth_gspo')
            run_name = self._get_config_value(
                self.config.logging, 'run_name', default='unsloth_gspo')

            # Get training parameters - DEFINE per_device_batch_size FIRST
            per_device_batch_size = self._get_config_value(
                self.config.train, 'per_device_batch_size', default=4)
            num_epochs = self._get_config_value(
                self.config.train, 'epochs', 'num_epochs', default=1)
            if num_epochs is None:
                num_epochs = 1
            gradient_accumulation_steps = self._get_config_value(
                self.config.train, 'gradient_accumulation_steps', default=1)
            learning_rate = self._get_config_value(
                self.config.train, 'learning_rate', default=5e-5)
            max_steps = self._get_config_value(
                self.config.train, 'max_steps', default=-1)
            if max_steps is None:
                max_steps = -1

            # Evaluation parameters
            eval_strategy = self._get_config_value(
                self.config.train, 'eval_strategy', default='epoch')
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
                self.config.train, 'max_grad_norm', default=0.5)

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

            # Generation parameters
            num_generations = self._get_config_value(
                self.config.train, 'num_generations', default=4)
            max_completion_length = self._get_config_value(
                self.config.train, 'max_completion_length', 'max_new_tokens', default=256)
            max_prompt_length = self._get_config_value(
                self.config.train,
                'max_prompt_length',
                default=self.config.model.max_seq_length // 2)
            temperature = self._get_config_value(
                self.config.train, 'temperature', default=0.7)
            top_p = self._get_config_value(
                self.config.train, 'top_p', default=0.95)

            # GRPO specific (used as GSPO approximation)
            beta = self._get_config_value(
                self.config.train, 'kl_coef', 'beta', default=0.1)

            # Adjust save strategy to save on epoch if eval_dataset exists
            if self.eval_dataset:
                save_strategy = "epoch"

            # Create GRPO configuration (GSPO approximation)
            grpo_config = GRPOConfig(
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

                # Generation parameters
                num_generations=num_generations,
                max_completion_length=max_completion_length,
                max_prompt_length=max_prompt_length,
                temperature=temperature,
                top_p=top_p,

                # GRPO-specific (GSPO approximation)
                beta=beta,

                # Seeds
                seed=seed,
                data_seed=data_seed,

                # Performance
                gradient_checkpointing=gradient_checkpointing,
                group_by_length=group_by_length,
                dataloader_pin_memory=False,
                importance_sampling_level = "sequence",

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

                # Use the internal _combined_reward_function method
                return self._combined_reward_function(completions, **kwargs)

            # Create GRPO trainer (using as approximation for GSPO)
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
                    logger.info(
                        "Using pure TRL, using reward_function parameter")
                    self.trainer = GRPOTrainer(
                        model=self.model,
                        args=grpo_config,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.eval_dataset,
                        processing_class=self.tokenizer,
                        reward_funcs=self._combined_reward_function,
                    )

            logger.info("GRPO trainer setup completed successfully!")

        except Exception as e:
            logger.error(f"Failed to setup GSPO trainer: {e}")
            raise

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute training step (handled by TRL trainer)."""
        # This is handled by TRL's GRPOTrainer internally
        return {"gspo_loss": 0.0}

    def create_data_loader(self) -> DataLoader:
        """Create data loader (handled by TRL trainer)."""
        # This is handled by TRL's GRPOTrainer internally
        return None

    def compute_reward(self, text: str) -> float:
        """Compute reward for a given text using configured reward functions."""
        try:
            total_reward = 0.0
            total_weight = 0.0

            for reward_item in self.reward_functions:
                try:
                    # Handle both dict structure and RewardFunction objects
                    if isinstance(reward_item, dict):
                        reward_func = reward_item["function"]
                        weight = reward_item["weight"]
                    else:
                        # Legacy: RewardFunction object
                        reward_func = reward_item
                        weight = getattr(reward_item, 'weight', 1.0)

                    # Call the reward function
                    if callable(reward_func):
                        reward = reward_func(text)
                    elif hasattr(reward_func, 'compute'):
                        reward = reward_func.compute(text)
                    else:
                        logger.warning(
                            f"Reward function is not callable, skipping")
                        continue

                    total_reward += reward * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Reward function failed: {e}")
                    continue

            if total_weight > 0:
                return total_reward / total_weight
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Failed to compute reward: {e}")
            return 0.0

    def train(self) -> Dict[str, Any]:
        """Execute GSPO training with Unsloth optimizations."""
        try:
            logger.info("Starting Unsloth GSPO training (using GRPOTrainer)")
            logger.info("=" * 80)
            start_time = time.time()

            # Setup components
            self.setup_model()
            self.setup_rewards()
            self.setup_data()
            self.setup_evaluator()
            self.setup_trainer()

            # Start training
            training_result = self.trainer.train()

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

            # Save model
            output_dir = self._get_config_value(
                self.config.logging, 'output_dir', default='./output/unsloth_gspo')
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            training_time = time.time() - start_time

            # Add to training history
            self.training_history.append({
                'timestamp': time.time(),
                'duration': training_time,
                'steps': getattr(training_result, 'global_step', 0),
                'task_type': 'generalized_scoring_proximal_objective',
                'model_path': output_dir,
            })

            # Compile results
            results = {
                "training_time": training_time,
                "final_loss": training_result.training_loss,
                "total_steps": training_result.global_step,
                "model_path": output_dir,
                "training_history": self.training_history,
                "reward_functions": len(self.reward_functions),
            }

            logger.info("=" * 80)
            logger.info(
                f"Unsloth GSPO training completed in {
                    training_time:.2f} seconds")
            logger.info(f"Final loss: {training_result.training_loss:.4f}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"GSPO training failed: {e}")
            raise

    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     metric_key_prefix: str = "eval",
    #     use_custom_evaluator: bool = True,
    #     **kwargs
    # ) -> Dict[str, float]:
    #     """GSPO-specific evaluation - auto-setup evaluators and delegate to parent."""

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

    def generate_gspo_samples(
            self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Generate sample outputs from the trained GSPO model."""
        try:
            logger.info(f"Generating {num_samples} GSPO samples")

            # Sample prompts for generation
            sample_prompts = [
                "Explain the concept of machine learning.",
                "Write a short story about a robot.",
                "What is the capital of France?",
                "How do you make a sandwich?",
                "Explain quantum computing in simple terms.",
            ]

            samples = []
            for i, prompt in enumerate(sample_prompts[:num_samples]):
                try:
                    # Format prompt for generation
                    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

                    # Tokenize input
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Generate response
                    with torch.no_grad():
                        outputs = self.unsloth_model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    # Decode response
                    response = self.tokenizer.decode(
                        outputs[0], skip_special_tokens=True)
                    response = response[len(formatted_prompt):].strip()

                    # Compute reward
                    reward = self.compute_reward(response)

                    samples.append({
                        "prompt": prompt,
                        "response": response,
                        "reward": reward,
                        "sample_id": i + 1
                    })

                except Exception as e:
                    logger.warning(
                        f"Failed to generate GSPO sample {
                            i + 1}: {e}")
                    samples.append({
                        "prompt": prompt,
                        "response": f"Generation failed: {e}",
                        "reward": 0.0,
                        "sample_id": i + 1
                    })

            logger.info(f"Generated {len(samples)} GSPO samples")
            return samples

        except Exception as e:
            logger.error(f"GSPO sample generation failed: {e}")
            return []

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained GSPO model."""
        try:
            save_path = path or self._get_config_value(
                self.config.logging, 'output_dir', default='./output/unsloth_gspo')

            logger.info(f"Saving Unsloth GSPO model to: {save_path}")

            # Save using Unsloth's optimized saving
            self.unsloth_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save training configuration
            config_path = Path(save_path) / "gspo_training_config.yaml"
            config_dict = {
                'model_name': self.config.model.name_or_path,
                'task_type': 'generalized_scoring_proximal_objective',
                'max_seq_length': self.config.model.max_seq_length,
                'num_reward_functions': len(self.reward_functions),
            }
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GSPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GSPO model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load a trained GSPO model."""
        try:
            logger.info(f"Loading Unsloth GSPO model from: {path}")

            import unsloth
            from unsloth import FastLanguageModel

            # Load model and tokenizer
            self.unsloth_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=self.config.model.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            self.model = self.unsloth_model

            logger.info("GSPO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load GSPO model: {e}")
            raise

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
                text = str(sample.get("prompt", sample.get("query", "")))

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
                reward_value = reward_func["function"](response)
                rewards[f"reward_{j}"] = reward_value

            sample_outputs.append({
                "input": text[:100] + "..." if len(text) > 100 else text,
                "output": response,
                "rewards": rewards,
                "model": "unsloth_gspo"
            })

        return sample_outputs

    def cleanup(self) -> None:
        """Cleanup Unsloth resources."""
        super().cleanup()

        if self.model is not None:
            # Clear model from memory
            del self.model
            del self.unsloth_model
            torch.cuda.empty_cache()

        logger.info("Unsloth GSPO trainer cleanup completed")

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
                'use_peft': True,  # Unsloth always uses PEFT
                'precision': self.config.model.precision,
                'num_reward_functions': len(self.reward_functions),
            },
            'dataset_info': {
                'train_size': len(self.train_dataset) if hasattr(self, 'dataset') and self.train_dataset else 0,
                'val_size': len(self.eval_dataset) if hasattr(self, 'eval_dataset') and self.eval_dataset else 0,
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(next(self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
                'has_peft': True,  # Unsloth always uses PEFT
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
            'use_peft': True,  # Unsloth always uses PEFT
            'precision': self.config.model.precision,
            'num_reward_functions': len(self.reward_functions),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Unsloth GSPO configuration saved to {path}")

    def run_experiment(self):
        """Run a complete experiment with enhanced features."""
        logger.info(
            f"Starting Unsloth GSPO experiment: {
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
            train_results = self.train()
            end_time = time.time()
            training_duration = end_time - start_time

            # Evaluate after training
            logger.info("Running post-training evaluation...")
            eval_results = self.evaluate()

            # Combine results
            results = {
                'mode': 'train_and_eval',
                'task_type': 'generalized_scoring_proximal_objective',
                'train_results': train_results,
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
                f"Unsloth GSPO experiment completed. Results saved to {results_path}")
            return results

        except Exception as e:
            logger.error(f"Unsloth GSPO experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up logging resources
            if self.logging_manager:
                self.logging_manager.finish()
