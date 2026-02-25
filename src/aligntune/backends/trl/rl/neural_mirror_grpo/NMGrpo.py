"""
TRL Handler for Neural Mirror GRPO Trainer

This module provides integration between the NeuralMirrorGRPOTrainer and the
aligntune training infrastructure, supporting all Neural Mirror-specific parameters.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.precision_handler import PrecisionHandler
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.utils.config_extractor import  extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class TRLNeuralMirrorGRPOTrainer(TrainerBase):
    """Handler for Neural Mirror GRPO trainer with full parameter support."""

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.eval_dataset = None
        self.reward_functions = []
        self.training_history = []
        self.dataset_dict = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if required dependencies are available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Check for the custom Neural Mirror trainer
            from .neural_mirror_grpo import NeuralMirrorGRPOTrainer, NeuralMirrorGRPOConfig
            return True
        except ImportError:
            return False

    def setup_trainer(self) -> None:
        """Setup the Neural Mirror GRPO trainer with configuration."""
        logger.info("=" * 80)
        logger.info("Setting up Neural Mirror GRPO Trainer")
        logger.info("=" * 80)

        # Precision handling
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "Neural Mirror GRPO")
        precision_args = PrecisionHandler.get_training_args_precision(
            precision)

        # Get all training parameters
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/neural_mirror_grpo')
        max_steps = self._get_config_value(
            self.config.train, 'max_steps', default=1600)
        if max_steps is None:
            max_steps = -1
        per_device_batch_size = self._get_config_value(
            self.config.train, 'per_device_train_batch_size', 'batch_size', default=16)
        num_generations = self._get_config_value(
            self.config.train, 'num_generations', default=32)
        learning_rate = self._get_config_value(
            self.config.train, 'learning_rate', 'lr', default=1e-5)
        gradient_accumulation_steps = self._get_config_value(
            self.config.train, 'gradient_accumulation_steps', default=8)
        max_completion_length = self._get_config_value(
            self.config.train, 'max_completion_length', default=256)
        max_prompt_length = self._get_config_value(
            self.config.train, 'max_prompt_length', default=768)
        temperature = self._get_config_value(
            self.config.train, 'temperature', default=0.8)
        top_p = self._get_config_value(self.config.train, 'top_p', default=0.9)

        # Neural Mirror GRPO specific parameters
        mirror_coefficient = self._get_config_value(
            self.config.train, 'mirror_coefficient', default=0.0001)
        mirror_init_scale = self._get_config_value(
            self.config.train, 'mirror_init_scale', default=0.01)
        mirror_seed = self._get_config_value(
            self.config.train, 'mirror_seed', default=42)
        divergence_type = self._get_config_value(
            self.config.train, 'divergence_type', default='neural_mirror')

        # Additional training parameters
        weight_decay = self._get_config_value(
            self.config.train, 'weight_decay', default=0.0)
        max_grad_norm = self._get_config_value(
            self.config.train, 'max_grad_norm', default=1.0)
        warmup_steps = self._get_config_value(
            self.config.train, 'warmup_steps', default=10)
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

        # Seed
        seed = self._get_config_value(self.config.train, 'seed', default=42)
        loss_type = self._get_config_value(
            self.config.train, 'loss_type', default='dapo')
        if loss_type == 'sigmoid':
            print('Sigmoid not supported')
            loss_type = 'dapo'

        # Use eval_dataset-aware defaults
        if self.eval_dataset:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
            eval_strategy = 'no'
            eval_steps = None

        logger.info("Neural Mirror GRPO Training Configuration")
        logger.info(f"Model: {self.config.model.name_or_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Batch size: {per_device_batch_size}")
        logger.info(f"Num generations: {num_generations}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"Max completion length: {max_completion_length}")
        logger.info(f"Max prompt length: {max_prompt_length}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Top-p: {top_p}")
        logger.info(f"Mirror coefficient: {mirror_coefficient}")
        logger.info(f"Mirror init scale: {mirror_init_scale}")
        logger.info(f"Mirror seed: {mirror_seed}")
        logger.info(f"Divergence type: {divergence_type}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Import Neural Mirror GRPO trainer
        from .neural_mirror_grpo import NeuralMirrorGRPOTrainer, NeuralMirrorGRPOConfig

        # Create Neural Mirror GRPO config with all parameters
        grpo_config = NeuralMirrorGRPOConfig(
            # Output and logging
            output_dir=output_dir,

            # Evaluation parameters
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,

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
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            seed=seed,

            # Generation parameters
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,

            # Neural Mirror GRPO specific
            divergence_type=divergence_type,
            mirror_coefficient=mirror_coefficient,
            mirror_init_scale=mirror_init_scale,
            mirror_seed=mirror_seed,
            loss_type=loss_type,

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

        # Create Neural Mirror GRPO trainer
        logger.info("Creating Neural Mirror GRPO trainer instance...")
        self.trainer = NeuralMirrorGRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self._combined_reward_function,
        )

        logger.info("Neural Mirror GRPO Trainer setup completed successfully")

    def train(self) -> Dict[str, Any]:
        """Execute Neural Mirror GRPO training with all setup steps."""
        # Setup all components in order
        logger.info("=" * 80)
        logger.info("Starting Neural Mirror GRPO Training Pipeline")
        logger.info("=" * 80)

        # Step 1: Setup model
        self.setup_model()

        # Step 2: Setup rewards
        self.setup_rewards()

        # Step 3: Setup data
        self.setup_data()

        # Step 4: Setup trainer
        self.setup_trainer()

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Neural Mirror GRPO Training")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info("=" * 80)

        # Start training
        train_result = self.trainer.train()

        # Log samples
        try:
            reward_callables = [rf["function"] for rf in self.reward_functions]
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
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/neural_mirror_grpo')
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Get Neural Mirror specific parameters for results
        mirror_coefficient = self._get_config_value(
            self.config.train, 'mirror_coefficient', default=0.0001)
        mirror_init_scale = self._get_config_value(
            self.config.train, 'mirror_init_scale', default=0.01)
        divergence_type = self._get_config_value(
            self.config.train, 'divergence_type', default='neural_mirror')

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
            "mirror_coefficient": mirror_coefficient,
            "mirror_init_scale": mirror_init_scale,
            "divergence_type": divergence_type,
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("Neural Mirror GRPO Training Completed!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)

        return results

    def setup_model(self) -> None:
        """Setup model with LoRA support if specified."""
        model_name = self._get_config_value(
            self.config.model,
            'name_or_path',
            'model_name',
            default='gpt2')
        trust_remote_code = self._get_config_value(
            self.config.model, 'trust_remote_code', default=False)
        device_map = self._get_config_value(
            self.config.model, 'device_map', default='auto')
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
        use_lora = self._get_config_value(
            self.config.model, 'use_lora', 'use_peft', default=False)

        # Precision handling
        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "Neural Mirror GRPO")
        dtype = PrecisionHandler.get_torch_dtype(precision)

        logger.info("=" * 80)
        logger.info(f"Setting up Neural Mirror GRPO model: {model_name}")
        logger.info(f"Precision: {precision} (dtype: {dtype})")
        logger.info(f"LoRA enabled: {use_lora}")
        logger.info("=" * 80)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad token to eos token")

        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Apply LoRA if specified
        if use_lora:
            logger.info("Applying LoRA configuration...")
            from peft import LoraConfig, get_peft_model

            lora_r = self._get_config_value(
                self.config.model, 'lora_r', 'r', default=64)
            lora_alpha = self._get_config_value(
                self.config.model, 'lora_alpha', 'alpha', default=64)
            lora_dropout = self._get_config_value(
                self.config.model, 'lora_dropout', 'dropout', default=0.05)
            lora_target_modules = self._get_config_value(
                self.config.model,
                'lora_target_modules',
                'target_modules',
                default=["q_proj", "k_proj", "v_proj", "o_proj"]
            )

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

        logger.info("Model setup completed successfully")

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
        """Setup reward functions using the registry system."""
        logger.info("Setting up reward functions for Neural Mirror GRPO...")

        # Get reward configurations
        rewards_config = []
        if hasattr(self.config, 'rewards'):
            rewards_config = self.config.rewards if isinstance(
                self.config.rewards, list) else []

        if not rewards_config:
            logger.warning(
                "No reward configurations found, using default code rewards")
            rewards_config = [
                {"type": "code_quality", "weight": 0.3, "params": {}},
                {"type": "code_correctness", "weight": 0.4, "params": {}},
                {"type": "length", "weight": 0.15, "params": {"min_length": 20, "max_length": 256}},
                {"type": "diversity", "weight": 0.15, "params": {}},
            ]

        # Load reward functions
        for reward_config in rewards_config:
            reward_type = reward_config.get('type', 'length')
            weight = reward_config.get('weight', 1.0)
            params = reward_config.get('params', {})

            try:
                # Handle custom reward functions
                if reward_type == 'custom' and 'reward_function' in params:
                    reward_func = params['reward_function']
                    logger.info(
                        f"Loaded custom reward function (weight: {weight})")
                else:
                    # Use rewards registry
                    from aligntune.rewards.registry import RewardRegistry as RewardsRegistry
                    from aligntune.rewards.core import RewardConfig, RewardType

                    # Map common variations
                    reward_type_mapping = {
                        'math': 'math_reasoning',
                        'code': 'code_quality',
                    }
                    reward_type = reward_type_mapping.get(
                        reward_type, reward_type)

                    try:
                        reward_type_enum = RewardType[reward_type.upper()]
                        reward_cfg = RewardConfig(
                            reward_type=reward_type_enum,
                            weight=1.0,
                            params=params
                        )
                        reward_func_obj = RewardsRegistry.get_reward_function(
                            reward_type, reward_cfg)

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
                        logger.warning(
                            f"Reward type '{reward_type}' not found, trying by name")
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

                # Store reward function
                self.reward_functions.append({
                    "function": reward_func,
                    "weight": weight,
                    "name": reward_type
                })

            except Exception as e:
                logger.warning(
                    f"Failed to load reward function '{reward_type}': {e}")
                continue

        if not self.reward_functions:
            logger.error(
                "No reward functions loaded! Adding default fallback.")

            def default_reward(text, reference=None, **kwargs):
                return 1.0 if len(text.split()) > 10 else 0.5

            self.reward_functions.append({
                "function": default_reward,
                "weight": 1.0,
                "name": "default"
            })

        logger.info(
            f"Configured {len(self.reward_functions)} reward functions")
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

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step."""
        # Neural Mirror GRPO uses internal training loop
        return {"loss": 0.0}

    def create_data_loader(self):
        """Create data loader."""
        # Neural Mirror GRPO uses internal data loading
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

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': 'neural_mirror_grpo',
                'dataset_name': self.config.dataset.name,
                'learning_rate': self._get_config_value(
                    self.config.train,
                    'learning_rate',
                    default=1e-5),
                'batch_size': self._get_config_value(
                    self.config.train,
                    'per_device_train_batch_size',
                    default=16),
                'mirror_coefficient': self._get_config_value(
                    self.config.train,
                    'mirror_coefficient',
                    default=0.0001),
                'num_reward_functions': len(
                    self.reward_functions),
            },
            'dataset_info': {
                'train_size': len(
                    self.train_dataset) if hasattr(
                    self,
                    'dataset') and self.train_dataset else 0,
                'val_size': len(
                    self.eval_dataset) if hasattr(
                    self,
                    'eval_dataset') and self.eval_dataset else 0,
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(
                    next(
                        self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(
                    self.tokenizer) if self.tokenizer else 0,
                'has_lora': hasattr(
                    self.model,
                    'peft_config') if self.model else False,
            },
            'training_history': self.training_history,
        }

        return stats

    def save_config(self, path: str):
        """Save configuration to file."""
        import yaml

        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': 'neural_mirror_grpo',
            'learning_rate': self._get_config_value(
                self.config.train,
                'learning_rate',
                default=1e-5),
            'batch_size': self._get_config_value(
                self.config.train,
                'per_device_train_batch_size',
                default=16),
            'mirror_coefficient': self._get_config_value(
                self.config.train,
                'mirror_coefficient',
                default=0.0001),
            'mirror_init_scale': self._get_config_value(
                self.config.train,
                'mirror_init_scale',
                default=0.01),
            'mirror_seed': self._get_config_value(
                self.config.train,
                'mirror_seed',
                default=42),
            'divergence_type': self._get_config_value(
                self.config.train,
                'divergence_type',
                default='neural_mirror'),
            'num_reward_functions': len(
                self.reward_functions),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Neural Mirror GRPO configuration saved to {path}")
    
    
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
            config_path = Path(save_path) / "nmgrpo_training_config"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(
                    self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"GRPO model saved successfully to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save GRPO model: {e}")
            raise

    