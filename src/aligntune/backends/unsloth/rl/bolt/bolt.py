"""
BOLT: Baseline-Optimized Learning Technique (Unsloth variant)

Unsloth-optimized GRPO with curriculum sampling and persistent baselines.
Provides 2-5x speedup over TRL variant using Unsloth's FastLanguageModel.

CRITICAL: This implementation overrides _generate_and_score_completions to
properly replace TRL's group-mean advantages with baseline advantages.
TRL always computes: A = r - mean(r_group)
We replace with: A = r - v̂(x)

Reuses baseline and curriculum modules from TRL BOLT backend.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np

from aligntune.backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
from aligntune.core.rl.config import UnifiedConfig

# Reuse BOLT components from TRL backend
from aligntune.backends.trl.rl.pace.baseline import UnifiedBaseline, make_prompt_key
from aligntune.backends.trl.rl.pace.curriculum import (
    DynamicWeightedSampler,
    DynamicallySampledDataset,
    CurriculumCallback,
)
from aligntune.backends.trl.rl.pace.pace import PaceConfig, PaceGRPOTrainer
from aligntune.core.precision_handler import PrecisionHandler

logger = logging.getLogger(__name__)


class UnslothPaceTrainer(UnslothGRPOTrainer):
    """
    BOLT with Unsloth acceleration (2-5x speedup).

    Extends UnslothGRPOTrainer with:
    - Uncertainty-based curriculum sampling
    - Persistent per-prompt baselines with KL-adaptive forgetting
    - SPO-style advantage computation: A = r - v̂(x)

    Uses Unsloth's FastLanguageModel for optimized model loading.

    IMPORTANT: Uses PaceGRPOTrainer.create_trainer_class() to properly override
    TRL's advantage computation. This works with Unsloth because it patches
    the GRPOTrainer class that we subclass.
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)

        # BOLT components
        self.baseline: Optional[UnifiedBaseline] = None
        self.curriculum_sampler: Optional[DynamicWeightedSampler] = None
        self.pace_config: Optional[PaceConfig] = None

        # Dataset storage for curriculum
        self._base_dataset_list: Optional[List[Dict[str, Any]]] = None
        self._prompt_keys: Optional[List[str]] = None
        self.model = None
        self.custom_evaluator = None
        self.eval_dataset = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth, TRL and BOLT dependencies are available."""
        try:
            import unsloth
            from unsloth import FastLanguageModel
            from trl import GRPOTrainer, GRPOConfig
            import numpy as np
            return True
        except ImportError:
            return False

    def _get_pace_config(self) -> PaceConfig:
        """Extract BOLT configuration from TrainingConfig."""
        if self.pace_config is not None:
            return self.pace_config

        train_cfg = self.config.train

        self.pace_config = PaceConfig(
            # Curriculum
            curriculum_enabled=self._get_config_value(
                train_cfg, "curriculum_enabled", default=False
            ),
            curriculum_epsilon=self._get_config_value(
                train_cfg, "curriculum_epsilon", default=0.05
            ),
            curriculum_update_freq=self._get_config_value(
                train_cfg, "curriculum_update_freq", default=10
            ),
            # Baseline
            baseline_enabled=self._get_config_value(
                train_cfg, "baseline_enabled", default=False
            ),
            baseline_rho_min=self._get_config_value(
                train_cfg, "baseline_rho_min", default=0.875
            ),
            baseline_rho_max=self._get_config_value(
                train_cfg, "baseline_rho_max", default=0.96
            ),
            baseline_D_half=self._get_config_value(
                train_cfg, "baseline_D_half", default=0.5
            ),
            baseline_warm_start=self._get_config_value(
                train_cfg, "baseline_warm_start", default=None
            ),
            # Advantage
            use_baseline_advantages=self._get_config_value(
                train_cfg, "use_baseline_advantages", default=False
            ),
        )

        logger.info("=" * 60)
        logger.info("BOLT Configuration (Unsloth):")
        logger.info(
            f"  Curriculum enabled: {
                self.pace_config.curriculum_enabled}")
        logger.info(f"  Baseline enabled: {self.pace_config.baseline_enabled}")
        logger.info(
            f"  Use baseline advantages: {
                self.pace_config.use_baseline_advantages}")
        if self.pace_config.curriculum_enabled:
            logger.info(
                f"  Curriculum epsilon: {
                    self.pace_config.curriculum_epsilon}")
            logger.info(
                f"  Curriculum update freq: {
                    self.pace_config.curriculum_update_freq}")
        if self.pace_config.baseline_enabled:
            logger.info(
                f"  Baseline rho: [{
                    self.pace_config.baseline_rho_min}, {
                    self.pace_config.baseline_rho_max}]")
            logger.info(
                f"  Baseline D_half: {
                    self.pace_config.baseline_D_half}")
            if self.pace_config.baseline_warm_start:
                logger.info(
                    f"  Baseline warm-start: {self.pace_config.baseline_warm_start}")
        logger.info("=" * 60)

        return self.pace_config

    def setup_model(self) -> None:
        """Setup Unsloth-optimized model and tokenizer."""
        try:
            import unsloth
            from unsloth import FastLanguageModel

            # Handle both dict and object config
            if isinstance(self.config.model, dict):
                model_name = self.config.model.get('name_or_path')
                max_seq_length = self.config.model.get('max_seq_length', 2048)
                quantization = self.config.model.get('quantization', {})
                lora_r = self.config.model.get('lora_r', 32)
                lora_alpha = self.config.model.get('lora_alpha', 32)
                lora_dropout = self.config.model.get('lora_dropout', 0.1)
                lora_target_modules = self.config.model.get(
                    'lora_target_modules', [
                        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            else:
                model_name = self.config.model.name_or_path
                max_seq_length = self.config.model.max_seq_length
                quantization = getattr(self.config.model, 'quantization', {})
                lora_r = getattr(self.config.model, 'lora_r', 32)
                lora_alpha = getattr(self.config.model, 'lora_alpha', 32)
                lora_dropout = getattr(self.config.model, 'lora_dropout', 0.1)
                lora_target_modules = getattr(
                    self.config.model, 'lora_target_modules', [
                        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

            logger.info("=" * 80)
            logger.info(
                f"Setting up Unsloth Counterfactual GRPO model: {model_name}")
            logger.info("=" * 80)

            # Check if fast_inference (vLLM) is enabled
            fast_inference = self._get_config_value(
                self.config.train, 'fast_inference', default=False)
            vllm_gpu_memory_utilization = self._get_config_value(
                self.config.train, 'vllm_gpu_memory_utilization', default=0.7)

            device_map = self.config.model.device_map or 'auto'

            # Configure Unsloth model parameters
            model_kwargs = {
                "max_seq_length": max_seq_length,
                "dtype": None,  # Auto-detect
                "load_in_4bit": quantization.get("load_in_4bit", True) if isinstance(quantization, dict) else True,
                "fast_inference": fast_inference,  # Unsloth's native vLLM integration
                # vLLM memory (0.95 for max speed)
                "gpu_memory_utilization": vllm_gpu_memory_utilization,
                "device_map": device_map,
            }

            logger.info(f"Loading model with kwargs: {model_kwargs}")

            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                **model_kwargs
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Configure model for training with LoRA
            logger.info(
                f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            logger.info(f"Target modules: {lora_target_modules}")

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_r,
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            logger.info(
                "Unsloth Counterfactual GRPO model setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Unsloth model: {e}")
            raise

    def _combined_reward_function(
            self,
            completions: List[str],
            **kwargs) -> List[float]:
        """Combined reward function with debug output for BOLT."""
        if not completions:
            return []

        batch_rewards = []

        # Extract batch data from kwargs
        test_lists = kwargs.get('test_list', [None] * len(completions))
        prompts = kwargs.get(
            'prompt',
            kwargs.get(
                'query',
                [None] *
                len(completions)))

        # Ensure lists
        if not isinstance(test_lists, list):
            test_lists = [test_lists] * len(completions)
        if not isinstance(prompts, list):
            prompts = [prompts] * len(completions)

        for idx, completion in enumerate(completions):
            total_reward = 0.0
            test_cases = test_lists[idx] if idx < len(test_lists) else None

            for rf in self.reward_functions:
                try:
                    reward_func = rf["function"]
                    weight = rf["weight"]

                    # Call reward function with test_cases
                    try:
                        reward = reward_func(completion, test_cases=test_cases)
                    except TypeError:
                        try:
                            reward = reward_func(completion)
                        except BaseException:
                            reward = 0.0

                    weighted_reward = weight * reward
                    total_reward += weighted_reward

                except Exception as e:
                    logger.warning(f"Error computing reward {rf['name']}: {e}")

            batch_rewards.append(total_reward)

        return batch_rewards

    def setup_data(self) -> None:
        """
        Setup datasets with BOLT curriculum sampling.

        Extends parent to:
        1. Initialize baseline (cold or warm-start)
        2. Wrap dataset with curriculum sampler (if enabled)
        """
        # Call parent to load dataset
        super().setup_data()

        pace_cfg = self._get_pace_config()

        # Skip if neither curriculum nor baseline enabled
        if not pace_cfg.curriculum_enabled and not pace_cfg.baseline_enabled and not pace_cfg.use_baseline_advantages:
            logger.info("BOLT features disabled, using standard Unsloth GRPO")
            return

        # Convert dataset to list and extract prompt keys
        logger.info("Preparing dataset for BOLT (Unsloth)...")

        # Handle different dataset sources
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            self._base_dataset_list = list(self.train_dataset)
        elif hasattr(self, 'dataset') and self.train_dataset is not None:
            self._base_dataset_list = list(self.train_dataset)
        else:
            logger.warning("No dataset found for BOLT curriculum")
            return

        self._prompt_keys = []
        for item in self._base_dataset_list:
            # Try different column names for prompt
            prompt = item.get("prompt") or item.get(
                "query") or item.get("question") or ""
            key = make_prompt_key(prompt)
            self._prompt_keys.append(key)

        logger.info(f"Extracted {len(self._prompt_keys)} prompt keys")

        # Initialize baseline
        if pace_cfg.baseline_enabled or pace_cfg.curriculum_enabled or pace_cfg.use_baseline_advantages:
            logger.info("Initializing BOLT baseline...")
            self.baseline = UnifiedBaseline(
                rho_min=pace_cfg.baseline_rho_min,
                rho_max=pace_cfg.baseline_rho_max,
                D_half=pace_cfg.baseline_D_half,
                epsilon=pace_cfg.curriculum_epsilon,
                track_timeline=False,
            )

            # Load warm-start if provided
            if pace_cfg.baseline_warm_start:
                logger.info(
                    f"Loading baseline warm-start from {pace_cfg.baseline_warm_start}")
                self.baseline.load(pace_cfg.baseline_warm_start)
                logger.info(
                    f"Loaded {len(self.baseline.tab)} baseline entries")

        # Setup curriculum sampling
        if pace_cfg.curriculum_enabled and self.baseline is not None:
            logger.info("Setting up curriculum sampling...")
            self.curriculum_sampler = DynamicWeightedSampler(
                baseline=self.baseline,
                dataset_size=len(self._base_dataset_list),
                prompt_keys=self._prompt_keys,
                epsilon=pace_cfg.curriculum_epsilon,
                oversample=2.0,
            )

            # Wrap dataset
            wrapped_dataset = DynamicallySampledDataset(
                base_dataset=self._base_dataset_list,
                sampler=self.curriculum_sampler,
            )

            # Update appropriate attribute
            if hasattr(self, 'train_dataset'):
                self.train_dataset = wrapped_dataset
            else:
                self.train_dataset = wrapped_dataset

            logger.info(
                f"Curriculum dataset size: {
                    len(wrapped_dataset)} (2x oversampled)")

    def setup_trainer(self) -> None:
        """
        Setup the BOLT trainer with all necessary configurations and callbacks.

        This configures the trainer but does not start training.
        Call train() to begin the training process.
        """
        pace_cfg = self._get_pace_config()

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
        save_steps = self._get_config_value(
            self.config.train, 'save_steps', default=100)
        logging_steps = self._get_config_value(
            self.config.train, 'logging_steps', default=10)
        eval_steps = self._get_config_value(
            self.config.train, 'eval_steps', default=None)

        # === UNIFIED PRECISION HANDLING ===
        from aligntune.core.precision_handler import PrecisionHandler

        precision = PrecisionHandler.get_precision_from_config(
            self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "BOLT (Unsloth)")
        precision_args = PrecisionHandler.get_training_args_precision(
            precision)

        gradient_checkpointing = self._get_config_value(
            self.config.model, 'gradient_checkpointing', default=False)
        output_dir = self._get_config_value(
            self.config.logging,
            'output_dir',
            default='./output/pace_unsloth')
        num_generations = self._get_config_value(
            self.config.train, 'num_generations', default=8)
        seed = self._get_config_value(self.config.train, 'seed', default=42)

        # GRPO-specific parameters
        beta = self._get_config_value(
            self.config.train, 'beta', 'kl_coef', default=0.1)
        epsilon = self._get_config_value(
            self.config.train, 'epsilon', 'cliprange', default=0.2)
        max_steps = self._get_config_value(
            self.config.train, 'max_steps', default=1)
        # Generation parameters
        max_completion_length = self._get_config_value(
            self.config.train, 'max_completion_length', 'max_new_tokens', default=512)
        max_prompt_length = self._get_config_value(
            self.config.train, 'max_prompt_length', default=512)
        temperature = self._get_config_value(
            self.config.train, 'temperature', default=0.7)
        top_p = self._get_config_value(
            self.config.train, 'top_p', default=0.95)
        top_k = self._get_config_value(self.config.train, 'top_k', default=50)
        eval_strategy = self._get_config_value(
            self.config.train, 'eval_strategy', default='epoch')
        if self.eval_dataset:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
            eval_strategy = 'no'
            eval_steps = None
            print(eval_strategy)

        logger.info("=" * 80)
        logger.info("BOLT Training Configuration (Unsloth)")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Batch size: {per_device_batch_size}")
        logger.info(f"Num generations (K): {num_generations}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"Max prompt length: {max_prompt_length}")
        logger.info(f"Max completion length: {max_completion_length}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup GRPO config
        from trl import GRPOConfig

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_steps is not None else save_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            remove_unused_columns=False,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            max_steps=max_steps,
            # Generation parameters
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # GRPO-specific
            beta=beta,
            eval_strategy=eval_strategy,
            epsilon=epsilon,
            # Precision (unified handling)
            **precision_args,
        )

        # ============ FIX: Better dataset retrieval ============
        # Get dataset with better error handling
        dataset = None
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            dataset = self.train_dataset
        elif hasattr(self, 'dataset') and self.train_dataset is not None:
            dataset = self.train_dataset

        # If still None, raise clear error
        if dataset is None:
            raise ValueError(
                "No dataset found! Make sure setup_data() was called before setup_trainer(). "
                "Expected either self.train_dataset or self.train_dataset to be set.")

        logger.info(f"Using dataset with {len(dataset)} examples")
        # ======================================================
        # ============ FIX: Better model retrieval ============
        # Ensure model exists
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError(
                "Model not found! Make sure setup_model() was called before setup_trainer(). "
                "Expected self.unsloth_model to be set.")

        logger.info(f"Using model: {type(self.model).__name__}")
        # ====================================================

        # Create trainer - use custom PaceGRPOTrainer if baseline advantages
        # enabled
        if pace_cfg.use_baseline_advantages and self.baseline is not None:
            logger.info(
                "Using PaceGRPOTrainer with baseline advantage computation (Unsloth)")
            TrainerClass = PaceGRPOTrainer.create_trainer_class(
                baseline=self.baseline,
                num_generations=num_generations,
            )
        else:
            logger.info("Using standard GRPOTrainer (no baseline advantages)")
            from trl import GRPOTrainer
            TrainerClass = GRPOTrainer

        # Create trainer instance with proper model reference
        self.trainer = TrainerClass(
            model=self.model,  # ← CRITICAL: Use unsloth_model, not model
            args=grpo_config,
            train_dataset=dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self._combined_reward_function,
        )

        # Add curriculum callback if enabled
        if pace_cfg.curriculum_enabled and self.curriculum_sampler is not None and self.baseline is not None:
            self.trainer.add_callback(CurriculumCallback(
                baseline=self.baseline,
                sampler=self.curriculum_sampler,
                dataset=dataset,
                update_freq=pace_cfg.curriculum_update_freq,
            ))
            logger.info(
                f"Added CurriculumCallback (update every {
                    pace_cfg.curriculum_update_freq} steps)")

        # Store output_dir for use in train()
        self.output_dir = output_dir

        logger.info("=" * 80)
        logger.info("BOLT Trainer Setup Complete (Unsloth)")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info(f"Baseline advantages: {pace_cfg.use_baseline_advantages}")
        logger.info(f"Curriculum sampling: {pace_cfg.curriculum_enabled}")
        logger.info("=" * 80)

    def train(self) -> Dict[str, Any]:
        """
        Execute BOLT training with Unsloth acceleration.

        This method performs all setup steps and then runs the training loop.
        """
        # Setup all components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()
        print(self.eval_dataset)
        self.setup_trainer()

        pace_cfg = self._get_pace_config()

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting BOLT Training (Unsloth)")
        logger.info("=" * 80)

        # Start training
        train_result = self.trainer.train()

        # Record training end
        end_time = time.time()
        training_duration = end_time - start_time

        logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Save baseline
        baseline_path = None
        if self.baseline is not None:
            baseline_path = Path(self.output_dir) / "pace_baseline.pkl"
            self.baseline.save(str(baseline_path))
            logger.info(f"Saved BOLT baseline to {baseline_path}")

        # Extract metrics
        metrics = {}
        if hasattr(train_result, 'metrics'):
            metrics = train_result.metrics

        # Add BOLT stats
        if self.baseline is not None:
            metrics.update(self.baseline.get_baseline_stats())
            metrics.update(self.baseline.get_curriculum_stats())

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

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
            "model_path": self.output_dir,
            "baseline_path": str(baseline_path) if baseline_path else None,
            "num_reward_functions": len(
                    self.reward_functions),
            "num_datasets": 1,
            "pace_config": {
                "curriculum_enabled": pace_cfg.curriculum_enabled,
                "baseline_enabled": pace_cfg.baseline_enabled,
                "use_baseline_advantages": pace_cfg.use_baseline_advantages,
                "backend": "unsloth",
            },
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("BOLT Training (Unsloth) Completed Successfully!")
        logger.info(f"Training time: {training_duration:.2f}s")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        if self.baseline:
            logger.info(f"Baseline prompts tracked: {len(self.baseline.tab)}")
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
