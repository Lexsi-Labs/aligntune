"""
BOLT: Baseline-Optimized Learning Technique

GRPO trainer with curriculum sampling and persistent baselines.

Key features:
1. Curriculum sampling: weight ∝ sqrt(v̂(1-v̂)) + ε (uncertainty-based)
2. Persistent baseline: A = r - v̂(x) (SPO-style advantage)
3. KL-adaptive forgetting: faster adaptation when policy shifts

CRITICAL: This implementation overrides _generate_and_score_completions to
properly replace TRL's group-mean advantages with baseline advantages.
TRL always computes: A = r - mean(r_group)
We replace with: A = r - v̂(x)

Configuration (in TrainingConfig):
- curriculum_enabled: Enable uncertainty-based sampling
- curriculum_epsilon: Floor for sampling weights (default: 0.05)
- curriculum_update_freq: Steps between weight updates (default: 10)
- baseline_enabled: Enable persistent baseline
- baseline_rho_min: Min forgetting factor (default: 0.875)
- baseline_rho_max: Max forgetting factor (default: 0.96)
- baseline_D_half: KL half-life (default: 0.5)
- baseline_warm_start: Path to JSON/PKL for warm-start
- use_baseline_advantages: Use A = r - v̂(x) vs group mean
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import numpy as np

from aligntune.backends.trl.rl.grpo.grpo import TRLGRPOTrainer
from aligntune.core.rl.config import UnifiedConfig
from aligntune.core.precision_handler import PrecisionHandler

from .baseline import UnifiedBaseline, make_prompt_key
from .curriculum import (
    DynamicWeightedSampler,
    DynamicallySampledDataset,
    CurriculumCallback,
    BaselineUpdateCallback,
    BaselineCheckpointCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class BoltConfig:
    """BOLT-specific configuration extracted from TrainingConfig."""

    # Curriculum
    curriculum_enabled: bool = False
    curriculum_epsilon: float = 0.05
    curriculum_update_freq: int = 10

    # Baseline
    baseline_enabled: bool = False
    baseline_rho_min: float = 0.875
    baseline_rho_max: float = 0.96
    baseline_D_half: float = 0.5
    baseline_warm_start: Optional[str] = None

    # Advantage computation
    use_baseline_advantages: bool = False


class BoltGRPOTrainer:
    """
    Custom GRPO trainer that properly overrides advantage computation.

    This class wraps TRL's GRPOTrainer and overrides:
    - _calculate_rewards: Store rewards/prompts for later use
    - _generate_and_score_completions: Replace TRL's advantages with baseline advantages

    IMPORTANT: TRL always computes advantages as:
        A = r - mean(r_group)

    We intercept this and replace with:
        A = r - v̂(x)

    where v̂(x) is the persistent baseline for prompt x.
    """

    @staticmethod
    def create_trainer_class(baseline: UnifiedBaseline, num_generations: int):
        """
        Dynamically create a GRPOTrainer subclass with baseline advantage computation.

        We use a factory pattern because we need to inject the baseline into the
        trainer class, and TRL's GRPOTrainer doesn't support this directly.
        """
        from trl import GRPOTrainer

        class _BoltGRPOTrainer(GRPOTrainer):
            """GRPOTrainer with persistent baseline advantages."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._bolt_baseline = baseline
                self._bolt_num_gen = num_generations
                self._bolt_step_count = 0
                self._last_rewards = None
                self._last_prompts = None

            def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
                """
                Override to store rewards and prompts for baseline advantage computation.

                TRL calls this to compute rewards, then uses them in _generate_and_score_completions
                to compute advantages. We intercept to store the data.
                """
                # Call parent to compute rewards
                rewards_per_func = super()._calculate_rewards(
                    inputs, prompts, completions, completion_ids_list
                )

                # Apply weights and sum (same as parent does in _generate_and_score_completions)
                device = rewards_per_func.device
                rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

                # Store for advantage computation in _generate_and_score_completions
                self._last_rewards = rewards
                self._last_prompts = prompts

                return rewards_per_func

            def _generate_and_score_completions(self, inputs):
                """
                Override to replace TRL's group-mean advantages with baseline advantages.

                TRL computes: A = r - mean(r_group)
                We replace with: A = r - v̂(x)
                """
                self._bolt_step_count += 1

                # Call parent to get everything computed (including TRL's wrong advantages)
                output = super()._generate_and_score_completions(inputs)

                # Skip if baseline advantages disabled or no stored data
                if self._bolt_baseline is None:
                    return output

                if self._last_rewards is None or self._last_prompts is None:
                    logger.warning(f"Step {self._bolt_step_count}: No stored rewards, using TRL advantages")
                    return output

                # Extract stored data
                rewards = self._last_rewards
                prompts_text = self._last_prompts
                num_gen = self._bolt_num_gen

                # Reshape rewards: (batch_size * num_gen,) -> (batch_size, num_gen)
                batch_size = len(rewards) // num_gen
                if batch_size == 0:
                    return output

                rewards_reshaped = rewards.view(batch_size, num_gen)

                # Get prompt keys (unique prompts - take every num_gen'th element)
                prompt_keys = []
                for i in range(0, len(prompts_text), num_gen):
                    if i < len(prompts_text):
                        key = make_prompt_key(prompts_text[i])
                        prompt_keys.append(key)
                prompt_keys = prompt_keys[:batch_size]

                # 1. PRE-UPDATE: Get baselines v̂(x) for each prompt (STORE OLD VALUES)
                old_v_hats = {k: self._bolt_baseline.get_v_hat(k) for k in prompt_keys}
                baselines = torch.tensor(
                    [old_v_hats[k] for k in prompt_keys],
                    device=rewards.device,
                    dtype=rewards.dtype
                ).unsqueeze(1).expand(-1, num_gen)

                # 2. Compute baseline advantages: A = r - v̂(x)
                advantages = rewards_reshaped - baselines
                advantages_flat = advantages.flatten()

                # 3. Global Z-normalization (same as TRL does)
                adv_mean = advantages_flat.mean()
                adv_std = advantages_flat.std().clamp(min=1e-6)
                advantages_normalized = (advantages_flat - adv_mean) / adv_std
                advantages_normalized = advantages_normalized.clamp(-5.0, 5.0)

                # 4. POST-UPDATE: Update baselines with new rewards
                mean_rewards_per_prompt = []
                with torch.no_grad():
                    for i, key in enumerate(prompt_keys):
                        group_rewards = rewards_reshaped[i].tolist()
                        mean_rewards_per_prompt.append(np.mean(group_rewards))
                        for reward in group_rewards:
                            # Estimate KL from performance shift
                            kl = self._bolt_baseline.estimate_kl_from_performance_shift(
                                key, group_rewards
                            )
                            self._bolt_baseline.update(key, reward, kl=kl)

                # DETAILED STEP LOGGING (every step for first 10 steps, then every 10 steps)
                verbose_logging = (self._bolt_step_count <= 10) or (self._bolt_step_count % 10 == 0)
                if verbose_logging:
                    self._bolt_baseline.print_step_update(
                        step=self._bolt_step_count,
                        prompt_keys=prompt_keys,
                        old_v_hats=old_v_hats,
                        rewards=mean_rewards_per_prompt
                    )

                # Summary logging every 50 steps
                if self._bolt_step_count % 50 == 0:
                    mean_baseline = np.mean([self._bolt_baseline.get_v_hat(k) for k in prompt_keys])
                    zero_adv = (advantages_normalized.abs() < 0.01).sum().item()
                    total = len(advantages_normalized)

                    logger.info(f"BOLT Step {self._bolt_step_count}:")
                    logger.info(f"  Baseline mean v̂: {mean_baseline:.3f}")
                    logger.info(f"  Advantage range: [{advantages_normalized.min().item():.3f}, {advantages_normalized.max().item():.3f}]")
                    logger.info(f"  Near-zero advantages: {zero_adv}/{total} ({100*zero_adv/total:.1f}%)")
                    logger.info(f"  Tracked prompts: {len(self._bolt_baseline.tab)}")

                # 5. REPLACE TRL's advantages with ours
                output["advantages"] = advantages_normalized

                # Clear stored data
                self._last_rewards = None
                self._last_prompts = None

                return output

        return _BoltGRPOTrainer


class TRLBoltTrainer(TRLGRPOTrainer):
    """
    BOLT: GRPO with curriculum sampling and persistent baselines.

    Extends TRLGRPOTrainer with:
    - Uncertainty-based curriculum sampling
    - Persistent per-prompt baselines with KL-adaptive forgetting
    - SPO-style advantage computation: A = r - v̂(x)

    Both features can be enabled independently:
    - Curriculum only: curriculum_enabled=True, use_baseline_advantages=False
    - Baseline only: curriculum_enabled=False, use_baseline_advantages=True
    - Both: curriculum_enabled=True, use_baseline_advantages=True
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__(config)

        # BOLT components
        self.baseline: Optional[UnifiedBaseline] = None
        self.curriculum_sampler: Optional[DynamicWeightedSampler] = None
        self.bolt_config: Optional[BoltConfig] = None

        # Dataset storage for curriculum
        self._base_dataset_list: Optional[List[Dict[str, Any]]] = None
        self._prompt_keys: Optional[List[str]] = None
        self.custom_evaluator = None 
        self.eval_dataset = None


    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and BOLT dependencies are available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import numpy as np
            return True
        except ImportError:
            return False

    def _get_bolt_config(self) -> BoltConfig:
        """Extract BOLT configuration from TrainingConfig."""
        if self.bolt_config is not None:
            return self.bolt_config

        train_cfg = self.config.train

        self.bolt_config = BoltConfig(
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
        logger.info("BOLT Configuration:")
        logger.info(f"  Curriculum enabled: {self.bolt_config.curriculum_enabled}")
        logger.info(f"  Baseline enabled: {self.bolt_config.baseline_enabled}")
        logger.info(f"  Use baseline advantages: {self.bolt_config.use_baseline_advantages}")
        if self.bolt_config.curriculum_enabled:
            logger.info(f"  Curriculum epsilon: {self.bolt_config.curriculum_epsilon}")
            logger.info(f"  Curriculum update freq: {self.bolt_config.curriculum_update_freq}")
        if self.bolt_config.baseline_enabled:
            logger.info(f"  Baseline rho: [{self.bolt_config.baseline_rho_min}, {self.bolt_config.baseline_rho_max}]")
            logger.info(f"  Baseline D_half: {self.bolt_config.baseline_D_half}")
            if self.bolt_config.baseline_warm_start:
                logger.info(f"  Baseline warm-start: {self.bolt_config.baseline_warm_start}")
        logger.info("=" * 60)

        return self.bolt_config

    def _combined_reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """Combined reward function with DETAILED debug output for BOLT.

        Shows complete prompts, test cases, and completions to debug reward=0 issues.
        """
        if not completions:
            return []

        batch_rewards = []

        # Extract batch data from kwargs
        test_lists = kwargs.get('test_list', [None] * len(completions))
        prompts = kwargs.get('prompt', kwargs.get('query', [None] * len(completions)))

        # Ensure lists
        if not isinstance(test_lists, list):
            test_lists = [test_lists] * len(completions)
        if not isinstance(prompts, list):
            prompts = [prompts] * len(completions)

        # Debug: Print first batch details ONCE per training run
        if not hasattr(self, '_debug_printed_first_batch'):
            print("\n" + "="*80)
            print("BOLT DEBUG: FIRST BATCH DETAILED INFO")
            print("="*80)
            print(f"kwargs keys: {list(kwargs.keys())}")
            print(f"Number of completions: {len(completions)}")
            print(f"Number of test_lists: {len(test_lists)}")
            print(f"Number of prompts: {len(prompts)}")

            # Show first example in full detail
            if len(completions) > 0:
                print("\n" + "-"*80)
                print("EXAMPLE 0 - FULL DETAILS:")
                print("-"*80)
                print(f"PROMPT:\n{prompts[0] if prompts[0] else 'None'}")
                print("-"*40)
                print(f"TEST CASES:\n{test_lists[0] if test_lists[0] else 'None'}")
                print("-"*40)
                print(f"COMPLETION (full):\n{completions[0]}")
                print("-"*80)

            self._debug_printed_first_batch = True

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
                        except:
                            reward = 0.0

                    weighted_reward = weight * reward
                    total_reward += weighted_reward

                except Exception as e:
                    logger.warning(f"Error computing reward {rf['name']}: {e}")

            batch_rewards.append(total_reward)

        # Print batch summary with MORE details
        if batch_rewards:
            successful = sum(1 for r in batch_rewards if r > 0.5)
            partial = sum(1 for r in batch_rewards if 0 < r <= 0.5)
            failed = sum(1 for r in batch_rewards if r == 0)

            print(f"\n{'='*80}")
            print(f"BATCH REWARDS: {successful} passed | {partial} partial | {failed} failed | total={len(batch_rewards)}")
            print(f"Reward stats: min={min(batch_rewards):.2f}, max={max(batch_rewards):.2f}, mean={sum(batch_rewards)/len(batch_rewards):.2f}")

            # Show first 2 examples with test cases
            for i in range(min(2, len(completions))):
                reward = batch_rewards[i]
                status = "✓" if reward > 0.5 else ("~" if reward > 0 else "✗")
                test_case = test_lists[i] if i < len(test_lists) else None

                print(f"\n  [{status}] Example {i} (reward={reward:.2f}):")
                if test_case:
                    test_preview = str(test_case)[:200]
                    print(f"      TEST: {test_preview}")
                print(f"      CODE: {completions[i][:400]}...")

            print(f"{'='*80}\n")

        return batch_rewards

    def _enhance_code_prompt(self, original_prompt: str, test_list: list, ref_code: str = "") -> str:
        """
        Enhance prompt for code generation tasks with explicit format requirements.

        Based on reference implementation in phase1_curriculum_code.py.
        This prevents the model from generating verbose explanations.
        """
        if not test_list:
            return original_prompt

        # Extract function name from test case
        test_str = test_list[0] if isinstance(test_list, list) else str(test_list)
        func_name = "your_function"
        func_args = []

        # Try to extract function name from assertion
        match = re.search(r'assert\s+(\w+)\s*\(', test_str)
        if match:
            func_name = match.group(1)

            # Try to extract function args from reference code
            if ref_code:
                func_match = re.search(rf'def\s+{re.escape(func_name)}\s*\(([^)]*)\)', ref_code)
                if func_match:
                    params_str = func_match.group(1)
                    func_args = [p.strip().split('=')[0].strip() for p in params_str.split(',') if p.strip()]

            # If no args found from code, estimate from test call
            if not func_args:
                call_match = re.search(rf'{re.escape(func_name)}\s*\(([^)]*)\)', test_str)
                if call_match:
                    args_str = call_match.group(1)
                    # Count comma-separated args (rough estimate)
                    arg_count = len([a for a in args_str.split(',') if a.strip()])
                    func_args = [f"arg{i+1}" for i in range(max(1, arg_count))]

        # Build structured prompt
        signature = f"def {func_name}({', '.join(func_args)}):"

        enhanced_prompt = f"""{original_prompt}

IMPLEMENT EXACTLY THIS FUNCTION:

{signature}
    # Your implementation here

REQUIREMENTS:
- Use ONLY this function name: {func_name}
- Accept exactly these parameters: {', '.join(func_args) if func_args else 'none'}
- Return a value, do not print
- Use only ASCII characters
- Output ONLY Python code
- No markdown, no explanations, no examples"""

        return enhanced_prompt

    def _enhance_dataset_prompts(self) -> None:
        """
        Re-process dataset prompts with code-specific enhancements.

        This rebuilds prompts with explicit format requirements and
        re-applies the chat template.
        """
        from datasets import Dataset

        # Get enable_thinking setting
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)

        def enhance_example(example):
            """Enhance a single example's prompt."""
            test_list = example.get('test_list')
            if not test_list:
                return example

            # Get reference code for extracting function signature
            ref_code = example.get('code', example.get('canonical_solution', ''))

            # Get original prompt text (try to extract from various sources)
            # The parent already formatted with chat template, so we need the raw text
            # We'll use the 'text' or 'prompt' column if available, or extract from formatted
            original_text = example.get('text', '')
            if not original_text:
                # Try to extract user content from formatted prompt
                formatted = example.get('query', example.get('prompt', ''))
                # Look for content after user role marker
                if '<|im_start|>user' in formatted:
                    # Qwen format
                    start = formatted.find('<|im_start|>user') + len('<|im_start|>user')
                    end = formatted.find('<|im_end|>', start)
                    if end > start:
                        original_text = formatted[start:end].strip()
                elif '[INST]' in formatted:
                    # Llama format
                    start = formatted.find('[INST]') + len('[INST]')
                    end = formatted.find('[/INST]', start)
                    if end > start:
                        original_text = formatted[start:end].strip()
                else:
                    # Fallback: use the formatted prompt as-is but strip common templates
                    original_text = formatted

            if not original_text:
                return example

            # Enhance the prompt
            enhanced_text = self._enhance_code_prompt(original_text, test_list, ref_code)

            # Re-apply chat template
            messages = [{"role": "user", "content": enhanced_text}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                try:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    formatted_prompt = enhanced_text

            # Update the example
            result = dict(example)
            result['query'] = formatted_prompt
            result['prompt'] = formatted_prompt
            return result

        # Apply enhancement to all examples
        self.train_dataset = self.train_dataset.map(
            enhance_example,
            desc="Enhancing code prompts"
        )


    def setup_data(self) -> None:
        """
        Setup datasets with BOLT curriculum sampling.

        Extends parent to:
        1. Enhance code prompts with explicit format requirements
        2. Initialize baseline (cold or warm-start)
        3. Wrap dataset with curriculum sampler (if enabled)
        """
        # Call parent to load dataset
        super().setup_data()

        # Enhance prompts for code generation tasks
        # Check if this is a code dataset (has test_list column)
        if hasattr(self, 'dataset') and self.train_dataset is not None:
            if 'test_list' in self.train_dataset.column_names:
                logger.info("Enhancing code prompts with explicit format requirements...")
                self._enhance_dataset_prompts()

        bolt_cfg = self._get_bolt_config()

        # Skip if neither curriculum nor baseline enabled
        if not bolt_cfg.curriculum_enabled and not bolt_cfg.baseline_enabled and not bolt_cfg.use_baseline_advantages:
            logger.info("BOLT features disabled, using standard GRPO")
            return

        # Convert dataset to list and extract prompt keys
        logger.info("Preparing dataset for BOLT...")
        self._base_dataset_list = list(self.train_dataset)
        self._prompt_keys = []

        for item in self._base_dataset_list:
            # Try different column names for prompt
            prompt = item.get("prompt") or item.get("query") or item.get("question") or ""
            key = make_prompt_key(prompt)
            self._prompt_keys.append(key)

        logger.info(f"Extracted {len(self._prompt_keys)} prompt keys")

        # Initialize baseline
        if bolt_cfg.baseline_enabled or bolt_cfg.curriculum_enabled or bolt_cfg.use_baseline_advantages:
            logger.info("Initializing BOLT baseline...")
            self.baseline = UnifiedBaseline(
                rho_min=bolt_cfg.baseline_rho_min,
                rho_max=bolt_cfg.baseline_rho_max,
                D_half=bolt_cfg.baseline_D_half,
                epsilon=bolt_cfg.curriculum_epsilon,
                track_timeline=False,  # Disable for production
            )

            # Load warm-start if provided
            if bolt_cfg.baseline_warm_start:
                logger.info(f"Loading baseline warm-start from {bolt_cfg.baseline_warm_start}")
                self.baseline.load(bolt_cfg.baseline_warm_start)
                logger.info(f"Loaded {len(self.baseline.tab)} baseline entries")

        # Setup curriculum sampling
        if bolt_cfg.curriculum_enabled and self.baseline is not None:
            logger.info("Setting up curriculum sampling...")
            self.curriculum_sampler = DynamicWeightedSampler(
                baseline=self.baseline,
                dataset_size=len(self._base_dataset_list),
                prompt_keys=self._prompt_keys,
                epsilon=bolt_cfg.curriculum_epsilon,
                oversample=2.0,
            )

            # Wrap dataset
            self.train_dataset = DynamicallySampledDataset(
                base_dataset=self._base_dataset_list,
                sampler=self.curriculum_sampler,
            )

            logger.info(f"Curriculum dataset size: {len(self.train_dataset)} (2x oversampled)")

    def setup_trainer(self) -> None:
        """
        Setup the BOLT trainer with all necessary configurations and callbacks.
        
        This configures the trainer but does not start training.
        Call train() to begin the training process.
        """
        bolt_cfg = self._get_bolt_config()

        # Get training parameters
        num_epochs = self._get_config_value(self.config.train, 'epochs', 'num_epochs', 'num_train_epochs', default=1)
        max_steps = self._get_config_value(self.config.train, 'max_steps', default=10)
        learning_rate = self._get_config_value(self.config.train, 'learning_rate', 'lr', default=1e-6)
        per_device_batch_size = self._get_config_value(self.config.train, 'per_device_batch_size', 'batch_size', default=1)
        gradient_accumulation_steps = self._get_config_value(self.config.train, 'gradient_accumulation_steps', default=32)
        max_grad_norm = self._get_config_value(self.config.train, 'max_grad_norm', default=1.0)
        weight_decay = self._get_config_value(self.config.train, 'weight_decay', default=0.0)
        warmup_steps = self._get_config_value(self.config.train, 'warmup_steps', default=10)
        save_steps = self._get_config_value(self.config.train, 'save_steps', default=100)
        logging_steps = self._get_config_value(self.config.train, 'logging_steps', default=10)
        eval_steps = self._get_config_value(self.config.train, 'eval_steps', default=None)

        # === UNIFIED PRECISION HANDLING ===
        precision = PrecisionHandler.get_precision_from_config(self.config, default="auto")
        precision = PrecisionHandler.validate_precision(precision)
        PrecisionHandler.log_precision_info(precision, "BOLT")
        precision_args = PrecisionHandler.get_training_args_precision(precision)

        gradient_checkpointing = self._get_config_value(self.config.model, 'gradient_checkpointing', default=False)
        output_dir = self._get_config_value(self.config.logging, 'output_dir', default='./output/bolt')
        num_generations = self._get_config_value(self.config.train, 'num_generations', default=8)
        seed = self._get_config_value(self.config.train, 'seed', default=42)

        # GRPO-specific parameters
        beta = self._get_config_value(self.config.train, 'beta', 'kl_coef', default=0.1)
        epsilon = self._get_config_value(self.config.train, 'epsilon', 'cliprange', default=0.2)

        # Generation parameters - CRITICAL for proper training
        max_completion_length = self._get_config_value(self.config.train, 'max_completion_length', 'max_new_tokens', default=512)
        max_prompt_length = self._get_config_value(self.config.train, 'max_prompt_length', default=512)
        temperature = self._get_config_value(self.config.train, 'temperature', default=0.7)
        top_p = self._get_config_value(self.config.train, 'top_p', default=0.95)
        top_k = self._get_config_value(self.config.train, 'top_k', default=50)
        
        # Evaluation parameters
        eval_strategy = self._get_config_value(self.config.train, 'eval_strategy', default='epoch')
        eval_dataset = self._get_config_value(self.config.train, 'eval_dataset', default=None)
        per_device_eval_batch_size = self._get_config_value(self.config.train, 'per_device_eval_batch_size', default=per_device_batch_size)

        # Logging parameters - enhanced
        # report_to = self._get_config_value(self.config.logging, 'report_to', default='none')
        report_to = self.config.logging.loggers if self.config.logging.loggers else []
        run_name = self._get_config_value(self.config.logging, 'run_name', default=None)
        logging_dir = self._get_config_value(self.config.logging, 'logging_dir', default=None)
        logging_first_step = self._get_config_value(self.config.train, 'logging_first_step', default=False)
        logging_nan_inf_filter = self._get_config_value(self.config.train, 'logging_nan_inf_filter', default=True)

        # Save/checkpoint parameters
        save_strategy = self._get_config_value(self.config.train, 'save_strategy', default='steps')
        save_total_limit = self._get_config_value(self.config.train, 'save_total_limit', default=None)
        load_best_model_at_end = self._get_config_value(self.config.train, 'load_best_model_at_end', default=False)
        metric_for_best_model = self._get_config_value(self.config.train, 'metric_for_best_model', default=None)
        greater_is_better = self._get_config_value(self.config.train, 'greater_is_better', default=False)

        # Additional training parameters
        save_safetensors = self._get_config_value(self.config.train, 'save_safetensors', default=True)
        save_only_model = self._get_config_value(self.config.train, 'save_only_model', default=False)

        # Use eval_dataset-aware defaults
        if self.eval_dataset != None:
            eval_strategy = eval_strategy if eval_strategy != 'no' else 'epoch'
        else:
            eval_strategy = 'no'
            eval_steps = None

        # Print comprehensive parameter summary
        print("\n" + "=" * 80)
        print("BOLT TRAINING PARAMETERS - CONFIG vs ACTUAL VALUES")
        print("=" * 80)
        print(f"{'Parameter':<35} {'Config Value':<20} {'Used Value':<20}")
        print("-" * 80)

        # Helper to get attr from dataclass or dict
        def get_cfg(obj, key, default='NOT SET'):
            if hasattr(obj, key):
                val = getattr(obj, key)
                return val if val is not None else default
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default

        # Training params
        cfg_train = self.config.train
        print(f"{'epochs':<35} {str(get_cfg(cfg_train, 'epochs')):<20} {num_epochs:<20}")
        print(f"{'max_steps':<35} {str(get_cfg(cfg_train, 'max_steps')):<20} {str(max_steps):<20}")
        print(f"{'per_device_batch_size':<35} {str(get_cfg(cfg_train, 'per_device_batch_size')):<20} {per_device_batch_size:<20}")
        print(f"{'gradient_accumulation_steps':<35} {str(get_cfg(cfg_train, 'gradient_accumulation_steps')):<20} {gradient_accumulation_steps:<20}")
        print(f"{'num_generations':<35} {str(get_cfg(cfg_train, 'num_generations')):<20} {num_generations:<20}")
        print(f"{'learning_rate':<35} {str(get_cfg(cfg_train, 'learning_rate')):<20} {learning_rate:<20}")
        print(f"{'max_prompt_length':<35} {str(get_cfg(cfg_train, 'max_prompt_length')):<20} {max_prompt_length:<20}")
        print(f"{'max_completion_length':<35} {str(get_cfg(cfg_train, 'max_completion_length')):<20} {max_completion_length:<20}")
        print(f"{'temperature':<35} {str(get_cfg(cfg_train, 'temperature')):<20} {temperature:<20}")
        print(f"{'top_p':<35} {str(get_cfg(cfg_train, 'top_p')):<20} {top_p:<20}")
        print(f"{'top_k':<35} {str(get_cfg(cfg_train, 'top_k', 'NOT SET')):<20} {top_k:<20}")
        print(f"{'save_steps':<35} {str(get_cfg(cfg_train, 'save_steps')):<20} {save_steps:<20}")
        print(f"{'logging_steps':<35} {str(get_cfg(cfg_train, 'logging_steps')):<20} {logging_steps:<20}")
        print(f"{'seed':<35} {str(get_cfg(cfg_train, 'seed')):<20} {seed:<20}")
        print(f"{'warmup_steps':<35} {str(get_cfg(cfg_train, 'warmup_steps')):<20} {warmup_steps:<20}")
        print(f"{'weight_decay':<35} {str(get_cfg(cfg_train, 'weight_decay')):<20} {weight_decay:<20}")
        print(f"{'max_grad_norm':<35} {str(get_cfg(cfg_train, 'max_grad_norm')):<20} {max_grad_norm:<20}")
        print(f"{'beta':<35} {str(get_cfg(cfg_train, 'beta')):<20} {beta:<20}")
        print(f"{'epsilon':<35} {str(get_cfg(cfg_train, 'epsilon')):<20} {epsilon:<20}")
        print(f"{'eval_steps':<35} {str(get_cfg(cfg_train, 'eval_steps')):<20} {eval_steps if eval_steps else save_steps:<20}")

        # Model params
        cfg_model = self.config.model
        print("-" * 80)
        print(f"{'precision':<35} {str(get_cfg(cfg_model, 'precision')):<20} {precision:<20}")
        print(f"{'gradient_checkpointing':<35} {str(get_cfg(cfg_model, 'gradient_checkpointing')):<20} {gradient_checkpointing:<20}")

        # Logging params
        cfg_logging = self.config.logging
        print("-" * 80)
        print(f"{'output_dir':<35} {str(get_cfg(cfg_logging, 'output_dir'))[:18]:<20} {str(output_dir)[:18]:<20}")

        # Effective batch size calculation
        effective_batch = per_device_batch_size * gradient_accumulation_steps
        samples_per_step = effective_batch * num_generations
        print("-" * 80)
        print(f"{'EFFECTIVE BATCH SIZE':<35} {'':<20} {effective_batch:<20}")
        print(f"{'SAMPLES PER STEP (batch*K)':<35} {'':<20} {samples_per_step:<20}")
        print("=" * 80 + "\n")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup GRPO config
        from trl import GRPOConfig

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
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
            # Generation parameters
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # GRPO-specific
            beta=beta,
            epsilon=epsilon,
            # Precision (unified handling)
            **precision_args,
             # Evaluation
            eval_strategy=eval_strategy,
            per_device_eval_batch_size=per_device_eval_batch_size,
            
            # Logging
            report_to=report_to,
            run_name=run_name,
            logging_dir=logging_dir,
            logging_first_step=logging_first_step,
            logging_nan_inf_filter=logging_nan_inf_filter,
            
            # Saving
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_safetensors=save_safetensors,
            save_only_model=save_only_model,
        )

        # Create trainer - use custom BoltGRPOTrainer if baseline advantages enabled
        if bolt_cfg.use_baseline_advantages and self.baseline is not None:
            logger.info("Using BoltGRPOTrainer with baseline advantage computation")
            TrainerClass = BoltGRPOTrainer.create_trainer_class(
                baseline=self.baseline,
                num_generations=num_generations,
            )
        else:
            logger.info("Using standard GRPOTrainer (no baseline advantages)")
            from trl import GRPOTrainer
            TrainerClass = GRPOTrainer

        # Create trainer instance
        self.trainer = TrainerClass(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self._combined_reward_function,
        )

        # Add curriculum callback if enabled
        if bolt_cfg.curriculum_enabled and self.curriculum_sampler is not None and self.baseline is not None:
            self.trainer.add_callback(CurriculumCallback(
                baseline=self.baseline,
                sampler=self.curriculum_sampler,
                dataset=self.train_dataset,
                update_freq=bolt_cfg.curriculum_update_freq,
            ))
            logger.info(f"Added CurriculumCallback (update every {bolt_cfg.curriculum_update_freq} steps)")

        # Add baseline checkpoint callback to save baseline with every checkpoint
        if self.baseline is not None:
            self.trainer.add_callback(BaselineCheckpointCallback(
                baseline=self.baseline,
                output_dir=output_dir,
            ))
            logger.info("Added BaselineCheckpointCallback (saves baseline with each checkpoint)")

        # Store output_dir for use in train()
        self.output_dir = output_dir

        logger.info("=" * 80)
        logger.info("BOLT Trainer Setup Complete")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        logger.info(f"Num reward functions: {len(self.reward_functions)}")
        logger.info(f"Baseline advantages: {bolt_cfg.use_baseline_advantages}")
        logger.info(f"Curriculum sampling: {bolt_cfg.curriculum_enabled}")
        logger.info("=" * 80)


    def train(self) -> Dict[str, Any]:
        """
        Execute BOLT training with curriculum and baseline callbacks.

        This method performs all setup steps and then runs the training loop.
        """
        # Setup all components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()
        self.setup_trainer()

        # Record training start
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting BOLT Training")
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
            baseline_path = Path(self.output_dir) / "bolt_baseline.pkl"
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

        # Get BOLT config for results
        bolt_cfg = self._get_bolt_config()

        # Compile results
        results = {
            "training_time": training_duration,
            "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else metrics.get('train_loss', 0.0),
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else 0,
            "model_path": self.output_dir,
            "baseline_path": str(baseline_path) if baseline_path else None,
            "num_reward_functions": len(self.reward_functions),
            "num_datasets": 1,
            "bolt_config": {
                "curriculum_enabled": bolt_cfg.curriculum_enabled,
                "baseline_enabled": bolt_cfg.baseline_enabled,
                "use_baseline_advantages": bolt_cfg.use_baseline_advantages,
            },
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("BOLT Training Completed Successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Model saved to: {results['model_path']}")
        if self.baseline:
            logger.info(f"Baseline prompts tracked: {len(self.baseline.tab)}")
        logger.info("=" * 80)

        return results

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
    
    