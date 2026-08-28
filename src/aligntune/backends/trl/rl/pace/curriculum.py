"""
BOLT Curriculum Module - Dynamic sampling and training callbacks.

Implements uncertainty-based curriculum learning:
- DynamicWeightedSampler: Samples prompts with weights ∝ sqrt(v̂(1-v̂)) + ε
- DynamicallySampledDataset: Dataset wrapper with refreshable indices
- CurriculumCallback: Updates weights every N steps
- BaselineUpdateCallback: Applies pending baseline updates
"""

import logging
from typing import Dict, List, Any, Optional, Iterator
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .baseline import UnifiedBaseline, make_prompt_key

logger = logging.getLogger(__name__)


class DynamicWeightedSampler(Sampler):
    """
    Sampler that draws samples based on dynamically updated weights.

    Weights are computed from baseline uncertainty: w(x) = sqrt(v̂(1-v̂)) + ε
    - Maximum weight at v̂ = 0.5 (most uncertain)
    - Low weight at v̂ → 0 or v̂ → 1 (confident)
    - ε provides a floor to ensure all prompts have some probability
    """

    def __init__(
        self,
        baseline: UnifiedBaseline,
        dataset_size: int,
        prompt_keys: List[str],
        epsilon: float = 0.05,
        oversample: float = 2.0,
    ):
        """
        Args:
            baseline: UnifiedBaseline for computing weights
            dataset_size: Number of prompts in dataset
            prompt_keys: List of prompt keys corresponding to dataset indices
            epsilon: Floor for sampling weights
            oversample: Multiplier for number of samples per epoch
        """
        self.baseline = baseline
        self.dataset_size = dataset_size
        self.prompt_keys = prompt_keys
        self.epsilon = epsilon
        self.num_samples = int(dataset_size * oversample)

        # Cached indices and weights
        self._weights: Optional[np.ndarray] = None
        self._cached_indices: Optional[List[int]] = None
        self._cache_epoch = -1

        # Initialize weights
        self.update_weights()

    def update_weights(self):
        """Recompute weights from current baseline state."""
        weights = []
        for key in self.prompt_keys:
            weight = self.baseline.get_sampling_weight(key)
            weights.append(weight)

        self._weights = np.array(weights, dtype=np.float64)
        # Invalidate cache
        self._cached_indices = None

        logger.debug(
            f"Updated curriculum weights: mean={self._weights.mean():.4f}, "
            f"std={self._weights.std():.4f}"
        )

    def get_indices_for_epoch(self, epoch: int) -> List[int]:
        """Get indices for this epoch (with caching)."""
        if self._cached_indices is None or epoch != self._cache_epoch:
            if self._weights is None:
                self.update_weights()

            probs = self._weights / self._weights.sum()
            self._cached_indices = np.random.choice(
                self.dataset_size,
                size=self.num_samples,
                replace=True,
                p=probs,
            ).tolist()
            self._cache_epoch = epoch

        return self._cached_indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_indices_for_epoch(self._cache_epoch + 1))

    def __len__(self) -> int:
        return self.num_samples


class DynamicallySampledDataset(Dataset):
    """
    Dataset wrapper that uses DynamicWeightedSampler for curriculum learning.

    Indices are refreshed when curriculum updates, allowing dynamic resampling
    during training without recreating the dataset.
    """

    def __init__(self, base_dataset: List[Dict[str, Any]], sampler: DynamicWeightedSampler):
        """
        Args:
            base_dataset: List of dataset items (dicts with 'prompt', etc.)
            sampler: DynamicWeightedSampler for curriculum sampling
        """
        self.base_dataset = base_dataset
        self.sampler = sampler
        self.current_indices = self.sampler.get_indices_for_epoch(0)
        self._refresh_count = 0

    def refresh_indices(self, epoch: Optional[int] = None):
        """
        Refresh indices from sampler (called by callback).

        Args:
            epoch: Optional epoch number for caching
        """
        self._refresh_count += 1
        if epoch is None:
            epoch = self._refresh_count
        self.current_indices = self.sampler.get_indices_for_epoch(epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self.current_indices[idx]
        return self.base_dataset[actual_idx]

    def __len__(self) -> int:
        return len(self.current_indices)


class BaselineRewardWrapper:
    """
    DEPRECATED: This class is no longer used for advantage computation.

    The correct approach is to override _generate_and_score_completions in the
    trainer class (see BoltGRPOTrainer in bolt.py). TRL always computes advantages
    as A = r - mean(r_group) in _generate_and_score_completions, so any wrapper-based
    approach will have its baseline subtraction cancelled out by TRL's group mean.

    This class is kept for backwards compatibility but should not be used.

    Original description:
    Wrapper that computes advantages using persistent baseline.

    Key flow:
    1. Pre-update read: Get v̂(x) before incorporating new reward
    2. Compute advantage: A = r - v̂(x)
    3. Queue update: Store (key, reward) for later application
    4. Post-step: Apply all pending updates with KL-adaptive forgetting

    This ensures advantages are computed against historical performance,
    not the current batch.
    """

    def __init__(self, baseline: UnifiedBaseline, normalize: bool = True):
        """
        Args:
            baseline: UnifiedBaseline for v̂(x) lookups
            normalize: Whether to normalize advantages across batch
        """
        self.baseline = baseline
        self.normalize = normalize
        self.pending_updates: List[tuple] = []  # [(key, reward), ...]

    def compute_advantages(
        self,
        rewards: List[float],
        prompt_keys: List[str],
    ) -> List[float]:
        """
        Compute advantages A = r - v̂(x) for each reward.

        Args:
            rewards: List of rewards
            prompt_keys: Corresponding prompt keys

        Returns:
            List of advantages (normalized if enabled)
        """
        advantages = []
        for reward, key in zip(rewards, prompt_keys):
            # Pre-update read
            v_hat = self.baseline.get_v_hat(key)

            # Compute advantage
            advantage = reward - v_hat
            advantages.append(advantage)

            # Queue for later update
            self.pending_updates.append((key, reward))

        # Normalize advantages
        if self.normalize and len(advantages) > 1:
            mean_adv = np.mean(advantages)
            std_adv = np.std(advantages) + 1e-8
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        return advantages

    def apply_pending_updates(self):
        """
        Apply all pending baseline updates with KL-adaptive forgetting.

        Groups rewards by prompt to estimate KL divergence.
        """
        if not self.pending_updates:
            return

        # Group by prompt key
        prompt_rewards: Dict[str, List[float]] = defaultdict(list)
        for key, reward in self.pending_updates:
            prompt_rewards[key].append(reward)

        # Apply updates with estimated KL
        for key, rewards in prompt_rewards.items():
            # Estimate KL from performance shift
            kl = self.baseline.estimate_kl_from_performance_shift(key, rewards)

            # Apply each reward update
            for reward in rewards:
                self.baseline.update(key, reward, kl=kl)

        # Clear pending
        self.pending_updates.clear()

        logger.debug(f"Applied baseline updates for {len(prompt_rewards)} prompts")


class CurriculumCallback(TrainerCallback):
    """
    Callback that updates curriculum sampling weights periodically.

    Every `update_freq` steps:
    1. Recompute weights from current baseline
    2. Refresh dataset indices
    3. Log curriculum statistics
    """

    def __init__(
        self,
        baseline: UnifiedBaseline,
        sampler: DynamicWeightedSampler,
        dataset: DynamicallySampledDataset,
        update_freq: int = 10,
    ):
        """
        Args:
            baseline: UnifiedBaseline for stats
            sampler: DynamicWeightedSampler to update
            dataset: DynamicallySampledDataset to refresh
            update_freq: Steps between updates
        """
        self.baseline = baseline
        self.sampler = sampler
        self.dataset = dataset
        self.update_freq = update_freq
        self.last_update_step = -1
        self.resample_count = 0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Before step - check if we need to resample."""
        if state.global_step > 0 and state.global_step % self.update_freq == 0:
            if state.global_step != self.last_update_step:
                self._update_and_resample(state.global_step)
                self.last_update_step = state.global_step

        return control

    def _update_and_resample(self, step: int):
        """Recompute weights and resample dataset based on current baseline."""
        self.resample_count += 1

        # Update sampler weights from baseline
        self.sampler.update_weights()

        # Refresh dataset indices
        self.dataset.refresh_indices(epoch=self.resample_count)

        # Log statistics
        if self.sampler._weights is not None:
            weights = self.sampler._weights
            probs = weights / weights.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            logger.info(
                f"Step {step}: Curriculum update #{self.resample_count} - "
                f"weight μ={weights.mean():.3f}, entropy={norm_entropy:.3f}"
            )


class BaselineUpdateCallback(TrainerCallback):
    """
    Callback that applies pending baseline updates after each optimization step.

    Ensures baseline updates happen AFTER the gradient update, maintaining
    the pre-update read semantics for advantage computation.
    """

    def __init__(
        self,
        baseline: UnifiedBaseline,
        reward_wrapper: Optional[BaselineRewardWrapper] = None,
    ):
        """
        Args:
            baseline: UnifiedBaseline to update
            reward_wrapper: Optional wrapper with pending updates
        """
        self.baseline = baseline
        self.reward_wrapper = reward_wrapper

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """After optimizer step - apply pending updates."""
        # Update baseline step counter
        self.baseline.set_step(state.global_step)

        # Apply pending updates if wrapper provided
        if self.reward_wrapper is not None:
            self.reward_wrapper.apply_pending_updates()

        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> TrainerControl:
        """Add baseline/curriculum stats to logs."""
        if logs is not None:
            logs.update(self.baseline.get_baseline_stats())
            logs.update(self.baseline.get_curriculum_stats())

        return control


class BaselineCheckpointCallback(TrainerCallback):
    """
    Callback that saves the baseline with every checkpoint.

    This ensures the baseline state is preserved alongside the model checkpoint,
    allowing training to be resumed with the correct baseline values.
    """

    def __init__(
        self,
        baseline: UnifiedBaseline,
        output_dir: str,
    ):
        """
        Args:
            baseline: UnifiedBaseline to save
            output_dir: Base output directory
        """
        self.baseline = baseline
        self.output_dir = output_dir
        self.saved_checkpoints = []

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Save baseline alongside model checkpoint."""
        import os
        from pathlib import Path

        # Determine checkpoint directory
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            checkpoint_dir = Path(args.output_dir)

        # Save baseline to checkpoint directory
        baseline_path = checkpoint_dir / "bolt_baseline.pkl"
        self.baseline.save(str(baseline_path))
        self.saved_checkpoints.append(str(baseline_path))

        # Print detailed summary
        print(f"\n{'='*80}")
        print(f"CHECKPOINT SAVED: Step {state.global_step}")
        print(f"{'='*80}")
        print(f"Baseline saved to: {baseline_path}")
        print(f"Total prompts tracked: {len(self.baseline.tab)}")

        if self.baseline.tab:
            v_hats = [a/(a+b) for a, b, _ in self.baseline.tab.values()]
            print(f"Baseline stats:")
            print(f"  Mean v̂: {np.mean(v_hats):.4f}")
            print(f"  Std v̂:  {np.std(v_hats):.4f}")
            print(f"  Min v̂:  {min(v_hats):.4f}")
            print(f"  Max v̂:  {max(v_hats):.4f}")

            # Distribution summary
            n_easy = sum(1 for v in v_hats if v >= 0.7)
            n_hard = sum(1 for v in v_hats if v <= 0.3)
            n_edge = sum(1 for v in v_hats if 0.3 < v < 0.7)
            print(f"  Easy (v̂≥0.7): {n_easy} | Edge (0.3<v̂<0.7): {n_edge} | Hard (v̂≤0.3): {n_hard}")

        print(f"{'='*80}\n")

        logger.info(f"Saved baseline to checkpoint: {baseline_path} ({len(self.baseline.tab)} prompts)")

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Save final baseline at end of training."""
        from pathlib import Path

        # Save final baseline
        final_path = Path(args.output_dir) / "bolt_baseline_final.pkl"
        self.baseline.save(str(final_path))

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final baseline saved to: {final_path}")
        print(f"Total prompts tracked: {len(self.baseline.tab)}")
        print(f"Checkpoints saved: {len(self.saved_checkpoints)}")
        for cp in self.saved_checkpoints[-5:]:  # Show last 5
            print(f"  - {cp}")
        if len(self.saved_checkpoints) > 5:
            print(f"  ... and {len(self.saved_checkpoints) - 5} more")
        print(f"{'='*80}\n")

        # Print final detailed summary
        self.baseline._print_baseline_summary(max_display=30)

        return control
