"""
Curriculum Learning Callback for adaptive difficulty sampling during training.

Integrates CurriculumSampler with trainer to track per-example pass rates
and adaptively adjust sampling based on training progress.
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json

from .base import TrainerCallback

logger = logging.getLogger(__name__)


class CurriculumCallback(TrainerCallback):
    """
    Callback that manages curriculum learning during training.

    Tracks per-example success rates and updates the curriculum sampler
    to adaptively adjust which examples are sampled based on difficulty.
    """

    def __init__(
        self,
        curriculum_sampler: Optional[Any] = None,
        pass_threshold: float = 0.5,
        update_interval: int = 10,
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize curriculum callback.

        Args:
            curriculum_sampler: CurriculumSampler instance to manage
            pass_threshold: Reward threshold above which an example counts as "passed"
            update_interval: Steps between logging curriculum progress
            log_dir: Directory to save curriculum progress logs
            enable_logging: Whether to log curriculum statistics
        """
        self.curriculum_sampler = curriculum_sampler
        self.pass_threshold = pass_threshold
        self.update_interval = update_interval
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir) if log_dir else None

        # Track current batch info for on_step_end callback
        self._last_rewards = None
        self._last_example_ids = None

        # Statistics tracking
        self._step_count = 0
        self._total_examples = 0
        self._total_passed = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Curriculum logs will be saved to {self.log_dir}")

        logger.info(
            f"CurriculumCallback initialized: pass_threshold={pass_threshold}, "
            f"update_interval={update_interval}"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of training.

        Initializes curriculum tracking and verifies sampler is configured.
        """
        logger.info("Starting curriculum learning training")

        if self.curriculum_sampler is None:
            logger.warning("No curriculum sampler provided - curriculum learning disabled")
            return

        # Log initial curriculum state
        if self.enable_logging:
            progress = self.curriculum_sampler.get_curriculum_progress()
            logger.info(
                f"Initial curriculum state: "
                f"avg_difficulty={progress.avg_difficulty:.3f}, "
                f"examples_with_data={progress.examples_with_data}"
            )

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step.

        Updates pass rates based on rewards from the step.
        """
        if self.curriculum_sampler is None:
            return

        self._step_count += 1

        # Extract rewards and example IDs from kwargs
        rewards = kwargs.get("rewards")
        example_ids = kwargs.get("example_ids")

        if rewards is not None and example_ids is not None:
            self._update_pass_rates(rewards, example_ids)

        # Log progress periodically
        if self.enable_logging and self._step_count % self.update_interval == 0:
            self._log_curriculum_progress(state)

    def _update_pass_rates(self, rewards, example_ids):
        """
        Update pass rates based on rewards.

        Args:
            rewards: Tensor or list of rewards for batch
            example_ids: Tensor or list of example indices in batch
        """
        if rewards is None or example_ids is None:
            return

        # Convert to lists if needed
        if hasattr(rewards, "cpu"):
            rewards = rewards.cpu().tolist()
        elif not isinstance(rewards, list):
            rewards = list(rewards)

        if hasattr(example_ids, "cpu"):
            example_ids = example_ids.cpu().tolist()
        elif not isinstance(example_ids, list):
            example_ids = list(example_ids)

        # Update pass rate for each example
        for example_id, reward in zip(example_ids, rewards):
            example_id = int(example_id)
            reward = float(reward)

            # Determine if example "passed"
            passed = reward >= self.pass_threshold

            # Update curriculum sampler
            self.curriculum_sampler.update_pass_rate(example_id, passed)

            # Track statistics
            self._total_examples += 1
            if passed:
                self._total_passed += 1

    def _log_curriculum_progress(self, state):
        """
        Log curriculum progress statistics.

        Args:
            state: Training state object
        """
        if self.curriculum_sampler is None:
            return

        progress = self.curriculum_sampler.get_curriculum_progress()

        # Log to logger
        logger.info(
            f"Curriculum progress (step {self._step_count}): "
            f"avg_difficulty={progress.avg_difficulty:.3f}, "
            f"avg_pass_rate={progress.avg_pass_rate:.3f}, "
            f"examples_trained={progress.examples_with_data}, "
            f"overall_pass_rate={self._total_passed / max(self._total_examples, 1):.3f}"
        )

        # Save to JSON log
        if self.log_dir:
            self._save_curriculum_log(progress)

    def _save_curriculum_log(self, progress):
        """
        Save curriculum progress to JSON file.

        Args:
            progress: CurriculumStats object
        """
        log_data = {
            "step": self._step_count,
            "avg_difficulty": progress.avg_difficulty,
            "avg_pass_rate": progress.avg_pass_rate,
            "min_difficulty": progress.min_difficulty,
            "max_difficulty": progress.max_difficulty,
            "examples_with_data": progress.examples_with_data,
            "num_sampled": progress.num_sampled,
            "num_passed": progress.num_passed,
            "overall_pass_rate": self._total_passed / max(self._total_examples, 1),
        }

        log_file = self.log_dir / "curriculum_progress.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Get current curriculum statistics.

        Returns:
            Dictionary with curriculum metrics
        """
        if self.curriculum_sampler is None:
            return {}

        progress = self.curriculum_sampler.get_curriculum_progress()
        difficulty_dist = self.curriculum_sampler.get_difficulty_distribution()

        return {
            "avg_difficulty": progress.avg_difficulty,
            "avg_pass_rate": progress.avg_pass_rate,
            "min_difficulty": progress.min_difficulty,
            "max_difficulty": progress.max_difficulty,
            "examples_with_data": progress.examples_with_data,
            "num_sampled": progress.num_sampled,
            "num_passed": progress.num_passed,
            "overall_pass_rate": self._total_passed / max(self._total_examples, 1),
            "difficulty_histogram": difficulty_dist["histogram"],
            "difficulty_mean": difficulty_dist["mean"],
            "difficulty_std": difficulty_dist["std"],
        }

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.

        Logs final curriculum statistics and summary.
        """
        if self.curriculum_sampler is None:
            return

        logger.info("Training completed - final curriculum statistics:")
        stats = self.get_curriculum_stats()

        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
