"""
Curriculum Learning Sampler for Adaptive Difficulty Sampling.

Implements TAROT-inspired curriculum learning strategies for RL training,
tracking per-example pass rates and sampling based on difficulty progression.

Paper: https://huggingface.co/papers/2602.15449
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Supported curriculum learning strategies."""
    EASY_FIRST = "easy_first"
    ADAPTIVE = "adaptive"
    MIXED = "mixed"


@dataclass
class CurriculumStats:
    """Statistics tracking for curriculum progress."""
    avg_difficulty: float = 0.0
    avg_pass_rate: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    num_sampled: int = 0
    num_passed: int = 0
    steps_completed: int = 0
    examples_with_data: int = 0


class CurriculumSampler:
    """
    Adaptive curriculum sampler that tracks per-example pass rates and samples
    based on difficulty progression.

    Strategies:
    - easy_first: Sample easy examples early, transition to hard examples
    - adaptive: UCB-style balance between exploitation (easy) and exploration (hard)
    - mixed: 70% current difficulty level, 30% random difficulty
    """

    def __init__(
        self,
        dataset_size: int,
        strategy: str = "adaptive",
        warmup_steps: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize curriculum sampler.

        Args:
            dataset_size: Total number of examples in dataset
            strategy: Curriculum strategy to use (easy_first/adaptive/mixed)
            warmup_steps: Number of steps before curriculum takes effect
            seed: Random seed for reproducibility
        """
        if strategy not in [s.value for s in CurriculumStrategy]:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {[s.value for s in CurriculumStrategy]}")

        self.dataset_size = dataset_size
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Track per-example pass rates
        # Format: {example_id: {"passed": int, "total": int, "pass_rate": float}}
        self._pass_rates: Dict[int, Dict[str, Any]] = {}
        self._init_pass_rates()

        # Track sampling statistics
        self._current_step = 0
        self._stats = CurriculumStats()

        logger.info(
            f"CurriculumSampler initialized: strategy={strategy}, "
            f"warmup_steps={warmup_steps}, dataset_size={dataset_size}"
        )

    def _init_pass_rates(self):
        """Initialize pass rate tracking for all examples."""
        for i in range(self.dataset_size):
            self._pass_rates[i] = {
                "passed": 0,
                "total": 0,
                "pass_rate": 0.0,  # 0.5 means neutral difficulty
            }

    def difficulty_score(self, example_id: int) -> float:
        """
        Compute difficulty score for an example.

        difficulty = 1 - pass_rate
        - Score 0.0: Easy (always passes)
        - Score 0.5: Medium (50% pass rate)
        - Score 1.0: Hard (never passes)

        Args:
            example_id: Index of example

        Returns:
            Difficulty score in range [0.0, 1.0]
        """
        if example_id not in self._pass_rates:
            return 0.5  # Unknown examples are neutral difficulty

        pass_rate = self._pass_rates[example_id]["pass_rate"]
        return 1.0 - pass_rate

    def sample(
        self,
        batch_size: int,
        current_step: int = 0,
    ) -> List[int]:
        """
        Sample examples based on curriculum strategy.

        Args:
            batch_size: Number of examples to sample
            current_step: Current training step (for scheduling)

        Returns:
            List of sampled example indices
        """
        self._current_step = current_step

        # During warmup, sample uniformly
        if current_step < self.warmup_steps:
            indices = self.rng.choice(self.dataset_size, size=batch_size, replace=True)
            return indices.tolist()

        # Apply curriculum strategy
        if self.strategy == "easy_first":
            return self._sample_easy_first(batch_size, current_step)
        elif self.strategy == "adaptive":
            return self._sample_adaptive(batch_size, current_step)
        elif self.strategy == "mixed":
            return self._sample_mixed(batch_size, current_step)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _sample_easy_first(self, batch_size: int, current_step: int) -> List[int]:
        """
        Easy-first strategy: sample easier examples early, harder later.

        Gradually increases difficulty threshold as training progresses.
        """
        # Compute progress [0, 1] - how far through training we are
        max_steps = self.warmup_steps * 10  # Assume ~10x warmup length for full training
        progress = min((current_step - self.warmup_steps) / max_steps, 1.0)

        # Difficulty threshold increases with progress
        # Early: sample examples with difficulty < 0.3
        # Late: sample examples with difficulty < 0.95
        difficulty_threshold = 0.3 + progress * 0.65

        # Get indices and their difficulties
        difficulties = np.array([self.difficulty_score(i) for i in range(self.dataset_size)])

        # Sample from below threshold (easier examples)
        easy_indices = np.where(difficulties <= difficulty_threshold)[0]

        if len(easy_indices) > batch_size // 2:
            # Enough easy examples - sample mostly easy, some random
            num_easy = int(batch_size * 0.8)
            num_random = batch_size - num_easy

            sampled = self.rng.choice(easy_indices, size=num_easy, replace=True).tolist()
            if num_random > 0:
                sampled.extend(self.rng.choice(self.dataset_size, size=num_random, replace=True).tolist())
            self.rng.shuffle(sampled)
            return sampled
        elif len(easy_indices) > 0:
            # Some easy examples available
            sampled = self.rng.choice(easy_indices, size=batch_size, replace=True)
            return sampled.tolist()
        else:
            # Fallback: sample uniformly if no easy examples
            sampled = self.rng.choice(self.dataset_size, size=batch_size, replace=True)
            return sampled.tolist()

    def _sample_adaptive(self, batch_size: int, current_step: int) -> List[int]:
        """
        Adaptive UCB-style strategy: balance exploitation (easy) vs exploration (hard).

        Uses upper confidence bound to select examples that are either:
        - Very easy (high exploitation)
        - High variance/uncertainty (exploration)
        """
        # Compute UCB scores for each example
        ucb_scores = np.zeros(self.dataset_size)

        for i in range(self.dataset_size):
            stats = self._pass_rates[i]
            total = max(stats["total"], 1)  # Avoid division by zero
            pass_rate = stats["pass_rate"]

            # UCB = -difficulty + confidence_bonus
            # Exploit easy examples, explore uncertain examples
            difficulty = self.difficulty_score(i)

            # For untrained examples (total == 0), use high exploration bonus
            if total == 1:  # Only initialized via set_pass_rate
                exploration_bonus = 1.0
            else:
                exploration_bonus = np.sqrt(2 * np.log(self._current_step + 1) / (total + 1))

            ucb_scores[i] = -difficulty + 0.3 * exploration_bonus

        # Normalize scores to probabilities - ensure all have positive weight
        min_score = ucb_scores.min()
        # Shift all scores to be positive
        shifted_scores = ucb_scores - min_score + 0.1
        probabilities = shifted_scores / shifted_scores.sum()

        # Sample based on UCB scores
        sampled = self.rng.choice(
            self.dataset_size,
            size=batch_size,
            replace=True,
            p=probabilities
        )

        return sampled.tolist()

    def _sample_mixed(self, batch_size: int, current_step: int) -> List[int]:
        """
        Mixed strategy: 70% current difficulty level, 30% random.

        Maintains focus on current difficulty while exploring other examples.
        """
        # Compute average difficulty
        difficulties = np.array([self.difficulty_score(i) for i in range(self.dataset_size)])
        avg_difficulty = difficulties.mean()

        # Find examples near current difficulty (within +/- 0.15)
        difficulty_tolerance = 0.15
        near_difficulty = np.where(
            np.abs(difficulties - avg_difficulty) <= difficulty_tolerance
        )[0]

        # Decide how many to sample from each source
        num_focused = int(batch_size * 0.7)
        num_random = batch_size - num_focused

        sampled = []

        # 70% from current difficulty
        if len(near_difficulty) > 0:
            focused = self.rng.choice(near_difficulty, size=num_focused, replace=True)
            sampled.extend(focused)
        else:
            # Fallback: sample uniformly
            fallback = self.rng.choice(self.dataset_size, size=num_focused, replace=True)
            sampled.extend(fallback)

        # 30% random
        random_samples = self.rng.choice(self.dataset_size, size=num_random, replace=True)
        sampled.extend(random_samples)

        # Shuffle to mix sources
        self.rng.shuffle(sampled)

        return sampled[:batch_size]

    def update_pass_rate(self, example_id: int, passed: bool):
        """
        Update pass rate for an example.

        Args:
            example_id: Index of example
            passed: Whether the example was passed/succeeded
        """
        if example_id not in self._pass_rates:
            return

        stats = self._pass_rates[example_id]
        stats["total"] += 1
        if passed:
            stats["passed"] += 1

        # Update pass rate as running average
        stats["pass_rate"] = stats["passed"] / stats["total"]

    def get_curriculum_progress(self) -> CurriculumStats:
        """
        Get curriculum progress statistics.

        Returns:
            CurriculumStats object with current metrics
        """
        # Compute statistics from pass rates
        difficulties = []
        pass_rates = []

        for i in range(self.dataset_size):
            stats = self._pass_rates[i]
            if stats["total"] > 0:  # Only include examples with data
                difficulties.append(self.difficulty_score(i))
                pass_rates.append(stats["pass_rate"])

        # Update stats
        if difficulties:
            self._stats.avg_difficulty = float(np.mean(difficulties))
            self._stats.min_difficulty = float(np.min(difficulties))
            self._stats.max_difficulty = float(np.max(difficulties))
            self._stats.examples_with_data = len(difficulties)

        if pass_rates:
            self._stats.avg_pass_rate = float(np.mean(pass_rates))

        self._stats.steps_completed = self._current_step

        # Count total passes
        total_passed = sum(s["passed"] for s in self._pass_rates.values())
        total_attempts = sum(s["total"] for s in self._pass_rates.values())
        self._stats.num_passed = total_passed
        self._stats.num_sampled = total_attempts

        return self._stats

    def get_difficulty_distribution(self) -> Dict[str, Any]:
        """
        Get histogram of difficulty distribution.

        Returns:
            Dictionary with difficulty bin statistics
        """
        difficulties = np.array([
            self.difficulty_score(i) for i in range(self.dataset_size)
        ])

        # Create bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        hist, bin_edges = np.histogram(difficulties, bins=bins)

        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(difficulties.mean()),
            "std": float(difficulties.std()),
            "min": float(difficulties.min()),
            "max": float(difficulties.max()),
        }

    def get_example_stats(self, example_id: int) -> Dict[str, Any]:
        """
        Get statistics for a specific example.

        Args:
            example_id: Index of example

        Returns:
            Dictionary with example statistics
        """
        if example_id not in self._pass_rates:
            return {}

        stats = self._pass_rates[example_id]
        return {
            "example_id": example_id,
            "total_attempts": stats["total"],
            "total_passed": stats["passed"],
            "pass_rate": stats["pass_rate"],
            "difficulty_score": self.difficulty_score(example_id),
        }

    def set_pass_rate(self, example_id: int, pass_rate: float):
        """
        Manually set pass rate for an example (for testing or initialization).

        Args:
            example_id: Index of example
            pass_rate: Pass rate value in [0, 1]
        """
        if example_id >= self.dataset_size:
            return

        if not 0 <= pass_rate <= 1:
            raise ValueError(f"Pass rate must be in [0, 1], got {pass_rate}")

        self._pass_rates[example_id]["pass_rate"] = pass_rate
        self._pass_rates[example_id]["total"] = 1  # Mark as initialized
