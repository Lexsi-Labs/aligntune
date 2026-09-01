"""
Example: Per-Component Reward Tracking and Visualization

This example demonstrates how to:
1. Use CompositeReward.batch_compute() with return_breakdown=True to get per-component rewards
2. Log reward breakdowns using UnifiedLogger.log_reward_breakdown()
3. Visualize reward trajectories and correlations
4. Analyze reward component interactions during training

Per-reward tracking and visualization utilities for AlignTune.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from aligntune.rewards.core import CompositeReward, RewardConfig, RewardType, RewardFunction
from aligntune.core.rl.logging_utils import UnifiedLogger
from aligntune.core.rl.config import LoggingConfig

logger = logging.getLogger(__name__)


class SimpleTextReward(RewardFunction):
    """Simple example reward based on text length."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return normalized length score."""
        if not text:
            return 0.0
        # Reward longer texts (up to a limit)
        length_score = min(1.0, len(text) / 200.0)
        return length_score

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Compute for batch."""
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


class SimpleQualityReward(RewardFunction):
    """Simple example reward based on text quality metrics."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return a mock quality score."""
        if not text:
            return 0.0

        # Simple heuristics: prefer texts with proper punctuation and reasonable length
        score = 0.5

        if "." in text or "!" in text or "?" in text:
            score += 0.2

        if len(text) > 50:
            score += 0.1

        if len(text.split()) > 10:
            score += 0.1

        return min(1.0, score)

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Compute for batch."""
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


class SimpleSafetyReward(RewardFunction):
    """Simple example safety reward."""

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return a mock safety score."""
        # In a real scenario, this would use a safety model
        # For this example, we check for common safe patterns
        unsafe_words = ["hate", "violence", "illegal"]

        text_lower = text.lower()
        if any(word in text_lower for word in unsafe_words):
            return 0.3

        return 0.9

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Compute for batch."""
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


def demonstrate_reward_breakdown():
    """Demonstrate per-component reward tracking."""

    print("\n" + "=" * 80)
    print("Per-Component Reward Tracking Example")
    print("=" * 80)

    # Setup: Create reward functions
    print("\n1. Setting up reward functions...")

    # Create configs for each reward type
    length_config = RewardConfig(reward_type=RewardType.LENGTH, weight=0.5)
    quality_config = RewardConfig(reward_type=RewardType.COHERENCE, weight=1.0)
    safety_config = RewardConfig(reward_type=RewardType.SAFETY, weight=1.5)

    # Create reward instances
    length_reward = SimpleTextReward(length_config)
    quality_reward = SimpleQualityReward(quality_config)
    safety_reward = SimpleSafetyReward(safety_config)

    # Combine into composite reward
    composite_reward = CompositeReward(
        reward_functions=[length_reward, quality_reward, safety_reward],
        weights=[0.5, 1.0, 1.5],  # Different weights for each component
    )

    print(f"  - Created composite reward with {len(composite_reward.reward_functions)} components")
    print(f"    - Length reward (weight: 0.5)")
    print(f"    - Quality reward (weight: 1.0)")
    print(f"    - Safety reward (weight: 1.5)")

    # Example texts
    texts = [
        "This is a great example of helpful text with proper punctuation.",
        "Short text",
        "This is a longer piece of text that should score well on multiple dimensions because it is well-formed and detailed.",
        "Text without proper ending",
        "This is a wonderful response that provides helpful information clearly and concisely!",
    ]

    print(f"\n2. Computing rewards for {len(texts)} texts...")
    print("-" * 80)

    # Compute rewards WITHOUT breakdown (backward compatible)
    print("\n  Standard mode (backward compatible):")
    composite_scores = composite_reward.batch_compute(texts)
    for i, (text, score) in enumerate(zip(texts, composite_scores)):
        print(f"    Text {i+1}: {score:.4f} | '{text[:40]}...'")

    # Compute rewards WITH breakdown (NEW FEATURE)
    print("\n  With per-component breakdown (NEW):")
    composite_scores_new, breakdowns = composite_reward.batch_compute(texts, return_breakdown=True)

    for i, (text, breakdown) in enumerate(zip(texts, breakdowns)):
        print(f"\n    Text {i+1}: '{text[:40]}...'")
        total = 0.0
        for reward_name, value in sorted(breakdown.items()):
            print(f"      {reward_name:12}: {value:.4f}")
            total += value
        weighted_sum = sum(
            value * weight
            for (reward_name, value), weight in zip(
                sorted(breakdown.items()),
                composite_reward.weights,
            )
        )
        print(f"      weighted sum: {composite_scores_new[i]:.4f}")

    # Setup logging and log reward breakdown
    print("\n3. Logging reward breakdowns with UnifiedLogger...")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_config = LoggingConfig(
            loggers=["tensorboard"],
            output_dir=tmpdir,
            run_name="reward_tracking_demo",
        )

        unified_logger = UnifiedLogger(log_config, backend=None)

        print(f"  Created logger with output directory: {tmpdir}")

        # Log breakdown for each sample
        for step, breakdown in enumerate(breakdowns):
            unified_logger.log_reward_breakdown(step=step, breakdown_dict=breakdown)

        # Save metrics
        unified_logger.save_metrics_history(tmpdir)
        print(f"  Saved metrics history with {len(unified_logger.metrics_history)} entries")

        # Show metrics history
        print("\n  Metrics history entries:")
        for entry in unified_logger.metrics_history:
            step = entry.get("step", "?")
            rewards = {k: v for k, v in entry.items() if "rewards/" in k}
            if rewards:
                print(f"    Step {step}: {rewards}")

        # Try to visualize (if matplotlib is available)
        try:
            from aligntune.eval.reward_viz import save_reward_visualization

            print("\n4. Generating visualization...")
            print("-" * 80)

            output_path = Path(tmpdir) / "reward_visualization.png"
            saved_path = save_reward_visualization(tmpdir, str(output_path))

            if saved_path and saved_path.exists():
                print(f"  Visualization saved to: {saved_path}")
                print(f"  File size: {saved_path.stat().st_size / 1024:.1f} KB")
            else:
                print("  Visualization generation skipped (matplotlib may not be available)")

        except ImportError:
            print("\n4. Visualization generation skipped (matplotlib not installed)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("- Use return_breakdown=True to get per-component rewards")
    print("- Log breakdowns with UnifiedLogger.log_reward_breakdown()")
    print("- Visualize trajectories and correlations with reward_viz module")
    print("- Component weights affect the composite score calculation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    demonstrate_reward_breakdown()
