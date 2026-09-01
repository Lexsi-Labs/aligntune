"""
Tests for per-reward tracking and visualization.

This module tests:
- CompositeReward.batch_compute() with return_breakdown=True
- Reward breakdown logging with UnifiedLogger.log_reward_breakdown()
- Reward visualization from logged metrics
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import reward and logging components
from aligntune.rewards.core import CompositeReward, RewardConfig, RewardType, RewardFunction
from aligntune.core.rl.logging_utils import UnifiedLogger
from aligntune.core.rl.config import LoggingConfig


class MockRewardFunction(RewardFunction):
    """Mock reward function for testing."""

    def __init__(self, config: RewardConfig, value: float = 0.5):
        super().__init__(config)
        self.value = value

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return a fixed mock value."""
        return self.value

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Return mock values for batch."""
        return [self.value] * len(texts)


@pytest.fixture
def simple_composite_reward():
    """Create a simple composite reward with 2 components."""
    config1 = RewardConfig(reward_type=RewardType.SENTIMENT, weight=1.0)
    config2 = RewardConfig(reward_type=RewardType.SAFETY, weight=1.0)

    reward1 = MockRewardFunction(config1, value=0.7)
    reward2 = MockRewardFunction(config2, value=0.5)

    return CompositeReward([reward1, reward2], weights=[1.0, 1.0])


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logging.

    ignore_cleanup_errors=True because TensorBoard's SummaryWriter flushes
    event files from a background thread, which can still be mid-write when
    this fixture tears down, intermittently leaving files behind that make
    plain TemporaryDirectory cleanup raise "Directory not empty".
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


class TestCompositeRewardBreakdown:
    """Tests for CompositeReward breakdown functionality."""

    def test_batch_compute_without_breakdown(self, simple_composite_reward):
        """Test batch_compute returns only scalar rewards by default."""
        texts = ["Hello world", "Test sentence", "Another text"]
        result = simple_composite_reward.batch_compute(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, float) for r in result)
        # Should be weighted average of 0.7 and 0.5 -> 0.6
        assert all(abs(r - 0.6) < 0.01 for r in result)

    def test_batch_compute_with_breakdown(self, simple_composite_reward):
        """Test batch_compute returns breakdown dict when requested."""
        texts = ["Hello world", "Test sentence", "Another text"]
        rewards, breakdowns = simple_composite_reward.batch_compute(
            texts, return_breakdown=True
        )

        assert isinstance(rewards, list)
        assert isinstance(breakdowns, list)
        assert len(rewards) == 3
        assert len(breakdowns) == 3

        # Check breakdown structure
        for breakdown in breakdowns:
            assert isinstance(breakdown, dict)
            # Should have keys for sentiment and safety
            reward_types = set(breakdown.keys())
            assert len(reward_types) >= 2

            # Values should be floats
            for value in breakdown.values():
                assert isinstance(value, float)
                assert 0 <= value <= 1 or -1 <= value <= 1  # Reasonable bounds

    def test_breakdown_dict_keys(self, simple_composite_reward):
        """Test that breakdown dict uses registered reward names."""
        text = "Test"
        _, breakdowns = simple_composite_reward.batch_compute(
            [text], return_breakdown=True
        )

        breakdown = breakdowns[0]

        # Keys should be the reward type values (strings like 'sentiment', 'safety')
        for key in breakdown.keys():
            assert isinstance(key, str)
            assert len(key) > 0
            # Should be a valid enum value
            assert any(rt.value == key for rt in RewardType)

    def test_backward_compatibility(self, simple_composite_reward):
        """Test that return_breakdown=False is default (backward compatible)."""
        texts = ["Test text"]

        # Old-style call (no return_breakdown parameter)
        result = simple_composite_reward.batch_compute(texts)

        # Should return List[float], not tuple
        assert isinstance(result, list)
        assert not isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_breakdown_with_references(self, simple_composite_reward):
        """Test breakdown with reference texts."""
        texts = ["Generated text"]
        references = ["Reference text"]

        rewards, breakdowns = simple_composite_reward.batch_compute(
            texts, references=references, return_breakdown=True
        )

        assert len(rewards) == 1
        assert len(breakdowns) == 1
        assert isinstance(breakdowns[0], dict)
        assert len(breakdowns[0]) >= 2


class TestUnifiedLoggerRewardBreakdown:
    """Tests for UnifiedLogger reward breakdown logging."""

    def test_log_reward_breakdown(self, temp_log_dir):
        """Test logging reward breakdown to metrics history."""
        config = LoggingConfig(
            loggers=["tensorboard"],
            output_dir=str(temp_log_dir),
            run_name="test_run"
        )
        logger = UnifiedLogger(config, backend=None)

        breakdown = {
            "sentiment": 0.7,
            "safety": 0.5,
            "coherence": 0.8
        }

        logger.log_reward_breakdown(step=1, breakdown_dict=breakdown)

        # Check metrics history
        assert len(logger.metrics_history) > 0
        latest_entry = logger.metrics_history[-1]
        assert latest_entry["step"] == 1

        # Check that rewards are prefixed correctly
        assert "rewards/sentiment" in latest_entry
        assert "rewards/safety" in latest_entry
        assert "rewards/coherence" in latest_entry

        # Check values
        assert latest_entry["rewards/sentiment"] == 0.7
        assert latest_entry["rewards/safety"] == 0.5
        assert latest_entry["rewards/coherence"] == 0.8

    def test_log_reward_breakdown_multiple_steps(self, temp_log_dir):
        """Test logging reward breakdown across multiple steps."""
        config = LoggingConfig(
            loggers=["tensorboard"],
            output_dir=str(temp_log_dir),
            run_name="test_run"
        )
        logger = UnifiedLogger(config, backend=None)

        for step in range(3):
            breakdown = {
                "sentiment": 0.5 + step * 0.1,
                "safety": 0.7 - step * 0.05,
            }
            logger.log_reward_breakdown(step=step, breakdown_dict=breakdown)

        # Check history
        reward_entries = [e for e in logger.metrics_history if "rewards/" in str(e)]
        assert len(reward_entries) >= 3

    def test_save_metrics_history_includes_rewards(self, temp_log_dir):
        """Test that metrics history file includes reward breakdowns."""
        config = LoggingConfig(
            loggers=["tensorboard"],
            output_dir=str(temp_log_dir),
            run_name="test_run"
        )
        logger = UnifiedLogger(config, backend=None)

        breakdown = {
            "sentiment": 0.7,
            "safety": 0.5,
        }
        logger.log_reward_breakdown(step=1, breakdown_dict=breakdown)

        # Save metrics
        logger.save_metrics_history(temp_log_dir)

        # Load and verify
        metrics_file = temp_log_dir / "metrics_history.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            history = json.load(f)

        # Should have at least one entry with reward breakdowns
        assert any("rewards/" in str(entry) for entry in history)


class TestRewardVisualization:
    """Tests for reward visualization."""

    def test_extract_reward_metrics(self, temp_log_dir):
        """Test extracting reward metrics from logged data."""
        from aligntune.eval.reward_viz import extract_reward_metrics

        # Create sample metrics
        metrics = {
            "rewards/sentiment": [(1, 0.5), (2, 0.6), (3, 0.7)],
            "rewards/safety": [(1, 0.8), (2, 0.75), (3, 0.7)],
            "loss": [(1, 0.5), (2, 0.4), (3, 0.3)],
            "learning_rate": [(1, 1e-4), (2, 1e-4), (3, 1e-5)],
        }

        reward_metrics = extract_reward_metrics(metrics)

        # Should only have reward metrics
        assert "sentiment" in reward_metrics
        assert "safety" in reward_metrics
        assert "loss" not in reward_metrics
        assert "learning_rate" not in reward_metrics

        # Values should be preserved
        assert reward_metrics["sentiment"] == metrics["rewards/sentiment"]
        assert reward_metrics["safety"] == metrics["rewards/safety"]

    def test_plot_reward_trajectory_with_json(self, temp_log_dir):
        """Test plotting from metrics.json file."""
        from aligntune.eval.reward_viz import save_reward_visualization

        # Create metrics history
        history = [
            {
                "step": i,
                "timestamp": i * 10,
                "rewards/sentiment": 0.5 + i * 0.05,
                "rewards/safety": 0.8 - i * 0.03,
            }
            for i in range(5)
        ]

        metrics_file = temp_log_dir / "metrics_history.json"
        with open(metrics_file, "w") as f:
            json.dump(history, f)

        # Try to generate visualization
        try:
            result = save_reward_visualization(str(temp_log_dir))
            # Should succeed if matplotlib is available
            if result:
                assert result.exists()
                assert str(result).endswith(".png")
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_metrics_json_structure(self, temp_log_dir):
        """Test reading metrics from JSON with proper structure."""
        from aligntune.eval.reward_viz import read_metrics_json

        # Create test metrics
        history = [
            {
                "step": 0,
                "timestamp": 0.0,
                "rewards/sentiment": 0.5,
                "rewards/safety": 0.8,
                "loss": 2.0,
            },
            {
                "step": 1,
                "timestamp": 10.0,
                "rewards/sentiment": 0.55,
                "rewards/safety": 0.75,
                "loss": 1.9,
            },
        ]

        metrics_file = temp_log_dir / "metrics_history.json"
        with open(metrics_file, "w") as f:
            json.dump(history, f)

        # Read metrics
        metrics = read_metrics_json(temp_log_dir)

        # Should have all numeric entries
        assert "rewards/sentiment" in metrics
        assert "rewards/safety" in metrics
        assert "loss" in metrics

        # Check structure
        assert len(metrics["rewards/sentiment"]) == 2
        assert metrics["rewards/sentiment"][0] == (0, 0.5)
        assert metrics["rewards/sentiment"][1] == (1, 0.55)


class TestIntegrationRewardTracking:
    """Integration tests for full reward tracking pipeline."""

    def test_full_pipeline(self, temp_log_dir):
        """Test end-to-end: reward computation -> logging -> visualization."""
        # Setup
        config1 = RewardConfig(
            reward_type=RewardType.SENTIMENT,
            weight=1.0
        )
        config2 = RewardConfig(
            reward_type=RewardType.SAFETY,
            weight=1.0
        )

        reward1 = MockRewardFunction(config1, value=0.7)
        reward2 = MockRewardFunction(config2, value=0.5)
        composite = CompositeReward([reward1, reward2])

        # Compute rewards with breakdown
        texts = ["Test 1", "Test 2", "Test 3"]
        rewards, breakdowns = composite.batch_compute(texts, return_breakdown=True)

        # Log via UnifiedLogger
        log_config = LoggingConfig(
            loggers=["tensorboard"],
            output_dir=str(temp_log_dir),
            run_name="test_run"
        )
        logger = UnifiedLogger(log_config, backend=None)

        for step, (reward, breakdown) in enumerate(zip(rewards, breakdowns)):
            logger.log_reward_breakdown(step=step, breakdown_dict=breakdown)

        # Save metrics
        logger.save_metrics_history(temp_log_dir)

        # Verify metrics file
        metrics_file = temp_log_dir / "metrics_history.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            history = json.load(f)

        # Should have reward entries
        reward_entries = [e for e in history if any("rewards/" in str(k) for k in e.keys())]
        assert len(reward_entries) >= len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
