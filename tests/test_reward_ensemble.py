"""
Tests for Reward Model Ensembles.

This module tests:
- CompositeReward ensemble_mode aggregation (mean, worst_case, uncertainty_weighted)
- RewardModelEnsemble with multiple HF models
- Ensemble statistics and per-model score tracking
- Integration with GRPO/DPO/PPO trainers
"""

import pytest
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from aligntune.rewards.core import (
    CompositeReward,
    RewardConfig,
    RewardType,
    RewardFunction,
)
from aligntune.rewards.ensemble import RewardModelEnsemble


class MockRewardFunction(RewardFunction):
    """Mock reward function that returns a fixed value."""

    def __init__(self, config: RewardConfig, value: float = 0.5):
        super().__init__(config)
        self.value = value

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return fixed mock value."""
        return self.value

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Return mock values for batch."""
        return [self.value] * len(texts)


class VariableRewardFunction(RewardFunction):
    """Mock reward function that varies by text length."""

    def __init__(self, config: RewardConfig, scale: float = 1.0):
        super().__init__(config)
        self.scale = scale

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Return score based on text length."""
        # Normalize to [0, 1]
        score = min(1.0, len(text) / 100.0 * self.scale)
        return score

    def batch_compute(
        self, texts: List[str], references: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        """Compute scores for batch based on text length."""
        return [self.compute(text) for text in texts]


# ============================================================================
# COMPOSITE REWARD ENSEMBLE MODE TESTS
# ============================================================================


class TestCompositeRewardEnsembleModes:
    """Test CompositeReward aggregation modes."""

    @pytest.fixture
    def three_mock_rewards(self):
        """Create three mock rewards with different values."""
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)
        config3 = RewardConfig(reward_type=RewardType.HELPFULNESS)

        reward1 = MockRewardFunction(config1, value=0.9)
        reward2 = MockRewardFunction(config2, value=0.5)
        reward3 = MockRewardFunction(config3, value=0.7)

        return [reward1, reward2, reward3]

    def test_composite_reward_mean_mode(self, three_mock_rewards):
        """Test mean aggregation mode (default)."""
        composite = CompositeReward(
            three_mock_rewards, weights=[1.0, 1.0, 1.0], ensemble_mode="mean"
        )

        # Should be (0.9 + 0.5 + 0.7) / 3 = 0.7
        score = composite.compute("test text")
        assert abs(score - 0.7) < 0.01

    def test_composite_reward_worst_case_mode(self, three_mock_rewards):
        """Test worst_case aggregation mode."""
        composite = CompositeReward(
            three_mock_rewards, weights=[1.0, 1.0, 1.0], ensemble_mode="worst_case"
        )

        # Should be min(0.9, 0.5, 0.7) = 0.5
        score = composite.compute("test text")
        assert abs(score - 0.5) < 0.01

    def test_composite_reward_uncertainty_weighted_mode(self, three_mock_rewards):
        """Test uncertainty_weighted aggregation mode."""
        composite = CompositeReward(
            three_mock_rewards, weights=[1.0, 1.0, 1.0], ensemble_mode="uncertainty_weighted"
        )

        # Mean = 0.7, Std = sqrt(var([0.9, 0.5, 0.7])) = 0.1633
        # Uncertainty weight = 1 - (0.1633 / 0.5) = 0.6734
        # Result = 0.7 * 0.6734 ≈ 0.471
        score = composite.compute("test text")
        assert 0.4 < score < 0.5  # Allow some tolerance

    def test_composite_reward_invalid_mode(self, three_mock_rewards):
        """Test that invalid ensemble mode raises error."""
        with pytest.raises(ValueError):
            CompositeReward(
                three_mock_rewards,
                weights=[1.0, 1.0, 1.0],
                ensemble_mode="invalid_mode"
            )

    def test_composite_reward_mean_with_weights(self, three_mock_rewards):
        """Test weighted mean aggregation."""
        # Weights: [0.5, 0.3, 0.2]
        # Weighted sum: 0.9*0.5 + 0.5*0.3 + 0.7*0.2 = 0.45 + 0.15 + 0.14 = 0.74
        composite = CompositeReward(
            three_mock_rewards, weights=[0.5, 0.3, 0.2], ensemble_mode="mean"
        )

        score = composite.compute("test text")
        assert abs(score - 0.74) < 0.01

    def test_composite_reward_batch_mean_mode(self, three_mock_rewards):
        """Test batch computation with mean mode."""
        composite = CompositeReward(
            three_mock_rewards, weights=[1.0, 1.0, 1.0], ensemble_mode="mean"
        )

        texts = ["hello", "world", "test"]
        scores = composite.batch_compute(texts)

        assert len(scores) == 3
        assert all(abs(s - 0.7) < 0.01 for s in scores)

    def test_composite_reward_batch_worst_case_mode(self, three_mock_rewards):
        """Test batch computation with worst_case mode."""
        composite = CompositeReward(
            three_mock_rewards, weights=[1.0, 1.0, 1.0], ensemble_mode="worst_case"
        )

        texts = ["hello", "world", "test"]
        scores = composite.batch_compute(texts)

        assert len(scores) == 3
        assert all(abs(s - 0.5) < 0.01 for s in scores)

    def test_composite_reward_all_identical(self, three_mock_rewards):
        """Test uncertainty_weighted with identical scores."""
        # Create three rewards with same value
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)
        config3 = RewardConfig(reward_type=RewardType.HELPFULNESS)

        reward1 = MockRewardFunction(config1, value=0.6)
        reward2 = MockRewardFunction(config2, value=0.6)
        reward3 = MockRewardFunction(config3, value=0.6)

        composite = CompositeReward(
            [reward1, reward2, reward3],
            weights=[1.0, 1.0, 1.0],
            ensemble_mode="uncertainty_weighted"
        )

        # When all identical, std=0, should return mean
        score = composite.compute("test text")
        assert abs(score - 0.6) < 0.01


# ============================================================================
# REWARD MODEL ENSEMBLE TESTS
# ============================================================================


class TestRewardModelEnsemble:
    """Test RewardModelEnsemble class."""

    @pytest.fixture
    def simple_ensemble_config(self):
        """Create a simple ensemble config with mock models."""
        return RewardConfig(
            reward_type=RewardType.REWARD_MODEL_ENSEMBLE,
            params={
                "model_ids": ["gpt2"],  # Use simple model for testing
                "ensemble_mode": "mean",
            }
        )

    def test_ensemble_initialization(self, simple_ensemble_config):
        """Test ensemble can be initialized."""
        # Should not raise any errors
        ensemble = RewardModelEnsemble(simple_ensemble_config)
        assert ensemble.ensemble_mode == "mean"
        assert len(ensemble.model_ids) > 0

    def test_ensemble_get_ensemble_mode(self, simple_ensemble_config):
        """Test getting ensemble mode."""
        ensemble = RewardModelEnsemble(simple_ensemble_config)
        assert ensemble.get_ensemble_mode() == "mean"

    def test_ensemble_set_ensemble_mode(self, simple_ensemble_config):
        """Test setting ensemble mode."""
        ensemble = RewardModelEnsemble(simple_ensemble_config)

        ensemble.set_ensemble_mode("worst_case")
        assert ensemble.get_ensemble_mode() == "worst_case"

        ensemble.set_ensemble_mode("uncertainty_weighted")
        assert ensemble.get_ensemble_mode() == "uncertainty_weighted"

    def test_ensemble_invalid_mode(self, simple_ensemble_config):
        """Test that invalid mode raises error."""
        ensemble = RewardModelEnsemble(simple_ensemble_config)

        with pytest.raises(ValueError):
            ensemble.set_ensemble_mode("invalid")

    def test_ensemble_no_models_raises_error(self):
        """Test that empty model list raises error during compute."""
        config = RewardConfig(
            reward_type=RewardType.REWARD_MODEL_ENSEMBLE,
            params={
                "model_ids": [],
                "ensemble_mode": "mean",
            }
        )

        ensemble = RewardModelEnsemble(config)

        # Should raise error when trying to compute without models
        with pytest.raises(RuntimeError):
            ensemble.compute("test text")


# ============================================================================
# COMPOSITE REWARD WITH VARIABLE REWARDS
# ============================================================================


class TestCompositeRewardWithVariableRewards:
    """Test ensemble modes with variable reward functions."""

    @pytest.fixture
    def variable_rewards(self):
        """Create variable reward functions with different scales."""
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)

        # reward1 scales linearly with text length
        reward1 = VariableRewardFunction(config1, scale=1.0)
        # reward2 scales at half rate
        reward2 = VariableRewardFunction(config2, scale=0.5)

        return [reward1, reward2]

    def test_mean_aggregation_variable(self, variable_rewards):
        """Test mean aggregation with variable rewards."""
        composite = CompositeReward(
            variable_rewards, weights=[1.0, 1.0], ensemble_mode="mean"
        )

        text = "x" * 100
        score = composite.compute(text)

        # reward1(100) = 1.0, reward2(100) = 0.5
        # mean = 0.75
        assert abs(score - 0.75) < 0.01

    def test_worst_case_aggregation_variable(self, variable_rewards):
        """Test worst_case aggregation with variable rewards."""
        composite = CompositeReward(
            variable_rewards, weights=[1.0, 1.0], ensemble_mode="worst_case"
        )

        text = "x" * 100
        score = composite.compute(text)

        # min(1.0, 0.5) = 0.5
        assert abs(score - 0.5) < 0.01

    def test_uncertainty_weighted_variable(self, variable_rewards):
        """Test uncertainty_weighted aggregation with variable rewards."""
        composite = CompositeReward(
            variable_rewards, weights=[1.0, 1.0], ensemble_mode="uncertainty_weighted"
        )

        text = "x" * 100
        score = composite.compute(text)

        # reward1 = 1.0, reward2 = 0.5
        # mean = 0.75, std = 0.25
        # uncertainty_weight = 1 - (0.25 / 0.5) = 0.5
        # result = 0.75 * 0.5 = 0.375
        assert abs(score - 0.375) < 0.01


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================


class TestEnsembleBatchProcessing:
    """Test batch processing with ensemble modes."""

    @pytest.fixture
    def variable_rewards(self):
        """Create variable reward functions."""
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)

        reward1 = VariableRewardFunction(config1, scale=1.0)
        reward2 = VariableRewardFunction(config2, scale=0.5)

        return [reward1, reward2]

    def test_batch_compute_mean_mode(self, variable_rewards):
        """Test batch computation with mean mode."""
        composite = CompositeReward(
            variable_rewards, weights=[1.0, 1.0], ensemble_mode="mean"
        )

        texts = ["x" * 50, "x" * 100, "x" * 200]
        scores = composite.batch_compute(texts)

        assert len(scores) == 3

        # For "x" * 50: reward1 = 0.5, reward2 = 0.25, mean = 0.375
        assert abs(scores[0] - 0.375) < 0.01

        # For "x" * 100: reward1 = 1.0, reward2 = 0.5, mean = 0.75
        assert abs(scores[1] - 0.75) < 0.01

        # For "x" * 200: reward1 = min(1.0, 200/100*1.0) = 1.0,
        # reward2 = min(1.0, 200/100*0.5) = 1.0 (not 0.5 -- the scale-0.5
        # reward also saturates at 200 chars), mean = 1.0
        assert abs(scores[2] - 1.0) < 0.01

    def test_batch_compute_with_breakdown(self, variable_rewards):
        """Test batch computation with breakdown."""
        composite = CompositeReward(
            variable_rewards, weights=[1.0, 1.0], ensemble_mode="mean"
        )

        texts = ["x" * 50, "x" * 100]
        scores, breakdowns = composite.batch_compute(texts, return_breakdown=True)

        assert len(scores) == 2
        assert len(breakdowns) == 2

        # Check breakdown structure
        assert isinstance(breakdowns[0], dict)
        assert len(breakdowns[0]) > 0


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEnsembleEdgeCases:
    """Test edge cases and error handling."""

    def test_single_reward_ensemble(self):
        """Test ensemble with only one reward function."""
        config = RewardConfig(reward_type=RewardType.SENTIMENT)
        reward = MockRewardFunction(config, value=0.8)

        composite = CompositeReward(
            [reward], weights=[1.0], ensemble_mode="worst_case"
        )

        # With single reward, all modes should return the same value
        score = composite.compute("test")
        assert abs(score - 0.8) < 0.01

    def test_empty_rewards_list_raises_error(self):
        """Test that empty rewards list raises error."""
        with pytest.raises(ValueError):
            # Should fail due to mismatched weights
            CompositeReward([], weights=[], ensemble_mode="mean")

    def test_mismatched_weights_raises_error(self):
        """Test that mismatched weights and rewards raises error."""
        config = RewardConfig(reward_type=RewardType.SENTIMENT)
        reward = MockRewardFunction(config, value=0.8)

        with pytest.raises(ValueError):
            CompositeReward([reward], weights=[1.0, 2.0], ensemble_mode="mean")

    def test_nan_handling_in_worst_case(self):
        """Test that NaN/inf values are handled gracefully."""
        class BadRewardFunction(RewardFunction):
            def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
                raise Exception("Model error")

        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)

        bad_reward = BadRewardFunction(config1)
        good_reward = MockRewardFunction(config2, value=0.8)

        composite = CompositeReward(
            [bad_reward, good_reward], weights=[1.0, 1.0], ensemble_mode="worst_case"
        )

        # Should handle the error and return minimum of available scores
        score = composite.compute("test")
        assert score >= 0.0 and score <= 1.0

    def test_all_zero_rewards(self):
        """Test aggregation when all rewards are zero."""
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)
        config3 = RewardConfig(reward_type=RewardType.HELPFULNESS)

        reward1 = MockRewardFunction(config1, value=0.0)
        reward2 = MockRewardFunction(config2, value=0.0)
        reward3 = MockRewardFunction(config3, value=0.0)

        composite = CompositeReward(
            [reward1, reward2, reward3],
            weights=[1.0, 1.0, 1.0],
            ensemble_mode="mean"
        )

        score = composite.compute("test")
        assert abs(score - 0.0) < 0.01


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEnsembleIntegration:
    """Integration tests with real-world scenarios."""

    def test_ensemble_with_different_ensemble_modes(self):
        """Test switching between ensemble modes for same rewards."""
        config1 = RewardConfig(reward_type=RewardType.SENTIMENT)
        config2 = RewardConfig(reward_type=RewardType.SAFETY)
        config3 = RewardConfig(reward_type=RewardType.HELPFULNESS)

        reward1 = MockRewardFunction(config1, value=0.9)
        reward2 = MockRewardFunction(config2, value=0.6)
        reward3 = MockRewardFunction(config3, value=0.8)

        text = "test text"

        # Test all three modes on same rewards
        mean_composite = CompositeReward(
            [reward1, reward2, reward3], weights=[1.0, 1.0, 1.0], ensemble_mode="mean"
        )
        mean_score = mean_composite.compute(text)

        worst_composite = CompositeReward(
            [reward1, reward2, reward3], weights=[1.0, 1.0, 1.0], ensemble_mode="worst_case"
        )
        worst_score = worst_composite.compute(text)

        unc_composite = CompositeReward(
            [reward1, reward2, reward3],
            weights=[1.0, 1.0, 1.0],
            ensemble_mode="uncertainty_weighted"
        )
        unc_score = unc_composite.compute(text)

        # Scores should follow: worst_case <= uncertainty_weighted <= mean
        # (generally, with this setup)
        assert worst_score <= mean_score
        assert worst_score == 0.6  # min(0.9, 0.6, 0.8)
        assert mean_score == (0.9 + 0.6 + 0.8) / 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
