"""
Tests for RubricReward (Rubric-Anchored RL).

All tests use mocks to avoid real LLM API calls.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
import numpy as np

from aligntune.rewards.rubric_reward import RubricReward
from aligntune.rewards.core import RewardConfig, RewardType
from aligntune.eval.llm_judge import LLMJudge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_judge():
    """Create a mock LLMJudge that returns deterministic scores."""
    judge = MagicMock(spec=LLMJudge)
    judge._cache = {}

    def score_side_effect(prompt, response, rubric):
        # Deterministic scoring based on response length
        # Longer responses get higher scores
        if not response:
            return 0.0
        return min(1.0, len(response) / 100.0)

    judge.score.side_effect = score_side_effect
    return judge


@pytest.fixture
def sample_rubric():
    """Sample rubric for testing."""
    return "Is this response helpful? Score 1.0 = very helpful, 0.0 = not helpful"


@pytest.fixture
def rubric_reward(mock_judge, sample_rubric):
    """Create a RubricReward instance for testing."""
    return RubricReward(rubric=sample_rubric, judge=mock_judge)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestRubricRewardInit:
    """Test RubricReward initialization."""

    def test_init_with_valid_judge_and_rubric(self, mock_judge, sample_rubric):
        """Test initialization with valid judge and rubric."""
        reward = RubricReward(rubric=sample_rubric, judge=mock_judge)
        assert reward.rubric == sample_rubric
        assert reward.judge == mock_judge

    def test_init_fails_with_none_judge(self, sample_rubric):
        """Test that initialization fails with None judge."""
        with pytest.raises(ValueError, match="Judge cannot be None"):
            RubricReward(rubric=sample_rubric, judge=None)

    def test_init_fails_with_non_llmjudge(self, sample_rubric):
        """Test that initialization fails with non-LLMJudge judge."""
        fake_judge = "not a judge"
        with pytest.raises(TypeError, match="Judge must be an LLMJudge instance"):
            RubricReward(rubric=sample_rubric, judge=fake_judge)

    def test_init_fails_with_empty_rubric(self, mock_judge):
        """Test that initialization fails with empty rubric."""
        with pytest.raises(ValueError, match="Rubric must be a non-empty string"):
            RubricReward(rubric="", judge=mock_judge)

    def test_init_fails_with_none_rubric(self, mock_judge):
        """Test that initialization fails with None rubric."""
        with pytest.raises(ValueError, match="Rubric must be a non-empty string"):
            RubricReward(rubric=None, judge=mock_judge)

    def test_init_with_custom_cache_size(self, mock_judge, sample_rubric):
        """Test initialization with custom cache size."""
        reward = RubricReward(rubric=sample_rubric, judge=mock_judge, cache_size=500)
        assert reward._cache_size == 500

    def test_init_creates_minimal_config(self, mock_judge, sample_rubric):
        """Test that init creates a minimal RewardConfig."""
        reward = RubricReward(rubric=sample_rubric, judge=mock_judge)
        assert isinstance(reward.config, RewardConfig)
        assert reward.config.weight == 1.0


# ---------------------------------------------------------------------------
# Single Compute Tests
# ---------------------------------------------------------------------------


class TestRubricRewardCompute:
    """Test RubricReward.compute() for single completions."""

    def test_compute_returns_float(self, rubric_reward):
        """Test that compute returns a float score."""
        score = rubric_reward.compute("This is a helpful response.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_compute_with_valid_completion(self, rubric_reward):
        """Test computing score for a valid completion."""
        completion = "This is a helpful response with substantial content."
        score = rubric_reward.compute(completion)
        assert 0.0 <= score <= 1.0
        # Score should reflect completion length (mocked behavior)
        assert score > 0.0

    def test_compute_with_short_completion(self, rubric_reward):
        """Test computing score for a short completion."""
        score = rubric_reward.compute("Hi")
        assert 0.0 <= score <= 1.0

    def test_compute_with_empty_completion(self, rubric_reward):
        """Test computing score for an empty completion."""
        score = rubric_reward.compute("")
        assert score == 0.0

    def test_compute_with_prompt(self, rubric_reward):
        """Test compute with provided prompt."""
        completion = "This is a response."
        prompt = "What is AI?"
        score = rubric_reward.compute(completion, prompt=prompt)
        assert 0.0 <= score <= 1.0

    def test_compute_calls_judge_score(self, mock_judge, rubric_reward):
        """Test that compute calls judge.score()."""
        completion = "Test completion"
        rubric_reward.compute(completion)
        mock_judge.score.assert_called()

    def test_compute_uses_provided_prompt(self, mock_judge, sample_rubric):
        """Test that compute uses the provided prompt."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.return_value = 0.8
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        prompt = "Custom prompt"
        completion = "Response"
        reward.compute(completion, prompt=prompt)

        # Verify judge was called with the provided prompt
        judge.score.assert_called_once()
        args = judge.score.call_args[0]
        assert args[0] == prompt  # First arg is prompt

    def test_compute_uses_fallback_prompt_when_not_provided(self, mock_judge, sample_rubric):
        """Test that compute uses fallback prompt when not provided."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.return_value = 0.7
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        completion = "Response"
        reward.compute(completion)

        judge.score.assert_called_once()
        args = judge.score.call_args[0]
        assert "[Original prompt unavailable]" in args[0]

    def test_compute_returns_fallback_on_error(self, sample_rubric):
        """Test that compute returns 0.5 fallback on judge error."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.side_effect = Exception("Judge error")
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        score = reward.compute("Test completion")
        assert score == 0.5


# ---------------------------------------------------------------------------
# Batch Compute Tests
# ---------------------------------------------------------------------------


class TestRubricRewardBatchCompute:
    """Test RubricReward.batch_compute() for multiple completions."""

    def test_batch_compute_returns_list(self, rubric_reward):
        """Test that batch_compute returns a list of floats."""
        completions = ["Response 1", "Response 2", "Response 3"]
        scores = rubric_reward.batch_compute(completions)
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_compute_empty_list(self, rubric_reward):
        """Test batch_compute with empty list."""
        scores = rubric_reward.batch_compute([])
        assert scores == []

    def test_batch_compute_with_numpy_array(self, rubric_reward):
        """Test batch_compute accepts numpy arrays."""
        completions = np.array(["Response 1", "Response 2"])
        scores = rubric_reward.batch_compute(completions)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_batch_compute_with_prompts(self, mock_judge, sample_rubric):
        """Test batch_compute with custom prompts."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.return_value = 0.8
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        completions = ["Response 1", "Response 2"]
        prompts = ["Prompt 1", "Prompt 2"]
        scores = reward.batch_compute(completions, prompts=prompts)

        assert len(scores) == 2
        # Verify judge was called with correct prompts
        assert judge.score.call_count == 2
        calls = [call[0] for call in judge.score.call_args_list]
        assert calls[0][0] == "Prompt 1"
        assert calls[1][0] == "Prompt 2"

    def test_batch_compute_with_mismatched_prompt_count(self, rubric_reward):
        """Test batch_compute handles mismatched prompt count."""
        completions = ["Response 1", "Response 2", "Response 3"]
        prompts = ["Prompt 1"]  # Only one prompt for three completions
        scores = rubric_reward.batch_compute(completions, prompts=prompts)
        assert len(scores) == 3

    def test_batch_compute_all_scores_in_range(self, rubric_reward):
        """Test that all batch scores are in [0, 1]."""
        completions = ["Short", "Medium length response", "This is a very long response with lots of content"]
        scores = rubric_reward.batch_compute(completions)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_compute_preserves_order(self, mock_judge, sample_rubric):
        """Test that batch_compute preserves order of responses."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}

        def score_by_length(prompt, response, rubric):
            return len(response) / 100.0

        judge.score.side_effect = score_by_length
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        completions = ["a", "aa", "aaa"]
        scores = reward.batch_compute(completions)

        # Scores should increase with length
        assert scores[0] < scores[1] < scores[2]


# ---------------------------------------------------------------------------
# Caching Tests
# ---------------------------------------------------------------------------


class TestRubricRewardCaching:
    """Test caching behavior (inherited from judge)."""

    def test_repeated_calls_use_judge_cache(self, mock_judge, sample_rubric):
        """Test that repeated calls with same completion use judge cache."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.return_value = 0.75
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        completion = "Same completion"

        # First call
        reward.compute(completion)
        assert judge.score.call_count == 1

        # Second call (should hit cache)
        reward.compute(completion)
        assert judge.score.call_count == 2  # Mock doesn't cache, but real judge would

    def test_get_cache_info(self, mock_judge, sample_rubric):
        """Test getting cache information."""
        reward = RubricReward(rubric=sample_rubric, judge=mock_judge)
        cache_info = reward.get_cache_info()
        assert isinstance(cache_info, dict)
        assert "cache_size" in cache_info

    def test_clear_cache(self, mock_judge, sample_rubric):
        """Test clearing the judge cache."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {"key1": 0.5, "key2": 0.7}
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        reward.clear_cache()
        assert len(judge._cache) == 0

    def test_clear_cache_on_judge_without_cache(self, sample_rubric):
        """Test clear_cache gracefully handles judges without cache."""
        judge = MagicMock(spec=LLMJudge)
        del judge._cache  # Remove cache attribute
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        # Should not raise an error
        reward.clear_cache()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestRubricRewardIntegration:
    """Integration tests with reward system."""

    def test_reward_config_creation(self, mock_judge, sample_rubric):
        """Test RubricReward with RewardConfig."""
        config = RewardConfig(
            reward_type=RewardType.RUBRIC,
            weight=0.8,
            params={"rubric": sample_rubric}
        )
        reward = RubricReward(rubric=sample_rubric, judge=mock_judge, config=config)
        assert reward.config.weight == 0.8

    def test_inherits_from_reward_function(self, rubric_reward):
        """Test that RubricReward is a RewardFunction."""
        from aligntune.rewards.core import RewardFunction
        assert isinstance(rubric_reward, RewardFunction)

    def test_batch_compute_method_signature(self, rubric_reward):
        """Test that batch_compute has correct signature."""
        # Should accept reference parameter for interface compatibility
        scores = rubric_reward.batch_compute(
            ["Response 1", "Response 2"],
            reference=["Ref 1", "Ref 2"]
        )
        assert len(scores) == 2

    def test_compute_with_kwargs(self, rubric_reward):
        """Test compute accepts arbitrary kwargs."""
        # Should not raise error even with unknown kwargs
        score = rubric_reward.compute("Response", extra_param="value")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestRubricRewardErrorHandling:
    """Test error handling and edge cases."""

    def test_compute_with_invalid_completion_type(self, rubric_reward):
        """Test compute with non-string completion."""
        score = rubric_reward.compute(123)  # Integer instead of string
        assert score == 0.0

    def test_compute_with_very_long_completion(self, rubric_reward):
        """Test compute with very long completion."""
        long_text = "word " * 10000  # Very long text
        score = rubric_reward.compute(long_text)
        assert 0.0 <= score <= 1.0

    def test_batch_compute_with_none_items(self, mock_judge, sample_rubric):
        """Test batch_compute with None items in list."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.return_value = 0.5
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        completions = ["Valid", None, "Another"]
        scores = reward.batch_compute(completions)
        assert len(scores) == 3

    def test_compute_logs_on_error(self, sample_rubric):
        """Test that compute logs errors."""
        judge = MagicMock(spec=LLMJudge)
        judge._cache = {}
        judge.score.side_effect = RuntimeError("Test error")
        reward = RubricReward(rubric=sample_rubric, judge=judge)

        with patch('aligntune.rewards.rubric_reward.logger') as mock_logger:
            score = reward.compute("Test")
            mock_logger.error.assert_called()


# ---------------------------------------------------------------------------
# Rubric Format Tests
# ---------------------------------------------------------------------------


class TestRubricFormat:
    """Test various rubric formats."""

    def test_simple_rubric(self, mock_judge):
        """Test with simple rubric."""
        rubric = "Good or bad?"
        reward = RubricReward(rubric=rubric, judge=mock_judge)
        score = reward.compute("Test")
        assert 0.0 <= score <= 1.0

    def test_detailed_rubric(self, mock_judge):
        """Test with detailed rubric."""
        rubric = """
        Evaluate helpfulness on scale 0-1:
        1.0 = Directly answers question with accurate, detailed info
        0.75 = Answers question but could be more detailed
        0.5 = Partially answers question
        0.25 = Mostly irrelevant
        0.0 = Completely unhelpful
        """
        reward = RubricReward(rubric=rubric, judge=mock_judge)
        score = reward.compute("This is helpful")
        assert 0.0 <= score <= 1.0

    def test_multiline_rubric(self, mock_judge):
        """Test with multi-line rubric."""
        rubric = "Line 1\nLine 2\nLine 3"
        reward = RubricReward(rubric=rubric, judge=mock_judge)
        score = reward.compute("Test")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------


class TestRubricRewardRegistry:
    """Test integration with reward registry."""

    def test_rubric_in_reward_type_enum(self):
        """Test that RUBRIC is in RewardType enum."""
        assert hasattr(RewardType, 'RUBRIC')
        assert RewardType.RUBRIC.value == "rubric"

    def test_rubric_registered_in_registry(self):
        """Test that rubric is registered in RewardRegistry."""
        from aligntune.rewards.registry import RewardRegistry
        rewards_list = RewardRegistry.list_rewards()
        assert "rubric" in rewards_list

    def test_create_rubric_reward_via_factory(self, mock_judge):
        """Test creating RubricReward via factory."""
        from aligntune.rewards.core import RewardFunctionFactory

        config = RewardConfig(
            reward_type=RewardType.RUBRIC,
            params={"judge": mock_judge}
        )

        # Note: Factory create_reward won't work for RUBRIC directly since it needs judge
        # This is expected - RubricReward is special and created differently
        # Verify the type exists
        assert RewardType.RUBRIC in RewardFunctionFactory._reward_classes


# ---------------------------------------------------------------------------
# Numerical Stability Tests
# ---------------------------------------------------------------------------


class TestRubricRewardNumericalStability:
    """Test numerical stability of scoring."""

    def test_scores_are_valid_floats(self, rubric_reward):
        """Test that scores are valid floats (not NaN or Inf)."""
        completions = ["Response 1", "Response 2", "Response 3"]
        scores = rubric_reward.batch_compute(completions)

        for score in scores:
            assert not np.isnan(score)
            assert not np.isinf(score)
            assert isinstance(score, (float, np.floating))

    def test_score_range_is_valid(self, rubric_reward):
        """Test that all scores are in valid range [0, 1]."""
        for _ in range(10):
            score = rubric_reward.compute(f"Test response {_}")
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
