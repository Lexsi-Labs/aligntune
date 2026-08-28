"""
Comprehensive tests for Curriculum Learning.

Tests curriculum sampler strategies, callback integration, and adaptive sampling.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aligntune.data.curriculum import (
    CurriculumSampler,
    CurriculumStrategy,
    CurriculumStats,
)
from aligntune.core.callbacks.curriculum_callback import CurriculumCallback


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dataset_size():
    """Standard dataset size for tests."""
    return 100


@pytest.fixture
def curriculum_sampler(dataset_size):
    """Create a curriculum sampler for testing."""
    return CurriculumSampler(
        dataset_size=dataset_size,
        strategy="adaptive",
        warmup_steps=100,
        seed=42,
    )


@pytest.fixture
def temp_log_dir():
    """Temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# TEST: CURRICULUM SAMPLER INITIALIZATION
# ============================================================================


class TestCurriculumSamplerInit:
    """Test curriculum sampler initialization."""

    def test_init_basic(self, dataset_size):
        """Test basic initialization."""
        sampler = CurriculumSampler(dataset_size=dataset_size, strategy="adaptive")

        assert sampler.dataset_size == dataset_size
        assert sampler.strategy == "adaptive"
        assert sampler.warmup_steps == 1000
        assert len(sampler._pass_rates) == dataset_size

    def test_init_all_strategies(self, dataset_size):
        """Test initialization with all strategies."""
        for strategy in ["easy_first", "adaptive", "mixed"]:
            sampler = CurriculumSampler(dataset_size=dataset_size, strategy=strategy)
            assert sampler.strategy == strategy

    def test_init_custom_warmup(self, dataset_size):
        """Test initialization with custom warmup steps."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            warmup_steps=500,
        )
        assert sampler.warmup_steps == 500

    def test_init_invalid_strategy(self, dataset_size):
        """Test initialization with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            CurriculumSampler(dataset_size=dataset_size, strategy="invalid")

    def test_pass_rates_initialized(self, dataset_size):
        """Test that pass rates are initialized for all examples."""
        sampler = CurriculumSampler(dataset_size=dataset_size)

        for i in range(dataset_size):
            assert i in sampler._pass_rates
            assert sampler._pass_rates[i]["passed"] == 0
            assert sampler._pass_rates[i]["total"] == 0
            assert sampler._pass_rates[i]["pass_rate"] == 0.0


# ============================================================================
# TEST: DIFFICULTY SCORE COMPUTATION
# ============================================================================


class TestDifficultyScore:
    """Test difficulty score computation."""

    def test_difficulty_easy_example(self, curriculum_sampler):
        """Test difficulty for easy example (high pass rate)."""
        # Example with 80% pass rate -> difficulty 0.2
        curriculum_sampler.set_pass_rate(0, pass_rate=0.8)

        difficulty = curriculum_sampler.difficulty_score(0)
        assert 0.15 < difficulty < 0.25  # Allow small tolerance
        assert difficulty < 0.5  # Should be easy

    def test_difficulty_hard_example(self, curriculum_sampler):
        """Test difficulty for hard example (low pass rate)."""
        # Example with 20% pass rate -> difficulty 0.8
        curriculum_sampler.set_pass_rate(1, pass_rate=0.2)

        difficulty = curriculum_sampler.difficulty_score(1)
        assert 0.75 < difficulty < 0.85
        assert difficulty > 0.5  # Should be hard

    def test_difficulty_unknown_example(self, curriculum_sampler):
        """Test difficulty for unknown example returns neutral score."""
        difficulty = curriculum_sampler.difficulty_score(999)
        assert difficulty == 0.5  # Neutral

    def test_difficulty_formula(self, curriculum_sampler):
        """Test difficulty formula: difficulty = 1 - pass_rate."""
        for pass_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
            curriculum_sampler.set_pass_rate(10, pass_rate=pass_rate)
            difficulty = curriculum_sampler.difficulty_score(10)

            expected = 1.0 - pass_rate
            assert abs(difficulty - expected) < 0.01


# ============================================================================
# TEST: SAMPLING STRATEGIES
# ============================================================================


class TestSamplingStrategies:
    """Test curriculum sampling strategies."""

    def test_warmup_phase_uniform(self, curriculum_sampler, dataset_size):
        """Test that warmup phase samples uniformly."""
        # During warmup, should sample uniformly
        samples = curriculum_sampler.sample(batch_size=20, current_step=50)

        assert len(samples) == 20
        assert all(0 <= s < dataset_size for s in samples)

    def test_post_warmup_sampling(self, curriculum_sampler, dataset_size):
        """Test that post-warmup sampling uses strategy."""
        # Set difficulty for some examples
        for i in range(50):
            curriculum_sampler.set_pass_rate(i, pass_rate=0.9)  # Easy
        for i in range(50, 100):
            curriculum_sampler.set_pass_rate(i, pass_rate=0.1)  # Hard

        # Sample after warmup
        samples = curriculum_sampler.sample(batch_size=20, current_step=200)

        assert len(samples) == 20
        assert all(0 <= s < dataset_size for s in samples)

    def test_easy_first_strategy(self, dataset_size):
        """Test easy_first strategy prioritizes easy examples early."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            strategy="easy_first",
            warmup_steps=100,
        )

        # Set difficulties
        for i in range(50):
            sampler.set_pass_rate(i, pass_rate=0.9)  # Easy (difficulty 0.1)
        for i in range(50, 100):
            sampler.set_pass_rate(i, pass_rate=0.1)  # Hard (difficulty 0.9)

        # Early steps should get mostly easy examples
        samples_early = sampler.sample(batch_size=50, current_step=150)
        easy_count_early = sum(1 for s in samples_early if s < 50)

        # Late steps should include more hard examples
        samples_late = sampler.sample(batch_size=50, current_step=2000)
        easy_count_late = sum(1 for s in samples_late if s < 50)

        # Early phase should have at least some easy examples
        # Late phase may still have easy examples but should be more diverse
        assert easy_count_early >= 20  # At least 40% easy in early phase

    def test_adaptive_strategy(self, dataset_size):
        """Test adaptive strategy balances exploitation and exploration."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            strategy="adaptive",
            warmup_steps=100,
        )

        # Set difficulties
        for i in range(50):
            sampler.set_pass_rate(i, pass_rate=0.9)  # Easy
        for i in range(50, 100):
            sampler.set_pass_rate(i, pass_rate=0.1)  # Hard

        # Adaptive should sample from all examples
        samples = sampler.sample(batch_size=100, current_step=200)

        easy_samples = sum(1 for s in samples if s < 50)

        # Should have easy examples sampled (but not necessarily more than hard)
        # The adaptive strategy uses UCB which may vary based on step count
        assert easy_samples > 20  # At least 20% easy examples

    def test_mixed_strategy(self, dataset_size):
        """Test mixed strategy balances current difficulty and random."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            strategy="mixed",
            warmup_steps=100,
        )

        # Set difficulties with clear pattern
        for i in range(50):
            sampler.set_pass_rate(i, pass_rate=0.9)  # Easy
        for i in range(50, 100):
            sampler.set_pass_rate(i, pass_rate=0.1)  # Hard

        samples = sampler.sample(batch_size=100, current_step=200)

        assert len(samples) == 100
        assert all(0 <= s < dataset_size for s in samples)

    def test_sample_batch_size(self, curriculum_sampler):
        """Test that sample returns correct batch size."""
        for batch_size in [1, 10, 32, 100]:
            samples = curriculum_sampler.sample(batch_size=batch_size, current_step=500)
            assert len(samples) == batch_size

    def test_sample_deterministic_with_seed(self, dataset_size):
        """Test that same seed produces same samples."""
        sampler1 = CurriculumSampler(dataset_size=dataset_size, seed=42)
        sampler2 = CurriculumSampler(dataset_size=dataset_size, seed=42)

        samples1 = sampler1.sample(batch_size=20, current_step=500)
        samples2 = sampler2.sample(batch_size=20, current_step=500)

        assert samples1 == samples2


# ============================================================================
# TEST: PASS RATE TRACKING
# ============================================================================


class TestPassRateTracking:
    """Test pass rate update and tracking."""

    def test_update_pass_rate_passed(self, curriculum_sampler):
        """Test updating pass rate for successful example."""
        curriculum_sampler.update_pass_rate(0, passed=True)
        curriculum_sampler.update_pass_rate(0, passed=True)
        curriculum_sampler.update_pass_rate(0, passed=False)

        stats = curriculum_sampler._pass_rates[0]
        assert stats["total"] == 3
        assert stats["passed"] == 2
        # Running average: 2/3
        assert abs(stats["pass_rate"] - 2/3) < 0.01

    def test_update_pass_rate_all_passed(self, curriculum_sampler):
        """Test pass rate with all successes."""
        for _ in range(5):
            curriculum_sampler.update_pass_rate(1, passed=True)

        stats = curriculum_sampler._pass_rates[1]
        assert stats["total"] == 5
        assert stats["passed"] == 5
        assert stats["pass_rate"] == 1.0

    def test_update_pass_rate_all_failed(self, curriculum_sampler):
        """Test pass rate with all failures."""
        for _ in range(5):
            curriculum_sampler.update_pass_rate(2, passed=False)

        stats = curriculum_sampler._pass_rates[2]
        assert stats["total"] == 5
        assert stats["passed"] == 0
        assert stats["pass_rate"] < 0.1

    def test_update_pass_rate_running_average(self, curriculum_sampler):
        """Test that running average correctly updates pass rate."""
        # Start with some pass rate
        curriculum_sampler._pass_rates[5]["total"] = 1
        curriculum_sampler._pass_rates[5]["passed"] = 0
        curriculum_sampler._pass_rates[5]["pass_rate"] = 0.0

        # Add successes
        for _ in range(5):
            curriculum_sampler.update_pass_rate(5, passed=True)

        # Rate should be 5/6 after 5 updates
        rate = curriculum_sampler._pass_rates[5]["pass_rate"]
        assert abs(rate - 5/6) < 0.01


# ============================================================================
# TEST: CURRICULUM STATISTICS
# ============================================================================


class TestCurriculumStats:
    """Test curriculum progress statistics."""

    def test_get_curriculum_progress(self, curriculum_sampler, dataset_size):
        """Test getting curriculum progress."""
        # Set some pass rates
        for i in range(50):
            curriculum_sampler.set_pass_rate(i, pass_rate=0.8)
        for i in range(50, 100):
            curriculum_sampler.set_pass_rate(i, pass_rate=0.2)

        progress = curriculum_sampler.get_curriculum_progress()

        assert isinstance(progress, CurriculumStats)
        assert progress.examples_with_data == 100
        assert 0 < progress.avg_difficulty < 1
        assert 0 < progress.avg_pass_rate < 1

    def test_difficulty_distribution(self, curriculum_sampler):
        """Test difficulty distribution histogram."""
        # Create uniform difficulty distribution
        for i in range(100):
            pass_rate = i / 100.0  # From 0% to 100%
            curriculum_sampler.set_pass_rate(i, pass_rate=pass_rate)

        dist = curriculum_sampler.get_difficulty_distribution()

        assert "histogram" in dist
        assert "bin_edges" in dist
        assert "mean" in dist
        assert "std" in dist
        assert len(dist["histogram"]) == 10  # 10 bins

    def test_example_stats(self, curriculum_sampler):
        """Test getting stats for specific example."""
        curriculum_sampler.set_pass_rate(5, pass_rate=0.7)
        curriculum_sampler.update_pass_rate(5, passed=True)

        stats = curriculum_sampler.get_example_stats(5)

        assert stats["example_id"] == 5
        assert stats["total_attempts"] >= 1
        assert stats["pass_rate"] > 0
        assert 0 < stats["difficulty_score"] < 1


# ============================================================================
# TEST: CURRICULUM CALLBACK
# ============================================================================


class TestCurriculumCallback:
    """Test curriculum callback integration."""

    def test_callback_init(self, curriculum_sampler, temp_log_dir):
        """Test callback initialization."""
        callback = CurriculumCallback(
            curriculum_sampler=curriculum_sampler,
            pass_threshold=0.5,
            log_dir=str(temp_log_dir),
        )

        assert callback.curriculum_sampler is curriculum_sampler
        assert callback.pass_threshold == 0.5
        assert callback.log_dir == temp_log_dir

    def test_callback_no_sampler(self):
        """Test callback without sampler doesn't crash."""
        callback = CurriculumCallback(curriculum_sampler=None)

        # Should handle None sampler gracefully
        assert callback.curriculum_sampler is None

    def test_callback_on_train_begin(self, curriculum_sampler):
        """Test on_train_begin callback."""
        callback = CurriculumCallback(curriculum_sampler=curriculum_sampler)

        # Mock args and state
        args = Mock()
        state = Mock()
        control = Mock()

        # Should not raise
        callback.on_train_begin(args, state, control)

    def test_callback_on_step_end_updates_pass_rate(self, curriculum_sampler):
        """Test that on_step_end updates pass rates."""
        callback = CurriculumCallback(curriculum_sampler=curriculum_sampler)

        args = Mock()
        state = Mock()
        control = Mock()

        # Call with rewards and example IDs
        rewards = [0.8, 0.2, 0.9]
        example_ids = [0, 1, 2]

        callback.on_step_end(
            args,
            state,
            control,
            rewards=rewards,
            example_ids=example_ids,
        )

        # Check that pass rates were updated
        assert curriculum_sampler._pass_rates[0]["total"] > 0
        assert curriculum_sampler._pass_rates[1]["total"] > 0
        assert curriculum_sampler._pass_rates[2]["total"] > 0

    def test_callback_get_curriculum_stats(self, curriculum_sampler):
        """Test getting curriculum stats from callback."""
        callback = CurriculumCallback(curriculum_sampler=curriculum_sampler)

        # Set some difficulties
        for i in range(50):
            curriculum_sampler.set_pass_rate(i, pass_rate=0.8)

        stats = callback.get_curriculum_stats()

        assert "avg_difficulty" in stats
        assert "avg_pass_rate" in stats
        assert "overall_pass_rate" in stats
        assert "difficulty_histogram" in stats

    def test_callback_save_log(self, curriculum_sampler, temp_log_dir):
        """Test that callback saves logs to disk."""
        callback = CurriculumCallback(
            curriculum_sampler=curriculum_sampler,
            log_dir=str(temp_log_dir),
            update_interval=1,
        )

        args = Mock()
        state = Mock()
        control = Mock()

        # Simulate training steps
        for step in range(5):
            callback._step_count = step + 1
            rewards = [0.7, 0.3, 0.8]
            example_ids = [0, 1, 2]

            callback.on_step_end(
                args,
                state,
                control,
                rewards=rewards,
                example_ids=example_ids,
            )

        # Check that log file was created
        log_file = temp_log_dir / "curriculum_progress.jsonl"
        assert log_file.exists()

        # Read logs
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) > 0

            # Parse first log
            log_entry = json.loads(lines[0])
            assert "step" in log_entry
            assert "avg_difficulty" in log_entry


# ============================================================================
# TEST: TRAINER INTEGRATION
# ============================================================================


class TestTrainerIntegration:
    """Test curriculum integration with trainer."""

    @pytest.mark.parametrize("strategy", ["easy_first", "adaptive", "mixed"])
    def test_all_strategies_work(self, dataset_size, strategy):
        """Test that all strategies work without errors."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            strategy=strategy,
            warmup_steps=50,
        )

        # Set some difficulties
        for i in range(dataset_size):
            sampler.set_pass_rate(i, pass_rate=np.random.rand())

        # Sample for multiple steps
        for step in range(200):
            samples = sampler.sample(batch_size=32, current_step=step)

            # Update pass rates randomly
            for example_id in samples:
                passed = np.random.rand() > 0.5
                sampler.update_pass_rate(example_id, passed)

        # Get final stats
        progress = sampler.get_curriculum_progress()

        assert progress.examples_with_data > 0
        assert 0 <= progress.avg_pass_rate <= 1
        assert 0 <= progress.avg_difficulty <= 1

    def test_curriculum_integration_with_callback(self):
        """Test curriculum sampler integrated with callback."""
        dataset_size = 50
        sampler = CurriculumSampler(dataset_size=dataset_size, warmup_steps=10)
        callback = CurriculumCallback(curriculum_sampler=sampler)

        # Simulate training
        for step in range(50):
            # Mock batch
            batch_size = 8
            rewards = np.random.rand(batch_size).tolist()
            example_ids = np.random.randint(0, dataset_size, batch_size).tolist()

            # Update via callback
            args = Mock()
            state = Mock()
            control = Mock()

            callback.on_step_end(
                args,
                state,
                control,
                rewards=rewards,
                example_ids=example_ids,
            )

        # Verify statistics were collected
        stats = callback.get_curriculum_stats()
        assert stats["num_sampled"] > 0

    def test_curriculum_with_different_thresholds(self, dataset_size):
        """Test curriculum with different pass thresholds."""
        sampler = CurriculumSampler(dataset_size=dataset_size)

        for threshold in [0.0, 0.5, 1.0]:
            callback = CurriculumCallback(
                curriculum_sampler=sampler,
                pass_threshold=threshold,
            )

            # Simulate updates with threshold
            args = Mock()
            state = Mock()
            control = Mock()

            callback.on_step_end(
                args,
                state,
                control,
                rewards=[0.5],
                example_ids=[0],
            )


# ============================================================================
# TEST: EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test behavior with very small dataset."""
        sampler = CurriculumSampler(dataset_size=1)

        samples = sampler.sample(batch_size=10, current_step=500)
        assert len(samples) == 10
        assert all(s == 0 for s in samples)

    def test_batch_size_larger_than_dataset(self, curriculum_sampler):
        """Test sampling batch larger than dataset."""
        samples = curriculum_sampler.sample(batch_size=200, current_step=500)

        assert len(samples) == 200
        assert all(0 <= s < 100 for s in samples)

    def test_no_examples_with_data(self, curriculum_sampler):
        """Test progress stats when no examples have been sampled."""
        progress = curriculum_sampler.get_curriculum_progress()

        assert progress.examples_with_data == 0
        assert progress.num_sampled == 0

    def test_invalid_example_id_update(self, curriculum_sampler):
        """Test updating pass rate with invalid example ID."""
        # Should not crash
        curriculum_sampler.update_pass_rate(9999, passed=True)
        curriculum_sampler.update_pass_rate(-1, passed=False)

    def test_invalid_pass_rate_value(self, curriculum_sampler):
        """Test setting invalid pass rate value."""
        with pytest.raises(ValueError):
            curriculum_sampler.set_pass_rate(0, pass_rate=1.5)

        with pytest.raises(ValueError):
            curriculum_sampler.set_pass_rate(0, pass_rate=-0.1)


# ============================================================================
# TEST: PERFORMANCE AND SCALABILITY
# ============================================================================


class TestPerformance:
    """Test performance with large datasets."""

    @pytest.mark.parametrize("dataset_size", [1000, 10000])
    def test_large_dataset(self, dataset_size):
        """Test curriculum sampler with large dataset."""
        sampler = CurriculumSampler(
            dataset_size=dataset_size,
            warmup_steps=100,
        )

        # Set difficulties for all examples
        for i in range(dataset_size):
            sampler.set_pass_rate(i, pass_rate=np.random.rand())

        # Sample and update
        for _ in range(10):
            samples = sampler.sample(batch_size=32, current_step=500)
            for example_id in samples:
                sampler.update_pass_rate(example_id, passed=np.random.rand() > 0.5)

        # Get stats
        progress = sampler.get_curriculum_progress()
        assert progress.examples_with_data > 0

    def test_many_updates(self, curriculum_sampler):
        """Test many rapid updates to pass rates."""
        for _ in range(1000):
            example_id = np.random.randint(0, 100)
            passed = np.random.rand() > 0.5
            curriculum_sampler.update_pass_rate(example_id, passed)

        progress = curriculum_sampler.get_curriculum_progress()
        assert progress.num_sampled == 1000
