"""
Tests for Reasoning Pipeline Components (v3.6).

Covers:
- ReasoningBenchmark loading for multiple datasets
- LM-eval task registration for reasoning benchmarks
- No actual model training or benchmark downloads
- CPU-only testing
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

from aligntune.eval.benchmarks.reasoning import (
    ReasoningBenchmark,
    ReasoningBenchmarkData,
)
from aligntune.eval.lm_eval_integration import (
    REASONING_TASKS,
    LMEVAL_TASKS,
    LMEvalTask,
    TaskCategory,
)


class TestReasoningBenchmark:
    """Tests for ReasoningBenchmark."""

    def test_benchmark_initialization(self):
        """Test benchmark initializes correctly."""
        benchmark = ReasoningBenchmark()
        assert benchmark.cache_dir is None
        assert len(benchmark._loaded_benchmarks) == 0

    def test_benchmark_list_supported(self):
        """Test listing supported benchmarks."""
        benchmark = ReasoningBenchmark()
        supported = benchmark.list_benchmarks()

        assert "AIME" in supported
        assert "MATH" in supported
        assert "GPQA" in supported
        assert "LiveCodeBench" in supported
        assert "GSM8K-CoT" in supported

    def test_load_aime_benchmark(self):
        """Test loading AIME benchmark."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("AIME", max_samples=1)

        assert data.name == "AIME"
        assert len(data.questions) >= 1
        assert len(data.solutions) >= 1
        assert data.steps is not None
        assert data.step_labels is not None

    def test_load_math_benchmark(self):
        """Test loading MATH benchmark."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("MATH", max_samples=1)

        assert data.name == "MATH"
        assert len(data.questions) >= 1
        assert len(data.solutions) >= 1

    def test_load_gpqa_benchmark(self):
        """Test loading GPQA benchmark."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("GPQA")

        assert data.name == "GPQA"
        assert len(data.questions) >= 1

    def test_load_livecodebench(self):
        """Test loading LiveCodeBench."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("LiveCodeBench")

        assert data.name == "LiveCodeBench"
        assert len(data.questions) >= 1

    def test_load_gsm8k_cot(self):
        """Test loading GSM8K-CoT."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("GSM8K-CoT")

        assert data.name == "GSM8K-CoT"
        assert len(data.questions) >= 1

    def test_unsupported_benchmark(self):
        """Test error on unsupported benchmark."""
        benchmark = ReasoningBenchmark()

        with pytest.raises(ValueError, match="Unsupported benchmark"):
            benchmark.load_benchmark("INVALID_BENCHMARK")

    def test_benchmark_caching(self):
        """Test benchmark results are cached."""
        benchmark = ReasoningBenchmark()

        # Load twice
        data1 = benchmark.load_benchmark("AIME", split="test")
        data2 = benchmark.load_benchmark("AIME", split="test")

        # Should be same cached object
        assert data1 is data2

    def test_benchmark_to_dict(self):
        """Test benchmark data converts to dictionary."""
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark("AIME", max_samples=1)

        data_dict = data.to_dict()

        assert "name" in data_dict
        assert "questions" in data_dict
        assert "solutions" in data_dict
        assert "steps" in data_dict
        assert "step_labels" in data_dict
        assert "num_samples" in data_dict


class TestReasoningBenchmarkData:
    """Tests for ReasoningBenchmarkData."""

    def test_initialization(self):
        """Test benchmark data initializes."""
        data = ReasoningBenchmarkData(
            name="test",
            questions=["Q1", "Q2"],
            solutions=["S1", "S2"],
            steps=[["step1"], ["step2"]],
            step_labels=[[1], [0]],
        )

        assert data.name == "test"
        assert len(data) == 2
        assert data.questions == ["Q1", "Q2"]

    def test_len(self):
        """Test __len__ returns correct count."""
        data = ReasoningBenchmarkData(
            name="test",
            questions=["Q1", "Q2", "Q3"],
            solutions=["S1", "S2", "S3"],
        )

        assert len(data) == 3


class TestLMEvalReasoningTasks:
    """Tests for lm-eval reasoning task registration."""

    def test_reasoning_tasks_dict(self):
        """Test REASONING_TASKS dictionary contains expected tasks."""
        assert "aime" in REASONING_TASKS
        assert "math" in REASONING_TASKS
        assert "gpqa" in REASONING_TASKS
        assert "livecode" in REASONING_TASKS
        assert "gsm8k_cot" in REASONING_TASKS

    def test_reasoning_task_properties(self):
        """Test reasoning tasks have correct properties."""
        task = REASONING_TASKS["aime"]

        assert task["name"] == "AIME"
        assert task["category"] == "reasoning"
        assert "lm_eval_task_name" in task
        assert "metric_type" in task
        assert task["metric_type"] == "exact_match"

    def test_lmeval_tasks_include_reasoning(self):
        """Test reasoning tasks registered in LMEVAL_TASKS."""
        # Check key reasoning tasks
        assert "aime" in LMEVAL_TASKS
        assert "math" in LMEVAL_TASKS
        assert "gpqa" in LMEVAL_TASKS
        assert "gsm8k_cot" in LMEVAL_TASKS

        # Check they are LMEvalTask instances
        assert isinstance(LMEVAL_TASKS["aime"], LMEvalTask)
        assert isinstance(LMEVAL_TASKS["math"], LMEvalTask)

    def test_reasoning_task_category(self):
        """Test reasoning tasks have correct category."""
        assert LMEVAL_TASKS["aime"].category == TaskCategory.REASONING
        assert LMEVAL_TASKS["math"].category == TaskCategory.REASONING
        assert LMEVAL_TASKS["gpqa"].category == TaskCategory.REASONING

    def test_reasoning_task_metrics(self):
        """Test reasoning tasks specify metrics."""
        aime_task = LMEVAL_TASKS["aime"]
        assert len(aime_task.metrics) > 0
        assert "exact_match" in aime_task.metrics


class TestReasoningPipelineIntegration:
    """Integration tests for reasoning pipeline components."""

    def test_all_reasoning_benchmarks_loadable(self):
        """Test all reasoning benchmarks can be loaded."""
        benchmark = ReasoningBenchmark()

        for bench_name in benchmark.list_benchmarks():
            data = benchmark.load_benchmark(bench_name, max_samples=1)
            assert data is not None
            assert len(data) >= 1
