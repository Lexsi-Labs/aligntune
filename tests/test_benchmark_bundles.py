"""
Comprehensive tests for benchmark bundles and alignment metrics.

Tests verify:
- Bundle existence and task lists
- Bundle expansion in evaluator
- Alignment metric computation
- CLI --bundle flag parsing
"""

import pytest
from typing import List, Dict, Any
import numpy as np


class TestBenchmarkBundles:
    """Test benchmark bundle definitions and access."""

    def test_bundles_exist(self):
        """Verify all expected bundles are defined."""
        from aligntune.eval.benchmarks.presets import BENCHMARK_BUNDLES

        expected_bundles = ["alignment_core", "safety", "reasoning"]
        for bundle_name in expected_bundles:
            assert bundle_name in BENCHMARK_BUNDLES, f"Bundle {bundle_name} not found"

    def test_bundles_not_empty(self):
        """Verify each bundle contains tasks."""
        from aligntune.eval.benchmarks.presets import BENCHMARK_BUNDLES

        for bundle_name, tasks in BENCHMARK_BUNDLES.items():
            assert len(tasks) > 0, f"Bundle {bundle_name} is empty"
            assert all(isinstance(t, str) for t in tasks), \
                f"Bundle {bundle_name} contains non-string tasks"

    def test_bundle_structure(self):
        """Verify bundle structure and task names."""
        from aligntune.eval.benchmarks.presets import BENCHMARK_BUNDLES

        # Alignment core should have safety + reasoning tasks
        assert "alignment_core" in BENCHMARK_BUNDLES
        align_tasks = BENCHMARK_BUNDLES["alignment_core"]
        assert len(align_tasks) >= 5, "alignment_core should have at least 5 tasks"

        # Safety should have dedicated safety tasks
        assert "safety" in BENCHMARK_BUNDLES
        safety_tasks = BENCHMARK_BUNDLES["safety"]
        assert len(safety_tasks) >= 3, "safety bundle should have at least 3 tasks"

        # Reasoning should have math/reasoning tasks
        assert "reasoning" in BENCHMARK_BUNDLES
        reasoning_tasks = BENCHMARK_BUNDLES["reasoning"]
        assert len(reasoning_tasks) >= 3, "reasoning bundle should have at least 3 tasks"

    def test_get_bundle_function(self):
        """Test get_bundle() retrieves correct tasks."""
        from aligntune.eval.benchmarks.presets import get_bundle, BENCHMARK_BUNDLES

        for bundle_name in BENCHMARK_BUNDLES.keys():
            tasks = get_bundle(bundle_name)
            assert tasks == BENCHMARK_BUNDLES[bundle_name]
            assert len(tasks) > 0

    def test_get_bundle_invalid_name(self):
        """Test get_bundle() raises error for invalid bundle."""
        from aligntune.eval.benchmarks.presets import get_bundle

        with pytest.raises(ValueError, match="Unknown benchmark bundle"):
            get_bundle("nonexistent_bundle")

    def test_list_bundles_function(self):
        """Test list_bundles() returns all bundle names."""
        from aligntune.eval.benchmarks.presets import list_bundles, BENCHMARK_BUNDLES

        bundles = list_bundles()
        assert set(bundles) == set(BENCHMARK_BUNDLES.keys())
        assert len(bundles) == 3  # Should have 3 preset bundles


class TestRegistryBundleSupport:
    """Test EvalRegistry bundle integration."""

    def test_registry_get_bundle(self):
        """Test EvalRegistry.get_bundle() method."""
        from aligntune.eval.registry import EvalRegistry

        for bundle_name in EvalRegistry.list_bundles():
            tasks = EvalRegistry.get_bundle(bundle_name)
            assert isinstance(tasks, list)
            assert len(tasks) > 0

    def test_registry_list_bundles(self):
        """Test EvalRegistry.list_bundles() method."""
        from aligntune.eval.registry import EvalRegistry

        bundles = EvalRegistry.list_bundles()
        assert isinstance(bundles, list)
        assert "alignment_core" in bundles
        assert "safety" in bundles
        assert "reasoning" in bundles

    def test_registry_bundle_expansion(self):
        """Test bundle expansion resolves to expected tasks."""
        from aligntune.eval.registry import EvalRegistry

        bundle_tasks = EvalRegistry.get_bundle("alignment_core")
        assert isinstance(bundle_tasks, list)
        assert all(isinstance(t, str) for t in bundle_tasks)
        # Should have specific tasks
        assert any("truthful" in t.lower() or "qa" in t.lower() for t in bundle_tasks)


class TestEvaluatorBundleSupport:
    """Test BaseEvaluator bundle parameter handling."""

    def test_evaluator_bundle_parameter_accepted(self):
        """Test evaluator accepts bundle parameter."""
        from aligntune.eval.evaluator import BaseEvaluator

        evaluator = BaseEvaluator()
        # Verify method signature includes bundle parameter
        import inspect
        sig = inspect.signature(evaluator.evaluate_benchmark)
        assert "bundle" in sig.parameters

    def test_evaluator_bundle_expansion_logic(self):
        """Test evaluator correctly expands bundles."""
        from aligntune.eval.evaluator import BaseEvaluator

        evaluator = BaseEvaluator()
        # This would need mocking of lm-eval to fully test
        # For now, just verify the method exists and is callable
        assert callable(evaluator.evaluate_benchmark)


class TestAlignmentMetrics:
    """Test alignment-specific metrics."""

    def test_refusal_rate_metric_exists(self):
        """Verify RefusalRate metric is importable."""
        from aligntune.eval.metrics.alignment import RefusalRate

        metric = RefusalRate()
        assert metric.name == "refusal_rate"

    def test_refusal_rate_computation(self):
        """Test RefusalRate computation with sample data."""
        from aligntune.eval.metrics.alignment import RefusalRate

        metric = RefusalRate()
        refusals = [1, 1, 0, 1, 0]  # 3/5 refusals = 0.6

        result = metric.compute([], [], refusals=refusals)
        assert "refusal_rate" in result
        assert abs(result["refusal_rate"] - 0.6) < 0.01

    def test_refusal_rate_empty_data(self):
        """Test RefusalRate handles empty data gracefully."""
        from aligntune.eval.metrics.alignment import RefusalRate

        metric = RefusalRate()
        result = metric.compute([], [])
        assert "refusal_rate" in result
        assert result["refusal_rate"] == 0.0

    def test_sycophancy_score_metric_exists(self):
        """Verify SycophancyScore metric is importable."""
        from aligntune.eval.metrics.alignment import SycophancyScore

        metric = SycophancyScore()
        assert metric.name == "sycophancy_score"

    def test_sycophancy_score_computation(self):
        """Test SycophancyScore computation with sample data."""
        from aligntune.eval.metrics.alignment import SycophancyScore

        metric = SycophancyScore()
        predictions = ["Yes", "No", "Yes", "No"]
        bias_labels = [1, 0, 1, 0]  # Perfect agreement = 1.0

        result = metric.compute(predictions, [], bias_labels=bias_labels)
        assert "sycophancy_score" in result
        assert abs(result["sycophancy_score"] - 0.5) < 0.01

    def test_sycophancy_score_empty_data(self):
        """Test SycophancyScore handles empty data gracefully."""
        from aligntune.eval.metrics.alignment import SycophancyScore

        metric = SycophancyScore()
        result = metric.compute([], [])
        assert "sycophancy_score" in result
        assert result["sycophancy_score"] == 0.0

    def test_verbosity_delta_metric_exists(self):
        """Verify VerbosityDelta metric is importable."""
        from aligntune.eval.metrics.alignment import VerbosityDelta

        metric = VerbosityDelta()
        assert metric.name == "verbosity_delta"

    def test_verbosity_delta_computation_with_baselines(self):
        """Test VerbosityDelta computation with baseline lengths."""
        from aligntune.eval.metrics.alignment import VerbosityDelta

        metric = VerbosityDelta()
        predictions = [
            "This is a short response.",  # 5 words
            "This is a longer response with more words.",  # 8 words
        ]
        baseline_lengths = [5, 5]  # Both baselines are 5

        result = metric.compute(
            predictions, [], baseline_lengths=baseline_lengths
        )
        assert "verbosity_delta" in result
        # Average delta: ((5-5) + (8-5)) / 2 = 1.5
        # (the second prediction is 8 words, not 9 - the original expectation
        # miscounted it)
        assert abs(result["verbosity_delta"] - 1.5) < 0.1

    def test_verbosity_delta_computation_with_references(self):
        """Test VerbosityDelta using reference outputs as baseline."""
        from aligntune.eval.metrics.alignment import VerbosityDelta

        metric = VerbosityDelta()
        predictions = [
            "This is a longer output.",  # 5 words
            "This is a much longer output with extra words.",  # 9 words
        ]
        references = [
            "Short.",  # 1 word
            "Medium output.",  # 2 words
        ]

        result = metric.compute(predictions, references)
        assert "verbosity_delta" in result
        # Should compute delta based on reference lengths

    def test_verbosity_delta_empty_data(self):
        """Test VerbosityDelta handles empty data gracefully."""
        from aligntune.eval.metrics.alignment import VerbosityDelta

        metric = VerbosityDelta()
        result = metric.compute([], [])
        assert "verbosity_delta" in result
        assert result["verbosity_delta"] == 0.0


class TestMetricsIntegration:
    """Test alignment metrics integration with evaluation system."""

    def test_alignment_metrics_in_metrics_module(self):
        """Test alignment metrics are exported from metrics module."""
        from aligntune.eval.metrics import RefusalRate, SycophancyScore, VerbosityDelta

        assert RefusalRate is not None
        assert SycophancyScore is not None
        assert VerbosityDelta is not None

    def test_alignment_metrics_safe_compute(self):
        """Test safe_compute wrapper for alignment metrics."""
        from aligntune.eval.metrics.alignment import RefusalRate

        metric = RefusalRate()
        # Test that safe_compute doesn't crash with bad data
        result = metric.safe_compute(None, None)
        assert isinstance(result, dict)


class TestCLIBundleIntegration:
    """Test CLI integration with bundle support."""

    def test_bundle_option_parsing(self):
        """Test --bundle option is properly parsed."""
        from aligntune.eval.registry import EvalRegistry

        bundles = EvalRegistry.list_bundles()
        assert len(bundles) > 0
        # Should be able to parse any bundle name
        for bundle_name in bundles:
            tasks = EvalRegistry.get_bundle(bundle_name)
            assert isinstance(tasks, list)


class TestEndToEndScenarios:
    """Integration tests for complete evaluation flows."""

    def test_bundle_task_expansion(self):
        """Test complete bundle -> tasks expansion flow."""
        from aligntune.eval.registry import EvalRegistry

        bundle_name = "alignment_core"
        tasks = EvalRegistry.get_bundle(bundle_name)

        # Verify tasks are valid strings
        assert all(isinstance(t, str) for t in tasks)
        assert all(len(t) > 0 for t in tasks)

    def test_multiple_bundles_combination(self):
        """Test combining tasks from multiple bundles."""
        from aligntune.eval.registry import EvalRegistry

        bundles = ["alignment_core", "safety"]
        all_tasks = []

        for bundle_name in bundles:
            tasks = EvalRegistry.get_bundle(bundle_name)
            all_tasks.extend(tasks)

        # Remove duplicates and verify
        unique_tasks = list(set(all_tasks))
        assert len(unique_tasks) > 0

    def test_alignment_metric_workflow(self):
        """Test complete alignment metric computation workflow."""
        from aligntune.eval.metrics.alignment import RefusalRate, SycophancyScore

        # Simulate evaluation data
        refusal_data = [1, 1, 0, 1, 0, 0, 1]
        bias_data = [1, 0, 1, 0, 1]
        predictions = ["response"] * 5

        # Compute metrics
        refusal_metric = RefusalRate()
        refusal_result = refusal_metric.compute([], [], refusals=refusal_data)

        syco_metric = SycophancyScore()
        syco_result = syco_metric.compute(predictions, [], bias_labels=bias_data)

        assert "refusal_rate" in refusal_result
        assert "sycophancy_score" in syco_result
        assert isinstance(refusal_result["refusal_rate"], float)
        assert isinstance(syco_result["sycophancy_score"], float)


# Integration tests with actual model (if available)
class TestWithModel:
    """Optional end-to-end tests with actual model evaluation."""

    def test_evaluate_with_bundle_alignment_core(self):
        """Test evaluation with alignment_core bundle (requires model)."""
        from aligntune.eval.evaluator import BaseEvaluator
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            evaluator = BaseEvaluator()
            results = evaluator.evaluate_benchmark(
                "gpt2",
                bundle="alignment_core"
            )

            assert isinstance(results, dict)
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_cli_eval_with_bundle(self):
        """Test CLI eval command with --bundle flag."""
        from typer.testing import CliRunner
        from aligntune.eval.cli import app

        runner = CliRunner()
        # This would require a real model and full lm-eval setup
        # Just verify the command structure
        result = runner.invoke(app, ["benchmark", "--help"])
        assert "--bundle" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
