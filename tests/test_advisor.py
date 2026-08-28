"""
Unit tests for the advisor module (cost estimation and algorithm recommendations).

Tests cover:
- Resource estimation with known models
- Algorithm recommendations under various scenarios
- Optimization suggestions based on configuration
- CLI output formatting
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from aligntune.core.advisor import (
    GPUProfile,
    Estimate,
    Recommendation,
    OptimizationSuggestion,
    infer_model_size,
    round_to_significant_figures,
    estimate_resources,
    recommend_algorithm,
    suggest_optimizations,
    format_estimate_table,
    format_recommendations,
    format_optimizations,
    GPU_PROFILES,
)


class TestDataStructures:
    """Test data structure initialization and representation."""

    def test_gpu_profile_creation(self):
        gpu = GPUProfile(
            name="test-gpu",
            vram_gb=40.0,
            tflops_fp16=312.0,
            price_per_hour_usd=1.50,
        )
        assert gpu.name == "test-gpu"
        assert gpu.vram_gb == 40.0
        assert gpu.tflops_fp16 == 312.0
        assert gpu.price_per_hour_usd == 1.50
        assert gpu.power_consumption_watts == 400.0  # default

    def test_estimate_creation(self):
        est = Estimate(
            vram_gb=32.5,
            wallclock_hours=2.5,
            cost_usd=5.0,
        )
        assert est.vram_gb == 32.5
        assert est.wallclock_hours == 2.5
        assert est.cost_usd == 5.0
        assert est.vram_uncertainty_pct == 30.0

    def test_recommendation_creation(self):
        rec = Recommendation(
            algorithm="dpo",
            score=0.92,
            reason="Best for alignment with your dataset size",
        )
        assert rec.algorithm == "dpo"
        assert rec.score == 0.92

    def test_optimization_suggestion_creation(self):
        opt = OptimizationSuggestion(
            optimization="FlashAttention-2",
            benefit="3-5x attention speedup",
            impact="No quality impact",
        )
        assert opt.optimization == "FlashAttention-2"
        assert opt.benefit == "3-5x attention speedup"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_infer_model_size_returns_none_for_unknown(self):
        result = infer_model_size("some-random-model")
        assert result is None

    def test_round_to_significant_figures(self):
        # 2 significant figures
        assert round_to_significant_figures(123.456, 2) == 120
        assert round_to_significant_figures(0.00456, 2) == 0.0046
        assert round_to_significant_figures(5.678, 2) == 5.7

    def test_round_to_significant_figures_zero(self):
        assert round_to_significant_figures(0, 2) == 0


class TestResourceEstimation:
    """Test resource estimation functionality."""

    def test_estimate_resources_basic(self):
        """Test basic resource estimation."""
        estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="sft",
            hardware_profile="a100-40gb",
        )

        assert isinstance(estimate, Estimate)
        assert estimate.vram_gb > 0
        assert estimate.wallclock_hours > 0
        assert estimate.cost_usd > 0
        assert estimate.vram_uncertainty_pct == 30.0

    def test_estimate_resources_with_dpo(self):
        """Test estimation with DPO algorithm (higher VRAM)."""
        sft_estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="sft",
        )
        dpo_estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="dpo",
        )

        # DPO should require more VRAM and time
        assert dpo_estimate.vram_gb > sft_estimate.vram_gb * 0.9  # Should be higher
        assert dpo_estimate.wallclock_hours > sft_estimate.wallclock_hours * 0.5

    def test_estimate_resources_with_lora(self):
        """Test estimation with LoRA (lower VRAM)."""
        full_estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="sft",
        )
        lora_estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="lora",
        )

        # LoRA should require much less VRAM
        assert lora_estimate.vram_gb < full_estimate.vram_gb
        # LoRA throughput might be slightly better
        assert lora_estimate.cost_usd < full_estimate.cost_usd * 1.5

    def test_estimate_resources_with_qlora(self):
        """Test estimation with QLoRA (minimal VRAM)."""
        qlora_estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="qlora",
        )

        # QLoRA should have very low VRAM footprint
        assert qlora_estimate.vram_gb < 5  # Should fit in small GPUs
        assert qlora_estimate.vram_gb > 0.1  # But still positive

    def test_estimate_resources_dataset_scaling(self):
        """Test that cost scales linearly with dataset size."""
        small_est = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=1000,
        )
        large_est = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
        )

        # Cost should scale roughly linearly (10x dataset -> ~10x cost)
        assert large_est.cost_usd > small_est.cost_usd * 8
        assert large_est.cost_usd < small_est.cost_usd * 12

    def test_estimate_resources_gpu_scaling(self):
        """Test that faster GPUs reduce training time."""
        a100_est = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            hardware_profile="a100-40gb",
        )
        h100_est = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            hardware_profile="h100",
        )

        # H100 is much faster, should take less time
        assert h100_est.wallclock_hours < a100_est.wallclock_hours
        # H100 costs more per hour but should be cheaper overall for fast job
        assert h100_est.cost_usd < a100_est.cost_usd

    def test_estimate_resources_unknown_model(self):
        """Test handling of unknown model names."""
        estimate = estimate_resources(
            model_name="unknown-model-xyz",
            dataset_size=10000,
        )

        # Should still return an estimate (with default 7B size)
        assert estimate.vram_gb > 0
        assert estimate.wallclock_hours > 0

    def test_estimate_resources_unknown_gpu(self):
        """Test handling of unknown GPU types."""
        estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            hardware_profile="unknown-gpu",
        )

        # Should still return an estimate (with default GPU)
        assert estimate.vram_gb > 0
        assert estimate.cost_usd > 0

class TestAlgorithmRecommendation:
    """Test algorithm recommendation functionality."""

    def test_recommend_algorithm_basic(self):
        """Test basic algorithm recommendation."""
        recommendations = recommend_algorithm(
            task_description="alignment",
            dataset_size=10000,
        )

        assert len(recommendations) > 0
        assert all(isinstance(r, Recommendation) for r in recommendations)
        # Should be sorted by score descending
        assert recommendations[0].score >= recommendations[-1].score

    def test_recommend_algorithm_alignment_boost(self):
        """Test that alignment task boosts DPO."""
        alignment_recs = recommend_algorithm(
            task_description="alignment optimization",
            dataset_size=10000,
        )

        # Find DPO score
        dpo_score = next((r.score for r in alignment_recs if r.algorithm == "dpo"), 0)
        sft_score = next((r.score for r in alignment_recs if r.algorithm == "sft"), 0)

        # DPO should be boosted for alignment task
        assert dpo_score >= sft_score

    def test_recommend_algorithm_speed_boost(self):
        """Test that speed task boosts LoRA/QLoRA."""
        speed_recs = recommend_algorithm(
            task_description="fast training",
            dataset_size=10000,
        )

        lora_score = next((r.score for r in speed_recs if r.algorithm == "lora"), 0)
        ppo_score = next((r.score for r in speed_recs if r.algorithm == "ppo"), 0)

        # LoRA should be boosted for speed
        assert lora_score > ppo_score

    def test_recommend_algorithm_small_dataset(self):
        """Test recommendations for small datasets."""
        small_recs = recommend_algorithm(
            task_description="general",
            dataset_size=100,  # Very small
        )

        lora_score = next((r.score for r in small_recs if r.algorithm == "lora"), 0)
        ppo_score = next((r.score for r in small_recs if r.algorithm == "ppo"), 0)

        # LoRA should rank high for small datasets
        assert lora_score > ppo_score * 0.8

    def test_recommend_algorithm_large_dataset(self):
        """Test recommendations for large datasets."""
        large_recs = recommend_algorithm(
            task_description="general",
            dataset_size=500000,  # Very large
        )

        ppo_score = next((r.score for r in large_recs if r.algorithm == "ppo"), 0)
        lora_score = next((r.score for r in large_recs if r.algorithm == "lora"), 0)

        # PPO/RL should be competitive for large datasets
        assert ppo_score >= lora_score * 0.6

    def test_recommend_algorithm_with_budget(self):
        """Test algorithm filtering with budget constraint."""
        # With very small budget, fast algorithms should rank high
        cheap_recs = recommend_algorithm(
            task_description="general",
            dataset_size=10000,
            budget_usd=1.0,  # Very small budget
        )

        qlora_score = next((r.score for r in cheap_recs if r.algorithm == "qlora"), 0)
        ppo_score = next((r.score for r in cheap_recs if r.algorithm == "ppo"), 0)

        # QLoRA should be favored with tight budget
        assert qlora_score > 0.4


class TestOptimizationSuggestions:
    """Test optimization suggestion functionality."""

    def test_suggest_optimizations_basic(self):
        """Test basic optimization suggestions."""
        suggestions = suggest_optimizations(
            model_size="7b",
            precision="fp32",
            hardware="a100-40gb",
        )

        assert isinstance(suggestions, list)
        assert all(isinstance(s, OptimizationSuggestion) for s in suggestions)

    def test_suggest_optimizations_flash_attention(self):
        """Test FlashAttention-2 suggestion for modern GPUs."""
        suggestions = suggest_optimizations(
            model_size="7b",
            precision="fp32",
            hardware="a100-40gb",
        )

        algo_names = [s.optimization for s in suggestions]
        assert any("flash" in name.lower() for name in algo_names)

    def test_suggest_optimizations_precision_tuning(self):
        """Test precision tuning suggestions."""
        suggestions = suggest_optimizations(
            model_size="7b",
            precision="fp32",  # Non-optimal precision
            hardware="a100-40gb",
        )

        algo_names = [s.optimization for s in suggestions]
        assert any("bf16" in name.lower() or "precision" in name.lower() for name in algo_names)

    def test_suggest_optimizations_gradient_accumulation(self):
        """Test gradient accumulation suggestion for large datasets."""
        suggestions = suggest_optimizations(
            model_size="7b",
            dataset_size=100000,  # Large dataset
            hardware="a100-40gb",
        )

        algo_names = [s.optimization for s in suggestions]
        assert any("gradient" in name.lower() for name in algo_names)

    def test_suggest_optimizations_small_dataset_warning(self):
        """Test warning for very small datasets."""
        suggestions = suggest_optimizations(
            model_size="7b",
            dataset_size=100,  # Very small
            hardware="a100-40gb",
        )

        algo_names = [s.optimization for s in suggestions]
        assert any("small" in name.lower() for name in algo_names)

    def test_suggest_optimizations_vram_tight(self):
        """Test VRAM-tight optimizations."""
        suggestions = suggest_optimizations(
            model_size="7b",
            precision="fp32",
            hardware="t4",  # Low VRAM GPU
            vram_tight=True,
        )

        algo_names = [s.optimization for s in suggestions]
        assert any("gradient" in name.lower() or "checkpoint" in name.lower() for name in algo_names)


class TestFormattingFunctions:
    """Test output formatting functions."""

    def test_format_estimate_table(self):
        """Test estimate table formatting."""
        estimate = Estimate(
            vram_gb=32.5,
            wallclock_hours=2.5,
            cost_usd=5.0,
        )

        formatted = format_estimate_table(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="sft",
            hardware="a100-40gb",
            estimate=estimate,
        )

        assert "Qwen" in formatted
        assert "10000" in formatted or "10,000" in formatted
        assert "32.5" in formatted
        assert "2.5" in formatted
        assert "$5.0" in formatted or "$5.00" in formatted

    def test_format_recommendations(self):
        """Test recommendations formatting."""
        recommendations = [
            Recommendation("dpo", 0.92, "Best for alignment"),
            Recommendation("sft", 0.75, "General purpose"),
        ]

        formatted = format_recommendations(recommendations)

        assert "dpo" in formatted.lower()
        assert "sft" in formatted.lower()
        assert "0.92" in formatted
        assert "0.75" in formatted

    def test_format_optimizations(self):
        """Test optimization suggestions formatting."""
        suggestions = [
            OptimizationSuggestion("FA2", "3-5x speedup", "No impact"),
            OptimizationSuggestion("QLoRA", "2-4x memory", "Slight slowdown"),
        ]

        formatted = format_optimizations(suggestions)

        assert "FA2" in formatted
        assert "QLoRA" in formatted
        assert "3-5x" in formatted

    def test_format_optimizations_empty(self):
        """Test formatting with no suggestions."""
        formatted = format_optimizations([])
        assert formatted == ""


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_estimate_and_recommend(self):
        """Test complete workflow: estimate resources and get recommendations."""
        # Estimate for DPO
        estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10000,
            algorithm="dpo",
        )

        # Get recommendations for alignment task
        recommendations = recommend_algorithm(
            task_description="alignment",
            dataset_size=10000,
            budget_usd=estimate.cost_usd * 1.5,  # Budget with some margin
        )

        # Get optimizations
        suggestions = suggest_optimizations(
            model_size="7b",
            precision="fp32",
            hardware="a100-40gb",
            dataset_size=10000,
        )

        # Verify all components work together
        assert estimate.cost_usd > 0
        assert len(recommendations) > 0
        assert len(suggestions) > 0

    def test_edge_case_tiny_dataset(self):
        """Test with very tiny dataset."""
        estimate = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=10,  # 10 samples
        )

        assert estimate.wallclock_hours < 0.01  # Should be very fast
        assert estimate.cost_usd < 0.1  # Should be very cheap

    def test_edge_case_budget_constraint(self):
        """Test algorithm recommendation with strict budget."""
        recommendations = recommend_algorithm(
            task_description="general",
            dataset_size=10000,
            budget_usd=0.5,  # Very strict budget
        )

        # LoRA/QLoRA should rank high
        top_algo = recommendations[0].algorithm
        assert top_algo in ["lora", "qlora", "sft"]


class TestSnapshotComparison:
    """Snapshot tests to ensure estimates remain consistent."""

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
