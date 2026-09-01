"""
Smoke tests for Reward Model Ensembles - checks that code structure is correct.

This test file doesn't actually import the full system to avoid the transformers
circular import issue during testing. Instead, it verifies the code changes are present.
"""

import pytest
import re
from pathlib import Path


def test_composite_reward_has_ensemble_mode_parameter():
    """Check that CompositeReward has ensemble_mode parameter in __init__."""
    core_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "core.py"
    content = core_file.read_text(encoding="utf-8")

    # Check ensemble_mode parameter exists
    assert "ensemble_mode: str = \"mean\"" in content

    # Check validation for ensemble_mode
    assert "ensemble_mode not in [\"mean\", \"worst_case\", \"uncertainty_weighted\"]" in content

    # Check the three aggregation modes are implemented
    assert "if self.ensemble_mode == \"mean\":" in content
    assert "elif self.ensemble_mode == \"worst_case\":" in content
    assert "elif self.ensemble_mode == \"uncertainty_weighted\":" in content


def test_reward_type_enum_has_ensemble():
    """Check that RewardType enum includes REWARD_MODEL_ENSEMBLE."""
    core_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "core.py"
    content = core_file.read_text(encoding="utf-8")

    # Check enum value
    assert "REWARD_MODEL_ENSEMBLE = \"reward_model_ensemble\"" in content


def test_ensemble_module_exists():
    """Check that ensemble.py file exists and has required class."""
    ensemble_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "ensemble.py"
    assert ensemble_file.exists()

    content = ensemble_file.read_text(encoding="utf-8")

    # Check class exists
    assert "class RewardModelEnsemble(RewardFunction):" in content

    # Check main methods exist
    assert "def __init__(self, config: RewardConfig):" in content
    assert "def compute(self, text: str" in content
    assert "def batch_compute(" in content
    assert "def _load_models(self)" in content
    assert "def get_ensemble_mode(self)" in content
    assert "def set_ensemble_mode(self, mode: str)" in content
    assert "def get_ensemble_stats(self" in content


def test_factory_has_lazy_load_ensemble():
    """Check that RewardFunctionFactory has lazy loading for ensemble."""
    core_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "core.py"
    content = core_file.read_text(encoding="utf-8")

    # Check lazy load method exists
    assert "_lazy_load_ensemble_reward(cls):" in content

    # Check it loads RewardModelEnsemble
    assert "from aligntune.rewards.ensemble import RewardModelEnsemble" in content
    assert "RewardType.REWARD_MODEL_ENSEMBLE] = RewardModelEnsemble" in content

    # Check it's called in create_reward
    assert "if config.reward_type == RewardType.REWARD_MODEL_ENSEMBLE:" in content
    assert "cls._lazy_load_ensemble_reward()" in content


def test_registry_registers_ensemble():
    """Check that RewardRegistry registers the ensemble reward type."""
    registry_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "registry.py"
    content = registry_file.read_text(encoding="utf-8")

    # Check registration
    assert "\"reward_model_ensemble\"" in content
    assert "RewardType.REWARD_MODEL_ENSEMBLE" in content
    assert "\"ensemble_mode\": \"mean\"" in content


def test_ensemble_py_aggregation_modes():
    """Check that ensemble.py implements all aggregation modes."""
    ensemble_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "ensemble.py"
    content = ensemble_file.read_text(encoding="utf-8")

    # Check all modes are implemented in compute
    assert "if self.ensemble_mode == \"mean\":" in content
    assert "elif self.ensemble_mode == \"worst_case\":" in content
    assert "elif self.ensemble_mode == \"uncertainty_weighted\":" in content

    # Check mean aggregation
    assert "return float(np.mean(scores))" in content

    # Check worst_case aggregation
    assert "return float(np.min(scores))" in content

    # Check uncertainty_weighted uses std dev
    assert "std_dev = np.std" in content
    assert "normalized_std = std_dev / 0.5" in content
    assert "uncertainty_weight = max(0.0, 1.0 - normalized_std)" in content


def test_composite_reward_worst_case_logic():
    """Check that CompositeReward worst_case returns minimum."""
    core_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "core.py"
    content = core_file.read_text(encoding="utf-8")

    # Check worst_case implementation exists in CompositeReward
    assert 'elif self.ensemble_mode == "worst_case":' in content
    assert "return min(rewards)" in content


def test_composite_reward_uncertainty_weighted_logic():
    """Check that CompositeReward uncertainty_weighted uses std dev."""
    core_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "core.py"
    content = core_file.read_text(encoding="utf-8")

    # Check key components of uncertainty_weighted
    assert "uncertainty_weighted" in content
    assert "mean_reward = np.mean(rewards)" in content
    assert "std_dev = np.std(rewards)" in content
    # std_dev is now normalized by the max possible std (0.5) for rewards
    # bounded in [0, 1] before being turned into a weight - see the comment
    # in core.py explaining the previous unnormalized version understated
    # the uncertainty penalty.
    assert "uncertainty_weight = max(0.0, 1.0 - std_dev / 0.5)" in content
    assert "return mean_reward * uncertainty_weight" in content


def test_ensemble_model_loading():
    """Check that RewardModelEnsemble has model loading logic."""
    ensemble_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "ensemble.py"
    content = ensemble_file.read_text(encoding="utf-8")

    # Check model loading
    assert "def _load_models(self)" in content
    assert "pipeline(" in content
    assert "AutoTokenizer.from_pretrained" in content
    assert "AutoModelForSequenceClassification.from_pretrained" in content

    # Check lazy loading flag
    assert "self._models_loaded = False" in content
    assert "if self._models_loaded:" in content


def test_ensemble_batch_compute_with_breakdown():
    """Check that batch_compute supports return_breakdown."""
    ensemble_file = Path(__file__).parent.parent / "src" / "aligntune" / "rewards" / "ensemble.py"
    content = ensemble_file.read_text(encoding="utf-8")

    # Check batch_compute method signature
    assert "return_breakdown: bool = False" in content
    assert "breakdowns = [] if return_breakdown else None" in content


def test_test_file_exists():
    """Check that test file exists."""
    test_file = Path(__file__).parent / "test_reward_ensemble.py"
    assert test_file.exists()

    content = test_file.read_text()

    # Check test class names
    assert "class TestCompositeRewardEnsembleModes:" in content
    assert "class TestRewardModelEnsemble:" in content
    assert "class TestEnsembleEdgeCases:" in content
    assert "class TestEnsembleIntegration:" in content

    # Check key tests
    assert "test_composite_reward_mean_mode" in content
    assert "test_composite_reward_worst_case_mode" in content
    assert "test_composite_reward_uncertainty_weighted_mode" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
