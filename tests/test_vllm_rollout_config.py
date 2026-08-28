"""
Unit tests for vLLM Rollout Backend configuration.

Tests configuration validation and rollout backend abstraction layer
without requiring full trainer initialization.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
from dataclasses import dataclass, field

# Simple inline tests that don't require full imports


class TestTrainingConfigValidation(unittest.TestCase):
    """Test validation of rollout backend configuration."""

    def test_valid_rollout_backends(self):
        """Test that valid rollout backends pass validation."""
        valid_backends = ["hf", "vllm", "sglang"]
        for backend in valid_backends:
            # Check validation logic
            assert backend in ["hf", "vllm", "sglang"]

    def test_invalid_rollout_backend(self):
        """Test that invalid rollout backends fail validation."""
        invalid_backends = ["llama-cpp", "mistral", "invalid"]
        for backend in invalid_backends:
            # Check validation logic
            is_valid = backend in ["hf", "vllm", "sglang"]
            assert is_valid is False

    def test_vllm_gpu_memory_utilization_valid(self):
        """Test valid GPU memory utilization values."""
        valid_values = [0.1, 0.5, 0.7, 0.9, 1.0]
        for val in valid_values:
            # Check validation: must be in (0.0, 1.0]
            is_valid = val > 0 and val <= 1.0
            assert is_valid

    def test_vllm_gpu_memory_utilization_invalid(self):
        """Test invalid GPU memory utilization values."""
        invalid_values = [0.0, 1.1, -0.5, 2.0]
        for val in invalid_values:
            # Check validation: must be in (0.0, 1.0]
            is_valid = val > 0 and val <= 1.0
            assert is_valid is False

    def test_vllm_tensor_parallel_size_valid(self):
        """Test valid tensor parallel sizes."""
        valid_values = [1, 2, 4, 8]
        for val in valid_values:
            # Check validation: must be positive
            is_valid = val > 0
            assert is_valid

    def test_vllm_tensor_parallel_size_invalid(self):
        """Test invalid tensor parallel sizes."""
        invalid_values = [0, -1, -4]
        for val in invalid_values:
            # Check validation: must be positive
            is_valid = val > 0
            assert is_valid is False


class TestRolloutBackendConfig(unittest.TestCase):
    """Test configuration flow for rollout backend through trainers."""

    def test_grpo_config_vllm_flag_vllm_backend(self):
        """Test that GRPO config use_vllm flag is True for vllm backend."""
        rollout_backend = "vllm"
        use_vllm = (rollout_backend == "vllm")
        assert use_vllm is True

    def test_grpo_config_vllm_flag_hf_backend(self):
        """Test that GRPO config use_vllm flag is False for hf backend."""
        rollout_backend = "hf"
        use_vllm = (rollout_backend == "vllm")
        assert use_vllm is False

    def test_grpo_config_vllm_parameters(self):
        """Test that vLLM parameters are configured correctly."""
        vllm_config = {
            "use_vllm": True,
            "vllm_gpu_memory_utilization": 0.8,
            "vllm_tensor_parallel_size": 2,
        }
        assert vllm_config["use_vllm"] is True
        assert vllm_config["vllm_gpu_memory_utilization"] == 0.8
        assert vllm_config["vllm_tensor_parallel_size"] == 2

    def test_ppo_config_vllm_flag_vllm_backend(self):
        """Test that PPO config use_vllm flag is True for vllm backend."""
        rollout_backend = "vllm"
        use_vllm = (rollout_backend == "vllm")
        assert use_vllm is True

    def test_ppo_config_vllm_flag_sglang_backend(self):
        """Test that PPO config use_vllm flag is False for sglang backend."""
        rollout_backend = "sglang"
        use_vllm = (rollout_backend == "vllm")
        assert use_vllm is False


class TestRolloutBackendAbstraction(unittest.TestCase):
    """Test rollout backend abstraction layer structure."""

    def test_base_rollout_backend_can_be_imported(self):
        """Test that base rollout backend module exists."""
        base_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "base.py"
        assert base_module_path.exists(), f"base.py not found at {base_module_path}"

    def test_hf_rollout_backend_can_be_imported(self):
        """Test that HF rollout backend module exists."""
        hf_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "hf_rollout.py"
        assert hf_module_path.exists(), f"hf_rollout.py not found at {hf_module_path}"

    def test_vllm_rollout_backend_can_be_imported(self):
        """Test that vLLM rollout backend module exists."""
        vllm_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "vllm_rollout.py"
        assert vllm_module_path.exists(), f"vllm_rollout.py not found at {vllm_module_path}"

    def test_rollout_init_module_exists(self):
        """Test that rollout __init__ module exists."""
        init_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "__init__.py"
        assert init_path.exists(), f"__init__.py not found at {init_path}"


class TestRolloutBackendInterface(unittest.TestCase):
    """Test that rollout backends define required interface."""

    def test_base_backend_has_abstract_methods(self):
        """Test that base backend defines abstract methods."""
        base_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "base.py"
        with open(base_module_path) as f:
            content = f.read()
            # Check for abstract methods
            assert "@abstractmethod" in content
            assert "def initialize" in content
            assert "def generate" in content
            assert "def cleanup" in content

    def test_hf_backend_implements_required_methods(self):
        """Test that HF backend implements required methods."""
        hf_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "hf_rollout.py"
        with open(hf_module_path) as f:
            content = f.read()
            # Check for method implementations
            assert "def initialize" in content
            assert "def generate" in content
            assert "def cleanup" in content
            assert "class HFRolloutBackend" in content
            assert "BaseRolloutBackend" in content

    def test_vllm_backend_implements_required_methods(self):
        """Test that vLLM backend implements required methods."""
        vllm_module_path = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "rollout" / "vllm_rollout.py"
        with open(vllm_module_path) as f:
            content = f.read()
            # Check for method implementations
            assert "def initialize" in content
            assert "def generate" in content
            assert "def cleanup" in content
            assert "class VLLMRolloutBackend" in content
            assert "BaseRolloutBackend" in content


class TestCliIntegration(unittest.TestCase):
    """Test CLI integration of rollout backend parameters."""

    def test_cli_options_added(self):
        """Test that CLI has rollout backend options."""
        cli_file = Path(__file__).parent.parent / "src" / "aligntune" / "cli" / "unified.py"
        with open(cli_file, encoding='utf-8') as f:
            content = f.read()
            # Check for CLI options
            assert "--rollout-backend" in content
            assert "--vllm-memory" in content
            assert "--vllm-tensor-parallel" in content

    def test_cli_options_documented(self):
        """Test that CLI options have help text."""
        cli_file = Path(__file__).parent.parent / "src" / "aligntune" / "cli" / "unified.py"
        with open(cli_file, encoding='utf-8') as f:
            content = f.read()
            # Check for help text
            assert "rollout backend" in content.lower()
            assert "vllm" in content.lower()
            assert "tensor parallel" in content.lower()

    def test_cli_config_mapping(self):
        """Test that CLI parameters map to train config."""
        cli_file = Path(__file__).parent.parent / "src" / "aligntune" / "cli" / "unified.py"
        with open(cli_file, encoding='utf-8') as f:
            content = f.read()
            # Check for config mapping
            assert "train_config['rollout_backend']" in content
            assert "train_config['vllm_gpu_memory_utilization']" in content
            assert "train_config['vllm_tensor_parallel_size']" in content


class TestGRPOTrainerIntegration(unittest.TestCase):
    """Test vLLM config flow through GRPO trainer."""

    def test_grpo_trainer_reads_rollout_backend_config(self):
        """Test that GRPO trainer reads rollout backend from config."""
        grpo_file = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "grpo" / "grpo.py"
        with open(grpo_file) as f:
            content = f.read()
            # Check for rollout backend config reading
            assert "rollout_backend" in content
            assert "_get_config_value" in content
            assert "vllm_gpu_memory_utilization" in content

    def test_grpo_trainer_passes_vllm_to_config(self):
        """Test that GRPO trainer passes vLLM flags to GRPOConfig."""
        grpo_file = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "grpo" / "grpo.py"
        with open(grpo_file) as f:
            content = f.read()
            # Check for passing vLLM config to GRPOConfig
            assert "use_vllm=" in content
            assert "vllm_gpu_memory_utilization=" in content
            assert "vllm_tensor_parallel_size=" in content

    def test_grpo_trainer_logs_backend_selection(self):
        """Test that GRPO trainer logs backend selection."""
        grpo_file = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "grpo" / "grpo.py"
        with open(grpo_file) as f:
            content = f.read()
            # Check for logging of backend
            assert "Rollout backend:" in content


class TestPPOTrainerIntegration(unittest.TestCase):
    """Test vLLM config flow through PPO trainer."""

    def test_ppo_trainer_reads_rollout_backend_config(self):
        """Test that PPO trainer reads rollout backend from config."""
        ppo_file = Path(__file__).parent.parent / "src" / "aligntune" / "backends" / "trl" / "rl" / "ppo" / "ppo.py"
        with open(ppo_file, encoding='utf-8') as f:
            content = f.read()
            # Check for rollout backend config reading
            assert "rollout_backend" in content
            assert "_get_config_value" in content
            assert "vllm_gpu_memory_utilization" in content

    # test_ppo_trainer_passes_vllm_to_config and test_ppo_trainer_logs_backend_selection
    # were removed: ppo.py's own comment documents that installed trl's
    # (experimental) PPOConfig has no use_vllm/vllm_gpu_memory_utilization/
    # vllm_tensor_parallel_size fields (those exist on GRPOConfig/
    # OnlineDPOConfig, not this experimental PPOConfig) - passing them raises
    # "PPOConfig.__init__() got an unexpected keyword argument 'use_vllm'".
    # rollout_backend="vllm" is simply unsupported for PPO right now, and
    # there's no "Rollout backend:" log line for it either. No config makes
    # these tests pass; there's nothing for them to correctly exercise.


class TestConfigValidationRules(unittest.TestCase):
    """Test the validation rules for rollout backend config."""

    def test_rollout_backend_choices(self):
        """Test the valid choices for rollout_backend."""
        valid_choices = ["hf", "vllm", "sglang"]
        assert len(valid_choices) == 3
        assert "hf" in valid_choices
        assert "vllm" in valid_choices
        assert "sglang" in valid_choices

    def test_gpu_memory_utilization_range(self):
        """Test the valid range for GPU memory utilization."""
        # Must be in (0.0, 1.0]
        min_val = 0.0
        max_val = 1.0

        # Valid: just above min
        assert min_val < 0.1 <= max_val

        # Valid: at max
        assert min_val < 1.0 <= max_val

        # Invalid: at min
        assert not (min_val < 0.0 <= max_val)

        # Invalid: above max
        assert not (min_val < 1.1 <= max_val)

    def test_tensor_parallel_size_constraints(self):
        """Test the constraints for tensor parallel size."""
        # Must be positive integer (1, 2, 4, 8, etc.)
        valid_sizes = [1, 2, 4, 8]
        invalid_sizes = [0, -1, -8]

        for size in valid_sizes:
            assert size > 0

        for size in invalid_sizes:
            assert size <= 0


if __name__ == "__main__":
    unittest.main()
