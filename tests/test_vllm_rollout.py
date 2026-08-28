"""
vLLM Rollout Backend Tests
Tests if vLLM rollout backend is working for ES trainer.
"""

import pytest


class TestVLLMRolloutAvailability:
    """Test vLLM rollout backend availability."""

    def test_vllm_import(self):
        """Test if vLLM rollout backend can be imported."""
        try:
            from aligntune.core.rollout import VLLMRolloutBackend, VLLM_AVAILABLE
            if VLLM_AVAILABLE:
                assert VLLMRolloutBackend is not None
            else:
                pytest.skip("vLLM not available")
        except ImportError as e:
            pytest.skip(f"vLLM import failed: {e}")

    def test_vllm_available_flag(self):
        """Test VLLM_AVAILABLE flag exists and is boolean."""
        from aligntune.core.rollout import VLLM_AVAILABLE
        assert isinstance(VLLM_AVAILABLE, bool)

    def test_base_rollout_backend_import(self):
        """Test base rollout backend imports."""
        from aligntune.core.rollout import BaseRolloutBackend
        assert BaseRolloutBackend is not None

    def test_hf_rollout_backend_import(self):
        """Test HuggingFace rollout backend imports."""
        from aligntune.core.rollout import HFRolloutBackend
        assert HFRolloutBackend is not None


class TestVLLMRolloutBackendCreation:
    """Test vLLM rollout backend creation."""

    def test_vllm_backend_creation(self):
        """Test creating vLLM rollout backend."""
        from aligntune.core.rollout import VLLMRolloutBackend
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        backend = VLLMRolloutBackend(
            model_name_or_path="Qwen/Qwen2.5-0.5B",
            tokenizer=tokenizer,
            dtype="auto",
            max_model_len=512,
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=8,
        )

        assert backend is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
