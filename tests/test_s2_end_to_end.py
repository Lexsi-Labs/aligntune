"""
End-to-end test for S2-Attention integration.

Tests S2-Attention with:
- Different group size ratios
- Different shift ratios
- Different minimum sequence lengths
- Different model families (TinyLlama, Qwen)
- QLoRA finetuning
- LongAlpaca dataset
- Variable passthrough verification

Run directly: python test_s2_end_to_end.py
Run with pytest: pytest test_s2_end_to_end.py -v
"""

import os
import sys
import pytest
import logging
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aligntune.core.backend_factory import create_sft_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestS2AttentionIntegration:
    """Test S2-Attention integration end-to-end."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for each test."""
        temp_dir = tempfile.mkdtemp(prefix="s2_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_s2_default_config(self, temp_output_dir):
        """Test S2-Attention with default configuration."""
        logger.info("Testing S2-Attention with default config")

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=4096,
            max_samples=5,
            max_steps=3,
            attn_implementation="s2",
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
            },
            use_peft=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify S2 config was passed through with defaults
        model = trainer.model
        config = model.config

        assert hasattr(config, 's2_group_size_ratio'), "Model config missing s2_group_size_ratio"
        assert config.s2_group_size_ratio == 0.25, \
            f"Expected s2_group_size_ratio=0.25, got {config.s2_group_size_ratio}"

        assert hasattr(config, 's2_min_seq_length'), "Model config missing s2_min_seq_length"
        assert config.s2_min_seq_length == 64, \
            f"Expected s2_min_seq_length=64, got {config.s2_min_seq_length}"

        assert hasattr(config, 's2_shift_ratio'), "Model config missing s2_shift_ratio"
        assert config.s2_shift_ratio == 0.5, \
            f"Expected s2_shift_ratio=0.5, got {config.s2_shift_ratio}"

        logger.info(f"Verified: s2_group_size_ratio = {config.s2_group_size_ratio}")
        logger.info(f"Verified: s2_min_seq_length = {config.s2_min_seq_length}")
        logger.info(f"Verified: s2_shift_ratio = {config.s2_shift_ratio}")

        logger.info("Starting training with S2-Attention (default config)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 default config test passed")

    def test_s2_small_groups(self, temp_output_dir):
        """Test S2-Attention with small groups (aggressive memory saving)."""
        logger.info("Testing S2-Attention with small groups (ratio=0.125)")

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=8192,
            max_samples=5,
            max_steps=3,
            attn_implementation="s2",
            s2_group_size_ratio=0.125,
            s2_shift_ratio=0.75,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify S2 config
        model = trainer.model
        config = model.config

        assert config.s2_group_size_ratio == 0.125, \
            f"Expected s2_group_size_ratio=0.125, got {config.s2_group_size_ratio}"
        assert config.s2_shift_ratio == 0.75, \
            f"Expected s2_shift_ratio=0.75, got {config.s2_shift_ratio}"

        logger.info(f"Verified: s2_group_size_ratio = {config.s2_group_size_ratio}")
        logger.info(f"Verified: s2_shift_ratio = {config.s2_shift_ratio}")

        logger.info("Starting training with S2-Attention (small groups)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 small groups test passed")

    def test_s2_large_groups(self, temp_output_dir):
        """Test S2-Attention with large groups (less aggressive)."""
        logger.info("Testing S2-Attention with large groups (ratio=0.5)")

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=8192,
            max_samples=5,
            max_steps=3,
            attn_implementation="s2",
            s2_group_size_ratio=0.5,
            s2_shift_ratio=0.25,
            s2_min_seq_length=128,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify S2 config
        model = trainer.model
        config = model.config

        assert config.s2_group_size_ratio == 0.5, \
            f"Expected s2_group_size_ratio=0.5, got {config.s2_group_size_ratio}"
        assert config.s2_shift_ratio == 0.25, \
            f"Expected s2_shift_ratio=0.25, got {config.s2_shift_ratio}"
        assert config.s2_min_seq_length == 128, \
            f"Expected s2_min_seq_length=128, got {config.s2_min_seq_length}"

        logger.info(f"Verified: s2_group_size_ratio = {config.s2_group_size_ratio}")
        logger.info(f"Verified: s2_shift_ratio = {config.s2_shift_ratio}")
        logger.info(f"Verified: s2_min_seq_length = {config.s2_min_seq_length}")

        logger.info("Starting training with S2-Attention (large groups)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 large groups test passed")

    def test_s2_different_model(self, temp_output_dir):
        """Test S2-Attention with different model (Qwen)."""
        logger.info("Testing S2-Attention with Qwen model")

        trainer = create_sft_trainer(
            model_name="Qwen/Qwen2-0.5B",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=8192,
            max_samples=3,
            max_steps=2,
            attn_implementation="s2",
            s2_group_size_ratio=0.25,
            s2_min_seq_length=64,
            s2_shift_ratio=0.5,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify S2 config
        model = trainer.model
        config = model.config

        assert hasattr(config, 's2_group_size_ratio'), "Model config missing s2_group_size_ratio"
        assert hasattr(config, 's2_min_seq_length'), "Model config missing s2_min_seq_length"
        assert hasattr(config, 's2_shift_ratio'), "Model config missing s2_shift_ratio"

        logger.info(f"Verified: s2_group_size_ratio = {config.s2_group_size_ratio}")
        logger.info(f"Verified: s2_min_seq_length = {config.s2_min_seq_length}")
        logger.info(f"Verified: s2_shift_ratio = {config.s2_shift_ratio}")

        logger.info("Starting training with S2-Attention (Qwen model)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 different model test passed")

    def test_s2_custom_min_seq_length(self, temp_output_dir):
        """Test S2-Attention with custom minimum sequence length."""
        logger.info("Testing S2-Attention with custom min_seq_length=256")

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=8192,
            max_samples=5,
            max_steps=3,
            attn_implementation="s2",
            s2_group_size_ratio=0.25,
            s2_min_seq_length=256,
            s2_shift_ratio=0.5,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify custom min_seq_length
        model = trainer.model
        config = model.config

        assert config.s2_min_seq_length == 256, \
            f"Expected s2_min_seq_length=256, got {config.s2_min_seq_length}"

        logger.info(f"Verified: s2_min_seq_length = {config.s2_min_seq_length}")

        logger.info("Starting training with S2-Attention (custom min_seq_length)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 custom min_seq_length test passed")

    def test_s2_config_verification(self, temp_output_dir):
        """Test that all S2 config parameters are correctly passed through."""
        logger.info("Testing S2 config parameter passthrough")

        test_group_ratio = 0.3
        test_min_seq = 100
        test_shift_ratio = 0.6

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            max_seq_length=8192,
            max_samples=3,
            max_steps=2,
            attn_implementation="s2",
            s2_group_size_ratio=test_group_ratio,
            s2_min_seq_length=test_min_seq,
            s2_shift_ratio=test_shift_ratio,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Detailed config verification
        model = trainer.model
        config = model.config

        logger.info("Verifying S2 configuration parameters:")

        # Check s2_group_size_ratio
        assert hasattr(config, 's2_group_size_ratio'), "Model config missing s2_group_size_ratio attribute"
        assert abs(config.s2_group_size_ratio - test_group_ratio) < 0.001, \
            f"s2_group_size_ratio mismatch: expected {test_group_ratio}, got {config.s2_group_size_ratio}"
        logger.info(f"  s2_group_size_ratio: {config.s2_group_size_ratio} (OK)")

        # Check s2_min_seq_length
        assert hasattr(config, 's2_min_seq_length'), "Model config missing s2_min_seq_length attribute"
        assert config.s2_min_seq_length == test_min_seq, \
            f"s2_min_seq_length mismatch: expected {test_min_seq}, got {config.s2_min_seq_length}"
        logger.info(f"  s2_min_seq_length: {config.s2_min_seq_length} (OK)")

        # Check s2_shift_ratio
        assert hasattr(config, 's2_shift_ratio'), "Model config missing s2_shift_ratio attribute"
        assert abs(config.s2_shift_ratio - test_shift_ratio) < 0.001, \
            f"s2_shift_ratio mismatch: expected {test_shift_ratio}, got {config.s2_shift_ratio}"
        logger.info(f"  s2_shift_ratio: {config.s2_shift_ratio} (OK)")

        logger.info("All S2 config parameters verified successfully")
        logger.info("S2 config verification test passed")

    def test_s2_without_quantization(self, temp_output_dir):
        """Test S2-Attention without quantization for baseline comparison."""
        logger.info("Testing S2-Attention without quantization")

        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=4096,
            max_samples=3,
            max_steps=2,
            attn_implementation="s2",
            s2_group_size_ratio=0.25,
            s2_min_seq_length=64,
            s2_shift_ratio=0.5,
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            gradient_checkpointing=True,
            bf16=True,
            seed=42,
            report_to="none",
        )

        assert trainer is not None, "Trainer creation failed"

        # Setup model to verify config
        trainer.setup_model()

        # Verify S2 config
        model = trainer.model
        config = model.config

        assert hasattr(config, 's2_group_size_ratio'), "Model config missing s2_group_size_ratio"
        assert hasattr(config, 's2_min_seq_length'), "Model config missing s2_min_seq_length"
        assert hasattr(config, 's2_shift_ratio'), "Model config missing s2_shift_ratio"

        logger.info(f"Verified: s2_group_size_ratio = {config.s2_group_size_ratio}")
        logger.info(f"Verified: s2_min_seq_length = {config.s2_min_seq_length}")
        logger.info(f"Verified: s2_shift_ratio = {config.s2_shift_ratio}")

        logger.info("Starting training with S2-Attention (no quantization)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("S2 without quantization test passed")


def run_quick_test():
    """Run a quick smoke test with minimal resources."""
    logger.info("=" * 80)
    logger.info("Running quick S2-Attention integration smoke test")
    logger.info("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="s2_quick_test_")

    try:
        logger.info("Creating S2-Attention trainer via backend factory")
        trainer = create_sft_trainer(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=4096,
            max_samples=3,
            max_steps=2,
            attn_implementation="s2",
            s2_group_size_ratio=0.25,
            s2_min_seq_length=64,
            s2_shift_ratio=0.5,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            bf16=True,
            seed=42,
            report_to="none",
        )

        # Setup model to verify config
        trainer.setup_model()

        # Verify config
        model = trainer.model
        config = model.config
        logger.info(f"Model config s2_group_size_ratio: {config.s2_group_size_ratio}")
        logger.info(f"Model config s2_min_seq_length: {config.s2_min_seq_length}")
        logger.info(f"Model config s2_shift_ratio: {config.s2_shift_ratio}")

        assert config.s2_group_size_ratio == 0.25, "S2 group_size_ratio config not applied"
        assert config.s2_min_seq_length == 64, "S2 min_seq_length config not applied"
        assert config.s2_shift_ratio == 0.5, "S2 shift_ratio config not applied"

        logger.info("Starting quick training run")
        trainer.train()

        logger.info("=" * 80)
        logger.info("Quick test PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Quick test FAILED: {e}")
        raise
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test S2-Attention integration")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test only",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (e.g., test_s2_default_config)",
    )
    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.test:
        pytest.main([__file__, f"-k", args.test, "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])
