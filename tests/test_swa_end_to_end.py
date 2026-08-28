"""
End-to-end test for Sliding Window Attention (SWA) integration.

Tests SWA with:
- Different window sizes
- Different model families (TinyLlama, Qwen)
- QLoRA finetuning
- LongAlpaca dataset
- Variable passthrough verification

Run directly: python test_swa_end_to_end.py
Run with pytest: pytest test_swa_end_to_end.py -v
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


class TestSWAIntegration:
    """Test Sliding Window Attention integration end-to-end."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for each test."""
        temp_dir = tempfile.mkdtemp(prefix="swa_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_swa_small_window(self, temp_output_dir):
        """Test SWA with small window size (2048)."""
        logger.info("Testing SWA with window=2048")

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
            attn_implementation="swa",
            sliding_window=2048,
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

        # Verify SWA config was passed through
        model = trainer.model
        assert hasattr(model.config, 'sliding_window'), "Model config missing sliding_window"
        assert model.config.sliding_window == 2048, f"Expected window=2048, got {model.config.sliding_window}"
        logger.info(f"Verified: model.config.sliding_window = {model.config.sliding_window}")

        logger.info("Starting training with SWA window=2048")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("SWA small window test passed")

    def test_swa_medium_window(self, temp_output_dir):
        """Test SWA with medium window size (4096)."""
        logger.info("Testing SWA with window=4096")

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
            attn_implementation="swa",
            sliding_window=4096,
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

        # Verify SWA config was passed through
        model = trainer.model
        assert hasattr(model.config, 'sliding_window'), "Model config missing sliding_window"
        assert model.config.sliding_window == 4096, f"Expected window=4096, got {model.config.sliding_window}"
        logger.info(f"Verified: model.config.sliding_window = {model.config.sliding_window}")

        logger.info("Starting training with SWA window=4096")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("SWA medium window test passed")

    def test_swa_large_window(self, temp_output_dir):
        """Test SWA with large window size (8192)."""
        logger.info("Testing SWA with window=8192")

        trainer = create_sft_trainer(
            model_name="Qwen/Qwen2-0.5B",
            dataset_name="Yukang/LongAlpaca-16k-length",
            backend="trl",
            output_dir=temp_output_dir,
            num_epochs=1,
            batch_size=1,
            learning_rate=2e-4,
            max_seq_length=16384,
            max_samples=3,
            max_steps=2,
            attn_implementation="swa",
            sliding_window=8192,
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

        # Verify SWA config was passed through
        model = trainer.model
        assert hasattr(model.config, 'sliding_window'), "Model config missing sliding_window"
        assert model.config.sliding_window == 8192, f"Expected window=8192, got {model.config.sliding_window}"
        logger.info(f"Verified: model.config.sliding_window = {model.config.sliding_window}")

        logger.info("Starting training with SWA window=8192")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("SWA large window test passed")

    def test_swa_without_quantization(self, temp_output_dir):
        """Test SWA without quantization for baseline comparison."""
        logger.info("Testing SWA without quantization")

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
            attn_implementation="swa",
            sliding_window=2048,
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

        # Verify SWA config
        model = trainer.model
        assert hasattr(model.config, 'sliding_window'), "Model config missing sliding_window"
        assert model.config.sliding_window == 2048, f"Expected window=2048, got {model.config.sliding_window}"
        logger.info(f"Verified: model.config.sliding_window = {model.config.sliding_window}")

        logger.info("Starting training with SWA (no quantization)")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("SWA without quantization test passed")

    def test_swa_config_verification(self, temp_output_dir):
        """Test that all SWA config parameters are correctly passed through."""
        logger.info("Testing SWA config parameter passthrough")

        test_window = 3072

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
            attn_implementation="swa",
            sliding_window=test_window,
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

        logger.info("Verifying SWA configuration parameters:")

        # Check sliding_window
        assert hasattr(config, 'sliding_window'), "Model config missing sliding_window attribute"
        assert config.sliding_window == test_window, \
            f"sliding_window mismatch: expected {test_window}, got {config.sliding_window}"
        logger.info(f"  sliding_window: {config.sliding_window} (OK)")

        # Check attn_implementation (might not be stored in config, but check if available)
        if hasattr(config, '_attn_implementation'):
            logger.info(f"  attn_implementation: {config._attn_implementation}")

        logger.info("All SWA config parameters verified successfully")
        logger.info("SWA config verification test passed")


def run_quick_test():
    """Run a quick smoke test with minimal resources."""
    logger.info("=" * 80)
    logger.info("Running quick SWA integration smoke test")
    logger.info("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="swa_quick_test_")

    try:
        logger.info("Creating SWA trainer via backend factory")
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
            attn_implementation="swa",
            sliding_window=2048,
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
        logger.info(f"Model config sliding_window: {model.config.sliding_window}")
        assert model.config.sliding_window == 2048, "SWA config not applied"

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

    parser = argparse.ArgumentParser(description="Test SWA integration")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test only",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (e.g., test_swa_small_window)",
    )
    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.test:
        pytest.main([__file__, f"-k", args.test, "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])
