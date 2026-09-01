"""
End-to-end test for RoPE long context integration with different model families.

Tests RoPE scaling with:
- Different model families (Llama, Mistral, Qwen)
- Different RoPE types (linear, dynamic, yarn)
- QLoRA finetuning
- LongAlpaca dataset

Run directly: python test_rope_end_to_end.py
Run with pytest: pytest test_rope_end_to_end.py -v
"""

import os
import sys
import pytest
import logging
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aligntune.core.sft.config import (
    SFTConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    LoggingConfig,
    PeftConfigData,
)
from aligntune.core.long_context.rope_config import RopeConfig
from aligntune.backends.trl.sft.sft_generation import TRLSFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRoPEIntegration:
    """Test RoPE integration with different model families and configurations."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for each test."""
        temp_dir = tempfile.mkdtemp(prefix="rope_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def create_base_config(self, model_name, rope_config, output_dir):
        """Create base SFT configuration with QLoRA and RoPE."""
        return SFTConfig(
            model=ModelConfig(
                name_or_path=model_name,
                max_seq_length=4096,
                precision="auto",
                quantization={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                },
                peft=PeftConfigData(
                    enabled=True,
                    variant="standard",
                    rank=16,
                    alpha=32,
                    dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
                rope=rope_config,
                gradient_checkpointing=True,
                use_unsloth=False,
            ),
            dataset=DatasetConfig(
                name="Yukang/LongAlpaca-16k-length",
                train_split="train",
                max_samples=10,
                
                task_type="sft",
                instruction_column="instruction",
                output_column="output",
                auto_detect_fields=True,
            ),
            train=TrainingConfig(
                per_device_batch_size=1,
                gradient_accumulation_steps=4,
                max_steps=5,
                epochs=1,
                learning_rate=2e-4,
                warmup_steps=1,
                eval_interval=10,
                save_interval=10,
                bf16=True,
                gradient_checkpointing=True,
                seed=42,
            ),
            logging=LoggingConfig(
                output_dir=output_dir,
                run_name="rope_test",
                log_level="INFO",
                report_to="none",
            ),
        )

    def test_llama_linear_rope(self, temp_output_dir):
        """Test Llama model with linear RoPE scaling."""
        logger.info("Testing Llama with linear RoPE scaling")

        rope_config = RopeConfig(
            rope_type="linear",
            target_max_seq_length=32768,
        )

        config = self.create_base_config(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training with linear RoPE")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("Llama linear RoPE test passed")

    def test_mistral_dynamic_rope(self, temp_output_dir):
        """Test Mistral model with dynamic RoPE scaling."""
        logger.info("Testing Mistral with dynamic RoPE scaling")

        rope_config = RopeConfig(
            rope_type="dynamic",
            target_max_seq_length=16384,
        )

        config = self.create_base_config(
            model_name="mistralai/Mistral-7B-v0.1",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        config.dataset.max_samples = 5
        config.train.max_steps = 3

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training with dynamic RoPE")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("Mistral dynamic RoPE test passed")

    def test_qwen_yarn_rope(self, temp_output_dir):
        """Test Qwen model with YaRN RoPE scaling."""
        logger.info("Testing Qwen with YaRN RoPE scaling")

        rope_config = RopeConfig(
            rope_type="yarn",
            target_max_seq_length=32768,
            original_max_position_embeddings=8192,
        )

        config = self.create_base_config(
            model_name="Qwen/Qwen2-0.5B",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        config.dataset.max_samples = 5
        config.train.max_steps = 3

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training with YaRN RoPE")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("Qwen YaRN RoPE test passed")

    def test_no_rope_baseline(self, temp_output_dir):
        """Test baseline training without RoPE scaling."""
        logger.info("Testing baseline without RoPE scaling")

        rope_config = RopeConfig(rope_type=None)

        config = self.create_base_config(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        config.dataset.max_samples = 5
        config.train.max_steps = 3

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training without RoPE")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("Baseline no RoPE test passed")

    def test_rope_error_fallback(self, temp_output_dir):
        """Test that invalid RoPE config gracefully falls back."""
        logger.info("Testing RoPE error fallback")

        rope_config = RopeConfig(
            rope_type="yarn",
            factor=2.0,
            original_max_position_embeddings=2048,
        )

        config = self.create_base_config(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        config.dataset.max_samples = 5
        config.train.max_steps = 3

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training with potentially problematic RoPE config")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("RoPE error fallback test passed")

    def test_default_rope_type(self, temp_output_dir):
        """Test default RoPE type (no scaling)."""
        logger.info("Testing default RoPE type")

        rope_config = RopeConfig(rope_type="default")

        config = self.create_base_config(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            rope_config=rope_config,
            output_dir=temp_output_dir,
        )

        config.dataset.max_samples = 5
        config.train.max_steps = 3

        trainer = TRLSFTTrainer(config)
        assert trainer is not None, "Trainer creation failed"

        logger.info("Starting training with default RoPE")
        trainer.train()

        assert os.path.exists(temp_output_dir), "Output directory not created"
        logger.info("Default RoPE type test passed")


class TestRoPEConfigValidation:
    """Test RoPE configuration validation and parameter extraction."""

    def test_auto_factor_calculation(self):
        """Test automatic factor calculation from target length."""
        from aligntune.core.long_context.rope_config import extract_rope_parameters
        from transformers import AutoConfig

        rope_config = RopeConfig(
            rope_type="linear",
            target_max_seq_length=16384,
        )

        model_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        params = extract_rope_parameters(
            rope_config=rope_config,
            model_config=model_config,
            max_seq_length=16384,
        )

        assert params is not None, "RoPE parameters should not be None"
        assert "factor" in params, "Factor should be auto-calculated"
        assert params["factor"] > 1.0, "Factor should be greater than 1"
        logger.info(f"Auto-calculated factor: {params['factor']}")

    def test_manual_factor_usage(self):
        """Test manual factor specification."""
        from aligntune.core.long_context.rope_config import extract_rope_parameters
        from transformers import AutoConfig

        rope_config = RopeConfig(
            rope_type="linear",
            factor=4.0,
        )

        model_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        params = extract_rope_parameters(
            rope_config=rope_config,
            model_config=model_config,
            max_seq_length=8192,
        )

        assert params is not None, "RoPE parameters should not be None"
        assert params["factor"] == 4.0, "Factor should match specified value"
        logger.info(f"Manual factor used: {params['factor']}")

    def test_yarn_specific_params(self):
        """Test YaRN-specific parameter handling."""
        from aligntune.core.long_context.rope_config import extract_rope_parameters
        from transformers import AutoConfig

        rope_config = RopeConfig(
            rope_type="yarn",
            target_max_seq_length=32768,
            original_max_position_embeddings=4096,
            attention_factor=1.0,
            beta_fast=32,
            beta_slow=1,
        )

        model_config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        params = extract_rope_parameters(
            rope_config=rope_config,
            model_config=model_config,
            max_seq_length=32768,
        )

        assert params is not None, "RoPE parameters should not be None"
        assert "original_max_position_embeddings" in params, "YaRN requires original max"
        assert params.get("attention_factor") == 1.0, "YaRN attention_factor should be set"
        logger.info(f"YaRN parameters: {params}")

    def test_unsupported_model_handling(self):
        """Test handling of models without RoPE support."""
        from aligntune.core.long_context.rope_config import extract_rope_parameters

        rope_config = RopeConfig(
            rope_type="linear",
            target_max_seq_length=8192,
        )

        class MockConfig:
            """Mock config without RoPE support."""
            model_type = "mock"

        model_config = MockConfig()

        params = extract_rope_parameters(
            rope_config=rope_config,
            model_config=model_config,
            max_seq_length=8192,
        )

        assert params is None, "Should return None for unsupported models"
        logger.info("Unsupported model correctly handled")


def run_quick_test():
    """Run a quick smoke test with minimal resources."""
    logger.info("=" * 80)
    logger.info("Running quick RoPE integration smoke test")
    logger.info("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="rope_quick_test_")

    try:
        rope_config = RopeConfig(
            rope_type="linear",
            target_max_seq_length=8192,
        )

        config = SFTConfig(
            model=ModelConfig(
                name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_seq_length=4096,
                precision="auto",
                quantization={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_quant_type": "nf4",
                },
                peft=PeftConfigData(
                    enabled=True,
                    rank=8,
                    alpha=16,
                ),
                rope=rope_config,
                use_unsloth=False,
            ),
            dataset=DatasetConfig(
                name="Yukang/LongAlpaca-16k-length",
                train_split="train",
                max_samples=3,
                task_type="sft",
                auto_detect_fields=True,
            ),
            train=TrainingConfig(
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
                max_steps=2,
                epochs=1,
                learning_rate=2e-4,
                bf16=True,
                seed=42,
            ),
            logging=LoggingConfig(
                output_dir=temp_dir,
                run_name="rope_quick_test",
                log_level="INFO",
                report_to="none",
            ),
        )

        logger.info("Creating trainer with RoPE configuration")
        trainer = TRLSFTTrainer(config)

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

    parser = argparse.ArgumentParser(description="Test RoPE integration")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test only",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (e.g., test_llama_linear_rope)",
    )
    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.test:
        pytest.main([__file__, f"-k", args.test, "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])
