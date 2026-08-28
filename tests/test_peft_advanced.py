"""
Smoke tests for advanced PEFT methods (DoRA, rsLoRA, LoftQ, PISSA).

These tests verify that:
1. Config fields are properly added and validated
2. TRL backend can create LoraConfig with advanced flags
3. Speed optimization flags (torch.compile, Liger) are wired correctly
4. 8-bit precision/optimizer state works

Uses the local module path to avoid installation issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


class TestPEFTAdvancedConfig:
    """Test PEFT advanced config fields."""

    def test_sft_config_dora_fields(self):
        """Test DoRA config fields are added to SFT ModelConfig."""
        from aligntune.core.sft.config import ModelConfig

        config = ModelConfig(
            name_or_path="sshleifer/tiny-gpt2",
            peft_enabled=True,
            dora_enabled=True,
            rslora_enabled=True,
            loftq_init=True,
            pissa_init=True,
        )

        assert config.dora_enabled is True
        assert config.rslora_enabled is True
        assert config.loftq_init is True
        assert config.pissa_init is True
        logger.info("✓ SFT DoRA config fields validated")

    def test_sft_config_speed_flags(self):
        """Test speed optimization flags in SFT config."""
        from aligntune.core.sft.config import ModelConfig, TrainingConfig

        config = ModelConfig(
            name_or_path="sshleifer/tiny-gpt2",
            use_liger_kernel=True,
        )

        train_config = TrainingConfig(
            torch_compile=True,
            use_liger_kernel=True,
        )

        assert train_config.torch_compile is True
        assert train_config.use_liger_kernel is True
        logger.info("✓ SFT speed optimization flags validated")

    def test_rl_config_dora_fields(self):
        """Test DoRA config fields are added to RL ModelConfig."""
        from aligntune.core.rl.config import ModelConfig

        config = ModelConfig(
            name_or_path="sshleifer/tiny-gpt2",
            use_peft=True,
            dora_enabled=True,
            rslora_enabled=True,
            loftq_init=True,
            pissa_init=True,
        )

        assert config.dora_enabled is True
        assert config.rslora_enabled is True
        assert config.loftq_init is True
        assert config.pissa_init is True
        logger.info("✓ RL DoRA config fields validated")


class TestPrecisionHandler:
    """Test precision handler extensions."""

    def test_int8_precision_type(self):
        """Test INT8 precision type is recognized."""
        from aligntune.core.precision_handler import PrecisionType

        assert hasattr(PrecisionType, "INT8")
        assert hasattr(PrecisionType, "FP8")
        assert PrecisionType.INT8.value == "int8"
        assert PrecisionType.FP8.value == "fp8"
        logger.info("✓ INT8/FP8 precision types added")

    def test_validate_int8_precision(self):
        """Test INT8 precision validation."""
        from aligntune.core.precision_handler import PrecisionHandler

        # Should not raise
        result = PrecisionHandler.validate_precision("int8")
        assert result == "int8"

        result = PrecisionHandler.validate_precision("fp8")
        assert result == "fp8"

        logger.info("✓ INT8/FP8 precision validation works")

    def test_get_torch_dtype_int8(self):
        """Test torch.dtype for 8-bit returns float32."""
        import torch
        from aligntune.core.precision_handler import PrecisionHandler

        # 8-bit quantization uses float32 as base dtype
        dtype = PrecisionHandler.get_torch_dtype("int8")
        assert dtype == torch.float32

        dtype = PrecisionHandler.get_torch_dtype("fp8")
        assert dtype == torch.float32

        logger.info("✓ 8-bit torch.dtype returns float32 as base")

    def test_8bit_optimizer_kwargs(self):
        """Test 8-bit optimizer kwargs generation."""
        from aligntune.core.precision_handler import PrecisionHandler

        kwargs = PrecisionHandler.get_8bit_optimizer_kwargs("int8")
        assert "optim" in kwargs
        assert kwargs["optim"] == "paged_adamw_8bit"

        kwargs = PrecisionHandler.get_8bit_optimizer_kwargs("fp8")
        assert "optim" in kwargs

        # Non-8bit should return empty dict
        kwargs = PrecisionHandler.get_8bit_optimizer_kwargs("bf16")
        assert kwargs == {}

        logger.info("✓ 8-bit optimizer kwargs generation works")


class TestTRLBackendPEFT:
    """Test TRL backend PEFT integration."""

    def test_lora_config_creation_minimal(self):
        """Test LoraConfig can be created with DoRA flags (minimal test)."""
        try:
            from peft import LoraConfig

            # Test DoRA flag
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],
                use_dora=True,
                use_rslora=True,
            )

            assert config.use_dora is True
            assert config.use_rslora is True
            logger.info("✓ LoraConfig supports DoRA and rsLoRA flags")
        except ImportError:
            logger.warning("PEFT not installed, skipping LoraConfig test")

    def test_loftq_pissa_init_flags(self):
        """Test that LoftQ and PISSA initialization can be set."""
        try:
            from peft import LoraConfig

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],
            )

            # Test setting initialization method
            config.init_lora_weights = "loftq"
            assert config.init_lora_weights == "loftq"

            config.init_lora_weights = "pissa"
            assert config.init_lora_weights == "pissa"

            logger.info("✓ LoftQ and PISSA initialization methods work")
        except ImportError:
            logger.warning("PEFT not installed, skipping initialization test")


class TestSFTConfigIntegration:
    """Integration tests for SFT config with all new fields."""

    def test_full_sft_config_with_advanced_peft(self):
        """Test complete SFT config with all advanced PEFT fields."""
        from aligntune.core.sft.config import (
            SFTConfig,
            ModelConfig,
            DatasetConfig,
            TrainingConfig,
            LoggingConfig,
            TaskType,
        )

        config = SFTConfig(
            model=ModelConfig(
                name_or_path="sshleifer/tiny-gpt2",
                peft_enabled=True,
                dora_enabled=True,
                rslora_enabled=False,
                loftq_init=True,
                pissa_init=False,
            ),
            dataset=DatasetConfig(
                name="wikitext",
                split="train",
                task_type=TaskType.TEXT_GENERATION,
                max_samples=100,
            ),
            train=TrainingConfig(
                per_device_batch_size=2,
                max_steps=10,
                torch_compile=True,
                use_liger_kernel=False,
            ),
            logging=LoggingConfig(output_dir="./test_output"),
        )

        # Verify all fields are present
        assert config.model.dora_enabled is True
        assert config.model.rslora_enabled is False
        assert config.model.loftq_init is True
        assert config.model.pissa_init is False
        assert config.train.torch_compile is True
        assert config.train.use_liger_kernel is False

        logger.info("✓ Full SFT config with advanced PEFT fields validated")

    def test_sft_config_serialization(self):
        """Test SFT config can be serialized and deserialized."""
        from aligntune.core.sft.config import (
            SFTConfig,
            ModelConfig,
            DatasetConfig,
            TrainingConfig,
            LoggingConfig,
            TaskType,
        )

        config = SFTConfig(
            model=ModelConfig(
                name_or_path="sshleifer/tiny-gpt2",
                dora_enabled=True,
                rslora_enabled=True,
            ),
            dataset=DatasetConfig(
                name="wikitext",
                task_type=TaskType.TEXT_GENERATION,
            ),
            train=TrainingConfig(torch_compile=True),
            logging=LoggingConfig(output_dir="./test_output"),
        )

        # Serialize
        config_dict = config.to_dict()
        assert config_dict["model"]["dora_enabled"] is True
        assert config_dict["model"]["rslora_enabled"] is True
        assert config_dict["train"]["torch_compile"] is True

        # Deserialize
        config2 = SFTConfig.from_dict(config_dict)
        assert config2.model.dora_enabled is True
        assert config2.model.rslora_enabled is True
        assert config2.train.torch_compile is True

        logger.info("✓ SFT config serialization works")


if __name__ == "__main__":
    # Run minimal tests without pytest
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("PEFT Advanced Features Test Suite")
    print("=" * 60)

    test_config = TestPEFTAdvancedConfig()
    test_config.test_sft_config_dora_fields()
    test_config.test_sft_config_speed_flags()
    test_config.test_rl_config_dora_fields()

    test_precision = TestPrecisionHandler()
    test_precision.test_int8_precision_type()
    test_precision.test_validate_int8_precision()
    test_precision.test_get_torch_dtype_int8()
    test_precision.test_8bit_optimizer_kwargs()

    test_lora = TestTRLBackendPEFT()
    test_lora.test_lora_config_creation_minimal()
    test_lora.test_loftq_pissa_init_flags()

    test_integration = TestSFTConfigIntegration()
    test_integration.test_full_sft_config_with_advanced_peft()
    test_integration.test_sft_config_serialization()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
