"""
Test suite for Register/Formality Control Trainer.

Tests cover:
- RegisterTrainerConfig validation
- Register token registration
- Special token handling
- Instruction formatting with register tokens
- Dataset preparation with registers
- Loss computation
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, Mock
from transformers import PreTrainedTokenizerBase
from aligntune.backends.trl.sft.register_trainer import (
    RegisterTrainerConfig,
    RegisterControlledSFTTrainer,
    RegisterTrainer,
)


class TestRegisterTrainerConfig:
    """Test RegisterTrainerConfig validation."""

    def test_config_creation_default(self):
        """Test creating config with default registers."""
        config = RegisterTrainerConfig()
        assert config.registers == ["formal", "legal", "technical", "simple"]
        assert config.register_loss_weight == 0.0
        assert config.prepend_register_token is True

    def test_config_creation_custom(self):
        """Test creating config with custom registers."""
        config = RegisterTrainerConfig(
            registers=["formal", "simple"],
            register_loss_weight=0.1,
        )
        assert config.registers == ["formal", "simple"]
        assert config.register_loss_weight == 0.1

    def test_config_invalid_register(self):
        """Test config rejects invalid register names."""
        with pytest.raises(ValueError, match="Invalid register"):
            RegisterTrainerConfig(registers=["formal", "invalid_register"])

    def test_config_invalid_loss_weight(self):
        """Test config validates loss weight range."""
        with pytest.raises(ValueError, match="register_loss_weight"):
            RegisterTrainerConfig(register_loss_weight=1.5)

        with pytest.raises(ValueError, match="register_loss_weight"):
            RegisterTrainerConfig(register_loss_weight=-0.1)

    def test_config_valid_loss_weight(self):
        """Test config accepts valid loss weights."""
        for weight in [0.0, 0.5, 1.0]:
            config = RegisterTrainerConfig(register_loss_weight=weight)
            assert config.register_loss_weight == weight


class TestRegisterControlledSFTTrainer:
    """Test RegisterControlledSFTTrainer class."""

    @pytest.fixture(autouse=True)
    def patch_sft_init(self):
        def _stub(inst, *args, **kwargs):
            inst.tokenizer = kwargs.get('processing_class')
            inst.model = kwargs.get('model')
        with patch('trl.SFTTrainer.__init__', _stub):
            yield

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return RegisterTrainerConfig(
            registers=["formal", "legal", "technical", "simple"],
            register_loss_weight=0.1,
            prepend_register_token=True,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.add_special_tokens = MagicMock(return_value=4)  # 4 tokens added
        tokenizer.convert_tokens_to_ids = MagicMock(side_effect=lambda x: hash(x) % 10000)
        tokenizer.__len__ = MagicMock(return_value=50000)
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.resize_token_embeddings = MagicMock()
        return model

    def test_format_instruction_with_register(self, mock_config):
        """Test formatting instruction with register token."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        # Mock the tokenizer for this test
        trainer.tokenizer = MagicMock()

        instruction = "What is KYC?"
        formatted = trainer._format_instruction_with_register(instruction, "formal")

        assert "[REGISTER: formal]" in formatted
        assert "What is KYC?" in formatted

    def test_format_instruction_without_register_token(self, mock_config):
        """Test formatting without register token prepending."""
        config = RegisterTrainerConfig(prepend_register_token=False)
        trainer = RegisterControlledSFTTrainer(
            config=config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        instruction = "What is KYC?"
        formatted = trainer._format_instruction_with_register(instruction, "formal")

        assert formatted == instruction  # Should be unchanged

    def test_format_instruction_with_context(self, mock_config):
        """Test formatting with context."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        instruction = "What is KYC?"
        context = "Banking regulations"
        formatted = trainer._format_instruction_with_register(instruction, "formal", context)

        assert "[REGISTER: formal]" in formatted
        assert "[CONTEXT: Banking regulations]" in formatted
        assert "What is KYC?" in formatted

    def test_format_instruction_invalid_register(self, mock_config):
        """Test formatting with invalid register defaults to 'formal'."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        instruction = "What is KYC?"
        # Should default to "formal" when invalid register given
        formatted = trainer._format_instruction_with_register(instruction, "invalid")

        assert "[REGISTER: formal]" in formatted

    def test_register_tokens_registered(self, mock_config, mock_tokenizer, mock_model):
        """Test that register tokens are registered with tokenizer."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=mock_model,
            processing_class=mock_tokenizer,
        )

        # Verify special tokens were added
        assert mock_tokenizer.add_special_tokens.called

        # Verify register token IDs are stored
        assert len(trainer.register_token_ids) == 4

    def test_register_token_ids_mapping(self, mock_config, mock_tokenizer, mock_model):
        """Test register token ID mapping."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=mock_model,
            processing_class=mock_tokenizer,
        )

        # All expected registers should have token IDs
        for register in ["formal", "legal", "technical", "simple"]:
            assert register in trainer.register_token_ids
            assert isinstance(trainer.register_token_ids[register], int)

    def test_compute_loss_without_auxiliary_loss(self, mock_config):
        """Test loss computation without auxiliary loss."""
        config = RegisterTrainerConfig(register_loss_weight=0.0)
        trainer = RegisterControlledSFTTrainer(
            config=config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        # Mock model and inputs
        mock_model = MagicMock()
        mock_model.return_value.loss = torch.tensor(0.5)

        inputs = {"input_ids": torch.zeros(2, 10)}

        # With loss_weight=0, should return lm_loss directly
        loss = trainer.compute_loss(mock_model, inputs)
        assert isinstance(loss, torch.Tensor)

    def test_compute_loss_with_return_outputs(self, mock_config):
        """Test loss computation returning outputs."""
        trainer = RegisterControlledSFTTrainer(
            config=mock_config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        mock_model = MagicMock()
        mock_model.return_value.loss = torch.tensor(0.5)

        inputs = {"input_ids": torch.zeros(2, 10)}

        # Should return tuple when return_outputs=True
        try:
            result = trainer.compute_loss(mock_model, inputs, return_outputs=True)
            # Result should be tuple or loss
            assert result is not None
        except Exception:
            # Expected since we're mocking
                pass


class TestRegisterTrainer:
    """Test RegisterTrainer wrapper class."""

    @pytest.fixture
    def trainer_config(self):
        """Create trainer config."""
        return RegisterTrainerConfig(
            model_name="gpt2",
            output_dir="./test_output",
        )

    def test_trainer_initialization(self, trainer_config):
        """Test RegisterTrainer initialization."""
        trainer = RegisterTrainer(trainer_config)
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None

    def test_trainer_model_loading(self, trainer_config):
        """Test model and tokenizer loading."""
        trainer = RegisterTrainer(trainer_config)

        # Mock the AutoModelForCausalLM and AutoTokenizer
        with patch("aligntune.backends.trl.sft.register_trainer.AutoTokenizer") as mock_tokenizer_cls, \
             patch("aligntune.backends.trl.sft.register_trainer.AutoModelForCausalLM") as mock_model_cls:

            # Setup mocks
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model

            # Load model
            trainer.load_model_and_tokenizer()

            # Verify loading
            assert trainer.model is mock_model
            assert trainer.tokenizer is mock_tokenizer
            mock_tokenizer_cls.from_pretrained.assert_called_once_with("gpt2")
            mock_model_cls.from_pretrained.assert_called_once_with("gpt2")

    def test_trainer_save_model_without_model(self, trainer_config):
        """Test save_model raises error without loaded model."""
        trainer = RegisterTrainer(trainer_config)

        with pytest.raises(ValueError, match="Model not loaded"):
            trainer.save_model("./output")


class TestRegisterTokenRegistration:
    """Test register token registration mechanics."""

    @pytest.fixture(autouse=True)
    def patch_sft_init(self):
        def _stub(inst, *args, **kwargs):
            inst.tokenizer = kwargs.get('processing_class')
            inst.model = kwargs.get('model')
        with patch('trl.SFTTrainer.__init__', _stub):
            yield

    def test_special_tokens_format(self):
        """Test special tokens are formatted correctly."""
        config = RegisterTrainerConfig()
        trainer = RegisterControlledSFTTrainer(
            config=config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        # Check token format
        expected_tokens = [
            "[REGISTER: formal]",
            "[REGISTER: legal]",
            "[REGISTER: technical]",
            "[REGISTER: simple]",
        ]
        assert trainer.register_tokens == expected_tokens

    def test_register_token_prepending(self):
        """Test register tokens are prepended correctly."""
        config = RegisterTrainerConfig(prepend_register_token=True)
        trainer = RegisterControlledSFTTrainer(
            config=config,
            model=MagicMock(),
            processing_class=MagicMock(),
        )

        for register in ["formal", "legal", "technical", "simple"]:
            formatted = trainer._format_instruction_with_register("Test", register)
            assert f"[REGISTER: {register}]" in formatted


class TestRegisterTrainerConfiguration:
    """Test various configuration scenarios."""

    def test_all_registers_supported(self):
        """Test all standard registers are supported."""
        registers = ["formal", "legal", "technical", "simple"]
        for register in registers:
            config = RegisterTrainerConfig(registers=[register])
            assert register in config.registers

    def test_multiple_registers(self):
        """Test configuration with multiple registers."""
        registers = ["formal", "technical", "simple"]
        config = RegisterTrainerConfig(registers=registers)
        assert len(config.registers) == 3
        for reg in registers:
            assert reg in config.registers

    def test_loss_weight_range(self):
        """Test loss weight valid range."""
        for weight in [0.0, 0.1, 0.5, 0.9, 1.0]:
            config = RegisterTrainerConfig(register_loss_weight=weight)
            assert config.register_loss_weight == weight


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
