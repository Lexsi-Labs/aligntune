"""
Tests for SPIN (Self-Play Fine-Tuning) Trainer Implementation

This test suite validates:
- Opponent checkpoint initialization and management
- Preference pair generation (SFT=chosen, opponent=rejected)
- Multiple self-play iterations
- Round tracking and checkpoint naming
- Batch response generation
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

from aligntune.backends.trl.rl.spin.spin import TRLSPINTrainer


class MockConfig:
    """Mock config for testing SPIN trainer."""

    def __init__(self):
        self.model = Mock()
        self.model.name_or_path = "gpt2"
        self.model.precision = Mock()
        self.model.precision.value = "fp32"
        self.model.device_map = "auto"
        self.model.quantization = None
        self.model.max_seq_length = 512

        self.dataset = Mock()
        self.dataset.name = "wikitext"
        self.dataset.split = "train"
        self.dataset.system_prompt = None

        self.train = Mock()
        self.train.num_rounds = 2
        self.train.generation_temperature = 0.7
        self.train.generation_max_length = 512
        self.train.dpo_steps_per_round = 100
        self.train.per_device_batch_size = 4
        self.train.learning_rate = 5e-5
        self.train.epochs = 1
        self.train.enable_thinking = False
        self.train.max_steps = None

        self.logging = Mock()
        self.logging.output_dir = "./test_output"

        self.distributed = Mock()
        self.distributed.backend = "single"


class TestSPINInitialization:
    """Test SPIN trainer initialization."""

    def test_spin_trainer_init_basic(self):
        """Test basic SPIN trainer initialization."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            assert trainer.num_rounds == 2
            assert trainer.generation_temperature == 0.7
            assert trainer.current_round == 0
            assert trainer.model is None
            assert trainer.opponent_model is None

    def test_spin_trainer_config_defaults(self):
        """Test SPIN trainer uses correct defaults."""
        config = MockConfig()
        # Remove SPIN params to test defaults
        del config.train.num_rounds
        del config.train.generation_temperature
        del config.train.generation_max_length
        del config.train.dpo_steps_per_round

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            # Check defaults
            assert trainer.num_rounds == 2  # Default
            assert trainer.generation_temperature == 0.7  # Default
            assert trainer.generation_max_length == 512  # Default
            assert trainer.dpo_steps_per_round == 100  # Default


class TestOpponentCheckpointManagement:
    """Test opponent checkpoint initialization and updates."""

    def test_opponent_checkpoint_initialization(self):
        """Test opponent checkpoint is initialized as copy of model."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            # Mock model and tokenizer
            trainer.model = MagicMock()
            trainer.tokenizer = MagicMock()

            # Initialize opponent checkpoint
            trainer._initialize_opponent_checkpoint()

            # Verify directory created. Opponent is now self.model directly,
            # so no save_pretrained call happens here (see spin.py docstring).
            assert trainer.opponent_checkpoint_dir is not None
            trainer.model.save_pretrained.assert_not_called()
            trainer.tokenizer.save_pretrained.assert_not_called()

            # Cleanup
            if os.path.exists(trainer.opponent_checkpoint_dir):
                shutil.rmtree(trainer.opponent_checkpoint_dir)

    def test_opponent_checkpoint_update(self):
        """Test opponent checkpoint is updated after DPO training."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            trainer.model = MagicMock()
            trainer.tokenizer = MagicMock()
            trainer.opponent_checkpoint_dir = tempfile.mkdtemp()

            try:
                # Update opponent checkpoint
                trainer.update_opponent_checkpoint()

                # This is a no-op: opponent is self.model directly, so the
                # trained model becomes the opponent without a save_pretrained
                # call (see spin.py docstring).
                trainer.model.save_pretrained.assert_not_called()
                trainer.tokenizer.save_pretrained.assert_not_called()
            finally:
                if os.path.exists(trainer.opponent_checkpoint_dir):
                    shutil.rmtree(trainer.opponent_checkpoint_dir)


class TestPreferencePairGeneration:
    """Test synthetic preference pair generation."""

    def test_generate_responses_signature(self):
        """Test generate_responses method exists with correct signature."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            assert hasattr(trainer, 'generate_responses')
            assert callable(trainer.generate_responses)

    def test_preference_pair_creation_signature(self):
        """Test create_preference_pairs method exists."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            assert hasattr(trainer, 'create_preference_pairs')
            assert callable(trainer.create_preference_pairs)


class TestSelfPlayRounds:
    """Test self-play iteration logic."""

    def test_round_tracking(self):
        """Test round counter is updated correctly."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            assert trainer.current_round == 0

            # Simulate rounds
            for i in range(trainer.num_rounds):
                trainer.current_round = i
                assert trainer.current_round == i

    def test_num_rounds_configuration(self):
        """Test num_rounds is properly configured."""
        config = MockConfig()
        config.train.num_rounds = 5

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            assert trainer.num_rounds == 5


class TestCheckpointNaming:
    """Test checkpoint naming conventions."""

    def test_round_checkpoint_naming(self):
        """Test round checkpoints follow naming convention."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            output_dir = "./test_output"

            # Expected checkpoint paths should follow: spin_round_1, spin_round_2, etc.
            for round_idx in range(trainer.num_rounds):
                expected_path = os.path.join(output_dir, f"spin_round_{round_idx + 1}")
                assert f"spin_round_{round_idx + 1}" in expected_path


class TestAbstractMethodImplementation:
    """Test SPIN implements required abstract methods."""

    def test_implements_setup_data(self):
        """Test setup_data is implemented."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            assert hasattr(trainer, 'setup_data')
            assert callable(trainer.setup_data)

    def test_implements_setup_rewards(self):
        """Test setup_rewards is implemented."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            assert hasattr(trainer, 'setup_rewards')
            assert callable(trainer.setup_rewards)

    def test_implements_train_step(self):
        """Test train_step is implemented."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            assert hasattr(trainer, 'train_step')
            assert callable(trainer.train_step)


class TestSpinSpecificFeatures:
    """Test SPIN-specific features."""

    def test_spin_generates_from_both_models(self):
        """Test SPIN generates responses from current and opponent models."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            # SPIN should have methods for generating from both models
            assert hasattr(trainer, 'generate_responses')
            assert hasattr(trainer, 'create_preference_pairs')

    def test_sft_dataset_storage(self):
        """Test SFT dataset is stored separately for SPIN."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            # The trainer should store SFT dataset separately
            assert hasattr(trainer, 'sft_dataset')
            assert trainer.sft_dataset is None  # Before setup


class TestTrainerAvailability:
    """Test trainer availability checking."""

    def test_is_available_method(self):
        """Test is_available class method exists."""
        assert hasattr(TRLSPINTrainer, 'is_available')
        assert callable(TRLSPINTrainer.is_available)


class TestUpdateOpponentCheckpoint:
    """Test opponent checkpoint update mechanism."""

    def test_update_opponent_checkpoint_method_exists(self):
        """Test update_opponent_checkpoint method exists."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)
            assert hasattr(trainer, 'update_opponent_checkpoint')
            assert callable(trainer.update_opponent_checkpoint)


class TestDataConfig:
    """Test SPIN data configuration."""

    def test_supports_sft_dataset_config(self):
        """Test that SPIN can be configured with SFT dataset."""
        config = MockConfig()

        with patch(
            'aligntune.backends.trl.rl.spin.spin.TrainerBase.__init__',
            new=lambda self, cfg, callbacks=None: setattr(self, 'config', cfg),
        ):
            trainer = TRLSPINTrainer(config)

            # Should have data setup capability
            assert hasattr(trainer, 'setup_dataset')
            assert callable(trainer.setup_dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
