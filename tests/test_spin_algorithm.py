"""
SPIN Algorithm Test Suite

Tests SPIN (Self-Play Improvement through No-regret learning) with:
- Import validation
- Backend availability
- Trainer creation
- Model/data setup
"""

import os
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = [
    os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B"),
]

MAX_STEPS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512

BACKEND = "trl"  # SPIN only uses TRL backend
DATASET_NAME = "yahma/alpaca-cleaned"  # imdb has no prompt column, incompatible with SPIN's schema
# DataManager.load_dataset() keeps whatever string is passed as `split` as the
# literal dict key (it doesn't slice it), so HF-style "train[:4]" leaves the
# result keyed "train[:4]" instead of "train" and downstream code that reads
# dataset_dict["train"] finds nothing. The supported way to limit sample count
# is split="train" + max_samples=N (see notebooks/18_spin.ipynb).
DATASET_SPLIT = "train"
MAX_SAMPLES = 4


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_trainer_created(trainer):
    """Assert trainer was created successfully."""
    assert trainer is not None, "Trainer creation failed"
    assert hasattr(trainer, "train"), "Trainer missing train() method"
    assert hasattr(trainer, "config"), "Trainer missing config attribute"


# =============================================================================
# IMPORT TESTS
# =============================================================================

def test_spin_imports():
    """Test SPIN module imports."""
    try:
        from aligntune.backends.trl.rl.spin.spin import TRLSPINTrainer
        assert TRLSPINTrainer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import TRLSPINTrainer: {e}")


def test_spin_backend_availability():
    """Test SPIN backend is available."""
    from aligntune.backends.trl.rl.spin.spin import TRLSPINTrainer

    is_available = TRLSPINTrainer.is_available()
    assert is_available, "SPIN backend not available (missing TRL dependencies?)"


# =============================================================================
# TRAINER CREATION TESTS
# =============================================================================

@pytest.mark.parametrize("model_name", MODELS, ids=[m.split("/")[-1] for m in MODELS])
def test_spin_trainer_creation(model_name):
    """Test SPIN trainer creation."""
    output_dir = tempfile.mkdtemp(prefix=f"spin_{model_name.split('/')[-1]}_")

    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend=BACKEND,
        output_dir=output_dir,
        num_rounds=1,
        dpo_steps_per_round=10,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.name_or_path == model_name


# =============================================================================
# SETUP TESTS
# =============================================================================

def test_spin_model_setup():
    """Test SPIN model setup."""
    output_dir = tempfile.mkdtemp(prefix="spin_model_setup_")

    trainer = create_rl_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend=BACKEND,
        output_dir=output_dir,
        num_rounds=1,
        dpo_steps_per_round=10,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        use_peft=True,
        lora_r=8,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model
    trainer.setup_model()

    # Check model components
    assert trainer.model is not None, "Policy model not initialized"
    assert trainer.tokenizer is not None, "Tokenizer not initialized"
    assert trainer.opponent_checkpoint_dir is not None, "Opponent checkpoint not initialized"

    # Reference model should be None for PEFT
    assert trainer.reference_model is None, "Reference model should be None for PEFT"


def test_spin_data_setup():
    """Test SPIN data setup."""
    output_dir = tempfile.mkdtemp(prefix="spin_data_setup_")

    trainer = create_rl_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend=BACKEND,
        output_dir=output_dir,
        num_rounds=1,
        dpo_steps_per_round=10,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model first (needed for tokenizer)
    trainer.setup_model()

    # Setup data
    trainer.setup_data()

    # Check datasets
    assert trainer.sft_dataset is not None, "SFT dataset not loaded"
    assert trainer.train_dataset is not None, "Train dataset not loaded"
    assert len(trainer.sft_dataset) > 0, "SFT dataset is empty"


def test_spin_trainer_config_setup():
    """Test SPIN DPO trainer config setup."""
    output_dir = tempfile.mkdtemp(prefix="spin_trainer_setup_")

    trainer = create_rl_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend=BACKEND,
        output_dir=output_dir,
        num_rounds=1,
        dpo_steps_per_round=10,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=512,
        use_peft=True,
        lora_r=8,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model and data
    trainer.setup_model()
    trainer.setup_data()

    # Setup DPO trainer config
    trainer.setup_trainer()

    # Check DPO config
    assert trainer.dpo_config is not None, "DPO config not created"
    assert trainer.dpo_config.max_steps == 10, "DPO steps per round not set correctly"
    assert trainer.dpo_config.learning_rate == LEARNING_RATE, "Learning rate not set correctly"

    # Check max_prompt_length < max_length (attribute may be absent on trl>=1.0, which
    # dropped max_prompt_length from DPOConfig in favor of max_length + truncation_mode)
    prompt_len = getattr(trainer.dpo_config, "max_prompt_length", trainer.dpo_config.max_length // 2)
    assert prompt_len < trainer.dpo_config.max_length, \
        f"max_prompt_length ({prompt_len}) should be < max_length ({trainer.dpo_config.max_length})"


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================

def test_spin_max_prompt_length_validation():
    """Test that max_prompt_length is validated correctly."""
    output_dir = tempfile.mkdtemp(prefix="spin_validation_")

    # Test with max_prompt_length not set (should default to half)
    trainer = create_rl_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend=BACKEND,
        output_dir=output_dir,
        num_rounds=1,
        dpo_steps_per_round=10,
        batch_size=BATCH_SIZE,
        max_seq_length=512,
        use_peft=True,
        lora_r=8,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_trainer()

    # Should default to max_length // 2 (attribute may be absent on trl>=1.0, which
    # dropped max_prompt_length from DPOConfig in favor of max_length + truncation_mode)
    actual = getattr(trainer.dpo_config, "max_prompt_length", 512 // 2)
    assert actual == 512 // 2, \
        f"max_prompt_length should default to {512 // 2}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
