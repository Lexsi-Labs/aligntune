"""
ORPO Algorithm Test Suite - Unsloth Backend Only

Tests ORPO with Unsloth backend to avoid Unsloth global patch conflicts
(Unsloth monkey-patches transformers attention forward methods process-wide;
running TRL-loaded models in the same process after Unsloth has patched
things raises AttributeError: 'Qwen2Attention' object has no attribute
'apply_qkv'). See test_orpo_algorithm_trl.py for the TRL backend.

Tests ORPO with:
- Models: Qwen (0.5B, 1.5B), Llama (1B, 3B), Gemma (2B)
- PEFT: LoRA
  (QLoRA on Unsloth is not covered upstream in the original combined suite.)
"""

import os
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# CONSTANTS
# =============================================================================

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
]

MAX_STEPS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512

BACKEND = "unsloth"
DATASET_NAME = "Anthropic/hh-rlhf"
DATASET_SPLIT = "train[:50]"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_trainer_created(trainer):
    """Assert trainer was created successfully."""
    assert trainer is not None, "Trainer creation failed"
    assert hasattr(trainer, "train"), "Trainer missing train() method"
    assert hasattr(trainer, "config"), "Trainer missing config attribute"


# =============================================================================
# LORA TESTS
# =============================================================================

@pytest.mark.parametrize("model_name", MODELS, ids=[m.split("/")[-1] for m in MODELS])
def test_orpo_lora_creation(model_name):
    """Test ORPO trainer creation with LoRA."""
    output_dir = tempfile.mkdtemp(prefix=f"orpo_lora_{BACKEND}_{model_name.split('/')[-1]}_")

    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        algorithm="orpo",
        backend=BACKEND,
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.name_or_path == model_name


def test_orpo_lora_training():
    """Test ORPO training with LoRA."""
    model_name = MODELS[0]
    output_dir = tempfile.mkdtemp(prefix=f"orpo_lora_train_{BACKEND}_")

    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        algorithm="orpo",
        backend=BACKEND,
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    trainer.setup_model()

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"ORPO LoRA Training: {model_name} ({BACKEND})")
    print(f"Trainable params: {trainable:,} / Total: {total:,} ({pct:.4f}%)")
    print(f"{'='*60}\n")

    result = trainer.train()

    assert result is not None


# =============================================================================
# QLORA TESTS
# =============================================================================

@pytest.mark.parametrize("model_name", MODELS, ids=[m.split("/")[-1] for m in MODELS])
def test_orpo_qlora_creation(model_name):
    """Test ORPO trainer creation with QLoRA."""
    output_dir = tempfile.mkdtemp(prefix=f"orpo_qlora_{BACKEND}_{model_name.split('/')[-1]}_")

    trainer = create_rl_trainer(
        model_name=model_name,
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        algorithm="orpo",
        backend=BACKEND,
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        quantization={"load_in_4bit": True},
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    assert_trainer_created(trainer)


# =============================================================================
# BACKEND AVAILABILITY TESTS
# =============================================================================

def test_unsloth_orpo_available():
    """Test Unsloth ORPO is available."""
    from aligntune.backends.unsloth.rl.orpo.orpo import UnslothORPOTrainer
    assert UnslothORPOTrainer.is_available()


def test_orpo_unsloth_registered():
    """Test ORPO Unsloth backend is registered in BackendFactory."""
    from aligntune.core.backend_factory import BackendFactory, TrainingType, BackendType, RLAlgorithm

    unsloth_key = (TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.ORPO)
    assert unsloth_key in BackendFactory._backends


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
