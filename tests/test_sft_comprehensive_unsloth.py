"""
Comprehensive SFT Training Test Suite — Unsloth backend only.

Split out from the former combined test_sft_comprehensive.py (which mixed
"trl" and "unsloth" backend parametrizations in one file/process) because
Unsloth monkey-patches transformers/trl globally on import, which was
silently flipping trl.SFTConfig's padding_free default for later,
TRL-only tests in the same process. See test_sft_comprehensive_trl.py
for the TRL-backend tests (including the only real training-loop test,
which only ever exercised "trl" rows in the original file).

Tests:
- Unsloth backend only
- Multiple models, PEFT adapters (LoRA, DoRA)
- High-level API tests using aligntune only
"""

import os
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_sft_trainer

# =============================================================================
# CONSTANTS - Small models and few steps for fast testing
# =============================================================================

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-8B",
]

BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 256  # Smaller for fast testing

DATASET_CONFIGS = [
    {
        "name": "tatsu-lab/alpaca",
        "split": "train[:50]",
        "task_type": "sft",
        "desc": "alpaca_instruction",
    },
    {
        "name": "HuggingFaceH4/instruction-dataset",
        "split": "train[:50]",
        "task_type": "sft",
        "desc": "instruction",
    },
    {
        "name": "philschmid/guanaco-sharegpt-style",
        "split": "train[:50]",
        "task_type": "sft",
        "desc": "multiturn_sharegpt",
    },
]

# High-level API tests - simpler configuration (Unsloth rows only)
HIGH_LEVEL_CONFIGS = [
    # Model, Backend, PEFT Variant, Dataset, ID
    (MODELS[0], "unsloth", "standard", DATASET_CONFIGS[1]),  # Qwen2.5 + instruction
    (MODELS[1], "unsloth", "dora", DATASET_CONFIGS[2]),      # Llama + multiturn
]

# Unit-like tests - test individual backend/adapter combinations (Unsloth rows only)
UNIT_CONFIGS = [
    ("unsloth", "standard"),
    ("unsloth", "dora"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_quantization_config(peft_variant: str) -> dict:
    """Get quantization config based on PEFT variant."""
    # Only apply 4-bit quantization to standard LoRA
    if peft_variant == "standard":
        return {
            "load_in_4bit": False,  # Disabled for fast testing
        }
    return {}


def assert_trainer_created(trainer):
    """Assert trainer was created successfully."""
    assert trainer is not None, "Trainer creation failed"
    assert hasattr(trainer, "train"), "Trainer missing train() method"
    assert hasattr(trainer, "config"), "Trainer missing config attribute"


# =============================================================================
# HIGH-LEVEL API TESTS
# =============================================================================

@pytest.mark.parametrize(
    "model_name,backend,peft_variant,dataset_config",
    HIGH_LEVEL_CONFIGS,
    ids=[
        f"{cfg[0].split('/')[-1]}_{cfg[1]}_{cfg[2]}_{cfg[3]['desc']}"
        for cfg in HIGH_LEVEL_CONFIGS
    ],
)
def test_sft_trainer_creation(model_name, backend, peft_variant, dataset_config):
    """Test SFT trainer creation with various configurations."""
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_{backend}_{peft_variant}_creation_"
    )

    trainer = create_sft_trainer(
        model_name=model_name,
        dataset_name=dataset_config["name"],
        backend=backend,
        output_dir=output_dir,
        max_steps=3,  # Just test creation, not training
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_variant=peft_variant,
        use_unsloth=True,
        quantization=get_quantization_config(peft_variant),
        eval_interval=2,
        split=dataset_config["split"],
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.name_or_path == model_name
    assert trainer.config.train.max_steps == 3


# =============================================================================
# UNIT-LIKE TESTS
# =============================================================================

@pytest.mark.parametrize(
    "backend,peft_variant",
    UNIT_CONFIGS,
    ids=[f"{cfg[0]}_{cfg[1]}" for cfg in UNIT_CONFIGS],
)
def test_sft_backend_adapter_combination(backend, peft_variant):
    """Test specific backend + PEFT adapter combinations (unit-level)."""
    model_name = MODELS[0]  # Use first model for consistency
    dataset_config = DATASET_CONFIGS[0]

    output_dir = tempfile.mkdtemp(
        prefix=f"sft_unit_{backend}_{peft_variant}_"
    )

    trainer = create_sft_trainer(
        model_name=model_name,
        dataset_name=dataset_config["name"],
        backend=backend,
        output_dir=output_dir,
        max_steps=5,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_variant=peft_variant,
        use_unsloth=True,
        quantization=get_quantization_config(peft_variant),
        split=dataset_config["split"],
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.peft.variant == peft_variant


def test_sft_backend_switching():
    """Test the Unsloth backend path (TRL side is in test_sft_comprehensive_trl.py)."""
    backend = "unsloth"
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_backend_{backend}_"
    )

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend=backend,
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        use_unsloth=True,
        split=DATASET_CONFIGS[0]["split"],
        seed=42,
    )

    assert_trainer_created(trainer)
    # Verify backend is correctly set (allow TRL fallback when unsloth is unavailable)
    backend_name = trainer.__class__.__module__
    if "trl" in backend_name.lower():
        pytest.skip("Unsloth not available, fell back to TRL backend")
    assert backend.lower() in backend_name.lower(), f"Expected {backend} in {backend_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
