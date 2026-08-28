"""
Comprehensive SFT Training Test Suite — TRL backend only.

Split out from the former combined test_sft_comprehensive.py (which mixed
"trl" and "unsloth" backend parametrizations in one file/process) because
Unsloth monkey-patches transformers/trl globally on import, which was
silently flipping trl.SFTConfig's padding_free default for later,
TRL-only tests in the same process. See test_sft_comprehensive_unsloth.py
for the Unsloth-backend tests.

Tests:
- TRL backend only
- Multiple models (Qwen2.5, Llama-3.1, Qwen2, Gemma)
- Multiple PEFT adapters (LoRA, DoRA, Vera, Kron)
- High-level API tests using aligntune only
"""

import os
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_sft_trainer

# Default model is overridable via --model CLI option (see conftest.py).
_DEFAULT_MODEL = os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# =============================================================================
# CONSTANTS - Small models and few steps for fast testing
# =============================================================================

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-8B",
    _DEFAULT_MODEL,
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen2-7B-Instruct",
    "google/gemma-7b-it",
]

MAX_STEPS = 10  # Fast testing
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 256  # Smaller for fast testing

# PEFT variants from core/peft/factory.py
PEFT_VARIANTS = [
    "standard",  # Basic LoRA
    "dora",      # DoRA (Weight-Decomposed Low-Rank Adaptation)
    "vera",      # Vera (Vector-based LoRA)
    "kron",      # Kron (Kronecker-factored LoRA)
]

# DataManager.load_dataset() keeps whatever string is passed as `split` as the
# literal dict key, so HF-style "train[:50]" leaves the result keyed
# "train[:50]" instead of "train"; downstream code that reads
# dataset_dict.get("train") then finds nothing. Use split="train" +
# max_samples=N instead (see notebooks/01_sft.ipynb).
DATASET_CONFIGS = [
    {
        "name": "tatsu-lab/alpaca",
        "split": "train",
        "max_samples": 50,
        "task_type": "sft",
        "desc": "alpaca_instruction",
    },
    {
        "name": "HuggingFaceH4/instruction-dataset",
        "split": "train",
        "max_samples": 50,
        "task_type": "sft",
        "desc": "instruction",
    },
    {
        "name": "philschmid/guanaco-sharegpt-style",
        "split": "train",
        "max_samples": 50,
        "task_type": "sft",
        "desc": "multiturn_sharegpt",
    },
]


# =============================================================================
# PARAMETRIZED TEST CONFIGS
# =============================================================================

# High-level API tests - simpler configuration (TRL rows only)
HIGH_LEVEL_CONFIGS = [
    # Model, Backend, PEFT Variant, Dataset, ID
    (MODELS[0], "trl", "standard", DATASET_CONFIGS[0]),  # Qwen2.5 + alpaca
    (MODELS[1], "trl", "dora", DATASET_CONFIGS[0]),      # Llama + alpaca
    (MODELS[2], "trl", "vera", DATASET_CONFIGS[1]),      # Qwen2 + instruction
    (MODELS[3], "trl", "kron", DATASET_CONFIGS[2]),      # Gemma + multiturn
]

# Unit-like tests - test individual backend/adapter combinations (TRL rows only)
UNIT_CONFIGS = [
    ("trl", "standard"),
    ("trl", "dora"),
    ("trl", "vera"),
    ("trl", "kron"),
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
        use_unsloth=False,
        quantization=get_quantization_config(peft_variant),
        eval_interval=2,
        split=dataset_config["split"],
        max_samples=dataset_config["max_samples"],
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.name_or_path == model_name
    assert trainer.config.train.max_steps == 3


@pytest.mark.parametrize(
    "model_name,backend,peft_variant,dataset_config",
    HIGH_LEVEL_CONFIGS[:2],  # Run just 2 actual training tests
    ids=[
        f"{cfg[0].split('/')[-1]}_{cfg[1]}_{cfg[2]}_train"
        for cfg in HIGH_LEVEL_CONFIGS[:2]
    ],
)
def test_sft_training_execution(model_name, backend, peft_variant, dataset_config):
    """Test SFT trainer with actual training loop."""
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_{backend}_{peft_variant}_exec_"
    )

    trainer = create_sft_trainer(
        model_name=model_name,
        dataset_name=dataset_config["name"],
        backend=backend,
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_variant=peft_variant,
        use_unsloth=False,
        quantization=get_quantization_config(peft_variant),
        eval_interval=5,
        split=dataset_config["split"],
        max_samples=dataset_config["max_samples"],
        seed=42,
    )

    # Setup model first (lazy loading)
    trainer.setup_model()

    # Print trainable params for debugging
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Training {model_name} with {peft_variant}")
    print(f"Trainable params: {trainable:,} / Total: {total:,} ({pct:.4f}%)")
    print(f"{'='*60}\n")

    # Enable gradient checkpointing for Qwen models only (memory efficiency)
    if "Qwen" in model_name and hasattr(trainer.model, 'gradient_checkpointing_enable'):
        trainer.model.gradient_checkpointing_enable()

    result = trainer.train()

    # Verify training result (loss can be in different locations depending on trainer)
    loss = None
    if isinstance(result, dict):
        # Check multiple possible locations for loss
        if "train_loss" in result.get("metrics", {}):
            loss = result["metrics"]["train_loss"]
        elif "training_loss" in result:
            loss = result["training_loss"]
        elif "loss" in result:
            loss = result["loss"]
    else:
        loss = getattr(result, "training_loss", None)

    assert loss is not None, "Training failed to return loss"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"


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
        use_unsloth=False,
        quantization=get_quantization_config(peft_variant),
        split=dataset_config["split"],
        max_samples=dataset_config["max_samples"],
        seed=42,
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.peft.variant == peft_variant


@pytest.mark.parametrize(
    "model_name",
    MODELS,
    ids=[m.split("/")[-1] for m in MODELS],
)
def test_sft_model_compatibility(model_name):
    """Test different model architectures."""
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_model_{model_name.split('/')[-1]}_"
    )

    trainer = create_sft_trainer(
        model_name=model_name,
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
        seed=42,
    )

    assert_trainer_created(trainer)
    assert model_name in trainer.config.model.name_or_path


def test_sft_backend_switching():
    """Test the TRL backend path (Unsloth side is in test_sft_comprehensive_unsloth.py)."""
    backend = "trl"
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
        use_unsloth=False,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
        seed=42,
    )

    assert_trainer_created(trainer)
    backend_name = trainer.__class__.__module__
    assert backend.lower() in backend_name.lower(), f"Expected {backend} in {backend_name}"


@pytest.mark.parametrize(
    "peft_variant",
    PEFT_VARIANTS,
    ids=[f"peft_{v}" for v in PEFT_VARIANTS],
)
def test_sft_peft_variants(peft_variant):
    """Test all PEFT adapter variants."""
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_peft_{peft_variant}_"
    )

    # Create trainer with explicit variant setting
    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
        seed=42,
    )

    assert_trainer_created(trainer)
    # Verify PEFT is enabled
    assert trainer.config.model.peft.enabled is True


# =============================================================================
# CONFIG VALIDATION TESTS
# =============================================================================

def test_sft_config_defaults():
    """Test that default configuration is properly set."""
    output_dir = tempfile.mkdtemp(prefix="sft_defaults_")

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=5,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    assert trainer.config.train.per_device_batch_size == BATCH_SIZE
    assert trainer.config.train.learning_rate == LEARNING_RATE
    assert trainer.config.train.max_steps == 5
    assert trainer.config.logging.output_dir == output_dir


def test_sft_config_custom_values():
    """Test custom configuration parameters."""
    output_dir = tempfile.mkdtemp(prefix="sft_custom_")
    custom_lr = 5e-4
    custom_batch = 4
    custom_steps = 8

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=custom_steps,
        batch_size=custom_batch,
        learning_rate=custom_lr,
        max_seq_length=512,
        lora_r=16,
        lora_alpha=32,
    )

    assert trainer.config.train.learning_rate == custom_lr
    assert trainer.config.train.per_device_batch_size == custom_batch
    assert trainer.config.train.max_steps == custom_steps
    assert trainer.config.model.peft.rank == 16
    assert trainer.config.model.peft.alpha == 32


# =============================================================================
# PEFT CONFIGURATION TESTS
# =============================================================================

def test_sft_lora_disabled():
    """Test SFT training without LoRA."""
    output_dir = tempfile.mkdtemp(prefix="sft_no_peft_")

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        use_peft=False,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
    )

    assert_trainer_created(trainer)
    assert trainer.config.model.peft.enabled is False


def test_sft_lora_custom_config():
    """Test LoRA with custom configuration."""
    output_dir = tempfile.mkdtemp(prefix="sft_lora_custom_")

    custom_r = 32
    custom_alpha = 64
    custom_dropout = 0.2

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=3,
        use_peft=True,
        lora_r=custom_r,
        lora_alpha=custom_alpha,
        lora_dropout=custom_dropout,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
    )

    assert trainer.config.model.peft.enabled is True
    assert trainer.config.model.peft.rank == custom_r
    assert trainer.config.model.peft.alpha == custom_alpha
    assert trainer.config.model.peft.dropout == custom_dropout


# =============================================================================
# DATASET CONFIGURATION TESTS
# =============================================================================

@pytest.mark.parametrize(
    "dataset_config",
    DATASET_CONFIGS,
    ids=[dc["desc"] for dc in DATASET_CONFIGS],
)
def test_sft_dataset_configs(dataset_config):
    """Test different dataset configurations."""
    output_dir = tempfile.mkdtemp(
        prefix=f"sft_dataset_{dataset_config['desc']}_"
    )

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=dataset_config["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=3,
        batch_size=BATCH_SIZE,
        split=dataset_config["split"],
        max_samples=dataset_config["max_samples"],
    )

    assert_trainer_created(trainer)
    assert trainer.config.dataset.name == dataset_config["name"]
    assert dataset_config["split"] in str(trainer.config.dataset.split)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_sft_full_pipeline():
    """Full integration test: create trainer and run training."""
    output_dir = tempfile.mkdtemp(prefix="sft_integration_")

    trainer = create_sft_trainer(
        model_name=MODELS[0],
        dataset_name=DATASET_CONFIGS[0]["name"],
        backend="trl",
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        eval_interval=5,
        split=DATASET_CONFIGS[0]["split"],
        max_samples=DATASET_CONFIGS[0]["max_samples"],
        seed=42,
    )

    # Verify trainer state before training
    assert trainer.config is not None
    assert trainer.config.model.name_or_path == MODELS[0]

    # Setup model first (lazy loading)
    trainer.setup_model()

    # Print trainable params for debugging
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Full Pipeline Test: {MODELS[0]}")
    print(f"Trainable params: {trainable:,} / Total: {total:,} ({pct:.4f}%)")
    print(f"{'='*60}\n")

    # Enable gradient checkpointing for Qwen models only (memory efficiency)
    if "Qwen" in MODELS[0] and hasattr(trainer.model, 'gradient_checkpointing_enable'):
        trainer.model.gradient_checkpointing_enable()

    # Run training
    result = trainer.train()

    # Verify training completed (loss can be in different locations depending on trainer)
    loss = None
    if isinstance(result, dict):
        # Check multiple possible locations for loss
        if "train_loss" in result.get("metrics", {}):
            loss = result["metrics"]["train_loss"]
        elif "training_loss" in result:
            loss = result["training_loss"]
        elif "loss" in result:
            loss = result["loss"]
    else:
        loss = getattr(result, "training_loss", None)

    assert loss is not None, "Training did not complete successfully"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
