"""
Online DPO End-to-End Test

Tests Online Iterative DPO with:
- Imports validation
- Backend availability
- Trainer creation
- Model/data setup
- Full training loop with Math Reward Function
"""

import os
import re
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# report_to="none" isn't honored consistently by every backend/trainer path,
# so force wandb off at the environment level to avoid a hard failure in
# CI/offline environments without a wandb API key.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# MATH REWARD FUNCTION
# =============================================================================

def extract_answer_from_completion(text):
    """Extract numerical answer from completion text."""
    # TRL conversational format
    if isinstance(text, list):
        text = text[-1]["content"]
    elif isinstance(text, dict):
        text = text["content"]

    # Look for #### format (GSM8K style)
    match = re.search(
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
    )

    if match:
        return match.group(1).replace(",", "")

    # Fallback: extract last number in text
    numbers = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text,
    )

    if numbers:
        return numbers[-1].replace(",", "")

    return None


def math_reward_function(completion, reference=None, **kwargs):
    """Score a completion based on correct mathematical answer.

    aligntune.core.rl.reward_handler.resolve_reward_call_kwargs() now calls
    custom reward functions one completion at a time, binding the completion
    text to whichever name the first positional parameter uses and aliasing
    the dataset's ground-truth column (gsm8k uses "answer", one of
    REFERENCE_ALIAS_KEYS) to `reference` if the function declares that param.
    The old batch-style `(prompts, completions, **kwargs) -> list[float]`
    signature is no longer how registry rewards are invoked - see the
    notebooks (e.g. notebooks/23_es.ipynb) for the current per-completion
    convention this now matches.
    """
    pred = extract_answer_from_completion(completion)

    try:
        return 1.0 if abs(float(pred) - float(reference)) < 1e-5 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL = os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
# DataManager.load_dataset() keeps whatever string is passed as `split` as the
# literal dict key, so HF-style "train[:4]" leaves the result keyed
# "train[:4]" instead of "train". Use split="train" + max_samples=N instead
# (see notebooks/03_online_dpo.ipynb).
DATASET_SPLIT = "train"
MAX_SAMPLES = 4
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512


# =============================================================================
# IMPORTS TEST
# =============================================================================

def test_online_dpo_imports():
    """Test Online DPO module imports."""
    try:
        from aligntune.backends.trl.rl.online_dpo.online_dpo import TRLOnlineDPOTrainer
        assert TRLOnlineDPOTrainer is not None
        print("✓ TRLOnlineDPOTrainer imported")
    except ImportError as e:
        pytest.fail(f"Failed to import TRLOnlineDPOTrainer: {e}")


def test_online_dpo_backend_availability():
    """Test Online DPO backend is available."""
    from aligntune.backends.trl.rl.online_dpo.online_dpo import TRLOnlineDPOTrainer

    is_available = TRLOnlineDPOTrainer.is_available()
    assert is_available, "Online DPO backend not available (missing TRL dependencies?)"
    print("✓ Online DPO backend available")


# =============================================================================
# TRAINER CREATION TEST
# =============================================================================

def test_online_dpo_trainer_creation():
    """Test Online DPO trainer creation with math reward function."""
    output_dir = tempfile.mkdtemp(prefix="online_dpo_test_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        reward_functions=[math_reward_function],  # Use custom reward function
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    assert trainer is not None, "Trainer creation failed!"
    assert trainer.config.model.name_or_path == MODEL
    print("✓ Online DPO trainer created with GSM8K dataset")


# =============================================================================
# SETUP TESTS
# =============================================================================

def test_online_dpo_model_setup():
    """Test Online DPO model setup."""
    output_dir = tempfile.mkdtemp(prefix="online_dpo_model_setup_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        reward_functions=[math_reward_function],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model
    trainer.setup_model()

    assert trainer.model is not None, "Policy model not initialized!"
    assert trainer.tokenizer is not None, "Tokenizer not initialized!"
    print(f"✓ Model: {type(trainer.model).__name__}")
    print(f"✓ Tokenizer: {type(trainer.tokenizer).__name__}")


def test_online_dpo_data_setup():
    """Test Online DPO data setup."""
    output_dir = tempfile.mkdtemp(prefix="online_dpo_data_setup_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        reward_functions=[math_reward_function],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model first (needed for tokenizer)
    trainer.setup_model()

    # Setup data
    trainer.setup_data()

    assert trainer.train_dataset is not None, "Train dataset not loaded!"
    assert len(trainer.train_dataset) > 0, "Train dataset is empty!"
    print(f"✓ Dataset size: {len(trainer.train_dataset)}")
    print(f"✓ Columns: {trainer.train_dataset.column_names}")


def test_online_dpo_trainer_config_setup():
    """Test Online DPO trainer config setup."""
    output_dir = tempfile.mkdtemp(prefix="online_dpo_trainer_setup_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        reward_functions=[math_reward_function],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Setup model and data
    trainer.setup_model()
    trainer.setup_data()

    # Setup DPO trainer config
    trainer.setup_trainer()

    # Check DPO config exists
    assert trainer.dpo_config is not None, "DPO config not created!"
    assert trainer.dpo_config.learning_rate == LEARNING_RATE
    print(f"✓ Online DPO config created")
    print(f"  - learning_rate: {trainer.dpo_config.learning_rate}")
    print(f"  - max_length: {trainer.dpo_config.max_length}")
    print(f"  - max_new_tokens: {trainer.dpo_config.max_new_tokens}")


def test_online_dpo_generation_params():
    """Test Online DPO generation parameters (temperature, top_p, etc)."""
    output_dir = tempfile.mkdtemp(prefix="online_dpo_gen_params_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        reward_functions=[math_reward_function],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )

    # Setup to create dpo_config
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_rewards()
    trainer.setup_trainer()

    # Check generation parameters are properly set
    assert trainer.dpo_config.temperature == 0.8, f"Expected temperature=0.8, got {trainer.dpo_config.temperature}"
    assert trainer.dpo_config.top_p == 0.95, f"Expected top_p=0.95, got {trainer.dpo_config.top_p}"
    assert trainer.dpo_config.top_k == 50, f"Expected top_k=50, got {trainer.dpo_config.top_k}"
    print(f"✓ Online DPO generation parameters:")
    print(f"  - temperature: {trainer.dpo_config.temperature}")
    print(f"  - top_p: {trainer.dpo_config.top_p}")
    print(f"  - top_k: {trainer.dpo_config.top_k}")


# =============================================================================
# FULL END-TO-END TRAINING TEST
# =============================================================================

def test_online_dpo_full_training():
    """Full end-to-end training test with actual training loop."""
    from datasets import load_dataset

    output_dir = tempfile.mkdtemp(prefix="online_dpo_training_")

    # Load GSM8K directly to preserve answer column for reward function
    print("\nLoading GSM8K dataset directly...")
    dataset = load_dataset("openai/gsm8k", "main", split="train[:4]")

    # Extract ground truth answers and prepare for Online DPO
    def extract_answer(example):
        match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", example["answer"])
        example["answer"] = match.group(1).replace(",", "") if match else None
        example["prompt"] = example["question"]  # Rename for OnlineDPO
        return example

    dataset = dataset.map(extract_answer)
    dataset = dataset.select_columns(["prompt", "answer"])  # Keep only prompt and answer
    print(f"✓ Dataset loaded with {len(dataset)} samples")
    print(f"✓ Columns: {dataset.column_names}")

    # Create trainer without dataset (we'll set manually)
    trainer = create_rl_trainer(
        model_name=MODEL,
        algorithm="online_dpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        reward_functions=[math_reward_function],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        num_train_epochs=1,
        max_steps=2,
        dataset_name=DATASET,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        
    )

    # Setup model and manually set dataset
    print("\nSetting up trainer...")
    trainer.setup_model()
    trainer.train_dataset = dataset
    trainer.eval_dataset = None
    trainer.setup_rewards()
    trainer.setup_trainer()
    print("✓ Trainer setup complete")

    # Run full training pipeline
    print("\nStarting training...")
    result = trainer.train()

    assert result is not None, "Training result is None"
    assert "status" in result, "Training result missing 'status' key"
    assert result["status"] == "success", f"Training failed with status: {result['status']}"
    assert "output_dir" in result, "Training result missing 'output_dir' key"

    print(f"\n✓ Full training completed successfully")
    print(f"  - Status: {result['status']}")
    print(f"  - Output dir: {result['output_dir']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
