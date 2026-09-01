"""
GBMPO Tests - Backend Factory Integration

Tests GBMPO with backend factory to verify params flow correctly.
"""

import os
import re
import tempfile
import time
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def extract_answer_from_completion(text):
    """Extract numerical answer from completion text."""
    if isinstance(text, list):
        text = text[-1]["content"] if text else ""
    elif isinstance(text, dict):
        text = text.get("content", "")

    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def math_answer_reward(completion, reference=None, **kwargs):
    """Reward function for math answer correctness.

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
    if reference is None:
        return 0.0
    try:
        pred = extract_answer_from_completion(completion)
        if pred is None:
            return 0.0
        pred_num = float(pred)
        gt_num = float(str(reference).replace(",", ""))
        return 1.0 if abs(pred_num - gt_num) < 1e-5 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# TEST CONFIG
# =============================================================================

MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train[:4]"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512


# =============================================================================
# TESTS
# =============================================================================

def test_gbmpo_imports():
    """Test GBMPO module imports."""
    from aligntune.backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
    assert TRLGBMPOTrainer is not None
    print("✓ TRLGBMPOTrainer imported")


def test_gbmpo_backend_availability():
    """Test GBMPO backend is available."""
    from aligntune.backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
    is_available = TRLGBMPOTrainer.is_available()
    assert is_available, "GBMPO backend not available"
    print("✓ GBMPO backend available")


def test_gbmpo_trainer_creation():
    """Test creating GBMPO trainer with GBMPO-specific parameters."""
    output_dir = tempfile.mkdtemp(prefix="gbmpo_test_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm="gbmpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        gbmpo_divergence_type="l2",
        gbmpo_l2_coefficient=0.0001,
        reward_functions=[math_answer_reward],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    assert trainer is not None
    assert trainer.config.model.name_or_path == MODEL
    print("✓ GBMPO trainer created")


def test_gbmpo_params_passed():
    """Test that GBMPO-specific params are correctly passed."""
    output_dir = tempfile.mkdtemp(prefix="gbmpo_params_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm="gbmpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        gbmpo_divergence_type="l2kl",
        gbmpo_l2_coefficient=0.0005,
        reward_functions=[math_answer_reward],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Check params in config.train
    assert hasattr(trainer.config.train, 'gbmpo_divergence_type')
    assert hasattr(trainer.config.train, 'gbmpo_l2_coefficient')
    assert trainer.config.train.gbmpo_divergence_type == "l2kl"
    assert trainer.config.train.gbmpo_l2_coefficient == 0.0005

    print("✓ GBMPO params in config.train")
    print(f"  gbmpo_divergence_type: {trainer.config.train.gbmpo_divergence_type}")
    print(f"  gbmpo_l2_coefficient: {trainer.config.train.gbmpo_l2_coefficient}")


def test_gbmpo_trainer_setup():
    """Test GBMPO internal trainer setup with patched loss."""
    output_dir = tempfile.mkdtemp(prefix="gbmpo_setup_")

    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm="gbmpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        use_peft=True,
        lora_r=8,
        gbmpo_divergence_type="l2",
        gbmpo_l2_coefficient=0.0001,
        reward_functions=[math_answer_reward],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_rewards()
    trainer.setup_trainer()

    # Verify internal trainer was created
    assert hasattr(trainer, 'trainer')
    assert trainer.trainer is not None

    # Verify GBMPO params stored on trainer
    assert hasattr(trainer.trainer, 'divergence_type')
    assert hasattr(trainer.trainer, 'l2_coefficient')
    assert trainer.trainer.divergence_type == "l2"
    assert trainer.trainer.l2_coefficient == 0.0001

    print("✓ Internal trainer patched successfully")
    print(f"  divergence_type: {trainer.trainer.divergence_type}")
    print(f"  l2_coefficient: {trainer.trainer.l2_coefficient}")


def test_gbmpo_full_training(divergence_type="l2", l2_coefficient=0.0001, beta=0.1):
    """Full end-to-end training test."""
    from datasets import load_dataset

    output_dir = tempfile.mkdtemp(prefix=f"gbmpo_{divergence_type}_")

    print(f"\n{'='*60}")
    print(f"Testing GBMPO ({divergence_type})")
    print('='*60)

    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train[:4]")

    def extract_answer(example):
        match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", example["answer"])
        example["answer"] = match.group(1).replace(",", "") if match else "0"
        example["prompt"] = example["question"]
        return example

    dataset = dataset.map(extract_answer)
    dataset = dataset.select_columns(["prompt", "answer"])

    # Create trainer
    trainer = create_rl_trainer(
        model_name=MODEL,
        algorithm="gbmpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        gbmpo_divergence_type=divergence_type,
        gbmpo_l2_coefficient=l2_coefficient,
        beta=beta,
        reward_functions=[math_answer_reward],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        num_train_epochs=1,
        max_steps=2,
        dataset_name=DATASET,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
    )

    # Setup
    trainer.setup_model()
    trainer.train_dataset = dataset
    trainer.eval_dataset = None
    trainer.setup_rewards()
    trainer.setup_trainer()

    # Verify params
    print(f"\n✓ Config.train.gbmpo_divergence_type: {trainer.config.train.gbmpo_divergence_type}")
    print(f"✓ Config.train.gbmpo_l2_coefficient: {trainer.config.train.gbmpo_l2_coefficient}")
    print(f"✓ Trainer.divergence_type: {trainer.trainer.divergence_type}")
    print(f"✓ Trainer.l2_coefficient: {trainer.trainer.l2_coefficient}")

    assert trainer.trainer.divergence_type == divergence_type
    assert trainer.trainer.l2_coefficient == l2_coefficient

    # Train
    start_time = time.time()
    for cb in trainer.get_hf_callbacks():
        trainer.trainer.add_callback(cb)

    result = trainer.trainer.train()
    duration = time.time() - start_time

    assert result is not None
    print(f"\n✓ GBMPO ({divergence_type}) training completed ({duration:.2f}s)")


def test_gbmpo_l2():
    """Test L2 variant."""
    test_gbmpo_full_training(divergence_type="l2", l2_coefficient=0.0001, beta=0.0)


def test_gbmpo_l2kl():
    """Test L2KL variant."""
    test_gbmpo_full_training(divergence_type="l2kl", l2_coefficient=0.0001, beta=0.1)


def test_gbmpo_prob_l2():
    """Test ProbL2 variant."""
    test_gbmpo_full_training(divergence_type="prob_l2", l2_coefficient=0.0001, beta=0.0)


def test_gbmpo_prob_l2kl():
    """Test ProbL2KL variant."""
    test_gbmpo_full_training(divergence_type="prob_l2kl", l2_coefficient=0.0001, beta=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
