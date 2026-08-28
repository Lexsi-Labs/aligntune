"""
SPIN End-to-End Test with Alpaca Dataset

Strict testing - no leniency!
"""

import os
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL = os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = "yahma/alpaca-cleaned"
# DataManager.load_dataset() keeps whatever string is passed as `split` as the
# literal dict key, so HF-style "train[:8]" leaves the result keyed
# "train[:8]" instead of "train". Use split="train" + max_samples=N instead
# (see notebooks/18_spin.ipynb).
DATASET_SPLIT = "train"
MAX_SAMPLES = 8
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512


# =============================================================================
# TEST SPIN FULL PIPELINE
# =============================================================================

def test_spin_full_pipeline():
    """Test SPIN full pipeline with alpaca dataset."""
    output_dir = tempfile.mkdtemp(prefix="spin_e2e_test_")

    print("\n" + "="*80)
    print("SPIN E2E TEST - ALPACA DATASET")
    print("="*80)

    # Create trainer
    print("\n[1/6] Creating SPIN trainer...")
    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        algorithm="spin",
        backend="trl",
        output_dir=output_dir,
        num_rounds=2,  # 2 rounds for testing
        dpo_steps_per_round=5,  # 5 steps per round
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
    assert trainer is not None, "Trainer creation failed!"
    print("✓ Trainer created")

    # Setup model
    print("\n[2/6] Setting up model...")
    trainer.setup_model()
    assert trainer.model is not None, "Model not initialized!"
    assert trainer.tokenizer is not None, "Tokenizer not initialized!"
    assert trainer.opponent_checkpoint_dir is not None, "Opponent checkpoint not initialized!"
    print(f"✓ Model: {type(trainer.model).__name__}")
    print(f"✓ Tokenizer: {type(trainer.tokenizer).__name__}")
    print(f"✓ Opponent checkpoint: {trainer.opponent_checkpoint_dir}")

    # Setup data
    print("\n[3/6] Setting up data...")
    trainer.setup_data()
    assert trainer.sft_dataset is not None, "SFT dataset not loaded!"
    assert len(trainer.sft_dataset) > 0, "SFT dataset is empty!"
    print(f"✓ Dataset size: {len(trainer.sft_dataset)}")
    print(f"✓ Columns: {trainer.sft_dataset.column_names}")

    # Check dataset has required columns
    sample = trainer.sft_dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")

    # Verify prompt/response columns exist
    has_prompt = any(col in sample for col in ['prompt', 'question', 'text', 'instruction'])
    has_response = any(col in sample for col in ['chosen', 'answer', 'response', 'output', 'completion'])
    assert has_prompt, f"No prompt column found in dataset! Keys: {list(sample.keys())}"
    assert has_response, f"No response column found in dataset! Keys: {list(sample.keys())}"
    print("✓ Dataset has required columns")

    # Setup DPO trainer config
    print("\n[4/6] Setting up DPO trainer config...")
    trainer.setup_trainer()
    assert trainer.dpo_config is not None, "DPO config not created!"
    assert trainer.dpo_config.max_steps == 5, f"Expected max_steps=5, got {trainer.dpo_config.max_steps}"
    # Attribute may be absent on trl>=1.0, which dropped max_prompt_length from
    # DPOConfig in favor of max_length + truncation_mode.
    prompt_len = getattr(trainer.dpo_config, "max_prompt_length", trainer.dpo_config.max_length // 2)
    assert prompt_len < trainer.dpo_config.max_length, \
        f"max_prompt_length ({prompt_len}) must be < max_length ({trainer.dpo_config.max_length})"
    print(f"✓ DPO config created")
    print(f"  - max_steps: {trainer.dpo_config.max_steps}")
    print(f"  - max_length: {trainer.dpo_config.max_length}")
    print(f"  - max_prompt_length: {prompt_len}")
    print(f"  - learning_rate: {trainer.dpo_config.learning_rate}")

    # Test preference pair generation
    print("\n[5/6] Testing preference pair generation...")
    preference_dataset = trainer.create_preference_pairs()
    assert preference_dataset is not None, "Preference dataset creation failed!"
    assert len(preference_dataset) > 0, "Preference dataset is empty!"
    assert "prompt" in preference_dataset.column_names, "Missing 'prompt' column!"
    assert "chosen" in preference_dataset.column_names, "Missing 'chosen' column!"
    assert "rejected" in preference_dataset.column_names, "Missing 'rejected' column!"
    print(f"✓ Generated {len(preference_dataset)} preference pairs")
    print(f"✓ Columns: {preference_dataset.column_names}")

    # Verify preference pair structure
    pair_sample = preference_dataset[0]
    assert isinstance(pair_sample['prompt'], str), "Prompt should be string!"
    assert isinstance(pair_sample['chosen'], str), "Chosen should be string!"
    assert isinstance(pair_sample['rejected'], str), "Rejected should be string!"
    assert len(pair_sample['prompt']) > 0, "Prompt is empty!"
    assert len(pair_sample['chosen']) > 0, "Chosen is empty!"
    assert len(pair_sample['rejected']) > 0, "Rejected is empty!"
    print(f"✓ Sample pair structure valid")
    print(f"  - Prompt length: {len(pair_sample['prompt'])}")
    print(f"  - Chosen length: {len(pair_sample['chosen'])}")
    print(f"  - Rejected length: {len(pair_sample['rejected'])}")

    # Test full training (2 rounds)
    print("\n[6/6] Testing full SPIN training (2 rounds)...")
    result = trainer.train()
    assert result is not None, "Training returned None!"
    assert "metrics" in result, "Training result missing 'metrics'!"
    assert "round_1" in result["metrics"], "Missing round_1 metrics!"
    assert "round_2" in result["metrics"], "Missing round_2 metrics!"
    print(f"✓ Training completed successfully")
    print(f"✓ Rounds completed: {len(result['metrics'])}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
