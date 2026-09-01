"""
GRPO End-to-End Test with GSM8K Dataset and Custom Reward Functions

Full pipeline testing for GRPO and variants (GSPO, DAPO, DR-GRPO)
Strict testing - no leniency!
"""

import os
import re
import tempfile
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from aligntune.core.backend_factory import create_rl_trainer


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL = os.environ.get("ALIGNTUNE_TEST_MODEL", "Qwen/Qwen2.5-0.5B")
DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train[:4]"
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 512


# =============================================================================
# REWARD FUNCTIONS (TRL Schema)
# =============================================================================

def extract_answer_from_completion(text):
    """Extract numerical answer from completion text."""
    if isinstance(text, list):
        text = text[-1]["content"] if text else ""
    elif isinstance(text, dict):
        text = text.get("content", "")

    # Look for #### format (GSM8K style)
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: extract last number
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


def reasoning_length_reward(completion, **kwargs):
    """Reward function for reasoning length (per-completion, see math_answer_reward)."""
    word_count = len(completion.split())
    if word_count < 50:
        return word_count / 50.0 * 0.5
    elif word_count > 500:
        return max(0.0, 1.0 - (word_count - 500) / 500.0)
    else:
        return 1.0


def step_count_reward(completion, **kwargs):
    """Reward function for reasoning steps (per-completion, see math_answer_reward)."""
    step_matches = re.findall(r"(?:^|\n)\s*(?:Step|step)\s+\d+", completion)
    step_count = len(step_matches)

    if step_count == 0:
        return 0.0
    elif step_count < 3:
        return step_count / 3.0 * 0.7
    elif step_count <= 5:
        return 1.0
    else:
        return max(0.8, 1.0 - (step_count - 5) / 10.0)


# =============================================================================
# TEST GRPO FULL PIPELINE
# =============================================================================

def test_grpo_full_pipeline():
    """Test GRPO full pipeline with GSM8K dataset."""
    output_dir = tempfile.mkdtemp(prefix="grpo_e2e_test_")

    print("\n" + "="*80)
    print("GRPO E2E TEST - GSM8K DATASET WITH REWARD FUNCTIONS")
    print("="*80)

    # Create trainer
    print("\n[1/5] Creating GRPO trainer...")
    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm="grpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        reward_functions=[
            math_answer_reward,
            reasoning_length_reward,
            step_count_reward,
        ],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        num_train_epochs=1,
        max_steps=1,
        seed=42,
    )
    assert trainer is not None, "Trainer creation failed!"
    assert trainer.config.model.name_or_path == MODEL, "Model name mismatch!"
    print("✓ Trainer created")
    print(f"  - Model: {MODEL}")
    print(f"  - Dataset: {DATASET}")
    print(f"  - Algorithm: grpo")

    # Setup model
    print("\n[2/5] Setting up model...")
    trainer.setup_model()
    assert trainer.model is not None, "Model not initialized!"
    assert trainer.tokenizer is not None, "Tokenizer not initialized!"
    print(f"✓ Model: {type(trainer.model).__name__}")
    print(f"✓ Tokenizer: {type(trainer.tokenizer).__name__}")

    # Setup data
    print("\n[3/5] Setting up data...")
    trainer.setup_data()
    assert trainer.train_dataset is not None, "Train dataset not loaded!"
    assert len(trainer.train_dataset) > 0, "Train dataset is empty!"
    print(f"✓ Dataset size: {len(trainer.train_dataset)}")
    print(f"✓ Columns: {trainer.train_dataset.column_names}")

    # Check dataset has required columns
    sample = trainer.train_dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    assert "prompt" in sample or "query" in sample, \
        f"No prompt column found! Keys: {list(sample.keys())}"
    print("✓ Dataset has required columns")

    # Setup rewards
    print("\n[4/5] Setting up reward functions...")
    trainer.setup_rewards()
    assert hasattr(trainer, 'reward_functions'), "No reward_functions attribute!"
    assert len(trainer.reward_functions) == 3, \
        f"Expected 3 rewards, got {len(trainer.reward_functions)}"
    print(f"✓ Loaded {len(trainer.reward_functions)} reward functions")
    print(f"  - math_answer_reward")
    print(f"  - reasoning_length_reward")
    print(f"  - step_count_reward")

    # Test reward functions with sample
    # (per-completion signature now, not the old batch (prompts, completions)
    # convention - see math_answer_reward's docstring above)
    print("\n[4.1] Testing reward functions on sample...")
    test_completion = "Step 1: Add 2+2\nStep 2: Result is 4\n#### 4"
    test_answer = "4"

    math_reward = math_answer_reward(test_completion, reference=test_answer)
    length_reward = reasoning_length_reward(test_completion)
    step_reward = step_count_reward(test_completion)

    assert math_reward == 1.0, f"Expected math reward 1.0, got {math_reward}"
    assert length_reward > 0.0, f"Expected positive length reward, got {length_reward}"
    assert step_reward > 0.0, f"Expected positive step reward, got {step_reward}"

    print(f"✓ Reward functions working correctly:")
    print(f"  - Math reward: {math_reward:.2f}")
    print(f"  - Length reward: {length_reward:.2f}")
    print(f"  - Step reward: {step_reward:.2f}")

    # Setup trainer config
    print("\n[5/5] Setting up GRPO trainer config...")
    trainer.setup_trainer()
    assert trainer.trainer is not None, "GRPO trainer not created!"
    assert hasattr(trainer.trainer, 'train'), "Trainer missing train method!"
    print(f"✓ GRPO trainer config created")
    print(f"  - loss_type: grpo")
    print(f"  - num_generations: {trainer.trainer.args.num_generations if hasattr(trainer.trainer.args, 'num_generations') else 'N/A'}")
    print(f"  - output_dir: {output_dir}")

    # Run training
    print("\n[BONUS] Running GRPO training (1 step)...")
    result = trainer.train()
    assert result is not None, "Training returned None!"
    print(f"✓ Training completed successfully")
    print(f"  - Status: {result.get('status', 'success')}")
    print(f"  - Output dir: {result.get('model_path', output_dir)}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)


# =============================================================================
# TEST GRPO VARIANTS
# =============================================================================

@pytest.mark.parametrize("algorithm", [
    "grpo",
    "gspo",
    "dapo",
    "drgrpo",
])
def test_grpo_variants_trl(algorithm):
    """Test GRPO variants with TRL backend."""
    output_dir = tempfile.mkdtemp(prefix=f"grpo_{algorithm}_trl_test_")

    print(f"\n{'='*80}")
    print(f"TRL GRPO VARIANT TEST: {algorithm.upper()}")
    print(f"{'='*80}")

    # Create trainer
    print(f"\n[1/3] Creating {algorithm.upper()} trainer with TRL backend...")
    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm=algorithm,
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        reward_functions=[
            math_answer_reward,
            reasoning_length_reward,
            step_count_reward,
        ],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        num_train_epochs=1,
        max_steps=1,
        seed=42,
    )
    assert trainer is not None, f"{algorithm.upper()} trainer creation failed!"
    print(f"✓ {algorithm.upper()} trainer created")

    # Setup all components
    print(f"\n[2/3] Setting up {algorithm.upper()} components...")
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_rewards()
    trainer.setup_trainer()

    assert trainer.model is not None, "Model not initialized!"
    assert trainer.train_dataset is not None, "Dataset not loaded!"
    assert len(trainer.reward_functions) == 3, "Expected 3 rewards!"
    print(f"✓ {algorithm.upper()} setup complete with {len(trainer.reward_functions)} rewards")

    # Run training
    print(f"\n[3/3] Running {algorithm.upper()} training...")
    result = trainer.train()
    assert result is not None, f"{algorithm.upper()} training returned None!"
    print(f"✓ {algorithm.upper()} training completed successfully")

    print(f"\n{'='*80}")
    print(f"✓ {algorithm.upper()} TEST PASSED!")
    print(f"{'='*80}")


def test_grpo_with_multiple_rewards():
    """Test GRPO with all three reward functions."""
    output_dir = tempfile.mkdtemp(prefix="grpo_multi_rewards_test_")

    print(f"\n{'='*80}")
    print(f"GRPO TEST: Multiple Reward Functions")
    print(f"{'='*80}")

    # Create trainer
    print(f"\nCreating GRPO trainer with 3 rewards...")
    trainer = create_rl_trainer(
        model_name=MODEL,
        dataset_name=DATASET,
        config_name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        algorithm="grpo",
        backend="trl",
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        use_peft=True,
        lora_r=8,
        reward_functions=[
            math_answer_reward,
            reasoning_length_reward,
            step_count_reward,
        ],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        num_train_epochs=1,
        max_steps=1,
        seed=42,
    )
    assert trainer is not None, "GRPO trainer creation failed!"
    print(f"✓ GRPO trainer created with 3 rewards")

    # Setup all components
    print(f"\nSetting up GRPO components...")
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_rewards()
    trainer.setup_trainer()

    assert trainer.model is not None, "Model not initialized!"
    assert trainer.train_dataset is not None, "Dataset not loaded!"
    assert len(trainer.reward_functions) == 3, "Expected 3 rewards!"
    print(f"✓ GRPO setup complete with {len(trainer.reward_functions)} rewards")

    # Run training
    print(f"\nRunning GRPO training with multiple rewards...")
    result = trainer.train()
    assert result is not None, "GRPO training returned None!"
    print(f"✓ GRPO training completed successfully")

    print(f"\n{'='*80}")
    print(f"✓ GRPO MULTI-REWARD TEST PASSED!")
    print(f"{'='*80}")


# =============================================================================
# TEST REWARD FUNCTIONS
# =============================================================================

def test_math_answer_reward_function():
    """Test math answer reward function."""
    print("\n[UNIT] Testing math_answer_reward function...")

    prompts = ["What is 2+2?", "What is 5+5?", "What is 10/2?"]
    completions = [
        "2+2 equals #### 4",
        "5+5 is #### 10",
        "10/2 gives #### 5",
    ]
    answers = ["4", "10", "5"]

    rewards = [
        math_answer_reward(completion, reference=answer)
        for completion, answer in zip(completions, answers)
    ]

    assert len(rewards) == 3, f"Expected 3 rewards, got {len(rewards)}"
    assert all(r == 1.0 for r in rewards), f"Expected all 1.0, got {rewards}"
    print(f"✓ Math answer reward: all correct answers recognized")


def test_reasoning_length_reward_function():
    """Test reasoning length reward function."""
    print("\n[UNIT] Testing reasoning_length_reward function...")

    prompts = ["Q1", "Q2", "Q3"]
    completions = [
        "Short answer",  # Too short
        " ".join(["word"] * 200),  # Optimal
        " ".join(["word"] * 800),  # Too long
    ]

    rewards = [reasoning_length_reward(completion) for completion in completions]

    assert len(rewards) == 3, f"Expected 3 rewards, got {len(rewards)}"
    assert 0.0 < rewards[0] < 1.0, f"Short should be partial, got {rewards[0]}"
    assert rewards[1] == 1.0, f"Optimal should be 1.0, got {rewards[1]}"
    assert 0.0 < rewards[2] < 1.0, f"Long should be partial, got {rewards[2]}"
    print(f"✓ Reasoning length reward: correct scoring")


def test_step_count_reward_function():
    """Test step count reward function."""
    print("\n[UNIT] Testing step_count_reward function...")

    prompts = ["Q1", "Q2", "Q3", "Q4"]
    completions = [
        "No steps here",  # 0 steps
        "Step 1: Start\nStep 2: Middle",  # 2 steps
        "Step 1: A\nStep 2: B\nStep 3: C\nStep 4: D",  # 4 steps
        "Step 1: A\nStep 2: B\nStep 3: C\nStep 4: D\nStep 5: E\nStep 6: F",  # 6 steps
    ]

    rewards = [step_count_reward(completion) for completion in completions]

    assert len(rewards) == 4, f"Expected 4 rewards, got {len(rewards)}"
    assert rewards[0] == 0.0, f"No steps should be 0.0, got {rewards[0]}"
    assert 0.0 < rewards[1] < 1.0, f"2 steps should be partial, got {rewards[1]}"
    assert rewards[2] == 1.0, f"4 steps should be 1.0, got {rewards[2]}"
    assert 0.0 < rewards[3] < 1.0, f"6 steps should be partial, got {rewards[3]}"
    print(f"✓ Step count reward: correct scoring")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
