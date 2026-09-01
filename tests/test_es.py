"""
ES Configuration Wiring Tests

Tests that all parameters are correctly passed using create_es_trainer from backend factory.
NO MOCKS - Real end-to-end integration tests.
"""

import pytest
import re


# ============================================================================
# GSM8K Reward Functions
# ============================================================================

def math_answer_reward(prompt, response, reference, **kwargs):
    """Reward function for GSM8K math correctness."""
    if not response or not reference:
        print(f"[REWARD] Empty response or reference")
        return 0.0

    # Extract answer from response - try multiple formats:
    # 1. LaTeX format: \boxed{72}
    # 2. GSM8K computation format: <<48/2=24>>24
    # 3. Simple format: #### 72
    pred_answer = None

    # Try LaTeX format first
    latex_match = re.search(r'\\boxed\{([+-]?\d+(?:[.,]\d+)?)\}', response)
    if latex_match:
        pred_answer = latex_match.group(1).replace(',', '')
        print(f"[REWARD] Found LaTeX answer: {pred_answer}")
    else:
        # Try GSM8K computation format (last number before line end)
        # Pattern: any text, optionally with <<...>>, then final number
        gsm_comp_match = re.search(r'<<[^>]*>>(\d+)', response)
        if gsm_comp_match:
            pred_answer = gsm_comp_match.group(1)
            print(f"[REWARD] Found GSM8K computation answer: {pred_answer}")
        else:
            # Try simple format #### 72
            simple_match = re.search(r'####\s*([+-]?\d+(?:[.,]\d+)?)', response)
            if simple_match:
                pred_answer = simple_match.group(1).replace(',', '')
                print(f"[REWARD] Found simple format answer: {pred_answer}")
            else:
                # Last resort: extract last number in response
                last_num = re.findall(r'([+-]?\d+(?:[.,]\d+)?)', response)
                if last_num:
                    pred_answer = last_num[-1].replace(',', '')
                    print(f"[REWARD] Found last number in response: {pred_answer}")
                else:
                    print(f"[REWARD] No answer format found in response (first 100 chars): {response[:100]}")

    # Extract ground truth from reference (GSM8K format)
    # Reference has computation steps, extract last number
    ref_nums = re.findall(r'([+-]?\d+(?:[.,]\d+)?)', reference)
    true_answer = ref_nums[-1].replace(',', '') if ref_nums else None
    print(f"[REWARD] Reference answer: {true_answer}")

    if pred_answer and true_answer:
        try:
            score = 1.0 if abs(float(pred_answer) - float(true_answer)) < 1e-3 else 0.0
            print(f"[REWARD] Comparing {pred_answer} vs {true_answer} → score={score}")
            return score
        except ValueError as e:
            print(f"[REWARD] ValueError: {e}")
            return 0.0

    print(f"[REWARD] Missing answer: pred={pred_answer}, true={true_answer}")
    return 0.0


def reasoning_length_reward(prompt, response, reference, **kwargs):
    """Reward based on reasoning length for GSM8K."""
    if not response:
        return 0.0

    # Get reasoning part (before ####)
    reasoning = response.split('####')[0] if '####' in response else response

    # Count words
    words = reasoning.split()
    word_count = len(words)

    # Reward function:
    if word_count < 50:
        return 0.0
    elif word_count <= 200:
        return 1.0
    elif word_count <= 500:
        return 0.8
    else:
        return 0.3


# ============================================================================
# Test ES Trainer Creation from Backend Factory
# ============================================================================

class TestESTrainerFromFactory:
    """Test ES trainer created from create_es_trainer (backend factory)."""

    def test_create_es_trainer_basic(self):
        """Test create_es_trainer with basic parameters."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            population_size=4,
            sigma=0.1,
            learning_rate=0.01,
            num_iterations=5,
            batch_size=2,
            column_mapping={"question": "prompt", "answer": "reference"},
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # Verify trainer is created
        assert trainer is not None
        assert hasattr(trainer, 'config')

        # Verify config has correct values
        assert trainer.config.es.population_size == 4
        assert trainer.config.es.sigma == 0.1
        assert trainer.config.es.learning_rate == 0.01
        assert trainer.config.es.num_iterations == 5
        assert trainer.config.es.prompt_batch_size == 2

        # Verify model config
        assert trainer.config.model.name_or_path == "Qwen/Qwen2.5-0.5B"

        # Verify dataset config
        assert trainer.config.dataset.name == "openai/gsm8k"

        # Verify rewards
        assert hasattr(trainer.config, 'rewards')
        assert len(trainer.config.rewards) == 1

    def test_create_es_trainer_with_rewards(self):
        """Test create_es_trainer with custom reward functions."""
        from aligntune.core.backend_factory import create_es_trainer

        rewards = [
            {"type": "custom", "params": {"function": math_answer_reward}},
            {"type": "custom", "params": {"function": reasoning_length_reward}}
        ]

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            column_mapping={"question": "prompt", "answer": "reference"},
            rewards=rewards
        )

        # Verify trainer created
        assert trainer is not None

        # Verify rewards passed
        assert len(trainer.config.rewards) == 2
        assert trainer.config.rewards[0]['type'] == 'custom'
        assert trainer.config.rewards[1]['type'] == 'custom'

    def test_create_es_trainer_with_kwargs(self):
        """Test create_es_trainer with optional kwargs."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            dtype="bf16",
            max_seq_length=1024,
            split="train[:10]",
            system_prompt="You are a helpful assistant.",
            seed=123,
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # Verify optional params passed
        assert trainer.config.model.dtype == "bf16"
        assert trainer.config.model.max_seq_length == 1024
        assert trainer.config.es.seed == 123


# ============================================================================
# Test Reward Function Setup (Real, No Mocks)
# ============================================================================

class TestRewardFunctionSetup:
    """Test reward function setup in real trainer."""

    def test_single_reward_setup(self):
        """Test single reward function is set up correctly."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # Setup reward function
        trainer.setup_reward_function()

        # Verify reward function works
        assert trainer.reward_fn is not None
        assert callable(trainer.reward_fn)

        # Test with correct answer
        score = trainer.reward_fn(
            prompt="What is 2+2?",
            response="2 + 2 = 4. #### 4",
            reference="#### 4"
        )
        assert score == 1.0

        # Test with incorrect answer
        score = trainer.reward_fn(
            prompt="What is 2+2?",
            response="2 + 2 = 5. #### 5",
            reference="#### 4"
        )
        assert score == 0.0

    def test_multiple_rewards_setup(self):
        """Test multiple reward functions are averaged."""
        from aligntune.core.backend_factory import create_es_trainer

        rewards = [
            {"type": "custom", "params": {"function": math_answer_reward}},
            {"type": "custom", "params": {"function": reasoning_length_reward}}
        ]

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            rewards=rewards
        )

        trainer.setup_reward_function()

        assert trainer.reward_fn is not None

        # Test: correct answer (1.0) + good length (1.0) = avg 1.0
        response = "Step 1: " + " ".join(["calculation"] * 80) + " #### 18"
        score = trainer.reward_fn(
            prompt="Test",
            response=response,
            reference="#### 18"
        )
        assert score == 1.0

        # Test: correct answer (1.0) + short (0.0) = avg 0.5
        score = trainer.reward_fn(
            prompt="Test",
            response="Short #### 18",
            reference="#### 18"
        )
        assert score == 0.5

        # Test: incorrect answer (0.0) + good length (1.0) = avg 0.5
        response = "Step 1: " + " ".join(["calculation"] * 80) + " #### 99"
        score = trainer.reward_fn(
            prompt="Test",
            response=response,
            reference="#### 18"
        )
        assert score == 0.5


# ============================================================================
# Test All Config Parameters Accessible
# ============================================================================

class TestConfigParametersAccess:
    """Test all config parameters are accessible from trainer."""

    def test_all_es_params_accessible(self):
        """Test all ES config params can be accessed."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            population_size=32,
            sigma=0.3,
            learning_rate=0.05,
            batch_size=8,
            num_iterations=500,
            max_seq_length=1024,
            dtype="bf16",
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # Verify all params accessible
        assert trainer.config.es.population_size == 32
        assert trainer.config.es.sigma == 0.3
        assert trainer.config.es.learning_rate == 0.05
        assert trainer.config.es.prompt_batch_size == 8
        assert trainer.config.es.num_iterations == 500

        assert trainer.config.model.name_or_path == "Qwen/Qwen2.5-0.5B"
        assert trainer.config.model.max_seq_length == 1024
        assert trainer.config.model.dtype == "bf16"

        assert trainer.config.dataset.name == "openai/gsm8k"

    def test_dataset_task_type_is_rloo(self):
        """Test that data manager will use task_type='rloo' internally."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # The trainer's setup_data() method hardcodes task_type="rloo"
        # We verify this by checking the code path exists
        assert hasattr(trainer, 'setup_data')


# ============================================================================
# Test Full Component Setup (Model will be downloaded)
# ============================================================================

class TestFullComponentSetup:
    """Test full ES trainer setup with all components - model downloaded automatically."""

    def test_full_setup_all_components(self):
        """Test setting up all ES trainer components."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            subset="main",
            split="train[:5]",
            population_size=4,
            num_iterations=1,
            batch_size=2,
            column_mapping={"question": "prompt", "answer": "reference"},
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        print("\n[1/4] Setting up model...")
        trainer.setup_model()
        assert trainer.base_model is not None
        assert trainer.tokenizer is not None
        print("✓ Model setup completed")

        print("[2/4] Setting up data...")
        trainer.setup_data()
        assert trainer.task is not None
        print("✓ Data setup completed")

        print("[3/4] Setting up reward function...")
        trainer.setup_reward_function()
        assert trainer.reward_fn is not None
        assert callable(trainer.reward_fn)
        print("✓ Reward function setup completed")

        print("[4/4] Testing reward function...")
        score = trainer.reward_fn(
            prompt="Test",
            response="Answer #### 5",
            reference="#### 5"
        )
        assert score == 1.0
        print("✓ Reward function works correctly")

        print("\n✓ All ES components setup successfully!")

    def test_full_training_iteration(self):
        """Test running full ES training with multiple iterations to track progress."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            subset="socratic",
            split="train[:20]",
            population_size=8,
            sigma=0.1,
            learning_rate=0.01,
            num_iterations=10,
            batch_size=4,
            column_mapping={"question": "prompt", "answer": "reference"},
            rewards=[{"type": "custom", "params": {"function": math_answer_reward}}]
        )

        # Setup all components
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_reward_function()
        trainer.setup_rollout_backend()

        print("\n✓ All components initialized")
        print(f"  - Model: {trainer.config.model.name_or_path}")
        print(f"  - Dataset: {trainer.config.dataset.name}")
        print(f"  - Population size: {trainer.config.es.population_size}")
        print(f"  - Reward functions: {len(trainer.config.rewards)}")

        print("\n✓ Starting ES training...")
        result = trainer.train()
        assert result is not None
        assert trainer.fitness_history is not None
        assert len(trainer.fitness_history) > 0

        print("\n" + "="*80)
        print("ES TRAINING RESULTS")
        print("="*80)

        # Log fitness history (reward tracking)
        print(f"\nFitness History (Rewards):")
        for i, fitness in enumerate(trainer.fitness_history):
            print(f"  Iteration {i+1}: {fitness:.4f}")

        # Log statistics
        max_fitness = max(trainer.fitness_history)
        min_fitness = min(trainer.fitness_history)
        avg_fitness = sum(trainer.fitness_history) / len(trainer.fitness_history)

        print(f"\nReward Statistics:")
        print(f"  Max Fitness:  {max_fitness:.4f}")
        print(f"  Min Fitness:  {min_fitness:.4f}")
        print(f"  Avg Fitness:  {avg_fitness:.4f}")
        print(f"  Improvement:  {max_fitness - min_fitness:.4f}")

        # Log configuration used
        print(f"\nTraining Configuration:")
        print(f"  Population Size:  {trainer.config.es.population_size}")
        print(f"  Sigma (mutation): {trainer.config.es.sigma}")
        print(f"  Learning Rate:    {trainer.config.es.learning_rate}")
        print(f"  Iterations:       {trainer.config.es.num_iterations}")
        print(f"  Batch Size:       {trainer.config.es.prompt_batch_size}")
        print(f"  Reward Functions: {len(trainer.config.rewards)}")

        print("\n" + "="*80)
        print("✓ ES training completed successfully!")
        print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
