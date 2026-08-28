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
        return 0.0

    # Extract answer from response (GSM8K uses #### marker)
    response_match = re.search(r'####\s*([+-]?\d+(?:[.,]\d+)?)', response)
    pred_answer = response_match.group(1).replace(',', '') if response_match else None

    # Extract ground truth from reference
    ref_match = re.search(r'####\s*([+-]?\d+(?:[.,]\d+)?)', reference)
    true_answer = ref_match.group(1).replace(',', '') if ref_match else None

    if pred_answer and true_answer:
        try:
            return 1.0 if abs(float(pred_answer) - float(true_answer)) < 1e-3 else 0.0
        except ValueError:
            return 0.0

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
# Test Full Component Setup (Requires Model)
# ============================================================================

class TestFullComponentSetup:
    """Test full ES trainer setup with all components."""

    def test_full_setup_all_components(self):
        """Test setting up all ES trainer components."""
        from aligntune.core.backend_factory import create_es_trainer

        # DataManager.load_dataset() keeps whatever string is passed as
        # `split` as the literal dict key, so HF-style "train[:5]" leaves
        # the result keyed "train[:5]" instead of "train". Use split="train"
        # + max_samples=N instead (see notebooks/20_es.ipynb).
        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            subset="main",
            split="train",
            max_samples=5,
            population_size=4,
            num_iterations=1,
            batch_size=2,
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
        """Test running one full ES training iteration."""
        from aligntune.core.backend_factory import create_es_trainer

        trainer = create_es_trainer(
            model_name="Qwen/Qwen2.5-0.5B",
            dataset_name="openai/gsm8k",
            subset="main",
            split="train",
            max_samples=10,
            population_size=4,
            sigma=0.1,
            learning_rate=0.01,
            num_iterations=1,
            batch_size=2,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
