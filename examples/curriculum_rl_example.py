"""
Example: Using Curriculum Learning with AlignTune RLHF Training

This example demonstrates how to use adaptive curriculum learning with TAROT-inspired
difficulty sampling during RLHF training. The curriculum sampler tracks per-example
pass rates and adaptively adjusts which examples are sampled based on training progress.

Paper: https://huggingface.co/papers/2602.15449
"""

import torch
from aligntune.data.curriculum import CurriculumSampler, CurriculumStrategy
from aligntune.core.callbacks.curriculum_callback import CurriculumCallback
from aligntune.core.rl.config import UnifiedConfig, TrainingConfig, ModelConfig, DatasetConfig


# ==============================================================================
# EXAMPLE 1: Basic Curriculum Learning Setup
# ==============================================================================

def example_basic_curriculum():
    """
    Basic example: Setup curriculum learning with adaptive strategy.
    """
    # Create curriculum sampler
    dataset_size = 1000
    sampler = CurriculumSampler(
        dataset_size=dataset_size,
        strategy="adaptive",  # Options: easy_first, adaptive, mixed
        warmup_steps=500,     # Steps before curriculum takes effect
        seed=42
    )

    # Create callback to integrate with trainer
    callback = CurriculumCallback(
        curriculum_sampler=sampler,
        pass_threshold=0.5,   # Reward threshold for "passed"
        update_interval=10,   # Log stats every N steps
        log_dir="./curriculum_logs"
    )

    print("Curriculum sampler initialized")
    print(f"  Strategy: {sampler.strategy}")
    print(f"  Dataset size: {sampler.dataset_size}")
    print(f"  Warmup steps: {sampler.warmup_steps}")

    return sampler, callback


# ==============================================================================
# EXAMPLE 2: Curriculum Sampling During Training
# ==============================================================================

def example_curriculum_sampling():
    """
    Example: How curriculum sampling works during training.
    """
    # Create sampler with mixed strategy
    sampler = CurriculumSampler(
        dataset_size=100,
        strategy="mixed",
        warmup_steps=50,
    )

    # Simulate training loop
    for step in range(200):
        # Sample batch
        batch_indices = sampler.sample(batch_size=32, current_step=step)

        # Simulate training on batch
        # In real training, rewards come from the RL algorithm
        import numpy as np
        rewards = np.random.randn(32).tolist()  # Example rewards

        # Update pass rates based on rewards
        # (In real training, this is done by CurriculumCallback)
        for example_id, reward in zip(batch_indices, rewards):
            passed = reward > 0.5  # Threshold for "success"
            sampler.update_pass_rate(example_id, passed)

        # Log progress periodically
        if step % 50 == 0:
            progress = sampler.get_curriculum_progress()
            print(f"Step {step}:")
            print(f"  Avg difficulty: {progress.avg_difficulty:.3f}")
            print(f"  Avg pass rate: {progress.avg_pass_rate:.3f}")
            print(f"  Examples with data: {progress.examples_with_data}")


# ==============================================================================
# EXAMPLE 3: Different Strategies
# ==============================================================================

def example_different_strategies():
    """
    Example: Compare different curriculum strategies.
    """
    strategies = ["easy_first", "adaptive", "mixed"]

    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 40)

        sampler = CurriculumSampler(
            dataset_size=100,
            strategy=strategy,
            warmup_steps=50,
        )

        # Set varying difficulties for examples
        for i in range(100):
            pass_rate = 0.9 if i < 30 else 0.1  # Easy vs hard split
            sampler.set_pass_rate(i, pass_rate=pass_rate)

        # Sample at different stages
        for step in [100, 500, 1500]:
            samples = sampler.sample(batch_size=50, current_step=step)
            easy_count = sum(1 for s in samples if s < 30)
            print(f"  Step {step}: {easy_count}/50 easy examples sampled")


# ==============================================================================
# EXAMPLE 4: Integration with Config
# ==============================================================================

def example_config_integration():
    """
    Example: Configure curriculum learning via UnifiedConfig.
    """
    config = UnifiedConfig(
        algo="ppo",
        model=ModelConfig(name_or_path="meta-llama/Llama-2-7b"),
        datasets=[
            DatasetConfig(
                name="ultrafeedback",
                split="train",
                max_samples=1000,
            )
        ],
        train=TrainingConfig(
            max_steps=1000,
            per_device_batch_size=4,
            # Curriculum learning settings
            curriculum_rl_enabled=True,
            curriculum_rl_strategy="adaptive",
            curriculum_rl_warmup_steps=100,
            curriculum_rl_pass_threshold=0.5,
        ),
    )

    print("Config with Curriculum Learning:")
    print(f"  Enabled: {config.train.curriculum_rl_enabled}")
    print(f"  Strategy: {config.train.curriculum_rl_strategy}")
    print(f"  Warmup steps: {config.train.curriculum_rl_warmup_steps}")
    print(f"  Pass threshold: {config.train.curriculum_rl_pass_threshold}")


# ==============================================================================
# EXAMPLE 5: Analyzing Curriculum Progress
# ==============================================================================

def example_analyzing_progress():
    """
    Example: Analyze curriculum learning progress.
    """
    sampler = CurriculumSampler(dataset_size=100)

    # Simulate training
    import numpy as np
    for step in range(100):
        samples = sampler.sample(batch_size=10, current_step=step)
        rewards = np.random.randn(10)

        for example_id, reward in zip(samples, rewards):
            sampler.update_pass_rate(example_id, reward > 0)

    # Get statistics
    progress = sampler.get_curriculum_progress()
    print("Curriculum Progress Statistics:")
    print(f"  Average difficulty: {progress.avg_difficulty:.3f}")
    print(f"  Average pass rate: {progress.avg_pass_rate:.3f}")
    print(f"  Min difficulty: {progress.min_difficulty:.3f}")
    print(f"  Max difficulty: {progress.max_difficulty:.3f}")
    print(f"  Examples trained: {progress.examples_with_data}")
    print(f"  Total passes: {progress.num_passed}/{progress.num_sampled}")

    # Get difficulty distribution
    dist = sampler.get_difficulty_distribution()
    print(f"\nDifficulty Distribution:")
    print(f"  Mean: {dist['mean']:.3f}")
    print(f"  Std: {dist['std']:.3f}")
    print(f"  Histogram: {dist['histogram']}")


# ==============================================================================
# EXAMPLE 6: Manual Example Control
# ==============================================================================

def example_manual_control():
    """
    Example: Manually set and inspect example statistics.
    """
    sampler = CurriculumSampler(dataset_size=10)

    # Set initial difficulties
    sampler.set_pass_rate(0, pass_rate=0.9)  # Easy
    sampler.set_pass_rate(1, pass_rate=0.5)  # Medium
    sampler.set_pass_rate(2, pass_rate=0.1)  # Hard

    # Get example stats
    for i in range(3):
        stats = sampler.get_example_stats(i)
        print(f"Example {i}:")
        print(f"  Pass rate: {stats['pass_rate']:.3f}")
        print(f"  Difficulty: {stats['difficulty_score']:.3f}")

    # Update with real feedback
    sampler.update_pass_rate(0, passed=True)
    sampler.update_pass_rate(1, passed=False)
    sampler.update_pass_rate(2, passed=False)

    print("\nAfter updates:")
    stats = sampler.get_example_stats(0)
    print(f"Example 0 pass rate: {stats['pass_rate']:.3f}")


# ==============================================================================
# EXAMPLE 7: Custom Training Loop with Curriculum
# ==============================================================================

def example_custom_training_loop():
    """
    Example: Custom training loop with curriculum learning.
    """
    # Setup
    sampler = CurriculumSampler(dataset_size=100, warmup_steps=20)
    callback = CurriculumCallback(
        curriculum_sampler=sampler,
        pass_threshold=0.5,
        update_interval=10,
    )

    # Simulate training
    import numpy as np

    print("Training with curriculum learning:")
    print("-" * 40)

    for step in range(100):
        # Sample batch based on curriculum
        batch_indices = sampler.sample(batch_size=8, current_step=step)

        # Simulate forward pass and rewards
        rewards = np.random.randn(8).tolist()

        # Update curriculum (would be done by trainer in real scenario)
        class MockState:
            pass

        state = MockState()
        callback.on_step_end(
            args=None,
            state=state,
            control=None,
            rewards=rewards,
            example_ids=batch_indices,
        )

        # Log periodically
        if step % 25 == 0:
            stats = callback.get_curriculum_stats()
            print(f"Step {step}: avg_difficulty={stats['avg_difficulty']:.3f}, "
                  f"avg_pass_rate={stats['avg_pass_rate']:.3f}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Basic Curriculum Learning Setup")
    print("=" * 60)
    sampler, callback = example_basic_curriculum()

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Curriculum Sampling During Training")
    print("=" * 60)
    example_curriculum_sampling()

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Strategies")
    print("=" * 60)
    example_different_strategies()

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Integration with Config")
    print("=" * 60)
    example_config_integration()

    print("\n" + "=" * 60)
    print("EXAMPLE 5: Analyzing Curriculum Progress")
    print("=" * 60)
    example_analyzing_progress()

    print("\n" + "=" * 60)
    print("EXAMPLE 6: Manual Example Control")
    print("=" * 60)
    example_manual_control()

    print("\n" + "=" * 60)
    print("EXAMPLE 7: Custom Training Loop with Curriculum")
    print("=" * 60)
    example_custom_training_loop()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
