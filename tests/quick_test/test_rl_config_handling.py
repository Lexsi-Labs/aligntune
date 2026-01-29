#!/usr/bin/env python3
"""
Simple RL Configuration Test
This script tests the basic RL configuration functionality.
"""

import sys
from pathlib import Path

# Add the parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from aligntune.core.backend_factory import create_rl_trainer
    print("✅ Successfully imported create_rl_trainer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the right directory or adjust the import path.")
    sys.exit(1)


def test_basic_config():
    """Test the most basic configuration."""
    print("\n" + "="*60)
    print("BASIC RL CONFIGURATION TEST")
    print("="*60)
    
    try:
        # Basic configuration using direct parameters
        trainer = create_rl_trainer(
            model_name="Qwen/Qwen3-0.6B",  # Small model for testing
            dataset_name="An-anthonny/ultrachat_1k_sample",  # Small dataset
            algorithm="ppo",  # Test with PPO
            backend="trl",  # Use TRL backend
            output_dir="./test_basic_rl",
            max_steps=10,  # Very small for quick testing
            batch_size=2,
            learning_rate=2e-5,
            split="train[:5]",  # Only 5 samples
            column_mapping={"messages": "text"},
            field_mappings={
                "prompt": "messages",
                "response": "completion"
            },
            rewards=[{
                "type": "length_reward",
                "weight": 0.1,
                "params": {}
            }],
            logging_steps=5,
        )
        
        print("✅ Configuration created successfully!")
        print(f"\nConfiguration details:")
        print(f"  Algorithm: {trainer.config.algo}")
        print(f"  Model: {trainer.config.model.name_or_path}")
        print(f"  Dataset: {trainer.config.datasets[0].name}")
        print(f"  Max steps: {trainer.config.train.max_steps}")
        print(f"  Batch size: {trainer.config.train.per_device_batch_size}")
        print(f"  Output dir: {trainer.config.logging.output_dir}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_config_file():
    """Test configuration from a YAML file."""
    print("\n" + "="*60)
    print("CONFIG FILE TEST")
    print("="*60)
    
    # Create a simple YAML config file
    yaml_content = """
model_name: "Qwen/Qwen3-0.6B"
dataset_name: "An-anthonny/ultrachat_1k_sample"
algorithm: "dpo"
backend: "trl"
output_dir: "./test_yaml_config"
max_steps: 10
batch_size: 2
learning_rate: 2e-5
split: "train[:3]"
column_mapping:
  messages: "text"
field_mappings:
  prompt: "messages"
  response: "completion"
rewards:
  - type: "length_reward"
    weight: 0.1
    params: {}
logging_steps: 5
"""
    
    # Write to temporary file
    config_path = Path("./test_config.yaml")
    config_path.write_text(yaml_content)
    
    try:
        trainer = create_rl_trainer(config=str(config_path))
        print("✅ YAML config loaded successfully!")
        return trainer
    except Exception as e:
        print(f"❌ YAML config failed: {e}")
        return None
    finally:
        # Clean up
        if config_path.exists():
            config_path.unlink()


def quick_train_test(trainer, steps=5):
    """Run a very quick training test."""
    print("\n" + "="*60)
    print("QUICK TRAINING TEST")
    print("="*60)
    
    if trainer is None:
        print("❌ No trainer available")
        return False
    
    print(f"Running {steps} training steps...")
    print("This might take a minute to download models and dataset...")
    
    try:
        # Save original max_steps
        original_steps = trainer.config.train.max_steps
        
        # Set to smaller value for quick test
        trainer.config.train.max_steps = steps
        
        # Start training
        print("\nStarting training...")
        trainer.train()
        
        # Restore original
        trainer.config.train.max_steps = original_steps
        
        print(f"\n✅ Successfully completed {steps} training steps!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("\n" + "="*60)
    print("RL CONFIGURATION TEST")
    print("="*60)
    
    # Test 1: Basic configuration
    trainer = test_basic_config()
    
    if trainer:
        # Ask if user wants to run a quick training test
        response = input("\nRun a quick training test (5 steps)? (y/n): ").strip().lower()
        if response == 'y':
            success = quick_train_test(trainer, steps=5)
            if success:
                print("\n🎉 Basic test passed! RL configuration is working.")
            else:
                print("\n⚠️  Configuration loaded but training failed.")
        else:
            print("\n✅ Configuration test passed (training skipped).")
    
    # Test 2: Config file
    print("\n" + "="*60)
    response = input("Test YAML config file loading? (y/n): ").strip().lower()
    if response == 'y':
        trainer2 = test_config_file()
        if trainer2:
            print("✅ YAML config test passed!")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import transformers
        print("Dependencies check:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Transformers: {transformers.__version__}")
        
        # Try to import TRL
        try:
            import trl
            print(f"  TRL: {trl.__version__}")
        except ImportError:
            print("  TRL: Not installed (required for TRL backend)")
            print("  Install with: pip install trl")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch transformers")
        sys.exit(1)
    
    main()