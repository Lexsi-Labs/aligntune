"""
SPIN Import and Registration Tests

Quick validation tests for SPIN:
- Import dependencies
- Backend registration
- Trainer creation
"""

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_spin_dependencies():
    """Test that all SPIN dependencies can be imported."""
    print("Testing SPIN dependencies...")

    # TRL dependencies
    try:
        from trl import DPOTrainer, DPOConfig
        print("✓ TRL DPO imports OK")
    except ImportError as e:
        print(f"✗ TRL import failed: {e}")
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Transformers imports OK")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        sys.exit(1)

    try:
        from aligntune.core.model_loader import build_model
        print("✓ Model loader imports OK")
    except ImportError as e:
        print(f"✗ Model loader import failed: {e}")
        sys.exit(1)

    try:
        from aligntune.data.manager import DataManager
        print("✓ DataManager imports OK")
    except ImportError as e:
        print(f"✗ DataManager import failed: {e}")
        sys.exit(1)

    print("All dependencies imported successfully!\n")


def test_spin_trainer_import():
    """Test SPIN trainer can be imported."""
    print("Testing SPIN trainer import...")

    try:
        from aligntune.backends.trl.rl.spin.spin import TRLSPINTrainer
        print("✓ TRLSPINTrainer imported OK")

        # Check availability
        if TRLSPINTrainer.is_available():
            print("✓ TRLSPINTrainer.is_available() = True")
        else:
            print("✗ TRLSPINTrainer.is_available() = False (TRL not installed?)")

    except ImportError as e:
        print(f"✗ TRLSPINTrainer import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()


def test_spin_registration():
    """Test SPIN is registered in BackendFactory."""
    print("Testing SPIN backend registration...")

    try:
        from aligntune.core.backend_factory import BackendFactory, TrainingType, BackendType, RLAlgorithm

        # Check SPIN enum exists
        if hasattr(RLAlgorithm, 'SPIN'):
            print("✓ RLAlgorithm.SPIN exists")
        else:
            print("✗ RLAlgorithm.SPIN not found in enum")
            sys.exit(1)

        # Check registration
        spin_key = (TrainingType.RL, BackendType.TRL, RLAlgorithm.SPIN)
        if spin_key in BackendFactory._backends:
            print(f"✓ SPIN registered in BackendFactory: {spin_key}")
        else:
            print(f"✗ SPIN not registered in BackendFactory")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Registration check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()


def test_spin_config_enum():
    """Test SPIN is in AlgorithmType enum."""
    print("Testing SPIN in AlgorithmType enum...")

    try:
        from aligntune.core.rl.config import AlgorithmType

        if hasattr(AlgorithmType, 'SPIN'):
            print(f"✓ AlgorithmType.SPIN = {AlgorithmType.SPIN.value}")
        else:
            print("✗ AlgorithmType.SPIN not found")
            sys.exit(1)

    except Exception as e:
        print(f"✗ AlgorithmType check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()


def test_spin_trainer_creation():
    """Test SPIN trainer can be created with minimal config."""
    print("Testing SPIN trainer creation...")

    try:
        from aligntune.core.backend_factory import create_rl_trainer

        # Minimal config for creation test
        trainer = create_rl_trainer(
            model_name="gpt2",
            dataset_name="imdb",
            split="train[:4]",
            algorithm="spin",
            backend="trl",
            output_dir="/tmp/spin_test",
            num_rounds=1,
            dpo_steps_per_round=10,
            batch_size=2,
            learning_rate=1e-5,
        )

        print(f"✓ Trainer created: {type(trainer).__name__}")
        print(f"✓ Trainer has config: {hasattr(trainer, 'config')}")
        print(f"✓ Trainer has train method: {hasattr(trainer, 'train')}")
        print(f"✓ Trainer has setup_model method: {hasattr(trainer, 'setup_model')}")
        print(f"✓ Trainer has setup_data method: {hasattr(trainer, 'setup_data')}")
        print(f"✓ Trainer has setup_trainer method: {hasattr(trainer, 'setup_trainer')}")

    except Exception as e:
        print(f"✗ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("SPIN Import & Registration Tests")
    print("=" * 60)
    print()

    test_spin_dependencies()
    test_spin_trainer_import()
    test_spin_registration()
    test_spin_config_enum()
    test_spin_trainer_creation()

    print("=" * 60)
    print("✓ All SPIN tests passed!")
    print("=" * 60)
