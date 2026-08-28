"""
Quick integration test for Online DPO trainer setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aligntune.backends.trl.rl.online_dpo.online_dpo import TRLOnlineDPOTrainer
from aligntune.core.backend_factory import RLAlgorithm
from aligntune.core.rl.config import BackendType
from unittest.mock import MagicMock

def test_online_dpo_trainer_import():
    """Test that Online DPO trainer can be imported."""
    print("Test 1: Import Online DPO trainer")
    assert TRLOnlineDPOTrainer is not None
    print("  PASSED: TRLOnlineDPOTrainer imported successfully")

def test_online_dpo_enum():
    """Test that ONLINE_DPO is in RLAlgorithm enum."""
    print("Test 2: Check RLAlgorithm.ONLINE_DPO enum")
    assert hasattr(RLAlgorithm, 'ONLINE_DPO')
    assert RLAlgorithm.ONLINE_DPO.value == "online_dpo"
    print("  PASSED: RLAlgorithm.ONLINE_DPO enum exists")

def test_is_available():
    """Test trainer availability check."""
    print("Test 3: Check trainer availability")
    is_available = TRLOnlineDPOTrainer.is_available()
    print(f"  TRLOnlineDPOTrainer.is_available() = {is_available}")
    print("  PASSED: Availability check completed")

if __name__ == "__main__":
    print("Running Online DPO Setup Tests\n" + "="*50)
    try:
        test_online_dpo_trainer_import()
        test_online_dpo_enum()
        test_is_available()
        print("\n" + "="*50)
        print("All tests PASSED!")
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
