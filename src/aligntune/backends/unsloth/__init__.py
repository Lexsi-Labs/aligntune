"""
Unsloth Backend implementations for AlignTune.


This module provides Unsloth-optimized backend implementations for different
training types and algorithms, offering faster training improvements over standard training.

IMPORTANT: Unsloth must be imported before other ML libraries. This module uses
lazy loading to ensure proper import order.
"""

def _check_unsloth_available():
    """Check if Unsloth is available without importing trainers."""
    try:
        from .._imports import _check_unsloth_available as _lazy_check
        return _lazy_check()
    except ImportError:
        return False

UNSLOTH_AVAILABLE = _check_unsloth_available()
UNSLOTH_SFT_AVAILABLE = UNSLOTH_AVAILABLE
UNSLOTH_DPO_AVAILABLE = UNSLOTH_AVAILABLE
UNSLOTH_PPO_AVAILABLE = UNSLOTH_AVAILABLE
UNSLOTH_GRPO_AVAILABLE = UNSLOTH_AVAILABLE
# GSPO is only supported by TRL, not Unsloth
UNSLOTH_GSPO_AVAILABLE = False

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
UnslothSFTTrainer = None
UnslothDPOTrainer = None
UnslothPPOTrainer = None
UnslothGRPOTrainer = None
# GSPO is only supported by TRL, not Unsloth
UnslothGSPOTrainer = None

__all__ = [
    "UNSLOTH_AVAILABLE",
    "UNSLOTH_SFT_AVAILABLE",
    "UNSLOTH_DPO_AVAILABLE",
    "UNSLOTH_PPO_AVAILABLE",
    "UNSLOTH_GRPO_AVAILABLE",
    "UNSLOTH_GSPO_AVAILABLE",
]
