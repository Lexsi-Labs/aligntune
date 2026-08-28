"""
Backend implementations for AlignTune.


This module provides organized backend implementations for different
training types and algorithms, supporting multiple backends for optimal
performance and flexibility.
"""

# Import TRL backends
try:
    from .trl.sft.sft import TRLSFTTrainer
    from .trl.rl.dpo.dpo import TRLDPOTrainer
    from .trl.rl.ppo.ppo import TRLPPOTrainer
    from .trl.rl.grpo.grpo import TRLGRPOTrainer
    from .trl.rl.gspo.gspo import TRLGSPOTrainer
    TRL_BACKENDS_AVAILABLE = True
except ImportError:
    TRL_BACKENDS_AVAILABLE = False
    TRLSFTTrainer = None
    TRLDPOTrainer = None
    TRLPPOTrainer = None
    TRLGRPOTrainer = None
    TRLGSPOTrainer = None

# Unsloth backends are loaded lazily via backend_factory
# Do NOT import them here to avoid global TRL patching
UNSLOTH_BACKENDS_AVAILABLE = None  # Will be checked lazily

def _check_unsloth_backends_available():
    """Lazy check for Unsloth backend availability."""
    global UNSLOTH_BACKENDS_AVAILABLE
    if UNSLOTH_BACKENDS_AVAILABLE is None:
        try:
            from .._imports import _check_unsloth_available
            UNSLOTH_BACKENDS_AVAILABLE = _check_unsloth_available()
        except ImportError:
            UNSLOTH_BACKENDS_AVAILABLE = False
    return UNSLOTH_BACKENDS_AVAILABLE

# Set to None to prevent accidental imports
UnslothSFTTrainer = None
UnslothDPOTrainer = None
UnslothPPOTrainer = None
UnslothGRPOTrainer = None
UnslothGSPOTrainer = None

__all__ = [
    # TRL Backends
    "TRLSFTTrainer",
    "TRLDPOTrainer",
    "TRLPPOTrainer",
    "TRLGRPOTrainer",
    "TRLGSPOTrainer",
    "TRL_BACKENDS_AVAILABLE",
    
    # Unsloth Backends
    "UnslothSFTTrainer",
    "UnslothDPOTrainer",
    "UnslothPPOTrainer",
    "UnslothGRPOTrainer",
    "UnslothGSPOTrainer",
    "UNSLOTH_BACKENDS_AVAILABLE",
]
