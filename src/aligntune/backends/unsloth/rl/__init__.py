"""
Unsloth RL Backend.

This module provides Unsloth-optimized Reinforcement Learning implementations.
"""

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
DPO_AVAILABLE = True
PPO_AVAILABLE = True
GRPO_AVAILABLE = True
GSPO_AVAILABLE = True
BOLT_AVAILABLE = True
COUNTERFACT_GRPO_AVAILABLE = True

UnslothDPOTrainer = None
UnslothPPOTrainer = None
UnslothGRPOTrainer = None
UnslothGSPOTrainer = None
UnslothBoltTrainer = None
UnslothCounterFactGRPOTrainer = None

__all__ = [
    "UnslothDPOTrainer",
    "UnslothPPOTrainer",
    "UnslothGRPOTrainer",
    "UnslothGSPOTrainer",
    "UnslothBoltTrainer",
    "UnslothCounterFactGRPOTrainer",
    "DPO_AVAILABLE",
    "PPO_AVAILABLE",
    "GRPO_AVAILABLE",
    "GSPO_AVAILABLE",
    "BOLT_AVAILABLE",
    "COUNTERFACT_GRPO_AVAILABLE",
]
