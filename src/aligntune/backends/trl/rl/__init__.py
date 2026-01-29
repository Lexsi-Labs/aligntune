"""
TRL RL Backend.

This module provides TRL-based Reinforcement Learning implementations.
"""

try:
    from .dpo.dpo import TRLDPOTrainer
    DPO_AVAILABLE = True
except ImportError:
    DPO_AVAILABLE = False
    TRLDPOTrainer = None

try:
    from .ppo.ppo import TRLPPOTrainer
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    TRLPPOTrainer = None

try:
    from .grpo.grpo import TRLGRPOTrainer
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    TRLGRPOTrainer = None

try:
    from .gspo.gspo import TRLGSPOTrainer
    GSPO_AVAILABLE = True
except ImportError:
    GSPO_AVAILABLE = False
    TRLGSPOTrainer = None

try:
    from .bolt.bolt import TRLBoltTrainer
    BOLT_AVAILABLE = True
except ImportError:
    BOLT_AVAILABLE = False
    TRLBoltTrainer = None

__all__ = [
    "TRLDPOTrainer",
    "TRLPPOTrainer",
    "TRLGRPOTrainer",
    "TRLGSPOTrainer",
    "TRLBoltTrainer",
    "DPO_AVAILABLE",
    "PPO_AVAILABLE",
    "GRPO_AVAILABLE",
    "GSPO_AVAILABLE",
    "BOLT_AVAILABLE",
]
