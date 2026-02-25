"""
TRL Backend implementations for AlignTune.


This module provides pure TRL backend implementations for reliable
and battle-tested training across all supported algorithms.
"""

try:
    from .sft.sft import TRLSFTTrainer
    SFT_AVAILABLE = True
except ImportError:
    SFT_AVAILABLE = False
    TRLSFTTrainer = None

try:
    from .rl.dpo.dpo import TRLDPOTrainer
    DPO_AVAILABLE = True
except ImportError:
    DPO_AVAILABLE = False
    TRLDPOTrainer = None

try:
    from .rl.ppo.ppo import TRLPPOTrainer
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    TRLPPOTrainer = None

try:
    from .rl.grpo.grpo import TRLGRPOTrainer
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    TRLGRPOTrainer = None

try:
    from .rl.gspo.gspo import TRLGSPOTrainer
    GSPO_AVAILABLE = True
except ImportError:
    GSPO_AVAILABLE = False
    TRLGSPOTrainer = None

__all__ = [
    "TRLSFTTrainer",
    "TRLDPOTrainer",
    "TRLPPOTrainer",
    "TRLGRPOTrainer",
    "TRLGSPOTrainer",
    "SFT_AVAILABLE",
    "DPO_AVAILABLE",
    "PPO_AVAILABLE",
    "GRPO_AVAILABLE",
    "GSPO_AVAILABLE",
]
