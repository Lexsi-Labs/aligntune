"""
TRL PPO Backend.

This module provides TRL-based Proximal Policy Optimization implementations.
"""

try:
    from .ppo import TRLPPOTrainer
    __all__ = ["TRLPPOTrainer"]
except ImportError:
    __all__ = []
