"""
TRL GRPO Backend.

This module provides TRL-based Group Relative Policy Optimization implementations.
"""

try:
    from .drgrpo import TRLDRGRPOTrainer
    __all__ = ["TRLDRGRPOTrainer"]
except ImportError:
    __all__ = []
