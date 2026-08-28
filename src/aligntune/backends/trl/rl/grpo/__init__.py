"""
TRL GRPO Backend.

This module provides TRL-based Group Relative Policy Optimization implementations.
"""

try:
    from .grpo import TRLGRPOTrainer
    __all__ = ["TRLGRPOTrainer"]
except ImportError:
    __all__ = []
