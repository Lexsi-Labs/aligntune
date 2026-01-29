"""
TRL GRPO Backend.

This module provides TRL-based Group Relative Policy Optimization implementations.
"""

try:
    from .grpo_counterfactual import TRLCounterFactGRPOTrainer
    __all__ = ["TRLCounterFactGRPOTrainer"]
except ImportError:
    __all__ = []
