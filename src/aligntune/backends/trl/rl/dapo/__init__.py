"""
TRL GRPO Backend.

This module provides TRL-based Group Relative Policy Optimization implementations.
"""

try:
    from .dapo import TRLDAPOTrainer
    __all__ = ["TRLDAPOTrainer"]
except ImportError:
    __all__ = []
