"""
TRL DPO Backend.

This module provides TRL-based Direct Preference Optimization implementations.
"""

try:
    from .dpo import TRLDPOTrainer
    __all__ = ["TRLDPOTrainer"]
except ImportError:
    __all__ = []
