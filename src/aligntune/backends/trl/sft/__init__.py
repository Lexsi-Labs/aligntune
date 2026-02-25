"""
TRL SFT Backend.

This module provides TRL-based Supervised Fine-Tuning implementations.
"""

try:
    from .sft import TRLSFTTrainer
    __all__ = ["TRLSFTTrainer"]
except ImportError:
    __all__ = []
