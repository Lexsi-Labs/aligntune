"""
TRL GSPO Backend.

This module provides TRL-based Generalized Scoring Proximal Objective implementations.
"""

try:
    from .gspo import TRLGSPOTrainer
    __all__ = ["TRLGSPOTrainer"]
except ImportError:
    __all__ = []
