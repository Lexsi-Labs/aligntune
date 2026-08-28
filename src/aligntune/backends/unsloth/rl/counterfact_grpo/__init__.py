"""
Unsloth Counterfactual GRPO Backend.

This module provides an Unsloth-optimized backend for Counterfactual GRPO training,
combining Unsloth's fast model loading with the custom counterfactual importance
weighting from the TRL implementation.
"""

from .counterfact_grpo import UnslothCounterFactGRPOTrainer

__all__ = ["UnslothCounterFactGRPOTrainer"]
