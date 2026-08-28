"""
Online DPO Trainer Module

Implements iterative online DPO training where the model continuously improves
by sampling from the current policy, ranking with a reward model, and training
on preference pairs.
"""

from .online_dpo import TRLOnlineDPOTrainer

__all__ = ["TRLOnlineDPOTrainer"]
