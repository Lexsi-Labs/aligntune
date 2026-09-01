"""
SPIN: Self-Play Fine-Tuning for Improved Alignment

This package implements SPIN (Self-Play Improvement through No-regret learning),
which trains models through self-play by:
1. Generating responses from current model and frozen opponent checkpoint
2. Creating synthetic preference pairs (SFT response = chosen, opponent = rejected)
3. Training DPO on synthetic pairs
4. Updating opponent checkpoint and repeating

Paper: https://arxiv.org/abs/2404.04291
"""

from .spin import TRLSPINTrainer

__all__ = ["TRLSPINTrainer"]
