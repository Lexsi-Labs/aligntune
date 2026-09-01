"""
RAFT (Retrieval Augmented Fine-Tuning) Backend for AlignTune.

Implements retrieval-augmented training where the model learns to:
1. Condition on golden (relevant) documents in the input context
2. Distinguish between golden and distractor (irrelevant) documents
3. Minimize citation hallucination by preferring cited sources over invented facts
"""

from .raft_trainer import RaftTrainer, RaftTrainerConfig, format_raft_example, raft_trainer_from_config

__all__ = ["RaftTrainer", "RaftTrainerConfig", "format_raft_example", "raft_trainer_from_config"]
