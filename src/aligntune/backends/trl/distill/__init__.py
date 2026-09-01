"""
TRL Knowledge Distillation Trainers.

Supports:
- Standard Distillation (MiniLLM)
- SDFT (Self-Distillation Fine-Tuning)
"""

from .distillation.distillation import TRLDistillationTrainer
from .sdft.sdft import TRLSDFTTrainer

__all__ = [
    "TRLDistillationTrainer",
    "TRLSDFTTrainer",
]
