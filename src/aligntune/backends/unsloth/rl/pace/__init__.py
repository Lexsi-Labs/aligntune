"""
BOLT: Baseline-Optimized Learning Technique (Unsloth variant)

Unsloth-optimized GRPO backend with curriculum sampling and persistent baselines.
Provides faster training over TRL variant using Unsloth's FastLanguageModel.
"""

try:
    from .pace import UnslothBoltTrainer
    BOLT_AVAILABLE = True
except ImportError:
    BOLT_AVAILABLE = False
    UnslothBoltTrainer = None

__all__ = [
    "UnslothBoltTrainer",
    "BOLT_AVAILABLE",
]
