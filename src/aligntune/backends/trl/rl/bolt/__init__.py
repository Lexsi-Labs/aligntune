"""
BOLT: Baseline-Optimized Learning Technique

A GRPO backend with curriculum sampling and persistent baselines.

Features:
- Uncertainty-based curriculum sampling: w(x) = sqrt(v̂(1-v̂)) + ε
- Persistent per-prompt baselines with KL-adaptive forgetting
- SPO-style advantage computation: A = r - v̂(x)
"""

try:
    from .bolt import TRLBoltTrainer, BoltConfig
    from .baseline import BaselineStore, UnifiedBaseline, make_prompt_key
    from .curriculum import (
        DynamicWeightedSampler,
        DynamicallySampledDataset,
        CurriculumCallback,
        BaselineUpdateCallback,
        BaselineRewardWrapper,
    )

    BOLT_AVAILABLE = True
except ImportError as e:
    BOLT_AVAILABLE = False
    TRLBoltTrainer = None
    BoltConfig = None
    BaselineStore = None
    UnifiedBaseline = None
    make_prompt_key = None
    DynamicWeightedSampler = None
    DynamicallySampledDataset = None
    CurriculumCallback = None
    BaselineUpdateCallback = None
    BaselineRewardWrapper = None

__all__ = [
    "TRLBoltTrainer",
    "BoltConfig",
    "BaselineStore",
    "UnifiedBaseline",
    "make_prompt_key",
    "DynamicWeightedSampler",
    "DynamicallySampledDataset",
    "CurriculumCallback",
    "BaselineUpdateCallback",
    "BaselineRewardWrapper",
    "BOLT_AVAILABLE",
]
