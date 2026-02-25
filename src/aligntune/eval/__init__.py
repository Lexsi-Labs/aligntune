# """
# Evaluation module for AlignTune.

# Provides unified evaluation for SFT and RL models.
# """

# from .core import BaseEvaluator
# from .rl_evaluator import RLEvaluator
# from .metrics.base import Metric
# from .metrics.generic import PerplexityMetric, AccuracyMetric
# from .metrics.text import BleuMetric, RougeMetric
# from .metrics.rl import KLDivergenceMetric, RewardAccuracyMetric, PolicyEntropyMetric

# __all__ = [
#     "BaseEvaluator",
#     "RLEvaluator",
#     "Metric",
#     "PerplexityMetric",
#     "AccuracyMetric",
#     "BleuMetric",
#     "RougeMetric",
#     "KLDivergenceMetric",
#     "RewardAccuracyMetric",
#     "PolicyEntropyMetric"
# ]

# MERGED INIT 

"""
<<<<<<< HEAD
Evaluation system for aligntune.
=======
Evaluation system for AlignTune.
>>>>>>> origin/check_chirag

This module provides a unified interface for evaluating models:
1. Universal Evaluator (New): Modular, backend-agnostic evaluation for SFT/RL.
2. Legacy Evaluator (Old): Compatible with existing CLI and EvalRunner workflows.
"""

# --- New Universal Framework Exports ---
from .evaluator import BaseEvaluator
from .rl_evaluator import RLEvaluator
from .metrics.base import Metric
from .metrics.generic import PerplexityMetric, AccuracyMetric
from .metrics.text import BleuMetric, RougeMetric
from .metrics.rl import KLDivergenceMetric, RewardAccuracyMetric, PolicyEntropyMetric
from .metrics.math import MathAccuracyMetric
from .metrics.code import PassAtKMetric

# --- Legacy Framework Exports (Backward Compatibility) ---
# Assumes the old 'core.py' is preserved in the directory
from .core import (
    EvalType,
    TaskCategory,
    EvalConfig,
    EvalTask,
    EvalResult,
    EvalLogger,
    EvalRegistry,
    EvalRunner,
)

from .lm_eval_integration import (
    LMEvalConfig,
    LMEvalTask,
    LMEvalRunner,
    LMEVAL_TASKS,
    get_available_lm_eval_tasks,
    get_lm_eval_task,
    run_standard_benchmark,
)

# Registry (Merged logic)
from . import registry

__all__ = [
    # New Universal Classes
    "BaseEvaluator",
    "RLEvaluator",
    "Metric",
    "PerplexityMetric",
    "AccuracyMetric",
    "BleuMetric",
    "RougeMetric",
    "KLDivergenceMetric",
    "RewardAccuracyMetric",
    "PolicyEntropyMetric",
    "MathAccuracyMetric",
    "PassAtKMetric",

    # Legacy Core Classes
    "EvalType",
    "TaskCategory", 
    "EvalConfig",
    "EvalTask",
    "EvalResult",
    "EvalLogger",
    "EvalRegistry",
    "EvalRunner",
    
    # lm-eval Integration
    "LMEvalConfig",
    "LMEvalTask", 
    "LMEvalRunner",
    "LMEVAL_TASKS",
    "get_available_lm_eval_tasks",
    "get_lm_eval_task",
    "run_standard_benchmark",
    
    # Registry
    "registry",
]