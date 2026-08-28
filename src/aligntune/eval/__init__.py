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
Evaluation system for AlignTune.

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
from .alignment_auditor import AlignmentAuditor, AlignmentDriftTracker, AuditReport
from .probes import load_all_probe_sets, load_custom_probes
from .model_adapters import (
    ModelAdapter,
    HFModelAdapter,
    VLLMModelAdapter,
    GGUFModelAdapter,
    OllamaModelAdapter,
    build_adapter,
)
from .quant_regression import (
    ExportedArtifact,
    RegressionThresholds,
    ArtifactResult,
    RegressionReport,
    QuantRegressionRunner,
)

# Tokenization Evaluation
from .tokenization import (
    evaluate_unreachable_tokens,
    evaluate_fertility,
    evaluate_tokenizer,
)

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

    # Alignment Auditing
    "AlignmentAuditor",
    "AlignmentDriftTracker",
    "AuditReport",
    "load_all_probe_sets",
    "load_custom_probes",

    # Model Adapters
    "ModelAdapter",
    "HFModelAdapter",
    "VLLMModelAdapter",
    "GGUFModelAdapter",
    "OllamaModelAdapter",
    "build_adapter",

    # Quantization Regression
    "ExportedArtifact",
    "RegressionThresholds",
    "ArtifactResult",
    "RegressionReport",
    "QuantRegressionRunner",

    # Tokenization Evaluation
    "evaluate_unreachable_tokens",
    "evaluate_fertility",
    "evaluate_tokenizer",

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
