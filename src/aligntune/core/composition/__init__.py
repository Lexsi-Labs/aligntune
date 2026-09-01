"""
Multi-stage training composition framework.

This module provides infrastructure for composing complex training pipelines
that chain multiple algorithms together (e.g., SFT → MoA → ES → DPO → audit).
"""

from .stages import (
    Stage,
    Composition,
    StageResult,
    CompositionLoader,
)
from .runner import (
    CompositionRunner,
    CompositionExecutor,
)

__all__ = [
    "Stage",
    "Composition",
    "StageResult",
    "CompositionLoader",
    "CompositionRunner",
    "CompositionExecutor",
]
