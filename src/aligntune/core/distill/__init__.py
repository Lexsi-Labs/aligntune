"""Knowledge Distillation module for AlignTune."""

from .config import UnifiedDistillConfig, DistillationType

# Alias for backwards compatibility
DistillConfig = UnifiedDistillConfig

__all__ = ["DistillConfig", "UnifiedDistillConfig", "DistillationType"]
