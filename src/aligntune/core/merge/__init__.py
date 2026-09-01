"""
Model merging subpackage.

Provides thin wrappers around third-party merging libraries:

* MergekitMerger — linear, task_arithmetic, ram via mergekit
* PEFTMerger     — Simple LoRA adapter merge via peft.PeftModel.merge_and_unload()
"""

from .base import BaseMerger
from .mergekit_merger import MergekitMerger
from .peft_merger import PEFTMerger

__all__ = [
    "BaseMerger",
    "MergekitMerger",
    "PEFTMerger",
]
