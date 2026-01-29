"""
Backward compatibility module for /sft/core/ imports.

This module provides backward-compatible imports for code that previously
imported from aligntune.sft.core. All imports are re-exported from the
new /core/sft/ location.
"""

# Re-export all SFT core components from the new location
from ...core.sft import *
