"""
Backward compatibility module for /rl/ imports.

This module provides backward-compatible imports for code that previously
imported from aligntune.rl.core. All imports are re-exported from the
new /core/rl/ location.
"""

# Re-export all RL core components from the new location
from ..core.rl import *
