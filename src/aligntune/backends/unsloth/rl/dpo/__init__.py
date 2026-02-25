"""
Unsloth DPO Backend.

This module provides Unsloth-optimized Direct Preference Optimization implementations.
"""

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
__all__ = []  # Empty to prevent accidental imports
