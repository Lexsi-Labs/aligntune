"""
Unsloth SFT Backend.

This module provides Unsloth-optimized Supervised Fine-Tuning implementations.
"""

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
__all__ = []  # Empty to prevent accidental imports