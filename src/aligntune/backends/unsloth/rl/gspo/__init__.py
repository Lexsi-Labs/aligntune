"""
Unsloth GSPO Backend.

This module provides Unsloth-optimized Generalized Scoring Proximal Objective implementations.
"""

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
__all__ = []  # Empty to prevent accidental imports
