"""
Unsloth Rewards Backend.

This module provides the Unsloth-registered custom reward model training entrypoint
(UnslothRewardModelTrainer). See training.py for why the classification backbone is
loaded with standard transformers classes rather than unsloth.FastLanguageModel.
"""

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
__all__ = []  # Empty to prevent accidental imports
