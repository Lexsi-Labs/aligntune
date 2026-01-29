"""
Unsloth PPO Backend.

This module provides Unsloth-optimized Proximal Policy Optimization implementations.
"""

# CRITICAL: Apply TRL patch IMMEDIATELY before any other imports
import sys
import os

# Force clear Unsloth cache to ensure fresh compilation
cache_dir = os.path.join(os.getcwd(), "unsloth_compiled_cache")
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("🗑️ Cleared Unsloth compiled cache")

# Set environment variable to force recompilation
os.environ['UNSLOTH_FORCE_RECOMPILE'] = '1'

# Apply TRL patch BEFORE any Unsloth imports
try:
    import trl.trainer.utils
    from trl.trainer.utils import first_true_indices
    import torch

    def patched_get_reward(
            model,
            query_responses,
            pad_token_id,
            context_length):
        """Patched get_reward to handle encoder models that don't support use_cache."""
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        lm_backbone = getattr(model, model.base_model_prefix)
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        # Detect encoder models
        model_type = getattr(model.config, 'model_type', '').lower()
        encoder_models = [
            'bert',
            'deberta',
            'deberta-v2',
            'roberta',
            'electra',
            'albert']

        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'return_dict': True,
            'output_hidden_states': True,
        }

        # Only add use_cache for decoder models
        if model_type not in encoder_models:
            kwargs['use_cache'] = False

        output = lm_backbone(**kwargs)
        reward_logits = model.score(output.hidden_states[-1])
        sequence_lengths = first_true_indices(
            query_responses[:, context_length:] == pad_token_id) - 1 + context_length

        return (reward_logits, reward_logits, sequence_lengths)

    # Apply the patch IMMEDIATELY
    trl.trainer.utils.get_reward = patched_get_reward
    print("✅ Applied TRL get_reward patch BEFORE Unsloth compilation")

except Exception as e:
    print(f"❌ Could not apply TRL patch: {e}")

# Trainers are loaded lazily via backend_factory
# Do NOT import them here to avoid triggering Unsloth's global TRL patching
__all__ = []  # Empty to prevent accidental imports
