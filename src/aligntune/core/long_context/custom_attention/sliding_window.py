"""
Sliding Window Attention (SWA) implementation for AlignTune.

This module provides a custom attention implementation that applies sliding window
masking to reduce memory usage for long sequences. Each token only attends to a
fixed-size window of surrounding tokens rather than the full sequence.

Typical usage:
    from aligntune.core.long_context.attention import register_sliding_window_attention

    # Register before model loading
    register_sliding_window_attention()

    # Load model with SWA
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        attn_implementation="swa"
    )
    model.config.sliding_window = 4096
"""

from typing import Optional
import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask, sliding_window_causal_mask_function


def swa_mask_function(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function=None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
):
    """
    Create a sliding window causal mask for attention computation.

    This is called by transformers during forward pass to create attention masks.
    Signature matches sdpa_mask from transformers.masking_utils.

    NOTE: this used to take `cache_position` (a required positional arg,
    matching an older transformers mask-interface calling convention).
    transformers.masking_utils.create_causal_mask() now calls mask
    interfaces with q_length/q_offset instead and never passes
    cache_position at all, so every call raised TypeError: swa_mask_function()
    missing 1 required positional argument: 'cache_position' before this
    function's body ever ran. Matching sdpa_mask's current signature fixes it.

    Args:
        batch_size: Number of sequences in the batch
        q_length: Length of the query sequence
        kv_length: Length of key/value sequence
        q_offset: Offset for query positions (used with KV cache)
        kv_offset: Offset for key/value positions (used with KV cache)
        mask_function: Base mask function (ignored, we override with sliding window)
        attention_mask: Optional padding mask from tokenizer
        local_size: Local attention window size (passed by transformers)
        allow_is_causal_skip: Whether to allow skipping mask creation for causal case
        **kwargs: Additional arguments (config is passed here)

    Returns:
        4D attention mask tensor of shape [batch_size, 1, q_length, kv_length]
    """
    config = kwargs.get('config')
    window = getattr(config, "sliding_window", None) if config else None

    if window is None:
        raise ValueError(
            "Model config must have 'sliding_window' attribute set. "
            "Example: model.config.sliding_window = 4096"
        )

    # Create sliding window mask using HuggingFace's built-in function
    mask = sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=sliding_window_causal_mask_function(window),
        attention_mask=attention_mask,
        local_size=window,  # Critical: prevents optimization from skipping mask
        allow_is_causal_skip=allow_is_causal_skip,
        **kwargs,
    )
    return mask


def register_sliding_window_attention():
    """
    Register sliding window attention implementation with transformers.

    This function registers two components:
    1. AttentionInterface: Uses SDPA for actual attention computation
    2. AttentionMaskInterface: Creates sliding window masks

    After registration, models can be loaded with attn_implementation="swa".

    Example:
        >>> from aligntune.core.long_context.attention import register_sliding_window_attention
        >>> register_sliding_window_attention()
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "meta-llama/Llama-2-7b",
        ...     attn_implementation="swa"
        ... )
        >>> model.config.sliding_window = 4096

    Note:
        This should be called before model loading, typically in model_loader.py
    """
    # Register attention computation (reuse SDPA)
    AttentionInterface.register("swa", sdpa_attention_forward)

    # Register mask creation (custom sliding window)
    AttentionMaskInterface.register("swa", swa_mask_function)


__all__ = [
    "swa_mask_function",
    "register_sliding_window_attention",
]
