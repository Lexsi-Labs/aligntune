"""
S2-Attention (Shift Short Attention) implementation for AlignTune.

S2-Attention reduces memory usage through grouped attention with shifted windows.
Unlike sliding window attention which uses fixed local windows, S2-Attention:
1. Divides sequence into groups
2. Shifts half the attention heads by group_size/2
3. Performs attention within groups only

This achieves global information flow through the shifting mechanism while
maintaining memory efficiency through grouping.

Typical usage:
    from aligntune.core.long_context.custom_attention import register_s2_attention

    register_s2_attention()

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        attn_implementation="s2"
    )

    # Optional: Configure S2 parameters
    model.config.s2_group_size_ratio = 0.25
    model.config.s2_min_seq_length = 64
    model.config.s2_shift_ratio = 0.5
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask, causal_mask_function


def _mask_pad_value(mask: torch.Tensor) -> Union[float, int, bool]:
    # 2D padding masks are int64 0/1; additive 4D masks are float.
    if mask.dtype.is_floating_point:
        return torch.finfo(mask.dtype).min
    if mask.dtype == torch.bool:
        return False
    return 0


def s2_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    S2-Attention forward pass with grouped and shifted attention.

    Configuration options (set on model.config):
        - s2_group_size_ratio: Group size as fraction of seq_len (default: 0.25)
        - s2_min_seq_length: Minimum seq_len to activate S2 (default: 64)
        - s2_shift_ratio: Fraction of heads to shift (default: 0.5)

    Args:
        module: The attention module
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        value: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask
        dropout: Dropout probability
        scaling: Optional scaling factor
        is_causal: Whether to use causal masking
        **kwargs: Additional arguments

    Returns:
        Tuple of (attention_output, None)
    """
    bsz, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # Read configuration from model config
    config = getattr(module, 'config', None)
    group_size_ratio = getattr(config, 's2_group_size_ratio', 0.25)
    min_seq_length = getattr(config, 's2_min_seq_length', 64)
    shift_ratio = getattr(config, 's2_shift_ratio', 0.5)

    # Calculate group size
    group_size = max(1, int(q_len * group_size_ratio))

    # Determine if GQA is used
    is_gqa = num_q_heads != num_kv_heads

    # Fallback to standard SDPA
    if not module.training or q_len < min_seq_length or group_size < 2:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout, scaling, is_causal, **kwargs
        )

    # Pad sequence to be divisible by group_size
    remainder = q_len % group_size
    if remainder > 0:
        pad_len = group_size - remainder
        padded_len = q_len + pad_len

        query = F.pad(query, (0, 0, 0, pad_len))
        key = F.pad(key, (0, 0, 0, pad_len))
        value = F.pad(value, (0, 0, 0, pad_len))

        if attention_mask is not None:
            pad_val = _mask_pad_value(attention_mask)
            if attention_mask.ndim == 4:
                attention_mask = F.pad(
                    attention_mask, (0, pad_len, 0, pad_len), value=pad_val
                )
            else:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=pad_val)
    else:
        pad_len = 0
        padded_len = q_len

    num_groups = padded_len // group_size
    shift = group_size // 2

    # Calculate how many heads to shift based on shift_ratio
    num_shifted_q = max(1, int(num_q_heads * shift_ratio))
    num_shifted_kv = max(1, int(num_kv_heads * shift_ratio))

    # Split heads: unshifted and shifted
    query_unshifted = query[:, :num_q_heads - num_shifted_q]
    query_shifted = query[:, num_q_heads - num_shifted_q:].roll(-shift, dims=2)
    query = torch.cat([query_unshifted, query_shifted], dim=1)

    key_unshifted = key[:, :num_kv_heads - num_shifted_kv]
    key_shifted = key[:, num_kv_heads - num_shifted_kv:].roll(-shift, dims=2)
    key = torch.cat([key_unshifted, key_shifted], dim=1)

    value_unshifted = value[:, :num_kv_heads - num_shifted_kv]
    value_shifted = value[:, num_kv_heads - num_shifted_kv:].roll(-shift, dims=2)
    value = torch.cat([value_unshifted, value_shifted], dim=1)

    # Reshape into groups
    query = query.transpose(1, 2).contiguous()
    query = query.reshape(bsz * num_groups, group_size, num_q_heads, head_dim)
    query = query.transpose(1, 2).contiguous()

    key = key.transpose(1, 2).contiguous()
    key = key.reshape(bsz * num_groups, group_size, num_kv_heads, head_dim)
    key = key.transpose(1, 2).contiguous()

    value = value.transpose(1, 2).contiguous()
    value = value.reshape(bsz * num_groups, group_size, num_kv_heads, head_dim)
    value = value.transpose(1, 2).contiguous()

    # Create grouped attention mask
    grouped_mask = None
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            grouped_mask = attention_mask[:, :, :group_size, :group_size].repeat(
                num_groups, 1, 1, 1
            )
        else:
            grouped_mask = attention_mask.view(bsz, num_groups, group_size)
            grouped_mask = grouped_mask.reshape(bsz * num_groups, 1, 1, group_size)
        if grouped_mask.dtype in (torch.int32, torch.int64, torch.int8, torch.uint8):
            grouped_mask = grouped_mask.bool()

    # Perform SDPA within each group
    sdpa_kwargs = {}
    if is_gqa:
        sdpa_kwargs["enable_gqa"] = True

    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=grouped_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=(grouped_mask is None),
        **sdpa_kwargs,
    )

    # Ungroup
    out = out.transpose(1, 2).contiguous()
    out = out.reshape(bsz, padded_len, num_q_heads, head_dim)
    out = out.transpose(1, 2).contiguous()

    # Undo shift
    out_unshifted = out[:, :num_q_heads - num_shifted_q]
    out_shifted = out[:, num_q_heads - num_shifted_q:].roll(shift, dims=2)
    out = torch.cat([out_unshifted, out_shifted], dim=1)

    # Remove padding
    if pad_len > 0:
        out = out[:, :, :q_len, :]

    # Return in expected format
    out = out.transpose(1, 2).contiguous()

    return out, None


def s2_mask_function(
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
    Create attention mask for S2-Attention.

    This is called by transformers during forward pass to create attention masks.
    S2 uses standard causal masking. Grouping happens in forward pass.
    Signature matches sdpa_mask from transformers.masking_utils.

    NOTE: this used to take `cache_position` (a required positional arg,
    matching an older transformers mask-interface calling convention).
    transformers.masking_utils.create_causal_mask() now calls mask
    interfaces with q_length/q_offset instead and never passes
    cache_position at all, so every call raised TypeError: s2_mask_function()
    missing 1 required positional argument: 'cache_position' before this
    function's body ever ran. Matching sdpa_mask's current signature fixes it.

    Args:
        batch_size: Number of sequences in the batch
        q_length: Length of the query sequence
        kv_length: Length of key/value sequence
        q_offset: Offset for query positions (used with KV cache)
        kv_offset: Offset for key/value positions (used with KV cache)
        mask_function: Base mask function (ignored, we use causal masking)
        attention_mask: Optional padding mask from tokenizer
        local_size: Local attention window size (not used for S2)
        allow_is_causal_skip: Whether to allow skipping mask creation for causal case
        **kwargs: Additional arguments (config is passed here)

    Returns:
        4D attention mask tensor
    """
    return sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=causal_mask_function,
        attention_mask=attention_mask,
        local_size=local_size,
        allow_is_causal_skip=allow_is_causal_skip,
        **kwargs,
    )


def register_s2_attention():
    """
    Register S2-Attention implementation with transformers.

    After registration, use: attn_implementation="s2"

    Configuration options (set on model.config):
        model.config.s2_group_size_ratio = 0.25   # Group size (0.0-1.0)
        model.config.s2_min_seq_length = 64       # Min seq to activate
        model.config.s2_shift_ratio = 0.5         # Fraction of heads to shift

    Example:
        >>> register_s2_attention()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "meta-llama/Llama-2-7b",
        ...     attn_implementation="s2"
        ... )
        >>> model.config.s2_group_size_ratio = 0.125  # Smaller groups
        >>> model.config.s2_shift_ratio = 0.75        # Shift 75% of heads
    """
    AttentionInterface.register("s2", s2_attention_forward)
    AttentionMaskInterface.register("s2", s2_mask_function)


__all__ = [
    "s2_attention_forward",
    "s2_mask_function",
    "register_s2_attention",
]
