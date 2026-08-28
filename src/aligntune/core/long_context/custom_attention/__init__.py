"""
Custom attention implementations for long-context training in AlignTune.

This package provides efficient attention mechanisms for handling long sequences:

- Sliding Window Attention (SWA): Each token attends to a fixed-size window
- S2-Attention (future): Shifted window attention for better long-range modeling

Available Implementations
-------------------------

Sliding Window Attention (SWA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces memory usage by limiting each token to attend to only a fixed window
of surrounding tokens. Works with any model architecture.

Example::

    from aligntune.core.long_context.attention import register_sliding_window_attention

    # Register SWA backend
    register_sliding_window_attention()

    # Load model with SWA
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        attn_implementation="swa"
    )
    model.config.sliding_window = 4096  # 4k window size

Registry Utilities
~~~~~~~~~~~~~~~~~~

The AttentionRegistry tracks registered implementations to avoid duplicates::

    from aligntune.core.long_context.attention import AttentionRegistry

    if not AttentionRegistry.is_registered("swa"):
        register_sliding_window_attention()

Integration with AlignTune
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your SFT config::

    from aligntune.core.sft.config import SFTConfig, ModelConfig

    config = SFTConfig(
        model=ModelConfig(
            name_or_path="meta-llama/Llama-2-7b",
            sliding_window=4096,  # Enable SWA with 4k window
            max_seq_length=8192,
        ),
        ...
    )

The model_loader will automatically register and apply SWA when sliding_window
is set in the config.
"""

from .sliding_window import (
    swa_mask_function,
    register_sliding_window_attention,
)
from .s2_attention import (
    s2_attention_forward,
    s2_mask_function,
    register_s2_attention,
)
from .registry import (
    AttentionRegistry,
    register_attention_implementation,
)

__all__ = [
    # Sliding Window Attention
    "swa_mask_function",
    "register_sliding_window_attention",
    # S2-Attention
    "s2_attention_forward",
    "s2_mask_function",
    "register_s2_attention",
    # Registry utilities
    "AttentionRegistry",
    "register_attention_implementation",
]
