"""
Long context support for AlignTune (v3.9).

This package provides utilities for extending the context window of large language
models beyond their original training length using Rotary Position Embedding (RoPE)
scaling strategies, together with dataset loaders for long-context SFT.

Supported scaling methods
-------------------------

- **linear**: Simple position rescaling by a constant factor. Fastest and most
  portable; works well for moderate extensions (up to ~4×). May degrade on
  out-of-distribution positions at high extension ratios.

- **dynamic**: NTK-aware dynamic scaling that adjusts the base frequency at
  inference time based on actual sequence length (LM-infinite style). Adapts
  without requiring re-training.

- **yarn**: Yet Another RoPE Extension (Peng et al., 2023). Uses a
  frequency-domain decomposition to preserve local attention patterns while
  extending the global context. Best quality for large extensions with fine-tuning.

- **ntk**: NTK-by-parts scaling developed by the LocalLLaMA community.
  Applies different scaling factors to different frequency bands, preserving
  short-range positional sensitivity while extending long-range capacity.

Dataset loaders
---------------

:class:`LongContextDatasetLoader` supports the following datasets:

- ``"longalpaca"``    — ``abacusai/LongAlpaca-16k`` from the Hugging Face Hub.
- ``"books3_chunks"`` — Books3 chunked to ≤32k segments (stub if not local).
- ``"arxiv_chunks"``  — arXiv papers chunked to ≤32k segments (stub if not local).

Typical usage
-------------

RoPE scaling::

    from aligntune.core.long_context import RopeScalingConfig, RopeScalingApplier

    rope_config = RopeScalingConfig(
        type="yarn",
        factor=4.0,
        original_max_position=4096,
        target_max_position=16384,
    )
    RopeScalingApplier.validate_config(rope_config)
    hf_rope_dict = RopeScalingApplier.build_rope_config(rope_config)
    # Pass hf_rope_dict as model_kwargs["rope_scaling"] to from_pretrained.

Long-context dataset loading::

    from aligntune.core.long_context import LongContextDatasetLoader, DataRecord

    loader = LongContextDatasetLoader()
    records: list[DataRecord] = loader.load("longalpaca", max_length=32768)
    for record in records[:3]:
        print(record["source"], record["length"])
"""

from aligntune.core.long_context.rope_scaling import (
    RopeScalingConfig,
    RopeScalingApplier,
)
from aligntune.core.long_context.datasets import (
    LongContextDatasetLoader,
    DataRecord,
)
from aligntune.core.long_context.packing import DocumentPacker
from aligntune.core.long_context.attention import (
    SlidingWindowAttentionConfig,
    LongContextAttentionHelper,
)
from aligntune.core.long_context.benchmarks import (
    LongContextBenchmark,
    NIAHResult,
)

# Auto-register custom attention implementations
# This makes "swa" and "s2" available as attn_implementation options
try:
    from aligntune.core.long_context.custom_attention import (
        register_sliding_window_attention,
        register_s2_attention,
        AttentionRegistry,
    )

    # Register SWA automatically on import
    if not AttentionRegistry.is_registered("swa"):
        register_sliding_window_attention()

    # Register S2 automatically on import
    if not AttentionRegistry.is_registered("s2"):
        register_s2_attention()
except ImportError:
    # Transformers may not be available or version too old
    pass

__all__ = [
    # RoPE scaling utilities
    "RopeScalingConfig",
    "RopeScalingApplier",
    # Long-context dataset loaders
    "LongContextDatasetLoader",
    "DataRecord",
    # Document packing
    "DocumentPacker",
    # Attention helpers
    "SlidingWindowAttentionConfig",
    "LongContextAttentionHelper",
    # Benchmarks
    "LongContextBenchmark",
    "NIAHResult",
    # Custom attention (auto-registered)
    "register_sliding_window_attention",
    "AttentionRegistry",
]
