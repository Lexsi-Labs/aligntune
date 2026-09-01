"""
Sliding-window and long-context attention helpers for AlignTune.

This module provides:

* :class:`SlidingWindowAttentionConfig` – a lightweight dataclass that
  captures the two parameters needed to configure sliding-window attention
  (window size and the number of globally attending tokens).

* :class:`LongContextAttentionHelper` – a stateless utility class that
  auto-detects the best attention implementation available in the current
  environment and returns the corresponding keyword-argument dict suitable
  for passing to ``transformers.AutoModelForCausalLM.from_pretrained``.

Attention implementation preference order
-----------------------------------------
1. ``flash_attention_2`` – requires the ``flash-attn`` package to be
   installed *and* a CUDA-capable device.  Provides the best throughput
   for long sequences and is required for memory-efficient long-context
   training.
2. ``sdpa`` – PyTorch Scaled Dot-Product Attention (available in
   PyTorch ≥ 2.0 via ``torch.nn.functional.scaled_dot_product_attention``).
   No extra package needed; good performance on modern GPUs.
3. ``eager`` – the plain HuggingFace Python attention loop.  Always
   available but slowest; suitable for CPU-only environments and debugging.

Typical usage::

    from aligntune.core.long_context.attention import (
        SlidingWindowAttentionConfig,
        LongContextAttentionHelper,
    )

    # Inspect sliding-window parameters for a Mistral-style model
    sw_cfg = SlidingWindowAttentionConfig(window_size=4096, global_tokens=1)

    # Resolve the best attention backend
    model_kwargs = LongContextAttentionHelper.get_attn_implementation(
        "mistralai/Mistral-7B-v0.1"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        **model_kwargs,
    )
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

AttnImplementation = Literal["flash_attention_2", "sdpa", "eager"]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindowAttentionConfig:
    """Configuration for sliding-window attention.

    Sliding-window attention (Beltagy et al., 2020; used in Mistral,
    Mixtral, Longformer, …) limits each token to attending over at most
    ``window_size`` neighbouring tokens.  A small number of *global* tokens
    (typically the CLS token or a fixed-size prefix) attend and are attended
    from the entire sequence.

    Attributes
    ----------
    window_size:
        Number of tokens each position can attend to (half-window on each
        side, so the effective receptive field is ``2 * window_size``
        tokens in bidirectional attention).  Must be positive.
        Typical values: 4 096 (Mistral 7B), 8 192 (Mixtral 8×7B).
    global_tokens:
        Number of tokens at the start of the sequence that attend globally
        (full sequence) and are attended to by all other tokens.  Setting
        this to 0 disables global attention entirely.  For CLS-based models
        set to 1; for prefix-based models set to the prefix length.
        Default: 0.

    Examples
    --------
    >>> cfg = SlidingWindowAttentionConfig(window_size=4096)
    >>> cfg.window_size
    4096
    >>> cfg.global_tokens
    0

    >>> cfg = SlidingWindowAttentionConfig(window_size=8192, global_tokens=64)
    >>> cfg.effective_window_tokens
    16384
    """

    window_size: int
    global_tokens: int = field(default=0)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be a positive integer, got {self.window_size}"
            )
        if self.global_tokens < 0:
            raise ValueError(
                f"global_tokens must be non-negative, got {self.global_tokens}"
            )

    @property
    def effective_window_tokens(self) -> int:
        """Total unique tokens visible per position including bidirectional reach.

        For a causal (decoder-only) model this is simply ``window_size``
        because attention is unidirectional.  For bidirectional models
        (encoders) each position sees ``2 * window_size`` tokens.  This
        property returns the bidirectional value as an upper bound.

        Returns
        -------
        int
            ``2 * window_size``.
        """
        return 2 * self.window_size

    def to_dict(self) -> Dict[str, int]:
        """Serialise to a plain dictionary.

        Returns
        -------
        Dict[str, int]
            ``{"window_size": ..., "global_tokens": ...}``
        """
        return {
            "window_size": self.window_size,
            "global_tokens": self.global_tokens,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "SlidingWindowAttentionConfig":
        """Deserialise from a plain dictionary.

        Parameters
        ----------
        d:
            Dictionary with keys ``"window_size"`` and optionally
            ``"global_tokens"``.

        Returns
        -------
        SlidingWindowAttentionConfig
        """
        return cls(
            window_size=d["window_size"],
            global_tokens=d.get("global_tokens", 0),
        )


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------


class LongContextAttentionHelper:
    """Stateless utility for resolving the best long-context attention backend.

    All public methods are class-methods so no instantiation is needed.
    """

    # Known models with published sliding-window sizes (informational only)
    _KNOWN_WINDOW_SIZES: Dict[str, int] = {
        "mistral": 4_096,
        "mixtral": 4_096,
        "longformer": 512,
        "bigbird": 64,
        "gemma": 8_192,
        "phi": 2_048,
        "qwen": 32_768,
    }

    @classmethod
    def get_attn_implementation(
        cls,
        model_name_or_path: str,
    ) -> Dict[str, object]:
        """Detect the best available attention implementation.

        Inspects the current Python environment to determine whether
        ``flash_attention_2``, PyTorch SDPA, or plain eager attention is
        available.  Returns a dict of keyword arguments that can be
        unpacked directly into
        ``AutoModelForCausalLM.from_pretrained(..., **kwargs)``.

        The method does *not* load the model; it only inspects packages
        and hardware.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace Hub model id or local path.  Currently used only
            for logging; future versions may inspect the model config to
            select a specialised backend.

        Returns
        -------
        Dict[str, object]
            A dictionary with the following keys:

            ``"attn_implementation"`` : ``str``
                One of ``"flash_attention_2"``, ``"sdpa"``, or
                ``"eager"``.
            ``"use_flash_attention_2"`` : ``bool``
                Convenience flag; ``True`` iff
                ``attn_implementation == "flash_attention_2"``.

        Examples
        --------
        >>> kwargs = LongContextAttentionHelper.get_attn_implementation(
        ...     "mistralai/Mistral-7B-v0.1"
        ... )
        >>> "attn_implementation" in kwargs
        True
        >>> kwargs["attn_implementation"] in ("flash_attention_2", "sdpa", "eager")
        True
        """
        impl = cls._resolve_implementation(model_name_or_path)
        use_fa2 = impl == "flash_attention_2"

        logger.info(
            "get_attn_implementation('%s') -> attn_implementation='%s', "
            "use_flash_attention_2=%s",
            model_name_or_path,
            impl,
            use_fa2,
        )

        return {
            "attn_implementation": impl,
            "use_flash_attention_2": use_fa2,
        }

    @classmethod
    def _resolve_implementation(cls, model_name_or_path: str) -> AttnImplementation:
        """Internal: pick the best attention implementation.

        Parameters
        ----------
        model_name_or_path:
            Model identifier (used for logging only).

        Returns
        -------
        AttnImplementation
            One of ``"flash_attention_2"``, ``"sdpa"``, ``"eager"``.
        """
        # 1. Try Flash Attention 2 first
        if cls._flash_attention_available():
            logger.debug(
                "flash_attention_2 selected for '%s' (flash-attn package found)",
                model_name_or_path,
            )
            return "flash_attention_2"

        # 2. Fall back to PyTorch SDPA
        if cls._sdpa_available():
            logger.debug(
                "sdpa selected for '%s' (torch >= 2.0, no flash-attn)",
                model_name_or_path,
            )
            return "sdpa"

        # 3. Last resort: eager
        logger.debug(
            "eager selected for '%s' (torch < 2.0 or SDPA unavailable)",
            model_name_or_path,
        )
        return "eager"

    @staticmethod
    def _flash_attention_available() -> bool:
        """Return ``True`` if ``flash_attn`` is importable.

        Flash Attention 2 requires:
        * The ``flash-attn`` Python package (``pip install flash-attn``).
        * A CUDA-capable GPU.
        * PyTorch with CUDA support.

        This check only inspects package availability; the GPU check is
        intentionally omitted because ``from_pretrained`` will raise a
        clear error if the hardware is absent.

        Returns
        -------
        bool
        """
        try:
            importlib.import_module("flash_attn")
            return True
        except ImportError:
            return False

    @staticmethod
    def _sdpa_available() -> bool:
        """Return ``True`` if PyTorch's SDPA kernel is available.

        SDPA (``torch.nn.functional.scaled_dot_product_attention``) was
        added in PyTorch 2.0 and is accelerated by fused CUDA kernels on
        Ampere+ GPUs and by efficient CPU paths on x86.

        Returns
        -------
        bool
        """
        try:
            import torch

            return hasattr(torch.nn.functional, "scaled_dot_product_attention")
        except ImportError:
            return False

    @classmethod
    def get_sliding_window_config(
        cls, model_name_or_path: str
    ) -> Optional[SlidingWindowAttentionConfig]:  # type: ignore[name-defined]
        """Try to infer a :class:`SlidingWindowAttentionConfig` from the model name.

        Matches the lower-cased model id against a table of known
        architectures and returns a pre-configured object.  Returns
        ``None`` for unknown architectures.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace Hub model id or local path.

        Returns
        -------
        SlidingWindowAttentionConfig or None
            A pre-populated config for known architectures; ``None``
            otherwise.
        """
        lower = model_name_or_path.lower()
        for key, window in cls._KNOWN_WINDOW_SIZES.items():
            if key in lower:
                cfg = SlidingWindowAttentionConfig(window_size=window)
                logger.debug(
                    "get_sliding_window_config('%s') -> window_size=%d (matched '%s')",
                    model_name_or_path,
                    window,
                    key,
                )
                return cfg

        logger.debug(
            "get_sliding_window_config('%s') -> None (no known architecture matched)",
            model_name_or_path,
        )
        return None

    @classmethod
    def build_model_kwargs(
        cls,
        model_name_or_path: str,
        dtype: str = "auto",
    ) -> Dict[str, object]:
        """Build a complete ``from_pretrained`` kwargs dict for long-context models.

        Combines the attention-implementation flags with a ``torch_dtype``
        setting.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace Hub model id or local path.
        dtype:
            One of ``"auto"``, ``"bfloat16"``, ``"float16"``, or
            ``"float32"``.  Passed directly to
            ``AutoModelForCausalLM.from_pretrained``.  Default: ``"auto"``.

        Returns
        -------
        Dict[str, object]
            Ready-to-unpack kwargs for ``from_pretrained``.
        """
        kwargs = cls.get_attn_implementation(model_name_or_path)

        if dtype != "auto":
            try:
                import torch

                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                if dtype in dtype_map:
                    kwargs["torch_dtype"] = dtype_map[dtype]
                else:
                    logger.warning(
                        "Unknown dtype '%s'; falling back to 'auto'", dtype
                    )
            except ImportError:
                pass
        else:
            kwargs["torch_dtype"] = "auto"

        return kwargs


# ---------------------------------------------------------------------------
# Re-export Optional for the type hint in get_sliding_window_config
# ---------------------------------------------------------------------------
from typing import Optional  # noqa: E402  (placed after class to avoid circular refs)
