"""
RoPE (Rotary Position Embedding) scaling utilities for long-context inference.

This module provides dataclasses and helper classes that encode the four main
RoPE extension strategies and translate them into the dictionary format expected
by HuggingFace ``transformers`` model configurations (the ``rope_scaling`` key
passed to ``AutoModelForCausalLM.from_pretrained``).

Strategy overview
-----------------

linear
    Introduced by Press et al. (2022) as "Position Interpolation".  Every
    position index is divided by the scaling factor before the sinusoidal
    projection, effectively compressing the position space.  No retraining is
    strictly required, though a short fine-tuning pass on longer sequences
    improves quality significantly.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
    Embedding", 2021; Chen et al., "Extending Context Window of Large Language
    Models via Positional Interpolation", 2023.

dynamic
    NTK-aware dynamic scaling (sometimes called "Code Llama" scaling or
    Lm-infinite style).  Instead of a fixed division of position indices, the
    base frequency ``θ`` of the RoPE sinusoids is scaled dynamically based on
    the actual input length at runtime.  The model therefore sees no
    out-of-distribution positions for sequences up to ``factor × original``.
    No fine-tuning is required.

    Reference: bloc97 LocalLLaMA post "NTK-Aware Scaled RoPE allows LLaMA
    models to have extended (8k+) context size without any fine-tuning and
    minimal perplexity degradation", 2023.

yarn
    Yet Another RoPE Extension (YaRN).  Decomposes the frequency spectrum of
    RoPE into three regions (low, mid, high frequency) and applies different
    interpolation strategies to each, preserving local attention patterns while
    still extending the global context window.  YaRN achieves state-of-the-art
    perplexity on long-context benchmarks when combined with short fine-tuning.

    Reference: Peng et al., "YaRN: Efficient Context Window Extension of Large
    Language Models", arXiv:2309.00071, 2023.

ntk
    NTK-by-parts scaling, developed by the LocalLLaMA community as an
    improvement over vanilla NTK scaling.  Applies a smoothly varying scale
    factor across frequency dimensions (``parts``) so that high-frequency
    dimensions (capturing short-range dependencies) are left mostly unscaled,
    while low-frequency dimensions (capturing long-range dependencies) receive
    the full scaling factor.  This preserves the model's ability to reason
    about local token order while extending global context.

    Reference: LocalLLaMA community thread, "NTK-By-Parts Scaling", 2023;
    further formalised in the YaRN paper (Peng et al., 2023, §4.1).

HuggingFace integration
-----------------------

All four strategies map onto the ``rope_scaling`` dict that ``transformers``
accepts when loading models::

    model_kwargs["rope_scaling"] = {
        "type": "linear" | "dynamic" | "yarn" | "longrope",
        "factor": <float>,
        # yarn / ntk also accept:
        "original_max_position_embeddings": <int>,
    }

The ``RopeScalingApplier`` class encapsulates this translation, validation, and
factor computation so that callers never need to know the exact key names
required by each transformers version.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported strategy literals
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"linear", "dynamic", "yarn", "ntk"})

# HuggingFace transformers uses "longrope" as the internal name for the
# NTK-by-parts approach in some model families (e.g., Phi-3).  We expose "ntk"
# as the user-facing name and translate it at build time.
_HF_TYPE_MAP: Dict[str, str] = {
    "linear": "linear",
    "dynamic": "dynamic",
    "yarn": "yarn",
    "ntk": "longrope",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RopeScalingConfig:
    """
    Immutable specification of a RoPE context-extension strategy.

    Parameters
    ----------
    type:
        One of ``"linear"``, ``"dynamic"``, ``"yarn"``, or ``"ntk"``.
        See module docstring for strategy descriptions.
    factor:
        Multiplicative extension factor relative to the base model.  A value of
        ``4.0`` means the target context window is four times the original.
        Must be strictly greater than 1.0; the identity (no extension) is
        represented by ``factor=1.0`` and would be rejected by
        :meth:`RopeScalingApplier.validate_config`.
    original_max_position:
        The base model's native maximum sequence length (from the model card or
        config.json ``max_position_embeddings``).  Typically 2048, 4096, or
        8192 for current open-weight LLMs.
    target_max_position:
        The desired extended context length, e.g. 32768 or 131072.  Must be
        strictly greater than ``original_max_position``.

        If not supplied explicitly, it is derived from
        ``original_max_position × factor`` during
        :meth:`RopeScalingApplier.validate_config`.

    Examples
    --------
    Linear 4× extension for a LLaMA-2 7B (base 4096):

    >>> cfg = RopeScalingConfig(
    ...     type="linear",
    ...     factor=4.0,
    ...     original_max_position=4096,
    ...     target_max_position=16384,
    ... )

    YaRN 8× extension for Mistral (base 8192) targeting 65536:

    >>> cfg = RopeScalingConfig(
    ...     type="yarn",
    ...     factor=8.0,
    ...     original_max_position=8192,
    ...     target_max_position=65536,
    ... )
    """

    type: Literal["linear", "dynamic", "yarn", "ntk"]
    """Scaling strategy.  One of ``linear``, ``dynamic``, ``yarn``, ``ntk``."""

    factor: float
    """Extension multiplier (e.g. 4.0 → 4× the base context length)."""

    original_max_position: int
    """Base model's native ``max_position_embeddings``."""

    target_max_position: int
    """Desired extended context length in tokens."""

    def __post_init__(self) -> None:
        """Coerce types and perform lightweight sanity checks at construction."""
        # Coerce numeric types to be safe
        self.factor = float(self.factor)
        self.original_max_position = int(self.original_max_position)
        self.target_max_position = int(self.target_max_position)

        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"RopeScalingConfig.type must be one of {sorted(_VALID_TYPES)}, "
                f"got {self.type!r}."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RopeScalingConfig("
            f"type={self.type!r}, "
            f"factor={self.factor}, "
            f"original_max_position={self.original_max_position}, "
            f"target_max_position={self.target_max_position})"
        )


# ---------------------------------------------------------------------------
# Applier
# ---------------------------------------------------------------------------


class RopeScalingApplier:
    """
    Stateless helper that converts :class:`RopeScalingConfig` instances into
    the ``rope_scaling`` dicts required by HuggingFace ``transformers``.

    All public methods are class methods; there is no need to instantiate this
    class.

    Design notes
    ------------
    - The class is stateless so it can be imported and called at any point in
      the model-loading pipeline without managing lifecycle.
    - Validation is deliberately separated from building so that callers can
      validate early (e.g. at config parse time) and build late (at model load
      time) without duplication.
    - Each strategy documents its mapping precisely so future maintainers can
      verify correctness against the upstream transformers implementation.

    Examples
    --------
    >>> cfg = RopeScalingConfig(
    ...     type="yarn",
    ...     factor=4.0,
    ...     original_max_position=4096,
    ...     target_max_position=16384,
    ... )
    >>> RopeScalingApplier.validate_config(cfg)          # raises if invalid
    >>> rope_dict = RopeScalingApplier.build_rope_config(cfg)
    >>> # Merge with the base config's rope_parameters so rope_theta (and any
    >>> # other model-specific defaults) survive the override -- passing
    >>> # rope_dict alone replaces rope_parameters wholesale.
    >>> base_config = AutoConfig.from_pretrained(model_name)
    >>> model_kwargs["rope_scaling"] = {**base_config.rope_parameters, **rope_dict}
    """

    # ------------------------------------------------------------------
    # Public class methods
    # ------------------------------------------------------------------

    @classmethod
    def validate_config(cls, rope_config: RopeScalingConfig) -> None:
        """
        Validate a :class:`RopeScalingConfig` and raise ``ValueError`` on any
        detected misconfiguration.

        Validation rules
        ----------------
        1. ``type`` must be one of the supported strategies.
        2. ``factor`` must be > 1.0 (factor=1.0 means no extension; callers
           should omit rope_scaling entirely in that case).
        3. ``original_max_position`` must be a positive integer.
        4. ``target_max_position`` must be strictly greater than
           ``original_max_position``.
        5. ``target_max_position`` must be consistent with ``factor`` within a
           10 % relative tolerance (allowing for rounding to a power of 2).

        Parameters
        ----------
        rope_config:
            The config to validate.

        Raises
        ------
        ValueError
            On any validation failure; message contains the violated rule.

        Examples
        --------
        >>> cfg = RopeScalingConfig("linear", 4.0, 4096, 16384)
        >>> RopeScalingApplier.validate_config(cfg)  # no exception

        >>> bad = RopeScalingConfig("linear", 0.5, 4096, 2048)
        >>> RopeScalingApplier.validate_config(bad)  # raises ValueError
        """
        errors: list[str] = []

        # Rule 1: valid type (already checked in __post_init__ but re-check for
        # belt-and-suspenders safety when objects are mutated externally)
        if rope_config.type not in _VALID_TYPES:
            errors.append(
                f"type must be one of {sorted(_VALID_TYPES)}, "
                f"got {rope_config.type!r}."
            )

        # Rule 2: factor > 1.0
        if rope_config.factor <= 1.0:
            errors.append(
                f"factor must be > 1.0 (got {rope_config.factor}). "
                "For no extension, omit rope_scaling entirely."
            )

        # Rule 3: original_max_position positive
        if rope_config.original_max_position <= 0:
            errors.append(
                f"original_max_position must be a positive integer, "
                f"got {rope_config.original_max_position}."
            )

        # Rule 4: target > original
        if rope_config.target_max_position <= rope_config.original_max_position:
            errors.append(
                f"target_max_position ({rope_config.target_max_position}) must be "
                f"strictly greater than original_max_position "
                f"({rope_config.original_max_position})."
            )

        # Rule 5: consistency between factor and target/original (10 % tol)
        if rope_config.original_max_position > 0 and rope_config.factor > 1.0:
            implied_target = rope_config.original_max_position * rope_config.factor
            relative_diff = abs(rope_config.target_max_position - implied_target) / implied_target
            if relative_diff > 0.10:
                errors.append(
                    f"target_max_position ({rope_config.target_max_position}) is "
                    f"inconsistent with original_max_position "
                    f"({rope_config.original_max_position}) × factor "
                    f"({rope_config.factor}) = {implied_target:.0f}. "
                    f"Relative difference {relative_diff:.1%} exceeds 10 % tolerance. "
                    "Either adjust target_max_position or recompute factor."
                )

        if errors:
            raise ValueError(
                "RopeScalingConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "RopeScalingConfig validated: type=%s, factor=%.2f, "
            "original=%d, target=%d",
            rope_config.type,
            rope_config.factor,
            rope_config.original_max_position,
            rope_config.target_max_position,
        )

    @classmethod
    def build_rope_config(cls, rope_config: RopeScalingConfig) -> Dict[str, Any]:
        """
        Convert a :class:`RopeScalingConfig` into the ``rope_scaling`` dict
        accepted by HuggingFace ``transformers``.

        The returned dict can be passed directly as
        ``model_kwargs["rope_scaling"]`` to ``AutoModelForCausalLM.from_pretrained``.

        Strategy-specific output shapes
        --------------------------------

        **linear**::

            {
                "type": "linear",
                "factor": <float>,
            }

        HuggingFace divides every position index by ``factor`` before computing
        sinusoidal embeddings, performing simple position interpolation.

        **dynamic**::

            {
                "type": "dynamic",
                "factor": <float>,
            }

        HuggingFace recomputes the RoPE base frequency at runtime when the
        sequence length exceeds the original maximum, scaling the base
        by ``factor ** (dim / (dim - 2))``.

        **yarn**::

            {
                "type": "yarn",
                "factor": <float>,
                "original_max_position_embeddings": <int>,
            }

        HuggingFace applies frequency-domain decomposition across three bands
        (low / mid / high) with different interpolation strategies for each.
        ``original_max_position_embeddings`` anchors the band boundaries.

        **ntk** (mapped to HuggingFace ``"longrope"``)::

            {
                "type": "longrope",
                "factor": <float>,
                "original_max_position_embeddings": <int>,
            }

        ``longrope`` is the HuggingFace name for NTK-by-parts / LongRoPE
        scaling.  A different scale factor is applied per frequency dimension,
        smoothly transitioning from 1.0 (high-frequency, short-range) to
        ``factor`` (low-frequency, long-range).

        Parameters
        ----------
        rope_config:
            A validated (or about to be validated) :class:`RopeScalingConfig`.

        Returns
        -------
        dict
            A ``rope_scaling`` dict ready for ``from_pretrained``.

        Raises
        ------
        ValueError
            If the config fails internal validation (delegates to
            :meth:`validate_config`).

        Examples
        --------
        >>> cfg = RopeScalingConfig("linear", 4.0, 4096, 16384)
        >>> RopeScalingApplier.build_rope_config(cfg)
        {'type': 'linear', 'factor': 4.0}

        >>> cfg = RopeScalingConfig("yarn", 4.0, 4096, 16384)
        >>> RopeScalingApplier.build_rope_config(cfg)
        {'type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 4096}
        """
        cls.validate_config(rope_config)

        hf_type = _HF_TYPE_MAP[rope_config.type]

        # Base dict shared by all strategies.
        # NOTE: newer transformers versions read the strategy name from the
        # "rope_type" key (config.rope_parameters["rope_type"]) instead of
        # "type". We set both so this dict works whether it reaches the model
        # via the legacy `type` convention or the current `rope_type` one.
        # Also note that passing this dict as `rope_scaling=` to
        # `from_pretrained` REPLACES the base model's rope_parameters wholesale
        # (transformers' config kwarg-override path does not merge), so
        # `rope_theta` is dropped unless the caller merges it back in from the
        # base config -- see the RopeScalingApplier module docstring.
        rope_dict: Dict[str, Any] = {
            "type": hf_type,
            "rope_type": hf_type,
            "factor": rope_config.factor,
        }

        # Strategies that need the original position count as an anchor
        if rope_config.type in {"yarn", "ntk"}:
            rope_dict["original_max_position_embeddings"] = (
                rope_config.original_max_position
            )

        logger.debug(
            "Built rope_scaling dict for strategy %r: %s",
            rope_config.type,
            rope_dict,
        )

        return rope_dict

    @classmethod
    def compute_scale_factor(
        cls,
        original_max: int,
        target_max: int,
    ) -> float:
        """
        Compute the RoPE scaling factor required to extend context from
        ``original_max`` to ``target_max`` tokens.

        This is a simple quotient, but is provided as an explicit method so that
        callers need not remember the formula or worry about integer division.

        The factor is always rounded to 6 decimal places to avoid floating-point
        noise propagating into config dicts and checksums.

        Parameters
        ----------
        original_max:
            The base model's native ``max_position_embeddings``.
        target_max:
            The desired extended context length.

        Returns
        -------
        float
            ``target_max / original_max``, rounded to 6 decimal places.

        Raises
        ------
        ValueError
            If ``original_max <= 0`` or ``target_max <= original_max``.

        Examples
        --------
        >>> RopeScalingApplier.compute_scale_factor(4096, 16384)
        4.0

        >>> RopeScalingApplier.compute_scale_factor(2048, 131072)
        64.0

        >>> RopeScalingApplier.compute_scale_factor(4096, 32768)
        8.0
        """
        if original_max <= 0:
            raise ValueError(
                f"original_max must be a positive integer, got {original_max}."
            )
        if target_max <= original_max:
            raise ValueError(
                f"target_max ({target_max}) must be strictly greater than "
                f"original_max ({original_max})."
            )

        factor = target_max / original_max
        factor = round(factor, 6)

        logger.debug(
            "compute_scale_factor: %d → %d = %.6f",
            original_max,
            target_max,
            factor,
        )

        return factor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _strategy_description(cls, strategy: str) -> str:
        """
        Return a human-readable description of a scaling strategy.

        Intended for logging and error messages only; not part of the public API.

        Parameters
        ----------
        strategy:
            One of the supported strategy names.

        Returns
        -------
        str
            A one-sentence description of the strategy.
        """
        descriptions: Dict[str, str] = {
            "linear": (
                "Linear position interpolation: divides every position index by "
                "the scale factor, compressing the position space uniformly. "
                "Fast and widely supported; quality degrades at very high factors "
                "without fine-tuning."
            ),
            "dynamic": (
                "NTK-aware dynamic scaling: adjusts the RoPE base frequency at "
                "inference time according to actual sequence length, so no "
                "out-of-distribution positions are ever seen. No retraining required."
            ),
            "yarn": (
                "YaRN (Peng et al., 2023): frequency-domain decomposition into "
                "low/mid/high bands with per-band interpolation strategies. Best "
                "quality for large extensions when combined with short fine-tuning."
            ),
            "ntk": (
                "NTK-by-parts (LocalLLaMA / LongRoPE): per-dimension scaling that "
                "preserves high-frequency (short-range) dimensions and applies full "
                "scaling only to low-frequency (long-range) dimensions."
            ),
        }
        return descriptions.get(strategy, f"Unknown strategy: {strategy!r}")

    @classmethod
    def _log_config_summary(cls, rope_config: RopeScalingConfig) -> None:
        """
        Emit an INFO-level log line summarising the RoPE scaling configuration.

        Called automatically by :meth:`build_rope_config` after successful
        validation.  Not part of the public API.

        Parameters
        ----------
        rope_config:
            The config to summarise.
        """
        logger.info(
            "RoPE scaling: strategy=%s, factor=%.2f, "
            "context %d → %d tokens. %s",
            rope_config.type,
            rope_config.factor,
            rope_config.original_max_position,
            rope_config.target_max_position,
            cls._strategy_description(rope_config.type),
        )

    @classmethod
    def _validate_factor_is_power_of_two_friendly(
        cls,
        factor: float,
        *,
        warn_only: bool = True,
    ) -> None:
        """
        Warn (or raise) if ``factor`` is not a power of two.

        Most published long-context recipes use powers of two (2×, 4×, 8×, …)
        because they align with the model's rotary frequency grid.  Non-power-
        of-two factors are valid but may yield suboptimal interpolation quality.

        Parameters
        ----------
        factor:
            The scale factor to check.
        warn_only:
            If ``True`` (default), emit a warning; if ``False``, raise
            ``ValueError``.

        Raises
        ------
        ValueError
            If ``warn_only=False`` and ``factor`` is not a power of two.
        """
        log2_factor = math.log2(factor)
        is_power_of_two = abs(log2_factor - round(log2_factor)) < 1e-9

        if not is_power_of_two:
            msg = (
                f"RoPE scale factor {factor} is not a power of two. "
                "Powers-of-two factors (2, 4, 8, …) align with the model's "
                "rotary frequency grid and typically yield better interpolation "
                "quality. Non-powers are valid but may degrade perplexity."
            )
            if warn_only:
                logger.warning(msg)
            else:
                raise ValueError(msg)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def make_rope_config_for_target(
    strategy: Literal["linear", "dynamic", "yarn", "ntk"],
    original_max_position: int,
    target_max_position: int,
) -> RopeScalingConfig:
    """
    Convenience factory that computes the scale factor automatically.

    Given the original model maximum and the desired extended context, this
    function computes the required ``factor`` via
    :meth:`RopeScalingApplier.compute_scale_factor` and returns a validated
    :class:`RopeScalingConfig`.

    Parameters
    ----------
    strategy:
        Scaling strategy: ``"linear"``, ``"dynamic"``, ``"yarn"``, or ``"ntk"``.
    original_max_position:
        Base model's native ``max_position_embeddings``.
    target_max_position:
        Desired extended context length.

    Returns
    -------
    RopeScalingConfig
        A fully initialised and validated config.

    Raises
    ------
    ValueError
        If any parameter is invalid or the resulting config fails validation.

    Examples
    --------
    >>> cfg = make_rope_config_for_target("yarn", 4096, 16384)
    >>> cfg.factor
    4.0
    >>> cfg.type
    'yarn'
    """
    factor = RopeScalingApplier.compute_scale_factor(
        original_max_position, target_max_position
    )
    config = RopeScalingConfig(
        type=strategy,
        factor=factor,
        original_max_position=original_max_position,
        target_max_position=target_max_position,
    )
    RopeScalingApplier.validate_config(config)
    return config


__all__ = [
    "RopeScalingConfig",
    "RopeScalingApplier",
    "make_rope_config_for_target",
]
