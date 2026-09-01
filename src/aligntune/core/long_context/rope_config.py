"""
RoPE (Rotary Position Embedding) Configuration for Long Context

This module provides RopeConfig dataclass and auto-calculation logic for extending
model context windows using various RoPE scaling strategies supported by TRL/Transformers.

Supported RoPE Types:
- "default": Standard RoPE (no scaling, original implementation)
- "linear": Simple position interpolation (PI)
- "dynamic": Dynamic NTK-aware scaling
- "yarn": YaRN frequency-domain decomposition
- "longrope": LongRoPE per-dimension scaling
- "llama3": Llama3-style RoPE scaling

High-Level Logic:
    1. User provides RopeConfig with rope_type and optional params
    2. If factor is provided → use it directly (manual mode)
    3. If factor is NOT provided → auto-calculate from target_length / original_length
    4. Build rope_parameters dict for TRL
    5. Add variant-specific params based on rope_type

Usage:
    # Simple: auto-calculate from context length
    rope_config = RopeConfig(rope_type="linear")
    params = extract_rope_parameters(rope_config, model_config, max_seq_length=32768)
    # Returns: {"rope_type": "linear", "factor": 8.0}

    # Explicit target context
    rope_config = RopeConfig(rope_type="yarn", target_max_seq_length=131072)
    params = extract_rope_parameters(rope_config, model_config)

    # Manual control (no auto-calculation)
    rope_config = RopeConfig(rope_type="yarn", factor=8.0, rope_theta=10000.0)
    params = extract_rope_parameters(rope_config, model_config)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Model architecture compatibility matrix for RoPE scaling types
ROPE_COMPATIBILITY = {
    "llama": ["default", "linear", "dynamic", "yarn", "llama3"],
    "mistral": ["default", "linear", "dynamic", "yarn"],
    "mixtral": ["default", "linear", "dynamic", "yarn"],
    "qwen": ["default", "linear", "dynamic", "yarn"],
    "qwen2": ["default", "linear", "dynamic", "yarn", "llama3"],
    "gemma": ["default", "linear", "dynamic", "yarn"],
    "gemma2": ["default", "linear", "dynamic", "yarn", "llama3"],
    "phi": ["default"],  # Phi has limited RoPE scaling support
    "phi3": ["default"],  # Phi3 has its own RoPE implementation
    "stablelm": ["default", "linear", "dynamic"],
}


@dataclass
class RopeConfig:
    """
    Configuration for RoPE (Rotary Position Embedding) scaling.

    Provides smart defaults and auto-calculation for extending model context windows.

    Attributes:
        rope_type: Type of RoPE scaling to use. Options:
            - None: No RoPE applied (skip rope_parameters entirely)
            - "default": Standard RoPE without scaling (original implementation)
            - "linear": Linear position interpolation
            - "dynamic": Dynamic NTK-aware scaling
            - "yarn": YaRN frequency-domain decomposition
            - "longrope": LongRoPE per-dimension scaling
            - "llama3": Llama3-style RoPE scaling

        target_max_seq_length: Target context window size in tokens.
            If provided (and factor is None), used to auto-calculate factor.
            Examples: 32768 (32k), 65536 (64k), 131072 (128k)

        factor: Scaling factor for context extension.
            If provided, used directly (no auto-calculation).
            If None, auto-calculated from target_max_seq_length / original_max.
            Example: factor=4.0 extends 32k → 128k

        rope_theta: Base period of RoPE embeddings (optional).
            Defaults to model's default_theta (typically 10000.0).

        original_max_position_embeddings: Original context window of the model (optional).
            If None, auto-extracted from model.config.max_position_embeddings.
            Required for: yarn, longrope, llama3

        partial_rotary_factor: Percentage of query/key head embedding to apply RoPE (optional).

        # YaRN-specific parameters
        attention_factor: Scaling factor for attention computation (optional).
            Auto-calculated by TRL if not provided.

        beta_fast: Boundary for extrapolation in linear ramp (YaRN only, optional).
            Default: 32

        beta_slow: Boundary for interpolation in linear ramp (YaRN only, optional).
            Default: 1

        # LongRoPE-specific parameters
        short_factor: Scaling factors for short contexts (< original_max, REQUIRED for longrope).
            Must be list[float] with length = hidden_size / num_heads / 2.

        long_factor: Scaling factors for long contexts (>= original_max, REQUIRED for longrope).
            Must be list[float] with length = hidden_size / num_heads / 2.

        # Llama3-specific parameters
        low_freq_factor: Scaling factor for low frequency components (REQUIRED for llama3).

        high_freq_factor: Scaling factor for high frequency components (REQUIRED for llama3).
    """

    # Core parameters
    rope_type: Optional[str] = None
    target_max_seq_length: Optional[int] = None
    factor: Optional[float] = None
    rope_theta: Optional[float] = None
    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None

    # YaRN-specific
    attention_factor: Optional[float] = None
    beta_fast: Optional[float] = None
    beta_slow: Optional[float] = None

    # LongRoPE-specific
    short_factor: Optional[List[float]] = None
    long_factor: Optional[List[float]] = None

    # Llama3-specific
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None

    def __post_init__(self):
        """Validate rope_type after initialization."""
        if self.rope_type is not None:
            valid_types = ["default", "linear", "dynamic", "yarn", "longrope", "llama3"]
            if self.rope_type not in valid_types:
                raise ValueError(
                    f"rope_type must be one of {valid_types}, got '{self.rope_type}'"
                )


def extract_rope_parameters(
    rope_config: RopeConfig,
    model_config: Any,
    max_seq_length: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract and auto-calculate RoPE parameters for model loading.

    Converts RopeConfig into rope_parameters dict for Transformers' from_pretrained().

    Logic:
        1. If rope_type is None → return None (no RoPE)
        2. If factor is provided → use it directly (manual mode, skip calculation)
        3. If factor is None → auto-calculate from target_length / original_length
        4. Build rope_parameters dict with rope_type and factor
        5. Add variant-specific parameters based on rope_type

    Args:
        rope_config: RopeConfig with user configuration
        model_config: Model config from AutoConfig.from_pretrained (for extracting original_max)
        max_seq_length: Max sequence length from training config (fallback for target)

    Returns:
        Dict for model_kwargs["rope_parameters"] or None if no RoPE

    Raises:
        ValueError: If required parameters are missing for the specified rope_type
    """

    # No RoPE scaling requested
    if rope_config.rope_type is None:
        logger.info("No RoPE scaling configured (rope_type=None)")
        return None

    # Check if model supports RoPE
    # Models with RoPE typically have rope_theta or rope_scaling in config
    supports_rope = (
        hasattr(model_config, 'rope_theta') or
        hasattr(model_config, 'rope_scaling') or
        hasattr(model_config, 'rotary_emb_base')
    )

    if not supports_rope:
        logger.warning(
            f"Model does not appear to support RoPE scaling (no rope_theta/rope_scaling in config). "
            f"Skipping RoPE configuration. Model type: {getattr(model_config, 'model_type', 'unknown')}"
        )
        return None

    logger.info(f"Configuring RoPE scaling: type={rope_config.rope_type}")

    # =========================================================================
    # STEP 1: Determine scaling factor
    # =========================================================================
    factor = None
    original_length = None
    target_length = None

    if rope_config.factor is not None:
        # MANUAL MODE: User provided factor, use it directly
        factor = rope_config.factor
        logger.info(f"Using user-provided scaling factor: {factor:.2f}x")

        # Get original_length for logging (optional)
        if rope_config.original_max_position_embeddings is not None:
            original_length = rope_config.original_max_position_embeddings
        elif hasattr(model_config, 'max_position_embeddings'):
            original_length = model_config.max_position_embeddings

    else:
        # AUTO-CALCULATION MODE: Calculate factor from target_length

        # "default" type doesn't need factor
        if rope_config.rope_type == "default":
            logger.info("RoPE type 'default' does not require scaling factor")
        else:
            # Get target length
            target_length = rope_config.target_max_seq_length or max_seq_length
            if target_length is None:
                raise ValueError(
                    "Cannot auto-calculate scaling factor. "
                    "Provide one of: rope_config.factor, rope_config.target_max_seq_length, or max_seq_length"
                )

            # Get original length from model
            original_length = rope_config.original_max_position_embeddings
            if original_length is None:
                if hasattr(model_config, 'max_position_embeddings'):
                    original_length = model_config.max_position_embeddings
                    logger.info(f"Extracted original context from model: {original_length} tokens")
                else:
                    original_length = 2048
                    logger.warning(
                        f"Could not extract max_position_embeddings from model, using default: {original_length}"
                    )

            # Check if we need to scale at all
            if target_length <= original_length:
                logger.info(
                    f"Target length ({target_length}) <= original length ({original_length}). "
                    f"No RoPE scaling needed. Skipping."
                )
                return None

            # Calculate factor
            factor = target_length / original_length
            logger.info(
                f"Auto-calculated scaling factor: {factor:.2f}x ({original_length} → {target_length} tokens)"
            )

    # =========================================================================
    # STEP 2: Build base rope_parameters dict
    # =========================================================================
    # Different models expect different keys: some use "type", some use "rope_type"
    # Add both for maximum compatibility
    rope_params: Dict[str, Any] = {
        "type": rope_config.rope_type,        # For Phi3, older models
        "rope_type": rope_config.rope_type    # For newer models
    }

    # Add factor if applicable (not needed for "default")
    if factor is not None:
        rope_params["factor"] = factor

    # Add rope_theta (required by transformers)
    # Extract from model config or use user-provided value
    rope_theta = rope_config.rope_theta
    if rope_theta is None:
        # Try to get from model config
        if hasattr(model_config, 'rope_theta') and model_config.rope_theta is not None:
            rope_theta = model_config.rope_theta
        else:
            # Default for most models (Llama 2, Mistral, etc.)
            rope_theta = 10000.0
            logger.info(f"rope_theta not found in config, using default: {rope_theta}")

    rope_params["rope_theta"] = rope_theta

    if rope_config.partial_rotary_factor is not None:
        rope_params["partial_rotary_factor"] = rope_config.partial_rotary_factor

    # =========================================================================
    # STEP 3: Add variant-specific parameters
    # =========================================================================

    if rope_config.rope_type == "default":
        # No additional params needed
        pass

    elif rope_config.rope_type in ["linear", "dynamic"]:
        # Only need factor (already added)
        pass

    elif rope_config.rope_type == "yarn":
        # Get original_length (required for yarn)
        if original_length is None:
            original_length = rope_config.original_max_position_embeddings
            if original_length is None and hasattr(model_config, 'max_position_embeddings'):
                original_length = model_config.max_position_embeddings
            if original_length is None:
                raise ValueError(
                    "YaRN requires original_max_position_embeddings. "
                    "Provide rope_config.original_max_position_embeddings"
                )

        rope_params["original_max_position_embeddings"] = original_length

        # Optional parameters with TRL defaults
        if rope_config.attention_factor is not None:
            rope_params["attention_factor"] = rope_config.attention_factor
        if rope_config.beta_fast is not None:
            rope_params["beta_fast"] = rope_config.beta_fast
        if rope_config.beta_slow is not None:
            rope_params["beta_slow"] = rope_config.beta_slow

    elif rope_config.rope_type == "longrope":
        # Get original_length (required)
        if original_length is None:
            original_length = rope_config.original_max_position_embeddings
            if original_length is None and hasattr(model_config, 'max_position_embeddings'):
                original_length = model_config.max_position_embeddings
            if original_length is None:
                raise ValueError(
                    "LongRoPE requires original_max_position_embeddings. "
                    "Provide rope_config.original_max_position_embeddings"
                )

        rope_params["original_max_position_embeddings"] = original_length

        # Optional attention_factor
        if rope_config.attention_factor is not None:
            rope_params["attention_factor"] = rope_config.attention_factor

        # Required: short_factor and long_factor
        if rope_config.short_factor is None or rope_config.long_factor is None:
            raise ValueError(
                "LongRoPE requires 'short_factor' and 'long_factor' parameters. "
                "These are model-specific lists. Check model documentation."
            )

        rope_params["short_factor"] = rope_config.short_factor
        rope_params["long_factor"] = rope_config.long_factor

    elif rope_config.rope_type == "llama3":
        # Get original_length (required)
        if original_length is None:
            original_length = rope_config.original_max_position_embeddings
            if original_length is None and hasattr(model_config, 'max_position_embeddings'):
                original_length = model_config.max_position_embeddings
            if original_length is None:
                raise ValueError(
                    "Llama3 RoPE requires original_max_position_embeddings. "
                    "Provide rope_config.original_max_position_embeddings"
                )

        rope_params["original_max_position_embeddings"] = original_length

        # Required: low_freq_factor and high_freq_factor
        if rope_config.low_freq_factor is None or rope_config.high_freq_factor is None:
            raise ValueError(
                "Llama3 RoPE requires 'low_freq_factor' and 'high_freq_factor'. "
                "Check model documentation for correct values."
            )

        rope_params["low_freq_factor"] = rope_config.low_freq_factor
        rope_params["high_freq_factor"] = rope_config.high_freq_factor

    # =========================================================================
    # STEP 4: Log final configuration
    # =========================================================================
    logger.info("=" * 80)
    logger.info("RoPE Scaling Configuration")
    logger.info(f"  Type: {rope_config.rope_type}")
    if factor is not None:
        logger.info(f"  Factor: {factor:.2f}x")
    if original_length is not None and target_length is not None:
        logger.info(f"  Context: {original_length} → {target_length} tokens")
    elif original_length is not None:
        logger.info(f"  Original context: {original_length} tokens")
    if rope_config.rope_theta is not None:
        logger.info(f"  Theta: {rope_config.rope_theta}")
    logger.info("=" * 80)

    return rope_params
