"""Inference utilities for aligntune."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def maybe_enable_unsloth_inference(model: Any) -> Any:
    """Best-effort Unsloth fast-inference wrapper for evaluation.

    - If Unsloth / FastLanguageModel is unavailable or disabled, returns model unchanged.
    - If the model is incompatible, logs at debug level and returns model unchanged.
    - Uses an attribute flag to avoid double-wrapping.
    """
    # Check if PURE_TRL_MODE or other disabling flags are set
    import os
    if (os.environ.get('TRL_ONLY_MODE', '0') == '1' or
        os.environ.get('DISABLE_UNSLOTH_FOR_TRL', '0') == '1' or
        os.environ.get('PURE_TRL_MODE', '0') == '1'):
        return model

    # Avoid double-wrapping
    if getattr(model, "_unsloth_inference_enabled", False):
        return model

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception:
        # Unsloth not actually importable here
        return model

    try:
        FastLanguageModel.for_inference(model)
        setattr(model, "_unsloth_inference_enabled", True)
        logger.info("Enabled Unsloth fast inference for evaluation model.")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Unable to enable Unsloth fast inference for model: {e}")

    return model