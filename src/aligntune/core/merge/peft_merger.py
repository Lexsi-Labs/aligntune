"""
PEFTMerger — lightweight LoRA adapter merger using the `peft` library.

Provides a simple alternative to MergekitMerger when only LoRA adapter
merging is needed and mergekit is not available.

Usage:
    merger = PEFTMerger()
    output = merger.merge(
        base_model="path/to/base_model",
        output_path="./merged_model",
        adapter_path="path/to/lora_adapter",
    )
"""

import logging
from pathlib import Path
from typing import Optional, Union

from .base import BaseMerger

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ["lora-merge"]


def _require_peft():
    """Raise a helpful ImportError if peft is not installed."""
    try:
        import peft  # noqa: F401
        return peft
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA adapter merging.\n"
            "Install it with:  pip install peft\n"
            "See https://github.com/huggingface/peft for details."
        ) from exc


def _require_transformers():
    """Raise a helpful ImportError if transformers is not installed."""
    try:
        import transformers  # noqa: F401
        return transformers
    except ImportError as exc:
        raise ImportError(
            "transformers is required for LoRA adapter merging.\n"
            "Install it with:  pip install transformers"
        ) from exc


class PEFTMerger(BaseMerger):
    """
    Merges a LoRA adapter into its base model using ``peft.PeftModel.merge_and_unload()``.

    This is a lightweight alternative when mergekit is not available.  Only
    ``lora-merge`` is supported — for TIES / DARE / SLERP use MergekitMerger.
    """

    def supports_method(self) -> list[str]:
        return list(SUPPORTED_METHODS)

    def merge_lora(
        self,
        base_model: Union[str, object],
        output_path: str,
        adapter_path: Optional[str] = None,
        tokenizer: Optional[object] = None,
        torch_dtype: str = "auto",
    ) -> str:
        """
        Merge a LoRA adapter into a base model.

        Args:
            base_model: Base model path/HF ID OR already loaded model object
            output_path: Directory where the merged model will be saved
            adapter_path: Path to LoRA adapter checkpoint. If None, base_model must be PeftModel
            tokenizer: Optional tokenizer to save alongside model
            torch_dtype: torch dtype (auto, float16, bfloat16, float32)

        Returns:
            Absolute path to the merged model directory

        Example:
            # Merge adapter checkpoint
            >>> merger = PEFTMerger()
            >>> merger.merge_lora(
            ...     base_model="gpt2",
            ...     adapter_path="./lora_checkpoint",
            ...     output_path="./merged"
            ... )

            # Merge already loaded model
            >>> merger.merge_lora(
            ...     base_model=trained_model,
            ...     output_path="./merged",
            ...     tokenizer=tokenizer
            ... )
        """
        peft = _require_peft()
        transformers = _require_transformers()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Handle base_model (string path or loaded model)
        if isinstance(base_model, str):
            base_model_path = base_model
            logger.info(f"Loading base model: {base_model_path}")

            import torch
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype_arg = dtype_map.get(torch_dtype, "auto")

            base_model_obj = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype_arg,
            )

            # Load adapter if provided
            if adapter_path is not None:
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                model = peft.PeftModel.from_pretrained(base_model_obj, adapter_path)
            else:
                # Treat base_model_path as PEFT model
                logger.info(f"Loading {base_model_path} as PEFT model")
                model = peft.PeftModel.from_pretrained(base_model_obj, base_model_path)

            # Load tokenizer if not provided
            if tokenizer is None:
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")

        else:
            # Already loaded model
            model = base_model
            base_model_path = None

            if adapter_path is not None:
                logger.info(f"Loading adapter from: {adapter_path}")
                model = peft.PeftModel.from_pretrained(model, adapter_path)

        # Check if model is PEFT model
        if not isinstance(model, peft.PeftModel):
            logger.warning("Model is not a PEFT model. Saving without merging.")
            model.save_pretrained(str(output_path))
            if tokenizer:
                tokenizer.save_pretrained(str(output_path))
            return str(output_path.resolve())

        # Keep the original Hub id on the merged config so Hub cards can
        # advertise `base_model: org/name` instead of the local save path.
        hub_base = None
        peft_cfg = getattr(model, "peft_config", None)
        adapter_cfg = None
        if isinstance(peft_cfg, dict):
            adapter_cfg = peft_cfg.get("default") or next(iter(peft_cfg.values()), None)
        else:
            adapter_cfg = peft_cfg
        if adapter_cfg is not None:
            hub_base = getattr(adapter_cfg, "base_model_name_or_path", None)
        if isinstance(base_model, str) and "/" in base_model and not str(base_model).startswith(("/", ".")):
            hub_base = hub_base or base_model

        # Merge adapters
        logger.info("Merging adapter weights (merge_and_unload)...")
        merged_model = model.merge_and_unload()
        if hub_base:
            merged_model.config._name_or_path = str(hub_base)

        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(str(output_path))

        if tokenizer:
            tokenizer.save_pretrained(str(output_path))
            logger.info("Tokenizer saved")

        logger.info(f"✓ LoRA merge complete: {output_path}")
        return str(output_path.resolve())

    def merge(
        self,
        models: list[str],
        output_path: str,
        adapter_path: Optional[str] = None,
        method: str = "lora-merge",
        torch_dtype: str = "auto",
        **kwargs,
    ) -> str:
        """
        Merge a LoRA adapter into a base model (legacy interface).

        Args:
            models: Single-element list containing the base model path or HF ID
            output_path: Directory where the merged model will be saved
            adapter_path: Path to the LoRA adapter directory
            method: Must be "lora-merge"
            torch_dtype: torch dtype (auto, float16, bfloat16, float32)

        Returns:
            Absolute path to the merged model directory
        """
        if method != "lora-merge":
            raise ValueError(
                f"PEFTMerger only supports 'lora-merge', got '{method}'. "
                "Use MergekitMerger for SLERP / TIES / DARE-TIES / linear."
            )

        self.validate_models(models)

        if not models:
            raise ValueError("'models' must contain at least one base model path.")

        return self.merge_lora(
            base_model=models[0],
            output_path=output_path,
            adapter_path=adapter_path,
            torch_dtype=torch_dtype,
        )
