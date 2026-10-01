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

from aligntune.utils.provenance import build_provenance, provenance_input, write_provenance

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


def _adapter_vocab_size(adapter_path: str, tokenizer: Optional[object]) -> Optional[int]:
    """Vocabulary size an adapter was trained with, or None if it cannot be determined.

    Adapters trained after a vocabulary extension carry resized embedding
    weights (``modules_to_save``); their row count is authoritative. Falls back
    to the adapter directory's tokenizer length.
    """
    adapter_dir = Path(adapter_path)
    weights_file = adapter_dir / "adapter_model.safetensors"
    if weights_file.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(weights_file), framework="pt") as f:
                for key in f.keys():
                    if "embed_tokens" in key or "wte" in key:
                        shape = f.get_slice(key).get_shape()
                        if len(shape) == 2 and "lora_" not in key:
                            return int(shape[0])
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(f"Could not inspect adapter embeddings: {exc}")
    if tokenizer is not None and (adapter_dir / "tokenizer_config.json").exists():
        return len(tokenizer)
    return None


def _grow_embeddings_for_adapter(model, adapter_path: str, tokenizer: Optional[object]) -> None:
    """Resize the base model's embeddings up to the adapter's vocabulary size."""
    target = _adapter_vocab_size(adapter_path, tokenizer)
    if target is None:
        return
    current = model.get_input_embeddings().weight.shape[0]
    if target > current:
        logger.info(f"Resizing base embeddings {current} -> {target} to match adapter vocabulary")
        model.resize_token_embeddings(target)


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
        provenance: Optional[dict] = None,
    ) -> str:
        """
        Merge a LoRA adapter into a base model.

        Args:
            base_model: Base model path/HF ID OR already loaded model object
            output_path: Directory where the merged model will be saved
            adapter_path: Path to LoRA adapter checkpoint. If None, base_model must be PeftModel
            tokenizer: Optional tokenizer to save alongside model
            torch_dtype: torch dtype (auto, float16, bfloat16, float32)
            provenance: ``lexsi_provenance.json`` object for the merged dir.
                Defaults to one whose input is the adapter (and its own
                provenance, when the adapter dir has one).

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

            # Prefer the adapter directory's tokenizer: it carries the chat
            # template and any added tokens used during training.
            if tokenizer is None:
                tokenizer_sources = []
                if adapter_path is not None and any(
                    (Path(adapter_path) / n).exists() for n in ("tokenizer.json", "tokenizer_config.json")
                ):
                    tokenizer_sources.append(adapter_path)
                tokenizer_sources.append(base_model_path)
                for source in tokenizer_sources:
                    try:
                        tokenizer = transformers.AutoTokenizer.from_pretrained(source)
                        break
                    except Exception as e:
                        logger.warning(f"Could not load tokenizer from {source}: {e}")

            # Load adapter if provided
            if adapter_path is not None:
                _grow_embeddings_for_adapter(base_model_obj, adapter_path, tokenizer)
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                model = peft.PeftModel.from_pretrained(base_model_obj, adapter_path)
            else:
                # Treat base_model_path as PEFT model
                logger.info(f"Loading {base_model_path} as PEFT model")
                model = peft.PeftModel.from_pretrained(base_model_obj, base_model_path)

            if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        else:
            # Already loaded model
            model = base_model
            base_model_path = None

            if adapter_path is not None:
                logger.info(f"Loading adapter from: {adapter_path}")
                model = peft.PeftModel.from_pretrained(model, adapter_path)

        if provenance is None:
            adapter_ref = adapter_path or base_model_path
            provenance = build_provenance(
                "merge_lora",
                base_model=base_model_path,
                inputs=[provenance_input(adapter_ref, "adapter")] if adapter_ref else [],
            )

        # Check if model is PEFT model
        if not isinstance(model, peft.PeftModel):
            logger.warning("Model is not a PEFT model. Saving without merging.")
            model.save_pretrained(str(output_path))
            if tokenizer:
                tokenizer.save_pretrained(str(output_path))
            write_provenance(output_path, provenance)
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
        write_provenance(output_path, provenance)

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
