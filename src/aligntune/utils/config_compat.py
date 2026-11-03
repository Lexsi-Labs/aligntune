"""
Make saved model configs readable by transformers 4.x loaders.

transformers 5 moved some ``config.json`` fields: ``rope_theta`` and
``rope_scaling`` now live only inside ``rope_parameters``, and
``torch_dtype`` is written as ``dtype``. Loaders still on transformers 4.x
(e.g. SGLang / vLLM builds pinned to 4.57) don't read the new keys and
silently fall back to defaults (``rope_theta=10000``), which loads fine but
badly degrades generation. See issue #38.

The helpers here only *add* the legacy top-level keys next to the new ones,
with identical values, so transformers 5 reads the config exactly as before.
The tokenizer config gets the same treatment: transformers 5's list-valued
``extra_special_tokens`` crashes 4.x, so it is written as
``additional_special_tokens`` instead, and its ``TokenizersBackend`` class
name as ``PreTrainedTokenizerFast``.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"


def _legacy_rope_fields(rope_parameters: dict) -> dict:
    """Return the transformers 4.x top-level fields equivalent to ``rope_parameters``."""
    # Models with per-layer-type RoPE (e.g. {"full_attention": {...},
    # "sliding_attention": {...}}) store one dict per layer type; the global
    # (full attention) entry is what 4.x read from the top-level fields.
    if rope_parameters and all(isinstance(v, dict) for v in rope_parameters.values()):
        rope_parameters = rope_parameters.get("full_attention")
        if not isinstance(rope_parameters, dict):
            return {}

    fields = {}
    if "rope_theta" in rope_parameters:
        fields["rope_theta"] = rope_parameters["rope_theta"]
    if "partial_rotary_factor" in rope_parameters:
        fields["partial_rotary_factor"] = rope_parameters["partial_rotary_factor"]

    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
    if rope_type != "default":
        fields["rope_scaling"] = {
            k: v for k, v in rope_parameters.items() if k not in ("rope_theta", "partial_rotary_factor")
        }
    return fields


def _add_legacy_fields(cfg: dict) -> list:
    """Add missing legacy keys to ``cfg`` (and nested sub-configs) in place; return what was added."""
    added = []

    rope_parameters = cfg.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        for key, value in _legacy_rope_fields(rope_parameters).items():
            if cfg.get(key) is None:
                cfg[key] = value
                added.append(key)

    if "dtype" in cfg and cfg.get("torch_dtype") is None and cfg["dtype"] is not None:
        cfg["torch_dtype"] = cfg["dtype"]
        added.append("torch_dtype")

    # Multimodal / composite models nest the language model config.
    for key, value in cfg.items():
        if key.endswith("_config") and isinstance(value, dict):
            added.extend(f"{key}.{k}" for k in _add_legacy_fields(value))
    return added


def add_legacy_config_fields(model_dir: Union[str, Path]) -> list:
    """
    Rewrite ``model_dir/config.json`` so transformers 4.x loaders read it correctly.

    Adds top-level ``rope_theta`` / ``rope_scaling`` / ``partial_rotary_factor``
    from ``rope_parameters`` and ``torch_dtype`` from ``dtype`` when missing.
    Existing keys are never overwritten.

    Returns:
        The list of keys that were added (empty if nothing changed).
    """
    path = Path(model_dir) / CONFIG_NAME
    if not path.is_file():
        return []
    cfg = json.loads(path.read_text())
    added = _add_legacy_fields(cfg)
    if added:
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        logger.info(f"Added transformers 4.x compatible config fields: {', '.join(added)}")
    return added


def add_legacy_tokenizer_fields(model_dir: Union[str, Path]) -> bool:
    """
    Rewrite ``model_dir/tokenizer_config.json`` so transformers 4.x can load it.

    transformers 5 saves the extra special tokens as a list under
    ``extra_special_tokens``; 4.x expects that key to be a dict (model-specific
    named tokens) and fails with ``'list' object has no attribute 'keys'``.
    The list is written back under ``additional_special_tokens``, the name both
    versions read, so transformers 5 loads the same tokenizer.

    Tokenizers without a model-specific class (e.g. Llama 3) are saved with
    ``tokenizer_class: "TokenizersBackend"``, which 4.x doesn't know; it is
    written as ``PreTrainedTokenizerFast``, the equivalent class in both versions.

    Returns:
        True if the file was changed.
    """
    path = Path(model_dir) / TOKENIZER_CONFIG_NAME
    if not path.is_file():
        return False
    cfg = json.loads(path.read_text())
    changed = []

    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        additional = list(cfg.get("additional_special_tokens") or [])
        additional += [tok for tok in extra if tok not in additional]
        cfg["additional_special_tokens"] = additional
        del cfg["extra_special_tokens"]
        changed.append("additional_special_tokens")

    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed.append("tokenizer_class")

    if not changed:
        return False
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"Wrote transformers 4.x compatible {', '.join(changed)} to tokenizer_config.json")
    return True


def _resolve_base_config(base_model: str) -> Optional[Path]:
    local = Path(base_model) / CONFIG_NAME
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(base_model, CONFIG_NAME))
    except Exception as e:
        logger.warning(f"Could not fetch {CONFIG_NAME} for base model {base_model}: {e}")
        return None


def restore_base_config(model_dir: Union[str, Path], base_model: str) -> bool:
    """
    Replace ``model_dir/config.json`` with the base model's original config.

    LoRA merging does not change the architecture, so the base config is
    correct for the merged weights. Fields the merge can change are carried
    over from the merged config: ``vocab_size`` (embeddings grown for added
    tokens) and the saved dtype (``torch_dtype`` / ``dtype``).

    Returns:
        True if the base config was written, False if it could not be found
        (the merged config is then left as is).
    """
    base_path = _resolve_base_config(base_model)
    if base_path is None:
        return False

    out_path = Path(model_dir) / CONFIG_NAME
    merged = json.loads(out_path.read_text()) if out_path.is_file() else {}
    cfg = json.loads(base_path.read_text())

    if merged.get("vocab_size") is not None:
        cfg["vocab_size"] = merged["vocab_size"]
    saved_dtype = merged.get("dtype") or merged.get("torch_dtype")
    if saved_dtype is not None:
        cfg["torch_dtype"] = saved_dtype
        if "dtype" in cfg:
            cfg["dtype"] = saved_dtype

    out_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote base model config from {base_model} to {out_path}")
    return True
