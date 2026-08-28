"""
LoRA adapter implementation.
"""
import logging
from typing import Any, List

from .embedding_utils import resolve_embedding_module_names

logger = logging.getLogger(__name__)


def get_peft_value(config: Any, *names: str, default: Any = None) -> Any:
    """Resolve a PEFT option from explicit factory kwargs or config fields."""
    extra_params = getattr(getattr(config, "train", None), "extra_params", {}) or {}
    peft_config = getattr(getattr(config, "model", None), "peft", None)
    model_config = getattr(config, "model", None)
    for name in names:
        if name in extra_params:
            return extra_params[name]
        if peft_config is not None and hasattr(peft_config, name):
            value = getattr(peft_config, name)
            if value is not None:
                return value
        if model_config is not None and hasattr(model_config, name):
            value = getattr(model_config, name)
            if value is not None:
                return value
    return default


class LoraAdapter:
    """Standard LoRA adapter with support for rsLoRA and initializations.

    Previously subclassed a PEFTAdapterBase ABC (core/peft/base.py, now
    merged in here). MoA/Text2LoRA/Doc2LoRA are the only other adapter
    variants in the codebase and are deliberately standalone tools with
    their own APIs (see docs/advanced/adapters.md) rather than
    PEFTAdapterBase subclasses, so the base class had exactly one
    implementation by design, not as a placeholder for more.
    """

    def __init__(self, config: Any):
        self.config = config
        self.target_modules = self._resolve_target_modules()

    def _resolve_target_modules(self) -> List[str]:
        if hasattr(self.config.model, 'peft') and self.config.model.peft is not None:
            if getattr(self.config.model.peft, 'target_modules', None) is not None:
                return self.config.model.peft.target_modules

        target_modules = getattr(self.config.model, 'target_modules', getattr(self.config.model, 'lora_target_modules', None))
        if target_modules is None:
            # Fallback will happen dynamically based on the model if needed
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        return target_modules

    def _resolve_all_linear_target_modules(self, model: Any) -> List[str]:
        """
        Resolve the HF-PEFT "all-linear" shorthand into an explicit list of leaf
        module names (e.g. "q_proj", "gate_proj", ...) for every nn.Linear
        submodule, excluding the output/LM-head layer. Unlike HF's own
        peft.LoraConfig, target_modules here is a plain string iterated
        character-by-character if left as "all-linear" - this resolves it to a
        real list before that happens, and works for both the transformers and
        Unsloth code paths (Unsloth does not understand the "all-linear" string
        either).
        """
        import torch.nn as nn

        output_module = None
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            output_module = get_output_embeddings()

        leaf_names = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if output_module is not None and module is output_module:
                continue
            leaf_names.add(name.rsplit(".", 1)[-1])

        return sorted(leaf_names)

    def _auto_detect_target_modules(self, available_modules: List[str], model_type: str) -> List[str]:
        patterns = [
            ("c_attn", "c_proj"),  # GPT-2 style
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),  # LLaMA style (full)
            ("q_proj", "k_proj", "v_proj", "o_proj"),  # LLaMA style (base)
            ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),  # Falcon/Bloom style
        ]

        for pattern in patterns:
            if all(any(m in name for name in available_modules) for m in pattern):
                return list(pattern)

        # Architecture-based fallback
        model_type = model_type.lower()
        if any(x in model_type for x in ["qwen", "llama", "mistral", "gemma"]):
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "phi" in model_type:
            return ["q_proj", "k_proj", "v_proj", "dense"]
        elif any(x in model_type for x in ["gpt2", "gpt", "dialogpt"]):
            return ["c_attn", "c_proj", "c_fc"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _resolve_lora_hyperparams(
        self,
        default_r: int = 16,
        default_alpha: int = 32,
        default_dropout: float = 0.05,
    ) -> "tuple[int, int, float]":
        """
        Resolve (r, alpha, dropout), preferring the nested config.model.peft.*
        (the PeftConfigData object create_sft_trainer()/create_rl_trainer()
        actually populate from lora_r=/lora_alpha=/lora_dropout=) over flat
        config.model.lora_r/lora_alpha/lora_dropout attributes, which most
        config types never set at all.

        Mirrors the resolution lora.py's LoraAdapter already does correctly;
        subclasses that read the flat attributes only (skipping .peft
        entirely) silently ignore whatever the caller actually configured
        and always fall through to this method's hardcoded defaults instead.
        """
        return (
            get_peft_value(self.config, 'lora_r', 'lora_rank', 'rank', default=default_r),
            get_peft_value(self.config, 'lora_alpha', 'alpha', default=default_alpha),
            get_peft_value(self.config, 'lora_dropout', 'dropout', default=default_dropout),
        )

    def _get_loftq_config(self) -> Any:
        if get_peft_value(self.config, 'loftq_init', default=False):
            try:
                from peft import LoftQConfig
                return LoftQConfig(loftq_bits=4)
            except ImportError:
                logger.warning("PEFT LoftQConfig not available. Ignoring loftq_init.")
        return None

    def apply_to_transformers(self, model: Any, task_type: str = "CAUSAL_LM") -> Any:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("PEFT required for LoRA. Install with: pip install peft") from e

        # Extract config values
        lora_r = get_peft_value(self.config, 'lora_r', 'lora_rank', 'rank', default=16)
        lora_alpha = get_peft_value(self.config, 'lora_alpha', 'alpha', default=32)
        lora_dropout = get_peft_value(self.config, 'lora_dropout', 'dropout', default=0.05)
        bias = get_peft_value(self.config, 'bias', default='none')
        if get_peft_value(self.config, 'use_dora', 'dora_enabled', default=False):
            logger.warning("DoRA is not available in this build. Falling back to standard LoRA.")
        use_dora = False
        use_rslora = get_peft_value(self.config, 'use_rslora', 'rslora_enabled', 'rslora', default=False)
        init_weights = get_peft_value(self.config, 'init_lora_weights', 'init_weights', default=True)

        # Check PiSSA
        if get_peft_value(self.config, 'pissa_init', default=False):
            init_weights = "pissa"

        # Auto-detect target modules if needed
        available_modules = [name for name, _ in model.named_modules()]
        model_type = getattr(model.config, "model_type", "unknown")

        if isinstance(self.target_modules, str) and self.target_modules.lower() == "all-linear":
            # Resolve the HF-PEFT shorthand ourselves instead of iterating the raw
            # string character-by-character below.
            target_modules = self._resolve_all_linear_target_modules(model)
        else:
            target_modules = self.target_modules

        valid_modules = [m for m in target_modules if any(m in name for name in available_modules)]
        if not valid_modules:
            valid_modules = self._auto_detect_target_modules(available_modules, model_type)

        if not valid_modules:
            logger.warning(
                f"Target modules {self.target_modules} not found. "
                f"Attempting to proceed with auto-detected: {valid_modules}"
            )

        modules_to_save = None
        if getattr(self.config.model, 'train_embeddings', False):
            modules_to_save = resolve_embedding_module_names(model)
            logger.info(
                "Training full embedding modules alongside LoRA: %s",
                modules_to_save,
            )

        logger.info(f"Building LoRA config: r={lora_r}, alpha={lora_alpha}, targets={valid_modules}")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=valid_modules,
            modules_to_save=modules_to_save,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
            use_dora=use_dora,
            use_rslora=use_rslora,
            init_lora_weights=init_weights,
        )

        if use_dora:
            logger.info("DoRA (Weight-Decomposed LoRA) enabled")
        if use_rslora:
            logger.info("rsLoRA (Rank-Stabilized LoRA) enabled")
        if init_weights in ["pissa", "pissa_niter_4", "pissa_niter_16"]:
            logger.info(f"PiSSA initialization enabled: {init_weights}")

        peft_model = get_peft_model(model, config)
        peft_model.print_trainable_parameters()
        return peft_model

    def apply_to_unsloth(self, model: Any, model_type: str, available_modules: List[str]) -> Any:
        from unsloth import FastLanguageModel

        if isinstance(self.target_modules, str) and self.target_modules.lower() == "all-linear":
            # Resolve the HF-PEFT shorthand ourselves - Unsloth's own get_peft_model
            # does not understand it either and would iterate the raw string too.
            target_modules = self._resolve_all_linear_target_modules(model)
        else:
            target_modules = self.target_modules

        valid_modules = [m for m in target_modules if any(m in name for name in available_modules)]
        if not valid_modules:
            valid_modules = self._auto_detect_target_modules(available_modules, model_type)

        if not valid_modules:
            raise ValueError(
                f"Target modules {target_modules} not found in the base model (type: {model_type}). "
                f"Available modules include: {available_modules[:10]}... "
            )

        logger.info(f"Using target modules for {model_type}: {valid_modules}")

        # Some config types (e.g. UnifiedESConfig) nest LoRA hyperparameters
        # under config.model.peft.* instead of flat config.model.lora_* -
        # apply_to_transformers() above already handles this; mirror it here.
        # Getting this wrong isn't just a wrong-default issue for ES: its
        # vLLM rollout backend is separately configured with
        # max_lora_rank=config.model.peft.rank, so an adapter built here at
        # the flat-attribute fallback (16) instead of the real configured
        # rank crashes vLLM with "LoRA rank 16 is greater than
        # max_lora_rank 8".
        lora_r = get_peft_value(self.config, 'lora_r', 'lora_rank', 'rank', default=16)
        lora_alpha = get_peft_value(self.config, 'lora_alpha', 'alpha', default=16)
        lora_dropout = get_peft_value(self.config, 'lora_dropout', 'dropout', default=0.0)
        # Unsloth's fast-patching path only fully optimizes bias="none" (a
        # non-"none" value still works - Unsloth just logs a one-time warning
        # and falls back to patching every other layer, at a performance
        # cost) - still forwarded rather than hardcoded so the config actually
        # takes effect, matching apply_to_transformers() above.
        bias = get_peft_value(self.config, 'bias', default='none')
        if get_peft_value(self.config, 'use_dora', 'dora_enabled', default=False):
            logger.warning("DoRA is not available in this build. Falling back to standard LoRA.")
        use_dora = False

        # Not every config type nests a `.train` section (e.g. UnifiedESConfig
        # has `.es` instead) - fall back to the same defaults `getattr` below
        # would have applied if `.train` existed but lacked these fields.
        train_config = getattr(self.config, 'train', None)

        unsloth_model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=valid_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=getattr(train_config, 'gradient_checkpointing', "unsloth"),
            random_state=getattr(train_config, 'seed', 3407),
            use_rslora=get_peft_value(self.config, 'use_rslora', 'rslora_enabled', 'rslora', default=False),
            use_dora=use_dora,
            loftq_config=self._get_loftq_config(),
        )
        return unsloth_model
