"""
Centralized SFT Model Loader Utility.

This module handles base model initialization, quantization setup,
tokenizer loading, and precision handling for all SFT trainers.
"""

import logging
import torch
from typing import Tuple, Any, Optional

from .sft.config import SFTConfig
from .registry import TaskType
from .precision_handler import PrecisionHandler
from .long_context.rope_config import extract_rope_parameters

logger = logging.getLogger(__name__)


def setup_tokenizer(tokenizer: Any, config: SFTConfig, task_type: TaskType) -> Any:
    """Setup tokenizer with task-specific configurations."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad token to eos token")

    # Handle both SFTConfig.dataset and UnifiedConfig.datasets (RL configs)
    dataset_config = getattr(config, 'dataset', None) or getattr(config, 'datasets', None)
    cfg_template = getattr(dataset_config, "chat_template", None) if dataset_config else None
    
    # Try applying custom template from config
    if cfg_template and isinstance(cfg_template, str):
        # Jinja templates
        if ("{%" in cfg_template or "{{" in cfg_template) and (
            not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None
        ):
            try:
                tokenizer.chat_template = cfg_template
                logger.info("Applied custom chat_template from config to tokenizer")
            except Exception as e:
                logger.debug(f"Failed to set tokenizer.chat_template from config: {e}")
        # Unsloth named templates (e.g., 'llama-3')
        elif not ("{%" in cfg_template or "{{" in cfg_template):
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template=cfg_template,
                    mapping=None,
                )
                logger.info(f"Applied Unsloth chat template: {cfg_template}")
            except Exception as e:
                logger.debug(f"Failed to apply named Unsloth chat template: {e}")

    # Fallbacks for SFT tasks
    if task_type == TaskType.SFT:
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}Assistant:"
            )
            logger.info("Added basic manual chat template to tokenizer")

        has_chat_template = (
            hasattr(tokenizer, "apply_chat_template") and
            hasattr(tokenizer, "chat_template") and
            tokenizer.chat_template is not None
        )
        if not has_chat_template:
            special_tokens = ["<|im_start|>", "<|im_end|>"]
            num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens for SFT")

    return tokenizer


def build_model(
    config: Any, 
    task_type: TaskType,
    use_unsloth: bool = False,
    apply_peft: bool = False,
    is_reference: bool = False
) -> Tuple[Any, Any]:
    """
    Builds the model and tokenizer based on configuration.
    Handles quantization, precision, and task-specific loading.
    """
    logger.info("=" * 80)
    backend_str = "Unsloth" if use_unsloth else "TRL"
    logger.info(f"Setting up {backend_str} SFT model: {config.model.name_or_path}")
    logger.info(f"Task Type: {task_type.value}")
    logger.info("=" * 80)
    
    precision = PrecisionHandler.get_precision_from_config(config, default="auto")
    precision = PrecisionHandler.validate_precision(precision)
    PrecisionHandler.log_precision_info(precision, f"{backend_str} SFT")
    dtype = PrecisionHandler.get_torch_dtype(precision)
    
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            
            model_kwargs = {
                "max_seq_length": config.model.max_seq_length,
                "dtype": dtype,
                "load_in_4bit": config.model.quantization.get("load_in_4bit", False) if config.model.quantization else False,
            }
            
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.model.name_or_path,
                    **model_kwargs
                )
            except (SyntaxError, RuntimeError) as e:
                if "unsloth_compiled_module" in str(e) or "unexpected indent" in str(e).lower():
                    raise RuntimeError(
                        f"Unsloth compilation error: {e}\n"
                        f"Please clear the Unsloth cache (rm -rf ~/.cache/unsloth/) and retry."
                    ) from e
                raise
            except (AssertionError, ValueError, NotImplementedError) as e:
                error_str = str(e).lower()
                if "not supported" in error_str or "unsloth does not support" in error_str or "architecture" in error_str:
                    logger.warning(
                        f"⚠️ Unsloth does not support model architecture '{config.model.name_or_path}'. "
                        f"Reason: {e}. "
                        "Falling back to standard Hugging Face TRL model loading."
                    )
                    # Gracefully fall back to TRL backend
                    return build_model(
                        config,
                        task_type,
                        use_unsloth=False,
                        apply_peft=apply_peft,
                        is_reference=is_reference,
                    )
                raise

            # Unlike the standard/TRL branch below, FastLanguageModel.from_pretrained()
            # always returns the tokenizer bundled with `name_or_path`, ignoring
            # `tokenizer_name_or_path` - swap in the configured tokenizer here
            # (e.g. a vocab-extended one) before resizing/adapting embeddings,
            # same as the standard branch does.
            tokenizer_name_or_path = getattr(config.model, 'tokenizer_name_or_path', None)
            if tokenizer_name_or_path and tokenizer_name_or_path != config.model.name_or_path:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name_or_path,
                    trust_remote_code=getattr(config.model, "trust_remote_code", True),
                )

            embedding_init_method = getattr(config.model, "embedding_init_method", None)
            if embedding_init_method is not None:
                from transformers import AutoTokenizer
                from .tokenization.embedding_adaptation import adapt_token_embeddings

                base_tokenizer = AutoTokenizer.from_pretrained(
                    config.model.name_or_path,
                    trust_remote_code=getattr(config.model, "trust_remote_code", True),
                )
                adaptation_report = adapt_token_embeddings(
                    model=model,
                    old_tokenizer=base_tokenizer,
                    new_tokenizer=tokenizer,
                    method=embedding_init_method,
                    pad_to_multiple_of=getattr(config.model, "embedding_pad_to_multiple_of", None),
                )
                logger.info("Tokenizer embedding adaptation complete: %s", adaptation_report)

            # Apply LoRA only when requested. Reference models must remain
            # adapter-free, and full-finetuning callers should not receive an
            # implicit adapter.
            if apply_peft and not is_reference:
                model_type = model.config.model_type.lower()
                from .peft.factory import PEFTFactory
                adapter = PEFTFactory.get_adapter(config)
                available_modules = [name for name, _ in model.named_modules()]
                model = adapter.apply_to_unsloth(
                    model=model,
                    model_type=model_type,
                    available_modules=available_modules,
                )

            # Preserve max_seq_length for trainers that need it (preference-based algorithms)
            if not hasattr(model, 'max_seq_length'):
                model.max_seq_length = config.model.max_seq_length
            if hasattr(model, 'model') and not hasattr(model.model, 'max_seq_length'):
                model.model.max_seq_length = config.model.max_seq_length

            tokenizer = setup_tokenizer(tokenizer, config, task_type)
            return model, tokenizer
            
        except ImportError as e:
            raise ImportError("Unsloth not available.") from e

    else:
        # TRL (Standard Transformers)
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer
            )
        except ImportError as e:
            raise ImportError("Transformers not available.") from e

        # Load tokenizer (use tokenizer_name_or_path if provided, else model path)
        tokenizer_path = getattr(config.model, 'tokenizer_name_or_path', None) or config.model.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        tokenizer = setup_tokenizer(tokenizer, config, task_type)

        if task_type == TaskType.TEXT_CLASSIFICATION:
            num_labels = getattr(config.model, 'num_labels', 2)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model.name_or_path,
                num_labels=num_labels,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            
        elif task_type == TaskType.TOKEN_CLASSIFICATION:
            num_labels = getattr(config.model, 'num_labels', 9)
            model = AutoModelForTokenClassification.from_pretrained(
                config.model.name_or_path,
                num_labels=num_labels,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            
        else:
            # Generation tasks
            # Load model config first to check RoPE support
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                config.model.name_or_path,
                trust_remote_code=getattr(config.model, 'trust_remote_code', True)
            )

            quantization_config = None
            quantization_dict = getattr(config.model, 'quantization', {})

            if quantization_dict.get('load_in_4bit', False) or quantization_dict.get('load_in_8bit', False):
                try:
                    from transformers import BitsAndBytesConfig
                    load_in_4bit = quantization_dict.get('load_in_4bit', False)
                    load_in_8bit = quantization_dict.get('load_in_8bit', False)

                    if load_in_4bit:
                        compute_dtype_str = quantization_dict.get('bnb_4bit_compute_dtype', 'bfloat16')
                        compute_dtype = torch.bfloat16 if compute_dtype_str == 'bfloat16' else torch.float16
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_quant_type=quantization_dict.get('bnb_4bit_quant_type', 'nf4'),
                            bnb_4bit_use_double_quant=quantization_dict.get('bnb_4bit_use_double_quant', False),
                        )
                    elif load_in_8bit:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                except ImportError:
                    logger.warning("BitsAndBytes not available.")

            # Check for DDP/Accelerate device mapping
            device_map = getattr(config.model, 'device_map', 'auto')
            import os
            is_ddp = device_map in ["DDP", "ddp"] or os.environ.get("ACCELERATE_USE_DDP") == "true" or os.environ.get("ACCELERATE_USE_FSDP") == "true"

            if is_ddp:
                try:
                    from accelerate import PartialState
                    device_string = PartialState().process_index
                    device_map = {"": device_string}
                    logger.info(f"DDP/FSDP mode detected: setting device_map={{'': {device_string}}}")
                except ImportError:
                    logger.warning("Accelerate not available for DDP device mapping, falling back to 'auto'")
                    device_map = "auto"
            elif device_map == "auto" and not torch.cuda.is_available():
                device_map = None

            model_kwargs = {
                'torch_dtype': dtype,
                'trust_remote_code': getattr(config.model, 'trust_remote_code', True),
                'device_map': device_map,
                'low_cpu_mem_usage': True,
            }

            max_memory = getattr(config.model, 'max_memory', None)
            if max_memory:
                model_kwargs['max_memory'] = max_memory

            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            # NOTE: intentionally NOT passing load_in_4bit=False/load_in_8bit=False
            # here when no quantization is requested - newer `transformers`
            # (5.x) removed those as direct from_pretrained()/model __init__
            # kwargs (superseded by quantization_config), so passing them
            # explicitly (even as False) raises
            # `TypeError: ...__init__() got an unexpected keyword argument
            # 'load_in_4bit'`. Omitting them is a no-op for the non-quantized
            # path since these both default to False anyway.

            # Set attention implementation if specified
            attn_implementation = getattr(config.model, 'attn_implementation', 'auto')
            if attn_implementation and attn_implementation != 'auto':
                # Register custom attention implementations before loading model
                if attn_implementation == 'swa':
                    from .long_context.custom_attention import (
                        register_sliding_window_attention,
                        AttentionRegistry
                    )
                    if not AttentionRegistry.is_registered('swa'):
                        register_sliding_window_attention()
                        logger.info("Registered SWA attention implementation")
                elif attn_implementation == 's2':
                    from .long_context.custom_attention import (
                        register_s2_attention,
                        AttentionRegistry
                    )
                    if not AttentionRegistry.is_registered('s2'):
                        register_s2_attention()
                        logger.info("Registered S2 attention implementation")

                model_kwargs['attn_implementation'] = attn_implementation
                logger.info(f"Using attention implementation: {attn_implementation}")

            # Extract RoPE parameters if configured
            rope_config = getattr(config.model, 'rope', None)
            rope_params = None
            if rope_config:
                try:
                    rope_params = extract_rope_parameters(
                        rope_config=rope_config,
                        model_config=model_config,
                        max_seq_length=config.model.max_seq_length
                    )
                    if rope_params:
                        model_kwargs['rope_scaling'] = rope_params
                        logger.info(f"Applied RoPE scaling configuration: {rope_params.get('rope_type')}")
                except Exception as e:
                    logger.warning(f"Failed to configure RoPE scaling: {e}. Continuing without RoPE scaling.")
                    rope_params = None

            # Try to load model with RoPE, fall back without it if it fails
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model.name_or_path,
                    **model_kwargs
                )
            except Exception as e:
                if rope_params and 'rope_scaling' in model_kwargs:
                    logger.warning(
                        f"Failed to load model with RoPE scaling: {e}. "
                        "Retrying without RoPE scaling..."
                    )
                    # Remove rope_scaling and retry
                    model_kwargs.pop('rope_scaling', None)
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model.name_or_path,
                        **model_kwargs
                    )
                else:
                    # Not a RoPE issue, re-raise
                    raise
            
            if not quantization_config and torch.cuda.is_available() and model_kwargs.get('device_map') != 'auto':
                model = model.cuda()
                
            # Reference models are quantized for inference only. They must remain
            # adapter-free and frozen, even when the trainable model uses QLoRA.
            # `apply_peft` is the source of truth because wrapper configs do not
            # always expose a `use_peft` attribute.
            if quantization_config and not is_reference and not apply_peft:
                logger.warning(
                    "Quantized training requires PEFT adapters or embedding-only "
                    "training. Set use_peft=True (QLoRA) or train_embeddings=True."
                )

        embedding_init_method = getattr(
            config.model, "embedding_init_method", None
        )
        if embedding_init_method is not None:
            from .tokenization.embedding_adaptation import (
                adapt_token_embeddings,
            )

            base_tokenizer = AutoTokenizer.from_pretrained(
                config.model.name_or_path,
                trust_remote_code=getattr(
                    config.model, "trust_remote_code", True
                ),
            )
            adaptation_report = adapt_token_embeddings(
                model=model,
                old_tokenizer=base_tokenizer,
                new_tokenizer=tokenizer,
                method=embedding_init_method,
                pad_to_multiple_of=getattr(
                    config.model,
                    "embedding_pad_to_multiple_of",
                    None,
                ),
            )
            logger.info(
                "Tokenizer embedding adaptation complete: %s",
                adaptation_report,
            )

        # Set max_seq_length for trainers that need it (preference-based algorithms)
        # This must be set BEFORE PEFT to ensure it persists through PEFT wrapping
        if not hasattr(model, 'max_seq_length'):
            model.max_seq_length = config.model.max_seq_length
        if hasattr(model, 'model') and not hasattr(model.model, 'max_seq_length'):
            model.model.max_seq_length = config.model.max_seq_length

        # Set sliding window if configured (for SWA attention)
        sliding_window = getattr(config.model, 'sliding_window', None)
        if sliding_window:
            model.config.sliding_window = sliding_window
            logger.info(f"Set model.config.sliding_window = {sliding_window}")

        # Set S2 attention parameters if configured
        s2_group_ratio = getattr(config.model, 's2_group_size_ratio', None)
        s2_min_seq = getattr(config.model, 's2_min_seq_length', None)
        s2_shift_ratio = getattr(config.model, 's2_shift_ratio', None)

        if s2_group_ratio is not None:
            model.config.s2_group_size_ratio = s2_group_ratio
            logger.info(f"Set model.config.s2_group_size_ratio = {s2_group_ratio}")
        if s2_min_seq is not None:
            model.config.s2_min_seq_length = s2_min_seq
            logger.info(f"Set model.config.s2_min_seq_length = {s2_min_seq}")
        if s2_shift_ratio is not None:
            model.config.s2_shift_ratio = s2_shift_ratio
            logger.info(f"Set model.config.s2_shift_ratio = {s2_shift_ratio}")

        # Apply PEFT if requested
        if apply_peft:
            from .peft.factory import PEFTFactory
            from peft import prepare_model_for_kbit_training

            adapter = PEFTFactory.get_adapter(config)

            # Prepare model for k-bit training if using quantization
            quantization_dict = getattr(config.model, 'quantization', {})
            if quantization_dict.get('load_in_4bit', False) or quantization_dict.get('load_in_8bit', False):
                logger.info("Preparing model for k-bit training...")
                model = prepare_model_for_kbit_training(model)

            task = "CAUSAL_LM" if task_type not in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION] else "SEQ_CLS"
            model = adapter.apply_to_transformers(model, task_type=task)

            # Re-apply max_seq_length after PEFT wrapping (in case PEFT wrapper loses it)
            if not hasattr(model, 'max_seq_length'):
                model.max_seq_length = config.model.max_seq_length

            # Re-apply attention config after PEFT wrapping
            if sliding_window:
                model.config.sliding_window = sliding_window
                logger.info(f"Re-applied model.config.sliding_window = {sliding_window} after PEFT")
            if s2_group_ratio is not None:
                model.config.s2_group_size_ratio = s2_group_ratio
                logger.info(f"Re-applied model.config.s2_group_size_ratio = {s2_group_ratio} after PEFT")
            if s2_min_seq is not None:
                model.config.s2_min_seq_length = s2_min_seq
                logger.info(f"Re-applied model.config.s2_min_seq_length = {s2_min_seq} after PEFT")
            if s2_shift_ratio is not None:
                model.config.s2_shift_ratio = s2_shift_ratio
                logger.info(f"Re-applied model.config.s2_shift_ratio = {s2_shift_ratio} after PEFT")
            if hasattr(model, 'model') and not hasattr(model.model, 'max_seq_length'):
                model.model.max_seq_length = config.model.max_seq_length

            # Enable gradient flow for frozen base model parameters (critical for PEFT/LoRA)
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
                logger.info("Enabled input_require_grads for PEFT gradient flow")

            logger.info(f"Applied PEFT natively via model_loader")

        elif getattr(config.model, 'train_embeddings', False):
            from .peft.embedding_utils import configure_embedding_only_training

            embedding_module_names = configure_embedding_only_training(model)

            logger.info(
                "Configured embedding-only training without PEFT: %s",
                embedding_module_names,
            )

        # Print trainable params info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(f"Trainable params: {trainable_params:,} / Total params: {total_params:,} ({trainable_percent:.4f}%)")

        # Ensure model is in training mode (unless it's a reference model)
        if not is_reference:
            model.train()

        # Freeze and set to eval mode if it's a reference model
        if is_reference:
            logger.info("Configuring model as frozen reference model")
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model, tokenizer

def build_ref_model(config: Any, base_model: Any = None, use_unsloth: bool = False) -> Any:
    """Build or extract reference model for RL."""
    try:
        from peft.peft_model import PeftModel
        is_peft = isinstance(base_model, PeftModel)
    except ImportError:
        is_peft = False

    # If base model is a PEFT model, reference model is implicitly the base weights
    # TRL's disable_adapter() handles this natively for PPO/DPO/GRPO
    if base_model is not None and is_peft:
        logger.info("Base model is a PEFT model. Reference model will use disable_adapter() inherently.")
        return None  
        
    # Otherwise, load a completely new frozen instance
    logger.info("Loading separate frozen reference model...")
    ref_model, _ = build_model(
        config=config,
        task_type=TaskType.CHAT_COMPLETION, # Typical for reference
        use_unsloth=use_unsloth,
        apply_peft=False, # Ref models never have PEFT
        is_reference=True
    )
    return ref_model

def _build_bnb_quantization_config(quantization_dict: dict):
    """Build a BitsAndBytesConfig from a raw quantization dict, or return None.

    Mirrors the working pattern in build_model() above. Newer `transformers`
    (5.x) removed load_in_4bit/load_in_8bit as direct from_pretrained()/model
    __init__ kwargs (superseded by quantization_config), so passing them raw -
    as build_reward_model()/build_value_model() used to - raises
    `TypeError: ...__init__() got an unexpected keyword argument 'load_in_4bit'`.
    """
    if not quantization_dict:
        return None
    load_in_4bit = quantization_dict.get('load_in_4bit', False)
    load_in_8bit = quantization_dict.get('load_in_8bit', False)
    if not (load_in_4bit or load_in_8bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("BitsAndBytes not available.")
        return None
    if load_in_4bit:
        compute_dtype_str = quantization_dict.get('bnb_4bit_compute_dtype', 'bfloat16')
        compute_dtype = torch.bfloat16 if compute_dtype_str == 'bfloat16' else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quantization_dict.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quantization_dict.get('bnb_4bit_use_double_quant', False),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def build_reward_model(config: Any) -> Any:
    """Build the reward model for RL, automatically wrapping it in UniversalRewardModelWrapper."""
    source = getattr(config.model, 'reward_model_source', None)
    if isinstance(source, dict):
        source_name = source.get('model_name') or source.get('model_path')
    else:
        source_name = (
            (getattr(source, 'model_name', None) or getattr(source, 'model_path', None))
            if source is not None else None
        )
    model_name = (
        getattr(config.model, 'reward_model_name', None)
        or getattr(config.model, 'reward_path', None)
        or source_name
        or config.model.name_or_path
    )
    logger.info(f"Loading reward model from: {model_name}")
    
    precision = PrecisionHandler.get_precision_from_config(config, default="auto")
    dtype = PrecisionHandler.get_torch_dtype(precision)
    device = getattr(config.model, 'reward_device', 'auto')
    quantization = getattr(config.model, 'reward_model_quantization', None) or {}
    
    # Check if Unsloth is requested specifically for reward model
    # NOTE: ModelConfig.reward_value_loading_type defaults to None (not
    # 'standard'), so getattr(..., 'standard') never falls back - the
    # attribute always exists, just with value None. Use `or` to actually
    # apply the 'standard' default, otherwise loading_type stays None and
    # neither the 'unsloth' nor 'standard' branch below runs, leaving
    # `reward_model` unbound.
    loading_type = getattr(config.model, 'reward_value_loading_type', None) or 'standard'
    
    if loading_type == 'unsloth':
        try:
            from unsloth import FastLanguageModel
            reward_model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=getattr(config.model, 'max_seq_length', 2048),
                dtype=dtype,
                load_in_4bit=config.model.quantization.get("load_in_4bit", False) if config.model.quantization else False,
            )
            # Add dummy score head if missing, though unsloth sequence classification is better
            if not hasattr(reward_model, 'score'):
                import torch
                reward_model.score = torch.nn.Linear(reward_model.config.hidden_size, 1, bias=False)
        except ImportError:
            logger.warning("Unsloth not available for reward model, falling back to standard TRL.")
            loading_type = 'standard'
            
    if loading_type == 'standard':
        from transformers import AutoModelForSequenceClassification
        reward_quantization_config = _build_bnb_quantization_config(quantization)
        reward_model_kwargs = {
            'num_labels': 1,
            'torch_dtype': dtype,
            'device_map': device,
            'trust_remote_code': True,
        }
        if reward_quantization_config is not None:
            reward_model_kwargs['quantization_config'] = reward_quantization_config
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **reward_model_kwargs,
        )
        
    # CRITICAL: Wrap with UniversalRewardModelWrapper for TRL compatibility
    try:
        from .rl.reward_model_wrapper import UniversalRewardModelWrapper
        reward_model = UniversalRewardModelWrapper(reward_model)
        logger.info("✅ Applied UniversalRewardModelWrapper to reward model inside model_loader")
    except ImportError:
        logger.warning("Could not import UniversalRewardModelWrapper! Reward model may crash with use_cache=False in TRL.")
        
    reward_model.eval()
    return reward_model

def build_value_model(config: Any, policy_model: Any = None) -> Any:
    """Build the value model expected by the installed TRL PPO backend.

    Current ``trl.experimental.ppo`` expects a sequence-classification-style
    value model exposing ``base_model_prefix`` and ``score``.  The older
    ``AutoModelForCausalLMWithValueHead`` wrapper does not satisfy that
    interface, so do not use it here.
    """
    value_model_name = getattr(config.model, 'reward_value_model', None)

    # An explicitly supplied value model is loaded through the same
    # sequence-classification path as a reward model.
    model_name = value_model_name or getattr(config.model, 'sft_path', None)
    if model_name is None:
        model_name = getattr(config.model, 'name_or_path', None)
    if not model_name:
        raise ValueError("A policy or value model path is required for PPO")

    logger.info("Loading PPO value model from: %s", model_name)

    from transformers import AutoModelForSequenceClassification

    precision = PrecisionHandler.get_precision_from_config(config, default="auto")
    dtype = PrecisionHandler.get_torch_dtype(precision)
    device_map = getattr(config.model, 'device_map', None) or "auto"
    quantization = (
        getattr(config.model, 'value_model_quantization', None)
        or getattr(config.model, 'quantization', None)
        or {}
    )
    value_quantization_config = _build_bnb_quantization_config(quantization)
    value_model_kwargs = {
        'num_labels': 1,
        'torch_dtype': dtype,
        'device_map': device_map,
        'trust_remote_code': getattr(config.model, 'trust_remote_code', False),
        'ignore_mismatched_sizes': True,
    }
    if value_quantization_config is not None:
        value_model_kwargs['quantization_config'] = value_quantization_config

    value_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **value_model_kwargs,
    )

    # TRL's PolicyAndValueWrapper accesses these attributes directly.  A
    # plain HF sequence-classification model keeps its parameters registered;
    # wrapping it in UniversalRewardModelWrapper here would hide them from
    # the optimizer.
    if not hasattr(value_model, 'base_model_prefix') or not hasattr(value_model, 'score'):
        raise TypeError(
            "TRL PPO requires a causal sequence-classification value model "
            "with `base_model_prefix` and `score`; "
            f"{type(value_model).__name__} does not provide that interface."
        )

    value_model.eval()
    return value_model
