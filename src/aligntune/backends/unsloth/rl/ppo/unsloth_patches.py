"""Patches for Unsloth compatibility with sequence classification models."""
import torch
import logging
import shutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# trl>=1.0 removed these two from trl.trainer.utils (present in the trl==0.23.0
# this backend was originally written against). Local fallbacks matching their
# historical trl implementations, used only when the installed trl no longer
# exports them.
def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """Index of the first True along the last dim of `bools` (row length if none is True)."""
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}{{message['role'].capitalize() + ': ' + "
    "message['content']}}\n\n{% endfor %}{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}{% endif %}"
)


def get_first_true_indices():
    """Return trl's first_true_indices if available, else the local fallback."""
    try:
        from trl.trainer.utils import first_true_indices as _trl_impl
        return _trl_impl
    except ImportError:
        logger.warning(
            "trl.trainer.utils.first_true_indices not found (removed in trl>=1.0) "
            "- using local fallback implementation")
        return first_true_indices


def get_simple_chat_template():
    """Return trl's SIMPLE_CHAT_TEMPLATE if available, else the local fallback."""
    try:
        from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE as _trl_impl
        return _trl_impl
    except ImportError:
        logger.warning(
            "trl.trainer.utils.SIMPLE_CHAT_TEMPLATE not found (removed in trl>=1.0) "
            "- using local fallback implementation")
        return SIMPLE_CHAT_TEMPLATE


def clear_all_unsloth_caches():
    """Aggressively clear all Unsloth caches before training."""
    logger.info("Clearing all Unsloth caches...")

    # Clear local cache
    cache_dir = Path.cwd() / "unsloth_compiled_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info(f"  Cleared: {cache_dir}")

    # Clear global cache if set
    if 'UNSLOTH_COMPILED_CACHE' in os.environ:
        global_cache = Path(os.environ['UNSLOTH_COMPILED_CACHE'])
        if global_cache.exists():
            shutil.rmtree(global_cache)
            logger.info(f"  Cleared: {global_cache}")

    # Force recompilation
    os.environ['UNSLOTH_FORCE_RECOMPILE'] = '1'
    logger.info("  Set UNSLOTH_FORCE_RECOMPILE=1")


def _apply_qkv(self, *args):
    """Apply QKV projections. Handles Unsloth's self.apply_qkv(self, hidden_states) pattern."""
    # Handle both calling patterns:
    # 1. self.apply_qkv(hidden_states) - 2 args
    # 2. self.apply_qkv(self, hidden_states) - 3 args (Unsloth pattern)
    if len(args) == 1:
        hidden_states = args[0]
    elif len(args) == 2:
        hidden_states = args[1]  # Skip the explicit self
    else:
        raise ValueError(
            f"apply_qkv called with {
                len(args)} arguments, expected 1 or 2")

    Q = self.q_proj(hidden_states)
    K = self.k_proj(hidden_states)
    V = self.v_proj(hidden_states)
    return Q, K, V


def _apply_o(self, *args):
    """Apply output projection. Handles Unsloth's self.apply_o(self, attn_output) pattern."""
    # Handle both calling patterns:
    # 1. self.apply_o(attn_output) - 2 args
    # 2. self.apply_o(self, attn_output) - 3 args (Unsloth pattern)
    if len(args) == 1:
        attn_output = args[0]
    elif len(args) == 2:
        attn_output = args[1]  # Skip the explicit self
    else:
        raise ValueError(
            f"apply_o called with {
                len(args)} arguments, expected 1 or 2")

    return self.o_proj(attn_output)


# Unsloth model modules whose Attention class may lack apply_qkv/apply_o -
# each of these shares the same q_proj/k_proj/v_proj/o_proj layout, so the
# same patch logic applies to all of them.
_UNSLOTH_ATTENTION_MODULES = (
    ("unsloth.models.qwen3", "Qwen3Attention"),
    ("unsloth.models.qwen2", "Qwen2Attention"),
)


def patch_attention_classes_globally():
    """Patch attention classes at the class level BEFORE model loading.

    This ensures Unsloth's compilation process sees the patched methods.
    """
    for module_path, class_name in _UNSLOTH_ATTENTION_MODULES:
        try:
            module = __import__(module_path, fromlist=[class_name])
            attention_cls = getattr(module, class_name)

            if hasattr(attention_cls, '_aligntune_patched'):
                logger.info(f"{class_name} already patched at class level")
                continue

            attention_cls.apply_qkv = _apply_qkv
            attention_cls.apply_o = _apply_o
            attention_cls._aligntune_patched = True

            logger.info(
                f"Successfully patched {class_name} class with apply_qkv and apply_o methods")

        except ImportError as e:
            logger.warning(f"Could not import {class_name} for patching: {e}")
        except Exception as e:
            logger.error(f"Error patching {class_name} class: {e}")
            raise


def verify_attention_patches(model):
    """Verify that attention modules have apply_qkv method."""
    verified = 0
    missing = []

    for name, module in model.named_modules():
        if 'Attention' in type(module).__name__:
            if hasattr(module, 'apply_qkv'):
                verified += 1
            else:
                missing.append(name)

    if missing:
        logger.error(
            f"Missing apply_qkv on {len(missing)} modules: {missing[:3]}...")
        return False

    if verified > 0:
        logger.info(f"Verified {verified} attention modules have apply_qkv")
        return True
    else:
        logger.warning("No attention modules found to verify")
        return True


def print_model_structure(model, max_depth=3, current_depth=0):
    """Print model structure for debugging."""
    indent = "  " * current_depth

    if current_depth == 0:
        logger.info(f"🏗️ Model structure for {type(model).__name__}:")

    if current_depth >= max_depth:
        logger.debug(f"{indent}... (max depth reached)")
        return

    for name, module in model.named_children():
        module_type = type(module).__name__
        has_apply_qkv = hasattr(module, 'apply_qkv')
        logger.debug(
            f"{indent}{name}: {module_type} (apply_qkv: {has_apply_qkv})")

        # Recursively print children if not at max depth
        if current_depth < max_depth - 1:
            print_model_structure(module, max_depth, current_depth + 1)


def patch_attention_apply_qkv(model):
    """
    Add apply_qkv method to attention modules that lack it.
    This is needed when loading models with AutoModelForSequenceClassification.
    """
    patched_count = 0
    total_modules = 0
    attention_modules = 0

    logger.info(
        f"🔍 Starting patch_attention_apply_qkv on model type: {
            type(model).__name__}")

    # Print model structure for debugging
    print_model_structure(model, max_depth=2)

    for name, module in model.named_modules():
        total_modules += 1
        module_type = type(module).__name__

        # Debug: Log all modules to understand the structure
        # Look for attention modules with various naming patterns
        is_attention_module = (
            'Attention' in module_type or
            'SelfAttention' in module_type or
            'MultiHeadAttention' in module_type or
            'Qwen3Attention' in module_type or
            'LlamaAttention' in module_type or
            'MistralAttention' in module_type or
            'GemmaAttention' in module_type
        )

        if is_attention_module:
            attention_modules += 1
            logger.info(
                f"Found attention module: {name} (type: {module_type})")

            # Check if this is an attention module without apply_qkv
            has_apply_qkv = hasattr(module, 'apply_qkv')
            logger.info(
                f"🔍 Attention module {name}: has_apply_qkv={has_apply_qkv}")

            # Force patch even if method exists (it might be broken)
            logger.info(
                f"🔧 Force patching attention module: {name} (type: {module_type})")

            # Debug: Print module attributes to understand structure
            logger.debug(
                f"Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")

            # Check if module has the required projection layers
            has_q_proj = hasattr(module, 'q_proj')
            has_k_proj = hasattr(module, 'k_proj')
            has_v_proj = hasattr(module, 'v_proj')
            has_o_proj = hasattr(module, 'o_proj')

            logger.debug(
                f"Projection layers - q_proj: {has_q_proj}, k_proj: {has_k_proj}, v_proj: {has_v_proj}, o_proj: {has_o_proj}")

            if not (has_q_proj and has_k_proj and has_v_proj and has_o_proj):
                logger.warning(
                    f"⚠️ Attention module {name} missing required projection layers, skipping")
                continue

            # Create a proper bound method that handles both calling patterns
            def make_apply_qkv(attn_module, module_name):
                def apply_qkv(*args):
                    """Apply QKV projections to hidden states."""
                    try:
                        # Handle the specific Unsloth calling pattern:
                        # self.apply_qkv(self, hidden_states)
                        if len(args) == 3 and args[0] is attn_module:
                            # Called as self.apply_qkv(self, hidden_states) -
                            # Unsloth pattern
                            hidden_states = args[2]
                        elif len(args) == 2:
                            # Called as self.apply_qkv(hidden_states)
                            hidden_states = args[1]
                        else:
                            # Try to extract hidden_states from the last
                            # argument
                            hidden_states = args[-1]

                        logger.debug(
                            f"apply_qkv called on {module_name} with {
                                len(args)} args, hidden_states shape: {
                                hidden_states.shape}")

                        # Apply QKV projections
                        Q = attn_module.q_proj(hidden_states)
                        K = attn_module.k_proj(hidden_states)
                        V = attn_module.v_proj(hidden_states)

                        logger.debug(
                            f"QKV projections completed - Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
                        return Q, K, V

                    except Exception as e:
                        logger.error(
                            f"❌ Error in apply_qkv for {module_name}: {e}")
                        logger.error(f"Args: {args}")
                        logger.error(f"Module type: {type(attn_module)}")
                        raise e
                return apply_qkv

            # Create apply_o method
            def make_apply_o(attn_module, module_name):
                def apply_o(*args):
                    """Apply output projection to attention output."""
                    try:
                        # Handle both self.apply_o(attn_output) and
                        # self.apply_o(self, attn_output)
                        if len(args) == 2:
                            # Called as self.apply_o(attn_output)
                            attn_output = args[1]
                        elif len(args) == 3:
                            # Called as self.apply_o(self, attn_output) - skip
                            # the explicit self
                            attn_output = args[2]
                        else:
                            raise ValueError(
                                f"apply_o called with {
                                    len(args)} arguments, expected 2 or 3")

                        logger.debug(
                            f"apply_o called on {module_name} with attn_output shape: {
                                attn_output.shape}, dtype: {
                                attn_output.dtype}")

                        # Apply output projection
                        output = attn_module.o_proj(attn_output)

                        logger.debug(
                            f"Output projection completed - output: {output.shape}")
                        return output

                    except Exception as e:
                        logger.error(
                            f"❌ Error in apply_o for {module_name}: {e}")
                        logger.error(f"Args: {args}")
                        logger.error(f"Module type: {type(attn_module)}")
                        raise e
                return apply_o

            # Bind both methods to the module
            import types
            module.apply_qkv = types.MethodType(
                make_apply_qkv(module, name), module)
            module.apply_o = types.MethodType(
                make_apply_o(module, name), module)
            patched_count += 1
            logger.info(f"✅ Patched apply_qkv and apply_o for {name}")
        else:
            # Log non-attention modules at debug level
            logger.debug(f"Non-attention module: {name} (type: {module_type})")

    logger.info(
        f"📊 Patch summary: {total_modules} total modules, {attention_modules} attention modules, {patched_count} patched")

    if patched_count > 0:
        logger.info(
            f"✅ Patched {patched_count} attention modules with apply_qkv method")
    else:
        logger.warning("⚠️ No attention modules were patched")

    return model


def handle_model_compatibility(model):
    """Handle different model types and ensure compatibility."""
    model_type = type(model).__name__
    logger.info(f"🔧 Handling compatibility for model type: {model_type}")

    # Check if model has the expected attributes
    has_base_model_prefix = hasattr(model, 'base_model_prefix')
    has_config = hasattr(model, 'config')
    has_modules = hasattr(model, 'modules')
    has_parameters = hasattr(model, 'parameters')

    logger.debug(
        f"Model attributes - base_model_prefix: {has_base_model_prefix}, config: {has_config}, modules: {has_modules}, parameters: {has_parameters}")

    # Handle different model architectures
    if 'GPT' in model_type or 'Qwen' in model_type:
        logger.info("🤖 Detected GPT/Qwen architecture")
    elif 'BERT' in model_type or 'DeBERTa' in model_type:
        logger.info("🤖 Detected BERT/DeBERTa architecture")
    elif 'T5' in model_type:
        logger.info("🤖 Detected T5 architecture")
    else:
        logger.info(f"🤖 Unknown architecture: {model_type}")

    return model


def disable_unsloth_forward(model):
    """Revert to standard transformers forward pass."""
    for name, module in model.named_modules():
        if hasattr(module, '__class__'):
            # Check if using Unsloth's fast forward
            if 'fast_forward' in str(
                module.__class__.__dict__.get(
                    'forward',
                    '')):
                # Revert to original forward
                original_class = module.__class__.__bases__[0]
                if hasattr(original_class, 'forward'):
                    module.forward = original_class.forward.__get__(
                        module, type(module))
                    logger.debug(f"Reverted {name} to standard forward")
    return model
