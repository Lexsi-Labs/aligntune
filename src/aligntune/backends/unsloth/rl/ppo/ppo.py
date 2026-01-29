"""
Unsloth PPO Backend Implementation - WORKING VERSION

Direct port from working TRL PPO code with Unsloth optimizations.
"""

# CRITICAL: Import Unsloth FIRST before any other ML libraries
import unsloth
from unsloth import FastLanguageModel

import logging
import time
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from datasets import load_dataset, concatenate_datasets
from accelerate import PartialState

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig
from aligntune.rewards.registry import RewardRegistry
from aligntune.core.rl.reward_model_wrapper import UniversalRewardModelWrapper
from aligntune.core.rl.sample_logger import generate_and_log_samples
from aligntune.utils.config_extractor import  extract_extra_and_missing_params

# Debug flag - set to True to enable verbose debug logging
DEBUG = False

logger = logging.getLogger(__name__)

# GLOBAL REFERENCE for patched_get_reward to access trainer state
GLOBAL_UNSLOTH_TRAINER_REF = None

def patched_get_reward(model, query_responses, pad_token_id, context_length):
    """Patched get_reward to handle encoder models and function-based rewards."""
    import os
    from trl.trainer.utils import first_true_indices
    import torch
    
    DEBUG_MODE = os.environ.get("FINETUNEHUB_DEBUG", "0") == "1"
    trainer_instance = GLOBAL_UNSLOTH_TRAINER_REF
    
    if DEBUG_MODE:
        print(f"DEBUG: Patched get_reward called with model type: {type(model).__name__}", flush=True)
    
    # Import reward model types inside function to avoid circular imports
    from aligntune.core.rl.function_based_reward_model import FunctionBasedRewardModel
    from aligntune.core.rl.reward_model_wrapper import UniversalRewardModelWrapper
    
    # Decode and re-tokenize if reward tokenizer is different from policy tokenizer
    if trainer_instance and hasattr(trainer_instance, 'reward_tokenizer') and trainer_instance.reward_tokenizer is not None:
        # Decode with policy tokenizer
        texts = trainer_instance.tokenizer.batch_decode(query_responses, skip_special_tokens=True)
        
        # Encode with reward tokenizer
        max_len = 512
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            max_len = model.config.max_position_embeddings
        elif hasattr(trainer_instance.reward_tokenizer, 'model_max_length'):
            tok_max = trainer_instance.reward_tokenizer.model_max_length
            if tok_max < 10000:
                max_len = tok_max
        
        inputs = trainer_instance.reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )
        
        device = next(model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        
        retokenized = True
    else:
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        retokenized = False
    
    if isinstance(model, FunctionBasedRewardModel):
        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True
        )
        reward_logits = model.score(output.hidden_states)
        sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        
        final_rewards = reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1)
        
        return (reward_logits, final_rewards, sequence_lengths)
    
    # Neural reward model
    try:
        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_dict': True
        }
        
        is_encoder = False
        if hasattr(model, 'config'):
            model_type = getattr(model.config, 'model_type', '').lower()
            encoder_models = ['bert', 'roberta', 'deberta', 'deberta-v2', 'electra', 'albert', 'distilbert', 'xlm-roberta']
            if any(enc in model_type for enc in encoder_models):
                is_encoder = True
        
        if is_encoder and 'position_ids' in locals() and position_ids is not None:
            kwargs['position_ids'] = position_ids
            
        if not is_encoder and hasattr(model, 'config'):
            kwargs['use_cache'] = False
        
        try:
            outputs = model(**kwargs)
        except TypeError as te:
            if 'position_ids' in str(te):
                if 'position_ids' in kwargs: del kwargs['position_ids']
                outputs = model(**kwargs)
            elif 'use_cache' in str(te):
                if 'use_cache' in kwargs: del kwargs['use_cache']
                outputs = model(**kwargs)
            else:
                raise te
        
        if hasattr(outputs, 'logits'):
            reward_logits = outputs.logits
        elif isinstance(outputs, tuple):
            reward_logits = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            reward_logits = outputs
        else:
            raise ValueError(f"Output type {type(outputs)} has no logits")
            
    except Exception as e:
        if DEBUG_MODE: logger.warning(f"Forward pass failed: {e}")
        # Fallback
        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(model, 'score'):
            reward_logits = model.score(output.hidden_states[-1])
        elif hasattr(model, 'classifier'):
            reward_logits = model.classifier(output.hidden_states[-1])
        else:
             raise AttributeError("Model has no score or classifier")

    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    
    if retokenized and reward_logits.dim() == 2:
         final_rewards = reward_logits.squeeze(-1)
    elif reward_logits.dim() == 3:
         final_rewards = reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1)
    else:
         final_rewards = reward_logits.squeeze(-1)

    if final_rewards.dim() == 0:
         final_rewards = final_rewards.unsqueeze(0)
    
    return (reward_logits, final_rewards, sequence_lengths)

# Apply patch IMMEDIATELY before any TRL imports
try:
    import trl.trainer.utils
    trl.trainer.utils.get_reward = patched_get_reward
    logger.info("✅ Applied TRL get_reward patch globally before ppo_trainer import")
except Exception as e:
    logger.warning(f"Could not apply global patch: {e}")

# Patch TRL's PolicyAndValueWrapper for Unsloth compatibility
# Patch TRL's PolicyAndValueWrapper for Unsloth compatibility
try:
    from trl.trainer.ppo_trainer import PolicyAndValueWrapper as _TRLPolicyAndValueWrapper
    
    if not hasattr(_TRLPolicyAndValueWrapper, "gradient_checkpointing_disable"):
        def _pav_disable_gc(self):
            """Disable gradient checkpointing on underlying models."""
            try:
                policy = getattr(self, "policy_model", None)
                if policy and hasattr(policy, "gradient_checkpointing_disable"):
                    policy.gradient_checkpointing_disable()
                base = getattr(self, "model", None)
                if base and hasattr(base, "gradient_checkpointing_disable"):
                    base.gradient_checkpointing_disable()
            except Exception:
                pass
        
        setattr(_TRLPolicyAndValueWrapper, "gradient_checkpointing_disable", _pav_disable_gc)
        logger.info("Patched PolicyAndValueWrapper.gradient_checkpointing_disable() for compatibility")
    
    if not hasattr(_TRLPolicyAndValueWrapper, "gradient_checkpointing_enable"):
        def _pav_enable_gc(self, gradient_checkpointing_kwargs=None):
            """Enable gradient checkpointing on underlying models."""
            try:
                policy = getattr(self, "policy_model", None)
                if policy and hasattr(policy, "gradient_checkpointing_enable"):
                    if gradient_checkpointing_kwargs:
                        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                    else:
                        policy.gradient_checkpointing_enable()
                base = getattr(self, "model", None)
                if base and hasattr(base, "gradient_checkpointing_enable"):
                    if gradient_checkpointing_kwargs:
                        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                    else:
                        base.gradient_checkpointing_enable()
            except Exception:
                pass
        
        setattr(_TRLPolicyAndValueWrapper, "gradient_checkpointing_enable", _pav_enable_gc)
        logger.info("Patched PolicyAndValueWrapper.gradient_checkpointing_enable() for compatibility")
    
    if not hasattr(_TRLPolicyAndValueWrapper, "generate"):
        def _pav_generate(self, *args, **kwargs):
            policy = getattr(self, "policy_model", None)
            if policy and hasattr(policy, "generate"):
                return policy.generate(*args, **kwargs)
            base = getattr(self, "model", None)
            if base and hasattr(base, "generate"):
                return base.generate(*args, **kwargs)
            raise AttributeError("No generate method available")
        
        setattr(_TRLPolicyAndValueWrapper, "generate", _pav_generate)
        logger.info("Patched PolicyAndValueWrapper.generate() for Unsloth compatibility")
        
except Exception as e:
    logger.warning(f"Could not patch PolicyAndValueWrapper: {e}")

class UnslothPPOTrainer(TrainerBase):
    """PPO trainer with hybrid Unsloth architecture.
    
    Model Architecture:
    - Policy Model: Unsloth FastLanguageModel + LoRA (trained, optimized)
    - Reference Model: Unsloth FastLanguageModel (frozen, memory efficient)
    - Reward Model: Standard HuggingFace (frozen, any architecture)
    - Value Model: Standard HuggingFace (trained alongside policy)
    
    Key Insight: Unsloth models need manual patches for apply_qkv method.
    Standard models work naturally with TRL without patches.
    
    Reward Model Options:
    1. Pretrained: Load from HuggingFace (DeBERTa, RoBERTa, etc.)
    2. Custom trained: Train using TRL's RewardTrainer
    3. Hybrid: Fine-tune pretrained with custom reward functions
    """
    
    def __init__(self, config: UnifiedConfig):
        super().__init__(config)
        
        # Validate Unsloth is properly loaded
        import os
        if os.environ.get("UNSLOTH_IS_PRESENT") != "1":
            raise RuntimeError(
                "Unsloth PPO backend requires Unsloth to be imported first. "
                "This should be handled by the factory, but validation failed. "
                "Please report this as a bug."
            )
        
        # Rest of existing initialization
        self.unsloth_model = None
        self.policy_model = None
        self.ref_model = None
        self.reward_model = None
        self.value_model = None
        self.trainer = None
        self.training_history = []
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset_dict = None
        
    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth and TRL are available and properly configured."""
        try:
            # Check Unsloth installation
            import unsloth
            from unsloth import FastLanguageModel
            
            # Check if Unsloth is properly configured
            import os
            if os.environ.get("UNSLOTH_IS_PRESENT") != "1":
                logger.warning("Unsloth not properly initialized")
                return False
                
            # Check TRL availability
            from trl import PPOTrainer, PPOConfig
            from transformers import AutoModelForSequenceClassification
            
            return True
        except ImportError as e:
            logger.warning(f"Unsloth PPO backend not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unsloth PPO backend configuration error: {e}")
            return False
    
    def _get_config_value(self, config_obj, key: str, default=None):
        """Helper to get config value from dict or object."""
        if isinstance(config_obj, dict):
            return config_obj.get(key, default)
        else:
            return getattr(config_obj, key, default)
    
    def _detect_model_type(self, model_name: str) -> str:
        """Detect if model should use Unsloth or standard loading.
        
        Returns:
            'unsloth': Use Unsloth FastLanguageModel
            'standard': Use standard transformers
        """
        # Check config override first
        model_config = self.config.model if hasattr(self.config, 'model') else {}
        if isinstance(model_config, dict):
            force_type = model_config.get('reward_value_loading_type')
        else:
            force_type = getattr(model_config, 'reward_value_loading_type', None)
        
        if force_type in ['unsloth', 'standard']:
            logger.info(f"Using config-specified loading type: {force_type}")
            return force_type
        
        # Models that CANNOT use Unsloth (encoder-only models)
        incompatible_models = ['bert', 'roberta', 'distilbert', 'albert', 'electra', 'deberta']
        if any(indicator in model_name.lower() for indicator in incompatible_models):
            logger.info(f"Using standard loading for encoder model: {model_name}")
            return 'standard'
        
        # Default to Unsloth for decoder models (Llama, GPT, Mistral, etc.)
        # This ensures they get proper rotary embedding patches
        logger.info(f"Using Unsloth loading for decoder model: {model_name}")
        return 'unsloth'
    
    def _load_reward_value_models_unsloth(self, model_name: str, max_seq_length: int, quantization: dict):
        """Load reward/value models using Unsloth for optimization."""
        logger.info(f"Loading reward/value models with Unsloth: {model_name}")
        
        # Check for integrated reward training first
        reward_model_name = self._get_config_value(self.config.model, "reward_model_name", None)
        
        # Check if integrated reward training is enabled
        if (hasattr(self.config, 'reward_training') and 
            self.config.reward_training and 
            self.config.reward_training.enabled):
            
            logger.info("🏋️ Training custom reward model before PPO...")
            reward_model_path = self._train_custom_reward_model()
            # Override reward_model_name with trained model path
            reward_model_name = reward_model_path
            logger.info(f"✅ Custom reward model trained and saved to: {reward_model_path}")
        
        # Get quantization settings (configurable per model)
        reward_quant = self._get_config_value(self.config.model, 'reward_model_quantization', quantization)
        value_quant = self._get_config_value(self.config.model, 'value_model_quantization', quantization)
        
        # Load reward model
        if reward_model_name:
            logger.info(f"Loading pre-trained reward model: {reward_model_name}")
            from transformers import AutoModelForSequenceClassification
            
            # Load pre-trained reward model
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Wrap with UniversalRewardModelWrapper for compatibility
            self.reward_model = UniversalRewardModelWrapper(self.reward_model)
            
            # DEBUG: Check wrapper
            logger.info(f"🔍 DEBUG: Wrapped reward model type: {type(self.reward_model).__name__}")
            logger.info(f"🔍 DEBUG: Wrapped reward model has score: {hasattr(self.reward_model, 'score')}")
            logger.info(f"🔍 DEBUG: Base model type: {type(self.reward_model._model).__name__}")
            logger.info(f"🔍 DEBUG: Base model has classifier: {hasattr(self.reward_model._model, 'classifier')}")
            
            # Verify score method exists for Unsloth PPO compatibility
            if not hasattr(self.reward_model, 'score'):
                logger.error("Reward model wrapper missing score method!")
                base_model = self.reward_model._model
                if hasattr(base_model, 'classifier'):
                    def score_method(hidden_states):
                        return base_model.classifier(hidden_states)
                    self.reward_model.score = score_method
                    logger.info("Added score method to Unsloth reward model wrapper")
                else:
                    raise AttributeError("Reward model must have score() or classifier() method")
            else:
                logger.info("Unsloth reward model wrapper has score method")
            
            logger.info("✅ Loaded pre-trained reward model with UniversalRewardModelWrapper")
            
        else:
            # Create reward model from scratch (NO quantization for TRL compatibility)
            reward_base, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=False,  # CRITICAL: Disable quantization for TRL compatibility
            )
            if hasattr(reward_base, "config"):
                reward_base.config.output_hidden_states = True
            import torch.nn as nn
            hidden_size = reward_base.config.hidden_size
            base_dtype = next(reward_base.parameters()).dtype
            self.reward_model = reward_base
            self.reward_model.score = nn.Linear(hidden_size, 1, bias=False).to(base_dtype).to(reward_base.device)
            
            # Verify score layer was added correctly
            if not hasattr(self.reward_model, 'score'):
                raise RuntimeError("Failed to add score layer to Unsloth reward model")
            logger.info(f"Unsloth reward model score layer: {self.reward_model.score}")
        
        # Load value model similarly (NO quantization to avoid .to() errors in PPOTrainer)
        value_base, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,  # CRITICAL: Disable quantization for TRL compatibility
        )
        if hasattr(value_base, "config"):
            value_base.config.output_hidden_states = True
        self.value_model = value_base
        value_dtype = next(value_base.parameters()).dtype
        self.value_model.score = nn.Linear(hidden_size, 1, bias=False).to(value_dtype).to(value_base.device)
        
        # Disable gradient checkpointing
        for model in [self.reward_model, self.value_model]:
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            if hasattr(model, 'config'):
                model.config.gradient_checkpointing = False
                model.config.use_cache = True
        
        logger.info("✅ Reward/value models loaded with Unsloth optimizations")
    
    def _load_reward_value_models_standard(self, model_name: str):
        """Load reward/value models using standard transformers (no Unsloth patches)."""
        logger.info(f"Loading reward/value models with standard transformers: {model_name}")
        
        from transformers import AutoModelForSequenceClassification
        
        # Get quantization settings (configurable per model)
        reward_quant = self._get_config_value(self.config.model, 'reward_model_quantization', {})
        value_quant = self._get_config_value(self.config.model, 'value_model_quantization', {})
        
        # Get precision from config
        precision = self._get_config_value(self.config.model, "precision", "bfloat16")
        if hasattr(precision, 'value'):
            precision = precision.value
        torch_dtype = getattr(torch, precision) if precision in ['float16', 'bfloat16', 'float32'] else torch.bfloat16
        
        # Load reward model
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch_dtype,
            device_map="auto",
            load_in_4bit=reward_quant.get("load_in_4bit", False),
        )
        if hasattr(self.reward_model, 'config'):
            self.reward_model.config.output_hidden_states = True
        
        # Wrap with UniversalRewardModelWrapper for compatibility
        self.reward_model = UniversalRewardModelWrapper(self.reward_model)
        
        # Verify score method for standard transformers path
        if not hasattr(self.reward_model, 'score'):
            base_model = self.reward_model._model
            if hasattr(base_model, 'classifier'):
                def score_method(hidden_states):
                    return base_model.classifier(hidden_states)
                self.reward_model.score = score_method
                logger.info("Added score method to standard reward model wrapper")
        else:
            logger.info("Standard reward model wrapper has score method")
        
        # Load value model
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch_dtype,
            device_map="auto",
            load_in_4bit=value_quant.get("load_in_4bit", False),
        )
        if hasattr(self.value_model, 'config'):
            self.value_model.config.output_hidden_states = True
        
        # Disable gradient checkpointing
        for model in [self.reward_model, self.value_model]:
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
            if hasattr(model, 'config'):
                model.config.gradient_checkpointing = False
                model.config.use_cache = True
        
        # Standard HuggingFace models don't need Unsloth patches
        # They work naturally with TRL's PPOTrainer
        
        logger.info("✅ Reward/value models loaded with standard transformers")
    
    def _load_value_model_unsloth(self, model_name: str, max_seq_length: int, quantization: dict):
        """Load only value model using Unsloth."""
        logger.info(f"Loading value model with Unsloth: {model_name}")
        
        value_quant = self._get_config_value(self.config.model, 'value_model_quantization', quantization)
        
        value_base, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=value_quant.get("load_in_4bit", True),
        )
        if hasattr(value_base, "config"):
            value_base.config.output_hidden_states = True
        
        import torch.nn as nn
        hidden_size = value_base.config.hidden_size
        self.value_model = value_base
        base_dtype = next(value_base.parameters()).dtype
        self.value_model.score = nn.Linear(hidden_size, 1, bias=False).to(base_dtype).to(value_base.device)
        
        if hasattr(self.value_model, 'gradient_checkpointing_disable'):
            self.value_model.gradient_checkpointing_disable()
        if hasattr(self.value_model, 'config'):
            self.value_model.config.gradient_checkpointing = False
            self.value_model.config.use_cache = True
        
        logger.info("✅ Value model loaded with Unsloth optimizations")

    def _load_value_model_standard(self, model_name: str):
        """Load only value model using standard transformers (NO Unsloth patches)."""
        logger.info(f"Loading value model with standard transformers: {model_name}")
        
        # Warn if trying to load Unsloth model with standard transformers
        if "unsloth" in model_name.lower():
            logger.warning(f"⚠️ Loading Unsloth model '{model_name}' with standard transformers")
            logger.warning("⚠️ This may cause compatibility issues with Unsloth's PPOTrainer")
            logger.warning("⚠️ Consider using a standard HuggingFace model or set reward_value_loading_type='unsloth'")
        
        from transformers import AutoModelForSequenceClassification
        
        value_quant = self._get_config_value(self.config.model, 'value_model_quantization', {})
        
        # Get precision from config
        precision = self._get_config_value(self.config.model, "precision", "bfloat16")
        if hasattr(precision, 'value'):
            precision = precision.value
        torch_dtype = getattr(torch, precision) if precision in ['float16', 'bfloat16', 'float32'] else torch.bfloat16
        
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch_dtype,
            device_map="auto",
            load_in_4bit=value_quant.get("load_in_4bit", False),
        )
        if hasattr(self.value_model, 'config'):
            self.value_model.config.output_hidden_states = True
        
        # DEBUG: Check value model dtype after loading
        if DEBUG:
            try:
                value_dtype = next(self.value_model.parameters()).dtype
                logger.info(f"🔍 DEBUG: Value model loaded with dtype: {value_dtype}")
                logger.info(f"🔍 DEBUG: Target dtype was: {torch_dtype}")
            except Exception as e:
                logger.warning(f"🔍 DEBUG: Could not get value model dtype: {e}")
        
        if hasattr(self.value_model, 'gradient_checkpointing_disable'):
            self.value_model.gradient_checkpointing_disable()
        if hasattr(self.value_model, 'config'):
            self.value_model.config.gradient_checkpointing = False
            self.value_model.config.use_cache = True
        
        # DO NOT apply Unsloth patches to standard models
        logger.info("✅ Value model loaded with standard transformers (no Unsloth patches)")
    
    def _get_config_hash(self):
        """Generate hash of relevant configuration for cache invalidation."""
        import hashlib
        import json
        
        # Get config values
        model_name = self._get_config_value(self.config.model, "name_or_path")
        max_seq_length = self._get_config_value(self.config.model, "max_seq_length", 512)
        quantization = self._get_config_value(self.config.model, "quantization", {})
        gradient_checkpointing = self._get_config_value(self.config.model, "gradient_checkpointing", True)
        reward_value_model = self._get_config_value(self.config.model, "reward_value_model", "meta-llama/Llama-3.2-1B-Instruct")
        
        # Get library versions
        try:
            import unsloth
            unsloth_version = getattr(unsloth, "__version__", "unknown")
        except:
            unsloth_version = "not_installed"
        
        try:
            import transformers
            transformers_version = getattr(transformers, "__version__", "unknown")
        except:
            transformers_version = "not_installed"
        
        try:
            import torch
            torch_version = getattr(torch, "__version__", "unknown")
        except:
            torch_version = "not_installed"
        
        config_dict = {
            'model_name': model_name,
            'max_seq_length': max_seq_length,
            'quantization': quantization,
            'gradient_checkpointing': gradient_checkpointing,
            'reward_value_model': reward_value_model,
            'unsloth_version': unsloth_version,
            'transformers_version': transformers_version,
            'torch_version': torch_version,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _clear_unsloth_cache(self, force: bool = False):
        """Clear Unsloth compiled cache to force recompilation.
        
        Args:
            force: If True, always clear cache. If False, only clear if config changed.
        """
        import shutil
        from pathlib import Path
        
        # Clear local cache
        cache_dir = Path.cwd() / "unsloth_compiled_cache"
        if cache_dir.exists():
            if force:
                logger.info("🗑️  Force clearing Unsloth compiled cache...")
                shutil.rmtree(cache_dir)
                logger.info(f"✅ Cleared local cache: {cache_dir}")
            else:
                logger.info(f"Cache exists at: {cache_dir}")
        
        # Clear global Unsloth cache if set
        import os
        if 'UNSLOTH_COMPILED_CACHE' in os.environ:
            global_cache = Path(os.environ['UNSLOTH_COMPILED_CACHE'])
            if global_cache.exists():
                if force:
                    logger.info(f"🗑️  Force clearing global Unsloth cache...")
                    shutil.rmtree(global_cache)
                    logger.info(f"✅ Cleared global cache: {global_cache}")
                else:
                    logger.info(f"Global cache exists at: {global_cache}")
        
        # Set environment variable to force recompilation
        if force:
            os.environ['UNSLOTH_FORCE_RECOMPILE'] = '1'
            logger.info("✅ Set UNSLOTH_FORCE_RECOMPILE=1")

    def _maybe_invalidate_unsloth_cache(self):
        """Invalidate Unsloth cache if configuration has changed."""
        from pathlib import Path
        import shutil
        
        cache_dir = Path.cwd() / "unsloth_compiled_cache"
        config_file = cache_dir / ".config_hash"
        
        current_hash = self._get_config_hash()
        logger.info(f"Current config hash: {current_hash[:8]}...")
        
        if config_file.exists():
            stored_hash = config_file.read_text().strip()
            if stored_hash != current_hash:
                logger.info(f"Config changed ({stored_hash[:8]}... -> {current_hash[:8]}...), invalidating cache")
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    logger.info(f"✓ Cleared Unsloth cache: {cache_dir}")
            else:
                logger.info(f"Config unchanged ({current_hash[:8]}...), reusing cache")
        else:
            logger.info("No previous cache found, will create new cache")
        
        # Ensure cache dir exists and store new hash
        cache_dir.mkdir(exist_ok=True)
        config_file.write_text(current_hash)

    def _validate_model_compatibility(self):
        """Validate that model types are compatible with loading strategy."""
        # Check value model
        model_name = self._get_config_value(self.config.model, "reward_value_model", "")
        loading_type = self._get_config_value(self.config.model, "reward_value_loading_type", "auto")
        
        if loading_type == "standard" and "unsloth" in model_name.lower():
            raise ValueError(
                f"Incompatible configuration: reward_value_model='{model_name}' "
                f"with reward_value_loading_type='standard'. "
                f"Either use a standard HuggingFace model or set loading_type='unsloth'"
            )
        
        # Check reward model
        reward_model_name = self._get_config_value(self.config.model, "reward_model_name", None)
        if reward_model_name and "unsloth" in reward_model_name.lower():
            logger.warning(f"⚠️ Using Unsloth reward model '{reward_model_name}' with standard loading")
            logger.warning("⚠️ This may cause compatibility issues. Consider using a standard HuggingFace reward model")
    def _disable_gradient_checkpointing_for_all(self):
        """Disable gradient checkpointing on all models - UNSLOTH-SAFE VERSION."""

        def _safe_replace_gc_attr(obj, name="(unknown)", indent=""):
            """Replace gradient checkpointing attributes safely."""
            try:
                # DON'T replace _gradient_checkpointing_func - just disable the flag
                if hasattr(obj, "gradient_checkpointing"):
                    obj.gradient_checkpointing = False
                # For config objects
                if hasattr(obj, "config"):
                    cfg = getattr(obj, "config", None)
                    if cfg and hasattr(cfg, "gradient_checkpointing"):
                        cfg.gradient_checkpointing = False
                    if cfg and hasattr(cfg, "use_cache"):
                        cfg.use_cache = True
                logger.info(f"{indent}✓ Disabled GC flags on {name}")
            except Exception as e:
                logger.warning(f"{indent}✗ Could not disable GC on {name}: {e}")

        def _recursive_disable(model, model_name="model", depth=0):
            if model is None or depth > 6:
                return

            indent = "  " * depth
            logger.info(f"{indent}Disabling GC on {model_name} (depth={depth})")

            # 1. Use official API if available
            if hasattr(model, "gradient_checkpointing_disable"):
                try:
                    model.gradient_checkpointing_disable()
                    logger.info(f"{indent}✓ Called gradient_checkpointing_disable()")
                except Exception as e:
                    logger.warning(f"{indent}✗ gradient_checkpointing_disable failed: {e}")

            # 2. Disable flags (but DON'T replace functions)
            _safe_replace_gc_attr(model, model_name, indent)

            # 3. Recurse into known wrappers
            for attr_name in ["model", "base_model", "transformer", "bert", "roberta"]:
                inner = getattr(model, attr_name, None)
                if inner is not None and inner is not model:
                    _recursive_disable(inner, f"{model_name}.{attr_name}", depth + 1)

        logger.info("=" * 80)
        logger.info("🔧 UNSLOTH-SAFE GRADIENT CHECKPOINTING DISABLE")
        logger.info("=" * 80)

        for model_name, model in [
            ("policy_model", getattr(self, "policy_model", None)),
            ("ref_model", getattr(self, "ref_model", None)),
            ("value_model", getattr(self, "value_model", None)),
            ("reward_model", getattr(self, "reward_model", None)),
        ]:
            if model is not None:
                logger.info(f"\n>>> Processing {model_name}")
                _recursive_disable(model, model_name, 0)

        logger.info("=" * 80)
        logger.info("✅ UNSLOTH-SAFE DISABLE COMPLETE")
        logger.info("=" * 80)

        for model_name, model in [
            ("policy_model", getattr(self, "policy_model", None)),
            ("ref_model", getattr(self, "ref_model", None)),
            ("value_model", getattr(self, "value_model", None)),
            ("reward_model", getattr(self, "reward_model", None)),
        ]:
            if model is not None:
                logger.info(f"\n>>> Processing {model_name}")
                _recursive_disable(model, model_name, 0)

        logger.info("=" * 80)
        logger.info("✅ NUCLEAR DISABLE COMPLETE (SAFE MODE)")
        logger.info("=" * 80)

    # =========================================================================================
    # FORCE DISABLE (used for specific model)
    # =========================================================================================
    def _force_disable_gradient_checkpointing_on_model(self, model, model_name="model"):
        """Forcefully disable gradient checkpointing - UNSLOTH SAFE."""
        if model is None:
            return

        try:
            # 1. Call disable() wherever possible
            for target in [
                model,
                getattr(model, "base_model", None),
                getattr(model, "model", None),
            ]:
                if target and hasattr(target, "gradient_checkpointing_disable"):
                    target.gradient_checkpointing_disable()
                    logger.info(f"✓ {model_name}: called gradient_checkpointing_disable()")

            # 2. Disable config flags
            for candidate in [model, getattr(model, "base_model", None)]:
                cfg = getattr(candidate, "config", None)
                if cfg:
                    if hasattr(cfg, "gradient_checkpointing"):
                        cfg.gradient_checkpointing = False
                    if hasattr(cfg, "use_cache"):
                        cfg.use_cache = True

            # 3. DON'T replace _gradient_checkpointing_func - Unsloth needs it!
            # Just set the flag to False on layers
            inner = getattr(model, "model", model)
            layers = getattr(inner, "layers", None)
            if layers:
                for layer in layers:
                    if hasattr(layer, "gradient_checkpointing"):
                        layer.gradient_checkpointing = False
                logger.info(f"✓ {model_name}: disabled GC flags on {len(layers)} layers")

            logger.info(f"✅ Force-disabled GC on {model_name} (Unsloth-safe)")

        except Exception as e:
            logger.warning(f"⚠️ Error disabling GC on {model_name}: {e}")
    def _patch_transformers_validation(self):
        """Global patch for transformers validation to ignore num_logits_to_keep."""
        try:
            from transformers.generation.utils import GenerationMixin
            
            if getattr(GenerationMixin, "_is_patched_validation", False):
                return

            original_validate = GenerationMixin._validate_model_kwargs
            
            def patched_validate(self, model_kwargs):
                # Strip num_logits_to_keep if present to avoid ValueError
                if 'num_logits_to_keep' in model_kwargs:
                    # logger.info("Global Patch: Stripped num_logits_to_keep from validation")
                    model_kwargs.pop('num_logits_to_keep')
                return original_validate(self, model_kwargs)
            
            GenerationMixin._validate_model_kwargs = patched_validate
            GenerationMixin._is_patched_validation = True
            logger.info("✅ Globally patched transformers._validate_model_kwargs")
        except Exception as e:
            logger.warning(f"⚠️ Failed to patch transformers validation: {e}")

    def _patch_model_forward_for_compatibility(self, model, model_name="model"):
        """Patch model's forward method to ignore incompatible kwargs (like position_ids)."""
        if model is None:
            return
            
        # Determine target model (handle wrappers)
        target_model = model
        if hasattr(model, "model"):
            target_model = model.model
        elif hasattr(model, "base_model"):
            target_model = model.base_model
            
        if not hasattr(target_model, "forward"):
            logger.warning(f"Cannot patch forward: {model_name} has no forward method")
            return

        # Check if model is Decoder (Llama, GPT, etc.) or Encoder (BERT, RoBERTa, DistilBERT)
        # Only patch decoders conditionally; ALWAYS strip position_ids for encoders
        is_decoder = False
        is_encoder = False
        if hasattr(model, 'config'):
            model_type = getattr(model.config, 'model_type', '').lower()
            decoder_models = ['llama', 'gpt', 'mistral', 'gemma', 'qwen', 'opt', 'bloom']
            encoder_models = ['bert', 'roberta', 'distilbert', 'albert', 'electra', 'deberta']
            if any(dec in model_type for dec in decoder_models):
                is_decoder = True
            elif any(enc in model_type for enc in encoder_models):
                is_encoder = True
        
        if not is_decoder and not is_encoder:
            return

        # Patch the forward method of the model itself
        original_forward = target_model.forward
        
        def patched_forward(self, *args, **kwargs):
            # For encoders: ALWAYS strip position_ids and use_cache (they don't support them)
            # For decoders: Only strip position_ids during training/eval (not inference)
            if is_encoder:
                if 'position_ids' in kwargs:
                    kwargs.pop('position_ids')
                if 'use_cache' in kwargs:
                    kwargs.pop('use_cache')
            else:  # is_decoder
                is_inference = kwargs.get('use_cache', False) or 'past_key_values' in kwargs
                if not is_inference and 'position_ids' in kwargs:
                    kwargs.pop('position_ids')
            return original_forward(*args, **kwargs)
        
        import types
        target_model.forward = types.MethodType(patched_forward, target_model)
        logger.info(f"✅ Patched {model_name} forward to conditionally ignore position_ids")
        
        # CRITICAL: Also patch the BACKBONE forward because get_reward calls it directly!
        # get_reward does: lm_backbone = getattr(model, model.base_model_prefix); lm_backbone(...)
        if hasattr(model, "base_model_prefix"):
            backbone_name = model.base_model_prefix
            if hasattr(model, backbone_name):
                backbone = getattr(model, backbone_name)
                if hasattr(backbone, "forward") and backbone is not target_model:
                    original_backbone_forward = backbone.forward
                    
                    def patched_backbone_forward(self, *args, **kwargs):
                        # For encoders: ALWAYS strip position_ids and use_cache
                        # For decoders: Only strip position_ids during training/eval
                        if is_encoder:
                            if 'position_ids' in kwargs:
                                kwargs.pop('position_ids')
                            if 'use_cache' in kwargs:
                                kwargs.pop('use_cache')
                        else:  # is_decoder
                            is_inference = kwargs.get('use_cache', False) or 'past_key_values' in kwargs
                            if not is_inference and 'position_ids' in kwargs:
                                kwargs.pop('position_ids')
                        return original_backbone_forward(*args, **kwargs)
                    
                    backbone.forward = types.MethodType(patched_backbone_forward, backbone)
                    logger.info(f"✅ Patched backbone ({backbone_name}) forward to conditionally ignore position_ids")

    def _patch_model_generate_for_compatibility(self, model, model_name="model"):
        """Patch model's generate method to remove incompatible kwargs."""
        if model is None:
            return

        # Recursive patching for wrappers (PeftModel, etc.)
        if hasattr(model, "base_model") and model.base_model is not model:
             self._patch_model_generate_for_compatibility(model.base_model, f"{model_name}.base_model")
            
        if not hasattr(model, "generate"):
            return

        # Avoid double patching
        if getattr(model.generate, "_is_patched_compatibility", False):
            return

        original_generate = model.generate
        
        def patched_generate(self, *args, **kwargs):
            # Strip num_logits_to_keep if present (causes transformers validation error)
            if 'num_logits_to_keep' in kwargs:
                kwargs.pop('num_logits_to_keep')
            
            # DEBUG: Force keys_to_ignore_at_inference to include num_logits_to_keep
            if hasattr(self, 'config') and hasattr(self.config, 'keys_to_ignore_at_inference'):
                if 'num_logits_to_keep' not in self.config.keys_to_ignore_at_inference:
                    # self.config.keys_to_ignore_at_inference.append('num_logits_to_keep') # Modify in place
                    # Some configs use tuples?
                    if isinstance(self.config.keys_to_ignore_at_inference, tuple):
                        self.config.keys_to_ignore_at_inference = list(self.config.keys_to_ignore_at_inference)
                    self.config.keys_to_ignore_at_inference.append('num_logits_to_keep')
            
            return original_generate(*args, **kwargs)
        
        patched_generate._is_patched_compatibility = True
        
        import types
        model.generate = types.MethodType(patched_generate, model)
        logger.info(f"✅ Patched {model_name} generate to ignore num_logits_to_keep")
        
        # CRITICAL: If model has _old_generate (Unsloth), patch it too!
        # Unsloth's generate might ADD num_logits_to_keep and then call _old_generate
        if hasattr(model, "_old_generate"):
            logger.info(f"Found _old_generate on {model_name}. Patching it.")
            original_old_generate = model._old_generate
            
            def patched_old_generate(self, *args, **kwargs):
                if 'num_logits_to_keep' in kwargs:
                    # logger.info(f"Stripped num_logits_to_keep in _old_generate of {model_name}")
                    kwargs.pop('num_logits_to_keep')
                return original_old_generate(*args, **kwargs)
            
            import types
            model._old_generate = types.MethodType(patched_old_generate, model)
            logger.info(f"✅ Patched {model_name} _old_generate to ignore num_logits_to_keep")
        else:
            logger.info(f"❌ _old_generate NOT found on {model_name} (Type: {type(model).__name__})")

    # =========================================================================================
    # Integration inside setup_model (simplified relevant part)
    # =========================================================================================
    def setup_model(self) -> None:
        """Setup Unsloth-optimized model and tokenizer for PPO."""
        try:
            from unsloth import FastLanguageModel
            # CRITICAL: Clear cache and patch classes BEFORE any model loading
            from .unsloth_patches import clear_all_unsloth_caches, patch_attention_classes_globally
            
            logger.info("Step 1: Clearing Unsloth caches...")
            clear_all_unsloth_caches()
            
            logger.info("Step 2: Patching attention classes at class level...")
            patch_attention_classes_globally()
            
            # Validate model compatibility
            self._validate_model_compatibility()
            
            # Check if user wants to clear cache (additional clearing if requested)
            clear_cache = self._get_config_value(self.config.model, 'clear_unsloth_cache', False)
            if clear_cache:
                logger.info("🗑️  Additional cache clearing requested (clear_unsloth_cache=True)")
                self._clear_unsloth_cache(force=True)
            else:
                # Smart cache invalidation based on config changes
                self._maybe_invalidate_unsloth_cache()
            
            # Get config values
            model_name = self._get_config_value(self.config.model, "name_or_path")
            max_seq_length = self._get_config_value(self.config.model, "max_seq_length", 512)
            quantization = self._get_config_value(self.config.model, "quantization", {})
            
            # Get precision from config to ensure consistency across all models
            precision = self._get_config_value(self.config.model, "precision", "bfloat16")
            if hasattr(precision, 'value'):
                precision = precision.value
            logger.info(f"Using precision: {precision} for all models")

            logger.info(f"Setting up Unsloth PPO model: {model_name}")
            
            # Load tokenizer first (before any model loading)
            from transformers import AutoTokenizer
            from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=False)

            # Tokenizer setup
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if self.tokenizer.chat_template is None:
                logger.info("Applying Unsloth chat template for PPO...")
                from unsloth import FastLanguageModel
                
                # Get chat template from config if available
                chat_template = self._get_config_value(self.config.dataset, 'chat_template', None)
                    
                # Default mapping
                mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
                
                try:
                    self.tokenizer = FastLanguageModel.get_chat_template(
                        self.tokenizer,
                        chat_template=chat_template if chat_template else "llama-3", # Default to llama-3
                        mapping=mapping,
                    )
                    logger.info(f"Applied chat template: {chat_template if chat_template else 'llama-3 (default)'}")
                except Exception as e:
                    logger.warning(f"Failed to apply Unsloth chat template: {e}. Falling back to SIMPLE_CHAT_TEMPLATE.")
                    self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

            logger.info(f"Tokenizer ready (vocab={len(self.tokenizer)})")

            # Check if separate reward model is specified
            reward_model_name = self._get_config_value(
                self.config.model, 
                "reward_model_name", 
                None
            )
            
            try:
              if self.config.model.reward_model_source.source_type=="custom_trained":
                  self.config.model.reward_model_source.training_config.base_model_name =  self._train_and_load_custom_reward_model(self.config.model.reward_model_source.training_config)
            except:
              pass
            
            if reward_model_name:
                # Check if reward model is already loaded and wrapped
                if hasattr(self, 'reward_model') and self.reward_model is not None:
                    logger.info(f"Reward model already loaded: {type(self.reward_model).__name__}")
                    return
                
                # Load specialized reward model (e.g., DeBERTa, RoBERTa, or Llama-based)
                logger.info(f"Loading specialized reward model: {reward_model_name}")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                # Detect if reward model is a decoder (Llama, Mistral, etc.) or encoder (BERT, RoBERTa, etc.)
                reward_model_type = self._detect_model_type(reward_model_name)
                
                # Load reward model with config precision
                torch_dtype = getattr(torch, precision) if precision in ['float16', 'bfloat16', 'float32'] else torch.bfloat16
                
                if reward_model_type == 'unsloth':
                    # Load decoder-based reward model via Unsloth
                    logger.info(f"Loading reward model via Unsloth (decoder model): {reward_model_name}")
                    from unsloth import FastLanguageModel
                    
                    self.reward_model, _ = FastLanguageModel.from_pretrained(
                        model_name=reward_model_name,
                        max_seq_length=max_seq_length,
                        dtype=torch_dtype,
                        load_in_4bit=False,  # Disable quantization for reward model
                    )
                    logger.info(f"✅ Reward model loaded via Unsloth")

                    # Unsloth loads as CausalLM. We need to add a score head if it's missing.
                    import torch.nn as nn
                    if not hasattr(self.reward_model, 'score') and not hasattr(self.reward_model, 'classifier'):
                        logger.info("Adding ad-hoc score head to Unsloth model")
                        hidden_size = self.reward_model.config.hidden_size
                        self.reward_model.score = nn.Linear(hidden_size, 1, bias=False)
                        self.reward_model.score.to(self.reward_model.device).to(torch_dtype)
                        logger.info("✅ Ad-hoc score head added")
                else:
                    # Load encoder-based reward model via standard transformers
                    logger.info(f"Loading reward model via standard transformers (encoder model): {reward_model_name}")
                    self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                        reward_model_name,
                        num_labels=1,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        ignore_mismatched_sizes=True,
                    )
                    logger.info(f"✅ Reward model loaded via standard transformers")
                
                # CRITICAL: Ensure ALL parameters are converted to the correct dtype
                if DEBUG:
                    logger.info(f"🔍 DEBUG: Before conversion - reward model dtype: {next(self.reward_model.parameters()).dtype}")
                
                # Convert the entire model to the target dtype
                self.reward_model = self.reward_model.to(torch_dtype)
                
                # Wrap with UniversalRewardModelWrapper for compatibility
                self.reward_model = UniversalRewardModelWrapper(self.reward_model)
                if DEBUG:
                    logger.info(f"🔍 DEBUG: After .to() conversion - reward model dtype: {next(self.reward_model.parameters()).dtype}")
                
                # Also ensure all submodules are converted
                for name, module in self.reward_model.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        old_dtype = module.weight.data.dtype
                        module.weight.data = module.weight.data.to(torch_dtype)
                        new_dtype = module.weight.data.dtype
                        if DEBUG and old_dtype != new_dtype:
                            logger.info(f"🔍 DEBUG: Converted {name}.weight from {old_dtype} to {new_dtype}")
                        elif DEBUG:
                            logger.info(f"🔍 DEBUG: {name}.weight already {new_dtype}")
                    if hasattr(module, 'bias') and module.bias is not None:
                        old_dtype = module.bias.data.dtype
                        module.bias.data = module.bias.data.to(torch_dtype)
                        new_dtype = module.bias.data.dtype
                        if DEBUG and old_dtype != new_dtype:
                            logger.info(f"🔍 DEBUG: Converted {name}.bias from {old_dtype} to {new_dtype}")
                        elif DEBUG:
                            logger.info(f"🔍 DEBUG: {name}.bias already {new_dtype}")
                
                # CRITICAL: Check the classification head specifically
                if DEBUG:
                    if hasattr(self.reward_model, 'classifier'):
                        logger.info(f"🔍 DEBUG: Classifier weight dtype: {self.reward_model.classifier.weight.dtype}")
                        logger.info(f"🔍 DEBUG: Classifier bias dtype: {self.reward_model.classifier.bias.dtype if self.reward_model.classifier.bias is not None else 'None'}")
                    if hasattr(self.reward_model, 'score'):
                        logger.info(f"🔍 DEBUG: Score weight dtype: {self.reward_model.score.weight.dtype}")
                        logger.info(f"🔍 DEBUG: Score bias dtype: {self.reward_model.score.bias.dtype if self.reward_model.score.bias is not None else 'None'}")
                    
                    # Final verification
                    final_dtype = next(self.reward_model.parameters()).dtype
                    logger.info(f"🔍 DEBUG: Final reward model dtype: {final_dtype}")
                    logger.info(f"🔍 DEBUG: Target dtype was: {torch_dtype}")
                logger.info(f"✅ Reward model loaded with {precision} precision")
                
                # CRITICAL: Check if classifier is actually in the right dtype
                # Handle different model architectures for head layer
                head_layer = None
                if hasattr(self.reward_model, 'classifier'):
                    head_layer = self.reward_model.classifier
                    if DEBUG:
                        logger.info(f"🔍 DEBUG: Using classifier - weight dtype: {head_layer.weight.dtype}")
                elif hasattr(self.reward_model, 'score'):
                    head_layer = self.reward_model.score
                    if DEBUG:
                        logger.info(f"🔍 DEBUG: Using score - weight dtype: {head_layer.weight.dtype}")
                else:
                    raise AttributeError(f"Model {type(self.reward_model)} has no classifier or score layer")
                
                # CRITICAL: Ensure head layer is in the right dtype
                try:
                    # Use .to() which handles recursion for complex heads (like RoBERTa)
                    head_layer.to(torch_dtype)
                    logger.info(f"✅ Converted head layer to {torch_dtype}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not convert head layer to {torch_dtype}: {e}")
                
                # Load reward tokenizer
                self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
                
                # Load value model using same model as policy
                reward_value_model = self._get_config_value(
                    self.config.model, 
                    "reward_value_model", 
                    model_name  # Use policy model for value model
                )
                if reward_value_model is None:
                    reward_value_model = model_name
                logger.info(f"Value model: {reward_value_model}")
                
                model_type = self._detect_model_type(reward_value_model)
                if model_type == 'unsloth':
                    self._load_value_model_unsloth(reward_value_model, max_seq_length, quantization)
                else:
                    self._load_value_model_standard(reward_value_model)
                
                # DEBUG: Check value model dtype
                if DEBUG and hasattr(self, 'value_model') and self.value_model is not None:
                    try:
                        value_dtype = next(self.value_model.parameters()).dtype
                        logger.info(f"🔍 DEBUG: Value model dtype: {value_dtype}")
                    except Exception as e:
                        logger.warning(f"🔍 DEBUG: Could not get value model dtype: {e}")
            else:
                # Original behavior: use same model for reward/value
                reward_value_model = self._get_config_value(
                    self.config.model, 
                    "reward_value_model", 
                    None # Default to None, we will set to model_name if missing
                )
                
                # Fallback to policy model if not specified
                if reward_value_model is None:
                    reward_value_model = model_name
                    
                logger.info(f"Reward/value model: {reward_value_model}")

                model_type = self._detect_model_type(reward_value_model)

                if model_type == 'unsloth':
                    self._load_reward_value_models_unsloth(
                    reward_value_model, 
                        max_seq_length, 
                        quantization
                    )
                else:
                    self._load_reward_value_models_standard(reward_value_model)

            # NOW build policy/ref models (Unsloth already imported at top)
            logger.info("Building policy/ref models with Unsloth...")

            # Load Unsloth policy model
            self.unsloth_model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,  # FIXED: Explicitly set dtype to bfloat16
                load_in_4bit=quantization.get("load_in_4bit", True),
            )

            # CRITICAL FIX: Apply LoRA with use_gradient_checkpointing=False
            logger.info("Applying LoRA WITHOUT gradient checkpointing...")
            self.policy_model = FastLanguageModel.get_peft_model(
                self.unsloth_model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,  # CHANGED: Set to False directly
                random_state=3407,
                use_rslora=False,
            )

            # Using class-level patches applied before model loading
            from .unsloth_patches import verify_attention_patches
            logger.info("Using class-level patches applied before model loading")
            verify_attention_patches(self.policy_model)
            
            # FIX: Add num_logits_to_keep to ignored keys to prevent generation error
            # Apply to all reachable configs to be safe
            models_to_fix = [self.policy_model, self.ref_model]
            if hasattr(self.policy_model, "base_model"): models_to_fix.append(self.policy_model.base_model)
            if hasattr(self.policy_model, "model"): models_to_fix.append(self.policy_model.model)
            
            for m in models_to_fix:
                if hasattr(m, "config"):
                    if not hasattr(m.config, "keys_to_ignore_at_inference"):
                        m.config.keys_to_ignore_at_inference = []
                    if "num_logits_to_keep" not in m.config.keys_to_ignore_at_inference:
                        m.config.keys_to_ignore_at_inference.append("num_logits_to_keep")
                        logger.info(f"✅ Added num_logits_to_keep to keys_to_ignore_at_inference for {type(m).__name__}")
            
            logger.info("✅ Policy model ready")

            # DEBUG: Check policy model dtype
            if DEBUG:
                try:
                    policy_dtype = next(self.policy_model.parameters()).dtype
                    logger.info(f"🔍 DEBUG: Policy model dtype: {policy_dtype}")
                except Exception as e:
                    logger.warning(f"🔍 DEBUG: Could not get policy model dtype: {e}")

            # Reference model (no quantization, no GC) - USE SAME DTYPE AS POLICY
            logger.info("Loading reference model...")
            self.ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16,  # FIXED: Use bfloat16 to match policy model
                load_in_4bit=False,
            )
            
            for p in self.ref_model.parameters():
                p.requires_grad = False
            
            # Using class-level patches applied before model loading
            logger.info("Using class-level patches applied before model loading")
            verify_attention_patches(self.ref_model)

            logger.info("✅ Reference model ready")
            
            # DEBUG: Check ref model dtype
            if DEBUG:
                try:
                    ref_dtype = next(self.ref_model.parameters()).dtype
                    logger.info(f"🔍 DEBUG: Reference model dtype: {ref_dtype}")
                except Exception as e:
                    logger.warning(f"🔍 DEBUG: Could not get ref model dtype: {e}")
            
            logger.info("✅ All models loaded successfully")
            
            # Log model architecture
            logger.info("=" * 60)
            logger.info("Model Architecture:")
            logger.info(f"  Policy:    {type(self.policy_model).__name__} (Unsloth + LoRA)")
            logger.info(f"  Reference: {type(self.ref_model).__name__} (Unsloth frozen)")
            if hasattr(self, 'reward_model') and self.reward_model is not None:
                logger.info(f"  Reward:    {type(self.reward_model).__name__} (Standard HF)")
            if hasattr(self, 'value_model') and self.value_model is not None:
                logger.info(f"  Value:     {type(self.value_model).__name__} (Standard HF)")
            logger.info("=" * 60)
            
            # Load reward model if specified
            self._load_reward_model_strict()

            # CRITICAL: Patch models to ignore position_ids (for compatibility with TRL get_reward)
            # Apply to ALL models including reward model (especially important for encoder models like BERT/RoBERTa/DistilBERT)
            self._patch_model_forward_for_compatibility(self.policy_model, "policy_model")
            self._patch_model_forward_for_compatibility(self.ref_model, "ref_model")
            self._patch_model_forward_for_compatibility(self.value_model, "value_model")
            
            # CRITICAL FIX: Patch reward model attention layers with apply_qkv from policy model
            # This is necessary when loading multiple Unsloth models, as the second instance might miss the patch
            if hasattr(self, 'reward_model') and self.reward_model is not None and hasattr(self, 'policy_model') and self.policy_model is not None:
                try:
                    policy_attn = None
                    for module in self.policy_model.modules():
                        if 'Attention' in module.__class__.__name__:
                            policy_attn = module
                            break
                    
                    if policy_attn is not None and hasattr(policy_attn, 'apply_qkv'):
                        logger.info("Patching reward model attention layers with Unsloth methods from policy model...")
                        count = 0
                        for module in self.reward_model.modules():
                            if 'Attention' in module.__class__.__name__:
                                patched_module = False
                                # Copy all apply_* methods (apply_qkv, apply_o, etc.)
                                for attr_name in dir(policy_attn):
                                    if attr_name.startswith('apply_'):
                                        if not hasattr(module, attr_name):
                                            setattr(module, attr_name, getattr(policy_attn, attr_name))
                                            patched_module = True
                                if patched_module:
                                    count += 1
                        if count > 0:
                            logger.info(f"✅ Patched {count} attention layers in reward model")
                    else:
                        logger.debug("Policy model has no apply_qkv to copy")
                except Exception as e:
                    logger.warning(f"Failed to patch apply_qkv on reward model: {e}")

            if hasattr(self, 'reward_model') and self.reward_model is not None:
                # For wrapped reward models, patch the underlying model
                if hasattr(self.reward_model, '_model'):
                    self._patch_model_forward_for_compatibility(self.reward_model._model, "reward_model._model")
                else:
                    self._patch_model_forward_for_compatibility(self.reward_model, "reward_model")
            
            # CRITICAL: Patch generate to ignore num_logits_to_keep
            self._patch_model_generate_for_compatibility(self.policy_model, "policy_model")
            self._patch_model_generate_for_compatibility(self.ref_model, "ref_model") # Although ref_model usually doesn't generate
            
            # GLOBAL: Patch transformers validation as a last resort
            self._patch_transformers_validation()

        except Exception as e:
            logger.error(f"❌ Failed to setup model: {e}")
            raise
    
    def _load_reward_model_strict(self):
        """Load reward model with strict validation, NO fallbacks."""
        if not hasattr(self.config.model, 'reward_model_source'):
            logger.info("No reward_model_source in config, skipping custom reward model")
            return
        
        if not self.config.model.reward_model_source:
            logger.info("reward_model_source is None, skipping custom reward model")
            return
        
        reward_source = self.config.model.reward_model_source
        
        # Validate source before loading
        from aligntune.rewards.training import RewardModelValidator
        RewardModelValidator.validate_reward_source(reward_source)
        
        logger.info(f"Loading reward model: source_type={reward_source.source_type}")
        
        try:
            if reward_source.source_type == "pretrained_hf":
                self.reward_model = self._load_pretrained_hf_reward_model(reward_source.model_name)
            elif reward_source.source_type == "pretrained_local":
                self.reward_model = self._load_local_reward_model(reward_source.model_path)
            elif reward_source.source_type == "custom_trained":
                # self.reward_model = self._train_and_load_custom_reward_model(reward_source.training_config)
                self.reward_model =  self._load_local_reward_model(reward_source.training_config.base_model_name)
            else:
                raise ValueError(f"Invalid source_type: {reward_source.source_type}")
            
            # CRITICAL: Wrap with UniversalRewardModelWrapper for TRL compatibility
            if not isinstance(self.reward_model, UniversalRewardModelWrapper):
                self.reward_model = UniversalRewardModelWrapper(self.reward_model)
                
                # CRITICAL: Add score method directly to the model for TRL compatibility
                # TRL's get_reward might bypass the wrapper and call model.score() directly
                base_model = self.reward_model._model
                if not hasattr(base_model, 'score'):
                    if hasattr(base_model, 'classifier'):
                        def score_method(hidden_states):
                            return base_model.classifier(hidden_states)
                        base_model.score = score_method
                        logger.info("✅ Added score method to base reward model")
                    elif hasattr(base_model, 'score'):
                        pass  # Already has it
                    else:
                        logger.warning("⚠️ Reward model has no classifier or score layer")
                
                # Verify wrapper score method exists
                if not hasattr(self.reward_model, 'score'):
                    logger.error("Reward model wrapper missing score method!")
                    if hasattr(base_model, 'classifier'):
                        def score_method(hidden_states):
                            return base_model.classifier(hidden_states)
                        self.reward_model.score = score_method
                        logger.info("Added score method to reward model wrapper")
                    else:
                        raise AttributeError("Reward model must have score() or classifier() method")
                else:
                    logger.info("✅ Reward model wrapper has score method")
            
            logger.info("✅ Reward model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load reward model: {e}")
            raise RuntimeError(f"Reward model loading failed: {e}") from e

    def _load_pretrained_hf_reward_model(self, model_name: str):
        """Load pre-trained reward model from HuggingFace Hub."""
        from aligntune.rewards.training import RewardModelLoader
        
        logger.info(f"Loading HF reward model: {model_name}")
        loader = RewardModelLoader()
        # Use ignore_mismatched_sizes to handle models trained with different num_labels
        return loader.load_from_huggingface(model_name, ignore_mismatched_sizes=True)

    def _load_local_reward_model(self, model_path: str):
        """Load reward model from local path."""
        from aligntune.rewards.training import RewardModelLoader
        
        logger.info(f"Loading local reward model: {model_path}")
        loader = RewardModelLoader()
        return loader.load_from_local(model_path)

    def _train_and_load_custom_reward_model(self, training_config):
        """Train custom reward model from reward functions."""
        from aligntune.rewards.training import RewardModelTrainer
        from aligntune.rewards.registry import RewardRegistry
        
        logger.info("Training custom reward model from reward functions")
        logger.info(f"Reward functions: {training_config.reward_functions}")
        logger.info(f"Training texts: {len(training_config.training_texts)} samples")
        
        # Get reward functions from registry
        reward_funcs = [
            RewardRegistry.get_reward_function(name) 
            for name in training_config.reward_functions
        ]
        
        # Create trainer
        trainer = RewardModelTrainer(
            base_model_name=training_config.base_model_name,
            reward_functions=reward_funcs,
            composite_weights=training_config.reward_weights
        )
        
        # Generate training data
        training_data = trainer.generate_training_data(
            texts=training_config.training_texts,
            references=training_config.reference_texts,
            batch_size=training_config.batch_size
        )
        
        # Train model
        model_path = trainer.train_reward_model(
            training_data=training_data,
            output_dir=training_config.output_dir,
            num_epochs=training_config.num_epochs,
            learning_rate=training_config.learning_rate,
            batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps
        )
        
        logger.info(f"✅ Custom reward model trained: {model_path}")
        
        # Load trained model
        # return self._load_local_reward_model(model_path)
        return model_path

    def setup_rewards(self) -> None:
        """Setup reward functions based on configuration mode."""
        logger.info("Setting up Unsloth PPO reward functions...")
        
        # Determine mode
        has_pretrained = hasattr(self, 'reward_model') and self.reward_model is not None
        has_reward_funcs = self.config.rewards and len(self.config.rewards) > 0
        fine_tune_mode = (
            has_pretrained and 
            has_reward_funcs and 
            hasattr(self.config.model, 'reward_model_source') and
            self.config.model.reward_model_source and
            getattr(self.config.model.reward_model_source, 'fine_tune_with_rewards', False)
        )
        
        # Check if reward training is enabled
        should_train_model = (
            hasattr(self.config, 'reward_training') and 
            self.config.reward_training and 
            self.config.reward_training.enabled
        )
        
        # Mode 1: Pretrained only
        if has_pretrained and not fine_tune_mode:
            logger.info("✅ Mode: Pretrained reward model (using as-is)")
            logger.info("   Reward functions will be ignored")
            self.reward_functions = []
            return
        
        # Mode 2: Custom trained (handled in _load_reward_model_strict)
        if not has_pretrained and has_reward_funcs:
            if should_train_model:
                logger.info("✅ Mode: Custom reward model training")
                logger.info("   Will train new model using reward functions")
                # Setup reward functions for training
                self._setup_reward_functions()
                return
            else:
                # NEW: Use reward functions directly with FunctionBasedRewardModel
                logger.info("✅ Mode: Direct reward function usage")
                logger.info("   Reward functions will be wrapped in FunctionBasedRewardModel")
                self._setup_reward_functions()
                return
        
        # Mode 3: Hybrid (fine-tune)
        if fine_tune_mode:
            logger.info("✅ Mode: Hybrid (fine-tune pretrained with reward functions)")
            self._setup_reward_functions()
            self._fine_tune_reward_model()
            return
        
        # No rewards configured
        logger.info("No reward model or functions configured")
        self.reward_functions = []
    
    def _setup_reward_functions(self):
        """Internal method to setup reward functions."""
        self.reward_functions = []
        
        for reward_config in self.config.rewards:
            reward_type = self._get_config_value(reward_config, 'type')
            reward_params = self._get_config_value(reward_config, 'params', {})
            reward_weight = self._get_config_value(reward_config, 'weight', 1.0)
            reward_clip = self._get_config_value(reward_config, 'clip', None)
            
            # Get reward function from registry
            reward_func = RewardRegistry.get_reward_function(reward_type)
            
            # Apply weight
            if reward_weight != 1.0:
                def weighted_reward(text, weight=reward_weight, func=reward_func, **kwargs):
                    return weight * func(text, **kwargs)
                reward_func = weighted_reward
            
            # Apply clipping
            if reward_clip is not None:
                def clipped_reward(text, clip=reward_clip, func=reward_func, **kwargs):
                    return max(-clip, min(clip, func(text, **kwargs)))
                reward_func = clipped_reward
            
            self.reward_functions.append(reward_func)
        
        logger.info(f"Configured {len(self.reward_functions)} reward functions")
    
    def _fine_tune_reward_model(self):
        """Fine-tune pretrained reward model with reward functions."""
        logger.info("Fine-tuning pretrained reward model...")
        # Use reward functions to generate additional training data
        # Fine-tune the loaded model
        # This is optional advanced feature
        logger.warning("⚠️  Fine-tuning not yet implemented, using pretrained as-is")
    
    def _compute_rewards(self, texts: List[str], references: Optional[List[str]] = None) -> List[float]:
        """Compute rewards for given texts using configured reward functions."""
        if not self.reward_functions:
            logger.warning("No reward functions configured, returning zero rewards")
            return [0.0] * len(texts)
        
        rewards = []
        for text in texts:
            total_reward = 0.0
            for reward_func in self.reward_functions:
                try:
                    reward = reward_func(text, reference=references[texts.index(text)] if references else None)
                    total_reward += reward
                except Exception as e:
                    logger.warning(f"Error computing reward for text: {e}")
                    total_reward += 0.0
            
            rewards.append(total_reward)
        
        return rewards
    
    def _detect_dataset_format(self, sample):
        """Detect dataset format."""
        if "messages" in sample:
            return "chat_format"
        elif "chosen" in sample and "rejected" in sample:
            return "hh_rlhf_with_prompt" if "prompt" in sample else "hh_rlhf_no_prompt"
        elif "instruction" in sample:
            return "alpaca_format"
        elif "text" in sample:
            return "simple_text"
        elif "input" in sample and "output" in sample:
            return "input_output"
        else:
            return "unknown"
    
    def _parse_element_by_format(self, element, format_type, tokenizer):
        """Parse dataset element - EXACT SAME AS WORKING CODE."""
        if format_type == "chat_format":
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
                tokenize=True,
            )
        elif format_type == "hh_rlhf_with_prompt":
            query_text = element["prompt"].strip()
            messages = [{"role": "user", "content": query_text}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                tokenize=True,
            )
        elif format_type == "hh_rlhf_no_prompt":
            chosen_text = element["chosen"].strip()
            
            if "\n\nHuman:" in chosen_text and "\n\nAssistant:" in chosen_text:
                parts = chosen_text.split("\n\nAssistant:")
                human_part = parts[0].replace("\n\nHuman:", "").strip() if len(parts) > 1 else chosen_text.split("\n\nHuman:")[-1].strip()
            elif "Human:" in chosen_text and "Assistant:" in chosen_text:
                parts = chosen_text.split("Assistant:")
                human_part = parts[0].replace("Human:", "").strip() if len(parts) > 1 else chosen_text.split("Human:")[-1].strip()
            else:
                lines = chosen_text.split('\n')
                human_part = lines[0].strip() if lines else chosen_text[:100].strip()
            
            if len(human_part) < 10:
                human_part = chosen_text[:200].strip()
            
            input_ids = tokenizer.encode(human_part, add_special_tokens=True, return_tensors=None)
            
            if len(input_ids) < 5:
                padding_ids = tokenizer.encode(" What do you think about this?", add_special_tokens=False)
                input_ids.extend(padding_ids)
        elif format_type == "alpaca_format":
            instruction = element["instruction"].strip()
            if "input" in element and element["input"] and element["input"].strip():
                query_text = f"{instruction}\n\nInput: {element['input'].strip()}"
            else:
                query_text = instruction
            messages = [{"role": "user", "content": query_text}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                tokenize=True,
            )
        elif format_type == "simple_text":
            query_text = element["text"].strip()
            messages = [{"role": "user", "content": query_text}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                tokenize=True,
            )
        elif format_type == "input_output":
            query_text = element["input"].strip()
            messages = [{"role": "user", "content": query_text}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            raise ValueError(f"Unknown format: {format_type}")
        
        # Remove trailing EOS
        try:
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and input_ids and input_ids[-1] == eos_id:
                input_ids = input_ids[:-1]
        except Exception:
            pass
        
        # Ensure minimum length
        if len(input_ids) < 5:
            padding_ids = tokenizer.encode(" Please elaborate.", add_special_tokens=False)
            input_ids.extend(padding_ids)
        
        return input_ids
    
    def setup_data(self) -> None:
        """Setup datasets for PPO training using unified DataManager."""
        logger.info("Setting up PPO datasets with DataManager...")
        
        # Extract dataset configuration
        dataset_config = None
        if hasattr(self.config, 'dataset') and self.config.dataset is not None:
            dataset_config = self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            dataset_config = self.config.datasets[0]
            if len(self.config.datasets) > 1:
                logger.warning(f"Multiple datasets provided, using first one")
        else:
            raise ValueError("No dataset configuration found")
        
        # Extract parameters
        dataset_name = self._get_config_value(dataset_config, 'name', default='imdb')
        split = self._get_config_value(dataset_config, 'split', default='train')
        config_name = self._get_config_value(dataset_config, 'config_name', default=None)
        system_prompt = self._get_config_value(dataset_config, 'system_prompt', default=None)
        enable_thinking = self._get_config_value(self.config.train, 'enable_thinking', default=False)
        
        # Advanced DataManager features
        column_mapping = self._get_config_value(dataset_config, 'column_mapping', default=None)
        processing_fn = self._get_config_value(dataset_config, 'processing_fn', default=None)
        processing_batched = self._get_config_value(dataset_config, 'processing_batched', default=False)
        max_samples = self._get_config_value(dataset_config, 'max_samples', default=None)
        percent = self._get_config_value(dataset_config, 'percent', default=None)
        
        logger.info(f"Loading dataset: {dataset_name} (split: {split}, config: {config_name})")
        
        # Initialize DataManager for PPO task
        from aligntune.data.manager import DataManager
        
        manager = DataManager(
            task_type="sft",
            system_prompt=system_prompt,
            tokenizer=self.tokenizer,
            enable_thinking=enable_thinking,
            column_mapping=column_mapping,
            processing_fn=processing_fn,
            max_samples = max_samples, 
            processing_batched=processing_batched
        )
        
        # Load dataset - DataManager handles everything
        dataset_dict = manager.load_dataset(
            dataset_name,
            config_name=config_name,
            split=split,
        )
        
        max_eval_samples =  self._get_config_value(dataset_config, 'max_eval_samples', default=None)
        # Extract train and validation splits
        self.train_dataset = dataset_dict.get("train", None)
       

        self.eval_dataset = dataset_dict.get("validation", None)

        self.dataset_dict = dataset_dict

            
        
        
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train examples")
        if self.eval_dataset:
            logger.info(f"Evaluation dataset: {len(self.eval_dataset)} examples")
        
        # Tokenize dataset for PPO
        max_prompt_length = self._get_config_value(self.config.train, 'max_prompt_length', default=512)
        
        def tokenize_function(examples):
            """Tokenize prompts for PPO training."""
            prompts = examples.get("prompt", examples.get("query", []))
            
            tokenized = self.tokenizer(
                prompts,
                padding=False,
                truncation=True,
                max_length=max_prompt_length,
            )
            input_ids = tokenized["input_ids"]
            lengths = [len(ids) for ids in input_ids]
            
            return {"input_ids": input_ids, "lengths": lengths}
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.eval_dataset.column_names,
                desc="Tokenizing eval dataset",
            )
        
        # Filter by length
        logger.info(f"Filtering sequences longer than {max_prompt_length} tokens...")
        self.train_dataset = self.train_dataset.filter(
            lambda x: x["lengths"] <= max_prompt_length,
            desc="Filtering train dataset",
        )
        
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.filter(
                lambda x: x["lengths"] <= max_prompt_length,
                desc="Filtering eval dataset",
            )
        
        logger.info(f"Final dataset sizes - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset) if self.eval_dataset else 0}")
        
        # Verify dataset structure
        assert self.train_dataset[0]["input_ids"][-1] != self.tokenizer.eos_token_id, \
            "The last token should not be an EOS token"
        
        # Log sample
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            decoded = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
            logger.info(f"Sample tokenized prompt (first 100 chars): {decoded[:100]}...")
            logger.info(f"Dataset columns: {self.train_dataset.column_names}")  

    def setup_trainer(self) -> None:
        """Setup TRL PPOTrainer."""
        # Set global reference for patched_get_reward
        global GLOBAL_UNSLOTH_TRAINER_REF
        GLOBAL_UNSLOTH_TRAINER_REF = self
        
        try:
            from trl import PPOTrainer, PPOConfig
            
            logger.info("Setting up PPOTrainer")
            
            # Logging configuration with defaults
            output_dir = getattr(self.config.logging, 'output_dir', './output/ppo')
            run_name = getattr(self.config.logging, 'run_name', None) or 'ppo_experiment'
            report_to = getattr(self.config.logging, 'loggers', 'none')
            
            # Training parameters from config with defaults
            num_epochs = getattr(self.config.train, 'epochs', None) or 1
            batch_size = getattr(self.config.train, 'per_device_batch_size', 1)
            grad_accum = getattr(self.config.train, 'gradient_accumulation_steps', 32)
            lr = getattr(self.config.train, 'learning_rate', 1e-6)
            kl_coef = getattr(self.config.train, 'kl_coef', 0.05)
            cliprange = getattr(self.config.train, 'cliprange', 0.2)
            cliprange_value = getattr(self.config.train, 'cliprange_value', 0.2)
            vf_coef = getattr(self.config.train, 'vf_coef', 0.1)
            gamma = getattr(self.config.train, 'gamma', 1.0)
            lam = getattr(self.config.train, 'lam', 0.95)
            max_grad_norm = getattr(self.config.train, 'max_grad_norm', 0.5)
            num_ppo_epochs = getattr(self.config.train, 'num_ppo_epochs', None) or 4
            whiten_rewards = getattr(self.config.train, 'whiten_rewards', False)
            response_length = getattr(self.config.train, 'response_length', 53)
            stop_token = getattr(self.config.train, 'stop_token', 'eos')
            missing_eos_penalty = getattr(self.config.train, 'missing_eos_penalty', 1.0)
            save_steps = getattr(self.config.train, 'save_steps', 500)
            save_strategy = getattr(self.config.train, 'save_strategy', 'steps')
            save_total_limit = getattr(self.config.train, 'save_total_limit', 5)
            eval_strategy = getattr(self.config.train, 'eval_strategy', 'steps')
            eval_steps = getattr(self.config.train, 'eval_steps', None) or 100
            logging_steps = getattr(self.config.train, 'logging_steps', 10)
            gradient_checkpointing = getattr(self.config.model, 'gradient_checkpointing', False)
            gradient_checkpointing_kwargs = getattr(self.config.train, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
            
            # Seed with default
            seed = getattr(self.config.distributed, 'seed', 42)
            
            # Precision handling - direct string comparison with default
            precision_str = getattr(self.config.model, 'precision', 'auto')
            if hasattr(precision_str, 'value'):  # Handle enum
                precision_str = precision_str.value
            
            bf16 = precision_str in ['bf16', 'auto']
            fp16 = precision_str == 'fp16'
            
            # Calculate total episodes
            total_episodes = len(self.train_dataset) * num_epochs
            
            # Create PPOConfig
            ppo_config = PPOConfig(
                exp_name=run_name,
                learning_rate=lr,
                batch_size=batch_size,
                mini_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                total_episodes=total_episodes,
                num_ppo_epochs=num_ppo_epochs,
                max_grad_norm=max_grad_norm,
                seed=seed,
                cliprange=cliprange,
                cliprange_value=cliprange_value,
                vf_coef=vf_coef,
                kl_coef=kl_coef,
                whiten_rewards=whiten_rewards,
                gamma=gamma,
                lam=lam,
                response_length=response_length,
                stop_token=stop_token,
                missing_eos_penalty=missing_eos_penalty,
                local_rollout_forward_batch_size=batch_size,
                output_dir=output_dir,
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                bf16=bf16,
                fp16=fp16,
                remove_unused_columns=False,
                run_name=run_name,
                gradient_checkpointing=False,
                report_to=report_to,
            )
            
            missing = extract_extra_and_missing_params(
                backend_config=ppo_config,
                config=self.config,
                algorithm='ppo'
            )

            for key, value in missing.items():
                setattr(ppo_config, key, value)
            
            
            # DEBUG: Final dtype check before creating trainer
            if DEBUG:
                logger.info("🔍 DEBUG: Final dtype check before PPOTrainer creation:")
                try:
                    if hasattr(self, 'reward_model') and self.reward_model is not None:
                        reward_dtype = next(self.reward_model.parameters()).dtype
                        logger.info(f"🔍 DEBUG: Reward model final dtype: {reward_dtype}")
                    if hasattr(self, 'value_model') and self.value_model is not None:
                        value_dtype = next(self.value_model.parameters()).dtype
                        logger.info(f"🔍 DEBUG: Value model final dtype: {value_dtype}")
                    if hasattr(self, 'policy_model') and self.policy_model is not None:
                        policy_dtype = next(self.policy_model.parameters()).dtype
                        logger.info(f"🔍 DEBUG: Policy model final dtype: {policy_dtype}")
                    
                    # CRITICAL: Test a forward pass to check hidden states dtype
                    logger.info("🔍 DEBUG: Testing forward pass dtypes...")
                    if hasattr(self, 'policy_model') and self.policy_model is not None:
                        test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
                        with torch.no_grad():
                            try:
                                outputs = self.policy_model(test_input)
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                    hidden_dtype = outputs.hidden_states[-1].dtype
                                    logger.info(f"🔍 DEBUG: Policy model hidden states dtype: {hidden_dtype}")
                                else:
                                    logger.info(f"🔍 DEBUG: Policy model outputs type: {type(outputs)}")
                            except Exception as e:
                                logger.warning(f"🔍 DEBUG: Could not test policy model forward pass: {e}")
                                
                except Exception as e:
                    logger.warning(f"🔍 DEBUG: Could not get final dtypes: {e}")
            
            # Check if we should use function-based reward model
            if not hasattr(self, 'reward_model') or self.reward_model is None:
                if hasattr(self, 'reward_functions') and self.reward_functions and len(self.reward_functions) > 0:
                    # Use function-based reward model
                    from aligntune.core.rl.function_based_reward_model import FunctionBasedRewardModel
                    
                    logger.info("=" * 60)
                    logger.info("✅ Using function-based reward model with reward functions")
                    logger.info(f"   {len(self.reward_functions)} reward function(s) configured")
                    logger.info("=" * 60)
                    
                    # Get device and dtype from policy model
                    device = next(self.policy_model.parameters()).device
                    dtype = next(self.policy_model.parameters()).dtype
                    
                    self.reward_model = FunctionBasedRewardModel(
                        reward_functions=self.reward_functions,
                        tokenizer=self.tokenizer,
                        device=str(device),
                        dtype=dtype
                    )
                    self.reward_model = self.reward_model.to(device)
                    
                    logger.info("✅ FunctionBasedRewardModel created and moved to device")
                else:
                    raise ValueError(
                        "TRL PPOTrainer requires either:\n"
                        "  - reward_model: A neural network reward model (pretrained or custom trained)\n"
                        "    Examples: 'OpenAssistant/reward-model-deberta-v3-large-v2', 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'\n"
                        "  - reward_functions: Rule-based reward functions (will be wrapped in FunctionBasedRewardModel)\n"
                        "    Example: [{'type': 'length', 'weight': 1.0}, {'type': 'sentiment', 'weight': 0.5}]\n"
                        "  - reward_training.enabled=True: Train a custom reward model from reward functions\n"
                        "Please provide one of these options."
                    )
            else:
                # Neural reward model provided
                logger.info("✅ Using neural network reward model")
                logger.info(f"   Reward model type: {type(self.reward_model).__name__}")
            
            # Create trainer with strict generation limits
            self.trainer = PPOTrainer(
                args=ppo_config,
                processing_class=self.tokenizer,
                model=self.policy_model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=None,
            )
            
            # Patch PolicyAndValueWrapper (keep this)
            self._patch_policy_wrapper()
            
            logger.info("PPOTrainer created successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise

    def _patch_policy_wrapper(self):
        """Patch PolicyAndValueWrapper to add missing methods for Unsloth compatibility."""
        try:
            wrapper = getattr(self.trainer, 'model', None)
            if wrapper is None:
                return
            
            # Add gradient_checkpointing_disable method if missing
            if not hasattr(wrapper, 'gradient_checkpointing_disable'):
                def _disable_gc():
                    """Disable gradient checkpointing on underlying models."""
                    try:
                        policy = getattr(wrapper, 'policy_model', None)
                        if policy and hasattr(policy, 'gradient_checkpointing_disable'):
                            policy.gradient_checkpointing_disable()
                        base = getattr(wrapper, 'model', None)
                        if base and hasattr(base, 'gradient_checkpointing_disable'):
                            base.gradient_checkpointing_disable()
                    except Exception:
                        pass
                
                setattr(wrapper, 'gradient_checkpointing_disable', _disable_gc)
                logger.info("Added gradient_checkpointing_disable() to PolicyAndValueWrapper")
            
            # Add gradient_checkpointing_enable method if missing
            if not hasattr(wrapper, 'gradient_checkpointing_enable'):
                def _enable_gc(gradient_checkpointing_kwargs=None):
                    """Enable gradient checkpointing on underlying models."""
                    try:
                        policy = getattr(wrapper, 'policy_model', None)
                        if policy and hasattr(policy, 'gradient_checkpointing_enable'):
                            if gradient_checkpointing_kwargs:
                                policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                            else:
                                policy.gradient_checkpointing_enable()
                        base = getattr(wrapper, 'model', None)
                        if base and hasattr(base, 'gradient_checkpointing_enable'):
                            if gradient_checkpointing_kwargs:
                                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                            else:
                                base.gradient_checkpointing_enable()
                    except Exception:
                        pass
                
                setattr(wrapper, 'gradient_checkpointing_enable', _enable_gc)
                logger.info("Added gradient_checkpointing_enable() to PolicyAndValueWrapper")
            
            # Add generate method if missing (for Unsloth)
            if not hasattr(wrapper, 'generate'):
                policy = getattr(wrapper, 'policy_model', None)
                base = getattr(wrapper, 'model', None)
                
                if policy and hasattr(policy, 'generate'):
                    target = policy.generate
                elif base and hasattr(base, 'generate'):
                    target = base.generate
                else:
                    return
                
                def _delegate_generate(*args, **kwargs):
                    return target(*args, **kwargs)
                
                setattr(wrapper, 'generate', _delegate_generate)
                logger.info("Added generate() to PolicyAndValueWrapper")
                
        except Exception as e:
            logger.warning(f"Could not patch PolicyAndValueWrapper: {e}")
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Not used - TRL handles training loop."""
        return {"loss": 0.0}
    
    def create_data_loader(self):
        """Not used - TRL handles data loading."""
        return None
    
    def train(self) -> Dict[str, Any]:
        """Execute PPO training."""
        try:
            logger.info("Starting Unsloth PPO training")
            start_time = time.time()
            
            # Setup everything
            self.setup_model()
            self.setup_rewards()
            self.setup_data()
            self.setup_trainer()

            # Train
            
            logger.info("Running PPO training...")
            training_result = self.trainer.train()
            try:
                generate_and_log_samples(
                    self.config.logging.sample_logging,
                    self.policy_model,
                    self.tokenizer,
                    getattr(self, 'reward_functions', None),
                    stage="post-train",
                    log=logger,
                )
            except Exception as sample_error:
                logger.warning(f"Unable to log qualitative samples: {sample_error}")
            
            # Save
            output_dir = self._get_config_value(self.config.logging, 'output_dir', './output/ppo')
            logger.info(f"Saving to: {output_dir}")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            training_time = time.time() - start_time

            # TRL PPOTrainer.train() may return None (or an object without global_step).
            # Use training_history length as a robust fallback so examples don't report 0 steps.
            fallback_steps = 0
            try:
                if hasattr(self, "training_history") and self.training_history:
                    fallback_steps = len(self.training_history)
            except Exception:
                fallback_steps = 0
            
            results = {
                "training_time": training_time,
                "final_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else 0.0,
                "total_steps": training_result.global_step if hasattr(training_result, 'global_step') else fallback_steps,
                "model_path": output_dir,
                "training_history": self.training_history,
                "num_reward_functions": 0,
                "num_datasets": len(self.config.datasets) if hasattr(self.config, 'datasets') else 0,
            }
            
            logger.info(f"Training completed in {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    
    def _train_custom_reward_model(self) -> str:
        """Train a custom reward model using the reward_training config."""
        from aligntune.rewards import RewardModelTrainer
        from aligntune.rewards.factory import create_reward_functions
        
        reward_config = self.config.reward_training
        
        # Create reward functions from config
        if reward_config.reward_functions:
            reward_functions = create_reward_functions(reward_config.reward_functions)
        else:
            # Default reward functions if none specified
            reward_functions = create_reward_functions([
                {'type': 'length', 'weight': 0.25, 'params': {'min_length': 10, 'max_length': 200}},
                {'type': 'sentiment', 'weight': 0.25, 'params': {'positive_weight': 1.0}},
                {'type': 'coherence', 'weight': 0.25, 'params': {'threshold': 0.7}},
                {'type': 'safety', 'weight': 0.25, 'params': {'strict': True}}
            ])
        
        # Create composite weights
        composite_weights = [1.0] * len(reward_functions)
        
        # Determine base model for reward training
        base_model_name = reward_config.base_model or self.config.model.name_or_path
        
        # Create reward model trainer
        trainer = RewardModelTrainer(
            base_model_name=base_model_name,
            reward_functions=reward_functions,
            composite_weights=composite_weights
        )
        
        # Load training data
        training_texts = trainer.load_training_data(
            texts=reward_config.training_texts,
            dataset_name=reward_config.dataset_name,
            dataset_path=reward_config.dataset_path,
            dataset_split=reward_config.dataset_split,
            text_column=reward_config.text_column,
            max_samples=reward_config.max_samples
        )
        
        # Generate training dataset
        training_data = trainer.generate_training_data(
            texts=training_texts,
            batch_size=reward_config.batch_size
        )
        
        # Train the reward model
        model_path = trainer.train_reward_model(
            training_data=training_data,
            output_dir=reward_config.output_dir,
            num_epochs=reward_config.num_epochs,
            learning_rate=reward_config.learning_rate,
            batch_size=reward_config.batch_size,
            save_steps=reward_config.save_steps,
            logging_steps=reward_config.logging_steps
        )
        
        logger.info(f"✅ Custom reward model training completed: {model_path}")
        return model_path
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model with fallback for missing memory tracker."""
        try:
            if not self.eval_dataset:
                logger.info("No evaluation dataset provided")
                return {}
            
            logger.info("Evaluating...")
            
            # Initialize memory tracker if missing
            if not hasattr(self.trainer, '_memory_tracker'):
                from transformers.trainer_utils import TrainerMemoryTracker
                self.trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)
                logger.info("Initialized missing memory tracker")
            
            eval_results = self.trainer.evaluate()
            return eval_results
            
        except AttributeError as e:
            if '_memory_tracker' in str(e):
                logger.warning(f"Memory tracker error during evaluation: {e}")
                logger.warning("Skipping evaluation - this is a known TRL/Transformers compatibility issue")
                return {"eval_skipped": True, "reason": "memory_tracker_missing"}
            else:
                logger.error(f"Evaluation failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model."""
        try:
            default_path = self._get_config_value(self.config.logging, 'output_dir', './output/ppo')
            save_path = path or default_path
            
            logger.info(f"Saving to: {save_path}")
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            config_path = Path(save_path) / "training_config.yaml"
            with open(config_path, "w") as f:
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info("Saved successfully")
            return save_path
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load model."""
        try:
            from unsloth import FastLanguageModel
            
            logger.info(f"Loading from: {path}")
            max_seq_length = self._get_config_value(self.config.model, 'max_seq_length', 2048)
            
            self.policy_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            
            logger.info("Loaded successfully")
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise