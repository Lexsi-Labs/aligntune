"""
Model wrapper classes for unified RLHF training.

This module provides wrapper classes for policy, reference, and value models
with support for precision control, quantization, and distributed training.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .config import ModelConfig, PrecisionType

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Base wrapper class for models."""
    
    def __init__(self, config: ModelConfig, model_type: str = "policy"):
        """Initialize model wrapper."""
        self.config = config
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = None
        
        logger.info(f"Initializing {model_type} model: {config.name_or_path}")
    
    def load_model(self) -> PreTrainedModel:
        """Load model with specified configuration."""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._get_torch_dtype(),
        }
        
        # Add quantization config if specified
        if self.config.quantization:
            model_kwargs["quantization_config"] = self._get_quantization_config()
        
        # Add attention implementation
        if self.config.attn_implementation != "auto":
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name_or_path,
                **model_kwargs
            )
            
            # Move to device
            if not self.config.quantization:  # Quantized models handle device placement
                self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Loaded {self.model_type} model: {self.config.name_or_path}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model: {e}")
            raise
        
        return self.model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name_or_path,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded tokenizer for {self.model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        return self.tokenizer
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype based on precision configuration."""
        if self.config.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.config.precision == PrecisionType.FP16:
            return torch.float16
        else:
            return torch.float32
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration."""
        if not self.config.quantization:
            return None
        
        quant_config = self.config.quantization
        
        if quant_config.get("4bit", False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=quant_config.get("use_double_quant", True)
            )
        elif quant_config.get("8bit", False):
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        return None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        synced_gpus: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Remove pad_token_id from kwargs if it exists to avoid duplicate
        kwargs.pop('pad_token_id', None)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                synced_gpus=synced_gpus,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                **kwargs
            )
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type,
            "model_name": self.config.name_or_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "precision": self.config.precision.value,
            "quantization": bool(self.config.quantization),
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "device": str(self.device) if self.device else "unknown"
        }


class PolicyModel(ModelWrapper):
    """Policy model wrapper for RLHF training."""
    
    def __init__(self, config: ModelConfig):
        """Initialize policy model."""
        super().__init__(config, "policy")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through policy model."""
        if self.model is None:
            raise RuntimeError("Policy model not loaded")
        
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits from policy model."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs.logits


class ReferenceModel(ModelWrapper):
    """Reference model wrapper for RLHF training."""
    
    def __init__(self, config: ModelConfig):
        """Initialize reference model."""
        super().__init__(config, "reference")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through reference model."""
        if self.model is None:
            raise RuntimeError("Reference model not loaded")
        
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits from reference model."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs.logits


class ValueModel(ModelWrapper):
    """Value model wrapper for RLHF training."""
    
    def __init__(self, config: ModelConfig):
        """Initialize value model."""
        super().__init__(config, "value")
        self.value_head = None
    
    def load_model(self) -> PreTrainedModel:
        """Load value model with value head."""
        # Load base model
        base_model = super().load_model()
        
        # Add value head if not present
        if not hasattr(base_model, 'value_head'):
            self.value_head = nn.Linear(base_model.config.hidden_size, 1)
            self.value_head = self.value_head.to(self.device)
            base_model.value_head = self.value_head
        
        return base_model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through value model."""
        if self.model is None:
            raise RuntimeError("Value model not loaded")
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Get value predictions
        if hasattr(self.model, 'value_head'):
            values = self.model.value_head(outputs.hidden_states[-1])
            outputs.values = values
        
        return outputs
    
    def get_values(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get value predictions."""
        outputs = self.forward(input_ids, attention_mask)
        return outputs.values if hasattr(outputs, 'values') else None


class ModelManager:
    """Manager for multiple models in RLHF training."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model manager."""
        self.config = config
        self.policy_model = None
        self.reference_model = None
        self.value_model = None
        self.tokenizer = None
    
    def load_policy_model(self) -> PolicyModel:
        """Load policy model."""
        self.policy_model = PolicyModel(self.config)
        self.policy_model.load_model()
        self.policy_model.load_tokenizer()
        
        # Use policy model's tokenizer as the main tokenizer
        if self.tokenizer is None:
            self.tokenizer = self.policy_model.tokenizer
        
        return self.policy_model
    
    def load_reference_model(self) -> ReferenceModel:
        """Load reference model."""
        # Use same config as policy model for reference
        self.reference_model = ReferenceModel(self.config)
        self.reference_model.load_model()
        
        # Use same tokenizer as policy model
        self.reference_model.tokenizer = self.tokenizer
        
        return self.reference_model
    
    def load_value_model(self, value_model_config: Optional[ModelConfig] = None) -> ValueModel:
        """Load value model."""
        # Use provided config or same as policy model
        config = value_model_config or self.config
        self.value_model = ValueModel(config)
        self.value_model.load_model()
        
        # Use same tokenizer as policy model
        self.value_model.tokenizer = self.tokenizer
        
        return self.value_model
    
    def get_all_models(self) -> Dict[str, ModelWrapper]:
        """Get all loaded models."""
        models = {}
        
        if self.policy_model:
            models["policy"] = self.policy_model
        if self.reference_model:
            models["reference"] = self.reference_model
        if self.value_model:
            models["value"] = self.value_model
        
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models."""
        info = {
            "tokenizer": "loaded" if self.tokenizer else "not_loaded",
            "models": {}
        }
        
        for name, model in self.get_all_models().items():
            info["models"][name] = model.get_model_info()
        
        return info
    
    def save_models(self, output_dir: str) -> None:
        """Save all models to output directory."""
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save models
        for name, model in self.get_all_models().items():
            if model.model:
                model_path = output_path / name
                model_path.mkdir(exist_ok=True)
                model.model.save_pretrained(model_path)
        
        logger.info(f"Saved all models to {output_path}")
    
    def cleanup(self) -> None:
        """Cleanup model resources."""
        for model in self.get_all_models().values():
            if model.model:
                del model.model
        
        if self.tokenizer:
            del self.tokenizer
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleaned up model resources")
