"""
Universal Reward Model Wrapper for use_cache Compatibility

This module provides a universal wrapper for reward models that handles compatibility
issues with TRL's get_reward() function, specifically the use_cache parameter that
is not supported by encoder-only models like DeBERTa.
"""

import logging
import inspect
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)


class UniversalRewardModelWrapper:
    """
    Universal wrapper for reward models that handles parameter compatibility.
    
    This wrapper:
    1. Inspects the model's forward signature to detect supported parameters
    2. Dynamically filters out unsupported kwargs (like use_cache)
    3. Ensures return_dict=True for consistent outputs
    4. Adds a score() method that works with hidden_states
    5. Delegates all other attributes to the wrapped model
    """
    
    def __init__(self, model):
        """
        Initialize the wrapper with a reward model.
        
        Args:
            model: The reward model to wrap (e.g., DeBERTaForSequenceClassification)
        """
        self._model = model
        self._model_type = type(model).__name__
        
        # Inspect the model's forward signature to detect supported parameters
        self._supported_params = self._inspect_forward_signature()
        
        # Ensure score method exists for TRL/Unsloth compatibility
        if hasattr(model, 'score'):
            # Model already has score (e.g., Unsloth models with nn.Linear)
            self.score = model.score
        elif hasattr(model, 'classifier'):
            # For AutoModelForSequenceClassification (BERT, DeBERTa, etc.)
            def score_from_classifier(hidden_states):
                # Sequence classification models expect pooled output (CLS token)
                # hidden_states shape: [batch_size, seq_len, hidden_size]
                
                # Take CLS token (first token) for classification
                cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
                
                # Get reward score from classifier
                reward_score = model.classifier(cls_hidden)  # [batch_size, 1]
                
                # Expand to match sequence length for TRL compatibility
                # TRL needs [batch_size, seq_len, 1] to index at sequence_lengths
                seq_len = hidden_states.size(1)
                reward_logits = reward_score.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 1]
                
                return reward_logits
            self.score = score_from_classifier
        else:
            # Fallback: use forward pass
            def score_from_forward(hidden_states):
                output = model(inputs_embeds=hidden_states)
                # Handle different output types
                if isinstance(output, tuple):
                    # AutoModelForCausalLMWithValueHead returns (logits, None, value)
                    return output[0]  # Return the logits
                elif hasattr(output, 'logits'):
                    return output.logits
                else:
                    # If output is already the logits tensor
                    return output
            self.score = score_from_forward
        
        logger.info(f"Wrapped {self._model_type} with UniversalRewardModelWrapper")
        logger.info(f"Supported forward parameters: {list(self._supported_params)}")
        logger.debug(f"UniversalRewardModelWrapper initialized with score method from {type(model).__name__}")
    
    def _inspect_forward_signature(self) -> set:
        """Inspect the model's forward signature to detect supported parameters."""
        try:
            # Get the forward method signature
            forward_sig = inspect.signature(self._model.forward)
            supported_params = set(forward_sig.parameters.keys())
            
            # Also check backbone model if it exists
            if hasattr(self._model, 'base_model_prefix'):
                backbone_name = self._model.base_model_prefix
                if hasattr(self._model, backbone_name):
                    backbone = getattr(self._model, backbone_name)
                    backbone_sig = inspect.signature(backbone.forward)
                    backbone_params = set(backbone_sig.parameters.keys())
                    
                    # Combine parameters from both model and backbone
                    supported_params.update(backbone_params)
                    logger.info(f"Backbone {backbone_name} parameters: {list(backbone_params)}")
            
            return supported_params
        except Exception as e:
            logger.warning(f"Could not inspect forward signature: {e}")
            # Fallback: assume common parameters are supported
            return {'input_ids', 'attention_mask', 'return_dict', 'output_hidden_states'}
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with parameter filtering.
        
        Filters out unsupported parameters before calling the original forward method.
        """
        # Make a copy of kwargs to avoid modifying the original
        filtered_kwargs = kwargs.copy()
        
        # Always ensure return_dict=True for consistent outputs
        filtered_kwargs['return_dict'] = True
        
        # Filter out unsupported parameters
        unsupported_params = []
        for param_name in list(filtered_kwargs.keys()):
            if param_name not in self._supported_params:
                unsupported_params.append(param_name)
                del filtered_kwargs[param_name]
        
        if unsupported_params:
            logger.debug(f"Filtered out unsupported parameters: {unsupported_params}")
        
        try:
            # Call the original forward method with filtered parameters
            result = self._model.forward(*args, **filtered_kwargs)
            
            # Ensure we return a proper output object, not a tuple
            if isinstance(result, tuple):
                # If it's a tuple, return the first element (usually the main output)
                return result[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"Args: {args}")
            logger.error(f"Filtered kwargs: {filtered_kwargs}")
            raise
    
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Score method that works with hidden_states from TRL's get_reward().
        
        Args:
            hidden_states: Hidden states from the model's backbone
            
        Returns:
            Reward logits
        """
        # Ensure dtype compatibility
        if hidden_states.dtype != next(self._model.parameters()).dtype:
            hidden_states = hidden_states.to(next(self._model.parameters()).dtype)
        
        with torch.no_grad():
            # For sequence classification models, use the classifier head directly
            if hasattr(self._model, 'classifier'):
                logits = self._model.classifier(hidden_states)
            elif hasattr(self._model, 'score'):
                # If it already has a score method, use it
                logits = self._model.score(hidden_states)
            else:
                # Fallback: try to find a classification head
                # Look for common classifier head names
                classifier_names = ['classifier', 'cls_head', 'classification_head', 'head']
                classifier_head = None
                
                for name in classifier_names:
                    if hasattr(self._model, name):
                        classifier_head = getattr(self._model, name)
                        break
                
                if classifier_head is not None:
                    logits = classifier_head(hidden_states)
                else:
                    # Last resort: create dummy scores
                    batch_size = hidden_states.shape[0]
                    device = hidden_states.device
                    dtype = next(self._model.parameters()).dtype
                    
                    logger.warning(f"No classifier head found in {self._model_type}, using dummy scores")
                    logits = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            
            return logits
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate all other attributes to the wrapped model.
        
        This ensures the wrapper is transparent - all model attributes,
        methods, and properties are accessible as if they were on the wrapper.
        """
        return getattr(self._model, name)
    
    def __call__(self, *args, **kwargs):
        """Handle direct calls to the model (e.g., model(input_ids))."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"UniversalRewardModelWrapper({self._model})"
    
    def __str__(self) -> str:
        """String representation of the wrapper."""
        return f"UniversalRewardModelWrapper({self._model})"
