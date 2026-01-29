"""
Function-Based Reward Model for TRL PPOTrainer

This module provides a wrapper that makes reward functions compatible
with TRL's PPOTrainer by implementing the required nn.Module interface.

The implementation follows TRL's reward computation flow:
1. TRL calls backbone.forward(input_ids) -> hidden_states
2. TRL calls model.score(hidden_states) -> reward_logits
3. We decode input_ids to text in forward()
4. We compute rewards from text in score()
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class DummyBackbone(nn.Module):
    """Dummy backbone that decodes input_ids to text for reward computation."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self._cached_texts = None
        self._cached_input_ids = None
    
    def to(self, device):
        """Move to device (tokenizer stays on CPU, which is fine)."""
        # nn.Module.to() will handle any parameters/buffers
        # Tokenizer doesn't need to be on device
        return super().to(device)
    
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        """
        Decode input_ids to text and cache for reward computation.
        
        Args:
            input_ids: Token IDs to decode
            attention_mask: Attention mask (unused but required by interface)
            return_dict: Whether to return dict-like object
            **kwargs: Additional arguments (unused)
            
        Returns:
            Object with hidden_states attribute (required by TRL)
        """
        self._cached_input_ids = input_ids
        
        # Decode to text
        texts = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        self._cached_texts = texts
        
        # Return dummy hidden states object (required by TRL)
        batch_size, seq_len = input_ids.shape
        hidden_size = 768  # Standard hidden size
        
        class HiddenStatesOutput:
            """Output object that mimics model output with hidden_states."""
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states
        
        # Create dummy hidden states
        hidden_states = torch.zeros(
            batch_size, seq_len, hidden_size,
            dtype=torch.float32,
            device=input_ids.device
        )
        
        output = HiddenStatesOutput(hidden_states)
        
        if return_dict:
            return output
        return output.hidden_states


class FunctionBasedRewardModel(nn.Module):
    """
    Reward model wrapper that uses reward functions directly.
    
    This class implements the interface required by TRL's PPOTrainer
    while using rule-based reward functions for actual computation.
    
    The flow is:
    1. TRL calls backbone.forward(input_ids) -> hidden_states
    2. TRL calls model.score(hidden_states) -> reward_logits
    3. We decode input_ids to text in forward()
    4. We compute rewards from text in score()
    
    Usage:
        reward_functions = [length_reward, sentiment_reward]
        reward_model = FunctionBasedRewardModel(
            reward_functions=reward_functions,
            tokenizer=tokenizer,
            device="cuda"
        )
        
        trainer = PPOTrainer(..., reward_model=reward_model)
    
    Args:
        reward_functions: List of callable reward functions that take text and return float
        tokenizer: Tokenizer for decoding input_ids to text
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        reward_functions: List[Callable],
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        if not reward_functions:
            raise ValueError("reward_functions cannot be empty")
        
        if not isinstance(reward_functions, list):
            raise TypeError(f"reward_functions must be a list, got {type(reward_functions)}")
        
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        
        self.reward_functions = reward_functions
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
        # Required by TRL: base_model_prefix
        self.base_model_prefix = "model"
        
        # Create dummy backbone
        self.model = DummyBackbone(tokenizer)
        
        # Dummy parameter for nn.Module compatibility
        self.dummy_param = nn.Parameter(torch.zeros(1, dtype=dtype))
        
        # Config for TRL compatibility
        self.config = type('Config', (), {
            'model_type': 'function_based',
            'hidden_size': 768,
            'num_labels': 1
        })()
        
        logger.info(
            f"✅ Initialized FunctionBasedRewardModel with {len(reward_functions)} reward functions"
        )
    
    def _compute_reward(self, text: str) -> float:
        """
        Compute total reward from all reward functions.
        
        Args:
            text: Text to compute reward for
            
        Returns:
            Total reward score (sum of all reward functions)
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)}, returning 0.0")
            return 0.0
        
        total_reward = 0.0
        
        for i, reward_func in enumerate(self.reward_functions):
            try:
                if not callable(reward_func):
                    logger.warning(f"Reward function {i} is not callable, skipping")
                    continue
                
                reward_value = reward_func(text)
                
                # Ensure reward is numeric
                if not isinstance(reward_value, (int, float)):
                    logger.warning(
                        f"Reward function {i} returned non-numeric value: {type(reward_value)}, "
                        f"converting to 0.0"
                    )
                    reward_value = 0.0
                
                total_reward += float(reward_value)
                
            except Exception as e:
                logger.debug(
                    f"Reward function {i} error for text '{text[:50]}...': {e}"
                )
                continue
        
        return total_reward
    
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Score method required by TRL's get_reward().
        
        This is called after forward() to extract reward scores.
        We compute rewards from cached texts.
        
        Args:
            hidden_states: Hidden states from backbone (unused, but required by interface)
            
        Returns:
            Reward logits in shape [batch_size, seq_len, 1]
        """
        # Get cached texts from backbone
        if self.model._cached_texts is None:
            logger.warning("No cached texts found in score(), returning zero rewards")
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            return torch.zeros(
                batch_size, seq_len, 1,
                dtype=self.dtype,
                device=hidden_states.device
            )
        
        texts = self.model._cached_texts
        
        # Compute rewards for each text
        batch_rewards = []
        for text in texts:
            reward = self._compute_reward(text)
            batch_rewards.append(reward)
        
        # Convert to tensor
        rewards = torch.tensor(
            batch_rewards,
            dtype=self.dtype,
            device=hidden_states.device
        )
        
        # Expand to match sequence length: [batch_size] -> [batch_size, seq_len, 1]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        reward_logits = rewards.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        reward_logits = reward_logits.expand(-1, seq_len, -1)  # [batch_size, seq_len, 1]
        
        # Clear cache after use (important for next batch)
        self.model._cached_texts = None
        
        return reward_logits
    
    def parameters(self):
        """Return dummy parameter for nn.Module compatibility."""
        yield self.dummy_param
    
    def to(self, device):
        """Move to device."""
        if isinstance(device, str):
            self.device = device
        else:
            self.device = str(device)
        return super().to(device)
    
    def eval(self):
        """Set to eval mode."""
        super().eval()
        return self
    
    def train(self, mode=True):
        """Set to train mode."""
        super().train(mode)
        return self
    
    @property
    def base_model(self):
        """Delegate base_model to self.model for TRL compatibility."""
        return self.model

