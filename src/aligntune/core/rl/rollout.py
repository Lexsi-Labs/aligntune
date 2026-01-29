"""
Rollout engine for batched generation in RLHF training.

This module provides utilities for generating rollouts, computing KL divergences,
and managing batched generation with distributed training support.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from .models import PolicyModel, ReferenceModel

logger = logging.getLogger(__name__)


class RolloutEngine:
    """Engine for generating rollouts and computing KL divergences."""
    
    def __init__(self, policy_model: PolicyModel, reference_model: Optional[ReferenceModel] = None):
        """Initialize rollout engine."""
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.device = policy_model.device
        
    def _should_use_cache(self, model):
        """Check if model supports and has caching enabled.
        
        Only decoder models (GPT, LLaMA, etc.) support use_cache.
        Encoder models (BERT, DeBERTa, etc.) do NOT support it.
        """
        try:
            if hasattr(model, 'config'):
                config = model.config
                model_type = getattr(config, 'model_type', '').lower()
                
                # Whitelist of decoder model types that support use_cache
                decoder_models = [
                    'gpt2', 'gpt', 'gpt_neo', 'gpt_neox', 'gptj', 'gpt_bigcode',
                    'llama', 'mistral', 'mixtral', 'qwen', 'qwen2', 'phi', 'phi3',
                    'opt', 'bloom', 'falcon', 'mpt', 'codegen', 'starcoder'
                ]
                
                # Only enable cache for decoder models
                if model_type in decoder_models:
                    if hasattr(config, 'use_cache'):
                        return config.use_cache
            
            # Default to False for encoder models and unknown types
            return False
        except Exception:
            return False
    
    def generate_rollouts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        synced_gpus: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate rollouts using the policy model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            synced_gpus: Whether to sync GPUs for distributed generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        if self.policy_model.model is None:
            raise RuntimeError("Policy model not loaded")
        
        # Prepare generation parameters
        generation_kwargs = {
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "synced_gpus": synced_gpus,
            "pad_token_id": self.policy_model.tokenizer.pad_token_id,
            "eos_token_id": self.policy_model.tokenizer.eos_token_id,
            "use_cache": self._should_use_cache(self.policy_model.model),
            **kwargs
        }
        
        # Use max_new_tokens if provided, otherwise use max_length
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generation_kwargs["max_length"] = max_length
        
        # Ensure input tensors are on the same device as the model
        device = next(self.policy_model.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Generate sequences
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        # Process outputs
        if num_return_sequences > 1:
            # Reshape outputs for multiple sequences
            batch_size = input_ids.size(0)
            outputs = outputs.view(batch_size, num_return_sequences, -1)
        
        # Compute log probabilities
        log_probs = self._compute_log_probs(input_ids, outputs, attention_mask)
        
        # Compute KL divergence if reference model is available
        kl_divergence = None
        if self.reference_model is not None:
            kl_divergence = self._compute_kl_divergence(input_ids, outputs, attention_mask)
        
        return {
            "generated_ids": outputs,
            "log_probs": log_probs,
            "kl_divergence": kl_divergence,
            "generation_kwargs": generation_kwargs
        }
    
    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities of generated sequences."""
        if self.policy_model.model is None:
            raise RuntimeError("Policy model not loaded")
        
        # Create attention mask for generated sequence (all tokens are valid)
        gen_attention_mask = torch.ones_like(generated_ids)
        
        # Get logits for generated sequences
        outputs = self.policy_model.forward(generated_ids, gen_attention_mask)
        logits = outputs.logits
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = generated_ids[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Create attention mask for shifted sequence
        shift_attention_mask = gen_attention_mask[..., 1:].contiguous()
        log_probs = log_probs * shift_attention_mask
        
        return log_probs
    
    def _compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference models."""
        if self.reference_model is None or self.reference_model.model is None:
            raise RuntimeError("Reference model not loaded")
        
        # Get logits from both models
        policy_outputs = self.policy_model.forward(generated_ids, attention_mask)
        reference_outputs = self.reference_model.forward(generated_ids, attention_mask)
        
        policy_logits = policy_outputs.logits
        reference_logits = reference_outputs.logits
        
        # Shift for next token prediction
        shift_policy_logits = policy_logits[..., :-1, :].contiguous()
        shift_reference_logits = reference_logits[..., :-1, :].contiguous()
        shift_labels = generated_ids[..., 1:].contiguous()
        
        # Compute log probabilities
        policy_log_probs = F.log_softmax(shift_policy_logits, dim=-1)
        reference_log_probs = F.log_softmax(shift_reference_logits, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            policy_log_probs,
            reference_log_probs,
            reduction='none',
            log_target=True
        )
        
        # Sum over vocabulary dimension
        kl_div = kl_div.sum(dim=-1)
        
        # Create attention mask for generated sequence
        gen_attention_mask = torch.ones_like(generated_ids)
        shift_attention_mask = gen_attention_mask[..., 1:].contiguous()
        kl_div = kl_div * shift_attention_mask
        
        return kl_div
    
    def compute_rewards(
        self,
        generated_ids: torch.Tensor,
        reward_functions: List[Any],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute rewards for generated sequences.
        
        Args:
            generated_ids: Generated token IDs
            reward_functions: List of reward functions
            **kwargs: Additional parameters for reward functions
            
        Returns:
            Tensor of rewards
        """
        if self.policy_model.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Decode generated sequences
        generated_texts = self.policy_model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        # Compute rewards for each sequence
        rewards = []
        for text in generated_texts:
            sequence_rewards = []
            for reward_func in reward_functions:
                try:
                    reward_value = reward_func(text, **kwargs)
                    sequence_rewards.append(reward_value)
                except Exception as e:
                    logger.warning(f"Reward function failed: {e}")
                    sequence_rewards.append(0.0)
            
            rewards.append(sequence_rewards)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward tensor
            values: Value predictions (optional)
            gamma: Discount factor
            lambda_: GAE parameter
            
        Returns:
            Advantage tensor
        """
        if values is None:
            # Simple advantage: just rewards
            return rewards
        
        # GAE computation
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(rewards.size(0))):
            if t == rewards.size(0) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lambda_ * advantage
            advantages[t] = advantage
        
        return advantages
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute returns (discounted cumulative rewards).
        
        Args:
            rewards: Reward tensor
            gamma: Discount factor
            
        Returns:
            Returns tensor
        """
        returns = torch.zeros_like(rewards)
        return_ = 0
        
        for t in reversed(range(rewards.size(0))):
            return_ = rewards[t] + gamma * return_
            returns[t] = return_
        
        return returns
    
    def batch_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate rollouts in batches to manage memory.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            batch_size: Batch size for generation
            **generation_kwargs: Generation parameters
            
        Returns:
            List of rollout dictionaries
        """
        total_samples = input_ids.size(0)
        rollouts = []
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_input_ids = input_ids[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx] if attention_mask is not None else None
            
            # Generate rollouts for this batch
            batch_rollouts = self.generate_rollouts(
                batch_input_ids,
                batch_attention_mask,
                **generation_kwargs
            )
            
            rollouts.append(batch_rollouts)
        
        return rollouts
    
    def compute_metrics(
        self,
        rollouts: List[Dict[str, torch.Tensor]],
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute rollout metrics.
        
        Args:
            rollouts: List of rollout dictionaries
            rewards: Reward tensor
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Combine all rollouts
        all_log_probs = torch.cat([r["log_probs"] for r in rollouts], dim=0)
        all_kl_divs = torch.cat([r["kl_divergence"] for r in rollouts], dim=0) if rollouts[0]["kl_divergence"] is not None else None
        
        # Log probability metrics
        metrics["mean_log_prob"] = all_log_probs.mean().item()
        metrics["std_log_prob"] = all_log_probs.std().item()
        metrics["min_log_prob"] = all_log_probs.min().item()
        metrics["max_log_prob"] = all_log_probs.max().item()
        
        # KL divergence metrics
        if all_kl_divs is not None:
            metrics["mean_kl_divergence"] = all_kl_divs.mean().item()
            metrics["std_kl_divergence"] = all_kl_divs.std().item()
            metrics["max_kl_divergence"] = all_kl_divs.max().item()
        
        # Reward metrics
        metrics["mean_reward"] = rewards.mean().item()
        metrics["std_reward"] = rewards.std().item()
        metrics["min_reward"] = rewards.min().item()
        metrics["max_reward"] = rewards.max().item()
        
        return metrics
    
    def cleanup(self) -> None:
        """Cleanup rollout engine resources."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleaned up rollout engine resources")
