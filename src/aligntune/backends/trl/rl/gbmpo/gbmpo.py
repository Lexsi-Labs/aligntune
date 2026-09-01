"""
TRL GBMPO Backend - Simplified Implementation via Loss Patching

This module provides GBMPO (Generalized Bregman Mirror Descent Policy Optimization)
by extending TRLGRPOTrainer and patching the loss computation to add L2 regularization.

Key Design:
- Inherit from TRLGRPOTrainer (reuse everything)
- Use TRL's GRPOTrainer directly (no custom trainer class)
- Patch _compute_loss to add L2 divergence term
- Minimal code, maximum reuse

Architecture:
    TRLGBMPOTrainer (this file)
    └── TRLGRPOTrainer (parent, handles model/data/rewards)
        └── Creates TRL GRPOTrainer
            └── Patched _compute_loss (adds L2 term)

Usage:
    from aligntune.core.backend_factory import create_rl_trainer

    trainer = create_rl_trainer(
        algorithm="gbmpo",
        gbmpo_divergence_type="l2",
        gbmpo_l2_coefficient=0.0001,
        ...
    )
"""

import logging
from typing import Dict, Any
import torch

from ..grpo.grpo import TRLGRPOTrainer
from aligntune.core.rl.config import UnifiedConfig

logger = logging.getLogger(__name__)


class TRLGBMPOTrainer(TRLGRPOTrainer):
    """
    GBMPO trainer via loss patching.

    Extends TRLGRPOTrainer and patches the TRL GRPOTrainer's _compute_loss
    method to add L2 divergence regularization. No custom trainer class needed.
    """

    # GBMPO consumes the shared prompt/reward schema used by GRPO-family trainers.
    TASK_TYPE = "grpo"

    def __init__(self, config: UnifiedConfig):
        """Initialize GBMPO trainer."""
        super().__init__(config)
        logger.info(f"Initialized TRLGBMPOTrainer (task_type={self.TASK_TYPE})")

    @classmethod
    def is_available(cls) -> bool:
        """Check if GBMPO dependencies are available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def setup_trainer(self) -> None:
        """
        Set up GRPO trainer and patch loss computation for GBMPO.

        This method:
        1. Extracts GBMPO-specific params (divergence_type, l2_coefficient)
        2. Calls parent to create TRL GRPOConfig and GRPOTrainer
        3. Patches _compute_loss method to add L2 regularization
        """
        logger.info("Setting up TRL GBMPO trainer (via loss patching)...")

        # === Extract GBMPO-specific parameters ===
        divergence_type = self._get_config_value(
            self.config.train, 'gbmpo_divergence_type', 'divergence_type', default='l2')
        l2_coefficient = self._get_config_value(
            self.config.train, 'gbmpo_l2_coefficient', 'l2_coefficient', default=0.0001)

        logger.info(f"GBMPO Parameters: divergence_type={divergence_type}, l2_coefficient={l2_coefficient}")

        # For L2-only variants (no KL), set beta to small value for TRL compatibility
        beta = self._get_config_value(
            self.config.train, 'beta', 'kl_coef', default=0.1)

        if divergence_type in ["l2", "prob_l2"] and beta == 0.0:
            # Temporarily set beta for TRL compatibility
            if hasattr(self.config.train, 'beta'):
                self.config.train.beta = 1e-10
            elif hasattr(self.config.train, 'kl_coef'):
                self.config.train.kl_coef = 1e-10
            logger.info(f"L2-only variant: setting beta=1e-10 for TRL compatibility")

        # === Call parent setup_trainer to create TRL GRPOTrainer ===
        super().setup_trainer()

        # === Patch _compute_loss to add L2 regularization ===
        logger.info(f"Patching _compute_loss with L2 regularization (divergence_type={divergence_type})...")

        original_compute_loss = self.trainer._compute_loss

        def patched_compute_loss(model, inputs):
            """Patched loss computation: GRPO + L2 regularization."""
            # Initialize step counter
            if not hasattr(self.trainer, '_l2_step_count'):
                self.trainer._l2_step_count = 0
            self.trainer._l2_step_count += 1

            # Get GRPO loss (includes KL if beta > 0)
            loss = original_compute_loss(model, inputs)

            # Add L2 regularization if coefficient > 0
            if l2_coefficient > 0.0:
                ref_per_token_logps = inputs.get("ref_per_token_logps")

                if ref_per_token_logps is not None:
                    # Get current policy logprobs
                    prompt_ids = inputs["prompt_ids"]
                    prompt_mask = inputs["prompt_mask"]
                    completion_ids = inputs["completion_ids"]
                    completion_mask = inputs["completion_mask"]

                    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                    logits_to_keep = completion_ids.size(1)

                    # trl's private _get_per_token_logps_and_entropies returns a 2-tuple
                    # (logps, entropies) on trl<1.0 and a 3-tuple (logps, entropies, aux_loss)
                    # on trl>=1.0 - take the first element regardless of arity.
                    logps_result = self.trainer._get_per_token_logps_and_entropies(
                        model, input_ids, attention_mask, logits_to_keep, compute_entropy=False
                    )
                    per_token_logps = logps_result[0]

                    # Compute L2 divergence
                    if divergence_type in ["l2", "l2kl"]:
                        # Log-space L2
                        per_token_l2 = (per_token_logps - ref_per_token_logps) ** 2
                    else:  # prob_l2, prob_l2kl
                        # Probability-space L2
                        policy_probs = torch.exp(per_token_logps)
                        ref_probs = torch.exp(ref_per_token_logps)
                        per_token_l2 = (policy_probs - ref_probs) ** 2

                    # Masked L2 loss
                    l2_loss = (per_token_l2 * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                    weighted_l2 = l2_coefficient * l2_loss

                    # Add to total loss
                    loss = loss + weighted_l2

            return loss

        # Apply patch
        self.trainer._compute_loss = patched_compute_loss

        # Store GBMPO params on trainer for introspection
        self.trainer.divergence_type = divergence_type
        self.trainer.l2_coefficient = l2_coefficient

        logger.info("✓ GBMPO trainer setup completed (loss patching applied)")

    def train(self) -> Dict[str, Any]:
        """
        Execute GBMPO training.

        Same as parent TRLGRPOTrainer but logs GBMPO-specific info.
        """
        # Setup components
        self.setup_model()
        self.setup_rewards()
        self.setup_data()
        self.setup_trainer()

        output_dir = self._get_config_value(
            self.config.logging, 'output_dir', default='./output/gbmpo_trl')

        import time
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting TRL GBMPO Training")
        logger.info(f"Dataset size: {len(self.train_dataset)}")
        logger.info(f"Divergence type: {self.trainer.divergence_type}")
        logger.info(f"L2 coefficient: {self.trainer.l2_coefficient}")
        logger.info("=" * 80)

        # Add callbacks
        for cb in self.get_hf_callbacks():
            self.trainer.add_callback(cb)

        # Run training
        train_result = self.trainer.train()

        end_time = time.time()
        training_duration = end_time - start_time

        # Extract metrics
        metrics = {}
        if hasattr(train_result, 'metrics'):
            metrics = train_result.metrics

        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Compile results
        results = {
            "training_time": training_duration,
            "final_loss": train_result.training_loss if hasattr(
                train_result, 'training_loss') else metrics.get('train_loss', 0.0),
            "total_steps": train_result.global_step if hasattr(
                train_result, 'global_step') else 0,
            "model_path": output_dir,
            "num_reward_functions": len(self.reward_functions),
            "divergence_type": self.trainer.divergence_type,
            "l2_coefficient": self.trainer.l2_coefficient,
            "metrics": metrics,
        }

        logger.info("=" * 80)
        logger.info("TRL GBMPO Training Completed Successfully!")
        logger.info(f"Final loss: {results['final_loss']:.4f}")
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Divergence type: {results['divergence_type']}")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 80)

        return results
