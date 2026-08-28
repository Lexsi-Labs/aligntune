"""
MoE-Aware SFT Trainer for AlignTune.

This module provides a Mixture of Experts (MoE) aware trainer that extends the
standard SFT trainer with support for router stability losses. It enables seamless
integration of MoE models into the AlignTune training pipeline.

Features:
- Automatic MoE model detection
- Router loss computation and weighting
- Combined loss calculation (LM loss + router losses)
- Configurable loss weights
- Monitoring and logging of router metrics
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from .router_losses import RouterStabilityLosses

logger = logging.getLogger(__name__)


class MoEConfig:
    """
    Configuration for MoE training parameters.

    Attributes:
        enabled (bool): Whether MoE training is enabled. Default is True.
        z_loss_weight (float): Weight for Z-loss. Default is 0.01.
        lb_loss_weight (float): Weight for load-balance loss. Default is 0.01.
        entropy_loss_weight (float): Weight for entropy loss. Default is 0.0.
        expert_capacity_multiplier (float): Multiplier for expert capacity calculation.
            capacity = num_experts * expert_capacity_multiplier. Default is 1.25.
        device (Optional[str]): Device for loss computation ('cpu', 'cuda', or None for auto).
            If None, automatically selects based on availability.
    """

    def __init__(
        self,
        enabled: bool = True,
        z_loss_weight: float = 0.01,
        lb_loss_weight: float = 0.01,
        entropy_loss_weight: float = 0.0,
        expert_capacity_multiplier: float = 1.25,
        device: Optional[str] = None,
    ):
        """
        Initialize MoE configuration.

        Args:
            enabled (bool): Whether MoE training is enabled. Default is True.
            z_loss_weight (float): Weight for Z-loss. Default is 0.01.
            lb_loss_weight (float): Weight for load-balance loss. Default is 0.01.
            entropy_loss_weight (float): Weight for entropy loss. Default is 0.0.
            expert_capacity_multiplier (float): Multiplier for expert capacity. Default is 1.25.
            device (Optional[str]): Device for computation. Default is None (auto).
        """
        self.enabled = enabled
        self.z_loss_weight = z_loss_weight
        self.lb_loss_weight = lb_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.expert_capacity_multiplier = expert_capacity_multiplier
        self.device = device

        logger.debug(
            f"MoEConfig initialized: enabled={enabled}, z_loss_weight={z_loss_weight}, "
            f"lb_loss_weight={lb_loss_weight}, entropy_loss_weight={entropy_loss_weight}"
        )

    def __repr__(self) -> str:
        """String representation of MoEConfig."""
        return (
            f"MoEConfig(enabled={self.enabled}, z_loss_weight={self.z_loss_weight}, "
            f"lb_loss_weight={self.lb_loss_weight}, entropy_loss_weight={self.entropy_loss_weight}, "
            f"expert_capacity_multiplier={self.expert_capacity_multiplier}, device={self.device})"
        )


class MoESFTTrainer:
    """
    MoE-aware Supervised Fine-Tuning Trainer.

    Extends standard SFT training with support for Mixture of Experts models by
    computing and combining router stability losses with the standard language modeling loss.

    This trainer should be used as a wrapper or mixin that can be integrated into
    existing SFT trainer implementations.

    Attributes:
        model: The MoE model to be trained.
        moe_config (MoEConfig): Configuration for MoE training.
        router_losses (RouterStabilityLosses): Utility for computing router losses.
        is_moe_model (bool): Whether the model is detected as an MoE model.

    Example:
        >>> moe_config = MoEConfig(z_loss_weight=0.01, lb_loss_weight=0.01)
        >>> trainer = MoESFTTrainer(model, moe_config)
        >>> combined_loss = trainer.compute_loss(logits, labels)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        moe_config: Optional[MoEConfig] = None,
    ):
        """
        Initialize MoE SFT Trainer.

        Args:
            model (Optional[torch.nn.Module]): The model to train. If provided, will detect
                if it's an MoE model.
            moe_config (Optional[MoEConfig]): MoE training configuration. If None, uses defaults.

        Raises:
            ValueError: If model is not properly initialized.
        """
        self.model = model
        self.moe_config = moe_config or MoEConfig()
        self.router_losses = RouterStabilityLosses(device=self.moe_config.device)

        # Detect if model is an MoE model
        self.is_moe_model = self._detect_moe_model() if model is not None else False

        if self.is_moe_model:
            logger.info("MoE model detected. MoE losses will be computed during training.")
        else:
            logger.debug("Standard model detected (not MoE).")

    def _detect_moe_model(self) -> bool:
        """
        Detect if the model is a Mixture of Experts model.

        Checks for common MoE indicators in model architecture:
        - Presence of router, gate, or expert-related modules
        - Specific attribute patterns from libraries like vLLM, HuggingFace Transformers

        Returns:
            bool: True if model appears to be an MoE model, False otherwise.
        """
        if self.model is None:
            return False

        # Check for common MoE indicators in model structure
        model_str = str(type(self.model))

        # Check for known MoE model classes
        moe_indicators = ["MoE", "MixtureOfExperts", "router", "experts", "gate"]
        for indicator in moe_indicators:
            if indicator.lower() in model_str.lower():
                return True

        # Check for expert layers in model (safely handle mock objects)
        try:
            for module_name, module in self.model.named_modules():
                if any(
                    indicator.lower() in module_name.lower()
                    for indicator in ["expert", "router", "gate", "moe"]
                ):
                    return True
        except (AttributeError, TypeError):
            # Model doesn't support named_modules or is not iterable
            pass

        return False

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        router_outputs: Optional[Dict[str, torch.Tensor]] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute combined loss for MoE training.

        Computes the standard language modeling loss and, if MoE is enabled,
        adds router stability losses scaled by their configured weights.

        The combined loss is:
            total_loss = lm_loss + z_loss * weight_z + lb_loss * weight_lb + entropy_loss * weight_entropy

        Args:
            logits (torch.Tensor): Model output logits of shape (batch_size * seq_len, vocab_size)
                or (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Target labels of shape (batch_size * seq_len,) or (batch_size, seq_len)
                with values in [0, vocab_size) or -100 for padding tokens.
            router_outputs (Optional[Dict[str, torch.Tensor]]): Dictionary containing router outputs:
                - 'router_logits': Router logits of shape (batch_size * seq_len, num_experts)
                - 'routing_probs': Routing probabilities (optional, computed if not provided)
                - 'expert_assignments': Hard expert assignments (optional, computed via argmax)
                - 'expert_mask': Mask for valid experts (optional)
            reduction (str): Reduction method for loss ('mean', 'sum'). Default is 'mean'.

        Returns:
            torch.Tensor: Scalar tensor containing the combined loss.

        Raises:
            ValueError: If logits and labels shapes are incompatible.
            RuntimeError: If router_outputs are provided but malformed.

        Example:
            >>> logits = torch.randn(128, 32000)
            >>> labels = torch.randint(0, 32000, (128,))
            >>> router_outputs = {
            ...     'router_logits': torch.randn(128, 8),
            ... }
            >>> loss = trainer.compute_loss(logits, labels, router_outputs)
        """
        # Reshape logits and labels if necessary
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)

        # Validate shapes
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits and labels first dimension mismatch: {logits.shape[0]} vs {labels.shape[0]}"
            )

        # Compute standard language modeling loss
        lm_loss = F.cross_entropy(
            logits,
            labels,
            reduction=reduction,
            ignore_index=-100,
        )

        # Return LM loss if MoE is not enabled
        if not self.is_moe_model or not self.moe_config.enabled:
            return lm_loss

        # Compute MoE router losses if enabled
        if router_outputs is None or len(router_outputs) == 0:
            logger.warning(
                "MoE model detected but router_outputs not provided. "
                "Returning LM loss only. Provide router_outputs dict with 'router_logits' key."
            )
            return lm_loss

        try:
            router_logits = router_outputs.get("router_logits")
            if router_logits is None:
                logger.warning(
                    "router_logits not found in router_outputs. Returning LM loss only."
                )
                return lm_loss

            # Compute routing probabilities if not provided
            routing_probs = router_outputs.get("routing_probs")
            if routing_probs is None:
                routing_probs = F.softmax(router_logits, dim=-1)

            # Get expert assignments and mask
            expert_assignments = router_outputs.get("expert_assignments")
            expert_mask = router_outputs.get("expert_mask")

            # Compute combined router losses
            combined_router_loss, loss_dict = self.router_losses.compute_combined_loss(
                router_logits=router_logits,
                routing_probs=routing_probs,
                expert_assignments=expert_assignments,
                expert_mask=expert_mask,
                z_loss_weight=self.moe_config.z_loss_weight,
                lb_loss_weight=self.moe_config.lb_loss_weight,
                entropy_loss_weight=self.moe_config.entropy_loss_weight,
            )

            # Combine LM loss and router losses
            total_loss = lm_loss + combined_router_loss

            # Log loss components for monitoring
            logger.debug(
                f"LM Loss: {lm_loss.item():.4f}, "
                f"Z-Loss: {loss_dict['z_loss']:.6f}, "
                f"LB-Loss: {loss_dict['lb_loss']:.6f}, "
                f"Entropy-Loss: {loss_dict['entropy_loss']:.6f}, "
                f"Total Router Loss: {loss_dict['total']:.6f}"
            )

            return total_loss

        except Exception as e:
            logger.error(f"Error computing MoE losses: {e}. Returning LM loss only.")
            return lm_loss

    def get_moe_loss_weights(self) -> Dict[str, float]:
        """
        Get current MoE loss weights.

        Returns:
            Dict[str, float]: Dictionary containing current loss weights:
                - 'z_loss_weight'
                - 'lb_loss_weight'
                - 'entropy_loss_weight'

        Example:
            >>> weights = trainer.get_moe_loss_weights()
            >>> print(weights)
            {'z_loss_weight': 0.01, 'lb_loss_weight': 0.01, 'entropy_loss_weight': 0.0}
        """
        return {
            "z_loss_weight": self.moe_config.z_loss_weight,
            "lb_loss_weight": self.moe_config.lb_loss_weight,
            "entropy_loss_weight": self.moe_config.entropy_loss_weight,
        }

    def set_moe_loss_weights(
        self,
        z_loss_weight: Optional[float] = None,
        lb_loss_weight: Optional[float] = None,
        entropy_loss_weight: Optional[float] = None,
    ) -> None:
        """
        Update MoE loss weights dynamically.

        Allows adjusting loss weights during training without reinitializing the trainer.

        Args:
            z_loss_weight (Optional[float]): New weight for Z-loss. If None, unchanged.
            lb_loss_weight (Optional[float]): New weight for load-balance loss. If None, unchanged.
            entropy_loss_weight (Optional[float]): New weight for entropy loss. If None, unchanged.

        Example:
            >>> trainer.set_moe_loss_weights(z_loss_weight=0.02, lb_loss_weight=0.02)
        """
        if z_loss_weight is not None:
            self.moe_config.z_loss_weight = z_loss_weight
            logger.debug(f"Updated z_loss_weight to {z_loss_weight}")

        if lb_loss_weight is not None:
            self.moe_config.lb_loss_weight = lb_loss_weight
            logger.debug(f"Updated lb_loss_weight to {lb_loss_weight}")

        if entropy_loss_weight is not None:
            self.moe_config.entropy_loss_weight = entropy_loss_weight
            logger.debug(f"Updated entropy_loss_weight to {entropy_loss_weight}")

    def is_moe_enabled(self) -> bool:
        """
        Check if MoE training is enabled.

        Returns:
            bool: True if both model is detected as MoE and config is enabled, False otherwise.
        """
        return self.is_moe_model and self.moe_config.enabled

    def __repr__(self) -> str:
        """String representation of MoESFTTrainer."""
        return (
            f"MoESFTTrainer(is_moe_model={self.is_moe_model}, "
            f"moe_enabled={self.is_moe_enabled()}, config={self.moe_config})"
        )
