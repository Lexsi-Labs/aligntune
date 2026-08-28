"""
Router Stability Losses for Mixture of Experts Models.

This module implements auxiliary losses for MoE routers to encourage stable expert
utilization and balanced load distribution during training. Implementations follow:
- ST-MoE paper: https://arxiv.org/abs/2202.08906
- Switch Transformer paper: https://arxiv.org/abs/2101.03961

Losses implemented:
1. Z-Loss (Auxiliary Loss): Variance of auxiliary loss across experts
2. Load Balance Loss: Encourages balanced load distribution
3. Entropy Loss: Router entropy regularization

All losses return scalar tensors and can be combined with standard training losses.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RouterStabilityLosses:
    """
    Compute auxiliary losses for MoE router stability.

    This class provides methods to compute various auxiliary losses that stabilize
    training of Mixture of Experts models by encouraging balanced expert utilization
    and preventing router collapse.

    Attributes:
        device (torch.device): Device to place tensors on (CPU or CUDA).

    Example:
        >>> losses = RouterStabilityLosses(device="cuda")
        >>> z_loss = losses.compute_z_loss(router_logits, expert_mask)
        >>> lb_loss = losses.compute_load_balance_loss(routing_probs, expert_assignments)
        >>> entropy_loss = losses.compute_entropy_loss(routing_probs)
        >>> total_loss = z_loss + 0.01 * lb_loss + 0.0 * entropy_loss
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize Router Stability Losses.

        Args:
            device (Optional[str]): Device placement ('cpu', 'cuda', or None for auto).
                If None, defaults to 'cuda' if available, else 'cpu'.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        logger.debug(f"RouterStabilityLosses initialized on device: {self.device}")

    def compute_z_loss(
        self,
        router_logits: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Z-Loss (Auxiliary Loss from ST-MoE paper).

        The Z-loss encourages uniform utilization by penalizing the variance of
        auxiliary losses across experts. It's defined as the variance of the sum
        of gating weights for each expert across tokens.

        The auxiliary loss for each expert is: aux_i = sum(softmax(logits)[:, i])
        Z-loss = variance(aux across experts)

        This prevents router collapse where all tokens route to a single expert.

        Mathematical formulation:
            aux_i = sum_tokens(routing_prob[tokens, i])
            z_loss = variance(aux_i for all experts)

        Args:
            router_logits (torch.Tensor): Router output logits of shape (batch_size, num_experts)
                or (sequence_length, num_experts) containing logits for each token's
                expert assignment.
            expert_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, num_experts)
                or (sequence_length, num_experts) with 0 for invalid experts, 1 for valid.
                If None, all experts are considered valid.

        Returns:
            torch.Tensor: Scalar tensor containing the Z-loss. Value is >= 0.
                Returns 0.0 if num_experts < 2.

        Raises:
            ValueError: If router_logits dimensions are invalid or mismatched with expert_mask.

        Example:
            >>> router_logits = torch.randn(32, 8)  # 32 tokens, 8 experts
            >>> z_loss = losses.compute_z_loss(router_logits)
            >>> print(z_loss.item())  # scalar value
        """
        # Validate inputs
        if router_logits.dim() not in (2,):
            raise ValueError(
                f"router_logits must be 2D (batch_size/seq_len, num_experts), "
                f"got shape {router_logits.shape}"
            )

        routing_probs = F.softmax(router_logits, dim=-1)

        if expert_mask is not None:
            if expert_mask.shape != router_logits.shape:
                raise ValueError(
                    f"expert_mask shape {expert_mask.shape} doesn't match "
                    f"router_logits shape {router_logits.shape}"
                )
            routing_probs = routing_probs * expert_mask

        # Compute auxiliary loss: sum of routing probabilities per expert
        # Shape: (num_experts,)
        num_tokens = routing_probs.shape[0]
        aux_loss_per_expert = routing_probs.sum(dim=0)

        num_experts = aux_loss_per_expert.shape[0]
        if num_experts < 2:
            return torch.tensor(0.0, dtype=routing_probs.dtype, device=routing_probs.device)

        # Normalize by number of tokens
        aux_loss_per_expert = aux_loss_per_expert / max(num_tokens, 1)

        # Z-loss is the variance of auxiliary losses across experts
        mean_aux = aux_loss_per_expert.mean()
        z_loss = torch.mean((aux_loss_per_expert - mean_aux) ** 2)

        return z_loss

    def compute_load_balance_loss(
        self,
        routing_probs: torch.Tensor,
        expert_assignments: Optional[torch.Tensor] = None,
        expert_capacity: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Load Balance Loss (from Switch Transformer paper).

        The load balance loss encourages balanced load distribution across experts by
        penalizing situations where some experts receive more load than others.

        Mathematical formulation:
            importance_i = sum_tokens(routing_probs[:, i]) / num_tokens
            load_i = num_tokens_assigned_to_expert_i / expert_capacity
            loss = sum((importance_i * load_i)^2)

        This formulation from Switch Transformer encourages:
        - Each expert to receive a fair share of tokens (importance)
        - Each expert to use its capacity efficiently (load)

        Args:
            routing_probs (torch.Tensor): Routing probabilities of shape (batch_size, num_experts)
                or (sequence_length, num_experts). Must be in range [0, 1].
            expert_assignments (Optional[torch.Tensor]): Hard assignment of tokens to experts,
                shape (batch_size,) or (sequence_length,) with values in [0, num_experts).
                If None, uses argmax of routing_probs.
            expert_capacity (Optional[int]): Maximum capacity per expert. If None, uses
                average number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Scalar tensor containing the load balance loss. Value is >= 0.

        Raises:
            ValueError: If routing_probs dimensions are invalid.

        Example:
            >>> routing_probs = torch.softmax(torch.randn(32, 8), dim=-1)
            >>> expert_assignments = torch.argmax(routing_probs, dim=-1)
            >>> lb_loss = losses.compute_load_balance_loss(
            ...     routing_probs, expert_assignments, expert_capacity=64
            ... )
            >>> print(lb_loss.item())
        """
        # Validate inputs
        if routing_probs.dim() not in (2,):
            raise ValueError(
                f"routing_probs must be 2D (batch_size/seq_len, num_experts), "
                f"got shape {routing_probs.shape}"
            )

        num_tokens = routing_probs.shape[0]
        num_experts = routing_probs.shape[1]

        # Compute importance per expert: fraction of tokens routed to expert
        # Shape: (num_experts,)
        importance = routing_probs.sum(dim=0) / num_tokens

        # Determine expert assignments if not provided
        if expert_assignments is None:
            expert_assignments = torch.argmax(routing_probs, dim=-1)

        # Validate expert_assignments
        if expert_assignments.shape[0] != num_tokens:
            raise ValueError(
                f"expert_assignments must have same length as routing_probs first dimension, "
                f"got {expert_assignments.shape[0]} vs {num_tokens}"
            )

        # Count tokens assigned to each expert
        # Shape: (num_experts,)
        load = torch.zeros(num_experts, dtype=routing_probs.dtype, device=routing_probs.device)
        load.scatter_add_(0, expert_assignments, torch.ones(num_tokens, dtype=routing_probs.dtype, device=routing_probs.device))

        # Normalize load by capacity
        if expert_capacity is None:
            # Use average load as capacity
            expert_capacity = float(num_tokens / num_experts)

        load = load / max(expert_capacity, 1.0)

        # Load balance loss = sum((importance * load)^2)
        lb_loss = torch.sum((importance * load) ** 2)

        return lb_loss

    def compute_entropy_loss(
        self,
        routing_probs: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute Entropy Loss for Router Regularization.

        The entropy loss encourages the router to make confident decisions by
        penalizing high entropy distributions. This is the negative of Shannon entropy,
        which discourages uniform distributions and encourages peaked distributions.

        Mathematical formulation:
            entropy_per_token = -sum_experts(p_i * log(p_i))
            entropy_loss = -mean(entropy_per_token)

        This regularization prevents the router from becoming indecisive and helps
        maintain routing clarity. The sign convention is such that lower entropy
        (more confident routing) results in lower loss values.

        Args:
            routing_probs (torch.Tensor): Routing probabilities of shape (batch_size, num_experts)
                or (sequence_length, num_experts). Must be in range [0, 1] and sum to 1
                across expert dimension.
            expert_mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, num_experts)
                or (sequence_length, num_experts) with 0 for invalid experts, 1 for valid.
                If None, all experts are considered valid.
            temperature (float): Temperature for entropy computation. Higher values increase entropy.
                Must be > 0. Default is 1.0.

        Returns:
            torch.Tensor: Scalar tensor containing the entropy loss (typically negative or small positive).
                This is the negative entropy, so lower values encourage more confident routing.

        Raises:
            ValueError: If routing_probs dimensions are invalid or temperature <= 0.
            RuntimeError: If numerical instability occurs in log computation.

        Example:
            >>> routing_probs = torch.softmax(torch.randn(32, 8), dim=-1)
            >>> entropy_loss = losses.compute_entropy_loss(routing_probs)
            >>> print(entropy_loss.item())
        """
        # Validate inputs
        if routing_probs.dim() not in (2,):
            raise ValueError(
                f"routing_probs must be 2D (batch_size/seq_len, num_experts), "
                f"got shape {routing_probs.shape}"
            )

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        # Apply temperature scaling
        scaled_probs = routing_probs / temperature

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs = torch.log(scaled_probs + epsilon)

        # Apply expert mask if provided
        if expert_mask is not None:
            if expert_mask.shape != routing_probs.shape:
                raise ValueError(
                    f"expert_mask shape {expert_mask.shape} doesn't match "
                    f"routing_probs shape {routing_probs.shape}"
                )
            log_probs = log_probs * expert_mask

        # Compute entropy: -sum(p * log(p))
        entropy_per_token = -torch.sum(scaled_probs * log_probs, dim=-1)

        # Return negative mean entropy (loss that decreases with more confident routing)
        entropy_loss = -torch.mean(entropy_per_token)

        # Validate output
        if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
            logger.warning(
                f"Numerical instability in entropy loss computation. "
                f"Got value: {entropy_loss}. Clamping to 0."
            )
            entropy_loss = torch.tensor(0.0, dtype=routing_probs.dtype, device=routing_probs.device)

        return entropy_loss

    def compute_combined_loss(
        self,
        router_logits: torch.Tensor,
        routing_probs: torch.Tensor,
        expert_assignments: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        z_loss_weight: float = 0.01,
        lb_loss_weight: float = 0.01,
        entropy_loss_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined MoE router losses.

        Convenience method to compute all three router losses and combine them with
        specified weights.

        Args:
            router_logits (torch.Tensor): Router logits of shape (batch_size, num_experts).
            routing_probs (torch.Tensor): Routing probabilities of shape (batch_size, num_experts).
            expert_assignments (Optional[torch.Tensor]): Hard expert assignments.
            expert_mask (Optional[torch.Tensor]): Mask for valid experts.
            z_loss_weight (float): Weight for Z-loss. Default is 0.01.
            lb_loss_weight (float): Weight for load-balance loss. Default is 0.01.
            entropy_loss_weight (float): Weight for entropy loss. Default is 0.0.

        Returns:
            Tuple[torch.Tensor, dict]: Combined loss tensor and dictionary with individual losses:
                - 'total': Combined loss
                - 'z_loss': Z-loss value
                - 'lb_loss': Load-balance loss value
                - 'entropy_loss': Entropy loss value

        Example:
            >>> router_logits = torch.randn(32, 8)
            >>> routing_probs = F.softmax(router_logits, dim=-1)
            >>> combined_loss, loss_dict = losses.compute_combined_loss(
            ...     router_logits, routing_probs,
            ...     z_loss_weight=0.01, lb_loss_weight=0.01
            ... )
            >>> print(f"Total loss: {loss_dict['total']:.4f}")
        """
        z_loss = self.compute_z_loss(router_logits, expert_mask)
        lb_loss = self.compute_load_balance_loss(routing_probs, expert_assignments)
        entropy_loss = self.compute_entropy_loss(routing_probs, expert_mask)

        combined_loss = (
            z_loss_weight * z_loss
            + lb_loss_weight * lb_loss
            + entropy_loss_weight * entropy_loss
        )

        loss_dict = {
            "total": combined_loss.item(),
            "z_loss": z_loss.item(),
            "lb_loss": lb_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        return combined_loss, loss_dict
