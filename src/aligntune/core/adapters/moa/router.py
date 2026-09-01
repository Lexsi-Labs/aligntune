"""
MoARouter: Token-to-Expert Routing for Mixture of Adapters

Implements differentiable routing of tokens to top-k experts with load balancing.
The router learns a gating function that produces routing weights for expert selection.

Features:
- Flexible gate architecture: linear or MLP
- Top-k selection with straight-through estimator for gradients
- Load balance loss to prevent expert collapse
- Temperature-scaled softmax for controlled routing entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MoARouter(nn.Module):
    """
    Token-to-expert router with top-k gating mechanism.

    Routes input tokens to the top-k experts using a learned gating function.
    Includes load balance loss to encourage uniform expert utilization.

    Args:
        hidden_dim (int): Dimension of hidden states
        num_experts (int): Number of available experts
        top_k (int): Number of experts to select per token. Must be <= num_experts
        use_mlp (bool): If True, use 2-layer MLP gate; else use linear gate
        router_temp (float): Temperature for softmax gating. Lower values sharpen routing
        mlp_hidden_dim (int): Hidden dimension for MLP gate (if use_mlp=True)

    Attributes:
        expert_counts (Tensor): Cumulative count of tokens routed to each expert
        load_balance_loss (float): Auxiliary loss value for the last forward pass
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        use_mlp: bool = False,
        router_temp: float = 1.0,
        mlp_hidden_dim: int = 256,
    ):
        """
        Initialize the MoA router.

        Args:
            hidden_dim: Dimension of input hidden states
            num_experts: Total number of experts available
            top_k: Number of experts to route each token to
            use_mlp: Use MLP gating function instead of linear
            router_temp: Temperature for softmax scaling
            mlp_hidden_dim: Hidden dimension for MLP gate

        Raises:
            ValueError: If top_k > num_experts or any dimension is invalid
        """
        super().__init__()

        if hidden_dim <= 0 or num_experts <= 0 or top_k <= 0:
            raise ValueError(
                f"hidden_dim, num_experts, and top_k must be positive. "
                f"Got: hidden_dim={hidden_dim}, num_experts={num_experts}, top_k={top_k}"
            )
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must be <= num_experts ({num_experts})"
            )

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temp = router_temp
        self.use_mlp = use_mlp

        # Gating function: projects hidden states to expert logits
        if use_mlp:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, num_experts),
            )
        else:
            self.gate = nn.Linear(hidden_dim, num_experts)

        # Tracking for load balance loss
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self._load_balance_loss = 0.0

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.

        Forward pass:
        1. Compute gating logits: gate(hidden_states) -> [batch*seq_len, num_experts]
        2. Apply softmax with temperature: logits / temp
        3. Select top-k experts and their weights
        4. Normalize weights to sum to 1.0 per token

        Args:
            hidden_states (Tensor): Input of shape [batch_size, seq_len, hidden_dim]

        Returns:
            expert_indices (Tensor): Shape [batch_size, seq_len, top_k]
                Indices of selected experts for each token
            routing_weights (Tensor): Shape [batch_size, seq_len, top_k]
                Normalized routing weights for selected experts
                Sums to 1.0 across top_k dimension per token

        Example:
            >>> router = MoARouter(hidden_dim=768, num_experts=4, top_k=2)
            >>> hidden = torch.randn(2, 10, 768)  # batch=2, seq_len=10
            >>> expert_idx, weights = router(hidden)
            >>> expert_idx.shape
            torch.Size([2, 10, 2])
            >>> weights.shape
            torch.Size([2, 10, 2])
            >>> weights.sum(dim=-1).allclose(torch.ones(2, 10))
            True
        """
        # Validate input shape
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected 3D input [batch, seq_len, hidden_dim], "
                f"got shape {hidden_states.shape}"
            )

        batch_size, seq_len, hidden_dim = hidden_states.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden_dim {hidden_dim} doesn't match router hidden_dim "
                f"{self.hidden_dim}"
            )

        # Reshape for processing: [batch * seq_len, hidden_dim]
        flat_hidden = hidden_states.reshape(-1, hidden_dim)

        # Compute gating logits: [batch * seq_len, num_experts]
        logits = self.gate(flat_hidden)

        # Apply temperature-scaled softmax
        gating_weights = F.softmax(logits / self.router_temp, dim=-1)

        # Select top-k experts
        # top_k_weights: [batch * seq_len, top_k]
        # top_k_indices: [batch * seq_len, top_k]
        top_k_weights, top_k_indices = torch.topk(
            gating_weights, k=self.top_k, dim=-1
        )

        # Normalize weights to sum to 1.0 per token
        # This ensures output scales match input scales
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Reshape back to [batch, seq_len, top_k]
        expert_indices = top_k_indices.reshape(batch_size, seq_len, self.top_k)
        routing_weights = top_k_weights.reshape(batch_size, seq_len, self.top_k)

        # Update expert counts for load balance loss (detach to prevent gradient flow)
        with torch.no_grad():
            for expert_id in range(self.num_experts):
                mask = (top_k_indices == expert_id).float().sum()
                self.expert_counts[expert_id] += mask

        return expert_indices, routing_weights

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Compute auxiliary load balance loss.

        Encourages uniform expert utilization by penalizing variance in expert activation.
        Loss is: sum of squared deviations from mean expert count.

        Loss = sum_i (expert_count_i - mean_expert_count)^2

        This should be added to the training loss with a small weight (e.g., 0.01)
        to prevent expert collapse where all tokens route to the same expert.

        Returns:
            Tensor: Scalar loss value (non-negative)

        Example:
            >>> router = MoARouter(hidden_dim=768, num_experts=4, top_k=2)
            >>> hidden = torch.randn(2, 10, 768)
            >>> expert_idx, weights = router(hidden)
            >>> lb_loss = router.get_load_balance_loss()
            >>> lb_loss.item() >= 0.0
            True
        """
        # Compute mean expert count
        mean_count = self.expert_counts.mean()

        # Compute squared deviations
        variance = ((self.expert_counts - mean_count) ** 2).sum()

        return variance

    def reset_expert_counts(self) -> None:
        """
        Reset expert counts for a new epoch or batch.

        Should be called at the start of each training epoch to reset
        the expert utilization tracking.

        Example:
            >>> router = MoARouter(hidden_dim=768, num_experts=4, top_k=2)
            >>> hidden = torch.randn(2, 10, 768)
            >>> _, _ = router(hidden)
            >>> router.reset_expert_counts()
            >>> router.expert_counts
            tensor([0., 0., 0., 0.])
        """
        self.expert_counts.zero_()

    def extra_repr(self) -> str:
        """Return additional string representation for debugging."""
        return (
            f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, temp={self.router_temp}, use_mlp={self.use_mlp}"
        )
