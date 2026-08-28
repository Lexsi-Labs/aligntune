"""
MoALoraLayer: Linear Layer with Multiple LoRA Experts and Dynamic Routing

Wraps PyTorch linear layers with N LoRA experts and a learned router.
Routes tokens to top-k experts and aggregates outputs weighted by routing scores.

Features:
- Multiple LoRA adapters per linear layer
- Dynamic token-to-expert routing
- Learnable routing weights
- Load balance loss for expert utilization
- Compatible with standard PyTorch models and PEFT
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MoALoraLayer(nn.Module):
    """
    Linear layer with Mixture of LoRA Adapters and top-k gating.

    Wraps a linear layer and augments it with N LoRA experts. A learned router
    selects the top-k best experts for each token, and outputs are aggregated
    from expert predictions weighted by routing scores.

    Architecture:
    1. Compute base linear output: W @ x
    2. Route tokens to top-k experts
    3. For each selected expert: apply corresponding LoRA adaptation
    4. Aggregate expert outputs using routing weights
    5. Add to base output

    Args:
        base_module (nn.Module): Base linear layer to augment (must have weight)
        num_experts (int): Number of LoRA experts
        lora_r (int): LoRA rank for all experts
        lora_alpha (int): LoRA scaling factor
        top_k (int): Number of experts to select per token
        router_temp (float): Temperature for router softmax
        use_mlp_router (bool): Use MLP gating instead of linear

    Attributes:
        base_module: The original linear layer
        router: MoARouter instance for expert selection
        lora_a_list: List of LoRA A matrices (input projection)
        lora_b_list: List of LoRA B matrices (output projection)
    """

    def __init__(
        self,
        base_module: nn.Linear,
        num_experts: int,
        lora_r: int,
        lora_alpha: int,
        top_k: int = 2,
        router_temp: float = 1.0,
        use_mlp_router: bool = False,
    ):
        """
        Initialize MoA Layer.

        Args:
            base_module: Base nn.Linear layer to wrap
            num_experts: Number of LoRA experts
            lora_r: LoRA rank
            lora_alpha: LoRA scaling/alpha parameter
            top_k: Number of top experts to select
            router_temp: Temperature for router gating
            use_mlp_router: Use MLP vs linear router

        Raises:
            ValueError: If base_module is not nn.Linear or parameters invalid
            AttributeError: If base_module lacks required weight attribute
        """
        super().__init__()

        if not isinstance(base_module, nn.Linear):
            raise ValueError(
                f"base_module must be nn.Linear, got {type(base_module)}"
            )
        if num_experts <= 0 or lora_r <= 0 or lora_alpha <= 0:
            raise ValueError(
                f"num_experts, lora_r, and lora_alpha must be positive. "
                f"Got: num_experts={num_experts}, lora_r={lora_r}, lora_alpha={lora_alpha}"
            )
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")

        self.base_module = base_module
        self.num_experts = num_experts
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.top_k = top_k
        self.scaling = lora_alpha / lora_r

        # Get dimensions from base module
        in_features = base_module.in_features
        out_features = base_module.out_features

        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"base_module dimensions invalid: in={in_features}, out={out_features}"
            )

        # Initialize router
        from .router import MoARouter

        self.router = MoARouter(
            hidden_dim=in_features,
            num_experts=num_experts,
            top_k=top_k,
            use_mlp=use_mlp_router,
            router_temp=router_temp,
        )

        # Initialize LoRA adapters for each expert
        # Each expert has: A matrix [in_features, r], B matrix [r, out_features]
        self.lora_a_list = nn.ModuleList(
            [nn.Linear(in_features, lora_r, bias=False) for _ in range(num_experts)]
        )
        self.lora_b_list = nn.ModuleList(
            [nn.Linear(lora_r, out_features, bias=False) for _ in range(num_experts)]
        )

        # Initialize LoRA weights with proper scaling
        self._init_lora_weights()

        logger.info(
            f"Initialized MoALoraLayer: in={in_features}, out={out_features}, "
            f"experts={num_experts}, rank={lora_r}, top_k={top_k}"
        )

    def _init_lora_weights(self) -> None:
        """
        Initialize LoRA weights with proper scaling.

        LoRA A: small random values to minimize change at initialization
        LoRA B: zero initialization for stable training
        """
        for lora_a, lora_b in zip(self.lora_a_list, self.lora_b_list):
            # Initialize A with small values
            nn.init.kaiming_uniform_(lora_a.weight, a=0, mode="fan_in")
            # Initialize B to zeros for minimal change at start
            nn.init.zeros_(lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoA routing and expert aggregation.

        Process:
        1. Apply base linear layer to all tokens
        2. Route tokens to top-k experts
        3. Apply each expert's LoRA to their routed tokens
        4. Aggregate expert outputs using routing weights
        5. Combine with base output

        Args:
            hidden_states (Tensor): Input of shape [batch_size, seq_len, in_features]

        Returns:
            Tensor: Output of shape [batch_size, seq_len, out_features]
                Same shape as base linear layer output

        Example:
            >>> base = nn.Linear(768, 3072)
            >>> moa_layer = MoALoraLayer(base, num_experts=4, lora_r=16, lora_alpha=32, top_k=2)
            >>> hidden = torch.randn(2, 10, 768)
            >>> output = moa_layer(hidden)
            >>> output.shape
            torch.Size([2, 10, 3072])
        """
        # Validate input
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected 3D input [batch, seq_len, in_features], "
                f"got {hidden_states.shape}"
            )

        batch_size, seq_len, in_features = hidden_states.shape

        if in_features != self.base_module.in_features:
            raise ValueError(
                f"Input feature dimension {in_features} doesn't match "
                f"base_module in_features {self.base_module.in_features}"
            )

        # Compute base output (without LoRA)
        base_output = self.base_module(hidden_states)

        # Route tokens to experts
        expert_indices, routing_weights = self.router(hidden_states)

        # Compute expert outputs and aggregate
        # Shape of expert_indices: [batch, seq_len, top_k]
        # Shape of routing_weights: [batch, seq_len, top_k]

        # Initialize aggregated expert outputs
        expert_outputs = torch.zeros_like(base_output)

        # Reshape for efficient batch processing
        flat_hidden = hidden_states.reshape(-1, in_features)  # [batch*seq_len, in_features]
        flat_expert_indices = expert_indices.reshape(-1, self.top_k)  # [batch*seq_len, top_k]
        flat_routing_weights = routing_weights.reshape(-1, self.top_k)  # [batch*seq_len, top_k]
        flat_expert_outputs = torch.zeros_like(base_output).reshape(-1, base_output.shape[-1])

        # Process each expert and route tokens to it
        for expert_id in range(self.num_experts):
            # Find positions where this expert is selected
            # mask: [batch*seq_len, top_k] boolean
            mask = flat_expert_indices == expert_id

            if mask.any():
                # Find which tokens use this expert (any position in top_k)
                # expert_token_mask: [batch*seq_len] boolean
                expert_token_mask = mask.any(dim=1)

                if expert_token_mask.any():
                    # Get the input tokens that use this expert
                    expert_hidden = flat_hidden[expert_token_mask]  # [num_tokens, in_features]

                    # Apply LoRA for this expert: A matrix -> B matrix
                    lora_out = self.lora_b_list[expert_id](
                        self.lora_a_list[expert_id](expert_hidden)
                    )  # [num_tokens, out_features]

                    # Extract routing weights for this expert at selected positions
                    # For each token using this expert, get its corresponding weight
                    expert_weights_list = []
                    for token_idx, uses_expert in enumerate(expert_token_mask):
                        if uses_expert:
                            # Find which position in top_k contains this expert
                            weight_indices = torch.where(flat_expert_indices[token_idx] == expert_id)[0]
                            if weight_indices.numel() > 0:
                                expert_weights_list.append(
                                    flat_routing_weights[token_idx, weight_indices[0]]
                                )

                    if expert_weights_list:
                        # Stack weights and apply to expert outputs
                        expert_weights = torch.stack(expert_weights_list)  # [num_tokens]
                        weighted_lora_out = lora_out * expert_weights.unsqueeze(-1)

                        # Accumulate to output at expert token positions
                        flat_expert_outputs[expert_token_mask] += weighted_lora_out

        # Reshape and scale by LoRA alpha/r, then add to base output
        expert_outputs = flat_expert_outputs.reshape_as(expert_outputs)
        output = base_output + expert_outputs * self.scaling

        return output

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Get load balance loss from router.

        Should be added to training loss with a small weight (e.g., 0.01).

        Returns:
            Tensor: Scalar loss value for expert load balancing
        """
        return self.router.get_load_balance_loss()

    def reset_load_balance_loss(self) -> None:
        """
        Reset expert counts for load balance loss computation.

        Should be called at the start of each epoch.
        """
        self.router.reset_expert_counts()

    def extra_repr(self) -> str:
        """Return additional string representation for debugging."""
        return (
            f"in_features={self.base_module.in_features}, "
            f"out_features={self.base_module.out_features}, "
            f"num_experts={self.num_experts}, lora_r={self.lora_r}, top_k={self.top_k}"
        )
