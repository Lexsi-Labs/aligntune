"""
Mixture of Adapters (MoA) Module for AlignTune v3.3 Advanced Parameterization Suite

MoA implements a routing mechanism over multiple LoRA experts with top-k gating.
Each layer routes tokens to the top-k most relevant experts, enabling parameter-efficient
sparse model adaptation.

Architecture:
- MoARouter: Learns a gating function that routes tokens to top-k experts
- MoALoraLayer: Wraps linear layers with multiple LoRA experts and dynamic routing

Key features:
- Per-token top-k expert selection with learnable routing weights
- Load balance loss to encourage uniform expert utilization
- Configurable expert count and top-k selection
- Compatible with standard PyTorch and PEFT workflows

References:
- Mixture of Experts (MoE): Switch Transformers paper
- LoRA: Low-Rank Adaptation of Large Language Models
"""

from .router import MoARouter
from .layer import MoALoraLayer

__all__ = [
    "MoARouter",
    "MoALoraLayer",
]
