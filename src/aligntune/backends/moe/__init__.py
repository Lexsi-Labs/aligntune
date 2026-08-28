"""
Mixture of Experts (MoE) support for AlignTune.

Provides:
- Per-expert PEFT adapter management for MoE models
- Router stability losses (Z-loss, load-balance loss, entropy loss)
- MoE-aware SFT trainer with integrated router loss computation

Supports Mixtral, DeepSeek-V2-Lite, and Qwen2.5-MoE architectures.

Modules:
    peft_moe: Apply per-expert LoRA adapters via PEFT
    router_losses: Compute router stability losses
    trainer: MoE-aware SFT trainer

Example:
    >>> from aligntune.backends.moe import PeftMoEWrapper, RouterStabilityLosses, MoESFTTrainer
    >>> wrapper = PeftMoEWrapper(model, experts, config, num_experts=8)
    >>> wrapped = wrapper.apply_per_expert_lora()
    >>> losses = RouterStabilityLosses()
    >>> z_loss = losses.compute_z_loss(router_logits)
    >>> trainer = MoESFTTrainer(model)
    >>> loss = trainer.compute_loss(logits, labels, router_outputs)
"""

try:
    from .peft_moe import PeftMoEWrapper, PeftMoEConfig

    MOE_BACKEND_AVAILABLE = True
except ImportError:
    MOE_BACKEND_AVAILABLE = False
    PeftMoEWrapper = None
    PeftMoEConfig = None

from .router_losses import RouterStabilityLosses
from .trainer import MoESFTTrainer, MoEConfig

__all__ = [
    "PeftMoEWrapper",
    "PeftMoEConfig",
    "MOE_BACKEND_AVAILABLE",
    "RouterStabilityLosses",
    "MoESFTTrainer",
    "MoEConfig",
]
