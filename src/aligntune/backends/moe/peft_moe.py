"""
PEFT adapter wrapper for Mixture of Experts (MoE) models.

This module provides per-expert LoRA adapter application and management for MoE models.
Enables fine-grained control over expert-specific adaptation while maintaining efficiency
through parameter isolation per expert.

Key features:
    - Per-expert LoRA adapter creation and registration
    - Adapter switching for expert selection
    - Parameter extraction and statistics
    - Integration with HuggingFace PEFT library
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from logging import getLogger
from dataclasses import dataclass

logger = getLogger(__name__)


@dataclass
class PeftMoEConfig:
    """Configuration for PEFT MoE wrapper.

    Attributes:
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to within experts.
        bias: Bias type for LoRA ("none", "all", "lora_only").
        task_type: Task type for PEFT ("CAUSAL_LM", "SEQ_2_SEQ_LM", etc.).
    """

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> Dict[str, Any]:
        """Convert to PEFT LoraConfig kwargs.

        Returns:
            Dictionary suitable for LoraConfig initialization.
        """
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules or ["linear"],
        }


class PeftMoEWrapper:
    """
    Applies per-expert LoRA adapters to Mixture of Experts models.

    This wrapper manages the creation, registration, and activation of independent
    LoRA adapters for each expert in a MoE model. Enables efficient multi-expert
    fine-tuning with isolated parameter spaces per expert.

    Attributes:
        model: Reference to the original model.
        expert_modules: Dictionary of expert modules keyed by layer name.
        num_experts: Total number of experts across all layers.
        peft_model: PEFT-wrapped model with adapters applied.
        expert_adapters: Dictionary tracking adapter names per expert.
        lora_config: Configuration dict for LoRA adapters.

    Example:
        >>> from aligntune.backends.moe import ExpertDiscovery, PeftMoEWrapper
        >>> discovery = ExpertDiscovery()
        >>> experts = discovery.discover_experts(model, "mixtral")
        >>> config = PeftMoEConfig(lora_r=16, lora_alpha=32)
        >>> wrapper = PeftMoEWrapper(model, experts, config)
        >>> wrapped_model = wrapper.apply_per_expert_lora()
        >>> wrapper.enable_expert_adapter(expert_id=0)
    """

    def __init__(
        self,
        model: nn.Module,
        expert_modules: Dict[str, List[nn.Module]],
        lora_config: Union[PeftMoEConfig, Dict[str, Any]],
        num_experts: int,
    ):
        """
        Initialize the PEFT MoE wrapper.

        Args:
            model: The MoE transformer model to wrap.
            expert_modules: Dictionary mapping layer identifiers to expert module lists.
                Format: {"layer_0_experts": [expert1, expert2, ...], ...}
            lora_config: PEFT LoRA configuration. Can be PeftMoEConfig instance or dict.
            num_experts: Total number of experts (for validation and tracking).

        Raises:
            ValueError: If expert_modules is empty or num_experts mismatch.
            ImportError: If PEFT library is not available.

        Example:
            >>> config = PeftMoEConfig(lora_r=16)
            >>> wrapper = PeftMoEWrapper(model, expert_dict, config, num_experts=8)
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library required. Install with: pip install peft"
            )

        self.model = model
        self.expert_modules = expert_modules
        self.num_experts = num_experts
        self.peft_model: Optional[nn.Module] = None
        self.expert_adapters: Dict[int, str] = {}
        self.active_expert_id: Optional[int] = None

        # Validate expert modules
        if not expert_modules:
            raise ValueError("expert_modules cannot be empty")

        total_experts = sum(len(experts) for experts in expert_modules.values())
        if total_experts != num_experts:
            logger.warning(
                f"Expert count mismatch: found {total_experts} experts, "
                f"expected {num_experts}. Using discovered count."
            )
            self.num_experts = total_experts

        # Convert config to dict if needed
        if isinstance(lora_config, PeftMoEConfig):
            self.lora_config = lora_config.to_peft_config()
        else:
            self.lora_config = lora_config or {}

        logger.info(
            f"Initialized PeftMoEWrapper for {self.num_experts} experts "
            f"across {len(expert_modules)} MoE layers"
        )

    def apply_per_expert_lora(self) -> nn.Module:
        """
        Apply per-expert LoRA adapters to the model.

        Creates independent LoRA adapters for each expert and registers them
        with the PEFT model. Each expert gets a uniquely named adapter that
        can be independently activated.

        Returns:
            PEFT-wrapped model with all expert adapters applied.

        Raises:
            RuntimeError: If PEFT wrapping fails.
            ValueError: If expert modules cannot be properly adapted.

        Example:
            >>> peft_model = wrapper.apply_per_expert_lora()
            >>> peft_model.print_trainable_parameters()
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library required. Install with: pip install peft"
            )

        logger.info("Applying per-expert LoRA adapters...")

        # Create base PEFT model with default adapter
        try:
            # Create default LoRA config
            default_lora_config = LoraConfig(**self.lora_config)

            # Wrap model with PEFT
            self.peft_model = get_peft_model(self.model, default_lora_config)
            logger.debug("Created base PEFT model with default adapter")

        except Exception as e:
            raise RuntimeError(f"Failed to create base PEFT model: {str(e)}")

        # Add adapters for each expert
        expert_id = 0
        for layer_name, experts_list in self.expert_modules.items():
            for expert_idx, expert_module in enumerate(experts_list):
                adapter_name = f"expert_{expert_id}"

                try:
                    # Create per-expert LoRA config
                    expert_lora_config = LoraConfig(**self.lora_config)

                    # Add adapter for this expert
                    self.peft_model.add_adapter(adapter_name, expert_lora_config)
                    self.expert_adapters[expert_id] = adapter_name

                    logger.debug(
                        f"Added adapter '{adapter_name}' for expert {expert_id} "
                        f"({layer_name}[{expert_idx}])"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to add adapter for expert {expert_id}: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to add expert adapter: {str(e)}"
                    )

                expert_id += 1

        logger.info(
            f"Successfully applied {len(self.expert_adapters)} expert adapters"
        )
        return self.peft_model

    def enable_expert_adapter(self, expert_id: int) -> None:
        """
        Switch to active LoRA adapter for a specific expert.

        Activates the adapter corresponding to the given expert ID, enabling
        its specialized LoRA weights for forward passes. Only one expert adapter
        can be active at a time (per standard PEFT behavior).

        Args:
            expert_id: Index of the expert to activate (0-indexed).

        Raises:
            ValueError: If expert_id is out of range or adapter not found.
            RuntimeError: If model is not yet PEFT-wrapped.

        Example:
            >>> wrapper.enable_expert_adapter(expert_id=3)
            >>> output = peft_model(input_ids)  # Uses expert_3 adapter
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        if expert_id not in self.expert_adapters:
            raise ValueError(
                f"Expert ID {expert_id} not found. "
                f"Valid range: 0-{self.num_experts - 1}"
            )

        adapter_name = self.expert_adapters[expert_id]

        try:
            # Set active adapter in PEFT model
            self.peft_model.set_adapter(adapter_name)
            self.active_expert_id = expert_id

            logger.debug(f"Activated adapter '{adapter_name}' for expert {expert_id}")

        except Exception as e:
            raise RuntimeError(f"Failed to activate adapter: {str(e)}")

    def get_expert_params(
        self, expert_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA parameters for one or all experts.

        Retrieves adapter weights (A and B matrices) for specified expert(s).
        If expert_id is None, returns all expert parameters organized by adapter.

        Args:
            expert_id: Specific expert ID to extract, or None for all experts.

        Returns:
            Dictionary mapping parameter names to tensors. Format depends on expert_id:
            - Single expert: {"lora_A": tensor, "lora_B": tensor, ...}
            - All experts: {"expert_0": {params}, "expert_1": {params}, ...}

        Example:
            >>> # Get params for expert 0
            >>> params = wrapper.get_expert_params(expert_id=0)
            >>> print(params.keys())
            >>> # Get all params
            >>> all_params = wrapper.get_expert_params()
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        if expert_id is not None:
            # Return params for specific expert
            if expert_id not in self.expert_adapters:
                raise ValueError(f"Expert ID {expert_id} not found")

            adapter_name = self.expert_adapters[expert_id]
            expert_params = {}

            # Extract adapter weights
            for name, param in self.peft_model.named_parameters():
                if adapter_name in name:
                    expert_params[name] = param.detach()

            return expert_params

        else:
            # Return params for all experts
            all_params = {}
            for exp_id, adapter_name in self.expert_adapters.items():
                expert_params = {}
                for name, param in self.peft_model.named_parameters():
                    if adapter_name in name:
                        expert_params[name] = param.detach()
                all_params[f"expert_{exp_id}"] = expert_params

            return all_params

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory efficiency.

        Reduces memory usage during training at the cost of recomputing
        activations during backward pass. Useful for large models.

        Example:
            >>> wrapper.enable_gradient_checkpointing()
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            self.peft_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning(
                "Gradient checkpointing not supported for this model"
            )

    def get_trainable_params_count(self) -> int:
        """
        Get total number of trainable parameters across all expert adapters.

        Returns:
            Total count of trainable parameters in all LoRA adapters.

        Example:
            >>> trainable = wrapper.get_trainable_params_count()
            >>> print(f"Trainable parameters: {trainable:,}")
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        trainable = sum(
            p.numel() for p in self.peft_model.parameters() if p.requires_grad
        )
        return trainable

    def get_adapter_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all expert adapters.

        Returns:
            Dictionary with adapter statistics and configuration.

        Example:
            >>> summary = wrapper.get_adapter_summary()
            >>> print(f"Total adapters: {summary['total_adapters']}")
            >>> print(f"Active adapter: {summary['active_adapter']}")
        """
        return {
            "total_adapters": len(self.expert_adapters),
            "num_experts": self.num_experts,
            "moe_layers": len(self.expert_modules),
            "active_adapter": (
                self.expert_adapters.get(self.active_expert_id)
                if self.active_expert_id is not None
                else None
            ),
            "adapter_names": list(self.expert_adapters.values()),
            "lora_config": self.lora_config,
        }

    def save_adapters(self, output_dir: str) -> None:
        """
        Save all expert adapters to disk.

        Saves adapter weights for all experts in separate subdirectories
        for later loading and inference.

        Args:
            output_dir: Directory to save adapters to.

        Raises:
            RuntimeError: If model is not PEFT-wrapped or save fails.

        Example:
            >>> wrapper.save_adapters("./output/moe_adapters")
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        try:
            self.peft_model.save_pretrained(output_dir)
            logger.info(f"Saved all expert adapters to {output_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to save adapters: {str(e)}")

    def load_adapters(self, input_dir: str) -> None:
        """
        Load expert adapters from disk.

        Restores previously saved adapter weights for all experts.

        Args:
            input_dir: Directory containing saved adapters.

        Raises:
            RuntimeError: If model is not PEFT-wrapped or load fails.

        Example:
            >>> wrapper.load_adapters("./output/moe_adapters")
        """
        if self.peft_model is None:
            raise RuntimeError(
                "Model not yet PEFT-wrapped. Call apply_per_expert_lora() first."
            )

        try:
            # Load adapters from directory
            self.peft_model = self.peft_model.from_pretrained(
                self.model, input_dir
            )
            logger.info(f"Loaded expert adapters from {input_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load adapters: {str(e)}")
