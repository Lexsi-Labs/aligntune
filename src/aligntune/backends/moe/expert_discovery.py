"""
Expert layer discovery for Mixture of Experts (MoE) architectures.

This module detects expert linear layers across different MoE model architectures
(Mixtral, DeepSeek-V2-Lite, Qwen2.5-MoE) and provides utilities for identifying
and quantifying expert parameters.

Supported architectures:
    - Mixtral-8x7B/8x22B: Uses block_sparse_moe with experts
    - DeepSeek-V2-Lite: Uses mlp.experts structure
    - Qwen2.5-MoE: Uses mlp.experts structure
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from logging import getLogger

logger = getLogger(__name__)


class ExpertDiscovery:
    """
    Discovers expert layers in Mixture of Experts models.

    This class provides methods to identify expert linear layers in different MoE
    architectures, enabling per-expert PEFT adapter application. Supports automatic
    architecture detection and architecture-specific discovery patterns.

    Attributes:
        supported_architectures: Set of supported MoE model architectures.
    """

    supported_architectures = {
        "mixtral",
        "deepseek-v2-lite",
        "qwen2.5-moe",
        "mixtral-moe",
        "deepseek_v2_lite",
        "qwen_moe",
    }

    def __init__(self):
        """Initialize ExpertDiscovery with supported architectures."""
        self.architecture = None
        self.experts_cache: Dict[str, Any] = {}

    def discover_experts(
        self, model: nn.Module, architecture: Optional[str] = None
    ) -> Dict[str, List[nn.Module]]:
        """
        Discover expert modules in a MoE model.

        Detects expert linear layers organized by their layer index. Automatically
        identifies architecture if not provided by examining model structure.

        Args:
            model: The MoE transformer model to analyze.
            architecture: Model architecture name (e.g., "mixtral", "deepseek-v2-lite").
                If None, attempts automatic detection.

        Returns:
            Dictionary mapping layer identifiers to lists of expert modules.
            Format: {"layer_0_experts": [expert_modules], "layer_1_experts": [...], ...}
            Returns empty dict if no experts found or architecture unsupported.

        Raises:
            ValueError: If architecture is provided but not supported.

        Example:
            >>> discovery = ExpertDiscovery()
            >>> experts = discovery.discover_experts(model, "mixtral")
            >>> for layer_name, expert_list in experts.items():
            ...     print(f"{layer_name}: {len(expert_list)} experts")
        """
        # Validate or detect architecture
        if architecture is None:
            architecture = self._detect_architecture(model)
        else:
            architecture = architecture.lower().strip()
            if architecture not in self.supported_architectures:
                raise ValueError(
                    f"Unsupported architecture: {architecture}. "
                    f"Supported: {self.supported_architectures}"
                )

        self.architecture = architecture

        # Route to architecture-specific discovery
        if architecture in ("mixtral", "mixtral-moe"):
            return self._discover_mixtral_experts(model)
        elif architecture in ("deepseek-v2-lite", "deepseek_v2_lite"):
            return self._discover_deepseek_experts(model)
        elif architecture in ("qwen2.5-moe", "qwen_moe"):
            return self._discover_qwen_experts(model)
        else:
            logger.warning(f"Unknown MoE architecture: {architecture}")
            return {}

    def _discover_mixtral_experts(self, model: nn.Module) -> Dict[str, List[nn.Module]]:
        """
        Discover expert layers in Mixtral models.

        Mixtral architecture organizes experts under:
        model.layers[i].block_sparse_moe.experts

        Each expert is a linear layer (typically Linear or a composite module).

        Args:
            model: Mixtral-based model.

        Returns:
            Dictionary of expert modules organized by layer index.

        Example:
            >>> experts = discovery._discover_mixtral_experts(mixtral_model)
            >>> layer_0_experts = experts.get("layer_0_experts", [])
        """
        experts_dict: Dict[str, List[nn.Module]] = {}

        # Access model layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            logger.warning("Could not find layers in Mixtral model")
            return {}

        # Iterate through layers to find MoE blocks
        for layer_idx, layer in enumerate(layers):
            # Check for block_sparse_moe structure
            if hasattr(layer, "block_sparse_moe"):
                moe_block = layer.block_sparse_moe
                if hasattr(moe_block, "experts"):
                    experts = moe_block.experts
                    if isinstance(experts, (nn.ModuleDict, nn.ModuleList)) or hasattr(
                        experts, "__iter__"
                    ):
                        # Extract expert modules
                        expert_list = []
                        if isinstance(experts, nn.ModuleDict):
                            expert_list = list(experts.values())
                        elif isinstance(experts, (nn.ModuleList, list)):
                            expert_list = list(experts)
                        else:
                            expert_list = [
                                getattr(experts, str(i))
                                for i in range(len(experts))
                                if hasattr(experts, str(i))
                            ]

                        if expert_list:
                            key = f"layer_{layer_idx}_experts"
                            experts_dict[key] = expert_list
                            logger.debug(
                                f"Found {len(expert_list)} experts in {key}"
                            )

        return experts_dict

    def _discover_deepseek_experts(self, model: nn.Module) -> Dict[str, List[nn.Module]]:
        """
        Discover expert layers in DeepSeek-V2-Lite models.

        DeepSeek-V2-Lite architecture organizes experts under:
        model.layers[i].mlp.experts

        Experts are typically organized as a ModuleList or ModuleDict.

        Args:
            model: DeepSeek-V2-Lite based model.

        Returns:
            Dictionary of expert modules organized by layer index.

        Example:
            >>> experts = discovery._discover_deepseek_experts(deepseek_model)
            >>> layer_0_experts = experts.get("layer_0_experts", [])
        """
        experts_dict: Dict[str, List[nn.Module]] = {}

        # Access model layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            logger.warning("Could not find layers in DeepSeek model")
            return {}

        # Iterate through layers to find MoE/MLP blocks
        for layer_idx, layer in enumerate(layers):
            # Check for mlp.experts structure
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                experts = layer.mlp.experts
                if isinstance(experts, (nn.ModuleDict, nn.ModuleList)) or hasattr(
                    experts, "__iter__"
                ):
                    # Extract expert modules
                    expert_list = []
                    if isinstance(experts, nn.ModuleDict):
                        expert_list = list(experts.values())
                    elif isinstance(experts, (nn.ModuleList, list)):
                        expert_list = list(experts)
                    else:
                        expert_list = [
                            getattr(experts, str(i))
                            for i in range(len(experts))
                            if hasattr(experts, str(i))
                        ]

                    if expert_list:
                        key = f"layer_{layer_idx}_experts"
                        experts_dict[key] = expert_list
                        logger.debug(
                            f"Found {len(expert_list)} experts in {key}"
                        )

        return experts_dict

    def _discover_qwen_experts(self, model: nn.Module) -> Dict[str, List[nn.Module]]:
        """
        Discover expert layers in Qwen2.5-MoE models.

        Qwen2.5-MoE architecture organizes experts under:
        model.layers[i].mlp.experts

        This is structurally similar to DeepSeek but may have different activation patterns.

        Args:
            model: Qwen2.5-MoE based model.

        Returns:
            Dictionary of expert modules organized by layer index.

        Example:
            >>> experts = discovery._discover_qwen_experts(qwen_model)
            >>> layer_0_experts = experts.get("layer_0_experts", [])
        """
        # Qwen2.5-MoE uses the same discovery pattern as DeepSeek-V2-Lite
        return self._discover_deepseek_experts(model)

    def _detect_architecture(self, model: nn.Module) -> Optional[str]:
        """
        Automatically detect MoE model architecture.

        Examines model structure to identify the MoE architecture type.
        Checks for characteristic attribute patterns of each supported architecture.

        Args:
            model: Model to identify.

        Returns:
            Architecture name (lowercase) or None if detection fails.

        Detection logic:
            1. Checks for block_sparse_moe -> "mixtral"
            2. Checks for mlp.experts in DeepSeek-style layout -> detects via config
            3. Falls back to model name/config analysis
        """
        # Access model layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            logger.warning("Could not find layers for architecture detection")
            return None

        # Check first few layers for architecture-specific patterns
        for layer in list(layers)[:3]:  # Check first 3 layers
            if hasattr(layer, "block_sparse_moe"):
                return "mixtral"
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                # Could be DeepSeek or Qwen, check config
                if hasattr(model, "config"):
                    config_model = getattr(model.config, "model_type", "").lower()
                    if "deepseek" in config_model:
                        return "deepseek-v2-lite"
                    elif "qwen" in config_model:
                        return "qwen2.5-moe"
                # Default to DeepSeek pattern
                return "deepseek-v2-lite"

        # Check model name/type for hints
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            model_type = model.config.model_type.lower()
            if "mixtral" in model_type:
                return "mixtral"
            elif "deepseek" in model_type:
                return "deepseek-v2-lite"
            elif "qwen" in model_type:
                return "qwen2.5-moe"

        return None

    def get_num_experts(
        self, model: nn.Module, architecture: Optional[str] = None
    ) -> int:
        """
        Get total number of experts across all layers.

        Discovers experts and counts total expert modules. Assumes consistent
        number of experts per layer (standard for most MoE models).

        Args:
            model: The MoE model.
            architecture: Model architecture (optional, auto-detected if None).

        Returns:
            Total number of experts. Returns 0 if no experts found.

        Example:
            >>> num_experts = discovery.get_num_experts(model, "mixtral")
            >>> print(f"Total experts: {num_experts}")
        """
        experts_dict = self.discover_experts(model, architecture)

        if not experts_dict:
            logger.warning("No experts found in model")
            return 0

        # Get expert count from first layer (assumes uniform distribution)
        first_layer_experts = next(iter(experts_dict.values()), [])
        if first_layer_experts:
            num_experts_per_layer = len(first_layer_experts)
            num_layers = len(experts_dict)
            total_experts = num_experts_per_layer * num_layers
            logger.info(
                f"Detected {num_experts_per_layer} experts per layer "
                f"across {num_layers} MoE layers ({total_experts} total)"
            )
            return total_experts

        return 0

    def get_expert_layer_count(self, model: nn.Module) -> int:
        """
        Get number of layers with experts.

        Args:
            model: The MoE model.

        Returns:
            Number of layers containing experts.

        Example:
            >>> num_moe_layers = discovery.get_expert_layer_count(model)
            >>> print(f"Model has {num_moe_layers} MoE layers")
        """
        experts_dict = self.discover_experts(model, self.architecture)
        return len(experts_dict)

    def get_expert_names(
        self, model: nn.Module, architecture: Optional[str] = None
    ) -> List[str]:
        """
        Get list of all expert layer identifiers.

        Args:
            model: The MoE model.
            architecture: Model architecture (optional, auto-detected if None).

        Returns:
            List of expert layer identifiers (e.g., ["layer_0_experts", "layer_1_experts", ...]).

        Example:
            >>> expert_names = discovery.get_expert_names(model, "mixtral")
            >>> for name in expert_names:
            ...     print(f"Expert group: {name}")
        """
        experts_dict = self.discover_experts(model, architecture)
        return list(experts_dict.keys())

    def get_experts_by_layer(
        self, model: nn.Module, layer_idx: int, architecture: Optional[str] = None
    ) -> Optional[List[nn.Module]]:
        """
        Get expert modules for a specific layer.

        Args:
            model: The MoE model.
            layer_idx: Index of the layer to retrieve experts for.
            architecture: Model architecture (optional, auto-detected if None).

        Returns:
            List of expert modules for the layer, or None if layer not found.

        Example:
            >>> experts = discovery.get_experts_by_layer(model, 0, "mixtral")
            >>> if experts:
            ...     print(f"Layer 0 has {len(experts)} experts")
        """
        experts_dict = self.discover_experts(model, architecture)
        key = f"layer_{layer_idx}_experts"
        return experts_dict.get(key)
