"""
GBMPO Configuration - Unified config for all divergence types.

This module provides a single configuration class for GBMPO (Generalized Bregman
Mirror Descent Policy Optimization) that supports multiple divergence types:
- L2: L2 norm regularization in log-space
- L2KL: Dual L2 + KL divergence
- ProbL2: L2 norm in probability space
- ProbL2KL: Dual probability-space L2 + KL

Key Design: Instead of 4 separate config classes, use one with divergence_type parameter.
This reduces duplication and makes adding new variants easier.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from trl import GRPOConfig


# Valid divergence type options
VALID_DIVERGENCE_TYPES = {"l2", "l2kl", "prob_l2", "prob_l2kl"}


@dataclass
class GBMPOConfig(GRPOConfig):
    """
    Unified GBMPO configuration extending TRL's GRPOConfig.

    This single config class handles all GBMPO divergence variants by using
    the divergence_type parameter. All variants inherit the same parent class
    (GRPOConfig) with only divergence-specific parameters differing.

    Schema:
        divergence_type (str): Which divergence to use. Options:
            - "l2": L2 norm regularization in log-space
            - "l2kl": Dual L2 + KL divergence (both λ and β terms)
            - "prob_l2": L2 norm in probability space
            - "prob_l2kl": Dual in probability space
            Default: "l2"

        l2_coefficient (float): Strength of L2 regularization term.
            Controls λ in: loss = base_loss + λ * L2_term
            Typical range: 0.0001 - 0.001
            Default: 0.0001

    Examples:
        # L2 variant
        config = GBMPOConfig(
            divergence_type="l2",
            l2_coefficient=0.0001,
            beta=0.0,  # No KL divergence
        )

        # L2KL variant (dual regularization)
        config = GBMPOConfig(
            divergence_type="l2kl",
            l2_coefficient=0.0001,  # L2 term
            beta=0.01,              # KL term (inherited from GRPOConfig)
        )

    Note:
        - All TRL GRPOConfig parameters are inherited and still available
        - divergence_type is the only new parameter needed to switch variants
        - No variant-specific __init__ needed (unlike old code)
    """

    # Divergence type selector - maps to DivergenceComputer variants
    divergence_type: str = "l2"
    """
    Which Bregman divergence to use for regularization.
    Must be one of: "l2", "l2kl", "prob_l2", "prob_l2kl"
    """

    # L2 regularization coefficient
    l2_coefficient: float = 0.0001
    """
    Strength of L2 regularization: λ in (policy_logp - ref_logp)^2
    - Higher value → policy stays closer to reference model
    - 0.0 disables L2 regularization
    Typical: 0.0001 (math), 0.0005 (code), 0.001 (aggressive)
    """

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate divergence type
        if self.divergence_type not in VALID_DIVERGENCE_TYPES:
            raise ValueError(
                f"Invalid divergence_type '{self.divergence_type}'. "
                f"Must be one of: {VALID_DIVERGENCE_TYPES}"
            )

        # Validate coefficient ranges
        if self.l2_coefficient < 0:
            raise ValueError(
                f"l2_coefficient must be >= 0, got {self.l2_coefficient}"
            )

        if self.l2_coefficient > 1.0:
            raise ValueError(
                f"l2_coefficient very large ({self.l2_coefficient}), "
                f"typical range is 0.0001-0.001. Are you sure?"
            )

        # Log configuration for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"[GBMPOConfig] Initialized with divergence_type='{self.divergence_type}', "
            f"l2_coefficient={self.l2_coefficient}, beta={self.beta}"
        )
