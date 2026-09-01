"""
TRL DAPO Backend - Thin Wrapper over GRPO

DAPO (Difficulty-Aware Policy Optimization) is a GRPO variant that normalizes
gradients by active tokens in the global accumulated batch, removing length bias.

This is a thin wrapper that inherits from TRLGRPOTrainer and sets loss_type='dapo'.

Reference: DAPO paper recommends this normalization for better training stability.
"""

import logging
from typing import Dict, Any
from aligntune.core.rl.config import UnifiedConfig
from ..grpo.grpo import TRLGRPOTrainer

logger = logging.getLogger(__name__)


class TRLDAPOTrainer(TRLGRPOTrainer):
    """
    DAPO trainer - inherits from TRLGRPOTrainer with loss_type='dapo'.

    DAPO normalizes by active tokens in the global accumulated batch (introduced in DAPO paper),
    which removes length bias that exists in standard GRPO normalization.

    All functionality is inherited from TRLGRPOTrainer - this class only overrides
    the default loss_type configuration parameter.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize DAPO trainer with loss_type='dapo'."""
        # Override default loss_type to 'dapo' if not explicitly set
        if getattr(config.train, 'loss_type', None) is None:
            config.train.loss_type = 'dapo'

        # Call parent constructor (does all the work)
        super().__init__(config)

        logger.info("=" * 80)
        logger.info("Initialized TRL DAPO trainer (GRPO with loss_type='dapo')")
        logger.info("DAPO removes length bias by normalizing with active tokens")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and dependencies are available."""
        return TRLGRPOTrainer.is_available()
