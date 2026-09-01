"""
TRL Dr. GRPO Backend - Thin Wrapper over GRPO

Dr. GRPO (Doctor GRPO) is a GRPO variant that normalizes by a global constant
(approximately max_completion_length), removing length bias.

This is a thin wrapper that inherits from TRLGRPOTrainer and sets loss_type='dr_grpo'.

Reference: Dr. GRPO paper introduces this constant normalization approach.
"""

import logging
from typing import Dict, Any
from aligntune.core.rl.config import UnifiedConfig
from ..grpo.grpo import TRLGRPOTrainer

logger = logging.getLogger(__name__)


class TRLDRGRPOTrainer(TRLGRPOTrainer):
    """
    Dr. GRPO trainer - inherits from TRLGRPOTrainer with loss_type='dr_grpo'.

    Dr. GRPO normalizes by a global constant (typically max_completion_length),
    which removes length bias while being simpler than DAPO's batch-based normalization.

    All functionality is inherited from TRLGRPOTrainer - this class only overrides
    the default loss_type configuration parameter.
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize Dr. GRPO trainer with loss_type='dr_grpo'."""
        # Override default loss_type to 'dr_grpo' if not explicitly set
        if getattr(config.train, 'loss_type', None) is None:
            config.train.loss_type = 'dr_grpo'

        # Call parent constructor (does all the work)
        super().__init__(config)

        logger.info("=" * 80)
        logger.info("Initialized TRL Dr. GRPO trainer (GRPO with loss_type='dr_grpo')")
        logger.info("Dr. GRPO removes length bias by normalizing with global constant")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and dependencies are available."""
        return TRLGRPOTrainer.is_available()
