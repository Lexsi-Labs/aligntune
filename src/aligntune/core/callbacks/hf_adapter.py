"""
Hugging Face Callback Adapter for Aligntune.

This module provides a bridge between Hugging Face's internal `TrainerCallback`
system and our custom `aligntune` logging framework. By injecting this adapter
into Hugging Face trainers (like `SFTTrainer` or `DPOTrainer`), we can intercept
internal metrics (loss, learning rate, KLD, rewards) and route them to our custom
WandB and console loggers.
"""

import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class HuggingFaceCallbackAdapter(TrainerCallback):
    """
    Adapter that intercepts Hugging Face Trainer events and pipes them to the
    aligntune callback handler and unified logger.
    """

    def __init__(self, callback_handler: Any, unified_logger: Any, aligntune_config: Any):
        """
        Initialize the adapter.

        Args:
            callback_handler: The aligntune CallbackHandler instance.
            unified_logger: The aligntune logger instance (SFTLogger or UnifiedLogger).
            aligntune_config: The aligntune UnifiedConfig/SFTConfig object.
        """
        self.callback_handler = callback_handler
        self.unified_logger = unified_logger
        self.aligntune_config = aligntune_config

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Intercept metric logs from Hugging Face and route to aligntune logger."""
        if logs is None:
            return

        # Pass metrics to custom logger (e.g. WandB, console)
        if hasattr(self.unified_logger, "log_metrics"):
            # HuggingFace step is passed to keep x-axis aligned
            self.unified_logger.log_metrics(logs, step=state.global_step)
        
        # Also trigger the generic on_log hook in aligntune's callback handler
        if hasattr(self.callback_handler, "on_log"):
            # We pass dummy state/control objects since HF manages the real state natively
            # when using trainer.train()
            self.callback_handler.on_log(self.aligntune_config, None, None, logs=logs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Intercept evaluation metrics."""
        if metrics is None:
            return

        if hasattr(self.unified_logger, "log_metrics"):
            self.unified_logger.log_metrics(metrics, step=state.global_step, prefix="eval/")

        if hasattr(self.callback_handler, "on_evaluate"):
            self.callback_handler.on_evaluate(self.aligntune_config, None, None, metrics=metrics)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Intercept checkpoint saves."""
        if hasattr(self.callback_handler, "on_save"):
            self.callback_handler.on_save(self.aligntune_config, None, None)