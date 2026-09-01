"""
AlignerCallback: Trainer integration for live monitoring.

Hooks into trainer lifecycle to update AlignerSession state and
process interactive commands.
"""

import logging
from typing import Dict, Any, Optional

from ..callbacks import TrainerCallback, TrainerControl

logger = logging.getLogger(__name__)


class AlignerCallback(TrainerCallback):
    """
    Callback for interactive training with AlignerSession.

    Monitors training metrics, example data, and processes commands
    from the AlignerSession.
    """

    def __init__(self, aligner_session=None):
        """
        Initialize callback.

        Args:
            aligner_session: AlignerSession instance to update
        """
        self.aligner_session = aligner_session
        self._step_count = 0

    def on_step_end(self, args, state, control, **kwargs) -> Optional[TrainerControl]:
        """
        Called after each training step.

        Updates state and handles commands from session.
        """
        if self.aligner_session is None:
            return control

        # Update step count
        self._step_count = getattr(state, "step", self._step_count)
        self.aligner_session.state.step = self._step_count

        # Handle pause request
        self._apply_pause()

        # Process commands from session
        self._process_commands(control)

        return control

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        """
        Called after logging.

        Updates metrics in aligner state.
        """
        if self.aligner_session is None or logs is None:
            return control

        # Update state with logged metrics
        self.aligner_session._update_state(logs)

        # Track elapsed time
        import time
        if hasattr(state, "start_time"):
            self.aligner_session.state.elapsed_time = time.time() - state.start_time

        return control

    def on_evaluate(self, args, state, control, metrics: Optional[Dict[str, float]] = None, **kwargs):
        """
        Called after evaluation.

        Updates evaluation metrics.
        """
        if self.aligner_session is None or metrics is None:
            return control

        # Store eval metrics
        self.aligner_session._update_state(metrics)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at training end."""
        if self.aligner_session:
            logger.info("Training completed - callback notified")

        return control

    def _apply_pause(self) -> None:
        """Handle pause request."""
        if self.aligner_session is None:
            return

        # Check if pause is requested
        if self.aligner_session.is_paused():
            logger.debug("Training paused - waiting for resume")
            # Wait for resume signal
            self.aligner_session._pause_event.wait()

    def _process_commands(self, control: TrainerControl) -> None:
        """
        Process commands from aligner session.

        Args:
            control: TrainerControl to modify if needed
        """
        if self.aligner_session is None:
            return

        # Non-blocking queue check
        import queue

        while True:
            try:
                cmd = self.aligner_session.command_queue.get_nowait()
            except queue.Empty:
                break

            try:
                if cmd.action == "set_hyperparams":
                    self._apply_hyperparameter_update(cmd.kwargs)
                elif cmd.action == "rollback":
                    self._handle_rollback_request(cmd.kwargs)
                elif cmd.action == "stop":
                    control.should_training_stop = True
                    logger.info("Stop command received")
            except Exception as e:
                logger.error(f"Error processing command {cmd.action}: {e}")

    def _apply_hyperparameter_update(self, hyperparams: Dict[str, Any]) -> None:
        """
        Apply hyperparameter updates to trainer.

        Modifies training arguments or optimizer state.

        Args:
            hyperparams: Dict of parameter_name -> value
        """
        if not self.aligner_session:
            return

        trainer = self.aligner_session.trainer

        try:
            # Update TrainingArguments if available
            if hasattr(trainer, "config") and hasattr(trainer.config, "train"):
                train_config = trainer.config.train

                if "learning_rate" in hyperparams:
                    train_config.learning_rate = hyperparams["learning_rate"]
                    logger.info(f"Updated learning_rate to {hyperparams['learning_rate']}")

                if "beta" in hyperparams:
                    train_config.beta = hyperparams["beta"]
                    logger.info(f"Updated beta to {hyperparams['beta']}")

                if "temperature" in hyperparams:
                    train_config.temperature = hyperparams["temperature"]
                    logger.info(f"Updated temperature to {hyperparams['temperature']}")

                if "batch_size" in hyperparams:
                    train_config.batch_size = hyperparams["batch_size"]
                    logger.info(f"Updated batch_size to {hyperparams['batch_size']}")

            # Update optimizer learning rate if available
            if hasattr(trainer, "optimizer") and trainer.optimizer and "learning_rate" in hyperparams:
                for param_group in trainer.optimizer.param_groups:
                    param_group["lr"] = hyperparams["learning_rate"]
                logger.info("Updated optimizer learning rate")

        except Exception as e:
            logger.error(f"Failed to apply hyperparameters: {e}")

    def _handle_rollback_request(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle rollback to earlier checkpoint.

        Args:
            kwargs: Dict with 'step' key
        """
        if not self.aligner_session:
            return

        step = kwargs.get("step")
        if step is None:
            logger.warning("Rollback request missing step argument")
            return

        try:
            trainer = self.aligner_session.trainer

            # Look for checkpoint manager
            if hasattr(trainer, "checkpoint_manager"):
                checkpoint_manager = trainer.checkpoint_manager
                checkpoint_path = checkpoint_manager.get_checkpoint(step)

                if checkpoint_path and checkpoint_path.exists():
                    # Load checkpoint
                    logger.info(f"Loading checkpoint from step {step}")
                    trainer.load_checkpoint(checkpoint_path)
                    logger.info(f"Rollback to step {step} completed")
                else:
                    logger.warning(f"Checkpoint for step {step} not found")
            else:
                logger.warning("Trainer does not have checkpoint manager")

        except Exception as e:
            logger.error(f"Rollback to step {step} failed: {e}")
