"""
AlignerSession: Interactive training session management.

Provides Python API for live training inspection, hyperparameter adjustment,
model sampling, and checkpoint rollback during training.
"""

import logging
import time
import threading
import queue
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AlignerState:
    """Current training state snapshot."""
    step: int = 0
    loss: float = 0.0
    reward: float = 0.0
    kl_divergence: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    elapsed_time: float = 0.0
    recent_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Command:
    """Command for trainer control via queue."""
    action: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


class AlignerSession:
    """
    Interactive session wrapper around a trainer.

    Manages trainer lifecycle with pause/resume/stop and provides read-write
    access to training state, hyperparameters, and model.

    Thread-safe: Background training thread + command queue + shared state.
    """

    def __init__(self, trainer):
        """
        Initialize AlignerSession with a trainer.

        Args:
            trainer: TrainerBase instance to wrap
        """
        self.trainer = trainer
        self.state = AlignerState()
        self._state_lock = threading.RLock()

        # Training control
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused
        self._stop_event = threading.Event()

        # Command queue for trainer modifications
        self.command_queue = queue.Queue()

        # Training thread
        self._training_thread: Optional[threading.Thread] = None

        # History tracking
        self._metrics_history: Dict[str, List[float]] = {}
        self._examples_buffer: deque = deque(maxlen=100)

        logger.info(f"AlignerSession initialized with {trainer.__class__.__name__}")

    def start(self) -> None:
        """Start training in background thread."""
        if self._training_thread is not None and self._training_thread.is_alive():
            logger.warning("Training already in progress")
            return

        self._stop_event.clear()
        self._pause_event.set()
        self._training_thread = threading.Thread(
            target=self._train_loop,
            daemon=True,
            name="AlignerTrainingThread"
        )
        self._training_thread.start()
        logger.info("Training started in background thread")

    def pause(self) -> None:
        """Pause training (does not stop it)."""
        self._pause_event.clear()
        logger.info("Training paused")

    def resume(self) -> None:
        """Resume paused training."""
        self._pause_event.set()
        logger.info("Training resumed")

    def stop(self) -> None:
        """Stop training cleanly."""
        self._stop_event.set()
        if self._training_thread:
            self._training_thread.join(timeout=10)
        logger.info("Training stopped")

    def peek(self) -> Dict[str, Any]:
        """
        Get current training state snapshot.

        Returns:
            Dict with step, loss, reward, kl, lr, batch_size, elapsed_time
        """
        with self._state_lock:
            return {
                "step": self.state.step,
                "loss": self.state.loss,
                "reward": self.state.reward,
                "kl_divergence": self.state.kl_divergence,
                "learning_rate": self.state.learning_rate,
                "batch_size": self.state.batch_size,
                "elapsed_time": self.state.elapsed_time,
            }

    def sample(self, prompt: str, max_length: int = 128, num_samples: int = 1) -> List[str]:
        """
        Generate samples from live model.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            num_samples: Number of samples to generate

        Returns:
            List of generated texts
        """
        if self.trainer.model is None:
            logger.warning("Model not yet initialized")
            return []

        self.trainer.model.eval()
        with threading.Lock():  # Prevent training interference
            try:
                inputs = self.trainer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                outputs = self.trainer.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                )

                samples = [
                    self.trainer.tokenizer.decode(
                        output, skip_special_tokens=True
                    )
                    for output in outputs
                ]
                return samples
            except Exception as e:
                logger.error(f"Sampling failed: {e}")
                return []

    def worst_examples(self, n: int = 5) -> List[Tuple[str, str, str, float]]:
        """
        Get worst performing examples (by reward delta).

        Returns:
            List of (prompt, chosen, rejected, reward_delta) tuples
        """
        with self._state_lock:
            if not self._examples_buffer:
                return []

            sorted_examples = sorted(
                self._examples_buffer,
                key=lambda x: x.get("reward_delta", 0.0)
            )

            results = []
            for ex in sorted_examples[:n]:
                results.append((
                    ex.get("prompt", ""),
                    ex.get("chosen", ""),
                    ex.get("rejected", ""),
                    ex.get("reward_delta", 0.0),
                ))
            return results

    def set(self, **hyperparams) -> None:
        """
        Hot-patch trainer hyperparameters.

        Supported: learning_rate, beta, temperature, etc.

        Args:
            **hyperparams: Key-value pairs of hyperparameters to update
        """
        cmd = Command(action="set_hyperparams", kwargs=hyperparams)
        self.command_queue.put(cmd)
        logger.info(f"Hyperparameter update queued: {hyperparams}")

    def rollback(self, step: int) -> None:
        """
        Rollback to earlier checkpoint.

        Args:
            step: Target step to rollback to
        """
        cmd = Command(action="rollback", kwargs={"step": step})
        self.command_queue.put(cmd)
        logger.info(f"Rollback to step {step} queued")

    def history(self) -> Dict[str, List[float]]:
        """
        Get metrics history over time.

        Returns:
            Dict of metric_name -> list of values
        """
        with self._state_lock:
            return {k: list(v) for k, v in self._metrics_history.items()}

    def _update_state(self, metrics: Dict[str, float]) -> None:
        """Update internal state from metrics."""
        with self._state_lock:
            if "loss" in metrics:
                self.state.loss = metrics["loss"]
            if "reward" in metrics:
                self.state.reward = metrics["reward"]
            if "kl_divergence" in metrics or "kl" in metrics:
                kl_key = "kl_divergence" if "kl_divergence" in metrics else "kl"
                self.state.kl_divergence = metrics[kl_key]
            if "learning_rate" in metrics:
                self.state.learning_rate = metrics["learning_rate"]
            if "batch_size" in metrics:
                self.state.batch_size = metrics["batch_size"]

            # Update history
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    if key not in self._metrics_history:
                        self._metrics_history[key] = []
                    self._metrics_history[key].append(val)

    def _add_example(self, example: Dict[str, Any]) -> None:
        """Add example to buffer."""
        with self._state_lock:
            self._examples_buffer.append(example)

    def _train_loop(self) -> None:
        """Main training loop with pause/resume/stop handling."""
        try:
            # Run trainer
            self.trainer.train()
        except Exception as e:
            logger.error(f"Training loop failed: {e}", exc_info=True)
        finally:
            self._stop_event.set()
            logger.info("Training loop completed")

    def is_training(self) -> bool:
        """Check if training is active."""
        return (
            self._training_thread is not None
            and self._training_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def is_paused(self) -> bool:
        """Check if training is paused."""
        return not self._pause_event.is_set()

    def get_trainer(self):
        """Get underlying trainer instance."""
        return self.trainer
