"""
Weights & Biases (wandb) integration for experiment tracking.

Provides a graceful wrapper around wandb that works seamlessly whether or not
wandb is installed. All methods are no-ops if wandb is unavailable.

This module enables optional experiment tracking across trainers without adding
wandb as a required dependency. Users can opt-in to wandb tracking by specifying
wandb configuration in their training config.

Example:
    >>> config = TrainingConfig(
    ...     wandb_project="my-project",
    ...     wandb_entity="my-entity",
    ...     wandb_tags=["v1", "experiment"]
    ... )
    >>> logger = WandBLogger(
    ...     project=config.wandb_project,
    ...     entity=config.wandb_entity,
    ...     config={"lr": 1e-5, "batch_size": 32},
    ...     tags=config.wandb_tags
    ... )
    >>> logger.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)
    >>> logger.log_artifact("/path/to/checkpoint")
    >>> logger.finalize()
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Graceful wrapper for Weights & Biases experiment tracking.

    Provides a unified interface for logging metrics, artifacts, and configurations
    to wandb. Silently degrades to no-ops if wandb is not installed, allowing code
    to work seamlessly in both scenarios.

    This is useful for optional experiment tracking where wandb shouldn't be a
    required dependency.

    Attributes:
        project: wandb project name
        entity: wandb entity (username or team)
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Optional notes about the run
        is_initialized: Whether wandb.init() was successfully called
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize WandBLogger.

        If wandb is not available or project is None, this logger becomes a no-op.
        All subsequent calls will silently do nothing.

        Args:
            project: wandb project name. If None, wandb tracking is disabled.
            entity: wandb entity (username or team). Optional.
            config: Configuration dictionary to log to wandb. Optional.
            tags: List of tags to attach to the run. Optional.
            notes: Optional notes about the run.
            name: Optional custom run name. If not provided, wandb generates one.

        Example:
            >>> logger = WandBLogger(
            ...     project="my-project",
            ...     entity="my-team",
            ...     config={"learning_rate": 1e-5, "batch_size": 32},
            ...     tags=["baseline", "test"],
            ...     notes="Initial experiment"
            ... )
        """
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.name = name
        self.is_initialized = False
        self.run = None

        # Determine if we should attempt initialization
        if not WANDB_AVAILABLE:
            logger.debug(
                "wandb is not installed. Experiment tracking is disabled. "
                "Install wandb to enable: pip install wandb"
            )
            return

        if project is None:
            logger.debug(
                "wandb_project not specified in config. "
                "Experiment tracking is disabled."
            )
            return

        # Attempt to initialize wandb
        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """
        Initialize wandb.init() with the provided configuration.

        Handles all exceptions gracefully to ensure training isn't disrupted
        if wandb initialization fails (e.g., network issues, permission issues).
        """
        if not WANDB_AVAILABLE or self.project is None:
            return

        try:
            # Build wandb.init kwargs
            init_kwargs = {
                "project": self.project,
                "config": self.config,
                "tags": self.tags,
                "notes": self.notes,
            }

            if self.entity:
                init_kwargs["entity"] = self.entity

            if self.name:
                init_kwargs["name"] = self.name

            # Initialize wandb
            self.run = wandb.init(**init_kwargs)
            self.is_initialized = True

            logger.info(
                f"WandB initialized: project={self.project}, "
                f"entity={self.entity or 'default'}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to initialize wandb: {str(e)}. "
                f"Experiment tracking is disabled, but training will continue."
            )
            self.is_initialized = False
            self.run = None

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to wandb.

        Logs a dictionary of metrics at a specific step. If wandb is not available
        or initialized, this is a silent no-op.

        Args:
            metrics: Dictionary of metric names to values.
                     Values can be scalars, lists, or wandb-compatible types.
            step: Optional step number. If provided, metrics are tagged with this step.

        Example:
            >>> logger.log_metrics({
            ...     "loss": 0.45,
            ...     "accuracy": 0.92,
            ...     "learning_rate": 1e-5
            ... }, step=100)
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        try:
            if step is not None:
                metrics = {**metrics, "step": step}

            self.run.log(metrics, step=step)

        except Exception as e:
            logger.warning(
                f"Failed to log metrics to wandb: {str(e)}. "
                f"Training will continue."
            )

    def log_artifact(
        self,
        path: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
    ) -> None:
        """
        Log an artifact (file or directory) to wandb.

        Useful for saving checkpoints, model weights, or evaluation results.
        If wandb is not available or initialized, this is a silent no-op.

        Args:
            path: Path to the file or directory to log. Can be absolute or relative.
            artifact_type: Type of artifact (e.g., "model", "dataset", "evaluation").
                          Defaults to "model".
            name: Optional custom name for the artifact. If not provided, the
                  filename is used.

        Example:
            >>> logger.log_artifact(
            ...     "/path/to/checkpoint",
            ...     artifact_type="model",
            ...     name="checkpoint-100"
            ... )
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        # Ensure path exists
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        try:
            # Use provided name or derive from path
            artifact_name = name or path_obj.name

            # Create wandb Artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
            )

            # Add file or directory to artifact
            if path_obj.is_file():
                artifact.add_file(str(path), name=artifact_name)
            else:
                artifact.add_dir(str(path), name=artifact_name)

            # Log the artifact
            self.run.log_artifact(artifact)

            logger.debug(f"Logged artifact: {artifact_name} ({artifact_type})")

        except Exception as e:
            logger.warning(
                f"Failed to log artifact to wandb: {str(e)}. "
                f"Training will continue."
            )

    def log_model(
        self,
        path: str,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a model artifact with wandb Model Registry.

        This creates a model artifact that can be tracked in the wandb Model Registry
        for versioning and deployment. If wandb is not available or initialized,
        this is a silent no-op.

        Args:
            path: Path to the model directory or weights.
            name: Model name in the registry. If not provided, uses directory name.
            aliases: List of aliases (e.g., ["latest", "production"]).

        Example:
            >>> logger.log_model(
            ...     "/path/to/model",
            ...     name="my-model",
            ...     aliases=["v1", "latest"]
            ... )
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Model path does not exist: {path}")
            return

        try:
            model_name = name or path_obj.name

            # Create artifact and link to model registry
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                aliases=aliases or [],
            )

            if path_obj.is_file():
                artifact.add_file(str(path))
            else:
                artifact.add_dir(str(path))

            self.run.log_artifact(artifact)
            logger.debug(f"Logged model: {model_name}")

        except Exception as e:
            logger.warning(
                f"Failed to log model to wandb: {str(e)}. "
                f"Training will continue."
            )

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration in an active wandb run.

        Useful for logging dynamic configuration that wasn't available at
        initialization time. If wandb is not available or initialized, this
        is a silent no-op.

        Args:
            updates: Dictionary of configuration updates.

        Example:
            >>> logger.update_config({"num_params": 7_000_000_000})
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        try:
            for key, value in updates.items():
                self.run.config[key] = value
            logger.debug(f"Updated config with {len(updates)} entries")
        except Exception as e:
            logger.warning(
                f"Failed to update wandb config: {str(e)}. "
                f"Training will continue."
            )

    def watch_model(
        self,
        model: Any,
        criterion: Optional[Any] = None,
        log_freq: int = 100,
    ) -> None:
        """
        Initialize wandb to watch model parameters and gradients.

        This enables gradient and parameter logging in wandb. If wandb is not
        available or initialized, this is a silent no-op.

        Args:
            model: PyTorch model to watch.
            criterion: Loss function (optional).
            log_freq: Frequency of logging (default: 100 steps).

        Example:
            >>> logger.watch_model(model, criterion=loss_fn, log_freq=100)
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        try:
            self.run.watch(model, criterion=criterion, log_freq=log_freq)
            logger.debug("Started watching model parameters and gradients")
        except Exception as e:
            logger.warning(
                f"Failed to set up model watching in wandb: {str(e)}. "
                f"Training will continue."
            )

    def finalize(self) -> None:
        """
        Finalize the wandb run.

        Should be called at the end of training to properly close the wandb connection.
        If wandb is not available or initialized, this is a silent no-op.

        Example:
            >>> try:
            ...     # training code
            ... finally:
            ...     logger.finalize()
        """
        if not self.is_initialized or not WANDB_AVAILABLE or self.run is None:
            return

        try:
            self.run.finish()
            self.is_initialized = False
            logger.info("WandB run finished")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {str(e)}")

    def is_enabled(self) -> bool:
        """
        Check if wandb is enabled and initialized.

        Returns:
            True if wandb is available and has been successfully initialized.
            False if wandb is not installed or initialization failed.
        """
        return self.is_initialized and WANDB_AVAILABLE and self.run is not None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures wandb is finalized."""
        self.finalize()
        return False
