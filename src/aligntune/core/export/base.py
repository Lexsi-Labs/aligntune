"""
Abstract base exporter class for model export pipeline.
"""

import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """
    Abstract base class for model exporters.

    All exporters should inherit from this class and implement the abstract methods.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the exporter.

        Args:
            output_dir: Base output directory for exported artifacts
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Validate that checkpoint path contains required model files.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            bool: True if checkpoint is valid
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
            return False

        # Check for model directory or model files
        model_dir = checkpoint_path / "model"
        if model_dir.exists():
            # Check for HF model files
            required_files = ["config.json"]
            for file in required_files:
                if not (model_dir / file).exists():
                    logger.warning(f"Missing {file} in checkpoint model directory")
            return True

        # Check for model files in checkpoint root
        if (checkpoint_path / "config.json").exists():
            return True

        logger.error(f"Invalid checkpoint: missing model files in {checkpoint_path}")
        return False

    def get_checkpoint_metadata(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint metadata to determine backend and training info.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        metadata = {}

        # Try to load checkpoint_metadata.json
        metadata_file = checkpoint_path / "checkpoint_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded checkpoint metadata from {metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")

        # Try to load training config
        config_file = checkpoint_path / "training_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    metadata["training_config"] = config
                    # Infer backend from config
                    if "backend" in config:
                        metadata["backend"] = config["backend"]
            except Exception as e:
                logger.warning(f"Failed to load training config: {e}")

        # Check for backend detection from model structure
        if "backend" not in metadata:
            metadata["backend"] = self._detect_backend(checkpoint_path)

        return metadata

    def _detect_backend(self, checkpoint_path: Union[str, Path]) -> str:
        """
        Detect which backend was used for training based on checkpoint structure.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            str: Detected backend name ("trl", "unsloth", "unknown")
        """
        checkpoint_path = Path(checkpoint_path)

        # Check for Unsloth-specific markers
        model_dir = checkpoint_path / "model"
        if model_dir.exists():
            # Unsloth often includes adapter configs if it was trained with LoRA
            if (model_dir / "adapter_config.json").exists():
                logger.info("Detected Unsloth backend (has adapter_config.json)")
                return "unsloth"

        # Check for training config indicators
        config_file = checkpoint_path / "training_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if "unsloth" in str(config).lower():
                        return "unsloth"
                    elif "trl" in str(config).lower():
                        return "trl"
            except:
                pass

        # Default to TRL
        logger.info("Unable to detect backend, defaulting to TRL")
        return "trl"

    @abstractmethod
    def prepare_model(self, checkpoint_path: Union[str, Path], **kwargs) -> Any:
        """
        Prepare model for export (load, merge adapters, etc).

        Args:
            checkpoint_path: Path to checkpoint directory
            **kwargs: Additional preparation arguments

        Returns:
            Prepared model object
        """
        pass

    @abstractmethod
    def export_model(self, model: Any, output_path: Union[str, Path], **kwargs) -> str:
        """
        Export the model to target format.

        Args:
            model: Prepared model object
            output_path: Output path for exported artifact
            **kwargs: Format-specific export arguments

        Returns:
            str: Path to exported artifact
        """
        pass

    def cleanup(self, *paths: Union[str, Path]):
        """
        Clean up temporary files created during export.

        Args:
            *paths: Paths to clean up
        """
        for path in paths:
            path = Path(path)
            if path.exists():
                try:
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    logger.info(f"Cleaned up: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {e}")

    def export(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Execute full export pipeline.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Output path (optional, uses output_dir if not provided)
            **kwargs: Format-specific export arguments

        Returns:
            str: Path to exported artifact
        """
        checkpoint_path = Path(checkpoint_path)

        # Validate checkpoint
        if not self.validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

        # Set output path
        if output_path is None:
            output_path = self.output_dir
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get checkpoint metadata
        metadata = self.get_checkpoint_metadata(checkpoint_path)
        logger.info(f"Checkpoint metadata: {metadata}")

        # Prepare model
        logger.info(f"Preparing model from {checkpoint_path}")
        model = self.prepare_model(checkpoint_path, **kwargs)

        # Export model
        logger.info(f"Exporting model to {output_path}")
        artifact_path = self.export_model(model, output_path, **kwargs)

        logger.info(f"Export completed: {artifact_path}")
        return artifact_path
