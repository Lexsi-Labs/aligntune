"""
Export callback for automatic model export on checkpoint save.

Triggers GGUF export or other formats when checkpoints are saved during training.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from .base import TrainerCallback

logger = logging.getLogger(__name__)


class ExportOnSaveCallback(TrainerCallback):
    """
    Callback that automatically exports models on checkpoint save.

    Triggered when training saves a checkpoint, can be configured to auto-export
    to GGUF, Ollama, or HF Hub formats.
    """

    def __init__(
        self,
        export_config: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str] = None,
    ):
        """
        Initialize export callback.

        Args:
            export_config: Export configuration dict with keys:
                - "enabled": bool - whether to enable auto-export
                - "format": str - export format ("gguf", "ollama", "hf_hub")
                - "quantization": str - quantization preset for GGUF
                - "converter": str - converter selection ("llama-cpp", "unsloth")
                - "create_ollama": bool - create Ollama model after GGUF export
                - "repo_id": str - HF Hub repo for HF Hub export
            export_dir: Base directory for exports (if not in config)
        """
        self.export_config = export_config or {}
        self.export_dir = Path(export_dir) if export_dir else None
        self.enabled = self.export_config.get("enabled", False)

        if self.enabled:
            logger.info(f"ExportOnSaveCallback initialized with config: {self.export_config}")
        else:
            logger.info("ExportOnSaveCallback disabled")

    def on_save(self, args, state, control, **kwargs):
        """
        Called after checkpoint save.

        Args:
            args: Training arguments
            state: Training state
            control: Control object
            **kwargs: Additional arguments including 'checkpoint_path'
        """
        if not self.enabled:
            return

        try:
            checkpoint_path = kwargs.get("checkpoint_path")
            if not checkpoint_path:
                logger.warning("No checkpoint_path in on_save callback")
                return

            checkpoint_path = Path(checkpoint_path)
            logger.info(f"ExportOnSaveCallback triggered for {checkpoint_path}")

            # Determine output directory
            export_dir = self.export_dir
            if not export_dir and hasattr(args, "output_dir"):
                export_dir = Path(args.output_dir) / "exports"
            if export_dir:
                export_dir = Path(export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)

            # Get export format
            export_format = self.export_config.get("format", "gguf")

            # Execute export based on format
            artifact_path = self._export_checkpoint(
                checkpoint_path,
                export_format,
                export_dir,
            )

            # Save artifact path to metrics
            if hasattr(state, "metrics") and isinstance(state.metrics, dict):
                state.metrics[f"export_{export_format}_path"] = str(artifact_path)
                logger.info(f"Saved export path to metrics: {artifact_path}")

            logger.info(f"Export callback completed for {checkpoint_path}")

        except Exception as e:
            # Log error but don't crash training
            logger.error(f"Export callback failed (non-blocking): {e}", exc_info=True)

    def _export_checkpoint(
        self,
        checkpoint_path: Path,
        export_format: str,
        export_dir: Optional[Path],
    ) -> Path:
        """
        Execute export for checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            export_format: Target export format
            export_dir: Output directory

        Returns:
            Path to exported artifact
        """
        # Lazy import to avoid circular dependencies
        from ..export import GGUFExporter, OllamaExporter, HFHubExporter

        logger.info(f"Exporting checkpoint to {export_format}")

        if export_format == "gguf":
            exporter = GGUFExporter(
                output_dir=export_dir,
                converter=self.export_config.get("converter"),
                quantization=self.export_config.get("quantization"),
            )
            artifact_path = exporter.export(checkpoint_path)

            # If requested, also export to Ollama
            if self.export_config.get("create_ollama", False):
                logger.info("Creating Ollama model from GGUF")
                ollama_exporter = OllamaExporter(
                    output_dir=export_dir,
                    gguf_path=artifact_path,
                    create_model=True,
                )
                modelfile_path = ollama_exporter.export(gguf_path=artifact_path)
                logger.info(f"Ollama model created: {modelfile_path}")

            return Path(artifact_path)

        elif export_format == "ollama":
            # Ollama export requires GGUF first
            logger.info("Creating GGUF for Ollama export")
            gguf_exporter = GGUFExporter(output_dir=export_dir)
            gguf_path = gguf_exporter.export(checkpoint_path)

            ollama_exporter = OllamaExporter(
                output_dir=export_dir,
                gguf_path=gguf_path,
                create_model=self.export_config.get("create_ollama", True),
            )
            artifact_path = ollama_exporter.export(gguf_path=gguf_path)
            return Path(artifact_path)

        elif export_format == "hf_hub":
            exporter = HFHubExporter(
                output_dir=export_dir,
                repo_id=self.export_config.get("repo_id"),
                adapter_only=self.export_config.get("adapter_only", False),
                private=self.export_config.get("private", False),
            )
            artifact_path = exporter.export(checkpoint_path)
            return Path(artifact_path)

        else:
            raise ValueError(f"Unknown export format: {export_format}")
