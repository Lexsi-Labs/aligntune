"""
HuggingFace Hub exporter for model export pipeline.

Uploads models or adapters directly to HuggingFace Hub.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union
from huggingface_hub import HfApi, HfFolder
from .base import BaseExporter

logger = logging.getLogger(__name__)


class HFHubExporter(BaseExporter):
    """
    Exporter for uploading models to HuggingFace Hub.

    Supports full model uploads or adapter-only uploads.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        adapter_only: bool = False,
        private: bool = False,
        token: Optional[str] = None,
    ):
        """
        Initialize HF Hub exporter.

        Args:
            output_dir: Local base output directory (for downloads)
            repo_id: Target HF Hub repository ID (e.g., "username/model-name")
            adapter_only: Upload only LoRA adapters, not full weights
            private: Make uploaded repository private
            token: HuggingFace API token (uses HF_TOKEN env var if not provided)
        """
        super().__init__(output_dir)
        self.repo_id = repo_id
        self.adapter_only = adapter_only
        self.private = private
        self.token = token or os.environ.get("HF_TOKEN") or HfFolder.get_token()

        if not self.token:
            logger.warning(
                "No HF token found. Set HF_TOKEN environment variable or "
                "login with 'huggingface-cli login'"
            )

    def _get_adapter_files(self, checkpoint_path: Union[str, Path]) -> list:
        """
        Find LoRA adapter files in checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            List of adapter file paths
        """
        checkpoint_path = Path(checkpoint_path)
        adapter_files = []

        # Check for adapter_config.json and adapter weights
        required_files = ["adapter_config.json"]
        optional_files = ["adapter_model.bin", "adapter_model.safetensors"]

        for file in required_files + optional_files:
            file_path = checkpoint_path / "model" / file
            if not file_path.exists():
                file_path = checkpoint_path / file
            if file_path.exists():
                adapter_files.append(file_path)

        logger.info(f"Found {len(adapter_files)} adapter files")
        return adapter_files

    def prepare_model(self, checkpoint_path: Union[str, Path], **kwargs) -> Union[str, Path]:
        """
        For HF Hub, "prepare_model" means validate checkpoint content.

        Args:
            checkpoint_path: Path to checkpoint directory
            **kwargs: Additional arguments

        Returns:
            Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        if self.adapter_only:
            # Check for adapter files
            adapter_files = self._get_adapter_files(checkpoint_path)
            if not adapter_files:
                raise ValueError(f"No adapter files found in {checkpoint_path}")
            logger.info("Validated adapter-only checkpoint")
        else:
            # Check for model files
            model_dir = checkpoint_path / "model"
            if not model_dir.exists():
                model_dir = checkpoint_path

            if not (model_dir / "config.json").exists():
                raise ValueError(f"No model config found in {checkpoint_path}")
            logger.info("Validated full model checkpoint")

        return checkpoint_path

    def export_model(
        self, checkpoint_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, **kwargs
    ) -> str:
        """
        Upload model or adapters to HF Hub.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Not used (ignored)
            **kwargs: Additional arguments

        Returns:
            URL to uploaded model on HF Hub
        """
        checkpoint_path = Path(checkpoint_path)

        if not self.repo_id:
            raise ValueError("repo_id must be provided for HF Hub upload")

        if not self.token:
            raise ValueError("HuggingFace token required for upload")

        api = HfApi(token=self.token)

        if self.adapter_only:
            logger.info(f"Uploading adapters to {self.repo_id}")
            # Upload only adapter files
            adapter_files = self._get_adapter_files(checkpoint_path)
            for adapter_file in adapter_files:
                logger.info(f"Uploading {adapter_file.name}...")
                api.upload_file(
                    path_or_fileobj=str(adapter_file),
                    path_in_repo=adapter_file.name,
                    repo_id=self.repo_id,
                    private=self.private,
                )
        else:
            logger.info(f"Uploading full model to {self.repo_id}")
            # Upload full checkpoint directory
            model_dir = checkpoint_path / "model"
            if not model_dir.exists():
                model_dir = checkpoint_path

            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=self.repo_id,
                private=self.private,
            )

            # Upload tokenizer if available
            tokenizer_dir = checkpoint_path / "tokenizer"
            if tokenizer_dir.exists():
                logger.info("Uploading tokenizer...")
                for file in tokenizer_dir.glob("*"):
                    if file.is_file():
                        api.upload_file(
                            path_or_fileobj=str(file),
                            path_in_repo=file.name,
                            repo_id=self.repo_id,
                            private=self.private,
                        )

        hub_url = f"https://huggingface.co/{self.repo_id}"
        logger.info(f"Upload completed: {hub_url}")
        return hub_url

    def export(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Execute full HF Hub export pipeline.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Not used
            repo_id: Target repository ID (overrides self.repo_id)
            **kwargs: Additional arguments

        Returns:
            URL to uploaded model on HF Hub
        """
        checkpoint_path = Path(checkpoint_path)

        if not self.validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

        if repo_id:
            self.repo_id = repo_id

        logger.info(
            f"Exporting to HF Hub (repo: {self.repo_id}, "
            f"adapter_only: {self.adapter_only})"
        )

        # Prepare (validate)
        checkpoint_path = self.prepare_model(checkpoint_path, **kwargs)

        # Export (upload)
        hub_url = self.export_model(checkpoint_path, output_path, **kwargs)

        logger.info(f"HF Hub export completed: {hub_url}")
        return hub_url
