"""
LoRA adapter merger for export pipeline.

Merges LoRA adapters with base model weights before export.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .base import BaseExporter

logger = logging.getLogger(__name__)


class MergeAdapterExporter(BaseExporter):
    """
    Exporter that merges LoRA adapters into base model weights.

    This is a pre-processing step for other exporters when working with LoRA checkpoints.
    """

    def prepare_model(self, checkpoint_path: Union[str, Path], **kwargs) -> tuple:
        """
        Load model with LoRA adapters and prepare for merging.

        Args:
            checkpoint_path: Path to checkpoint directory
            **kwargs: Additional arguments

        Returns:
            Tuple of (model, tokenizer)
        """
        checkpoint_path = Path(checkpoint_path)
        model_dir = checkpoint_path / "model"
        if not model_dir.exists():
            model_dir = checkpoint_path

        logger.info(f"Loading model from {model_dir}")

        try:
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype="auto",
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )

            # Check if this is a PEFT model (has adapters)
            if hasattr(model, "peft_config"):
                logger.info("Model has PEFT adapters, will merge them")
            else:
                logger.warning("Model does not have PEFT adapters, returning as-is")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export_model(self, model: tuple, output_path: Union[str, Path], **kwargs) -> str:
        """
        Merge adapters and export merged model.

        Args:
            model: Tuple of (model, tokenizer)
            output_path: Output directory
            **kwargs: Additional export arguments

        Returns:
            Path to merged model directory
        """
        model_obj, tokenizer = model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Merging LoRA adapters...")

        try:
            # Merge adapters if present
            if isinstance(model_obj, PeftModel):
                logger.info("Merging PEFT model...")
                merged_model = model_obj.merge_and_unload()
            else:
                logger.warning("Model is not a PEFT model, no merging needed")
                merged_model = model_obj

            # Save merged model
            logger.info(f"Saving merged model to {output_path}")
            merged_model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}")
            raise

    def export(
        self,
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Execute full merge and export pipeline.

        Args:
            checkpoint_path: Path to checkpoint directory
            output_path: Output path (optional)
            **kwargs: Additional arguments

        Returns:
            Path to merged model directory
        """
        checkpoint_path = Path(checkpoint_path)

        if not self.validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

        if output_path is None:
            output_path = self.output_dir / "merged"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Merging checkpoint from {checkpoint_path}")

        # Prepare model (load with adapters)
        model, tokenizer = self.prepare_model(checkpoint_path, **kwargs)

        # Export (merge and save)
        merged_path = self.export_model((model, tokenizer), output_path, **kwargs)

        logger.info(f"Merge completed: {merged_path}")
        return merged_path
