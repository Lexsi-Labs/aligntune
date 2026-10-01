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
from aligntune.utils.provenance import build_provenance, provenance_input, write_provenance

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
            adapter_config_path = model_dir / "adapter_config.json"
            if adapter_config_path.exists():
                # Adapter-only checkpoint: load the base model it was trained on
                # and attach the adapter so merge_and_unload() has something to merge.
                import json

                with open(adapter_config_path) as f:
                    base_name = json.load(f).get("base_model_name_or_path")
                base_name = kwargs.get("base_model") or base_name
                if not base_name:
                    raise ValueError(
                        f"{adapter_config_path} has no base_model_name_or_path; "
                        "pass base_model=... to merge this adapter."
                    )
                logger.info(f"Loading base model {base_name} for adapter {model_dir}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                tokenizer_source = (
                    model_dir
                    if any((model_dir / n).exists() for n in ("tokenizer.json", "tokenizer_config.json"))
                    else base_name
                )
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
                # Adapters trained after a vocabulary extension need the base
                # embeddings resized before their weights can be loaded.
                from aligntune.core.merge.peft_merger import _grow_embeddings_for_adapter

                _grow_embeddings_for_adapter(base_model, str(model_dir), tokenizer)
                model = PeftModel.from_pretrained(base_model, str(model_dir))
                logger.info("Model has PEFT adapters, will merge them")
                return model, tokenizer

            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )
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
        write_provenance(
            merged_path,
            build_provenance("merge_lora", inputs=[provenance_input(str(checkpoint_path), "adapter")]),
        )

        logger.info(f"Merge completed: {merged_path}")
        return merged_path
