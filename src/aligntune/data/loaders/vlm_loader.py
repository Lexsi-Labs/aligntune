"""
VLM Data Loader for Vision-Language Model Training.

This module provides a loader for image-text pair datasets used in VLM SFT.
Handles local images, URLs, and HuggingFace datasets with automatic validation.
"""

import logging
from typing import Optional, Any, Dict, List
from pathlib import Path
from datasets import load_dataset, Dataset
from .base import BaseLoader

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class VLMLoader(BaseLoader):
    """Loader for vision-language model datasets with image validation."""

    def __init__(
        self,
        name: str,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        image_column: str = "image",
        text_column: str = "text",
        **kwargs: Any
    ):
        """
        Args:
            name: Dataset name or path (HF repo, local directory, etc.)
            config_name: Dataset configuration/subset
            split: Dataset split to load
            image_column: Column name containing images
            text_column: Column name containing text
            **kwargs: Additional arguments (max_samples, cache_dir, etc.)
        """
        if not PILLOW_AVAILABLE:
            raise ImportError("PIL is required for VLM loading. Install with: pip install pillow")

        self.name = name
        self.config_name = config_name
        self.split = split
        self.image_column = image_column
        self.text_column = text_column
        self.kwargs = kwargs

    def load(self) -> Dataset:
        """Load and validate vision-language dataset."""
        logger.info(f"Loading VLM dataset: {self.name}")

        # Extract AlignTune-specific kwargs
        max_samples = self.kwargs.pop("max_samples", None)

        # Load dataset from HuggingFace
        load_args = [self.name]
        if self.config_name:
            load_args.append(self.config_name)

        try:
            dataset = load_dataset(
                *load_args,
                split=self.split,
                **self.kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.name}: {e}")
            raise

        logger.info(f"Loaded {len(dataset)} samples")

        # Apply max_samples if specified
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")

        # Validate and clean dataset
        dataset = self._validate_images(dataset)

        return dataset

    def _validate_images(self, dataset: Dataset) -> Dataset:
        """Validate images in dataset and skip corrupted ones."""
        logger.info("Validating images...")

        valid_indices = []
        for idx, sample in enumerate(dataset):
            if self.image_column not in sample:
                logger.warning(f"Sample {idx} missing image column '{self.image_column}'")
                continue

            image = sample[self.image_column]

            # Skip if image is None
            if image is None:
                logger.debug(f"Sample {idx} has None image, skipping")
                continue

            # Try to convert to PIL Image if needed
            try:
                if not isinstance(image, Image.Image):
                    # Handle PIL Image objects that may be stored as dicts
                    if isinstance(image, dict) and "path" in image:
                        image = Image.open(image["path"]).convert("RGB")
                    elif isinstance(image, dict) and "bytes" in image:
                        from io import BytesIO
                        image = Image.open(BytesIO(image["bytes"])).convert("RGB")
                    else:
                        # Try direct conversion
                        image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image

                # Validate image is not empty
                if image.size == (0, 0):
                    logger.debug(f"Sample {idx} has empty image, skipping")
                    continue

                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Sample {idx} has invalid image: {e}, skipping")
                continue

        logger.info(f"Valid samples: {len(valid_indices)}/{len(dataset)}")

        if not valid_indices:
            raise ValueError("No valid image-text pairs found in dataset")

        return dataset.select(valid_indices)

    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset to standardized format."""
        logger.info("Preprocessing VLM dataset...")

        def _preprocess_fn(sample):
            """Ensure standard format: {"image": PIL.Image, "text": str, "id": str}"""
            image = sample.get(self.image_column)
            text = sample.get(self.text_column, "")

            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                if isinstance(image, dict):
                    if "path" in image:
                        image = Image.open(image["path"]).convert("RGB")
                    elif "bytes" in image:
                        from io import BytesIO

                        image = Image.open(BytesIO(image["bytes"])).convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")

            # Ensure text is string
            if not isinstance(text, str):
                text = str(text)

            result = {
                "image": image,
                "text": text,
                "id": sample.get("id", str(hash(text)) if text else "unknown"),
            }

            return result

        return dataset.map(_preprocess_fn, remove_columns=dataset.column_names)

    def validate(self) -> bool:
        """Validate that dataset is properly formatted and accessible."""
        try:
            dataset = self.load()
            if len(dataset) == 0:
                logger.error("Dataset is empty")
                return False

            # Check first sample
            first_sample = dataset[0]
            if self.image_column not in first_sample or self.text_column not in first_sample:
                logger.error(f"Required columns missing. Expected: {self.image_column}, {self.text_column}")
                return False

            logger.info("Dataset validation passed")
            return True
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False


__all__ = ["VLMLoader"]
