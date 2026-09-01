"""
Corpus loading utilities for tokenization training.

Uses AlignTune's LoaderResolver pattern for consistent data loading
across all data sources (HuggingFace datasets, local files, etc.).
"""

import logging
from typing import List, Optional, Iterator, Union
from datasets import Dataset, DatasetDict, IterableDataset

logger = logging.getLogger(__name__)


class CorpusLoader:
    """
    Loads text corpora for tokenization training using AlignTune's LoaderResolver.

    Supports:
    - HuggingFace datasets (e.g., "wikimedia/wikipedia")
    - Local files (.txt, .json, .jsonl, .csv, .parquet)
    - Directories with text files
    - Streaming datasets for large corpora

    Examples:
        >>> loader = CorpusLoader(
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.hi",
        ...     text_column="text",
        ...     max_samples=10000
        ... )
        >>> corpus = loader.load()
    """

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        max_samples: Optional[int] = None,
        streaming: bool = False,
        **loader_kwargs
    ):
        """
        Initialize corpus loader.

        Args:
            dataset_name: Dataset name/path (HF dataset ID or local path)
            config_name: Dataset configuration (e.g., "20231101.hi" for Wikipedia)
            split: Dataset split to load (default: "train")
            text_column: Column containing text data (default: "text")
            max_samples: Maximum number of samples to load
            streaming: Use streaming for large datasets
            **loader_kwargs: Additional arguments passed to LoaderResolver
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.text_column = text_column
        self.max_samples = max_samples
        self.streaming = streaming
        self.loader_kwargs = loader_kwargs

    def load(self) -> Union[List[str], Iterator[str]]:
        """
        Load corpus as list or iterator of text strings.

        Returns:
            List[str] or Iterator[str] depending on streaming mode
        """
        from aligntune.data.loaders.resolver import LoaderResolver

        logger.info(f"Loading corpus from: {self.dataset_name}")
        if self.config_name:
            logger.info(f"Config: {self.config_name}")
        if self.max_samples:
            logger.info(f"Max samples: {self.max_samples}")

        # Prepare loader kwargs
        loader_kwargs = {
            'config_name': self.config_name,
            'split': self.split,
            'streaming': self.streaming,
            **self.loader_kwargs
        }

        # Remove None values
        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        # Load dataset using LoaderResolver
        loader = LoaderResolver.resolve(self.dataset_name, **loader_kwargs)
        dataset = loader.load()

        # Handle different dataset types
        if isinstance(dataset, DatasetDict):
            if self.split in dataset:
                dataset = dataset[self.split]
            else:
                # Use first available split
                available_splits = list(dataset.keys())
                logger.warning(
                    f"Split '{self.split}' not found. "
                    f"Using '{available_splits[0]}' instead."
                )
                dataset = dataset[available_splits[0]]

        # Extract text from dataset
        if self.streaming or isinstance(dataset, IterableDataset):
            return self._extract_text_streaming(dataset)
        else:
            return self._extract_text_batch(dataset)

    def _extract_text_batch(self, dataset: Dataset) -> List[str]:
        """Extract text from batch dataset."""
        # Check if text column exists
        if self.text_column not in dataset.column_names:
            raise ValueError(
                f"Column '{self.text_column}' not found. "
                f"Available columns: {dataset.column_names}"
            )

        # Apply max_samples limit
        if self.max_samples and len(dataset) > self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples")
            dataset = dataset.select(range(self.max_samples))

        # Extract text column
        texts = dataset[self.text_column]

        logger.info(f"Loaded {len(texts)} text samples")
        return texts

    def _extract_text_streaming(self, dataset: IterableDataset) -> Iterator[str]:
        """Extract text from streaming dataset."""
        logger.info("Using streaming mode")

        count = 0
        for example in dataset:
            if self.text_column not in example:
                raise ValueError(
                    f"Column '{self.text_column}' not found in example. "
                    f"Available columns: {list(example.keys())}"
                )

            yield example[self.text_column]
            count += 1

            if self.max_samples and count >= self.max_samples:
                logger.info(f"Reached max_samples limit: {self.max_samples}")
                break


def load_corpus(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = False,
    **loader_kwargs
) -> Union[List[str], Iterator[str]]:
    """
    Convenience function to load text corpus.

    Args:
        dataset_name: Dataset name/path (HF dataset ID or local path)
        config_name: Dataset configuration (e.g., "20231101.hi" for Wikipedia)
        split: Dataset split to load (default: "train")
        text_column: Column containing text data (default: "text")
        max_samples: Maximum number of samples to load
        streaming: Use streaming for large datasets
        **loader_kwargs: Additional arguments passed to LoaderResolver

    Returns:
        List[str] or Iterator[str] of text samples

    Examples:
        >>> # Load Hindi Wikipedia
        >>> corpus = load_corpus(
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.hi",
        ...     max_samples=10000
        ... )

        >>> # Load from local file
        >>> corpus = load_corpus(
        ...     dataset_name="./hindi_corpus.txt",
        ...     text_column="text"
        ... )

        >>> # Streaming for large datasets
        >>> corpus = load_corpus(
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.hi",
        ...     streaming=True
        ... )
    """
    loader = CorpusLoader(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        text_column=text_column,
        max_samples=max_samples,
        streaming=streaming,
        **loader_kwargs
    )
    return loader.load()
