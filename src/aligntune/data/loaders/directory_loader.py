import logging
from pathlib import Path
from typing import Iterable, Optional, List
from datasets import DatasetDict, Dataset, concatenate_datasets
from .json_loader import JSONLoader
from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .markdown_loader import MarkdownLoader
from .base import BaseLoader

logger = logging.getLogger(__name__)


class DirectoryLoader(BaseLoader):
    """
    Loader for directories containing multiple file types.

    Supports auto-routing files to appropriate loaders based on file extension:
    - .json, .jsonl → JSONLoader
    - .csv → CSVLoader
    - .parquet → ParquetLoader
    - .pdf → PDFLoader
    - .docx → DocxLoader
    - .md → MarkdownLoader
    """

    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".parquet": "parquet",
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown",
    }

    # Metadata that CuratorKIT (and other Lexsi tools) write next to the data.
    # These are never data splits.
    IGNORE_FILES = (
        "manifest.json",
        "rejected.jsonl",
        "dataset_card.md",
        "lexsi_provenance.json",
        "diagnostic_summary.json",
    )

    def __init__(
        self,
        path: str,
        pattern: Optional[str] = None,
        recurse: bool = True,
        config_name: Optional[str] = None,
        ignore_files: Optional[Iterable[str]] = None,
    ):
        """
        Initialize DirectoryLoader.

        Args:
            path: Path to directory
            pattern: Optional glob pattern to filter files (e.g., "*.md")
            recurse: Whether to recursively search subdirectories
            config_name: Only load data files with this stem (e.g. "dpo" for
                ``train/dpo.jsonl`` + ``val/dpo.jsonl``).
            ignore_files: File names never loaded as data. Defaults to
                ``IGNORE_FILES``.
        """
        self.path = Path(path)
        self.pattern = pattern or "*"
        self.recurse = recurse
        self.config_name = config_name
        self.ignore_files = set(self.IGNORE_FILES if ignore_files is None else ignore_files)

    def load(self) -> DatasetDict | Dataset:
        """Load all files from directory and return as DatasetDict or Dataset."""
        return self.load_directory()

    def load_directory(self) -> DatasetDict | Dataset:
        """
        Load all supported files from the directory.

        Returns:
            DatasetDict mapping split names (see ``_split_key``) to Datasets
        """
        if not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

        glob_func = self.path.rglob if self.recurse else self.path.glob
        files = [
            f for f in glob_func(self.pattern)
            if f.name not in self.ignore_files
            and (self.config_name is None or f.stem == self.config_name)
        ]

        if not files:
            raise ValueError(f"No files matching pattern '{self.pattern}' found in {self.path}")

        datasets = {}
        raw_data = []

        for file_path in sorted(files):
            if not file_path.is_file():
                continue
            try:
                loader = self._get_loader_for_file(file_path)
                if loader is None:
                    logger.warning(f"No loader found for {file_path}, skipping")
                    continue

                dataset = loader.load()
                # datasets.load_dataset("json", data_files=...) returns a DatasetDict
                # with a single "train" split; each file here is one split of ours.
                if isinstance(dataset, DatasetDict):
                    dataset = dataset["train"] if "train" in dataset else concatenate_datasets(list(dataset.values()))
                if dataset is None or len(dataset) == 0:
                    logger.warning(f"No data loaded from {file_path}")
                    continue
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

            key = self._split_key(file_path)
            if key in datasets:
                raise ValueError(
                    f"More than one file in {self.path} maps to split {key!r}; "
                    "pass config_name=<file stem> or a single file"
                )
            datasets[key] = dataset
            raw_data.append(dataset)

        if not datasets:
            raise ValueError(f"No supported files found in {self.path}")

        # For raw document formats (pdf, docx, md), merge into single dataset
        # For structured formats (json, csv, parquet), return as DatasetDict
        if raw_data and all(self._is_raw_format(f) for f in files if f.is_file()):
            # All files are raw format - merge into single dataset
            return concatenate_datasets(raw_data)

        # Return DatasetDict for mixed or structured formats
        return DatasetDict(datasets)

    def _split_key(self, file_path: Path) -> str:
        """Split name for one data file.

        A structured file in a subdirectory (``train/dpo.jsonl``,
        ``val/dpo.jsonl``) is that directory's split. A top-level file is keyed
        by its stem, or is ``train`` when ``config_name`` selected it. Raw
        documents are merged later, so they keep their relative path.
        """
        if self._is_raw_format(file_path):
            return file_path.relative_to(self.path).as_posix()
        if file_path.parent != self.path:
            return file_path.parent.name
        return "train" if self.config_name is not None else file_path.stem

    def _get_loader_for_file(self, file_path: Path):
        """
        Get appropriate loader for a file based on extension.

        Args:
            file_path: Path to file

        Returns:
            Loader instance or None if no loader found
        """
        suffix = file_path.suffix.lower()

        if suffix in [".json", ".jsonl"]:
            return JSONLoader(str(file_path))
        elif suffix == ".csv":
            return CSVLoader(str(file_path))
        elif suffix == ".parquet":
            return ParquetLoader(str(file_path))
        elif suffix == ".pdf":
            return PDFLoader(str(file_path))
        elif suffix == ".docx":
            return DocxLoader(str(file_path))
        elif suffix == ".md":
            return MarkdownLoader(str(file_path))
        else:
            return None

    def _is_raw_format(self, file_path: Path) -> bool:
        """
        Check if file is a raw document format (vs structured data format).

        Args:
            file_path: Path to file

        Returns:
            True if file is raw format (pdf, docx, md)
        """
        suffix = file_path.suffix.lower()
        return suffix in [".pdf", ".docx", ".md"]

    @staticmethod
    def supported_formats() -> List[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return list(DirectoryLoader.SUPPORTED_FORMATS.keys())
