import logging
from pathlib import Path
from typing import Optional, List
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

    def __init__(
        self,
        path: str,
        pattern: Optional[str] = None,
        recurse: bool = True,
    ):
        """
        Initialize DirectoryLoader.

        Args:
            path: Path to directory
            pattern: Optional glob pattern to filter files (e.g., "*.md")
            recurse: Whether to recursively search subdirectories
        """
        self.path = Path(path)
        self.pattern = pattern or "*"
        self.recurse = recurse

    def load(self) -> DatasetDict | Dataset:
        """Load all files from directory and return as DatasetDict or Dataset."""
        return self.load_directory()

    def load_directory(self) -> DatasetDict | Dataset:
        """
        Load all supported files from the directory.

        Returns:
            DatasetDict mapping filename stems to Datasets
        """
        if not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

        glob_func = self.path.rglob if self.recurse else self.path.glob
        files = list(glob_func(self.pattern))

        if not files:
            raise ValueError(f"No files matching pattern '{self.pattern}' found in {self.path}")

        datasets = {}
        raw_data = []

        for file_path in sorted(files):
            if file_path.is_file():
                try:
                    loader = self._get_loader_for_file(file_path)
                    if loader is None:
                        logger.warning(f"No loader found for {file_path}, skipping")
                        continue

                    dataset = loader.load()
                    if dataset is None or len(dataset) == 0:
                        logger.warning(f"No data loaded from {file_path}")
                        continue

                    # Store by filename stem
                    file_key = file_path.stem
                    datasets[file_key] = dataset
                    raw_data.append(dataset)

                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

        if not datasets:
            raise ValueError(f"No supported files found in {self.path}")

        # For raw document formats (pdf, docx, md), merge into single dataset
        # For structured formats (json, csv, parquet), return as DatasetDict
        if raw_data and all(self._is_raw_format(f) for f in files if f.is_file()):
            # All files are raw format - merge into single dataset
            return concatenate_datasets(raw_data)

        # Return DatasetDict for mixed or structured formats
        return DatasetDict(datasets)

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
