from pathlib import Path
from typing import Any
from .hf_loader import HFLoader
from .json_loader import JSONLoader
from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader
from .directory_loader import DirectoryLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .markdown_loader import MarkdownLoader


class LoaderResolver:
    """
    Resolves the appropriate loader based on source type.

    Supported formats:
    - Local files: .json, .jsonl, .csv, .parquet, .pdf, .docx, .md
    - Directories: auto-detects file types and routes appropriately
    - HuggingFace datasets: dataset identifiers
    """

    LOADER_REGISTRY = {
        "json": JSONLoader,
        "csv": CSVLoader,
        "parquet": ParquetLoader,
        "pdf": PDFLoader,
        "docx": DocxLoader,
        "markdown": MarkdownLoader,
        "directory": DirectoryLoader,
    }

    @staticmethod
    def resolve(source: str, **kwargs: Any):
        """
        Determines the correct loader based on the source string.
        Passes **kwargs (like config_name, delimiter) to the specific loader.

        Args:
            source: Source path, HuggingFace dataset identifier, or Indian domain name
            **kwargs: Loader-specific arguments (e.g., delimiter, chunk_size, scheme_name)

        Returns:
            Appropriate loader instance
        """
        path = Path(source)

        if path.exists():
            if path.is_dir():
                pattern = kwargs.get("pattern")
                recurse = kwargs.get("recurse", True)
                return DirectoryLoader(source, pattern=pattern, recurse=recurse)

            if source.endswith(".json") or source.endswith(".jsonl"):
                return JSONLoader(source)

            if source.endswith(".csv"):
                delimiter = kwargs.get("delimiter")
                return CSVLoader(source, delimiter=delimiter)

            if source.endswith(".parquet"):
                return ParquetLoader(source)

            if source.endswith(".pdf"):
                chunk_size = kwargs.get("chunk_size", 512)
                chunk_overlap = kwargs.get("chunk_overlap", 0)
                return PDFLoader(source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if source.endswith(".docx"):
                chunk_size = kwargs.get("chunk_size")
                chunk_overlap = kwargs.get("chunk_overlap", 0)
                return DocxLoader(source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if source.endswith(".md"):
                chunk_by = kwargs.get("chunk_by", "paragraph")
                chunk_size = kwargs.get("chunk_size")
                chunk_overlap = kwargs.get("chunk_overlap", 0)
                return MarkdownLoader(
                    source,
                    chunk_by=chunk_by,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

        # Default to HuggingFace
        # We pass all kwargs here (config_name, split, cache_dir, etc.)
        return HFLoader(source, **kwargs)

    @staticmethod
    def get_loader_by_format(format_name: str):
        """
        Get a loader class by format name.

        Args:
            format_name: Format name (e.g., "pdf", "docx", "markdown")

        Returns:
            Loader class

        Raises:
            ValueError if format not found in registry
        """
        format_name = format_name.lower()
        if format_name not in LoaderResolver.LOADER_REGISTRY:
            raise ValueError(
                f"Unknown format: {format_name}. "
                f"Supported formats: {list(LoaderResolver.LOADER_REGISTRY.keys())}"
            )
        return LoaderResolver.LOADER_REGISTRY[format_name]

    @staticmethod
    def supported_formats() -> list:
        """
        Get list of supported formats.

        Returns:
            List of supported format names
        """
        return list(LoaderResolver.LOADER_REGISTRY.keys())