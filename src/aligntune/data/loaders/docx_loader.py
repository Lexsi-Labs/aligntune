import logging
from pathlib import Path
from typing import Optional, List
from datasets import Dataset
from docx import Document
from docx.oxml.text.paragraph import CT_P

from .base import BaseLoader

logger = logging.getLogger(__name__)


class DocxLoader(BaseLoader):
    """
    Loader for DOCX files that extracts text preserving paragraph structure.

    Features:
    - Extracts text from all paragraphs
    - Preserves paragraph-level structure
    - Supports optional chunking
    - Handles corrupted files gracefully
    - Returns standardized format with source and paragraph metadata
    """

    def __init__(
        self,
        path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
    ):
        """
        Initialize DocxLoader.

        Args:
            path: Path to DOCX file or directory containing DOCX files
            chunk_size: Number of tokens per chunk (None to disable chunking)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self) -> Dataset:
        """Load DOCX file(s) and return as Dataset."""
        if self.path.is_dir():
            return self.load_directory()
        else:
            return self.load_file()

    def load_file(self) -> Dataset:
        """
        Load a single DOCX file.

        Returns:
            Dataset with columns: text, source, paragraph_idx
        """
        if not self.path.exists():
            raise FileNotFoundError(f"DOCX file not found: {self.path}")

        data = []
        filename = self.path.name

        try:
            doc = Document(str(self.path))
            paragraphs = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Skip empty paragraphs
                    paragraphs.append(text)

            if not paragraphs:
                logger.warning(f"No text extracted from {filename}")
                return Dataset.from_dict({"text": [], "source": [], "paragraph_idx": []})

            # Process paragraphs
            if self.chunk_size and self.chunk_size > 0:
                # Concatenate all paragraphs and chunk
                full_text = "\n".join(paragraphs)
                chunks = self._chunk_text(full_text)
                for chunk_text in chunks:
                    data.append({
                        "text": chunk_text,
                        "source": filename,
                        "paragraph_idx": -1,  # -1 indicates chunked across multiple paragraphs
                    })
            else:
                # Keep paragraph-level granularity
                for idx, para_text in enumerate(paragraphs):
                    data.append({
                        "text": para_text,
                        "source": filename,
                        "paragraph_idx": idx,
                    })

        except Exception as e:
            logger.error(f"Failed to load DOCX {filename}: {e}")
            return Dataset.from_dict({"text": [], "source": [], "paragraph_idx": []})

        if not data:
            logger.warning(f"No valid data extracted from {filename}")
            return Dataset.from_dict({"text": [], "source": [], "paragraph_idx": []})

        return Dataset.from_dict({
            "text": [d["text"] for d in data],
            "source": [d["source"] for d in data],
            "paragraph_idx": [d["paragraph_idx"] for d in data],
        })

    def load_directory(self, pattern: str = "*.docx", recurse: bool = True) -> Dataset:
        """
        Load all DOCX files from a directory.

        Args:
            pattern: Glob pattern for files to load (default: "*.docx")
            recurse: Whether to recursively search subdirectories

        Returns:
            Combined Dataset from all DOCX files
        """
        if not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

        glob_func = self.path.rglob if recurse else self.path.glob
        docx_files = list(glob_func(pattern))

        if not docx_files:
            logger.warning(f"No DOCX files found in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "paragraph_idx": []})

        all_data = []
        for docx_file in sorted(docx_files):
            loader = DocxLoader(
                str(docx_file),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            dataset = loader.load_file()
            if len(dataset) > 0:
                all_data.append(dataset)

        if not all_data:
            logger.warning(f"No valid data extracted from DOCX files in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "paragraph_idx": []})

        # Concatenate all datasets
        combined = all_data[0]
        for dataset in all_data[1:]:
            combined = combined.concatenate(dataset)

        return combined

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into roughly equal-sized pieces.

        Simple word-based chunking (not token-aware for efficiency).

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not self.chunk_size or self.chunk_size <= 0:
            return [text]

        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        # Rough estimate: 1 word ≈ 1.3 tokens
        words_per_chunk = max(1, int(self.chunk_size / 1.3))
        overlap_words = max(0, int(self.chunk_overlap / 1.3)) if self.chunk_overlap > 0 else 0

        for word in words:
            current_chunk.append(word)
            current_size += 1

            if current_size >= words_per_chunk:
                chunks.append(" ".join(current_chunk))
                # Apply overlap
                if overlap_words > 0 and len(current_chunk) > overlap_words:
                    current_chunk = current_chunk[-overlap_words:]
                    current_size = overlap_words
                else:
                    current_chunk = []
                    current_size = 0

        # Add remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]
