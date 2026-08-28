import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datasets import Dataset
from pypdf import PdfReader

from .base import BaseLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    Loader for PDF files that extracts text with optional chunking support.

    Features:
    - Extracts text from all pages in a PDF
    - Supports chunking by token count
    - Handles corrupted PDFs gracefully
    - Returns standardized format with source and page metadata
    """

    def __init__(
        self,
        path: str,
        chunk_size: Optional[int] = 512,
        chunk_overlap: int = 0,
    ):
        """
        Initialize PDFLoader.

        Args:
            path: Path to PDF file or directory containing PDFs
            chunk_size: Number of tokens per chunk (None to disable chunking)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self) -> Dataset:
        """Load PDF file(s) and return as Dataset."""
        if self.path.is_dir():
            return self.load_directory()
        else:
            return self.load_file()

    def load_file(self) -> Dataset:
        """
        Load a single PDF file.

        Returns:
            Dataset with columns: text, source, page
        """
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.path}")

        data = []
        filename = self.path.name

        try:
            if not self._validate_pdf(self.path):
                logger.warning(f"PDF validation failed for {filename}, skipping")
                return Dataset.from_dict({"text": [], "source": [], "page": []})

            reader = PdfReader(str(self.path))
            text_by_page = []

            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        text_by_page.append((text, page_idx))
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_idx} in {filename}: {e}")

            # Process extracted text
            if self.chunk_size and self.chunk_size > 0:
                # Concatenate all pages and chunk
                full_text = "\n".join([text for text, _ in text_by_page])
                chunks = self._chunk_text(full_text)
                for chunk_text in chunks:
                    data.append({
                        "text": chunk_text,
                        "source": filename,
                        "page": -1,  # -1 indicates chunked across multiple pages
                    })
            else:
                # Keep page-level granularity
                for text, page_idx in text_by_page:
                    data.append({
                        "text": text,
                        "source": filename,
                        "page": page_idx,
                    })

        except Exception as e:
            logger.error(f"Failed to load PDF {filename}: {e}")
            return Dataset.from_dict({"text": [], "source": [], "page": []})

        if not data:
            logger.warning(f"No text extracted from {filename}")
            return Dataset.from_dict({"text": [], "source": [], "page": []})

        return Dataset.from_dict({
            "text": [d["text"] for d in data],
            "source": [d["source"] for d in data],
            "page": [d["page"] for d in data],
        })

    def load_directory(self, pattern: str = "*.pdf", recurse: bool = True) -> Dataset:
        """
        Load all PDF files from a directory.

        Args:
            pattern: Glob pattern for files to load (default: "*.pdf")
            recurse: Whether to recursively search subdirectories

        Returns:
            Combined Dataset from all PDFs
        """
        if not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

        glob_func = self.path.rglob if recurse else self.path.glob
        pdf_files = list(glob_func(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "page": []})

        all_data = []
        for pdf_file in sorted(pdf_files):
            loader = PDFLoader(
                str(pdf_file),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            dataset = loader.load_file()
            if len(dataset) > 0:
                all_data.append(dataset)

        if not all_data:
            logger.warning(f"No valid data extracted from PDFs in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "page": []})

        # Concatenate all datasets
        combined = all_data[0]
        for dataset in all_data[1:]:
            combined = combined.concatenate(dataset)

        return combined

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into roughly equal-sized pieces.

        Simple word-based chunking (not token-aware for efficiency).
        For token-aware chunking, use transformers.AutoTokenizer.

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

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """
        Validate PDF file integrity.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF is valid and readable, False otherwise
        """
        try:
            reader = PdfReader(str(pdf_path))
            # Check if we can access at least some pages
            if len(reader.pages) == 0:
                return False
            # Try to extract text from first page
            _ = reader.pages[0].extract_text()
            return True
        except Exception as e:
            logger.warning(f"PDF validation failed for {pdf_path.name}: {e}")
            return False
