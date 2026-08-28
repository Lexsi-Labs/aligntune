import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datasets import Dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files with structure-aware text extraction.

    Features:
    - Parses markdown files with heading awareness
    - Supports heading-based chunking (split on heading boundaries)
    - Supports paragraph-based chunking
    - Preserves heading context in metadata
    - Handles corrupted files gracefully
    - Returns standardized format with source and heading metadata
    """

    def __init__(
        self,
        path: str,
        chunk_by: str = "paragraph",  # "paragraph" or "heading"
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
    ):
        """
        Initialize MarkdownLoader.

        Args:
            path: Path to markdown file or directory containing markdown files
            chunk_by: Chunking strategy - "paragraph" or "heading"
            chunk_size: Number of tokens per chunk (None to disable chunking)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.path = Path(path)
        self.chunk_by = chunk_by
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self) -> Dataset:
        """Load markdown file(s) and return as Dataset."""
        if self.path.is_dir():
            return self.load_directory()
        else:
            return self.load_file()

    def load_file(self) -> Dataset:
        """
        Load a single markdown file.

        Returns:
            Dataset with columns: text, source, heading, section
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.path}")

        data = []
        filename = self.path.name

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"Markdown file is empty: {filename}")
                return Dataset.from_dict({"text": [], "source": [], "heading": [], "section": []})

            # Parse content
            if self.chunk_by == "heading":
                sections = self._parse_by_heading(content)
            else:
                sections = self._parse_by_paragraph(content)

            # Apply chunking if needed
            if self.chunk_size and self.chunk_size > 0:
                for section in sections:
                    heading = section["heading"]
                    section_text = section["text"]
                    chunks = self._chunk_text(section_text)
                    for chunk_idx, chunk_text in enumerate(chunks):
                        data.append({
                            "text": chunk_text,
                            "source": filename,
                            "heading": heading,
                            "section": section["section"],
                        })
            else:
                for section in sections:
                    data.append({
                        "text": section["text"],
                        "source": filename,
                        "heading": section["heading"],
                        "section": section["section"],
                    })

        except Exception as e:
            logger.error(f"Failed to load markdown file {filename}: {e}")
            return Dataset.from_dict({"text": [], "source": [], "heading": [], "section": []})

        if not data:
            logger.warning(f"No valid data extracted from {filename}")
            return Dataset.from_dict({"text": [], "source": [], "heading": [], "section": []})

        return Dataset.from_dict({
            "text": [d["text"] for d in data],
            "source": [d["source"] for d in data],
            "heading": [d["heading"] for d in data],
            "section": [d["section"] for d in data],
        })

    def load_directory(self, pattern: str = "*.md", recurse: bool = True) -> Dataset:
        """
        Load all markdown files from a directory.

        Args:
            pattern: Glob pattern for files to load (default: "*.md")
            recurse: Whether to recursively search subdirectories

        Returns:
            Combined Dataset from all markdown files
        """
        if not self.path.is_dir():
            raise ValueError(f"{self.path} is not a directory")

        glob_func = self.path.rglob if recurse else self.path.glob
        md_files = list(glob_func(pattern))

        if not md_files:
            logger.warning(f"No markdown files found in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "heading": [], "section": []})

        all_data = []
        for md_file in sorted(md_files):
            loader = MarkdownLoader(
                str(md_file),
                chunk_by=self.chunk_by,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            dataset = loader.load_file()
            if len(dataset) > 0:
                all_data.append(dataset)

        if not all_data:
            logger.warning(f"No valid data extracted from markdown files in {self.path}")
            return Dataset.from_dict({"text": [], "source": [], "heading": [], "section": []})

        # Concatenate all datasets
        combined = all_data[0]
        for dataset in all_data[1:]:
            combined = combined.concatenate(dataset)

        return combined

    def _parse_by_heading(self, content: str) -> List[Dict]:
        """
        Parse markdown content by heading boundaries.

        Args:
            content: Markdown file content

        Returns:
            List of sections with heading and text
        """
        sections = []
        lines = content.split("\n")

        current_heading = ""
        current_section = []
        section_idx = 0

        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r"^(#+)\s+(.*)", line)
            if heading_match:
                # Save previous section if it has content
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append({
                        "heading": current_heading,
                        "text": section_text,
                        "section": section_idx,
                    })
                    section_idx += 1

                # Start new section
                current_heading = heading_match.group(2)
                current_section = []
            else:
                current_section.append(line)

        # Add final section
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "text": section_text,
                "section": section_idx,
            })

        return sections if sections else [{"heading": "", "text": content, "section": 0}]

    def _parse_by_paragraph(self, content: str) -> List[Dict]:
        """
        Parse markdown content by paragraph boundaries.

        Treats each non-empty block of text (separated by blank lines) as a paragraph.

        Args:
            content: Markdown file content

        Returns:
            List of sections with heading and text
        """
        sections = []
        paragraphs = re.split(r"\n\s*\n", content)

        current_heading = ""
        section_idx = 0

        for para_text in paragraphs:
            para_text = para_text.strip()
            if not para_text:
                continue

            # Check if paragraph starts with a heading
            heading_match = re.match(r"^(#+)\s+(.*)", para_text)
            if heading_match:
                current_heading = heading_match.group(2)
                # Skip headings from content
                continue

            sections.append({
                "heading": current_heading,
                "text": para_text,
                "section": section_idx,
            })
            section_idx += 1

        return sections if sections else [{"heading": "", "text": content, "section": 0}]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into roughly equal-sized pieces.

        Simple word-based chunking.

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
