"""
Long-context dataset loaders for AlignTune (v3.9).

This module provides a unified interface for loading and pre-processing datasets
that are specifically curated for long-context supervised fine-tuning.  Each
dataset is chunked or filtered so that every example fits within a configurable
``max_length`` token budget.

Supported datasets
------------------

``longalpaca``
    ``abacusai/LongAlpaca-16k`` from the Hugging Face Hub.  Contains ~51k
    instruction-following examples with answers that span up to 16k tokens.
    Loaded natively via ``datasets.load_dataset``.

``books3_chunks``
    Books3 corpus chunked into segments of at most ``max_length`` tokens.
    Books3 is not freely redistributable via the Hub; this loader is a *stub*
    that generates synthetic placeholder documents until a local Books3 path is
    configured via the ``BOOKS3_DATA_DIR`` environment variable.

``arxiv_chunks``
    arXiv full-text papers chunked into segments of at most ``max_length``
    tokens.  Like ``books3_chunks`` this is a *stub* that yields synthetic
    placeholders until a local arXiv dump path is configured via the
    ``ARXIV_DATA_DIR`` environment variable.

Dataset record format
---------------------

Every record returned by :class:`LongContextDatasetLoader` conforms to::

    {
        "text":   str,   # UTF-8 text of the chunk / document
        "length": int,   # approximate token count (word-level proxy)
        "source": str,   # dataset name (e.g. "longalpaca")
    }

Example usage
-------------

.. code-block:: python

    from aligntune.core.long_context.datasets import LongContextDatasetLoader

    loader = LongContextDatasetLoader()
    records = loader.load("longalpaca", max_length=32768)
    for record in records[:3]:
        print(record["source"], record["length"])
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public type alias for a single dataset record.
# ---------------------------------------------------------------------------

#: A single dataset record produced by this module.
DataRecord = Dict[str, object]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_token_estimate(text: str) -> int:
    """Return a fast, whitespace-based token-count proxy.

    This is intentionally lightweight — it splits on whitespace and multiplies
    by 1.3 to approximate the subword overhead of BPE tokenisers.  Use a real
    tokeniser when exact counts are required.

    Args:
        text: Input string.

    Returns:
        Estimated token count as an integer.
    """
    return int(len(text.split()) * 1.3)


def _chunk_text(text: str, max_length: int, stride: int = 0) -> List[str]:
    """Split *text* into word-boundary chunks of at most *max_length* tokens.

    The chunking is performed on whitespace-delimited words.  Each chunk
    contains at most ``max_length // 1.3`` words (inverting the 1.3× BPE
    multiplier from :func:`_word_token_estimate`).

    Args:
        text: The full document text to split.
        max_length: Maximum token budget per chunk (estimated).
        stride: Number of overlapping words between consecutive chunks.
            ``0`` means no overlap (contiguous packing).

    Returns:
        A list of string chunks, each fitting within *max_length* tokens.
    """
    words = text.split()
    max_words = max(1, int(max_length / 1.3))
    step = max(1, max_words - stride)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Per-dataset loader functions
# ---------------------------------------------------------------------------

def _load_longalpaca(max_length: int) -> List[DataRecord]:
    """Load ``abacusai/LongAlpaca-16k`` from the Hugging Face Hub.

    The dataset contains instruction-following pairs.  Each record is
    assembled from the ``instruction``, ``input``, and ``output`` fields
    (mirroring the Alpaca data format) into a single ``text`` field, then
    filtered to those whose estimated token length does not exceed
    *max_length*.

    Args:
        max_length: Maximum token length per record.  Records longer than
            this limit are silently dropped to keep GPU memory predictable.

    Returns:
        A list of :data:`DataRecord` dictionaries.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        Exception: Propagates any Hugging Face Hub connectivity error.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load longalpaca. "
            "Install it with: pip install datasets"
        ) from exc

    logger.info("Loading abacusai/LongAlpaca-16k from Hugging Face Hub …")
    raw = load_dataset("abacusai/LongAlpaca-16k", split="train", trust_remote_code=False)

    records: List[DataRecord] = []
    for row in raw:
        instruction: str = row.get("instruction", "") or ""
        inp: str = row.get("input", "") or ""
        output: str = row.get("output", "") or ""

        # Assemble Alpaca-style prompt + completion.
        if inp.strip():
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        length = _word_token_estimate(text)
        if length > max_length:
            continue  # skip examples that exceed the target window

        records.append({"text": text, "length": length, "source": "longalpaca"})

    logger.info(f"longalpaca: retained {len(records):,} records ≤ {max_length} tokens")
    return records


def _load_books3_chunks(max_length: int) -> List[DataRecord]:
    """Load or synthesise Books3 chunks.

    Books3 (part of The Pile) is not publicly available on the Hugging Face
    Hub.  This loader checks the ``BOOKS3_DATA_DIR`` environment variable for
    a local path containing ``.txt`` files.  When no path is configured it
    falls back to a deterministic set of synthetic placeholder documents that
    are clearly marked as stubs, so downstream code remains functional during
    development and CI.

    The returned chunks are at most *max_length* tokens long, produced by
    :func:`_chunk_text` with no stride (contiguous packing).

    Args:
        max_length: Maximum token length per chunk.

    Returns:
        A list of :data:`DataRecord` dictionaries.
    """
    data_dir: Optional[str] = os.environ.get("BOOKS3_DATA_DIR")

    if data_dir and os.path.isdir(data_dir):
        logger.info(f"Loading Books3 from local directory: {data_dir}")
        return list(_stream_text_files(data_dir, max_length=max_length, source="books3_chunks"))

    # ── Stub mode ────────────────────────────────────────────────────────────
    logger.warning(
        "BOOKS3_DATA_DIR is not set or does not exist. "
        "Returning synthetic Books3 stub records. "
        "Set the environment variable to a directory of .txt files to use real data."
    )
    stub_text = (
        "[STUB] This is a synthetic Books3 placeholder record generated by "
        "AlignTune's long-context dataset loader. "
        "It simulates the format of a real Books3 chunk (≤32k tokens) for "
        "development and testing purposes. "
        "Replace this stub by setting the BOOKS3_DATA_DIR environment variable "
        "to a directory containing plain-text (.txt) book files. " * 20
    )
    # Return a small number of synthetic records so pipelines don't break.
    records: List[DataRecord] = []
    for i in range(5):
        text = f"[STUB BOOK {i}] " + stub_text
        length = _word_token_estimate(text)
        records.append({"text": text, "length": length, "source": "books3_chunks"})
    return records


def _load_arxiv_chunks(max_length: int) -> List[DataRecord]:
    """Load or synthesise arXiv paper chunks.

    arXiv full-text dumps are not freely available on the Hugging Face Hub in
    a single unified dataset.  This loader checks the ``ARXIV_DATA_DIR``
    environment variable for a local path containing ``.txt`` files (e.g.
    exported from S3 using ``arxiv-public-datasets``).  When no path is
    configured it returns synthetic placeholder records.

    The returned chunks are at most *max_length* tokens long.

    Args:
        max_length: Maximum token length per chunk.

    Returns:
        A list of :data:`DataRecord` dictionaries.
    """
    data_dir: Optional[str] = os.environ.get("ARXIV_DATA_DIR")

    if data_dir and os.path.isdir(data_dir):
        logger.info(f"Loading arXiv papers from local directory: {data_dir}")
        return list(_stream_text_files(data_dir, max_length=max_length, source="arxiv_chunks"))

    # ── Stub mode ────────────────────────────────────────────────────────────
    logger.warning(
        "ARXIV_DATA_DIR is not set or does not exist. "
        "Returning synthetic arXiv stub records. "
        "Set the environment variable to a directory of .txt files to use real data."
    )
    stub_text = (
        "[STUB] Abstract: This synthetic record simulates an arXiv paper chunk "
        "produced by AlignTune's long-context dataset loader. "
        "1. Introduction. Long-context language models require training data "
        "that exercises the full context window. "
        "Replace this stub by setting the ARXIV_DATA_DIR environment variable "
        "to a directory containing plain-text arXiv paper files. " * 20
    )
    records: List[DataRecord] = []
    for i in range(5):
        text = f"[STUB ARXIV {i}] " + stub_text
        length = _word_token_estimate(text)
        records.append({"text": text, "length": length, "source": "arxiv_chunks"})
    return records


def _stream_text_files(
    directory: str,
    max_length: int,
    source: str,
) -> Generator[DataRecord, None, None]:
    """Yield chunked :data:`DataRecord` objects from all ``.txt`` files in *directory*.

    Walks *directory* recursively, reads each ``.txt`` file, splits it into
    chunks via :func:`_chunk_text`, and yields one record per chunk.

    Args:
        directory: Root path to scan for ``.txt`` files.
        max_length: Maximum token length per yielded chunk.
        source: Value to store in the ``"source"`` field of each record.

    Yields:
        :data:`DataRecord` dictionaries, one per text chunk.
    """
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    full_text = fh.read()
            except OSError as exc:
                logger.warning(f"Could not read {fpath}: {exc}")
                continue

            for chunk in _chunk_text(full_text, max_length=max_length, stride=0):
                length = _word_token_estimate(chunk)
                if length == 0:
                    continue
                yield {"text": chunk, "length": length, "source": source}


# ---------------------------------------------------------------------------
# Public loader class
# ---------------------------------------------------------------------------

#: Registry mapping dataset names to their loader functions.
_LOADERS = {
    "longalpaca": _load_longalpaca,
    "books3_chunks": _load_books3_chunks,
    "arxiv_chunks": _load_arxiv_chunks,
}


class LongContextDatasetLoader:
    """Unified loader for long-context SFT datasets.

    Provides a single :meth:`load` method that dispatches to the appropriate
    per-dataset implementation based on the dataset name string.  All returned
    records share the same schema::

        {
            "text":   str,   # document / chunk text
            "length": int,   # estimated token count
            "source": str,   # dataset identifier
        }

    Supported dataset names:

    * ``"longalpaca"``   — ``abacusai/LongAlpaca-16k``
    * ``"books3_chunks"``— Books3 chunked to ≤32k segments (stub if not local)
    * ``"arxiv_chunks"`` — arXiv papers chunked to ≤32k segments (stub if not local)

    Example::

        loader = LongContextDatasetLoader()

        # Load all long-context datasets used in the Qwen 128k recipe.
        for name in ["longalpaca", "books3_chunks", "arxiv_chunks"]:
            records = loader.load(name, max_length=32768)
            print(f"{name}: {len(records)} records")
    """

    # Default maximum token length used when *max_length* is not specified.
    DEFAULT_MAX_LENGTH: int = 32_768

    def load(
        self,
        name: str,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> List[DataRecord]:
        """Load a named long-context dataset.

        Args:
            name: Dataset identifier.  Must be one of ``"longalpaca"``,
                ``"books3_chunks"``, or ``"arxiv_chunks"``.
            max_length: Maximum token length (estimated) allowed per record.
                Records or chunks exceeding this limit are discarded or split.
                Defaults to :attr:`DEFAULT_MAX_LENGTH` (32 768 tokens).

        Returns:
            A list of :data:`DataRecord` dictionaries.  Each record has at
            least the keys ``"text"`` (str), ``"length"`` (int), and
            ``"source"`` (str).

        Raises:
            ValueError: If *name* is not a recognised dataset identifier.
            ImportError: If a required optional dependency (e.g. ``datasets``)
                is not installed when loading from the Hugging Face Hub.

        Example::

            loader = LongContextDatasetLoader()
            records = loader.load("longalpaca", max_length=16384)
            assert all(r["source"] == "longalpaca" for r in records)
        """
        if name not in _LOADERS:
            supported = ", ".join(f'"{k}"' for k in _LOADERS)
            raise ValueError(
                f"Unknown long-context dataset: '{name}'. "
                f"Supported datasets: {supported}."
            )

        if max_length < 1:
            raise ValueError(
                f"max_length must be a positive integer, got {max_length}."
            )

        loader_fn = _LOADERS[name]
        logger.info(f"LongContextDatasetLoader: loading '{name}' (max_length={max_length})")
        records = loader_fn(max_length)
        logger.info(
            f"LongContextDatasetLoader: '{name}' returned {len(records):,} records"
        )
        return records

    @staticmethod
    def supported_datasets() -> List[str]:
        """Return the list of recognised dataset names.

        Returns:
            Sorted list of dataset name strings that can be passed to
            :meth:`load`.

        Example::

            >>> LongContextDatasetLoader.supported_datasets()
            ['arxiv_chunks', 'books3_chunks', 'longalpaca']
        """
        return sorted(_LOADERS.keys())
