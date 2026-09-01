"""
Document packing utilities for long-context SFT training.

This module implements two complementary strategies for efficiently packing
short documents into fixed-length sequences suitable for long-context training:

1. **Greedy packing** (``DocumentPacker.pack_documents``): concatenates
   documents back-to-back, separated by the tokenizer's EOS token when
   configured, until the chunk is full.  Any remainder at the end of the
   dataset is padded to ``context_length`` so every sequence in the returned
   dataset has exactly the same length.

2. **Sliding-window packing** (``DocumentPacker.pack_with_stride``): produces
   overlapping chunks from the flat token stream so that each context boundary
   has two chances to be modelled – once as the right half of a window and once
   as the left half of the next window.

Both methods return a ``datasets.Dataset`` (or ``datasets.DatasetDict``) with
``input_ids`` and ``attention_mask`` columns and uniform sequence length,
making them drop-in replacements for the raw dataset fed to
``transformers.SFTTrainer``.

Typical usage::

    from aligntune.core.long_context.packing import DocumentPacker

    packer = DocumentPacker(tokenizer, context_length=32768, eos_between_docs=True)
    packed_ds = packer.pack_documents(raw_dataset, text_column="text")

    # Or with sliding-window overlap
    strided_ds = packer.pack_with_stride(raw_dataset, stride=512)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DocumentPacker:
    """Pack short documents into fixed-length sequences for long-context SFT.

    Parameters
    ----------
    tokenizer:
        A HuggingFace-compatible tokenizer that exposes ``encode``,
        ``pad_token_id``, and ``eos_token_id``.
    context_length:
        Target sequence length in tokens.  Every chunk in the returned
        dataset will have exactly this many tokens.  Default: 32 768.
    stride:
        Default stride used by :py:meth:`pack_with_stride` when the caller
        does not supply one.  Ignored by :py:meth:`pack_documents`.
        Default: 0 (no overlap between greedy chunks).
    eos_between_docs:
        If ``True``, the tokenizer's EOS token is appended between
        consecutive documents so the model can learn document boundaries.
        Default: ``True``.
    """

    def __init__(
        self,
        tokenizer,
        context_length: int = 32_768,
        stride: int = 0,
        eos_between_docs: bool = True,
    ) -> None:
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if stride < 0:
            raise ValueError(f"stride must be non-negative, got {stride}")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.eos_between_docs = eos_between_docs

        # Resolve separator token id (prefer EOS; fall back to PAD; then 0)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self._eos_id: int = eos_id if eos_id is not None else (pad_id if pad_id is not None else 0)
        self._pad_id: int = pad_id if pad_id is not None else (eos_id if eos_id is not None else 0)

        logger.debug(
            "DocumentPacker initialised: context_length=%d, stride=%d, "
            "eos_between_docs=%s, eos_id=%d, pad_id=%d",
            context_length,
            stride,
            eos_between_docs,
            self._eos_id,
            self._pad_id,
        )

    # ------------------------------------------------------------------
    # Primary public methods
    # ------------------------------------------------------------------

    def pack_documents(
        self,
        dataset,
        text_column: str = "text",
    ):
        """Greedily pack documents into fixed-length chunks.

        Each document is tokenised individually.  Documents are concatenated
        sequentially; when adding a document would exceed ``context_length``
        the current chunk is finalised (padded if needed) and a new chunk is
        started.  An EOS token is inserted between consecutive documents when
        ``self.eos_between_docs`` is ``True``.

        Parameters
        ----------
        dataset:
            A ``datasets.Dataset`` (or any object supporting iteration and
            ``__getitem__``) whose rows contain a text column.
        text_column:
            Name of the column that holds raw text.  Default: ``"text"``.

        Returns
        -------
        datasets.Dataset
            A new dataset with columns ``input_ids`` (``List[int]``) and
            ``attention_mask`` (``List[int]``).  Every row has exactly
            ``context_length`` elements in both lists.

        Raises
        ------
        KeyError
            If ``text_column`` is not present in the dataset.
        ValueError
            If the dataset is empty.
        """
        try:
            _ = dataset[0]
        except (IndexError, KeyError):
            raise ValueError("Dataset is empty – cannot pack zero documents.")

        if text_column not in dataset.column_names:
            raise KeyError(
                f"Column '{text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        logger.info(
            "pack_documents: tokenising %d documents (context_length=%d, "
            "eos_between_docs=%s)",
            len(dataset),
            self.context_length,
            self.eos_between_docs,
        )

        flat_ids = self._build_flat_token_stream(dataset, text_column)
        chunks = self._greedy_chunk(flat_ids)

        logger.info(
            "pack_documents: produced %d packed chunks from %d tokens total",
            len(chunks),
            len(flat_ids),
        )

        return self._make_dataset(chunks)

    def pack_with_stride(
        self,
        dataset,
        stride: Optional[int] = None,
        text_column: str = "text",
    ):
        """Pack documents using a sliding-window approach.

        Unlike :py:meth:`pack_documents`, this method generates *overlapping*
        chunks so that every token appears in two windows: once near the right
        end of a window and once near the left end of the next.  This is
        especially useful for documents that are longer than
        ``context_length``; the model sees each token in at least two
        distinct contexts.

        Parameters
        ----------
        dataset:
            A ``datasets.Dataset`` whose rows contain a text column.
        stride:
            Number of tokens to advance between consecutive windows.  Must be
            in ``[1, context_length]``.  When ``stride == context_length``
            the result is identical to :py:meth:`pack_documents` (no overlap).
            Defaults to ``self.stride`` if that is non-zero, or
            ``context_length // 2`` otherwise.
        text_column:
            Name of the column containing raw text.  Default: ``"text"``.

        Returns
        -------
        datasets.Dataset
            A new dataset with ``input_ids`` and ``attention_mask`` columns,
            each row of length exactly ``context_length``.

        Raises
        ------
        ValueError
            If ``stride`` is outside the valid range or the dataset is empty.
        """
        if stride is None:
            stride = self.stride if self.stride > 0 else max(1, self.context_length // 2)

        if not (1 <= stride <= self.context_length):
            raise ValueError(
                f"stride must be in [1, context_length={self.context_length}], got {stride}"
            )

        try:
            _ = dataset[0]
        except (IndexError, KeyError):
            raise ValueError("Dataset is empty – cannot pack zero documents.")

        if text_column not in dataset.column_names:
            raise KeyError(
                f"Column '{text_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        logger.info(
            "pack_with_stride: tokenising %d documents (context_length=%d, stride=%d)",
            len(dataset),
            self.context_length,
            stride,
        )

        flat_ids = self._build_flat_token_stream(dataset, text_column)
        chunks = self._sliding_chunk(flat_ids, stride)

        logger.info(
            "pack_with_stride: produced %d chunks from %d tokens (stride=%d)",
            len(chunks),
            len(flat_ids),
            stride,
        )

        return self._make_dataset(chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_single(self, text: str) -> List[int]:
        """Tokenise a single document and return token ids as a plain list.

        The method deliberately avoids requesting padding or truncation so
        the caller retains full control over length management.

        Parameters
        ----------
        text:
            Raw text to tokenise.

        Returns
        -------
        List[int]
            Token ids for *text*.  Empty list if *text* is blank.
        """
        if not text or not text.strip():
            return []
        result = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return list(result["input_ids"])

    def _build_flat_token_stream(
        self,
        dataset,
        text_column: str,
    ) -> List[int]:
        """Concatenate all documents into a single token stream.

        Documents are separated by the EOS token when
        ``self.eos_between_docs`` is ``True``.

        Parameters
        ----------
        dataset:
            HuggingFace Dataset to iterate over.
        text_column:
            Column name containing raw text.

        Returns
        -------
        List[int]
            Flat list of token ids covering the entire dataset.
        """
        flat: List[int] = []
        n_skipped = 0

        for i in range(len(dataset)):
            row = dataset[i]
            text = row.get(text_column, "") if isinstance(row, dict) else row
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            ids = self._tokenize_single(text)
            if not ids:
                n_skipped += 1
                continue

            if flat and self.eos_between_docs:
                flat.append(self._eos_id)

            flat.extend(ids)

        if n_skipped:
            logger.warning(
                "_build_flat_token_stream: skipped %d empty/blank documents", n_skipped
            )

        return flat

    def _greedy_chunk(self, flat_ids: List[int]) -> List[List[int]]:
        """Split *flat_ids* into non-overlapping chunks of ``context_length``.

        The last chunk is right-padded with ``self._pad_id`` to reach exactly
        ``context_length`` tokens.

        Parameters
        ----------
        flat_ids:
            Flat token stream produced by :py:meth:`_build_flat_token_stream`.

        Returns
        -------
        List[List[int]]
            List of token-id lists, each of length ``context_length``.
        """
        chunks: List[List[int]] = []
        n = self.context_length
        total = len(flat_ids)

        if total == 0:
            logger.warning("_greedy_chunk received an empty token stream; returning empty list")
            return chunks

        for start in range(0, total, n):
            chunk = flat_ids[start : start + n]
            # Pad the last (potentially short) chunk
            if len(chunk) < n:
                chunk = chunk + [self._pad_id] * (n - len(chunk))
            chunks.append(chunk)

        return chunks

    def _sliding_chunk(self, flat_ids: List[int], stride: int) -> List[List[int]]:
        """Split *flat_ids* into overlapping windows.

        Each window is exactly ``context_length`` tokens.  Windows begin at
        positions ``0, stride, 2*stride, …`` until the remaining tokens are
        fewer than ``context_length``.  The final partial window is padded and
        included unless the token stream is shorter than ``context_length``
        (in which case a single padded chunk is returned).

        Parameters
        ----------
        flat_ids:
            Flat token stream.
        stride:
            Advance between consecutive window starts.

        Returns
        -------
        List[List[int]]
            List of token-id lists, each of length ``context_length``.
        """
        chunks: List[List[int]] = []
        n = self.context_length
        total = len(flat_ids)

        if total == 0:
            logger.warning("_sliding_chunk received an empty token stream; returning empty list")
            return chunks

        start = 0
        while start < total:
            chunk = flat_ids[start : start + n]
            if len(chunk) < n:
                chunk = chunk + [self._pad_id] * (n - len(chunk))
            chunks.append(chunk)
            start += stride
            # Avoid duplicating a terminal padded chunk
            if start >= total:
                break

        return chunks

    @staticmethod
    def _build_attention_mask(input_ids: List[int], pad_id: int) -> List[int]:
        """Build an attention mask: 1 for real tokens, 0 for padding.

        Parameters
        ----------
        input_ids:
            Sequence of token ids.
        pad_id:
            The id used for padding.

        Returns
        -------
        List[int]
            Binary mask of the same length as *input_ids*.
        """
        return [0 if tok == pad_id else 1 for tok in input_ids]

    def _make_dataset(self, chunks: List[List[int]]):
        """Convert a list of token-id chunks into a ``datasets.Dataset``.

        Parameters
        ----------
        chunks:
            List of fixed-length token-id lists produced by
            :py:meth:`_greedy_chunk` or :py:meth:`_sliding_chunk`.

        Returns
        -------
        datasets.Dataset
            Dataset with ``input_ids`` and ``attention_mask`` columns.

        Raises
        ------
        ImportError
            If the ``datasets`` package is not installed.
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required by DocumentPacker. "
                "Install it with: pip install datasets"
            ) from exc

        if not chunks:
            logger.warning("_make_dataset called with zero chunks; returning empty dataset")
            return Dataset.from_dict({"input_ids": [], "attention_mask": []})

        attention_masks = [
            self._build_attention_mask(chunk, self._pad_id) for chunk in chunks
        ]

        return Dataset.from_dict(
            {
                "input_ids": chunks,
                "attention_mask": attention_masks,
            }
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def separator_token_id(self) -> int:
        """Token id inserted between documents (EOS).

        Returns
        -------
        int
            The EOS token id, or the pad token id when EOS is unavailable.
        """
        return self._eos_id

    @property
    def padding_token_id(self) -> int:
        """Token id used to pad incomplete chunks.

        Returns
        -------
        int
            The pad token id, or the EOS token id when PAD is unavailable.
        """
        return self._pad_id

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DocumentPacker("
            f"context_length={self.context_length}, "
            f"stride={self.stride}, "
            f"eos_between_docs={self.eos_between_docs})"
        )
