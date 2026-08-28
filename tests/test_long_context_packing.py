"""
CPU-only tests for DocumentPacker (v3.9 Long Context Support).

These tests use a lightweight mock tokenizer so no GPU, downloaded model, or
network access is required.  The mock tokenizer maps each whitespace-separated
word to a unique integer id in a deterministic way (hash-based), which is
sufficient to verify packing logic, boundary conditions, stride/overlap
correctness, and EOS insertion behaviour.

Test coverage:
- DocumentPacker.__init__: parameter validation
- DocumentPacker.pack_documents: context_length boundary enforcement
- DocumentPacker.pack_documents: EOS insertion between docs
- DocumentPacker.pack_documents: empty/blank document handling
- DocumentPacker.pack_with_stride: overlap correctness
- DocumentPacker.pack_with_stride: stride validation
- DocumentPacker._build_flat_token_stream: flat stream construction
- DocumentPacker._greedy_chunk: chunk lengths and padding
- DocumentPacker._sliding_chunk: window positions and lengths
- DocumentPacker._build_attention_mask: mask shape and values
- Edge cases: single document, all-padding chunk, very short dataset
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make sure the src tree is importable without an editable install
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aligntune.core.long_context.packing import DocumentPacker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: minimal mock tokenizer and dataset
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal HuggingFace-style tokenizer for unit testing.

    Encodes text by mapping each whitespace-delimited token to a numeric id
    (hash-mod-9000 + 100 to keep ids away from 0 and 1 which are reserved
    for pad/eos).

    Special token ids
    -----------------
    eos_token_id : 1
    pad_token_id : 0
    """

    eos_token_id: int = 1
    pad_token_id: int = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
        return_attention_mask: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """Encode *text* into token ids."""
        words = text.split()
        ids = [self._word_to_id(w) for w in words]

        if truncation and max_length is not None:
            ids = ids[:max_length]

        result: Dict[str, Any] = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(ids)
        return result

    @staticmethod
    def _word_to_id(word: str) -> int:
        """Map a word to a pseudo-random id in [100, 9099]."""
        return (abs(hash(word)) % 9_000) + 100

    @property
    def column_names(self):  # pragma: no cover
        return []


def _make_hf_dataset(texts: List[str]):
    """Build a minimal HuggingFace Dataset from a list of strings.

    Falls back to a plain list-based stub if ``datasets`` is not installed
    so that the import machinery can still be exercised on minimal envs.
    """
    try:
        from datasets import Dataset

        return Dataset.from_dict({"text": texts})
    except ImportError:
        # Lightweight stand-in: indexable, has column_names, len()
        class _StubDataset:
            def __init__(self, texts):
                self._texts = texts
                self.column_names = ["text"]

            def __len__(self):
                return len(self._texts)

            def __getitem__(self, idx):
                return {"text": self._texts[idx]}

        return _StubDataset(texts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tokenizer():
    """Return a fresh mock tokenizer instance."""
    return _MockTokenizer()


@pytest.fixture()
def short_texts():
    """Five short documents, each ~10 words long."""
    return [
        "The quick brown fox jumps over the lazy dog today",
        "Machine learning models require large amounts of training data",
        "Long context windows enable models to attend over many tokens",
        "Document packing improves throughput during supervised fine tuning",
        "Sliding window attention reduces memory usage for long sequences",
    ]


@pytest.fixture()
def packer_ctx16(tokenizer):
    """DocumentPacker with context_length=16 for easy boundary testing."""
    return DocumentPacker(tokenizer, context_length=16, eos_between_docs=True)


@pytest.fixture()
def packer_ctx32(tokenizer):
    """DocumentPacker with context_length=32."""
    return DocumentPacker(tokenizer, context_length=32, eos_between_docs=True)


# ---------------------------------------------------------------------------
# __init__ parameter validation
# ---------------------------------------------------------------------------


class TestDocumentPackerInit:
    """Validate constructor parameter checking."""

    def test_valid_construction(self, tokenizer):
        """Standard construction succeeds without error."""
        packer = DocumentPacker(tokenizer, context_length=512, stride=64)
        assert packer.context_length == 512
        assert packer.stride == 64
        assert packer.eos_between_docs is True

    def test_eos_between_docs_false(self, tokenizer):
        """eos_between_docs=False is stored correctly."""
        packer = DocumentPacker(tokenizer, context_length=128, eos_between_docs=False)
        assert packer.eos_between_docs is False

    def test_invalid_context_length_zero(self, tokenizer):
        """context_length=0 must raise ValueError."""
        with pytest.raises(ValueError, match="context_length"):
            DocumentPacker(tokenizer, context_length=0)

    def test_invalid_context_length_negative(self, tokenizer):
        """Negative context_length must raise ValueError."""
        with pytest.raises(ValueError, match="context_length"):
            DocumentPacker(tokenizer, context_length=-1)

    def test_invalid_stride_negative(self, tokenizer):
        """Negative stride must raise ValueError."""
        with pytest.raises(ValueError, match="stride"):
            DocumentPacker(tokenizer, context_length=64, stride=-1)

    def test_eos_id_fallback_when_eos_missing(self):
        """When eos_token_id is None, pad_token_id is used as the separator."""
        tok = _MockTokenizer()
        tok.eos_token_id = None  # type: ignore[assignment]
        tok.pad_token_id = 5
        packer = DocumentPacker(tok, context_length=32)
        assert packer.separator_token_id == 5

    def test_pad_id_fallback_when_both_none(self):
        """When both eos_token_id and pad_token_id are None, fall back to 0."""
        tok = _MockTokenizer()
        tok.eos_token_id = None  # type: ignore[assignment]
        tok.pad_token_id = None  # type: ignore[assignment]
        packer = DocumentPacker(tok, context_length=32)
        assert packer.separator_token_id == 0
        assert packer.padding_token_id == 0


# ---------------------------------------------------------------------------
# context_length boundary enforcement
# ---------------------------------------------------------------------------


class TestPackDocumentsContextLength:
    """All returned sequences must have exactly context_length tokens."""

    def test_all_chunks_have_correct_length(self, packer_ctx16, short_texts):
        """Every row in the packed dataset has exactly context_length tokens."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        for i in range(len(packed)):
            assert len(packed[i]["input_ids"]) == 16, (
                f"Row {i}: expected 16 tokens, got {len(packed[i]['input_ids'])}"
            )

    def test_attention_mask_matches_input_ids_length(self, packer_ctx16, short_texts):
        """attention_mask length must equal input_ids length for every row."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        for i in range(len(packed)):
            assert len(packed[i]["attention_mask"]) == len(packed[i]["input_ids"])

    def test_no_chunks_produced_for_empty_dataset(self, tokenizer):
        """Empty dataset must raise ValueError."""
        ds = _make_hf_dataset([])
        packer = DocumentPacker(tokenizer, context_length=16)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            packer.pack_documents(ds)

    def test_single_short_document_produces_one_padded_chunk(self, tokenizer):
        """A single document shorter than context_length produces one padded chunk."""
        ds = _make_hf_dataset(["hello world"])
        packer = DocumentPacker(tokenizer, context_length=16, eos_between_docs=True)
        packed = packer.pack_documents(ds)
        assert len(packed) == 1
        assert len(packed[0]["input_ids"]) == 16

    def test_long_single_document_produces_multiple_chunks(self, tokenizer):
        """A document longer than context_length is split across multiple chunks."""
        # 60 words -> 60 tokens (mock tokenizer is word-level)
        long_text = " ".join([f"word{i}" for i in range(60)])
        ds = _make_hf_dataset([long_text])
        packer = DocumentPacker(tokenizer, context_length=16, eos_between_docs=False)
        packed = packer.pack_documents(ds)
        # 60 tokens / 16 = 3 full chunks + 1 padded = 4 chunks
        assert len(packed) == 4
        for i in range(len(packed)):
            assert len(packed[i]["input_ids"]) == 16

    def test_missing_text_column_raises(self, packer_ctx16, short_texts):
        """Requesting a non-existent text column raises KeyError."""
        ds = _make_hf_dataset(short_texts)
        with pytest.raises(KeyError, match="nonexistent_col"):
            packer_ctx16.pack_documents(ds, text_column="nonexistent_col")


# ---------------------------------------------------------------------------
# EOS insertion between documents
# ---------------------------------------------------------------------------


class TestEOSInsertion:
    """Verify that EOS tokens are inserted correctly between documents."""

    def test_eos_present_in_flat_stream(self, tokenizer):
        """With eos_between_docs=True, EOS id appears in the flat token stream."""
        texts = ["hello world", "foo bar"]
        ds = _make_hf_dataset(texts)
        packer = DocumentPacker(tokenizer, context_length=32, eos_between_docs=True)
        flat = packer._build_flat_token_stream(ds, "text")
        assert tokenizer.eos_token_id in flat, (
            "EOS token should appear between documents"
        )

    def test_eos_absent_when_disabled(self, tokenizer):
        """With eos_between_docs=False, EOS id must not appear between docs."""
        texts = ["hello world", "foo bar"]
        ds = _make_hf_dataset(texts)
        packer = DocumentPacker(tokenizer, context_length=32, eos_between_docs=False)
        flat = packer._build_flat_token_stream(ds, "text")
        # The flat stream may still contain the eos_id if the tokenizer happens
        # to generate it as part of a word hash – we instead verify that the
        # *count* of eos_ids matches what the individual tokenizations produce.
        individual_eos_count = sum(
            packer._tokenize_single(t).count(tokenizer.eos_token_id) for t in texts
        )
        assert flat.count(tokenizer.eos_token_id) == individual_eos_count, (
            "No extra EOS should be inserted when eos_between_docs=False"
        )

    def test_eos_count_equals_n_docs_minus_one(self, tokenizer):
        """With eos_between_docs=True, the flat stream has exactly n_docs-1 separators.

        This test counts the number of eos_id occurrences that are actually
        separator insertions (at boundary positions) rather than incidentally
        generated by the tokenizer.  We use a tokenizer whose word->id mapping
        cannot produce eos_token_id (ids start at 100) so the count is exact.
        """
        texts = [f"word{i} token{i}" for i in range(5)]
        ds = _make_hf_dataset(texts)
        packer = DocumentPacker(tokenizer, context_length=128, eos_between_docs=True)
        flat = packer._build_flat_token_stream(ds, "text")
        # With _MockTokenizer, no word hashes to id 1, so every 1 is an EOS separator
        eos_count = flat.count(tokenizer.eos_token_id)
        assert eos_count == len(texts) - 1, (
            f"Expected {len(texts) - 1} EOS separators, found {eos_count}"
        )

    def test_blank_documents_are_skipped(self, tokenizer):
        """Blank/empty documents do not contribute tokens or extra EOS separators."""
        texts = ["hello world", "", "   ", "foo bar"]
        ds = _make_hf_dataset(texts)
        packer = DocumentPacker(tokenizer, context_length=64, eos_between_docs=True)
        flat = packer._build_flat_token_stream(ds, "text")
        # Only 2 non-empty docs -> 1 EOS separator
        eos_count = flat.count(tokenizer.eos_token_id)
        assert eos_count == 1


# ---------------------------------------------------------------------------
# Stride / sliding-window correctness
# ---------------------------------------------------------------------------


class TestPackWithStride:
    """Verify stride/overlap behaviour."""

    def test_stride_equal_context_length_matches_greedy(self, tokenizer, short_texts):
        """stride == context_length produces the same chunks as pack_documents."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=16, eos_between_docs=True)
        greedy = packer.pack_documents(ds)
        strided = packer.pack_with_stride(ds, stride=16)
        assert len(greedy) == len(strided), (
            "stride==context_length should produce the same number of chunks as greedy"
        )
        for i in range(len(greedy)):
            assert greedy[i]["input_ids"] == strided[i]["input_ids"]

    def test_smaller_stride_produces_more_chunks(self, tokenizer, short_texts):
        """A stride smaller than context_length produces more chunks."""
        ds = _make_hf_dataset(short_texts)
        ctx = 16
        packer = DocumentPacker(tokenizer, context_length=ctx, eos_between_docs=False)
        greedy = packer.pack_documents(ds)
        strided = packer.pack_with_stride(ds, stride=8)
        assert len(strided) >= len(greedy), (
            "Overlapping windows should produce at least as many chunks"
        )

    def test_consecutive_windows_overlap_by_correct_amount(self, tokenizer):
        """Consecutive windows overlap by exactly context_length - stride tokens."""
        texts = [f"word{i}" for i in range(40)]
        ds = _make_hf_dataset([" ".join(texts)])
        ctx = 16
        stride = 8
        packer = DocumentPacker(tokenizer, context_length=ctx, eos_between_docs=False)
        packed = packer.pack_with_stride(ds, stride=stride)
        if len(packed) >= 2:
            ids_0 = packed[0]["input_ids"]
            ids_1 = packed[1]["input_ids"]
            expected_overlap = ctx - stride
            # The suffix of window 0 should match the prefix of window 1
            assert ids_0[stride:] == ids_1[:expected_overlap], (
                "Consecutive windows must overlap by (context_length - stride) tokens"
            )

    def test_all_stride_chunks_have_correct_length(self, tokenizer, short_texts):
        """Every chunk produced by pack_with_stride has exactly context_length tokens."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=12, eos_between_docs=True)
        packed = packer.pack_with_stride(ds, stride=6)
        for i in range(len(packed)):
            assert len(packed[i]["input_ids"]) == 12

    def test_invalid_stride_zero_raises(self, tokenizer, short_texts):
        """stride=0 is not allowed in pack_with_stride."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=16)
        with pytest.raises(ValueError, match="stride"):
            packer.pack_with_stride(ds, stride=0)

    def test_invalid_stride_exceeds_context_length(self, tokenizer, short_texts):
        """stride > context_length must raise ValueError."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=16)
        with pytest.raises(ValueError, match="stride"):
            packer.pack_with_stride(ds, stride=17)

    def test_default_stride_falls_back_to_half_context(self, tokenizer, short_texts):
        """When stride is not supplied and self.stride==0, use context_length // 2."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=16, stride=0)
        # Should not raise; uses context_length // 2 = 8 automatically
        packed = packer.pack_with_stride(ds)
        assert len(packed) > 0

    def test_instance_stride_used_as_default(self, tokenizer, short_texts):
        """When pack_with_stride is called without stride, self.stride is used."""
        ds = _make_hf_dataset(short_texts)
        packer = DocumentPacker(tokenizer, context_length=16, stride=4)
        # stride=4 < context_length=16, should produce more chunks than greedy
        packed_strided = packer.pack_with_stride(ds)
        packed_greedy = packer.pack_documents(ds)
        assert len(packed_strided) >= len(packed_greedy)


# ---------------------------------------------------------------------------
# Attention mask correctness
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """Verify attention mask values."""

    def test_mask_is_one_for_real_tokens(self, tokenizer):
        """Positions with non-pad token ids get mask value 1."""
        input_ids = [100, 200, 300]  # All non-pad ids
        mask = DocumentPacker._build_attention_mask(input_ids, pad_id=0)
        assert all(m == 1 for m in mask)

    def test_mask_is_zero_for_pad_tokens(self, tokenizer):
        """Positions with pad token id get mask value 0."""
        input_ids = [100, 0, 0]  # Two padding positions
        mask = DocumentPacker._build_attention_mask(input_ids, pad_id=0)
        assert mask == [1, 0, 0]

    def test_mask_shape_matches_input_ids(self, tokenizer):
        """Mask length equals input_ids length for arbitrary sequences."""
        for length in [1, 8, 16, 32]:
            ids = [100] * (length // 2) + [0] * (length - length // 2)
            mask = DocumentPacker._build_attention_mask(ids, pad_id=0)
            assert len(mask) == length

    def test_packed_mask_has_zeros_only_in_padding_region(self, tokenizer):
        """In a packed dataset, trailing zeros correspond to the padding region."""
        # Single document with 3 words -> 3 tokens; context_length=8 -> 5 padding
        ds = _make_hf_dataset(["alpha beta gamma"])
        packer = DocumentPacker(tokenizer, context_length=8, eos_between_docs=False)
        packed = packer.pack_documents(ds)
        assert len(packed) == 1
        mask = packed[0]["attention_mask"]
        ids = packed[0]["input_ids"]
        # Wherever id==pad_id, mask must be 0
        for m, tok_id in zip(mask, ids):
            if tok_id == tokenizer.pad_token_id:
                assert m == 0
            else:
                assert m == 1


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Direct tests for private packing helpers."""

    def test_greedy_chunk_returns_correct_number_of_chunks(self, packer_ctx16):
        """_greedy_chunk produces ceil(n / context_length) chunks."""
        flat = list(range(33))  # 33 tokens, context=16 -> ceil(33/16)=3 chunks
        chunks = packer_ctx16._greedy_chunk(flat)
        assert len(chunks) == 3

    def test_greedy_chunk_last_chunk_padded(self, packer_ctx16):
        """The final chunk is padded to context_length if short."""
        flat = list(range(10))  # Shorter than context_length=16
        chunks = packer_ctx16._greedy_chunk(flat)
        assert len(chunks) == 1
        assert len(chunks[0]) == 16
        # Last 6 positions should be pad_id
        assert chunks[0][10:] == [packer_ctx16.padding_token_id] * 6

    def test_greedy_chunk_empty_input(self, packer_ctx16):
        """_greedy_chunk on an empty list returns an empty list."""
        chunks = packer_ctx16._greedy_chunk([])
        assert chunks == []

    def test_sliding_chunk_window_positions(self, packer_ctx16):
        """_sliding_chunk windows start at multiples of stride."""
        flat = list(range(40))
        stride = 8
        chunks = packer_ctx16._sliding_chunk(flat, stride=stride)
        # Windows at 0, 8, 16, 24, 32 (each covers 16 tokens)
        expected_starts = [0, 8, 16, 24, 32]
        assert len(chunks) == len(expected_starts)
        for i, start in enumerate(expected_starts):
            expected = flat[start : start + 16]
            if len(expected) < 16:
                expected = expected + [packer_ctx16.padding_token_id] * (16 - len(expected))
            assert chunks[i] == expected

    def test_tokenize_single_empty_string(self, tokenizer):
        """Tokenising an empty string returns an empty list."""
        packer = DocumentPacker(tokenizer, context_length=16)
        ids = packer._tokenize_single("")
        assert ids == []

    def test_tokenize_single_whitespace_only(self, tokenizer):
        """Tokenising whitespace-only text returns an empty list."""
        packer = DocumentPacker(tokenizer, context_length=16)
        ids = packer._tokenize_single("   ")
        assert ids == []

    def test_tokenize_single_returns_list(self, tokenizer):
        """_tokenize_single always returns a plain Python list."""
        packer = DocumentPacker(tokenizer, context_length=16)
        result = packer._tokenize_single("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)


# ---------------------------------------------------------------------------
# Dataset output structure
# ---------------------------------------------------------------------------


class TestPackedDatasetStructure:
    """Verify the structure of the dataset returned by pack_documents."""

    def test_output_has_input_ids_column(self, packer_ctx16, short_texts):
        """Packed dataset must have an 'input_ids' column."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        assert "input_ids" in packed.column_names

    def test_output_has_attention_mask_column(self, packer_ctx16, short_texts):
        """Packed dataset must have an 'attention_mask' column."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        assert "attention_mask" in packed.column_names

    def test_output_has_at_least_one_row(self, packer_ctx16, short_texts):
        """Non-empty input must produce at least one packed sequence."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        assert len(packed) >= 1

    def test_input_ids_are_lists_of_ints(self, packer_ctx16, short_texts):
        """Every input_ids cell must be a list of integers."""
        ds = _make_hf_dataset(short_texts)
        packed = packer_ctx16.pack_documents(ds)
        for i in range(len(packed)):
            ids = packed[i]["input_ids"]
            assert isinstance(ids, list)
            assert all(isinstance(x, int) for x in ids)
