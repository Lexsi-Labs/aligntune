"""
Tests for tokenizer extension.

Tests cover:
- TokenizerExtender for vocabulary expansion
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Import modules to test
from aligntune.core.tokenization import (
    TokenizerExtender,
)


# ==================== Fixtures ====================

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.vocab_size = 32000
    # extension.py's TokenizerExtender.__init__ does
    # `getattr(base_tokenizer, 'vocab_size', len(getattr(base_tokenizer, 'get_vocab', lambda: {})()))`
    # - Python evaluates that default expression eagerly regardless of
    # whether `vocab_size` is actually missing, so get_vocab() must return
    # something len()-able, matching what real HF tokenizers provide.
    tokenizer.get_vocab = Mock(return_value={f"tok_{i}": i for i in range(32000)})
    tokenizer.encode = Mock(side_effect=lambda text: list(range(len(text))))
    tokenizer.decode = Mock(side_effect=lambda ids: "".join(chr(65 + i % 26) for i in ids))
    return tokenizer


# ==================== TokenizerExtender Tests ====================

class TestTokenizerExtender:
    """Tests for TokenizerExtender class."""

    def test_init_with_valid_tokenizer(self, mock_tokenizer):
        """Test initialization with a valid tokenizer."""
        extender = TokenizerExtender(mock_tokenizer, target_vocab_size=128256)
        assert extender.base_tokenizer == mock_tokenizer
        assert extender.target_vocab_size == 128256
        assert extender.scripts == []
        assert len(extender.extended_vocab) == 0

    def test_init_without_encode_method(self):
        """Test initialization fails without encode method."""
        invalid_tokenizer = Mock(spec=['decode'])
        with pytest.raises(TypeError):
            TokenizerExtender(invalid_tokenizer)

    def test_init_without_decode_method(self):
        """Test initialization fails without decode method."""
        invalid_tokenizer = Mock(spec=['encode'])
        with pytest.raises(TypeError):
            TokenizerExtender(invalid_tokenizer)

    def test_target_vocab_size_validation(self, mock_tokenizer):
        """Test that target_vocab_size cannot be smaller than base vocab."""
        # target_vocab_size < base vocab_size should be handled gracefully
        extender = TokenizerExtender(mock_tokenizer, target_vocab_size=16000)
        # Should use base vocab size (32000) instead
        assert extender.target_vocab_size >= mock_tokenizer.vocab_size

    def test_extend_for_scripts_with_valid_scripts(self, mock_tokenizer):
        """Test extending tokenizer with valid Indic scripts."""
        extender = TokenizerExtender(mock_tokenizer, target_vocab_size=128256)

        corpus_texts = [
            "नमस्ते कैसे हो",  # Devanagari (Hindi)
            "வணக்கம்",  # Tamil
        ]

        result = extender.extend_for_scripts(
            scripts=["devanagari", "tamil"],
            corpus_texts=corpus_texts,
            num_new_tokens=100
        )

        assert result["scripts"] == ["devanagari", "tamil"]
        # num_new_tokens reflects len(bpe_merges) actually achievable from the
        # corpus (see extension.py docstring: "Number of tokens added"), not
        # an echo of the requested cap - a tiny 2-sentence corpus can't yield
        # 100 distinct merges.
        assert 0 < result["num_new_tokens"] <= 100
        assert "new_vocab_ids" in result
        assert "bpe_merges" in result
        assert "coverage_stats" in result
        assert "devanagari" in result["coverage_stats"]
        assert "tamil" in result["coverage_stats"]

    def test_extend_for_scripts_with_invalid_script(self, mock_tokenizer):
        """Test extending with invalid script name."""
        extender = TokenizerExtender(mock_tokenizer)
        corpus_texts = ["नमस्ते"]

        with pytest.raises(ValueError):
            extender.extend_for_scripts(
                scripts=["invalid_script"],
                corpus_texts=corpus_texts
            )

    def test_extend_for_scripts_with_empty_corpus(self, mock_tokenizer):
        """Test extending with empty corpus."""
        extender = TokenizerExtender(mock_tokenizer)

        with pytest.raises(ValueError):
            extender.extend_for_scripts(
                scripts=["devanagari"],
                corpus_texts=[]
            )

    def test_extend_for_scripts_all_supported_scripts(self, mock_tokenizer):
        """Test extending with all supported scripts."""
        extender = TokenizerExtender(mock_tokenizer)

        corpus_texts = [
            "नमस्ते",  # Devanagari
            "வணக்கம்",  # Tamil
            "తెలుగు",  # Telugu
            "ಕನ್ನಡ",  # Kannada
            "বাংলা",  # Bengali
            "മലയാളം",  # Malayalam
        ]

        result = extender.extend_for_scripts(
            scripts=["devanagari", "tamil", "telugu", "kannada", "bengali", "malayalam"],
            corpus_texts=corpus_texts,
            num_new_tokens=50
        )

        assert len(result["scripts"]) == 6
        assert len(result["coverage_stats"]) == 6

    def test_validate_round_trip_with_ascii_text(self, mock_tokenizer):
        """Test round-trip validation with ASCII text."""
        extender = TokenizerExtender(mock_tokenizer)

        # Mock tokenizer that returns same text
        mock_tokenizer.encode = Mock(return_value=[65, 66, 67])
        mock_tokenizer.decode = Mock(return_value="ABC")

        result = extender.validate_round_trip(["ABC"])

        assert result["total_samples"] == 1
        assert result["exact_matches"] == 1
        assert result["match_rate"] == 100.0

    def test_validate_round_trip_with_indic_text(self, mock_tokenizer):
        """Test round-trip validation with Indic text."""
        extender = TokenizerExtender(mock_tokenizer)

        # Mock tokenizer
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="नमस्ते")

        result = extender.validate_round_trip(["नमस्ते"])

        assert result["total_samples"] == 1
        assert "exact_matches" in result
        assert "match_rate" in result
        assert "samples" in result

    def test_validate_round_trip_with_max_length(self, mock_tokenizer):
        """Test round-trip validation with text length limit."""
        extender = TokenizerExtender(mock_tokenizer)

        mock_tokenizer.encode = Mock(return_value=[1, 2])
        mock_tokenizer.decode = Mock(return_value="नम")

        result = extender.validate_round_trip(
            ["नमस्ते"],
            max_text_length=2
        )

        assert result["total_samples"] == 1

    def test_save_extended_tokenizer(self, mock_tokenizer, tmp_path):
        """Test saving extended tokenizer."""
        extender = TokenizerExtender(mock_tokenizer)

        # First, extend the tokenizer
        corpus_texts = ["नमस्ते"]
        extender.extend_for_scripts(
            scripts=["devanagari"],
            corpus_texts=corpus_texts,
            num_new_tokens=10
        )

        # Save to temp directory
        save_path = str(tmp_path / "tokenizer")
        extender.save_extended_tokenizer(save_path)

        # Verify files were created
        from pathlib import Path
        save_dir = Path(save_path)
        assert (save_dir / "extended_vocab.json").exists()
        assert (save_dir / "bpe_merges.json").exists()
        assert (save_dir / "metadata.json").exists()

