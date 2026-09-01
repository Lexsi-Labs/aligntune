"""
Tokenizer expansion module for Indic language support.

This module provides utilities to extend base tokenizers (e.g., from LLaMA, Mistral)
to support Indic languages through script-aware BPE merges and vocabulary extension.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
from collections import Counter
import re

logger = logging.getLogger(__name__)


class TokenizerExtender:
    """
    Extends a base tokenizer to support Indic scripts through vocabulary expansion.

    This class provides methods to:
    - Extend tokenizer vocabulary with Indic script-specific tokens
    - Perform script-aware BPE merges for better compression
    - Validate round-trip encoding/decoding
    - Save extended tokenizers in HuggingFace format

    Attributes:
        base_tokenizer: The base tokenizer to extend (must have encode/decode methods)
        target_vocab_size: Target vocabulary size for the extended tokenizer
        scripts: List of supported Indic scripts
        extended_vocab: Dictionary mapping token strings to their IDs
    """

    # Indic Unicode ranges for different scripts
    SCRIPT_RANGES = {
        "devanagari": (0x0900, 0x097F),
        "tamil": (0x0B80, 0x0BFF),
        "bengali": (0x0980, 0x09FF),
        "telugu": (0x0C00, 0x0C7F),
        "kannada": (0x0C80, 0x0CFF),
        "malayalam": (0x0D00, 0x0D7F),
    }

    SUPPORTED_SCRIPTS = list(SCRIPT_RANGES.keys())

    def __init__(
        self,
        base_tokenizer: Any,
        target_vocab_size: int = 128256,
    ) -> None:
        """
        Initialize the TokenizerExtender.

        Args:
            base_tokenizer: A tokenizer object with encode() and decode() methods.
                          Typically from transformers library (LlamaTokenizer, etc.)
            target_vocab_size: Target vocabulary size for the extended tokenizer.
                             Default: 128256 (suitable for LLaMA-like models)

        Raises:
            TypeError: If base_tokenizer doesn't have required encode/decode methods
            ValueError: If target_vocab_size is less than current vocab size
        """
        if not hasattr(base_tokenizer, 'encode') or not hasattr(base_tokenizer, 'decode'):
            raise TypeError("base_tokenizer must have encode() and decode() methods")

        self.base_tokenizer = base_tokenizer
        self.target_vocab_size = target_vocab_size
        self.scripts: List[str] = []
        self.extended_vocab: Dict[str, int] = {}
        self.bpe_merges: List[Tuple[str, str]] = []

        # Get initial vocab size
        base_vocab_size = getattr(base_tokenizer, 'vocab_size',
                                 len(getattr(base_tokenizer, 'get_vocab', lambda: {})()))

        if target_vocab_size < base_vocab_size:
            logger.warning(
                f"target_vocab_size ({target_vocab_size}) is less than "
                f"base vocab_size ({base_vocab_size}). Using base vocab size."
            )
            self.target_vocab_size = base_vocab_size

        logger.info(
            f"TokenizerExtender initialized with base vocab size: {base_vocab_size}, "
            f"target vocab size: {target_vocab_size}"
        )

    def extend_for_scripts(
        self,
        scripts: List[str],
        corpus_texts: List[str],
        num_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extend tokenizer vocabulary for specified Indic scripts using corpus-aware merges.

        This method:
        1. Validates that requested scripts are supported
        2. Extracts script-specific characters and subword units from corpus
        3. Performs BPE-style merges prioritizing high-frequency multi-character units
        4. Creates new token IDs for merged units

        Args:
            scripts: List of script names to support. Valid values:
                    ["devanagari", "tamil", "bengali", "telugu", "kannada", "malayalam"]
            corpus_texts: List of example texts in the target scripts.
                         Used to identify frequent character sequences.
            num_new_tokens: Number of new tokens to add. If None, uses space from target_vocab_size.

        Returns:
            Dictionary containing:
                - "scripts": List of added scripts
                - "num_new_tokens": Number of tokens added
                - "new_vocab_ids": List of new token IDs
                - "bpe_merges": List of (token1, token2) tuples representing BPE merges
                - "coverage_stats": Dict with script character coverage statistics

        Raises:
            ValueError: If any script is not in SUPPORTED_SCRIPTS
            ValueError: If corpus_texts is empty

        Examples:
            >>> extender = TokenizerExtender(base_tokenizer, target_vocab_size=128256)
            >>> result = extender.extend_for_scripts(
            ...     scripts=["devanagari", "tamil"],
            ...     corpus_texts=["नमस्ते कैसे हो", "வணக்கம்"]
            ... )
            >>> print(f"Added {result['num_new_tokens']} new tokens")
        """
        # Validation
        invalid_scripts = [s for s in scripts if s not in self.SUPPORTED_SCRIPTS]
        if invalid_scripts:
            raise ValueError(
                f"Unsupported scripts: {invalid_scripts}. "
                f"Supported scripts: {self.SUPPORTED_SCRIPTS}"
            )

        if not corpus_texts:
            raise ValueError("corpus_texts cannot be empty")

        self.scripts = scripts

        # Calculate space for new tokens
        base_vocab_size = getattr(self.base_tokenizer, 'vocab_size',
                                 len(getattr(self.base_tokenizer, 'get_vocab', lambda: {})()))
        available_space = self.target_vocab_size - base_vocab_size
        num_new_tokens = num_new_tokens or available_space

        if num_new_tokens <= 0:
            logger.warning("No space available for new tokens. Returning empty result.")
            return {
                "scripts": scripts,
                "num_new_tokens": 0,
                "new_vocab_ids": [],
                "bpe_merges": [],
                "coverage_stats": {},
            }

        logger.info(f"Extending tokenizer for scripts: {scripts} with {num_new_tokens} new tokens")

        # Extract script-specific character sequences
        char_frequencies = self._extract_script_sequences(corpus_texts, scripts)

        # Perform BPE-style merges
        bpe_merges = self._perform_bpe_merges(char_frequencies, num_new_tokens)
        self.bpe_merges = bpe_merges

        # Create new token IDs
        new_vocab_ids = list(range(base_vocab_size, base_vocab_size + len(bpe_merges)))

        # Build extended vocabulary
        for i, (token1, token2) in enumerate(bpe_merges):
            merged_token = f"{token1}▁{token2}" if token1 and token2 else token1 or token2
            self.extended_vocab[merged_token] = base_vocab_size + i

        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage_stats(corpus_texts, scripts)

        result = {
            "scripts": scripts,
            "num_new_tokens": len(bpe_merges),
            "new_vocab_ids": new_vocab_ids,
            "bpe_merges": bpe_merges,
            "coverage_stats": coverage_stats,
        }

        logger.info(
            f"Extended tokenizer vocabulary with {len(bpe_merges)} new tokens. "
            f"Coverage stats: {coverage_stats}"
        )

        return result

    def validate_round_trip(
        self,
        test_texts: List[str],
        max_text_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate round-trip encoding/decoding for test texts.

        Encodes texts to token IDs and decodes back, checking for information loss.

        Args:
            test_texts: List of text samples to validate
            max_text_length: Maximum length of each text sample (for truncation)

        Returns:
            Dictionary containing:
                - "total_samples": Number of samples tested
                - "exact_matches": Number of exact round-trip matches
                - "match_rate": Percentage of exact matches (0-100)
                - "mean_edit_distance": Average normalized edit distance
                - "samples": List of detailed results per sample with:
                    - "original": Original text
                    - "decoded": Decoded text
                    - "match": Boolean if exact match
                    - "edit_distance": Edit distance from original

        Examples:
            >>> result = extender.validate_round_trip(
            ...     test_texts=["नमस्ते", "வணக்கம்"]
            ... )
            >>> print(f"Match rate: {result['match_rate']:.1f}%")
        """
        if not hasattr(self.base_tokenizer, 'encode') or not hasattr(self.base_tokenizer, 'decode'):
            raise RuntimeError("Tokenizer must have encode() and decode() methods")

        results_list = []
        exact_matches = 0
        total_edit_distance = 0.0

        for text in test_texts:
            # Truncate if needed
            if max_text_length:
                text = text[:max_text_length]

            try:
                # Encode and decode
                token_ids = self.base_tokenizer.encode(text)
                decoded_text = self.base_tokenizer.decode(token_ids)

                # Calculate edit distance
                edit_dist = self._edit_distance(text, decoded_text)
                normalized_dist = edit_dist / max(len(text), len(decoded_text), 1)

                # Check exact match
                is_match = text == decoded_text
                if is_match:
                    exact_matches += 1

                total_edit_distance += normalized_dist

                results_list.append({
                    "original": text,
                    "decoded": decoded_text,
                    "match": is_match,
                    "edit_distance": edit_dist,
                })

            except Exception as e:
                logger.warning(f"Round-trip validation failed for text '{text[:50]}...': {e}")
                results_list.append({
                    "original": text,
                    "decoded": None,
                    "match": False,
                    "error": str(e),
                })

        total_samples = len(test_texts)
        match_rate = (exact_matches / total_samples * 100) if total_samples > 0 else 0.0
        mean_edit_distance = total_edit_distance / total_samples if total_samples > 0 else 0.0

        result = {
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "match_rate": match_rate,
            "mean_edit_distance": mean_edit_distance,
            "samples": results_list,
        }

        logger.info(
            f"Round-trip validation complete: {exact_matches}/{total_samples} "
            f"exact matches ({match_rate:.1f}%), mean edit distance: {mean_edit_distance:.3f}"
        )

        return result

    def save_extended_tokenizer(self, path: str) -> None:
        """
        Save the extended tokenizer metadata in HuggingFace-compatible format.

        Creates a JSON file with extended vocabulary and BPE merge information
        that can be used to reconstruct the tokenizer.

        Args:
            path: Directory path where tokenizer files will be saved.
                 Creates:
                 - extended_vocab.json: Mapping of tokens to IDs
                 - bpe_merges.json: List of BPE merge operations
                 - metadata.json: Extension metadata (scripts, stats, etc.)

        Raises:
            OSError: If directory cannot be created or written to

        Examples:
            >>> extender.save_extended_tokenizer("/path/to/tokenizer")
            >>> # Creates:
            >>> # - /path/to/tokenizer/extended_vocab.json
            >>> # - /path/to/tokenizer/bpe_merges.json
            >>> # - /path/to/tokenizer/metadata.json
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save extended vocabulary
        vocab_path = save_path / "extended_vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.extended_vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved extended vocabulary to {vocab_path}")

        # Save BPE merges
        merges_path = save_path / "bpe_merges.json"
        # Convert tuples to lists for JSON serialization
        merges_list = [[t1, t2] for t1, t2 in self.bpe_merges]
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved BPE merges to {merges_path}")

        # Save metadata
        metadata = {
            "scripts": self.scripts,
            "num_extended_tokens": len(self.extended_vocab),
            "target_vocab_size": self.target_vocab_size,
            "bpe_merge_count": len(self.bpe_merges),
        }
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    # ==================== Private Helper Methods ====================

    def _extract_script_sequences(
        self,
        corpus_texts: List[str],
        scripts: List[str],
    ) -> Counter:
        """
        Extract frequent character sequences from corpus for given scripts.

        Args:
            corpus_texts: List of texts to analyze
            scripts: List of script names

        Returns:
            Counter of (char1, char2) bigrams from script-specific characters
        """
        char_frequencies = Counter()

        for script in scripts:
            start, end = self.SCRIPT_RANGES[script]
            script_chars = set(chr(c) for c in range(start, end + 1))

            for text in corpus_texts:
                # Filter text to only characters in this script
                script_text = ''.join(c for c in text if c in script_chars)

                # Extract bigrams
                for i in range(len(script_text) - 1):
                    bigram = (script_text[i], script_text[i + 1])
                    char_frequencies[bigram] += 1

        # Also add individual characters
        for script in scripts:
            start, end = self.SCRIPT_RANGES[script]
            script_chars = set(chr(c) for c in range(start, end + 1))

            for text in corpus_texts:
                script_text = ''.join(c for c in text if c in script_chars)
                for char in script_text:
                    char_frequencies[(char,)] += 1

        return char_frequencies

    def _perform_bpe_merges(
        self,
        char_frequencies: Counter,
        num_merges: int,
    ) -> List[Tuple[str, str]]:
        """
        Perform BPE-style merges on character frequencies.

        Args:
            char_frequencies: Counter of character sequences
            num_merges: Number of merges to perform

        Returns:
            List of (token1, token2) tuples representing merges
        """
        merges = []

        # Sort by frequency and take top items
        sorted_items = char_frequencies.most_common(num_merges)

        for item, freq in sorted_items:
            if isinstance(item, tuple) and len(item) == 2:
                # Convert single-character tuples back to strings
                token1 = item[0] if len(item[0]) > 0 else ""
                token2 = item[1] if len(item[1]) > 0 else ""
                merges.append((token1, token2))
            elif isinstance(item, tuple) and len(item) == 1:
                # Single character - represented as (char, "")
                merges.append((item[0], ""))

        return merges[:num_merges]

    def _calculate_coverage_stats(
        self,
        corpus_texts: List[str],
        scripts: List[str],
    ) -> Dict[str, float]:
        """
        Calculate coverage statistics for each script in corpus.

        Args:
            corpus_texts: List of texts
            scripts: List of scripts to analyze

        Returns:
            Dictionary mapping script names to coverage percentages
        """
        coverage_stats = {}

        for script in scripts:
            start, end = self.SCRIPT_RANGES[script]
            script_chars = set(chr(c) for c in range(start, end + 1))

            total_chars = 0
            script_chars_found = 0

            for text in corpus_texts:
                total_chars += len(text)
                script_chars_found += sum(1 for c in text if c in script_chars)

            coverage = (script_chars_found / total_chars * 100) if total_chars > 0 else 0.0
            coverage_stats[script] = coverage

        return coverage_stats

    @staticmethod
    def _edit_distance(text1: str, text2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.

        Args:
            text1: First string
            text2: Second string

        Returns:
            Edit distance (int)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]
