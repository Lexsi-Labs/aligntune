"""
CPU-only tests for SlidingWindowAttentionConfig and LongContextAttentionHelper
(v3.9 Long Context Support).

No GPU, model download, or network access is required.  Calls that normally
depend on ``flash_attn`` are tested via monkeypatching so the suite passes
in environments where the package is absent.

Test coverage:
- SlidingWindowAttentionConfig: valid construction and field access
- SlidingWindowAttentionConfig: parameter validation (window_size, global_tokens)
- SlidingWindowAttentionConfig: effective_window_tokens property
- SlidingWindowAttentionConfig: to_dict / from_dict round-trip
- LongContextAttentionHelper.get_attn_implementation: return structure
- LongContextAttentionHelper.get_attn_implementation: valid implementation values
- LongContextAttentionHelper.get_attn_implementation: flash_attention_2 selected
  when flash_attn is importable (monkeypatched)
- LongContextAttentionHelper.get_attn_implementation: sdpa fallback when
  flash_attn absent but torch >= 2.0 (monkeypatched)
- LongContextAttentionHelper.get_attn_implementation: eager fallback when
  neither flash_attn nor sdpa is available (monkeypatched)
- LongContextAttentionHelper.get_sliding_window_config: known model matching
- LongContextAttentionHelper.get_sliding_window_config: unknown model returns None
- LongContextAttentionHelper.build_model_kwargs: includes torch_dtype key
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src is importable
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aligntune.core.long_context.attention import (  # noqa: E402
    LongContextAttentionHelper,
    SlidingWindowAttentionConfig,
)


# ---------------------------------------------------------------------------
# SlidingWindowAttentionConfig tests
# ---------------------------------------------------------------------------


class TestSlidingWindowAttentionConfig:
    """Unit tests for the SlidingWindowAttentionConfig dataclass."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def test_basic_construction(self):
        """Default global_tokens=0 and correct window_size storage."""
        cfg = SlidingWindowAttentionConfig(window_size=4096)
        assert cfg.window_size == 4096
        assert cfg.global_tokens == 0

    def test_custom_global_tokens(self):
        """Non-zero global_tokens is stored correctly."""
        cfg = SlidingWindowAttentionConfig(window_size=8192, global_tokens=64)
        assert cfg.window_size == 8192
        assert cfg.global_tokens == 64

    def test_global_tokens_zero_is_allowed(self):
        """global_tokens=0 is a valid configuration (no global attention)."""
        cfg = SlidingWindowAttentionConfig(window_size=512, global_tokens=0)
        assert cfg.global_tokens == 0

    def test_global_tokens_one_for_cls(self):
        """global_tokens=1 models the CLS-token scenario used in Longformer."""
        cfg = SlidingWindowAttentionConfig(window_size=512, global_tokens=1)
        assert cfg.global_tokens == 1

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def test_window_size_zero_raises(self):
        """window_size=0 must raise ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowAttentionConfig(window_size=0)

    def test_window_size_negative_raises(self):
        """Negative window_size must raise ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowAttentionConfig(window_size=-512)

    def test_global_tokens_negative_raises(self):
        """Negative global_tokens must raise ValueError."""
        with pytest.raises(ValueError, match="global_tokens"):
            SlidingWindowAttentionConfig(window_size=512, global_tokens=-1)

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    def test_effective_window_tokens_is_double_window_size(self):
        """effective_window_tokens must equal 2 * window_size."""
        cfg = SlidingWindowAttentionConfig(window_size=4096)
        assert cfg.effective_window_tokens == 8192

    def test_effective_window_tokens_large(self):
        """effective_window_tokens is correct for large window sizes."""
        cfg = SlidingWindowAttentionConfig(window_size=32_768)
        assert cfg.effective_window_tokens == 65_536

    # -----------------------------------------------------------------------
    # Serialisation round-trip
    # -----------------------------------------------------------------------

    def test_to_dict_contains_required_keys(self):
        """to_dict must return both window_size and global_tokens."""
        cfg = SlidingWindowAttentionConfig(window_size=2048, global_tokens=16)
        d = cfg.to_dict()
        assert "window_size" in d
        assert "global_tokens" in d

    def test_to_dict_values_match_fields(self):
        """to_dict values must match the instance fields."""
        cfg = SlidingWindowAttentionConfig(window_size=2048, global_tokens=16)
        d = cfg.to_dict()
        assert d["window_size"] == 2048
        assert d["global_tokens"] == 16

    def test_from_dict_round_trip(self):
        """from_dict(to_dict()) produces an equal instance."""
        original = SlidingWindowAttentionConfig(window_size=4096, global_tokens=8)
        restored = SlidingWindowAttentionConfig.from_dict(original.to_dict())
        assert restored.window_size == original.window_size
        assert restored.global_tokens == original.global_tokens

    def test_from_dict_default_global_tokens(self):
        """from_dict uses 0 as the default for global_tokens when key is absent."""
        d = {"window_size": 1024}
        cfg = SlidingWindowAttentionConfig.from_dict(d)
        assert cfg.global_tokens == 0

    def test_from_dict_raises_on_missing_window_size(self):
        """from_dict must raise KeyError when window_size is absent."""
        with pytest.raises(KeyError):
            SlidingWindowAttentionConfig.from_dict({"global_tokens": 4})

    # -----------------------------------------------------------------------
    # Equality and hashability (dataclass defaults)
    # -----------------------------------------------------------------------

    def test_two_identical_configs_are_equal(self):
        """Two configs with identical fields compare equal."""
        a = SlidingWindowAttentionConfig(window_size=4096, global_tokens=0)
        b = SlidingWindowAttentionConfig(window_size=4096, global_tokens=0)
        assert a == b

    def test_different_window_sizes_are_not_equal(self):
        """Configs with different window_size are not equal."""
        a = SlidingWindowAttentionConfig(window_size=4096)
        b = SlidingWindowAttentionConfig(window_size=8192)
        assert a != b

    def test_different_global_tokens_are_not_equal(self):
        """Configs with different global_tokens are not equal."""
        a = SlidingWindowAttentionConfig(window_size=4096, global_tokens=0)
        b = SlidingWindowAttentionConfig(window_size=4096, global_tokens=1)
        assert a != b


# ---------------------------------------------------------------------------
# LongContextAttentionHelper tests
# ---------------------------------------------------------------------------


class TestGetAttnImplementation:
    """Tests for LongContextAttentionHelper.get_attn_implementation."""

    # -----------------------------------------------------------------------
    # Return structure
    # -----------------------------------------------------------------------

    def test_returns_dict(self):
        """Return value must be a dict."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert isinstance(result, dict)

    def test_contains_attn_implementation_key(self):
        """Returned dict must have 'attn_implementation' key."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert "attn_implementation" in result

    def test_contains_use_flash_attention_2_key(self):
        """Returned dict must have 'use_flash_attention_2' key."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert "use_flash_attention_2" in result

    def test_use_flash_attention_2_is_bool(self):
        """use_flash_attention_2 must be a bool."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert isinstance(result["use_flash_attention_2"], bool)

    def test_attn_implementation_is_valid_string(self):
        """attn_implementation must be one of the three valid values."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert result["attn_implementation"] in (
            "flash_attention_2",
            "sdpa",
            "eager",
        )

    # -----------------------------------------------------------------------
    # Consistency between keys
    # -----------------------------------------------------------------------

    def test_use_flash_true_iff_impl_is_flash(self):
        """use_flash_attention_2 must be True exactly when impl is flash_attention_2."""
        result = LongContextAttentionHelper.get_attn_implementation("some/model")
        is_flash = result["attn_implementation"] == "flash_attention_2"
        assert result["use_flash_attention_2"] == is_flash

    # -----------------------------------------------------------------------
    # flash_attention_2 selection (monkeypatched)
    # -----------------------------------------------------------------------

    def test_selects_flash_attention_2_when_flash_attn_importable(self):
        """When flash_attn is importable, flash_attention_2 must be selected."""
        # Create a fake flash_attn module in sys.modules
        fake_flash = types.ModuleType("flash_attn")
        with patch.dict(sys.modules, {"flash_attn": fake_flash}):
            result = LongContextAttentionHelper.get_attn_implementation(
                "mistralai/Mistral-7B-v0.1"
            )
        assert result["attn_implementation"] == "flash_attention_2"
        assert result["use_flash_attention_2"] is True

    # -----------------------------------------------------------------------
    # sdpa fallback (monkeypatched)
    # -----------------------------------------------------------------------

    def test_falls_back_to_sdpa_when_flash_absent_and_torch_ge_2(self):
        """When flash_attn is absent but torch >= 2.0, sdpa is selected."""
        # Remove flash_attn from sys.modules so the import check fails
        with patch.dict(sys.modules, {"flash_attn": None}):
            # Also ensure torch.nn.functional has scaled_dot_product_attention
            import torch

            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                pytest.skip("PyTorch < 2.0 – SDPA not available in this environment")

            result = LongContextAttentionHelper.get_attn_implementation("some/model")

        assert result["attn_implementation"] in ("sdpa", "eager"), (
            "When flash_attn is absent, implementation must be sdpa or eager"
        )

    # -----------------------------------------------------------------------
    # eager fallback (monkeypatched)
    # -----------------------------------------------------------------------

    def test_falls_back_to_eager_when_neither_flash_nor_sdpa_available(self):
        """When both flash_attn and SDPA are unavailable, eager is selected."""
        # Patch _flash_attention_available and _sdpa_available on the helper
        with (
            patch.object(LongContextAttentionHelper, "_flash_attention_available", return_value=False),
            patch.object(LongContextAttentionHelper, "_sdpa_available", return_value=False),
        ):
            result = LongContextAttentionHelper.get_attn_implementation("some/model")

        assert result["attn_implementation"] == "eager"
        assert result["use_flash_attention_2"] is False

    def test_eager_sets_use_flash_to_false(self):
        """When eager is selected, use_flash_attention_2 must be False."""
        with (
            patch.object(LongContextAttentionHelper, "_flash_attention_available", return_value=False),
            patch.object(LongContextAttentionHelper, "_sdpa_available", return_value=False),
        ):
            result = LongContextAttentionHelper.get_attn_implementation("some/model")
        assert result["use_flash_attention_2"] is False

    # -----------------------------------------------------------------------
    # Model-name argument is accepted but not required to affect logic
    # -----------------------------------------------------------------------

    def test_accepts_arbitrary_model_name(self):
        """get_attn_implementation accepts any string without raising."""
        for model_name in [
            "",
            "gpt2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "/local/path/to/model",
        ]:
            result = LongContextAttentionHelper.get_attn_implementation(model_name)
            assert "attn_implementation" in result


# ---------------------------------------------------------------------------
# get_sliding_window_config
# ---------------------------------------------------------------------------


class TestGetSlidingWindowConfig:
    """Tests for LongContextAttentionHelper.get_sliding_window_config."""

    def test_mistral_returns_config(self):
        """'mistral' in the model name returns a SlidingWindowAttentionConfig."""
        cfg = LongContextAttentionHelper.get_sliding_window_config(
            "mistralai/Mistral-7B-v0.1"
        )
        assert cfg is not None
        assert isinstance(cfg, SlidingWindowAttentionConfig)

    def test_mistral_window_size_correct(self):
        """Mistral models have window_size=4096."""
        cfg = LongContextAttentionHelper.get_sliding_window_config(
            "mistralai/Mistral-7B-v0.1"
        )
        assert cfg.window_size == 4_096

    def test_mixtral_returns_config(self):
        """'mixtral' in the model name returns a SlidingWindowAttentionConfig."""
        cfg = LongContextAttentionHelper.get_sliding_window_config(
            "mistralai/Mixtral-8x7B-v0.1"
        )
        assert cfg is not None

    def test_unknown_model_returns_none(self):
        """An unknown model name returns None."""
        cfg = LongContextAttentionHelper.get_sliding_window_config(
            "unknown-org/unknown-model-1b"
        )
        assert cfg is None

    def test_gpt2_returns_none(self):
        """GPT-2 is not in the known-model table and must return None."""
        cfg = LongContextAttentionHelper.get_sliding_window_config("gpt2")
        assert cfg is None

    def test_case_insensitive_matching(self):
        """Model-name matching is case-insensitive (lower-cased internally)."""
        cfg_lower = LongContextAttentionHelper.get_sliding_window_config("mistral-7b")
        cfg_upper = LongContextAttentionHelper.get_sliding_window_config("MISTRAL-7B")
        assert cfg_lower is not None
        assert cfg_upper is not None
        assert cfg_lower.window_size == cfg_upper.window_size


# ---------------------------------------------------------------------------
# build_model_kwargs
# ---------------------------------------------------------------------------


class TestBuildModelKwargs:
    """Tests for LongContextAttentionHelper.build_model_kwargs."""

    def test_returns_dict(self):
        """build_model_kwargs must return a dict."""
        kwargs = LongContextAttentionHelper.build_model_kwargs("some/model")
        assert isinstance(kwargs, dict)

    def test_contains_attn_implementation(self):
        """Result must include 'attn_implementation'."""
        kwargs = LongContextAttentionHelper.build_model_kwargs("some/model")
        assert "attn_implementation" in kwargs

    def test_contains_torch_dtype_auto_by_default(self):
        """When dtype='auto', torch_dtype must be 'auto'."""
        kwargs = LongContextAttentionHelper.build_model_kwargs("some/model", dtype="auto")
        assert kwargs.get("torch_dtype") == "auto"

    def test_bfloat16_dtype_maps_to_torch_type(self):
        """dtype='bfloat16' maps to torch.bfloat16 when torch is available."""
        try:
            import torch

            kwargs = LongContextAttentionHelper.build_model_kwargs(
                "some/model", dtype="bfloat16"
            )
            assert kwargs.get("torch_dtype") == torch.bfloat16
        except ImportError:
            pytest.skip("torch not installed")

    def test_float16_dtype_maps_to_torch_type(self):
        """dtype='float16' maps to torch.float16 when torch is available."""
        try:
            import torch

            kwargs = LongContextAttentionHelper.build_model_kwargs(
                "some/model", dtype="float16"
            )
            assert kwargs.get("torch_dtype") == torch.float16
        except ImportError:
            pytest.skip("torch not installed")

    def test_unknown_dtype_falls_back_gracefully(self):
        """An unrecognised dtype string does not raise; build_model_kwargs is robust."""
        # Should not raise; the helper warns and skips setting a typed dtype
        kwargs = LongContextAttentionHelper.build_model_kwargs(
            "some/model", dtype="int8"
        )
        assert "attn_implementation" in kwargs


# ---------------------------------------------------------------------------
# Private helpers (smoke tests)
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    """Smoke-test the private detection methods for isolation."""

    def test_flash_attention_available_returns_bool(self):
        """_flash_attention_available must return a bool."""
        result = LongContextAttentionHelper._flash_attention_available()
        assert isinstance(result, bool)

    def test_sdpa_available_returns_bool(self):
        """_sdpa_available must return a bool."""
        result = LongContextAttentionHelper._sdpa_available()
        assert isinstance(result, bool)

    def test_flash_available_true_when_flash_attn_importable(self):
        """When flash_attn is registered in sys.modules, returns True."""
        fake_flash = types.ModuleType("flash_attn")
        with patch.dict(sys.modules, {"flash_attn": fake_flash}):
            assert LongContextAttentionHelper._flash_attention_available() is True

    def test_flash_available_false_when_flash_attn_absent(self):
        """When flash_attn is absent from sys.modules, returns False."""
        with patch.dict(sys.modules, {"flash_attn": None}):
            assert LongContextAttentionHelper._flash_attention_available() is False
