"""
Tests for v3.9 Long Context Support — RoPE scaling utilities.

All tests are CPU-only and do not load any model weights.  They exercise:

- :class:`~aligntune.core.long_context.rope_scaling.RopeScalingConfig`
  construction and type coercion.
- :meth:`~aligntune.core.long_context.rope_scaling.RopeScalingApplier.validate_config`
  acceptance and rejection of valid / invalid configs.
- :meth:`~aligntune.core.long_context.rope_scaling.RopeScalingApplier.build_rope_config`
  output structure for each strategy.
- :meth:`~aligntune.core.long_context.rope_scaling.RopeScalingApplier.compute_scale_factor`
  arithmetic and edge-case handling.
- :func:`~aligntune.core.long_context.rope_scaling.make_rope_config_for_target`
  convenience factory.

No GPU, no internet, no transformers model loading.
"""

from __future__ import annotations

import math
import pytest

from aligntune.core.long_context.rope_scaling import (
    RopeScalingApplier,
    RopeScalingConfig,
    make_rope_config_for_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_cfg() -> RopeScalingConfig:
    """A valid linear 4× config (LLaMA-2 base 4096 → 16384)."""
    return RopeScalingConfig(
        type="linear",
        factor=4.0,
        original_max_position=4096,
        target_max_position=16384,
    )


@pytest.fixture
def dynamic_cfg() -> RopeScalingConfig:
    """A valid dynamic 4× config."""
    return RopeScalingConfig(
        type="dynamic",
        factor=4.0,
        original_max_position=4096,
        target_max_position=16384,
    )


@pytest.fixture
def yarn_cfg() -> RopeScalingConfig:
    """A valid YaRN 8× config (Mistral base 8192 → 65536)."""
    return RopeScalingConfig(
        type="yarn",
        factor=8.0,
        original_max_position=8192,
        target_max_position=65536,
    )


@pytest.fixture
def ntk_cfg() -> RopeScalingConfig:
    """A valid NTK-by-parts 4× config."""
    return RopeScalingConfig(
        type="ntk",
        factor=4.0,
        original_max_position=4096,
        target_max_position=16384,
    )


# ---------------------------------------------------------------------------
# RopeScalingConfig — construction
# ---------------------------------------------------------------------------


class TestRopeScalingConfigConstruction:
    """Tests for dataclass construction, type coercion, and __post_init__."""

    def test_valid_linear_construction(self, linear_cfg: RopeScalingConfig) -> None:
        """Constructing a valid linear config should succeed without exception."""
        assert linear_cfg.type == "linear"
        assert linear_cfg.factor == 4.0
        assert linear_cfg.original_max_position == 4096
        assert linear_cfg.target_max_position == 16384

    def test_valid_dynamic_construction(self, dynamic_cfg: RopeScalingConfig) -> None:
        assert dynamic_cfg.type == "dynamic"

    def test_valid_yarn_construction(self, yarn_cfg: RopeScalingConfig) -> None:
        assert yarn_cfg.type == "yarn"
        assert yarn_cfg.factor == 8.0

    def test_valid_ntk_construction(self, ntk_cfg: RopeScalingConfig) -> None:
        assert ntk_cfg.type == "ntk"

    def test_factor_coerced_to_float(self) -> None:
        """Integer factor should be silently coerced to float."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=4,  # int, not float
            original_max_position=2048,
            target_max_position=8192,
        )
        assert isinstance(cfg.factor, float)
        assert cfg.factor == 4.0

    def test_positions_coerced_to_int(self) -> None:
        """Float positions should be coerced to int."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=4096.0,  # type: ignore[arg-type]
            target_max_position=16384.0,   # type: ignore[arg-type]
        )
        assert isinstance(cfg.original_max_position, int)
        assert isinstance(cfg.target_max_position, int)

    def test_invalid_type_raises_value_error(self) -> None:
        """An unrecognised strategy name should raise ValueError at construction."""
        with pytest.raises(ValueError, match="must be one of"):
            RopeScalingConfig(
                type="superrope",  # type: ignore[arg-type]
                factor=4.0,
                original_max_position=4096,
                target_max_position=16384,
            )

    def test_all_four_strategy_names_accepted(self) -> None:
        """All four canonical strategy names must construct without error."""
        base_kwargs = dict(factor=4.0, original_max_position=2048, target_max_position=8192)
        for strategy in ("linear", "dynamic", "yarn", "ntk"):
            cfg = RopeScalingConfig(type=strategy, **base_kwargs)  # type: ignore[arg-type]
            assert cfg.type == strategy


# ---------------------------------------------------------------------------
# RopeScalingApplier.validate_config
# ---------------------------------------------------------------------------


class TestRopeScalingApplierValidation:
    """Tests for validate_config acceptance and rejection paths."""

    def test_valid_linear_passes(self, linear_cfg: RopeScalingConfig) -> None:
        """A well-formed linear config should not raise."""
        RopeScalingApplier.validate_config(linear_cfg)  # no exception

    def test_valid_dynamic_passes(self, dynamic_cfg: RopeScalingConfig) -> None:
        RopeScalingApplier.validate_config(dynamic_cfg)

    def test_valid_yarn_passes(self, yarn_cfg: RopeScalingConfig) -> None:
        RopeScalingApplier.validate_config(yarn_cfg)

    def test_valid_ntk_passes(self, ntk_cfg: RopeScalingConfig) -> None:
        RopeScalingApplier.validate_config(ntk_cfg)

    def test_factor_equal_to_one_raises(self) -> None:
        """factor=1.0 means no extension; should be rejected."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=1.0,
            original_max_position=4096,
            target_max_position=4096,
        )
        with pytest.raises(ValueError, match="factor must be > 1.0"):
            RopeScalingApplier.validate_config(cfg)

    def test_factor_less_than_one_raises(self) -> None:
        """factor < 1.0 would compress context; should be rejected."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=0.5,
            original_max_position=4096,
            target_max_position=2048,
        )
        with pytest.raises(ValueError):
            RopeScalingApplier.validate_config(cfg)

    def test_original_max_zero_raises(self) -> None:
        """original_max_position=0 is invalid."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=0,
            target_max_position=16384,
        )
        with pytest.raises(ValueError, match="original_max_position must be a positive integer"):
            RopeScalingApplier.validate_config(cfg)

    def test_target_less_than_original_raises(self) -> None:
        """target_max_position ≤ original_max_position should be rejected."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=4096,
            target_max_position=2048,
        )
        with pytest.raises(ValueError, match="strictly greater than original_max_position"):
            RopeScalingApplier.validate_config(cfg)

    def test_target_equal_to_original_raises(self) -> None:
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=4096,
            target_max_position=4096,
        )
        with pytest.raises(ValueError, match="strictly greater than original_max_position"):
            RopeScalingApplier.validate_config(cfg)

    def test_inconsistent_factor_and_target_raises(self) -> None:
        """factor and target_max_position that disagree by > 10 % should raise."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=4096,
            # Actual implied target = 16384, but we pass 32768 (100 % off)
            target_max_position=32768,
        )
        with pytest.raises(ValueError, match="inconsistent"):
            RopeScalingApplier.validate_config(cfg)

    def test_near_consistent_within_tolerance_passes(self) -> None:
        """target rounded to power-of-2 within 10 % tolerance should pass."""
        # 4096 × 4 = 16384.  Round to 16000 → diff = 384/16384 ≈ 2.3 %, within tol.
        cfg = RopeScalingConfig(
            type="linear",
            factor=4.0,
            original_max_position=4096,
            target_max_position=16000,
        )
        RopeScalingApplier.validate_config(cfg)  # should NOT raise

    def test_validation_error_message_lists_all_failures(self) -> None:
        """When multiple rules fail the error message should mention each."""
        cfg = RopeScalingConfig(
            type="linear",
            factor=0.5,
            original_max_position=0,
            target_max_position=2048,
        )
        with pytest.raises(ValueError) as exc_info:
            RopeScalingApplier.validate_config(cfg)
        msg = str(exc_info.value)
        # Both rule 2 (factor) and rule 3 (original_max) should appear
        assert "factor" in msg
        assert "original_max_position" in msg


# ---------------------------------------------------------------------------
# RopeScalingApplier.build_rope_config — output structure
# ---------------------------------------------------------------------------


class TestBuildRopeConfig:
    """Tests for the dict output of build_rope_config for each strategy."""

    # ---- linear ----

    def test_linear_keys(self, linear_cfg: RopeScalingConfig) -> None:
        """Linear config should produce exactly {type, rope_type, factor}."""
        result = RopeScalingApplier.build_rope_config(linear_cfg)
        assert set(result.keys()) == {"type", "rope_type", "factor"}

    def test_linear_type_value(self, linear_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(linear_cfg)
        assert result["type"] == "linear"

    def test_linear_factor_value(self, linear_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(linear_cfg)
        assert result["factor"] == pytest.approx(4.0)

    # ---- dynamic ----

    def test_dynamic_keys(self, dynamic_cfg: RopeScalingConfig) -> None:
        """Dynamic config should produce exactly {type, rope_type, factor}."""
        result = RopeScalingApplier.build_rope_config(dynamic_cfg)
        assert set(result.keys()) == {"type", "rope_type", "factor"}

    def test_dynamic_type_value(self, dynamic_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(dynamic_cfg)
        assert result["type"] == "dynamic"

    # ---- yarn ----

    def test_yarn_keys(self, yarn_cfg: RopeScalingConfig) -> None:
        """YaRN config should include original_max_position_embeddings."""
        result = RopeScalingApplier.build_rope_config(yarn_cfg)
        assert "original_max_position_embeddings" in result

    def test_yarn_type_value(self, yarn_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(yarn_cfg)
        assert result["type"] == "yarn"

    def test_yarn_original_max_position_embeddings(self, yarn_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(yarn_cfg)
        assert result["original_max_position_embeddings"] == 8192

    def test_yarn_factor_value(self, yarn_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(yarn_cfg)
        assert result["factor"] == pytest.approx(8.0)

    # ---- ntk ----

    def test_ntk_maps_to_longrope_type(self, ntk_cfg: RopeScalingConfig) -> None:
        """NTK strategy must be translated to 'longrope' for HuggingFace."""
        result = RopeScalingApplier.build_rope_config(ntk_cfg)
        assert result["type"] == "longrope"

    def test_ntk_includes_original_max_position_embeddings(
        self, ntk_cfg: RopeScalingConfig
    ) -> None:
        result = RopeScalingApplier.build_rope_config(ntk_cfg)
        assert "original_max_position_embeddings" in result
        assert result["original_max_position_embeddings"] == 4096

    def test_ntk_factor_preserved(self, ntk_cfg: RopeScalingConfig) -> None:
        result = RopeScalingApplier.build_rope_config(ntk_cfg)
        assert result["factor"] == pytest.approx(4.0)

    # ---- general ----

    def test_build_calls_validate_and_rejects_invalid(self) -> None:
        """build_rope_config must internally validate and raise for bad configs."""
        bad_cfg = RopeScalingConfig(
            type="linear",
            factor=0.5,
            original_max_position=4096,
            target_max_position=2048,
        )
        with pytest.raises(ValueError):
            RopeScalingApplier.build_rope_config(bad_cfg)

    def test_returned_dict_is_new_object_each_call(
        self, linear_cfg: RopeScalingConfig
    ) -> None:
        """build_rope_config must return a fresh dict (no shared-state bugs)."""
        result1 = RopeScalingApplier.build_rope_config(linear_cfg)
        result2 = RopeScalingApplier.build_rope_config(linear_cfg)
        assert result1 is not result2

    def test_result_values_are_python_primitives(
        self, yarn_cfg: RopeScalingConfig
    ) -> None:
        """All values in the returned dict should be JSON-serialisable primitives."""
        import json

        result = RopeScalingApplier.build_rope_config(yarn_cfg)
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# RopeScalingApplier.compute_scale_factor
# ---------------------------------------------------------------------------


class TestComputeScaleFactor:
    """Tests for the factor computation helper."""

    def test_exact_power_of_two_extension(self) -> None:
        """4096 → 16384 = factor 4.0."""
        assert RopeScalingApplier.compute_scale_factor(4096, 16384) == pytest.approx(4.0)

    def test_two_times_extension(self) -> None:
        """2048 → 4096 = factor 2.0."""
        assert RopeScalingApplier.compute_scale_factor(2048, 4096) == pytest.approx(2.0)

    def test_eight_times_extension(self) -> None:
        """4096 → 32768 = factor 8.0."""
        assert RopeScalingApplier.compute_scale_factor(4096, 32768) == pytest.approx(8.0)

    def test_64x_extension(self) -> None:
        """2048 → 131072 = factor 64.0."""
        assert RopeScalingApplier.compute_scale_factor(2048, 131072) == pytest.approx(64.0)

    def test_non_power_of_two_extension(self) -> None:
        """Non-power-of-two extension should return the quotient."""
        factor = RopeScalingApplier.compute_scale_factor(3000, 9000)
        assert factor == pytest.approx(3.0)

    def test_result_is_float(self) -> None:
        """Return type must be float."""
        result = RopeScalingApplier.compute_scale_factor(4096, 16384)
        assert isinstance(result, float)

    def test_result_rounded_to_six_decimal_places(self) -> None:
        """Result should be rounded to 6 dp (no floating-point noise)."""
        # 7000 / 3000 = 2.333333...
        factor = RopeScalingApplier.compute_scale_factor(3000, 7000)
        assert factor == round(7000 / 3000, 6)

    def test_original_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            RopeScalingApplier.compute_scale_factor(0, 16384)

    def test_original_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            RopeScalingApplier.compute_scale_factor(-4096, 16384)

    def test_target_equal_to_original_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater than"):
            RopeScalingApplier.compute_scale_factor(4096, 4096)

    def test_target_less_than_original_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater than"):
            RopeScalingApplier.compute_scale_factor(4096, 2048)

    def test_large_extension_64k_to_1m(self) -> None:
        """Extreme extension should still compute correctly."""
        factor = RopeScalingApplier.compute_scale_factor(65536, 1_048_576)
        expected = round(1_048_576 / 65536, 6)
        assert factor == pytest.approx(expected)

    @pytest.mark.parametrize("original,target,expected_factor", [
        (2048, 4096, 2.0),
        (2048, 8192, 4.0),
        (2048, 16384, 8.0),
        (4096, 16384, 4.0),
        (4096, 32768, 8.0),
        (8192, 65536, 8.0),
        (8192, 131072, 16.0),
    ])
    def test_parametrized_known_factors(
        self, original: int, target: int, expected_factor: float
    ) -> None:
        """Verify compute_scale_factor against hand-computed expected values."""
        assert RopeScalingApplier.compute_scale_factor(original, target) == pytest.approx(
            expected_factor
        )


# ---------------------------------------------------------------------------
# make_rope_config_for_target convenience factory
# ---------------------------------------------------------------------------


class TestMakeRopeConfigForTarget:
    """Tests for the module-level factory function."""

    def test_basic_yarn_4x(self) -> None:
        """Factory should produce a correct YaRN 4× config."""
        cfg = make_rope_config_for_target("yarn", 4096, 16384)
        assert cfg.type == "yarn"
        assert cfg.factor == pytest.approx(4.0)
        assert cfg.original_max_position == 4096
        assert cfg.target_max_position == 16384

    def test_basic_linear_8x(self) -> None:
        cfg = make_rope_config_for_target("linear", 4096, 32768)
        assert cfg.type == "linear"
        assert cfg.factor == pytest.approx(8.0)

    def test_ntk_strategy(self) -> None:
        cfg = make_rope_config_for_target("ntk", 2048, 8192)
        assert cfg.type == "ntk"
        assert cfg.factor == pytest.approx(4.0)

    def test_dynamic_strategy(self) -> None:
        cfg = make_rope_config_for_target("dynamic", 4096, 16384)
        assert cfg.type == "dynamic"

    def test_config_is_valid_after_factory(self) -> None:
        """Config produced by the factory should pass validate_config."""
        cfg = make_rope_config_for_target("yarn", 8192, 65536)
        RopeScalingApplier.validate_config(cfg)  # should not raise

    def test_factory_then_build_produces_correct_dict(self) -> None:
        """Full pipeline: factory → build → dict structure correct."""
        cfg = make_rope_config_for_target("yarn", 4096, 16384)
        result = RopeScalingApplier.build_rope_config(cfg)
        assert result["type"] == "yarn"
        assert result["factor"] == pytest.approx(4.0)
        assert result["original_max_position_embeddings"] == 4096

    def test_invalid_target_less_than_original_raises(self) -> None:
        """Factory should propagate ValueError for invalid target."""
        with pytest.raises(ValueError):
            make_rope_config_for_target("linear", 4096, 2048)

    def test_invalid_strategy_raises(self) -> None:
        """Factory should reject unknown strategy names."""
        with pytest.raises(ValueError):
            make_rope_config_for_target("superrope", 4096, 16384)  # type: ignore[arg-type]

    @pytest.mark.parametrize("strategy", ["linear", "dynamic", "yarn", "ntk"])
    def test_all_strategies_via_factory(self, strategy: str) -> None:
        """All four strategies should work through the factory without error."""
        cfg = make_rope_config_for_target(strategy, 4096, 16384)  # type: ignore[arg-type]
        assert cfg.type == strategy
        result = RopeScalingApplier.build_rope_config(cfg)
        assert "type" in result
        assert "factor" in result


# ---------------------------------------------------------------------------
# Integration-style: config → build → verify round-trip for all strategies
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end tests: construct config, validate, build, inspect output."""

    @pytest.mark.parametrize("strategy,expected_hf_type,needs_orig_max", [
        ("linear", "linear", False),
        ("dynamic", "dynamic", False),
        ("yarn", "yarn", True),
        ("ntk", "longrope", True),
    ])
    def test_full_round_trip(
        self,
        strategy: str,
        expected_hf_type: str,
        needs_orig_max: bool,
    ) -> None:
        """
        For each strategy:
        1. Construct a RopeScalingConfig.
        2. Validate it (no exception).
        3. Build the HF dict.
        4. Verify type and factor.
        5. Verify presence / absence of original_max_position_embeddings.
        """
        cfg = RopeScalingConfig(
            type=strategy,  # type: ignore[arg-type]
            factor=4.0,
            original_max_position=4096,
            target_max_position=16384,
        )
        RopeScalingApplier.validate_config(cfg)
        result = RopeScalingApplier.build_rope_config(cfg)

        assert result["type"] == expected_hf_type
        assert result["factor"] == pytest.approx(4.0)

        if needs_orig_max:
            assert "original_max_position_embeddings" in result
            assert result["original_max_position_embeddings"] == 4096
        else:
            assert "original_max_position_embeddings" not in result

    def test_128k_context_extension(self) -> None:
        """Simulate LLaMA-3 128k context extension (8192 → 131072 = 16×)."""
        cfg = make_rope_config_for_target("yarn", 8192, 131072)
        result = RopeScalingApplier.build_rope_config(cfg)
        assert result["factor"] == pytest.approx(16.0)
        assert result["original_max_position_embeddings"] == 8192

    def test_mistral_32k_context(self) -> None:
        """Simulate Mistral sliding window + 32k extension (8192 → 32768 = 4×)."""
        cfg = make_rope_config_for_target("dynamic", 8192, 32768)
        result = RopeScalingApplier.build_rope_config(cfg)
        assert result["type"] == "dynamic"
        assert result["factor"] == pytest.approx(4.0)
        assert "original_max_position_embeddings" not in result
