"""
Tests for carbon/energy tracking.

All tests are fully offline — no external API calls are made.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aligntune.core.advisor import (
    REGION_CARBON_INTENSITY,
    GPU_POWER_WATTS,
    CarbonEstimate,
    estimate_carbon,
    estimate_resources,
)
from aligntune.core.callbacks.carbon_tracker import (
    CarbonReport,
    CarbonTracker,
    CarbonTrackerCallback,
)


# ===========================================================================
# 1. Region intensity lookup
# ===========================================================================

class TestRegionCarbonIntensity:
    def test_known_aws_regions_present(self):
        for region in ("us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"):
            assert region in REGION_CARBON_INTENSITY, f"{region} missing from REGION_CARBON_INTENSITY"

    def test_known_gcp_regions_present(self):
        for region in ("us-central1", "europe-west4", "asia-east1"):
            assert region in REGION_CARBON_INTENSITY

    def test_known_azure_regions_present(self):
        for region in ("eastus", "westeurope"):
            assert region in REGION_CARBON_INTENSITY

    def test_default_region_present(self):
        assert "default" in REGION_CARBON_INTENSITY

    def test_values_are_positive_floats(self):
        for region, intensity in REGION_CARBON_INTENSITY.items():
            assert isinstance(intensity, float), f"{region} intensity is not a float"
            assert intensity > 0, f"{region} intensity must be positive"

    def test_us_west_2_lower_than_us_east_1(self):
        # us-west-2 is known to be significantly greener
        assert REGION_CARBON_INTENSITY["us-west-2"] < REGION_CARBON_INTENSITY["us-east-1"]

    def test_specific_values(self):
        assert REGION_CARBON_INTENSITY["us-east-1"] == pytest.approx(380.0)
        assert REGION_CARBON_INTENSITY["us-west-2"] == pytest.approx(136.0)
        assert REGION_CARBON_INTENSITY["default"] == pytest.approx(475.0)


# ===========================================================================
# 2. GPU_POWER_WATTS constant
# ===========================================================================

class TestGpuPowerWatts:
    def test_all_expected_gpus_present(self):
        for gpu in ("a100-40gb", "a100-80gb", "h100", "l4", "t4", "rtx3090", "rtx4090"):
            assert gpu in GPU_POWER_WATTS, f"{gpu} missing from GPU_POWER_WATTS"

    def test_values_are_positive(self):
        for gpu, watts in GPU_POWER_WATTS.items():
            assert watts > 0, f"{gpu} power must be positive"

    def test_h100_highest_power(self):
        # H100 is the most powerful GPU in the list
        assert GPU_POWER_WATTS["h100"] == max(GPU_POWER_WATTS.values())

    def test_specific_values(self):
        assert GPU_POWER_WATTS["a100-40gb"] == pytest.approx(300.0)
        assert GPU_POWER_WATTS["h100"] == pytest.approx(700.0)
        assert GPU_POWER_WATTS["l4"] == pytest.approx(72.0)


# ===========================================================================
# 3. estimate_carbon() formula
# ===========================================================================

class TestEstimateCarbonFormula:
    def test_basic_calculation(self):
        # 1 GPU A100-40GB (300W) for 1 hour in us-east-1 (380 gCO2/kWh)
        # kwh = (300 * 1 * 1) / 1000 = 0.3
        # co2 = 0.3 * 380 = 114 g
        result = estimate_carbon(
            wallclock_hours=1.0,
            gpu_type="a100-40gb",
            num_gpus=1,
            region="us-east-1",
        )
        assert result.kwh == pytest.approx(0.3, rel=1e-3)
        assert result.co2_grams == pytest.approx(114.0, rel=1e-3)

    def test_multi_gpu_scales_linearly(self):
        single = estimate_carbon(1.0, "h100", num_gpus=1, region="us-west-2")
        multi = estimate_carbon(1.0, "h100", num_gpus=4, region="us-west-2")
        assert multi.kwh == pytest.approx(single.kwh * 4, rel=1e-3)
        assert multi.co2_grams == pytest.approx(single.co2_grams * 4, rel=1e-3)

    def test_longer_run_scales_linearly(self):
        one_hour = estimate_carbon(1.0, "t4", region="eu-west-1")
        five_hours = estimate_carbon(5.0, "t4", region="eu-west-1")
        assert five_hours.kwh == pytest.approx(one_hour.kwh * 5, rel=1e-3)

    def test_returns_carbon_estimate_dataclass(self):
        result = estimate_carbon(2.0, "l4", num_gpus=2, region="us-central1")
        assert isinstance(result, CarbonEstimate)

    def test_region_stored_correctly(self):
        result = estimate_carbon(1.0, "rtx4090", region="us-west-2")
        assert result.region == "us-west-2"

    def test_intensity_stored_correctly(self):
        result = estimate_carbon(1.0, "rtx4090", region="us-east-1")
        assert result.intensity == pytest.approx(REGION_CARBON_INTENSITY["us-east-1"])

    def test_unknown_region_falls_back_to_default(self):
        result = estimate_carbon(1.0, "a100-40gb", region="zz-unknown-99")
        assert result.intensity == pytest.approx(REGION_CARBON_INTENSITY["default"])

    def test_unknown_gpu_uses_fallback_power(self):
        # Should not raise; falls back to 400W generic default
        result = estimate_carbon(1.0, "unknown-gpu-xyz", region="default")
        assert result.kwh == pytest.approx(0.4, rel=1e-3)

    def test_zero_hours_gives_zero_emissions(self):
        result = estimate_carbon(0.0, "h100", region="us-east-1")
        assert result.kwh == pytest.approx(0.0, abs=1e-6)
        assert result.co2_grams == pytest.approx(0.0, abs=1e-6)

    def test_greener_region_gives_lower_co2(self):
        dirty = estimate_carbon(1.0, "a100-80gb", region="ap-southeast-1")   # 493
        clean = estimate_carbon(1.0, "a100-80gb", region="us-west-2")         # 136
        assert clean.co2_grams < dirty.co2_grams


# ===========================================================================
# 4. estimate_resources() integration
# ===========================================================================

class TestEstimateResourcesCarbon:
    def test_returns_carbon_by_default(self):
        est = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=1000,
            algorithm="sft",
            hardware_profile="a100-40gb",
        )
        assert est.carbon is not None
        assert isinstance(est.carbon, CarbonEstimate)

    def test_region_propagated(self):
        est = estimate_resources(
            model_name="mistral-7b",
            dataset_size=500,
            region="us-west-2",
        )
        assert est.carbon.region == "us-west-2"
        assert est.carbon.intensity == pytest.approx(REGION_CARBON_INTENSITY["us-west-2"])

    def test_existing_fields_unchanged(self):
        est = estimate_resources(
            model_name="mistral-7b",
            dataset_size=500,
        )
        assert est.vram_gb > 0
        assert est.wallclock_hours > 0
        assert est.cost_usd > 0

    def test_carbon_kwh_positive(self):
        est = estimate_resources("llama-7b", dataset_size=100)
        assert est.carbon.kwh > 0
        assert est.carbon.co2_grams > 0


# ===========================================================================
# 5. CarbonTracker start / stop / report
# ===========================================================================

class TestCarbonTracker:
    def test_basic_lifecycle(self):
        tracker = CarbonTracker()
        tracker.start_tracking(gpu_type="t4", num_gpus=1, region="us-east-1")
        time.sleep(0.1)  # 100ms so the rounded duration_hours > 0.0 (min resolution = 0.0001h)
        tracker.stop_tracking()
        report = tracker.get_report()

        assert isinstance(report, CarbonReport)
        assert report.duration_hours >= 0          # rounded value, may be very small
        assert report.kwh >= 0
        assert report.co2_grams >= 0
        assert report.region == "us-east-1"
        assert report.gpu_type == "t4"
        assert report.num_gpus == 1
        assert report.backend in ("static", "codecarbon")

    def test_get_report_before_start_raises(self):
        tracker = CarbonTracker()
        with pytest.raises(RuntimeError):
            tracker.get_report()

    def test_get_report_without_stop_uses_current_time(self):
        tracker = CarbonTracker()
        tracker.start_tracking(gpu_type="l4", region="us-west-2")
        # Do NOT call stop_tracking — report should still succeed
        report = tracker.get_report()
        assert report.duration_hours >= 0

    def test_static_backend_used_when_no_codecarbon(self):
        with patch("aligntune.core.callbacks.carbon_tracker._CODECARBON_AVAILABLE", False):
            tracker = CarbonTracker()
            tracker.start_tracking(gpu_type="h100", num_gpus=2, region="eu-west-1")
            tracker.stop_tracking()
            report = tracker.get_report()
            assert report.backend == "static"

    def test_formula_matches_estimate_carbon(self):
        """Static tracker should produce results consistent with estimate_carbon()."""
        with patch("aligntune.core.callbacks.carbon_tracker._CODECARBON_AVAILABLE", False):
            tracker = CarbonTracker()
            tracker._start_time = 0.0  # override for deterministic duration
            tracker._stop_time = 3600.0  # exactly 1 hour
            tracker._gpu_type = "a100-40gb"
            tracker._num_gpus = 1
            tracker._region = "us-east-1"

            report = tracker.get_report()
            reference = estimate_carbon(1.0, "a100-40gb", num_gpus=1, region="us-east-1")

            assert report.co2_grams == pytest.approx(reference.co2_grams, rel=1e-3)
            assert report.kwh == pytest.approx(reference.kwh, rel=1e-3)

    def test_carbon_report_to_dict(self):
        tracker = CarbonTracker()
        tracker.start_tracking(gpu_type="rtx4090", region="westeurope")
        tracker.stop_tracking()
        report = tracker.get_report()
        d = report.to_dict()
        assert "co2_grams" in d
        assert "kwh" in d
        assert "duration_hours" in d
        assert "region" in d
        assert "backend" in d


# ===========================================================================
# 6. CarbonTrackerCallback integration
# ===========================================================================

class TestCarbonTrackerCallback:
    def _make_args(self, tmp_path):
        args = MagicMock()
        args.output_dir = str(tmp_path)
        return args

    def _make_state_control(self):
        return MagicMock(), MagicMock()

    def test_callback_full_lifecycle(self, tmp_path):
        cb = CarbonTrackerCallback(
            gpu_type="t4",
            num_gpus=1,
            region="us-east-1",
            output_dir=str(tmp_path),
        )
        args = self._make_args(tmp_path)
        state, control = self._make_state_control()

        cb.on_train_begin(args, state, control)
        time.sleep(0.01)
        cb.on_train_end(args, state, control)

        assert cb.report is not None
        assert isinstance(cb.report, CarbonReport)

    def test_carbon_report_json_saved(self, tmp_path):
        cb = CarbonTrackerCallback(
            gpu_type="a100-40gb",
            region="us-west-2",
            output_dir=str(tmp_path),
        )
        args = self._make_args(tmp_path)
        state, control = self._make_state_control()

        cb.on_train_begin(args, state, control)
        cb.on_train_end(args, state, control)

        report_path = tmp_path / "carbon_report.json"
        assert report_path.exists(), "carbon_report.json not written"
        data = json.loads(report_path.read_text())
        assert "co2_grams" in data
        assert "kwh" in data
        assert "duration_hours" in data

    def test_unified_logger_called_with_carbon_metrics(self, tmp_path):
        mock_logger = MagicMock()
        cb = CarbonTrackerCallback(
            gpu_type="h100",
            region="eu-west-1",
            output_dir=str(tmp_path),
            logger_instance=mock_logger,
        )
        args = self._make_args(tmp_path)
        state, control = self._make_state_control()

        cb.on_train_begin(args, state, control)
        cb.on_train_end(args, state, control)

        mock_logger.log_metrics.assert_called_once()
        call_kwargs = mock_logger.log_metrics.call_args[0][0]
        assert "carbon/co2_grams" in call_kwargs
        assert "carbon/kwh" in call_kwargs
        assert "carbon/duration_hours" in call_kwargs

    def test_output_dir_falls_back_to_args(self, tmp_path):
        cb = CarbonTrackerCallback(gpu_type="l4", region="default")  # no output_dir
        args = MagicMock()
        args.output_dir = str(tmp_path)
        state, control = self._make_state_control()

        cb.on_train_begin(args, state, control)
        cb.on_train_end(args, state, control)

        assert (tmp_path / "carbon_report.json").exists()

    def test_report_none_before_train_end(self):
        cb = CarbonTrackerCallback(gpu_type="t4")
        assert cb.report is None

    def test_on_train_end_without_begin_does_not_crash(self):
        """stop_tracking before start should log error gracefully, not raise."""
        cb = CarbonTrackerCallback(gpu_type="t4")
        args, state, control = MagicMock(), MagicMock(), MagicMock()
        # Calling stop without start will raise RuntimeError in get_report;
        # the callback should catch it and not propagate.
        cb.on_train_end(args, state, control)
        assert cb.report is None


# ===========================================================================
# 7. CLI --region flag (unit-level, no subprocess)
# ===========================================================================

class TestCLIRegion:
    """Test that the CLI estimate command correctly threads the region parameter."""

    def test_estimate_resources_accepts_region(self):
        """Passing region= to estimate_resources should not raise."""
        result = estimate_resources(
            model_name="Qwen/Qwen2.5-7B",
            dataset_size=1000,
            algorithm="lora",
            hardware_profile="a100-40gb",
            region="us-west-2",
        )
        assert result.carbon is not None
        assert result.carbon.region == "us-west-2"

    def test_estimate_resources_default_region(self):
        """Default region should use global average intensity."""
        result = estimate_resources(
            model_name="mistral-7b",
            dataset_size=500,
            region="default",
        )
        assert result.carbon.intensity == pytest.approx(REGION_CARBON_INTENSITY["default"])

    def test_different_regions_produce_different_carbon(self):
        dirty = estimate_resources(
            model_name="llama-7b", dataset_size=500, region="ap-southeast-1"
        )
        clean = estimate_resources(
            model_name="llama-7b", dataset_size=500, region="us-west-2"
        )
        assert clean.carbon.co2_grams < dirty.carbon.co2_grams

    def test_format_estimate_table_includes_carbon(self):
        from aligntune.core.advisor import format_estimate_table

        est = estimate_resources(
            model_name="mistral-7b", dataset_size=500, region="us-east-1"
        )
        table = format_estimate_table(
            "mistral-7b", 500, "sft", "a100-40gb", est, region="us-east-1"
        )
        assert "Carbon" in table or "CO2" in table or "kWh" in table
