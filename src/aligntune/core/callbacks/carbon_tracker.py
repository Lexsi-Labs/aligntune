"""
Carbon and energy tracking for AlignTune training runs.

Provides two complementary classes:

* ``CarbonTracker`` — standalone tracker that records wall-clock runtime and
  computes CO2/kWh using static region constants (no external API calls).
  Optionally delegates to ``codecarbon`` ``EmissionsTracker`` when that
  library is installed.

* ``CarbonTrackerCallback`` — a ``TrainerCallback`` that wraps
  ``CarbonTracker`` and integrates with the training loop, logging results
  under the ``carbon/`` metric prefix and saving a JSON report to
  ``output_dir``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .base import TrainerCallback
from ..advisor import estimate_carbon, CarbonEstimate, REGION_CARBON_INTENSITY, GPU_POWER_WATTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional codecarbon integration
# ---------------------------------------------------------------------------

try:
    from codecarbon import EmissionsTracker as _CodeCarbonTracker  # type: ignore
    _CODECARBON_AVAILABLE = True
    logger.debug("codecarbon is available — will use EmissionsTracker for accurate measurements")
except ImportError:
    _CodeCarbonTracker = None  # type: ignore
    _CODECARBON_AVAILABLE = False
    logger.debug("codecarbon not installed — falling back to static carbon calculation")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CarbonReport:
    """Actual carbon / energy usage measured during a training run."""
    co2_grams: float
    kwh: float
    duration_hours: float
    region: str
    intensity: float          # gCO2eq/kWh
    gpu_type: str
    num_gpus: int
    backend: str              # "codecarbon" | "static"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class CarbonTracker:
    """
    Tracks carbon emissions for a training run.

    Usage::

        tracker = CarbonTracker()
        tracker.start_tracking(gpu_type="a100-40gb", num_gpus=2, region="us-west-2")
        # ... training ...
        tracker.stop_tracking()
        report = tracker.get_report()
        print(report.co2_grams, report.kwh)

    When ``codecarbon`` is installed it will be used automatically for more
    accurate hardware-level measurements.  Otherwise the calculation falls
    back to the static ``GPU_POWER_WATTS`` constants from ``advisor.py``.
    """

    def __init__(self) -> None:
        self._gpu_type: str = "a100-40gb"
        self._num_gpus: int = 1
        self._region: str = "default"
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._codecarbon_tracker: Optional[object] = None
        self._codecarbon_emissions_kg: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_tracking(
        self,
        gpu_type: str = "a100-40gb",
        num_gpus: int = 1,
        region: str = "default",
    ) -> None:
        """
        Begin tracking.

        Args:
            gpu_type: GPU key (e.g., "a100-40gb", "h100").
            num_gpus: Number of GPUs in the training job.
            region: Cloud region key for carbon intensity lookup.
        """
        self._gpu_type = gpu_type.lower()
        self._num_gpus = num_gpus
        self._region = region
        self._start_time = time.monotonic()
        self._stop_time = None
        self._codecarbon_emissions_kg = None

        if _CODECARBON_AVAILABLE:
            try:
                # Map our region keys to ISO country codes where possible;
                # codecarbon accepts country_iso_code or cloud_region.
                self._codecarbon_tracker = _CodeCarbonTracker(
                    measure_power_secs=15,
                    save_to_file=False,
                    log_level="error",
                )
                self._codecarbon_tracker.start()
                logger.info("CarbonTracker: codecarbon EmissionsTracker started")
            except Exception as exc:
                logger.warning(f"CarbonTracker: codecarbon start failed ({exc}), using static fallback")
                self._codecarbon_tracker = None
        else:
            logger.info(
                f"CarbonTracker: static mode — GPU={gpu_type} x{num_gpus}, region={region}"
            )

    def stop_tracking(self) -> None:
        """Stop tracking and record end time."""
        self._stop_time = time.monotonic()

        if self._codecarbon_tracker is not None:
            try:
                emissions_kg = self._codecarbon_tracker.stop()
                if emissions_kg is not None:
                    self._codecarbon_emissions_kg = float(emissions_kg)
                    logger.info(
                        f"CarbonTracker: codecarbon reported {self._codecarbon_emissions_kg * 1000:.1f}g CO2"
                    )
            except Exception as exc:
                logger.warning(f"CarbonTracker: codecarbon stop failed ({exc}), using static fallback")
                self._codecarbon_tracker = None

    def get_report(self) -> CarbonReport:
        """
        Compute and return the carbon report.

        Raises:
            RuntimeError: If ``start_tracking()`` has not been called.
        """
        if self._start_time is None:
            raise RuntimeError("CarbonTracker.get_report() called before start_tracking()")

        # Use wall-clock duration; if stop was never called use now
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        duration_hours = (end - self._start_time) / 3600.0

        if self._codecarbon_emissions_kg is not None:
            # codecarbon gives us total CO2 in kg
            co2_grams = self._codecarbon_emissions_kg * 1000.0
            # Back-calculate kWh from the static intensity so the report is consistent
            intensity = REGION_CARBON_INTENSITY.get(
                self._region, REGION_CARBON_INTENSITY["default"]
            )
            kwh = co2_grams / intensity if intensity > 0 else 0.0
            backend = "codecarbon"
        else:
            # Static calculation
            carbon: CarbonEstimate = estimate_carbon(
                wallclock_hours=duration_hours,
                gpu_type=self._gpu_type,
                num_gpus=self._num_gpus,
                region=self._region,
            )
            co2_grams = carbon.co2_grams
            kwh = carbon.kwh
            intensity = carbon.intensity
            backend = "static"

        logger.info(
            f"CarbonTracker report: {co2_grams:.1f}g CO2, {kwh:.4f} kWh, "
            f"{duration_hours:.4f}h ({backend})"
        )

        return CarbonReport(
            co2_grams=round(co2_grams, 2),
            kwh=round(kwh, 4),
            duration_hours=round(duration_hours, 4),
            region=self._region,
            intensity=intensity,
            gpu_type=self._gpu_type,
            num_gpus=self._num_gpus,
            backend=backend,
        )


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class CarbonTrackerCallback(TrainerCallback):
    """
    Trainer callback that measures and logs carbon emissions.

    Integrates with ``TrainerCallback`` lifecycle:

    * ``on_train_begin`` — calls ``CarbonTracker.start_tracking()``
    * ``on_train_end`` — calls ``stop_tracking()``, logs metrics under the
      ``carbon/`` prefix, and writes a JSON report to ``output_dir``.

    Logged metrics (available in WandB / TensorBoard via UnifiedLogger):
        * ``carbon/co2_grams``
        * ``carbon/kwh``
        * ``carbon/duration_hours``

    Args:
        gpu_type: GPU key used for power-draw lookup.
        num_gpus: Number of GPUs in the training job.
        region: Cloud region for carbon intensity.
        output_dir: Directory where ``carbon_report.json`` will be saved.
            Falls back to ``args.output_dir`` when available in the callback.
        logger_instance: Optional ``UnifiedLogger`` (or any object with a
            ``log_metrics(dict)`` method).  When ``None``, Python ``logging``
            is used.
    """

    def __init__(
        self,
        gpu_type: str = "a100-40gb",
        num_gpus: int = 1,
        region: str = "default",
        output_dir: Optional[str] = None,
        logger_instance=None,
    ) -> None:
        self._gpu_type = gpu_type
        self._num_gpus = num_gpus
        self._region = region
        self._output_dir = Path(output_dir) if output_dir else None
        self._unified_logger = logger_instance
        self._tracker = CarbonTracker()
        self._report: Optional[CarbonReport] = None

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """Start carbon tracking at the beginning of training."""
        self._tracker.start_tracking(
            gpu_type=self._gpu_type,
            num_gpus=self._num_gpus,
            region=self._region,
        )
        logger.info(
            f"CarbonTrackerCallback: started — GPU={self._gpu_type} x{self._num_gpus}, "
            f"region={self._region}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        """Stop tracking, log metrics, and write JSON report."""
        self._tracker.stop_tracking()

        try:
            self._report = self._tracker.get_report()
        except RuntimeError as exc:
            logger.error(f"CarbonTrackerCallback: failed to generate report: {exc}")
            return

        # Log metrics
        metrics = {
            "carbon/co2_grams": self._report.co2_grams,
            "carbon/kwh": self._report.kwh,
            "carbon/duration_hours": self._report.duration_hours,
        }
        self._log_metrics(metrics)

        # Save JSON report
        out_dir = self._output_dir
        if out_dir is None and hasattr(args, "output_dir") and args.output_dir:
            out_dir = Path(args.output_dir)

        if out_dir is not None:
            self._save_report(out_dir)

        logger.info(
            f"CarbonTrackerCallback: {self._report.co2_grams:.1f}g CO2, "
            f"{self._report.kwh:.4f} kWh, {self._report.duration_hours:.4f}h "
            f"({self._report.backend})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def report(self) -> Optional[CarbonReport]:
        """Return the most recent CarbonReport (available after on_train_end)."""
        return self._report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: dict) -> None:
        """Log to UnifiedLogger if present, else fall back to Python logging."""
        if self._unified_logger is not None:
            try:
                self._unified_logger.log_metrics(metrics)
                return
            except Exception as exc:
                logger.warning(f"CarbonTrackerCallback: UnifiedLogger.log_metrics failed: {exc}")
        # Fallback
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    def _save_report(self, out_dir: Path) -> None:
        """Write carbon_report.json to out_dir."""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "carbon_report.json"
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(self._report.to_dict(), fh, indent=2)
            logger.info(f"CarbonTrackerCallback: report saved to {report_path}")
        except Exception as exc:
            logger.error(f"CarbonTrackerCallback: failed to save report: {exc}")
