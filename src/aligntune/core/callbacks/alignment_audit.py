"""
Alignment audit callback for training integration.

Runs post-evaluation alignment audits during training to detect:
- Reward hacking
- Sycophancy
- Refusal collapse
- Verbosity gaming

Non-blocking (errors logged but don't crash training).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .base import TrainerCallback
from ..rl.logging_utils import UnifiedLogger
from ...eval.alignment_auditor import AlignmentAuditor, AlignmentDriftTracker, AuditReport
from ...eval.probes import load_all_probe_sets, load_custom_probes

logger = logging.getLogger(__name__)


class AlignmentAuditCallback(TrainerCallback):
    """
    Callback that runs alignment audits at evaluation checkpoints.

    Logs results via UnifiedLogger under "audit/" prefix and optionally
    detects drift against baseline.

    Set ``judge_type`` to ``"openai"``, ``"anthropic"``, or ``"local"`` to
    enable LLM-as-judge scoring for sycophancy and refusal metrics.
    When ``judge_type`` is ``None`` (default), deterministic v1 scoring is used.
    """

    def __init__(
        self,
        enable_alignment_audit: bool = True,
        audit_probe_set: str = "all",  # "all" or path to custom JSONL
        audit_baseline_report: Optional[str] = None,
        baseline_model: Optional[str] = None,
        device: str = "cuda",
        regression_threshold: float = 0.1,
        output_dir: Optional[str] = None,
        judge_type: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_api_key: Optional[str] = None,
    ):
        """
        Initialize alignment audit callback.

        Args:
            enable_alignment_audit: Whether to run audits
            audit_probe_set: Either "all" to load default probes, or path to custom JSONL
            audit_baseline_report: Path to baseline AuditReport JSON for drift detection
            baseline_model: Model name for baseline verbosity comparison
            device: Device to run audits on
            regression_threshold: Flag if any metric regresses by more than this
            output_dir: Directory to save audit reports
            judge_type: Optional LLM judge type — ``"openai"``, ``"anthropic"``,
                        or ``"local"``.  ``None`` uses deterministic v1 scoring.
            judge_model: Model ID for the judge (uses judge-class default if ``None``).
            judge_api_key: API key for OpenAI/Anthropic judges.
        """
        self.enable_alignment_audit = enable_alignment_audit
        self.audit_probe_set = audit_probe_set
        self.baseline_model = baseline_model
        self.device = device
        self.regression_threshold = regression_threshold
        self.output_dir = Path(output_dir) if output_dir else None

        # Optionally build an LLM judge
        judge = None
        if judge_type is not None:
            try:
                from ...eval.llm_judge import JudgeFactory

                judge = JudgeFactory.create(
                    judge_type=judge_type,
                    model=judge_model,
                    api_key=judge_api_key,
                )
                logger.info(
                    f"AlignmentAuditCallback: LLM judge created "
                    f"({type(judge).__name__}, model={judge.model})"
                )
            except Exception as exc:
                logger.error(
                    f"Failed to create LLM judge (judge_type={judge_type!r}): {exc}. "
                    "Falling back to deterministic v1 scoring."
                )

        # Initialize auditor and tracker
        self.auditor = AlignmentAuditor(
            baseline_model=baseline_model,
            device=device,
            judge=judge,
        )
        self.drift_tracker = AlignmentDriftTracker(
            regression_threshold=regression_threshold
        )

        # Load baseline if provided
        if audit_baseline_report:
            try:
                baseline_path = Path(audit_baseline_report)
                self.drift_tracker.set_baseline_from_json(baseline_path)
                logger.info(f"Loaded baseline report from {baseline_path}")
            except Exception as e:
                logger.error(f"Failed to load baseline report: {e}")

        # Load probe sets
        self._load_probes()

        # Track audit count for logging
        self.audit_count = 0

        if self.enable_alignment_audit:
            logger.info("AlignmentAuditCallback initialized and enabled")
        else:
            logger.info("AlignmentAuditCallback initialized but disabled")

    def _load_probes(self) -> None:
        """Load probe sets based on configuration."""
        try:
            if self.audit_probe_set == "all":
                self.probe_sets = load_all_probe_sets()
                logger.info("Loaded default probe sets (all)")
            else:
                # Load custom probes
                custom_path = Path(self.audit_probe_set)
                if not custom_path.exists():
                    logger.warning(f"Custom probe path not found: {custom_path}")
                    self.probe_sets = load_all_probe_sets()
                else:
                    probes = load_custom_probes(custom_path)
                    self.probe_sets = {"custom": probes}
                    logger.info(f"Loaded custom probes from {custom_path}")
        except Exception as e:
            logger.error(f"Failed to load probe sets: {e}")
            self.probe_sets = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation. Runs alignment audit if enabled.

        Args:
            args: Training arguments
            state: Training state
            control: Control object
            metrics: Evaluation metrics dict
            **kwargs: Additional arguments including model and tokenizer
        """
        if not self.enable_alignment_audit:
            return

        try:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")

            if model is None or tokenizer is None:
                logger.warning("Model or tokenizer not available for alignment audit")
                return

            # Run audit
            logger.info(f"Running alignment audit (audit #{self.audit_count + 1})")
            report = self.auditor.score(model, tokenizer, self.probe_sets)
            self.audit_count += 1

            # Log results
            self._log_audit_results(report, state)

            # Check drift if baseline available
            if self.drift_tracker.baseline_report:
                deltas = self.drift_tracker.check_drift(report)
                self._log_drift_results(deltas, state)

            # Save report if output_dir available
            if self.output_dir:
                self._save_report(report, state)

        except Exception as e:
            # Non-blocking: log error but don't crash training
            logger.error(f"Alignment audit failed (non-blocking): {e}", exc_info=True)

    def _log_audit_results(self, report: AuditReport, state) -> None:
        """Log audit results via UnifiedLogger."""
        # Extract step from state if available
        step = getattr(state, "global_step", 0)

        audit_metrics = {
            "reward_hacking": report.reward_hacking,
            "sycophancy": report.sycophancy,
            "refusal_collapse": report.refusal_collapse,
            "verbosity_gain": report.verbosity_gain,
        }

        logger.info(
            f"Alignment Audit Results (step {step}): "
            f"reward_hacking={report.reward_hacking:.3f}, "
            f"sycophancy={report.sycophancy:.3f}, "
            f"refusal_collapse={report.refusal_collapse:.3f}, "
            f"verbosity_gain={report.verbosity_gain:+.3f}"
        )

    def _log_drift_results(self, deltas: Dict[str, float], state) -> None:
        """Log drift detection results."""
        step = getattr(state, "global_step", 0)

        logger.info(
            f"Alignment Drift Detection (step {step}): "
            f"reward_hacking_delta={deltas.get('reward_hacking_delta', 0):+.3f}, "
            f"sycophancy_delta={deltas.get('sycophancy_delta', 0):+.3f}, "
            f"refusal_collapse_delta={deltas.get('refusal_collapse_delta', 0):+.3f}, "
            f"verbosity_gain_delta={deltas.get('verbosity_gain_delta', 0):+.3f}"
        )

    def _save_report(self, report: AuditReport, state) -> None:
        """Save audit report to JSON file."""
        try:
            step = getattr(state, "global_step", 0)
            report_path = self.output_dir / f"audit_report_step_{step}.json"
            report.to_json(report_path)
            logger.info(f"Saved audit report to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save audit report: {e}")
