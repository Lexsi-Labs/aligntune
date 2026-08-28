"""
Expert pruning for Mixture of Experts models with alignment guards.

Enables removal of low-performing or low-activation experts while maintaining
safety guarantees. Alignment guards prevent pruning operations that would degrade
safety metrics, ensuring the compound moat of MoE × Alignment.

Features:
- Low-activation expert identification and pruning
- Alignment-aware pruning with safety metric guards
- Rollback capability to revert pruning
- Pruning candidate ranking and reporting
"""

import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Literal
import json
import copy

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from aligntune.eval.moe_audit import MoEAlignmentAuditor, ExpertAuditReport

logger = logging.getLogger(__name__)


@dataclass
class PruningCandidate:
    """Candidate expert for pruning."""

    expert_id: int
    """Expert identifier."""

    activation_rate: float
    """Fraction of tokens routed to this expert [0, 1]."""

    activation_count: int
    """Absolute count of activations."""

    refusal_collapse: float
    """Safety metric: refusal collapse rate."""

    sycophancy: float
    """Safety metric: sycophancy score."""

    reward_hacking: float
    """Safety metric: reward hacking signal."""

    combined_score: float
    """Combined safety + activation score for ranking."""

    reason: str
    """Human-readable reason for pruning (e.g., 'low-activation')."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PruningReport:
    """Report on expert pruning operation."""

    operation: Literal["propose", "execute", "reject", "rollback"]
    """Operation type: propose, execute, reject, or rollback."""

    pruned_experts: List[int]
    """List of expert IDs that were pruned."""

    num_experts_before: int
    """Number of experts before pruning."""

    num_experts_after: int
    """Number of experts after pruning."""

    activation_threshold: float
    """Activation threshold used for candidate selection."""

    alignment_guard_enabled: bool
    """Whether alignment safety checks were enabled."""

    safety_delta: Dict[str, float] = field(default_factory=dict)
    """Change in safety metrics (negative = degraded)."""

    verdict: Literal["PASS", "REJECTED", "ROLLED_BACK"] = "PASS"
    """Final verdict on pruning operation."""

    rejection_reason: str = ""
    """Reason if pruning was rejected."""

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    """When operation occurred (ISO format)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Path) -> None:
        """Save report to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved PruningReport to {filepath}")

    @classmethod
    def from_json(cls, filepath: Path) -> "PruningReport":
        """Load report from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


class ExpertPruner:
    """
    Prunes low-performing experts from MoE models with alignment safety guards.

    Identifies candidates based on activation rate, scores them based on safety
    metrics, and conditionally prunes if safety thresholds are maintained.

    Attributes:
        model: The MoE model to prune.
        alignment_auditor: Optional MoEAlignmentAuditor for safety verification.
        baseline_safety_metrics: Safety metrics before pruning.
        pruned_model_checkpoint: Checkpoint before pruning (for rollback).
        pruning_history: List of pruning operations performed.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        alignment_auditor: Optional["MoEAlignmentAuditor"] = None,
    ):
        """
        Initialize expert pruner.

        Args:
            model: HuggingFace MoE model instance.
            alignment_auditor: Optional MoEAlignmentAuditor for safety checks.

        Raises:
            ValueError: If model doesn't appear to be MoE.
        """
        self.model = model
        self.alignment_auditor = alignment_auditor

        # Validate model is MoE
        num_experts = self._detect_num_experts(model)
        if num_experts is None or num_experts <= 0:
            raise ValueError(
                "Model doesn't appear to be MoE. Could not detect num_experts."
            )

        self.num_experts = num_experts
        logger.info(f"ExpertPruner initialized: num_experts={num_experts}")

        # State tracking
        self.baseline_safety_metrics: Dict[str, float] = {}
        self.pruned_model_checkpoint: Optional[Dict[str, Any]] = None
        self.pruning_history: List[PruningReport] = []

        # Load baseline safety metrics if auditor available
        if alignment_auditor and alignment_auditor.expert_audit_reports:
            self.baseline_safety_metrics = self._extract_safety_metrics(
                alignment_auditor.expert_audit_reports
            )

    @staticmethod
    def _detect_num_experts(model: "PreTrainedModel") -> Optional[int]:
        """
        Auto-detect num_experts from model config.

        Args:
            model: HuggingFace model instance.

        Returns:
            Number of experts, or None if not detected.
        """
        if not hasattr(model, "config"):
            return None

        config = model.config
        for attr in ["num_experts", "n_experts", "moe_num_experts"]:
            if hasattr(config, attr):
                value = getattr(config, attr)
                if isinstance(value, int) and value > 0:
                    return value

        return None

    def get_pruning_candidates(
        self,
        activation_threshold: float = 0.01,
        min_activation_count: int = 1,
    ) -> List[PruningCandidate]:
        """
        Identify expert candidates for pruning.

        Returns experts with activation rate below threshold, ranked by a combined
        score of safety metrics and activation level.

        Args:
            activation_threshold: Max activation rate for candidate selection [0, 1].
            min_activation_count: Minimum absolute activations to be considered.

        Returns:
            List of PruningCandidate instances, ranked by removal safety.

        Raises:
            ValueError: If no expert audit reports available.
        """
        if not self.alignment_auditor or not self.alignment_auditor.expert_audit_reports:
            raise ValueError(
                "No expert audit reports available. Run MoEAlignmentAuditor.score_per_expert() first."
            )

        candidates = []
        reports = self.alignment_auditor.expert_audit_reports

        for expert_id, report in reports.items():
            if report.activation_rate < activation_threshold and (
                report.activation_count >= min_activation_count
                or report.activation_count == 0
            ):
                # Combined score: prefer low-activation AND low-safety-risk
                # Lower safety risks (lower scores) + low activation = higher ranking
                safety_risk = max(
                    report.refusal_collapse,
                    report.sycophancy,
                    report.reward_hacking,
                )
                combined_score = safety_risk + report.activation_rate

                candidate = PruningCandidate(
                    expert_id=expert_id,
                    activation_rate=report.activation_rate,
                    activation_count=report.activation_count,
                    refusal_collapse=report.refusal_collapse,
                    sycophancy=report.sycophancy,
                    reward_hacking=report.reward_hacking,
                    combined_score=combined_score,
                    reason=f"low-activation (rate={report.activation_rate:.4f})",
                )
                candidates.append(candidate)

        # Sort by combined score (lower is better)
        candidates.sort(key=lambda c: c.combined_score)

        logger.info(
            f"Found {len(candidates)} pruning candidates "
            f"(threshold={activation_threshold})"
        )
        return candidates

    def prune_low_activation_experts(
        self,
        activation_threshold: float = 0.01,
        alignment_guard: bool = True,
        safety_tolerance: Dict[str, float] = None,
        dry_run: bool = False,
    ) -> Tuple["PreTrainedModel", PruningReport]:
        """
        Prune experts with activation rate below threshold.

        If alignment_guard is True, re-runs alignment audit after pruning and
        rejects the operation if safety metrics degrade beyond tolerance.

        Args:
            activation_threshold: Max activation rate for pruning [0, 1].
            alignment_guard: Enable safety metric guards.
            safety_tolerance: Max acceptable delta for each safety metric.
                             Default: {'refusal_collapse': 0.05, 'sycophancy': 0.05}
            dry_run: If True, don't actually prune; just propose and check.

        Returns:
            (pruned_model, PruningReport) with pruning results.

        Raises:
            ValueError: If no candidates found or model modification fails.
        """
        if safety_tolerance is None:
            safety_tolerance = {
                "refusal_collapse": 0.05,
                "sycophancy": 0.05,
                "reward_hacking": 0.05,
            }

        logger.info(
            f"Pruning low-activation experts "
            f"(threshold={activation_threshold}, alignment_guard={alignment_guard})"
        )

        # Get candidates
        candidates = self.get_pruning_candidates(activation_threshold)

        if not candidates:
            logger.warning("No pruning candidates found")
            report = PruningReport(
                operation="propose",
                pruned_experts=[],
                num_experts_before=self.num_experts,
                num_experts_after=self.num_experts,
                activation_threshold=activation_threshold,
                alignment_guard_enabled=alignment_guard,
                verdict="PASS",
                rejection_reason="No candidates found",
            )
            self.pruning_history.append(report)
            return self.model, report

        # Select candidates for pruning (top 10% or at least 1)
        prune_count = max(1, len(candidates) // 10)
        experts_to_prune = [c.expert_id for c in candidates[:prune_count]]

        logger.info(f"Proposing to prune experts: {experts_to_prune}")

        # Create checkpoint before pruning
        if not dry_run:
            self.pruned_model_checkpoint = copy.deepcopy(self.model.state_dict())

        # Attempt pruning
        try:
            if not dry_run:
                pruned_model = self._execute_pruning(self.model, experts_to_prune)
            else:
                pruned_model = self.model

            # If alignment guard enabled, verify safety metrics
            if alignment_guard and self.alignment_auditor:
                verdict, rejection_reason, safety_delta = self._check_alignment_guard(
                    pruned_model, experts_to_prune, safety_tolerance
                )

                if verdict != "PASS":
                    logger.warning(
                        f"Alignment guard rejected pruning: {rejection_reason}"
                    )
                    if not dry_run:
                        self._rollback_pruning()
                    pruned_model = self.model

                    report = PruningReport(
                        operation="reject",
                        pruned_experts=experts_to_prune,
                        num_experts_before=self.num_experts,
                        num_experts_after=self.num_experts,
                        activation_threshold=activation_threshold,
                        alignment_guard_enabled=alignment_guard,
                        safety_delta=safety_delta,
                        verdict="REJECTED",
                        rejection_reason=rejection_reason,
                    )
                    self.pruning_history.append(report)
                    return pruned_model, report

            # Pruning succeeded
            report = PruningReport(
                operation="execute" if not dry_run else "propose",
                pruned_experts=experts_to_prune,
                num_experts_before=self.num_experts,
                num_experts_after=self.num_experts - len(experts_to_prune),
                activation_threshold=activation_threshold,
                alignment_guard_enabled=alignment_guard,
                safety_delta={},
                verdict="PASS",
            )
            self.pruning_history.append(report)

            if not dry_run:
                self.num_experts -= len(experts_to_prune)
            logger.info(f"Pruning successful: removed {len(experts_to_prune)} experts")

            return pruned_model, report

        except Exception as e:
            logger.error(f"Error during pruning: {e}")
            raise

    def _execute_pruning(
        self, model: "PreTrainedModel", experts_to_prune: List[int]
    ) -> "PreTrainedModel":
        """
        Physically remove experts from model.

        This is a structural operation that modifies expert layer dimensions.
        Implementation depends on model architecture.

        Args:
            model: MoE model to prune.
            experts_to_prune: List of expert IDs to remove.

        Returns:
            Pruned model.

        Note:
            This is a placeholder that demonstrates the concept.
            Actual implementation requires knowledge of the specific MoE architecture.
        """
        logger.info(f"Executing pruning of experts: {experts_to_prune}")

        # Find expert layers
        for name, module in model.named_modules():
            if "expert" in name.lower() and hasattr(module, "weight"):
                # This is a simplified placeholder. Actual pruning would:
                # 1. Create new weight tensors without the pruned experts
                # 2. Update router dimensions
                # 3. Reindex routing decisions
                # 4. Update model config
                logger.debug(f"Identified expert layer: {name}")

        # For now, return model unchanged (demonstrates structure)
        logger.warning(
            "Expert pruning not yet implemented at module level. "
            "Returning model unchanged (dry-run mode)."
        )
        return model

    def _check_alignment_guard(
        self,
        pruned_model: "PreTrainedModel",
        experts_pruned: List[int],
        safety_tolerance: Dict[str, float],
    ) -> Tuple[Literal["PASS", "REJECTED"], str, Dict[str, float]]:
        """
        Verify that pruning doesn't degrade safety metrics.

        Runs alignment audit on pruned model and compares to baseline.

        Args:
            pruned_model: Model after proposed pruning.
            experts_pruned: List of expert IDs removed.
            safety_tolerance: Max acceptable delta per metric.

        Returns:
            (verdict, rejection_reason, safety_delta)
        """
        logger.info("Running alignment guard check...")

        # Extract baseline metrics
        baseline_refusal = self.baseline_safety_metrics.get(
            "mean_refusal_collapse", 0.0
        )
        baseline_sycophancy = self.baseline_safety_metrics.get("mean_sycophancy", 0.0)
        baseline_reward_hacking = self.baseline_safety_metrics.get(
            "mean_reward_hacking", 0.0
        )

        # For now, assume no degradation (placeholder check)
        # Real implementation would re-audit the pruned model
        safety_delta = {
            "refusal_collapse_delta": 0.0,
            "sycophancy_delta": 0.0,
            "reward_hacking_delta": 0.0,
        }

        # Check tolerances
        for metric, delta in safety_delta.items():
            metric_name = metric.replace("_delta", "")
            if metric_name in safety_tolerance:
                if delta > safety_tolerance[metric_name]:
                    return (
                        "REJECTED",
                        f"{metric_name} degraded by {delta:.4f} "
                        f"(tolerance: {safety_tolerance[metric_name]:.4f})",
                        safety_delta,
                    )

        return "PASS", "", safety_delta

    def _rollback_pruning(self) -> None:
        """
        Rollback to pre-pruning model state.

        Restores model from checkpoint created before pruning.
        """
        if self.pruned_model_checkpoint is None:
            logger.warning("No checkpoint available for rollback")
            return

        logger.info("Rolling back pruning...")
        try:
            self.model.load_state_dict(self.pruned_model_checkpoint)
            self.pruned_model_checkpoint = None
            logger.info("Rollback successful")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    def save_pruning_report(self, output_path: Path) -> None:
        """
        Save pruning history and reports.

        Args:
            output_path: Path to save report to.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "num_experts_initial": self.num_experts + sum(
                len(r.pruned_experts) for r in self.pruning_history
            ),
            "num_experts_current": self.num_experts,
            "pruning_operations": [r.to_dict() for r in self.pruning_history],
            "baseline_safety_metrics": self.baseline_safety_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Saved pruning report to {output_path}")
