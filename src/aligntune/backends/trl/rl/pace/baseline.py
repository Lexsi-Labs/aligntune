"""
BOLT Baseline Module - Beta-Bernoulli tracker with KL-adaptive forgetting.

Implements persistent per-prompt baselines for:
1. Curriculum sampling: weight ∝ sqrt(v̂(1-v̂)) + ε (uncertainty-based)
2. Advantage computation: A = r - v̂(x) (SPO-style persistent baseline)

Key features:
- Pre-update reads: v̂ reflects history before current reward
- Post-update writes: incorporates new reward with forgetting
- KL-adaptive forgetting: faster adaptation when policy shifts
"""

import os
import math
import json
import pickle
import logging
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class BaselineStore:
    """
    Beta-Bernoulli tracker for persistent per-prompt baselines.

    Maintains running estimate v̂(x) of success probability for each prompt,
    with adaptive forgetting based on KL divergence.

    Key equation (SPO Eq 5):
        ρ(KL) = 2^(-KL/D_half)
        - At KL = D_half: ρ = 0.5 (half-decay)
        - At KL = 0: ρ = 1.0 (full memory)
        - At KL = 2×D_half: ρ = 0.25 (quarter memory)
    """

    def __init__(
        self,
        rho_min: float = 0.875,
        rho_max: float = 0.96,
        D_half: float = 0.5,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        track_timeline: bool = False,
    ):
        """
        Initialize baseline store.

        Args:
            rho_min: Minimum forgetting factor (faster adaptation)
            rho_max: Maximum forgetting factor (slower adaptation)
            D_half: KL half-life for adaptive forgetting
            alpha_init: Initial Beta prior α (pseudo-successes)
            beta_init: Initial Beta prior β (pseudo-failures)
            track_timeline: Whether to track v̂ evolution over time
        """
        # Baseline storage: prompt_key -> (α, β, kl_ema)
        self.tab: Dict[str, Tuple[float, float, float]] = {}

        self.rho_min = rho_min
        self.rho_max = rho_max
        self.D_half = D_half
        self.alpha_init = alpha_init
        self.beta_init = beta_init

        # Timeline tracking: prompt_key -> [(step, v_hat, alpha, beta, reward)]
        self.track_timeline = track_timeline
        self.timeline: Optional[Dict[str, List[Tuple[int, float, float, float, float]]]] = (
            {} if track_timeline else None
        )
        self.current_step = 0

    def get(self, key: str) -> Tuple[float, float]:
        """
        Get current baseline estimate (pre-update read).

        Args:
            key: Prompt identifier

        Returns:
            (v_hat, N_eff): Success probability estimate and effective sample size
        """
        a, b, _ = self.tab.get(key, (self.alpha_init, self.beta_init, 0.0))
        v_hat = a / (a + b)
        N_eff = a + b
        return v_hat, N_eff

    def get_v_hat(self, key: str) -> float:
        """Get baseline estimate v̂(x) = α/(α+β)"""
        v_hat, _ = self.get(key)
        return v_hat

    def get_uncertainty(self, key: str) -> float:
        """Get uncertainty sqrt(v̂(1-v̂)) for curriculum sampling"""
        v_hat = self.get_v_hat(key)
        return math.sqrt(v_hat * (1 - v_hat))

    def set_step(self, step: int):
        """Set current training step for timeline tracking."""
        self.current_step = step

    def update(self, key: str, reward: float, kl: Optional[float] = None):
        """
        Update baseline with new reward (post-update write).

        Uses KL-adaptive forgetting: higher KL → faster adaptation.

        Args:
            key: Prompt identifier
            reward: Binary reward (0 or 1, or continuous 0-1)
            kl: Optional KL divergence for adaptive forgetting
        """
        a, b, kl_ema = self.tab.get(key, (self.alpha_init, self.beta_init, 0.0))

        # KL-adaptive forgetting (SPO Equation 5)
        if kl is not None:
            # Higher KL → lower rho → faster forgetting
            rho = 2 ** (-kl / self.D_half)
            rho = max(self.rho_min, min(self.rho_max, rho))
            # Update KL EMA for tracking
            kl_ema = 0.9 * kl_ema + 0.1 * kl
        else:
            # Conservative: slow forgetting when KL unavailable
            rho = self.rho_max

        # Beta update with forgetting
        # Binarize reward for Beta distribution
        r_binary = 1.0 if reward > 0.5 else 0.0

        # Exponential moving average with forgetting
        a_new = rho * a + r_binary
        b_new = rho * b + (1.0 - r_binary)

        self.tab[key] = (a_new, b_new, kl_ema)

        # Track timeline if enabled
        if self.track_timeline and self.timeline is not None:
            v_hat_new = a_new / (a_new + b_new)
            if key not in self.timeline:
                self.timeline[key] = []
            self.timeline[key].append((self.current_step, v_hat_new, a_new, b_new, r_binary))

    def get_stats(self) -> Dict[str, float]:
        """
        Compute aggregate statistics for logging.

        Returns:
            Dictionary with baseline metrics
        """
        if not self.tab:
            return {
                "baseline/avg_v_hat": 0.0,
                "baseline/avg_N_eff": 0.0,
                "baseline/num_prompts": 0,
            }

        v_hats = [a / (a + b) for a, b, _ in self.tab.values()]
        N_effs = [a + b for a, b, _ in self.tab.values()]

        return {
            "baseline/avg_v_hat": sum(v_hats) / len(v_hats),
            "baseline/avg_N_eff": sum(N_effs) / len(N_effs),
            "baseline/num_prompts": len(self.tab),
            "baseline/min_v_hat": min(v_hats),
            "baseline/max_v_hat": max(v_hats),
        }

    def save(self, path: str):
        """Save baseline store to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.tab, f)
        logger.info(f"Saved baseline to {path}")

    def load(self, path: str):
        """Load baseline store from disk."""
        with open(path, "rb") as f:
            self.tab = pickle.load(f)
        logger.info(f"Loaded baseline from {path}: {len(self.tab)} prompts")


class UnifiedBaseline(BaselineStore):
    """
    Extended baseline with curriculum stats and advantage computation.

    Single baseline v̂(x) per prompt serves BOTH purposes:
    1. Curriculum sampling: weight ∝ sqrt(v̂(1-v̂)) + ε (uncertainty-based)
    2. Advantage computation: A = r - v̂(x) (SPO-style persistent baseline)

    Benefits:
    - Enables small group sizes (K=1-2): baseline provides stable advantage
    - Faster dataset iteration: cover more prompts per epoch
    - KL-adaptive forgetting: automatically adjusts to policy shifts
    """

    def __init__(
        self,
        rho_min: float = 0.875,
        rho_max: float = 0.96,
        D_half: float = 0.5,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        epsilon: float = 0.05,
        track_timeline: bool = False,
    ):
        """
        Args:
            rho_min: Min forgetting factor (fast adaptation)
            rho_max: Max forgetting factor (slow adaptation)
            D_half: KL half-life for adaptive decay
            alpha_init: Initial Beta α (pseudo-successes)
            beta_init: Initial Beta β (pseudo-failures)
            epsilon: Floor for sampling weights
            track_timeline: Track v̂ evolution over time
        """
        super().__init__(
            rho_min=rho_min,
            rho_max=rho_max,
            D_half=D_half,
            alpha_init=alpha_init,
            beta_init=beta_init,
            track_timeline=track_timeline,
        )
        self.epsilon = epsilon

        # Sampling stats
        self.sampling_counts: Dict[str, int] = defaultdict(int)
        self.rollout_history: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)

    def get_sampling_weight(self, key: str) -> float:
        """Get curriculum sampling weight: sqrt(v̂(1-v̂)) + ε"""
        return self.get_uncertainty(key) + self.epsilon

    def get_params(self, key: str) -> Tuple[float, float, float]:
        """Get full Beta parameters (α, β, N_eff)"""
        a, b, _ = self.tab.get(key, (self.alpha_init, self.beta_init, 0.0))
        N_eff = a + b
        return a, b, N_eff

    def update(self, key: str, reward: float, kl: Optional[float] = None):
        """Update baseline and track rollout history."""
        super().update(key, reward, kl)

        # Track rollout history
        v_hat = self.get_v_hat(key)
        self.rollout_history[key].append((reward, v_hat, self.current_step))

    def estimate_kl_from_performance_shift(self, key: str, batch_rewards: List[float]) -> float:
        """
        Estimate KL divergence from performance shift.

        Proxy: |current_batch_mean - v̂| × 2.0
        Larger shifts indicate policy has changed → higher KL → faster forgetting

        Args:
            key: Prompt identifier
            batch_rewards: Rewards for this prompt in current batch

        Returns:
            Estimated KL divergence
        """
        v_hat = self.get_v_hat(key)
        batch_mean = sum(batch_rewards) / len(batch_rewards) if batch_rewards else v_hat

        # Scale performance shift to KL range
        # |shift| ∈ [0, 1] → KL ∈ [0, 2]
        estimated_kl = abs(batch_mean - v_hat) * 2.0

        return estimated_kl

    def increment_sampling_count(self, key: str):
        """Track how many times prompt was sampled"""
        self.sampling_counts[key] += 1

    def get_curriculum_stats(self) -> Dict[str, float]:
        """Compute curriculum metrics for logging"""
        if not self.tab:
            return {}

        v_hats = [self.get_v_hat(k) for k in self.tab]
        uncertainties = [self.get_uncertainty(k) for k in self.tab]
        weights = [self.get_sampling_weight(k) for k in self.tab]

        n_total = len(v_hats)
        n_easy = sum(1 for v in v_hats if v > 0.8)
        n_hard = sum(1 for v in v_hats if v < 0.2)
        n_uncertain = sum(1 for v in v_hats if 0.4 <= v <= 0.6)
        n_learning_edge = sum(1 for v in v_hats if 0.3 <= v <= 0.7)

        return {
            "curriculum/n_tracked": n_total,
            "curriculum/mean_v_hat": float(np.mean(v_hats)),
            "curriculum/std_v_hat": float(np.std(v_hats)),
            "curriculum/mean_uncertainty": float(np.mean(uncertainties)),
            "curriculum/max_uncertainty": float(np.max(uncertainties)) if uncertainties else 0.0,
            "curriculum/mean_weight": float(np.mean(weights)),
            "curriculum/pct_easy": 100 * n_easy / n_total if n_total > 0 else 0.0,
            "curriculum/pct_hard": 100 * n_hard / n_total if n_total > 0 else 0.0,
            "curriculum/pct_uncertain": 100 * n_uncertain / n_total if n_total > 0 else 0.0,
            "curriculum/pct_learning_edge": 100 * n_learning_edge / n_total if n_total > 0 else 0.0,
        }

    def get_baseline_stats(self) -> Dict[str, float]:
        """Compute baseline metrics for logging"""
        if not self.tab:
            return {
                "baseline/avg_v_hat": 0.0,
                "baseline/avg_N_eff": 0.0,
                "baseline/num_prompts": 0,
            }

        v_hats = [a / (a + b) for a, b, _ in self.tab.values()]
        N_effs = [a + b for a, b, _ in self.tab.values()]
        kl_emas = [kl for _, _, kl in self.tab.values()]

        return {
            "baseline/avg_v_hat": float(np.mean(v_hats)),
            "baseline/avg_N_eff": float(np.mean(N_effs)),
            "baseline/avg_kl_ema": float(np.mean(kl_emas)),
            "baseline/num_prompts": len(self.tab),
            "baseline/min_v_hat": min(v_hats),
            "baseline/max_v_hat": max(v_hats),
        }

    def save(self, path: str):
        """Save baseline store to disk with full state."""
        data = {
            "tab": self.tab,
            "sampling_counts": dict(self.sampling_counts),
            "rollout_history": dict(self.rollout_history),
            "timeline": self.timeline,
            "current_step": self.current_step,
            "config": {
                "rho_min": self.rho_min,
                "rho_max": self.rho_max,
                "D_half": self.D_half,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init,
                "epsilon": self.epsilon,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved unified baseline to {path}: {len(self.tab)} prompts")

    def load(self, path: str, verbose: bool = True):
        """Load baseline store from disk."""
        if path.endswith(".json"):
            self.load_from_json(path, verbose=verbose)
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "tab" in data:
            self.tab = data.get("tab", {})
            self.sampling_counts = defaultdict(int, data.get("sampling_counts", {}))
            self.rollout_history = defaultdict(list, data.get("rollout_history", {}))
            self.timeline = data.get("timeline", {} if self.track_timeline else None)
            self.current_step = data.get("current_step", 0)

            # Load config if available
            config = data.get("config", {})
            self.rho_min = config.get("rho_min", self.rho_min)
            self.rho_max = config.get("rho_max", self.rho_max)
            self.D_half = config.get("D_half", self.D_half)
            self.alpha_init = config.get("alpha_init", self.alpha_init)
            self.beta_init = config.get("beta_init", self.beta_init)
            self.epsilon = config.get("epsilon", self.epsilon)
        else:
            # Legacy format: just the tab dict
            self.tab = data

        v_hats = [a/(a+b) for a,b,_ in self.tab.values()]
        logger.info(
            f"Loaded unified baseline from {path}: {len(self.tab)} prompts, "
            f"mean v̂={np.mean(v_hats):.3f}"
        )

        # Print detailed baseline info
        if verbose:
            self._print_baseline_summary()

    def load_from_json(self, path: str, verbose: bool = True):
        """
        Load baseline values from JSON file (v̂ dictionary).

        JSON format:
        {
            "baselines": {
                "prompt_key": v_hat_value,
                ...
            }
        }
        or simply:
        {
            "prompt_key": v_hat_value,
            ...
        }
        """
        if not os.path.exists(path):
            logger.warning(f"Baseline file not found: {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Handle both formats
        if "baselines" in data:
            baseline_values = data["baselines"]
        else:
            baseline_values = data

        # Convert v̂ values to Beta(α, β) parameters
        # Using method of moments with equivalent of 10 observations for warm start
        n_equiv = 10.0
        for key, v_hat in baseline_values.items():
            if not isinstance(v_hat, (int, float)):
                continue
            v_hat = max(0.01, min(0.99, float(v_hat)))  # Clip to avoid extremes
            alpha = v_hat * n_equiv
            beta = (1 - v_hat) * n_equiv
            self.tab[key] = (alpha, beta, 0.0)

        logger.info(
            f"Loaded baseline from JSON {path}: {len(self.tab)} prompts, "
            f"mean v̂={np.mean(list(baseline_values.values())):.3f}"
        )

        # Print detailed baseline info
        if verbose:
            self._print_baseline_summary()

    def _print_baseline_summary(self, max_display: int = 20):
        """Print detailed summary of loaded baselines."""
        if not self.tab:
            print("\n" + "=" * 80)
            print("BASELINE SUMMARY: No prompts loaded")
            print("=" * 80 + "\n")
            return

        # Compute stats
        items = [(k, a/(a+b), a+b) for k, (a, b, _) in self.tab.items()]
        items_sorted = sorted(items, key=lambda x: x[1])  # Sort by v̂

        v_hats = [v for _, v, _ in items]
        n_effs = [n for _, _, n in items]

        print("\n" + "=" * 80)
        print("BASELINE SUMMARY")
        print("=" * 80)
        print(f"Total prompts: {len(self.tab)}")
        print(f"Mean v̂: {np.mean(v_hats):.4f}")
        print(f"Std v̂:  {np.std(v_hats):.4f}")
        print(f"Min v̂:  {min(v_hats):.4f}")
        print(f"Max v̂:  {max(v_hats):.4f}")
        print(f"Mean N_eff: {np.mean(n_effs):.2f}")
        print()

        # Distribution buckets
        n_total = len(v_hats)
        buckets = [
            ("Very Hard (v̂ < 0.1)", sum(1 for v in v_hats if v < 0.1)),
            ("Hard (0.1 ≤ v̂ < 0.3)", sum(1 for v in v_hats if 0.1 <= v < 0.3)),
            ("Learning Edge (0.3 ≤ v̂ < 0.7)", sum(1 for v in v_hats if 0.3 <= v < 0.7)),
            ("Easy (0.7 ≤ v̂ < 0.9)", sum(1 for v in v_hats if 0.7 <= v < 0.9)),
            ("Very Easy (v̂ ≥ 0.9)", sum(1 for v in v_hats if v >= 0.9)),
        ]
        print("Distribution:")
        for label, count in buckets:
            pct = 100 * count / n_total if n_total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:30s}: {count:5d} ({pct:5.1f}%) {bar}")
        print()

        # Show sample prompts
        print(f"Sample prompts (showing {min(max_display, len(items_sorted))} of {len(items_sorted)}):")
        print("-" * 80)

        # Show hardest prompts
        n_show = min(max_display // 2, len(items_sorted))
        print(f"HARDEST {n_show} prompts:")
        for key, v_hat, n_eff in items_sorted[:n_show]:
            prompt_preview = key[:60] + "..." if len(key) > 60 else key
            prompt_preview = prompt_preview.replace("\n", " ")
            print(f"  v̂={v_hat:.3f} (N={n_eff:.1f}): {prompt_preview}")
        print()

        # Show easiest prompts
        print(f"EASIEST {n_show} prompts:")
        for key, v_hat, n_eff in items_sorted[-n_show:]:
            prompt_preview = key[:60] + "..." if len(key) > 60 else key
            prompt_preview = prompt_preview.replace("\n", " ")
            print(f"  v̂={v_hat:.3f} (N={n_eff:.1f}): {prompt_preview}")

        print("=" * 80 + "\n")

    def print_step_update(self, step: int, prompt_keys: List[str],
                          old_v_hats: Dict[str, float], rewards: List[float]):
        """Print v̂ changes after a training step."""
        print(f"\n--- BOLT Step {step} Baseline Update ---")
        print(f"Prompts updated: {len(prompt_keys)}")

        changes = []
        for i, key in enumerate(prompt_keys):
            old_v = old_v_hats.get(key, 0.5)
            new_v = self.get_v_hat(key)
            reward = rewards[i] if i < len(rewards) else 0.0
            delta = new_v - old_v
            changes.append((key, old_v, new_v, delta, reward))

        # Sort by absolute change
        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        # Show top changes
        n_show = min(5, len(changes))
        print(f"\nTop {n_show} changes:")
        for key, old_v, new_v, delta, reward in changes[:n_show]:
            prompt_preview = key[:50] + "..." if len(key) > 50 else key
            prompt_preview = prompt_preview.replace("\n", " ")
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {arrow} v̂: {old_v:.3f} → {new_v:.3f} (Δ={delta:+.3f}, r={reward:.1f}): {prompt_preview}")

        # Summary stats
        deltas = [c[3] for c in changes]
        print(f"\nSummary: mean Δv̂={np.mean(deltas):+.4f}, max |Δv̂|={max(abs(d) for d in deltas):.4f}")
        print(f"Current baseline: mean v̂={np.mean([self.get_v_hat(k) for k in prompt_keys]):.4f}")
        print("-" * 40)


def make_prompt_key(prompt: str) -> str:
    """
    Create stable key for prompt.

    Uses first 200 chars for readability in logs.
    For very long prompts, appends hash for uniqueness.

    Args:
        prompt: Full prompt text

    Returns:
        Stable string key
    """
    if len(prompt) <= 200:
        return prompt
    return f"{prompt[:100]}...{hash(prompt)}"
