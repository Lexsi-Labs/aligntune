#!/usr/bin/env python3
"""Analyze and plot baseline dynamics across PACE training runs."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")
SEEDS = ["s81", "s82", "s84"]
CONDITIONS = ["vanilla", "baseline_only", "curriculum_only", "full"]
CONDITION_LABELS = {
    "vanilla": "Vanilla GRPO",
    "baseline_only": "Baseline Only",
    "curriculum_only": "Curriculum Only",
    "full": "PACE (Full)"
}
CONDITION_COLORS = {
    "vanilla": "#888888",
    "baseline_only": "#2196F3",
    "curriculum_only": "#FF9800",
    "full": "#4CAF50"
}
GENERATIONS = {
    "g16": ("bolt_{seed}_{cond}", "G=16"),
    "g4": ("bolt_g4_{seed}_{cond}", "G=4"),
    "g2": ("bolt_g2_{seed}_{cond}", "G=2"),
}

def load_baseline_logs():
    """Load all baseline_state_log.jsonl files."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for gen_key, (pattern, label) in GENERATIONS.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_name = pattern.format(seed=seed, cond=cond)
                log_file = OUTPUT_DIR / run_name / "baseline_state_log.jsonl"

                if not log_file.exists():
                    continue

                steps = []
                global_mean = []
                global_std = []
                batch_reward = []
                pct_easy = []
                pct_hard = []
                pct_learning = []

                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        steps.append(entry["step"])
                        global_mean.append(entry["global_v_hat"]["mean"])
                        global_std.append(entry["global_v_hat"]["std"])
                        batch_reward.append(entry["batch_mean_reward"])
                        pct_easy.append(entry["curriculum"]["pct_easy"])
                        pct_hard.append(entry["curriculum"]["pct_hard"])
                        pct_learning.append(entry["curriculum"]["pct_learning_edge"])

                data[gen_key][cond][seed] = {
                    "steps": np.array(steps),
                    "global_mean": np.array(global_mean),
                    "global_std": np.array(global_std),
                    "batch_reward": np.array(batch_reward),
                    "pct_easy": np.array(pct_easy),
                    "pct_hard": np.array(pct_hard),
                    "pct_learning": np.array(pct_learning),
                }

    return data

def smooth(x, window=50):
    """Simple moving average smoothing."""
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_global_baseline_drift(data):
    """Plot how global baseline mean changes over training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # Aggregate across seeds
            all_means = []
            min_len = float('inf')

            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    vals = data[gen_key][cond][seed]["global_mean"]
                    all_means.append(vals)
                    min_len = min(min_len, len(vals))

            if not all_means:
                continue

            # Truncate to same length and compute mean/std
            all_means = [m[:min_len] for m in all_means]
            mean = np.mean(all_means, axis=0)
            std = np.std(all_means, axis=0)
            steps = data[gen_key][cond][SEEDS[0]]["steps"][:min_len]

            # Smooth
            mean_smooth = smooth(mean)
            steps_smooth = steps[:len(mean_smooth)]

            color = CONDITION_COLORS[cond]
            ax.plot(steps_smooth, mean_smooth, color=color,
                   label=CONDITION_LABELS[cond], linewidth=2)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1000)

        if idx == 0:
            ax.set_ylabel("Global Baseline Mean (v̂)", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig

def plot_curriculum_dynamics(data):
    """Plot curriculum sampling dynamics (easy/hard/learning edge)."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    gen_order = ["g2", "g4", "g16"]
    metrics = [("pct_easy", "% Easy (>0.8)", "#4CAF50"),
               ("pct_learning", "% Learning Edge (0.2-0.8)", "#FF9800"),
               ("pct_hard", "% Hard (<0.2)", "#F44336")]

    for row, (metric, metric_label, color) in enumerate(metrics):
        for col, gen_key in enumerate(gen_order):
            ax = axes[row, col]
            label = GENERATIONS[gen_key][1]

            for cond in ["curriculum_only", "full"]:  # Only curriculum conditions
                if cond not in data[gen_key]:
                    continue

                all_vals = []
                min_len = float('inf')

                for seed in SEEDS:
                    if seed in data[gen_key][cond]:
                        vals = data[gen_key][cond][seed][metric]
                        all_vals.append(vals)
                        min_len = min(min_len, len(vals))

                if not all_vals:
                    continue

                all_vals = [v[:min_len] for v in all_vals]
                mean = np.mean(all_vals, axis=0)
                steps = data[gen_key][cond][SEEDS[0]]["steps"][:min_len]

                mean_smooth = smooth(mean)
                steps_smooth = steps[:len(mean_smooth)]

                linestyle = '-' if cond == "full" else '--'
                ax.plot(steps_smooth, mean_smooth, color=color,
                       linestyle=linestyle, label=CONDITION_LABELS[cond], linewidth=2)

            if row == 0:
                ax.set_title(f"{label}", fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if row == 2:
                ax.set_xlabel("Training Steps", fontsize=11)

            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.01), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    return fig

def plot_batch_reward_progress(data):
    """Plot batch reward over training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            all_rewards = []
            min_len = float('inf')

            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    vals = data[gen_key][cond][seed]["batch_reward"]
                    all_rewards.append(vals)
                    min_len = min(min_len, len(vals))

            if not all_rewards:
                continue

            all_rewards = [r[:min_len] for r in all_rewards]
            mean = np.mean(all_rewards, axis=0)
            steps = data[gen_key][cond][SEEDS[0]]["steps"][:min_len]

            mean_smooth = smooth(mean)
            steps_smooth = steps[:len(mean_smooth)]

            color = CONDITION_COLORS[cond]
            ax.plot(steps_smooth, mean_smooth, color=color,
                   label=CONDITION_LABELS[cond], linewidth=2)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1)

        if idx == 0:
            ax.set_ylabel("Batch Mean Reward", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig

def print_summary_stats(data):
    """Print summary statistics."""
    print("\n=== Baseline Dynamics Summary ===\n")

    for gen_key in ["g2", "g4", "g16"]:
        print(f"\n{GENERATIONS[gen_key][1]}:")
        print("-" * 50)

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # Get final values
            final_means = []
            final_rewards = []

            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    final_means.append(data[gen_key][cond][seed]["global_mean"][-1])
                    final_rewards.append(np.mean(data[gen_key][cond][seed]["batch_reward"][-100:]))

            if final_means:
                print(f"  {CONDITION_LABELS[cond]:20s}: "
                      f"final_v̂={np.mean(final_means):.3f}±{np.std(final_means):.3f}, "
                      f"final_reward={np.mean(final_rewards):.3f}±{np.std(final_rewards):.3f}")

def main():
    print("Loading baseline logs...")
    data = load_baseline_logs()

    print_summary_stats(data)

    print("\nGenerating plots...")

    # Plot 1: Global baseline drift
    fig1 = plot_global_baseline_drift(data)
    fig1.savefig("plots/baseline_drift.png", dpi=150, bbox_inches='tight')
    fig1.savefig("plots/baseline_drift.pdf", bbox_inches='tight')
    print("Saved: plots/baseline_drift.png/.pdf")

    # Plot 2: Curriculum dynamics
    fig2 = plot_curriculum_dynamics(data)
    fig2.savefig("plots/curriculum_dynamics.png", dpi=150, bbox_inches='tight')
    fig2.savefig("plots/curriculum_dynamics.pdf", bbox_inches='tight')
    print("Saved: plots/curriculum_dynamics.png/.pdf")

    # Plot 3: Batch reward progress
    fig3 = plot_batch_reward_progress(data)
    fig3.savefig("plots/batch_reward_progress.png", dpi=150, bbox_inches='tight')
    fig3.savefig("plots/batch_reward_progress.pdf", bbox_inches='tight')
    print("Saved: plots/batch_reward_progress.png/.pdf")

if __name__ == "__main__":
    main()
