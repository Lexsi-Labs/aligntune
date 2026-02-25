#!/usr/bin/env python3
"""Plot PACE ablation results across G=2, G=4, G=16."""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Config
EVAL_DIR = Path("eval_results")
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

def load_results():
    """Load all eval results into nested dict."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for gen_key, (pattern, label) in GENERATIONS.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_name = pattern.format(seed=seed, cond=cond)
                run_dir = EVAL_DIR / run_name

                if not run_dir.exists():
                    continue

                for cp_dir in run_dir.iterdir():
                    if not cp_dir.is_dir() or not cp_dir.name.startswith("cp"):
                        continue

                    summary_file = cp_dir / "summary.json"
                    if not summary_file.exists():
                        continue

                    step = int(cp_dir.name[2:])  # "cp500" -> 500

                    with open(summary_file) as f:
                        data = json.load(f)

                    results[gen_key][cond][step][seed] = data["mbpp_plus_pass@1"]

    return results

def compute_stats(results):
    """Compute mean and std across seeds for each step."""
    stats = defaultdict(lambda: defaultdict(dict))

    for gen_key in results:
        for cond in results[gen_key]:
            steps = sorted(results[gen_key][cond].keys())
            means = []
            stds = []
            valid_steps = []

            for step in steps:
                values = list(results[gen_key][cond][step].values())
                if len(values) >= 2:  # Need at least 2 seeds for std
                    valid_steps.append(step)
                    means.append(np.mean(values))
                    stds.append(np.std(values))

            stats[gen_key][cond] = {
                "steps": valid_steps,
                "mean": means,
                "std": stds
            }

    return stats

def plot_results(stats):
    """Create subplot for each generation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in stats[gen_key]:
                continue

            data = stats[gen_key][cond]
            steps = data["steps"]
            mean = np.array(data["mean"])
            std = np.array(data["std"])

            color = CONDITION_COLORS[cond]
            label_text = CONDITION_LABELS[cond]

            ax.plot(steps, mean, color=color, label=label_text, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1000)

        if idx == 0:
            ax.set_ylabel("MBPP+ Pass@1", fontsize=12)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return fig

def plot_final_comparison(results):
    """Bar chart comparing final performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_order = ["g2", "g4", "g16"]
    x = np.arange(len(gen_order))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        final_means = []
        final_stds = []

        for gen_key in gen_order:
            if gen_key in results and cond in results[gen_key]:
                max_step = max(results[gen_key][cond].keys())
                values = list(results[gen_key][cond][max_step].values())
                final_means.append(np.mean(values))
                final_stds.append(np.std(values))
            else:
                final_means.append(0)
                final_stds.append(0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, final_means, width, yerr=final_stds,
                      label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
                      capsize=3)

    ax.set_ylabel("MBPP+ Pass@1", fontsize=12)
    ax.set_xlabel("Number of Generations (G)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["G=2", "G=4", "G=16"])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Final Performance by Condition and Generation Count", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    print("Loading results...")
    results = load_results()

    print("Computing statistics...")
    stats = compute_stats(results)

    # Print summary
    print("\n=== Final Performance (step 1000) ===")
    for gen_key in ["g2", "g4", "g16"]:
        print(f"\n{GENERATIONS[gen_key][1]}:")
        for cond in CONDITIONS:
            if gen_key in results and cond in results[gen_key]:
                max_step = max(results[gen_key][cond].keys())
                values = list(results[gen_key][cond][max_step].values())
                print(f"  {CONDITION_LABELS[cond]:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\nGenerating plots...")

    # Learning curves plot
    fig1 = plot_results(stats)
    fig1.savefig("plots/pace_ablation_curves.png", dpi=150, bbox_inches='tight')
    fig1.savefig("plots/pace_ablation_curves.pdf", bbox_inches='tight')
    print("Saved: plots/pace_ablation_curves.png/.pdf")

    # Final comparison bar chart
    fig2 = plot_final_comparison(results)
    fig2.savefig("plots/pace_ablation_final.png", dpi=150, bbox_inches='tight')
    fig2.savefig("plots/pace_ablation_final.pdf", bbox_inches='tight')
    print("Saved: plots/pace_ablation_final.png/.pdf")

    plt.show()

if __name__ == "__main__":
    main()
