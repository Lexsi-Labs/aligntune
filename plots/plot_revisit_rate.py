#!/usr/bin/env python3
"""Analyze and plot revisit rates across PACE training runs.

Revisit rate measures how often each problem is sampled during training,
which is a key indicator of curriculum sampling effectiveness.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
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


def load_sampling_logs():
    """Load sampling_log.jsonl files and compute revisit statistics."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for gen_key, (pattern, label) in GENERATIONS.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_name = pattern.format(seed=seed, cond=cond)
                log_file = OUTPUT_DIR / run_name / "sampling_log.jsonl"

                if not log_file.exists():
                    continue

                prompt_counts = Counter()
                total_entries = 0
                max_step = 0

                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        prompt_counts[entry["prompt_idx"]] += 1
                        total_entries += 1
                        max_step = max(max_step, entry["step"])

                if total_entries == 0:
                    continue

                # Compute statistics
                counts = np.array(list(prompt_counts.values()))
                num_unique = len(prompt_counts)

                # Expected count under uniform sampling
                # Each step samples batch_size prompts, so expected = total_entries / num_unique
                expected_uniform = total_entries / num_unique if num_unique > 0 else 1

                data[gen_key][cond][seed] = {
                    "prompt_counts": prompt_counts,
                    "counts_array": counts,
                    "total_entries": total_entries,
                    "max_step": max_step,
                    "num_unique": num_unique,
                    "expected_uniform": expected_uniform,
                    "mean_count": np.mean(counts),
                    "std_count": np.std(counts),
                    "max_count": np.max(counts),
                    "min_count": np.min(counts),
                    # Gini coefficient (inequality measure)
                    "gini": compute_gini(counts),
                    # Coefficient of variation
                    "cv": np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0,
                }

    # For G=2 and G=4: vanilla should use baseline_only data (both are uniform sampling)
    for gen_key in ["g2", "g4"]:
        if "baseline_only" in data[gen_key]:
            data[gen_key]["vanilla"] = data[gen_key]["baseline_only"]

    return data


def compute_gini(values):
    """Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def plot_revisit_histogram(data):
    """Plot histogram of revisit counts for each condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # Aggregate counts across seeds
            all_counts = []
            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    all_counts.extend(data[gen_key][cond][seed]["counts_array"])

            if not all_counts:
                continue

            counts = np.array(all_counts)
            color = CONDITION_COLORS[cond]

            # Histogram with density normalization
            bins = np.linspace(0, max(counts) + 1, 30)
            ax.hist(counts, bins=bins, alpha=0.5, color=color,
                   label=CONDITION_LABELS[cond], density=True)

        ax.set_xlabel("Times Sampled", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Density", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


def plot_revisit_gini(data):
    """Bar chart comparing Gini coefficients (sampling inequality)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_order = ["g2", "g4", "g16"]
    x = np.arange(len(gen_order))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        ginis = []
        stds = []

        for gen_key in gen_order:
            if gen_key in data and cond in data[gen_key]:
                vals = [data[gen_key][cond][seed]["gini"]
                       for seed in SEEDS if seed in data[gen_key][cond]]
                if vals:
                    ginis.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    ginis.append(0)
                    stds.append(0)
            else:
                ginis.append(0)
                stds.append(0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, ginis, width, yerr=stds,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
               capsize=3)

    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_xlabel("Number of Generations (G)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["G=2", "G=4", "G=16"])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Sampling Inequality (Gini)\nHigher = More Focused Curriculum", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_revisit_cv(data):
    """Bar chart comparing coefficient of variation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_order = ["g2", "g4", "g16"]
    x = np.arange(len(gen_order))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        cvs = []
        stds = []

        for gen_key in gen_order:
            if gen_key in data and cond in data[gen_key]:
                vals = [data[gen_key][cond][seed]["cv"]
                       for seed in SEEDS if seed in data[gen_key][cond]]
                if vals:
                    cvs.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    cvs.append(0)
                    stds.append(0)
            else:
                cvs.append(0)
                stds.append(0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, cvs, width, yerr=stds,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
               capsize=3)

    ax.set_ylabel("Coefficient of Variation", fontsize=12)
    ax.set_xlabel("Number of Generations (G)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["G=2", "G=4", "G=16"])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Sampling Variability (CV)\nHigher = More Non-Uniform Sampling", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_max_revisit(data):
    """Plot maximum revisit count (how many times the most-sampled problem was seen)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_order = ["g2", "g4", "g16"]
    x = np.arange(len(gen_order))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        max_counts = []
        stds = []

        for gen_key in gen_order:
            if gen_key in data and cond in data[gen_key]:
                vals = [data[gen_key][cond][seed]["max_count"]
                       for seed in SEEDS if seed in data[gen_key][cond]]
                if vals:
                    max_counts.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    max_counts.append(0)
                    stds.append(0)
            else:
                max_counts.append(0)
                stds.append(0)

        offset = (i - 1.5) * width
        ax.bar(x + offset, max_counts, width, yerr=stds,
               label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
               capsize=3)

    ax.set_ylabel("Max Times Sampled", fontsize=12)
    ax.set_xlabel("Number of Generations (G)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["G=2", "G=4", "G=16"])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Maximum Problem Revisit Count", fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_coverage_over_training(data):
    """Plot how many unique problems have been seen over training steps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # We need to reload and track cumulative coverage
            # For now, just use the summary stats
            for seed in SEEDS:
                if seed not in data[gen_key][cond]:
                    continue

                stats = data[gen_key][cond][seed]
                # Plot a simple point showing final coverage
                coverage = stats["num_unique"]
                total = stats["total_entries"]

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Unique Problems Seen", fontsize=12)

    plt.tight_layout()
    return fig


def print_summary(data):
    """Print summary statistics."""
    print("\n=== Revisit Rate Summary ===\n")

    for gen_key in ["g2", "g4", "g16"]:
        print(f"\n{GENERATIONS[gen_key][1]}:")
        print("-" * 70)
        print(f"{'Condition':<20} {'Mean±Std':<15} {'Max':<8} {'Gini':<8} {'CV':<8}")
        print("-" * 70)

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            means = []
            maxes = []
            ginis = []
            cvs = []

            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    stats = data[gen_key][cond][seed]
                    means.append(stats["mean_count"])
                    maxes.append(stats["max_count"])
                    ginis.append(stats["gini"])
                    cvs.append(stats["cv"])

            if means:
                print(f"  {CONDITION_LABELS[cond]:<18} "
                      f"{np.mean(means):.1f}±{np.std(means):.1f}    "
                      f"{np.mean(maxes):.0f}      "
                      f"{np.mean(ginis):.3f}    "
                      f"{np.mean(cvs):.3f}")


def load_sampling_with_vhat():
    """Load sampling logs and track v̂ values for each problem."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for gen_key, (pattern, label) in GENERATIONS.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_name = pattern.format(seed=seed, cond=cond)
                log_file = OUTPUT_DIR / run_name / "sampling_log.jsonl"

                if not log_file.exists():
                    continue

                prompt_counts = Counter()
                prompt_final_vhat = {}  # Last observed v̂ for each prompt

                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        idx = entry["prompt_idx"]
                        prompt_counts[idx] += 1
                        prompt_final_vhat[idx] = entry["new_v_hat"]

                if not prompt_counts:
                    continue

                data[gen_key][cond][seed] = {
                    "counts": prompt_counts,
                    "final_vhat": prompt_final_vhat,
                }

    # For G=2 and G=4: vanilla should use baseline_only data (both are uniform sampling)
    for gen_key in ["g2", "g4"]:
        if "baseline_only" in data[gen_key]:
            data[gen_key]["vanilla"] = data[gen_key]["baseline_only"]

    return data


def plot_vhat_vs_revisit(data):
    """Scatter plot: problem difficulty (v̂) vs revisit count."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    gen_order = ["g2", "g4", "g16"]

    # All 4 conditions: Vanilla, Baseline, Curriculum, PACE
    conditions_to_plot = ["vanilla", "baseline_only", "curriculum_only", "full"]

    for row, cond in enumerate(conditions_to_plot):
        for col, gen_key in enumerate(gen_order):
            ax = axes[row, col]
            label = GENERATIONS[gen_key][1]

            if cond not in data[gen_key]:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                continue

            # Aggregate across seeds
            all_vhat = []
            all_counts = []

            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    seed_data = data[gen_key][cond][seed]
                    for idx, count in seed_data["counts"].items():
                        if idx in seed_data["final_vhat"]:
                            all_vhat.append(seed_data["final_vhat"][idx])
                            all_counts.append(count)

            if not all_vhat:
                continue

            vhat = np.array(all_vhat)
            counts = np.array(all_counts)

            # Scatter with alpha for density
            ax.scatter(vhat, counts, alpha=0.3, s=10,
                      color=CONDITION_COLORS[cond])

            # Add binned mean line
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            for i in range(len(bins) - 1):
                mask = (vhat >= bins[i]) & (vhat < bins[i+1])
                if np.sum(mask) > 0:
                    bin_means.append(np.mean(counts[mask]))
                else:
                    bin_means.append(np.nan)

            ax.plot(bin_centers, bin_means, 'k-', linewidth=2, label='Binned mean')

            # Highlight learning edge region
            ax.axvspan(0.2, 0.8, alpha=0.1, color='green', label='Learning Edge')

            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"{label}", fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{CONDITION_LABELS[cond]}\nTimes Sampled", fontsize=11)
            if row == 3:  # Last row
                ax.set_xlabel("Problem Difficulty (v̂)", fontsize=11)

    fig.suptitle("Revisit Count vs Problem Difficulty\n(Green = Learning Edge: 0.2 < v̂ < 0.8)",
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def load_staleness_data():
    """Load sampling logs and compute staleness (steps between consecutive visits)."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for gen_key, (pattern, label) in GENERATIONS.items():
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_name = pattern.format(seed=seed, cond=cond)
                log_file = OUTPUT_DIR / run_name / "sampling_log.jsonl"

                if not log_file.exists():
                    continue

                # Track step when each prompt was last seen
                prompt_last_step = {}
                # Track all inter-visit intervals per prompt
                prompt_intervals = defaultdict(list)
                # Track intervals over training time (binned)
                intervals_by_step = defaultdict(list)

                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        idx = entry["prompt_idx"]
                        step = entry["step"]

                        if idx in prompt_last_step:
                            interval = step - prompt_last_step[idx]
                            prompt_intervals[idx].append(interval)
                            # Bin by step (every 100 steps)
                            step_bin = (step // 100) * 100
                            intervals_by_step[step_bin].append(interval)

                        prompt_last_step[idx] = step

                if not prompt_intervals:
                    continue

                # Compute summary stats
                all_intervals = [i for intervals in prompt_intervals.values() for i in intervals]

                data[gen_key][cond][seed] = {
                    "all_intervals": np.array(all_intervals),
                    "intervals_by_step": dict(intervals_by_step),
                    "mean_interval": np.mean(all_intervals) if all_intervals else 0,
                    "median_interval": np.median(all_intervals) if all_intervals else 0,
                }

    # For G=2 and G=4: vanilla should use baseline_only data
    for gen_key in ["g2", "g4"]:
        if "baseline_only" in data[gen_key]:
            data[gen_key]["vanilla"] = data[gen_key]["baseline_only"]

    return data


def plot_staleness_over_time(data):
    """Plot mean inter-visit interval over training steps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # Aggregate intervals_by_step across seeds
            all_step_intervals = defaultdict(list)
            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    for step_bin, intervals in data[gen_key][cond][seed]["intervals_by_step"].items():
                        all_step_intervals[step_bin].extend(intervals)

            if not all_step_intervals:
                continue

            # Compute mean interval per step bin (exclude last bin - edge effect)
            steps = sorted([s for s in all_step_intervals.keys() if s < 1000])
            means = [np.mean(all_step_intervals[s]) for s in steps]

            color = CONDITION_COLORS[cond]
            ax.plot(steps, means, color=color, label=CONDITION_LABELS[cond], linewidth=2)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 900)

        if idx == 0:
            ax.set_ylabel("Mean Steps Between Revisits", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    fig.suptitle("Baseline Staleness: Steps Between Consecutive Problem Visits",
                fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig


def plot_staleness_histogram(data):
    """Histogram of inter-visit intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gen_order = ["g2", "g4", "g16"]

    for idx, gen_key in enumerate(gen_order):
        ax = axes[idx]
        label = GENERATIONS[gen_key][1]

        for cond in CONDITIONS:
            if cond not in data[gen_key]:
                continue

            # Aggregate all intervals across seeds
            all_intervals = []
            for seed in SEEDS:
                if seed in data[gen_key][cond]:
                    all_intervals.extend(data[gen_key][cond][seed]["all_intervals"])

            if not all_intervals:
                continue

            intervals = np.array(all_intervals)
            color = CONDITION_COLORS[cond]

            # Histogram
            max_val = min(np.percentile(intervals, 99), 500)  # Cap at 99th percentile or 500
            bins = np.linspace(0, max_val, 30)
            ax.hist(intervals, bins=bins, alpha=0.5, color=color,
                   label=CONDITION_LABELS[cond], density=True)

        ax.set_xlabel("Steps Between Revisits", fontsize=12)
        ax.set_title(f"{label}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Density", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


def main():
    print("Loading sampling logs...")
    data = load_sampling_logs()

    print_summary(data)

    print("\nGenerating plots...")

    # Plot 1: Revisit histogram
    fig1 = plot_revisit_histogram(data)
    fig1.savefig("plots/revisit_histogram.png", dpi=150, bbox_inches='tight')
    fig1.savefig("plots/revisit_histogram.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_histogram.png/.pdf")

    # Plot 2: Gini coefficient comparison
    fig2 = plot_revisit_gini(data)
    fig2.savefig("plots/revisit_gini.png", dpi=150, bbox_inches='tight')
    fig2.savefig("plots/revisit_gini.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_gini.png/.pdf")

    # Plot 3: CV comparison
    fig3 = plot_revisit_cv(data)
    fig3.savefig("plots/revisit_cv.png", dpi=150, bbox_inches='tight')
    fig3.savefig("plots/revisit_cv.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_cv.png/.pdf")

    # Plot 4: Max revisit count
    fig4 = plot_max_revisit(data)
    fig4.savefig("plots/revisit_max.png", dpi=150, bbox_inches='tight')
    fig4.savefig("plots/revisit_max.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_max.png/.pdf")

    # Plot 5: v̂ vs revisit count (curriculum focus analysis)
    print("\nLoading sampling logs with v̂ data...")
    vhat_data = load_sampling_with_vhat()
    fig5 = plot_vhat_vs_revisit(vhat_data)
    fig5.savefig("plots/revisit_vhat_correlation.png", dpi=150, bbox_inches='tight')
    fig5.savefig("plots/revisit_vhat_correlation.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_vhat_correlation.png/.pdf")

    # Plot 6: Staleness over time
    print("\nLoading staleness data...")
    staleness_data = load_staleness_data()
    fig6 = plot_staleness_over_time(staleness_data)
    fig6.savefig("plots/revisit_staleness.png", dpi=150, bbox_inches='tight')
    fig6.savefig("plots/revisit_staleness.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_staleness.png/.pdf")

    # Plot 7: Staleness histogram
    fig7 = plot_staleness_histogram(staleness_data)
    fig7.savefig("plots/revisit_staleness_hist.png", dpi=150, bbox_inches='tight')
    fig7.savefig("plots/revisit_staleness_hist.pdf", bbox_inches='tight')
    print("Saved: plots/revisit_staleness_hist.png/.pdf")


if __name__ == "__main__":
    main()
