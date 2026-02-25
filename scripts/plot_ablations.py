#!/usr/bin/env python3
"""
Plot ablation study results for BOLT paper.
Generates publication-quality figures comparing:
- Vanilla GRPO
- Baseline advantages only
- Curriculum only
- Full BOLT (curriculum + baseline)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style settings for paper
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme
COLORS = {
    'vanilla': '#d62728',      # Red
    'baseline_only': '#ff7f0e', # Orange
    'curriculum_only': '#2ca02c', # Green
    'full': '#1f77b4',         # Blue
}

LABELS = {
    'vanilla': 'Vanilla GRPO',
    'baseline_only': 'Baseline Only',
    'curriculum_only': 'Curriculum Only',
    'full': 'Full BOLT',
}

def load_degenerate_logs(base_dir: str, seed: int = 80) -> dict:
    """Load degenerate log data from all ablation runs."""
    data = {}
    conditions = ['vanilla', 'baseline_only', 'curriculum_only', 'full']

    for cond in conditions:
        log_path = Path(base_dir) / f"bolt_s{seed}_{cond}" / "degenerate_log.jsonl"
        if log_path.exists():
            entries = []
            with open(log_path) as f:
                for line in f:
                    entries.append(json.loads(line))
            data[cond] = entries
            print(f"Loaded {len(entries)} entries for {cond}")
        else:
            print(f"Warning: No log found for {cond} at {log_path}")

    return data

def load_baseline_state_logs(base_dir: str, seed: int = 80) -> dict:
    """Load baseline state logs for curriculum metrics."""
    data = {}
    conditions = ['baseline_only', 'curriculum_only', 'full']

    for cond in conditions:
        log_path = Path(base_dir) / f"bolt_s{seed}_{cond}" / "baseline_state_log.jsonl"
        if log_path.exists():
            entries = []
            with open(log_path) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
            data[cond] = entries
            print(f"Loaded {len(entries)} baseline state entries for {cond}")

    return data

def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

def plot_degenerate_rate(data: dict, output_dir: str):
    """Plot degenerate group rate over training - THE KEY FIGURE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['degenerate_rate'] * 100 for e in entries]

        # Plot raw data with low alpha
        ax.plot(steps, rates, alpha=0.3, color=COLORS[cond], linewidth=1)

        # Plot smoothed
        if len(rates) > 5:
            smooth_rates = smooth(np.array(rates), window=5)
            smooth_steps = steps[2:-2] if len(steps) > 4 else steps
            ax.plot(smooth_steps[:len(smooth_rates)], smooth_rates,
                   color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Degenerate Group Rate (%)')
    ax.set_title('Degenerate Groups Over Training')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/degenerate_rate.png")
    plt.savefig(f"{output_dir}/degenerate_rate.pdf")
    print(f"Saved: {output_dir}/degenerate_rate.png")
    plt.close()

def plot_cumulative_degenerate(data: dict, output_dir: str):
    """Plot cumulative degenerate rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['cumulative_degenerate_rate'] * 100 for e in entries]
        ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cumulative Degenerate Rate (%)')
    ax.set_title('Cumulative Degenerate Group Rate')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_degenerate.png")
    plt.savefig(f"{output_dir}/cumulative_degenerate.pdf")
    print(f"Saved: {output_dir}/cumulative_degenerate.png")
    plt.close()

def plot_success_rate(data: dict, output_dir: str):
    """Plot success rate (reward > 0.5) over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['success_rate'] * 100 for e in entries]

        # Smoothed
        if len(rates) > 5:
            smooth_rates = smooth(np.array(rates), window=5)
            smooth_steps = steps[2:-2]
            ax.plot(smooth_steps[:len(smooth_rates)], smooth_rates,
                   color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Training Success Rate (reward > 0.5)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate.png")
    plt.savefig(f"{output_dir}/success_rate.pdf")
    print(f"Saved: {output_dir}/success_rate.png")
    plt.close()

def plot_advantage_stats(data: dict, output_dir: str):
    """Plot advantage statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Advantage std
    ax = axes[0]
    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        stds = [e['advantage_std'] for e in entries]
        ax.plot(steps, stds, color=COLORS[cond], linewidth=2, label=LABELS[cond])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Advantage Std')
    ax.set_title('Advantage Standard Deviation')
    ax.legend()

    # Advantage nonzero rate
    ax = axes[1]
    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['advantage_nonzero_rate'] * 100 for e in entries]
        ax.plot(steps, rates, color=COLORS[cond], linewidth=2, label=LABELS[cond])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Non-zero Advantage Rate (%)')
    ax.set_title('Rate of Non-zero Advantages (|A| > 0.01)')
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/advantage_stats.png")
    plt.savefig(f"{output_dir}/advantage_stats.pdf")
    print(f"Saved: {output_dir}/advantage_stats.png")
    plt.close()

def plot_degenerate_breakdown(data: dict, output_dir: str):
    """Plot all-pass vs all-fail breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All-pass rate
    ax = axes[0]
    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['all_pass'] / e['batch_size'] * 100 if e['batch_size'] > 0 else 0 for e in entries]
        if len(rates) > 5:
            smooth_rates = smooth(np.array(rates), window=5)
            smooth_steps = steps[2:-2]
            ax.plot(smooth_steps[:len(smooth_rates)], smooth_rates,
                   color=COLORS[cond], linewidth=2, label=LABELS[cond])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('All-Pass Rate (%)')
    ax.set_title('Groups Where All Samples Pass')
    ax.legend()
    ax.set_ylim(0, 100)

    # All-fail rate
    ax = axes[1]
    for cond, entries in data.items():
        steps = [e['step'] for e in entries]
        rates = [e['all_fail'] / e['batch_size'] * 100 if e['batch_size'] > 0 else 0 for e in entries]
        if len(rates) > 5:
            smooth_rates = smooth(np.array(rates), window=5)
            smooth_steps = steps[2:-2]
            ax.plot(smooth_steps[:len(smooth_rates)], smooth_rates,
                   color=COLORS[cond], linewidth=2, label=LABELS[cond])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('All-Fail Rate (%)')
    ax.set_title('Groups Where All Samples Fail')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/degenerate_breakdown.png")
    plt.savefig(f"{output_dir}/degenerate_breakdown.pdf")
    print(f"Saved: {output_dir}/degenerate_breakdown.png")
    plt.close()

def plot_summary_bars(data: dict, output_dir: str):
    """Plot summary bar chart of final metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = list(data.keys())
    x = np.arange(len(conditions))
    width = 0.6

    # Final degenerate rate
    ax = axes[0]
    final_degen = [data[c][-1]['cumulative_degenerate_rate'] * 100 for c in conditions]
    bars = ax.bar(x, final_degen, width, color=[COLORS[c] for c in conditions])
    ax.set_ylabel('Cumulative Degenerate Rate (%)')
    ax.set_title('Final Degenerate Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], rotation=15, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, final_degen):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Final success rate
    ax = axes[1]
    final_success = [data[c][-1]['success_rate'] * 100 for c in conditions]
    bars = ax.bar(x, final_success, width, color=[COLORS[c] for c in conditions])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Final Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], rotation=15, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, final_success):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Average reward
    ax = axes[2]
    avg_reward = [np.mean([e['reward_mean'] for e in data[c]]) for c in conditions]
    bars = ax.bar(x, avg_reward, width, color=[COLORS[c] for c in conditions])
    ax.set_ylabel('Average Reward')
    ax.set_title('Mean Reward Over Training')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conditions], rotation=15, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, avg_reward):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_bars.png")
    plt.savefig(f"{output_dir}/summary_bars.pdf")
    print(f"Saved: {output_dir}/summary_bars.png")
    plt.close()

def print_summary_table(data: dict):
    """Print summary statistics table."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Condition':<20} {'Degen Rate':<12} {'Success':<12} {'Avg Reward':<12} {'Steps':<8}")
    print("-"*80)

    for cond, entries in data.items():
        final_degen = entries[-1]['cumulative_degenerate_rate'] * 100
        final_success = entries[-1]['success_rate'] * 100
        avg_reward = np.mean([e['reward_mean'] for e in entries])
        n_steps = len(entries)
        print(f"{LABELS[cond]:<20} {final_degen:>10.1f}% {final_success:>10.1f}% {avg_reward:>10.3f} {n_steps:>8}")

    print("="*80)

def main():
    base_dir = "output"
    output_dir = "plots/ablations"
    seed = 80

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading degenerate logs...")
    data = load_degenerate_logs(base_dir, seed)

    if not data:
        print("No data found!")
        return

    # Print summary
    print_summary_table(data)

    # Generate plots
    print("\nGenerating plots...")
    plot_degenerate_rate(data, output_dir)
    plot_cumulative_degenerate(data, output_dir)
    plot_success_rate(data, output_dir)
    plot_advantage_stats(data, output_dir)
    plot_degenerate_breakdown(data, output_dir)
    plot_summary_bars(data, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
