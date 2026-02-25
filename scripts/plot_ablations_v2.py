#!/usr/bin/env python3
"""
Plot ablation study results for BOLT paper - v2 with tensorboard data.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("Warning: tensorboard not available")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

COLORS = {
    'vanilla': '#d62728',
    'baseline_only': '#ff7f0e',
    'curriculum_only': '#2ca02c',
    'full': '#1f77b4',
}

LABELS = {
    'vanilla': 'Vanilla GRPO',
    'baseline_only': 'Baseline Only',
    'curriculum_only': 'Curriculum Only',
    'full': 'Full BOLT',
}

def load_tensorboard_scalar(run_dir: str, tag: str):
    """Load scalar data from tensorboard logs."""
    if not HAS_TB:
        return [], []

    # Find the latest events file
    events_files = glob(f"{run_dir}/runs/*/events*")
    if not events_files:
        return [], []

    # Use the most recent
    events_file = sorted(events_files)[-1]
    event_dir = os.path.dirname(events_file)

    ea = event_accumulator.EventAccumulator(event_dir)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def load_degenerate_log(run_dir: str):
    """Load degenerate log if exists."""
    log_path = Path(run_dir) / "degenerate_log.jsonl"
    if not log_path.exists():
        return None

    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

def smooth(y, window=5):
    if len(y) < window:
        return np.array(y)
    return np.convolve(y, np.ones(window)/window, mode='valid')

def main():
    base_dir = "output"
    output_dir = "plots/ablations"
    seed = 80
    os.makedirs(output_dir, exist_ok=True)

    conditions = ['vanilla', 'baseline_only', 'curriculum_only', 'full']

    # Collect data
    all_data = {}
    for cond in conditions:
        run_dir = f"{base_dir}/bolt_s{seed}_{cond}"
        if not os.path.exists(run_dir):
            print(f"Missing: {run_dir}")
            continue

        all_data[cond] = {
            'degenerate_log': load_degenerate_log(run_dir),
        }

        # Load tensorboard metrics
        for tag in ['train/reward', 'train/frac_reward_zero_std', 'train/kl', 'train/loss']:
            steps, values = load_tensorboard_scalar(run_dir, tag)
            tag_short = tag.split('/')[-1]
            all_data[cond][f'tb_{tag_short}_steps'] = steps
            all_data[cond][f'tb_{tag_short}_values'] = values
            if values:
                print(f"  {cond}/{tag}: {len(values)} points")

    # =====================
    # PLOT 1: Degenerate Rate (THE KEY FIGURE)
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        if cond not in all_data:
            continue

        # Use degenerate_log if available, else tensorboard frac_reward_zero_std
        dlog = all_data[cond].get('degenerate_log')
        if dlog:
            steps = [e['step'] for e in dlog]
            rates = [e['degenerate_rate'] * 100 for e in dlog]
        else:
            steps = all_data[cond].get('tb_frac_reward_zero_std_steps', [])
            rates = [v * 100 for v in all_data[cond].get('tb_frac_reward_zero_std_values', [])]

        if not steps:
            continue

        # Plot raw with low alpha
        ax.plot(steps, rates, alpha=0.3, color=COLORS[cond], linewidth=1)

        # Smoothed
        if len(rates) > 5:
            sr = smooth(np.array(rates), 5)
            ss = steps[2:2+len(sr)]
            ax.plot(ss, sr, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Degenerate Group Rate (%)')
    ax.set_title('Degenerate Groups Over Training\n(Groups where all samples have identical rewards)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_degenerate_rate.png")
    plt.savefig(f"{output_dir}/fig1_degenerate_rate.pdf")
    print(f"Saved: fig1_degenerate_rate.png")
    plt.close()

    # =====================
    # PLOT 2: Training Reward
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        if cond not in all_data:
            continue

        steps = all_data[cond].get('tb_reward_steps', [])
        values = all_data[cond].get('tb_reward_values', [])

        if not steps:
            # Try degenerate log
            dlog = all_data[cond].get('degenerate_log')
            if dlog:
                steps = [e['step'] for e in dlog]
                values = [e['reward_mean'] for e in dlog]

        if not steps:
            continue

        if len(values) > 5:
            sv = smooth(np.array(values), 5)
            ss = steps[2:2+len(sv)]
            ax.plot(ss, sv, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, values, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Reward Over Time')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_reward.png")
    plt.savefig(f"{output_dir}/fig2_reward.pdf")
    print(f"Saved: fig2_reward.png")
    plt.close()

    # =====================
    # PLOT 3: Success Rate (from degenerate logs)
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        if cond not in all_data:
            continue

        dlog = all_data[cond].get('degenerate_log')
        if not dlog:
            continue

        steps = [e['step'] for e in dlog]
        rates = [e['success_rate'] * 100 for e in dlog]

        if len(rates) > 5:
            sr = smooth(np.array(rates), 5)
            ss = steps[2:2+len(sr)]
            ax.plot(ss, sr, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (Reward > 0.5) Over Training')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_success_rate.png")
    plt.savefig(f"{output_dir}/fig3_success_rate.pdf")
    print(f"Saved: fig3_success_rate.png")
    plt.close()

    # =====================
    # PLOT 4: Summary Bar Chart
    # =====================
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Collect final metrics
    final_metrics = {}
    for cond in conditions:
        if cond not in all_data:
            continue

        dlog = all_data[cond].get('degenerate_log')
        if dlog:
            final_metrics[cond] = {
                'degen': dlog[-1]['cumulative_degenerate_rate'] * 100,
                'success': dlog[-1]['success_rate'] * 100,
                'reward': np.mean([e['reward_mean'] for e in dlog]),
            }
        else:
            # Use tensorboard
            tb_frac = all_data[cond].get('tb_frac_reward_zero_std_values', [])
            tb_reward = all_data[cond].get('tb_reward_values', [])
            if tb_frac and tb_reward:
                final_metrics[cond] = {
                    'degen': np.mean(tb_frac[-10:]) * 100 if len(tb_frac) > 10 else np.mean(tb_frac) * 100,
                    'success': 0,  # Not available from TB
                    'reward': np.mean(tb_reward[-10:]) if len(tb_reward) > 10 else np.mean(tb_reward),
                }

    conds = list(final_metrics.keys())
    x = np.arange(len(conds))
    width = 0.6

    # Degenerate rate
    ax = axes[0]
    vals = [final_metrics[c]['degen'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Cumulative Degenerate Rate (%)')
    ax.set_title('Final Degenerate Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Success rate
    ax = axes[1]
    vals = [final_metrics[c]['success'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Final Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Avg reward
    ax = axes[2]
    vals = [final_metrics[c]['reward'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Mean Reward')
    ax.set_title('Average Training Reward')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_summary.png")
    plt.savefig(f"{output_dir}/fig4_summary.pdf")
    print(f"Saved: fig4_summary.png")
    plt.close()

    # =====================
    # Print summary table
    # =====================
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Condition':<20} {'Degen%':<10} {'Success%':<10} {'Reward':<10}")
    print("-"*70)
    for cond, m in final_metrics.items():
        print(f"{LABELS[cond]:<20} {m['degen']:>8.1f}% {m['success']:>8.1f}% {m['reward']:>8.3f}")
    print("="*70)

    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
