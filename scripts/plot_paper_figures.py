#!/usr/bin/env python3
"""
Generate all figures for BOLT paper from ablation study.
Uses degenerate_log.jsonl and baseline_state_log.jsonl
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
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

def load_logs(base_dir: str, seed: int, max_steps: int = 250):
    """Load all log files, limiting to max_steps for fair comparison."""
    data = {}
    conditions = ['vanilla', 'baseline_only', 'curriculum_only', 'full']

    for cond in conditions:
        run_dir = Path(base_dir) / f"bolt_s{seed}_{cond}"
        data[cond] = {'degenerate': [], 'baseline_state': []}

        # Degenerate log
        dlog = run_dir / "degenerate_log.jsonl"
        if dlog.exists():
            with open(dlog) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry['step'] <= max_steps:
                        data[cond]['degenerate'].append(entry)

        # Baseline state log
        blog = run_dir / "baseline_state_log.jsonl"
        if blog.exists():
            with open(blog) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry['step'] <= max_steps:
                            data[cond]['baseline_state'].append(entry)
                    except:
                        pass

        # Try tensorboard for vanilla
        if cond == 'vanilla' and not data[cond]['degenerate']:
            try:
                from tensorboard.backend.event_processing import event_accumulator
                events = glob(str(run_dir / "runs/*/events*"))
                if events:
                    ea = event_accumulator.EventAccumulator(os.path.dirname(sorted(events)[-1]))
                    ea.Reload()
                    if 'train/frac_reward_zero_std' in ea.Tags()['scalars']:
                        frac = ea.Scalars('train/frac_reward_zero_std')
                        reward = ea.Scalars('train/reward') if 'train/reward' in ea.Tags()['scalars'] else []
                        for i, e in enumerate(frac):
                            if e.step > max_steps:
                                break
                            entry = {
                                'step': e.step,
                                'degenerate_rate': e.value,
                                'cumulative_degenerate_rate': np.mean([f.value for f in frac[:i+1]]),
                                'reward_mean': reward[i].value if i < len(reward) else 0,
                                'success_rate': 0,
                                'advantage_std': 1.0,
                                'advantage_nonzero_rate': 1.0,
                            }
                            data[cond]['degenerate'].append(entry)
            except Exception as ex:
                print(f"Warning: Could not load tensorboard for vanilla: {ex}")

        print(f"{cond}: {len(data[cond]['degenerate'])} degenerate entries, "
              f"{len(data[cond]['baseline_state'])} baseline entries")

    return data

def smooth(y, window=5):
    if len(y) < window:
        return np.array(y)
    return np.convolve(y, np.ones(window)/window, mode='valid')

def plot_figure1_degenerate(data, output_dir):
    """THE KEY FIGURE: Degenerate group rate over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        rates = [e['degenerate_rate'] * 100 for e in entries]

        # Raw data (faded)
        ax.plot(steps, rates, alpha=0.25, color=COLORS[cond], linewidth=1)

        # Smoothed
        if len(rates) >= 5:
            sr = smooth(np.array(rates), 5)
            ss = steps[2:2+len(sr)]
            ax.plot(ss, sr, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2.5, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Degenerate Group Rate (%)')
    ax.set_title('Degenerate Groups Over Training\n(Groups where all K samples have identical binary rewards)')
    ax.legend(loc='best')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(ax.get_xlim()[1]*0.98, 52, 'random baseline', ha='right', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_degenerate_rate.png")
    plt.savefig(f"{output_dir}/fig1_degenerate_rate.pdf")
    print(f"Saved: fig1_degenerate_rate")
    plt.close()

def plot_figure2_curriculum_focus(data, output_dir):
    """Where is training focused? v_hat distribution over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: pct_learning_edge over time
    ax = axes[0]
    for cond in ['curriculum_only', 'full']:
        entries = data[cond]['baseline_state']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        pcts = [e.get('curriculum', {}).get('pct_learning_edge', 0) * 100 for e in entries]

        if pcts:
            ax.plot(steps, pcts, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('% at Learning Edge')
    ax.set_title('Training Focus: % Problems at Learning Edge\n(0.3 < v̂ < 0.7)')
    ax.legend()
    ax.set_ylim(0, 100)

    # Right: sampling entropy over time
    ax = axes[1]
    for cond in ['curriculum_only', 'full']:
        entries = data[cond]['baseline_state']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        entropy = [e.get('curriculum', {}).get('sampling_entropy', 1.0) for e in entries]

        if entropy:
            ax.plot(steps, entropy, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Curriculum Diversity\n(1.0 = uniform, lower = more focused)')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_curriculum_focus.png")
    plt.savefig(f"{output_dir}/fig2_curriculum_focus.pdf")
    print(f"Saved: fig2_curriculum_focus")
    plt.close()

def plot_figure3_advantage_quality(data, output_dir):
    """Advantage statistics showing gradient quality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: advantage_std
    ax = axes[0]
    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        stds = [e.get('advantage_std', 1.0) for e in entries]
        ax.plot(steps, stds, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Advantage Std')
    ax.set_title('Advantage Standard Deviation\n(Higher = more gradient signal)')
    ax.legend()

    # Right: advantage_nonzero_rate
    ax = axes[1]
    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        rates = [e.get('advantage_nonzero_rate', 1.0) * 100 for e in entries]
        ax.plot(steps, rates, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Non-zero Rate (%)')
    ax.set_title('Rate of Non-zero Advantages\n(|A| > 0.01)')
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_advantage_quality.png")
    plt.savefig(f"{output_dir}/fig3_advantage_quality.pdf")
    print(f"Saved: fig3_advantage_quality")
    plt.close()

def plot_figure4_summary(data, output_dir):
    """Summary bar chart of final metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Collect final metrics
    metrics = {}
    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        metrics[cond] = {
            'degen': entries[-1].get('cumulative_degenerate_rate', entries[-1]['degenerate_rate']) * 100,
            'success': entries[-1].get('success_rate', 0) * 100,
            'reward': np.mean([e.get('reward_mean', 0) for e in entries]),
        }

    conds = list(metrics.keys())
    x = np.arange(len(conds))
    width = 0.6

    # Degenerate rate
    ax = axes[0]
    vals = [metrics[c]['degen'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Cumulative Degen Rate (%)')
    ax.set_title('Degenerate Groups (↓ better)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Success rate
    ax = axes[1]
    vals = [metrics[c]['success'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Training Success (↑ better)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Average reward
    ax = axes[2]
    vals = [metrics[c]['reward'] for c in conds]
    bars = ax.bar(x, vals, width, color=[COLORS[c] for c in conds])
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Reward (↑ better)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20, ha='right')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_summary.png")
    plt.savefig(f"{output_dir}/fig4_summary.pdf")
    print(f"Saved: fig4_summary")
    plt.close()

    return metrics

def plot_figure5_degenerate_breakdown(data, output_dir):
    """All-pass vs all-fail breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All-pass
    ax = axes[0]
    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        rates = [e.get('all_pass', 0) / e.get('batch_size', 1) * 100 for e in entries]

        if len(rates) >= 5:
            sr = smooth(np.array(rates), 5)
            ss = steps[2:2+len(sr)]
            ax.plot(ss, sr, color=COLORS[cond], linewidth=2, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('All-Pass Rate (%)')
    ax.set_title('Groups Where All K Samples Pass\n(Model mastered this problem)')
    ax.legend()
    ax.set_ylim(0, 100)

    # All-fail
    ax = axes[1]
    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        entries = data[cond]['degenerate']
        if not entries:
            continue

        steps = [e['step'] for e in entries]
        rates = [e.get('all_fail', 0) / e.get('batch_size', 1) * 100 for e in entries]

        if len(rates) >= 5:
            sr = smooth(np.array(rates), 5)
            ss = steps[2:2+len(sr)]
            ax.plot(ss, sr, color=COLORS[cond], linewidth=2, label=LABELS[cond])
        else:
            ax.plot(steps, rates, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel('Training Step')
    ax.set_ylabel('All-Fail Rate (%)')
    ax.set_title('Groups Where All K Samples Fail\n(Problem too hard)')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_degenerate_breakdown.png")
    plt.savefig(f"{output_dir}/fig5_degenerate_breakdown.pdf")
    print(f"Saved: fig5_degenerate_breakdown")
    plt.close()

def print_latex_table(metrics):
    """Print LaTeX table for paper."""
    print("\n" + "="*60)
    print("LaTeX Table")
    print("="*60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & Degen. Rate $\downarrow$ & Success $\uparrow$ & Reward $\uparrow$ \\")
    print(r"\midrule")

    for cond in ['vanilla', 'baseline_only', 'curriculum_only', 'full']:
        if cond not in metrics:
            continue
        m = metrics[cond]
        print(f"{LABELS[cond]} & {m['degen']:.1f}\\% & {m['success']:.1f}\\% & {m['reward']:.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Ablation study results on MBPP.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")
    print("="*60)

def main():
    base_dir = "output"
    output_dir = "plots/paper"
    seed = 80

    os.makedirs(output_dir, exist_ok=True)

    print("Loading logs...")
    data = load_logs(base_dir, seed)

    print("\nGenerating figures...")
    plot_figure1_degenerate(data, output_dir)
    plot_figure2_curriculum_focus(data, output_dir)
    plot_figure3_advantage_quality(data, output_dir)
    metrics = plot_figure4_summary(data, output_dir)
    plot_figure5_degenerate_breakdown(data, output_dir)

    print_latex_table(metrics)

    print(f"\nAll figures saved to: {output_dir}/")

if __name__ == "__main__":
    main()
