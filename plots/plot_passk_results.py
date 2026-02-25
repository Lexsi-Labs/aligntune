#!/usr/bin/env python3
"""Plot Pass@k and Maj@k results."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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
CONDITIONS = ["vanilla", "baseline_only", "curriculum_only", "full"]

def load_results():
    results = defaultdict(lambda: defaultdict(list))

    for f in sorted(Path("eval_results").glob("*_passk/summary.json")):
        run = f.parent.name.replace("_passk", "")
        with open(f) as fp:
            data = json.load(fp)

        parts = run.split('_')
        if run.startswith('bolt_g'):
            gen = parts[1]
            cond = '_'.join(parts[3:])
        else:
            gen = "g16"
            cond = '_'.join(parts[2:])

        key = (gen, cond)
        results[key]["mbpp"].append(data["results"]["mbpp"])
        results[key]["humaneval"].append(data["results"]["humaneval"])

    return results

def plot_passk_comparison(results, benchmark, metric_prefix="pass"):
    """Bar chart comparing pass@k or maj@k across conditions and G values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    ks = [1, 8, 32] if metric_prefix == "pass" else [8, 32]
    gen_order = ["g2", "g4", "g16"]

    for idx, gen in enumerate(gen_order):
        ax = axes[idx]
        x = np.arange(len(ks))
        width = 0.2

        for i, cond in enumerate(CONDITIONS):
            key = (gen, cond)
            if key not in results:
                continue

            data = results[key][benchmark]
            means = []
            stds = []

            for k in ks:
                metric = f"{metric_prefix}@{k}"
                vals = [d[metric] for d in data]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            offset = (i - 1.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
                   capsize=3)

        ax.set_xlabel(f"{metric_prefix.capitalize()}@k", fontsize=12)
        ax.set_title(f"{gen.upper()}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in ks])
        ax.grid(True, alpha=0.3, axis='y')

        if idx == 0:
            benchmark_name = "MBPP+" if benchmark == "mbpp" else "HumanEval+"
            ax.set_ylabel(f"{benchmark_name} {metric_prefix.capitalize()}@k", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig

def plot_g2_highlight(results):
    """Highlight G=2 results where PACE shines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (benchmark, title) in enumerate([("mbpp", "MBPP+"), ("humaneval", "HumanEval+")]):
        ax = axes[idx]

        metrics = ["pass@1", "pass@8", "pass@32", "maj@8", "maj@32"]
        x = np.arange(len(metrics))
        width = 0.2

        for i, cond in enumerate(CONDITIONS):
            key = ("g2", cond)
            data = results[key][benchmark]

            means = [np.mean([d[m] for d in data]) for m in metrics]
            stds = [np.std([d[m] for d in data]) for m in metrics]

            offset = (i - 1.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=CONDITION_LABELS[cond], color=CONDITION_COLORS[cond],
                   capsize=2)

        ax.set_ylabel(f"{title} Score", fontsize=12)
        ax.set_title(f"{title} @ G=2", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.4, 0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig

def main():
    print("Loading results...")
    results = load_results()

    print("Generating plots...")

    # MBPP+ Pass@k
    fig1 = plot_passk_comparison(results, "mbpp", "pass")
    fig1.savefig("plots/mbpp_passk.png", dpi=150, bbox_inches='tight')
    fig1.savefig("plots/mbpp_passk.pdf", bbox_inches='tight')
    print("Saved: plots/mbpp_passk.png/.pdf")

    # HumanEval+ Pass@k
    fig2 = plot_passk_comparison(results, "humaneval", "pass")
    fig2.savefig("plots/humaneval_passk.png", dpi=150, bbox_inches='tight')
    fig2.savefig("plots/humaneval_passk.pdf", bbox_inches='tight')
    print("Saved: plots/humaneval_passk.png/.pdf")

    # MBPP+ Maj@k
    fig3 = plot_passk_comparison(results, "mbpp", "maj")
    fig3.savefig("plots/mbpp_majk.png", dpi=150, bbox_inches='tight')
    fig3.savefig("plots/mbpp_majk.pdf", bbox_inches='tight')
    print("Saved: plots/mbpp_majk.png/.pdf")

    # HumanEval+ Maj@k
    fig4 = plot_passk_comparison(results, "humaneval", "maj")
    fig4.savefig("plots/humaneval_majk.png", dpi=150, bbox_inches='tight')
    fig4.savefig("plots/humaneval_majk.pdf", bbox_inches='tight')
    print("Saved: plots/humaneval_majk.png/.pdf")

    # G=2 highlight
    fig5 = plot_g2_highlight(results)
    fig5.savefig("plots/g2_pace_highlight.png", dpi=150, bbox_inches='tight')
    fig5.savefig("plots/g2_pace_highlight.pdf", bbox_inches='tight')
    print("Saved: plots/g2_pace_highlight.png/.pdf")

if __name__ == "__main__":
    main()
