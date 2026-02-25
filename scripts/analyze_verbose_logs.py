#!/usr/bin/env python3
"""
Analyze verbose training logs to find interesting paper examples.

Usage:
    python scripts/analyze_verbose_logs.py verbose_logs/run_xxx.jsonl

This script will:
1. Load all examples from the JSONL log
2. Compute statistics across examples
3. Identify interesting cases:
   - High-contrast importance (some spans much more important than others)
   - Correct vs incorrect examples
   - Different span types (arithmetic vs sentence)
4. Output a summary and top N examples for the paper
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_verbose_log(log_path: str) -> List[Dict]:
    """Load examples from a verbose log JSONL file."""
    examples = []
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "example":
                examples.append(entry)
    return examples


def compute_importance_stats(example: Dict) -> Dict:
    """Compute statistics about importance distribution for an example."""
    importance = np.array(example.get("importance_normalized", []))
    weights = np.array(example.get("weights_final", []))

    if len(importance) == 0:
        return {}

    # Filter to non-zero
    nonzero_imp = importance[importance > 0.1]
    nonzero_weights = weights[weights > 0.1]

    stats = {
        "importance_mean": float(importance.mean()),
        "importance_std": float(importance.std()),
        "importance_max": float(importance.max()),
        "importance_min": float(importance[importance > 0].min()) if any(importance > 0) else 0,
        "importance_spread": float(importance.max() - importance.min()),
        "weight_mean": float(weights.mean()) if len(weights) > 0 else 1.0,
        "weight_std": float(weights.std()) if len(weights) > 0 else 0.0,
        "weight_spread": float(weights.max() - weights.min()) if len(weights) > 0 else 0.0,
        "num_spans": len(example.get("spans", [])),
        "num_tokens": len(example.get("tokens", [])),
    }

    # Compute span importance concentration
    spans = example.get("spans", [])
    if spans:
        span_drops = [s.get("importance_drop", 0) for s in spans]
        stats["max_span_drop"] = max(span_drops) if span_drops else 0
        stats["span_drop_variance"] = float(np.var(span_drops)) if len(span_drops) > 1 else 0

    return stats


def score_example_interest(example: Dict, stats: Dict) -> float:
    """Score how interesting an example is for the paper."""
    score = 0.0

    # High importance spread is interesting
    score += stats.get("importance_spread", 0) * 2.0

    # High weight spread is interesting
    score += stats.get("weight_spread", 0) * 1.5

    # Variance in span drops is interesting (shows differentiation)
    score += stats.get("span_drop_variance", 0) * 0.5

    # Examples with more spans are more illustrative
    score += min(stats.get("num_spans", 0), 10) * 0.1

    # Prefer longer completions (more to analyze)
    score += min(stats.get("num_tokens", 0), 200) * 0.01

    # Bonus for correct examples (cleaner to explain)
    if example.get("is_correct", False):
        score += 0.5

    return score


def format_example_for_paper(example: Dict, rank: int) -> str:
    """Format an example nicely for paper inclusion."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"EXAMPLE #{rank} (Step {example.get('step', '?')}, {'CORRECT' if example.get('is_correct') else 'INCORRECT'})")
    lines.append(f"{'='*80}")

    lines.append(f"\n--- PROMPT ---")
    prompt = example.get("prompt_text", "")[:500]
    lines.append(prompt + ("..." if len(example.get("prompt_text", "")) > 500 else ""))

    lines.append(f"\n--- COMPLETION ---")
    completion = example.get("completion_text", "")
    lines.append(completion[:1000] + ("..." if len(completion) > 1000 else ""))

    lines.append(f"\n--- PREDICTED ANSWER: {example.get('predicted_answer', 'N/A')} ---")
    lines.append(f"--- ADVANTAGE: {example.get('advantage', 0):.4f} ---")

    lines.append(f"\n--- DETECTED SPANS ({len(example.get('spans', []))}) ---")
    for i, span in enumerate(example.get("spans", [])[:10]):
        drop = span.get("importance_drop", 0)
        text = span.get("text", "")[:60]
        span_type = span.get("span_type", "?")
        lines.append(f"  [{i+1}] {span_type:10} | Drop: {drop:+.4f} | \"{text}\"")

    lines.append(f"\n--- TOP 10 WEIGHTED TOKENS ---")
    for t in example.get("top_weighted_tokens", [])[:10]:
        lines.append(f"  Pos {t['pos']:3d} | Weight: {t['weight']:.3f} | Imp: {t['importance']:.3f} | '{t['token']}'")

    lines.append(f"\n--- BOTTOM 10 WEIGHTED TOKENS ---")
    for t in example.get("bottom_weighted_tokens", [])[:10]:
        lines.append(f"  Pos {t['pos']:3d} | Weight: {t['weight']:.3f} | Imp: {t['importance']:.3f} | '{t['token']}'")

    stats = example.get("weight_stats", {})
    lines.append(f"\n--- WEIGHT STATS ---")
    lines.append(f"  Mean: {stats.get('mean', 1):.4f}, Std: {stats.get('std', 0):.4f}, Min: {stats.get('min', 1):.4f}, Max: {stats.get('max', 1):.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze verbose training logs for paper examples")
    parser.add_argument("log_path", help="Path to verbose log JSONL file")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top examples to show")
    parser.add_argument("--output", "-o", help="Output file for formatted examples")
    parser.add_argument("--correct-only", action="store_true", help="Only show correct examples")
    parser.add_argument("--incorrect-only", action="store_true", help="Only show incorrect examples")
    args = parser.parse_args()

    print(f"Loading examples from {args.log_path}...")
    examples = load_verbose_log(args.log_path)
    print(f"Loaded {len(examples)} examples")

    if len(examples) == 0:
        print("No examples found!")
        return

    # Filter if requested
    if args.correct_only:
        examples = [e for e in examples if e.get("is_correct", False)]
        print(f"Filtered to {len(examples)} correct examples")
    elif args.incorrect_only:
        examples = [e for e in examples if not e.get("is_correct", False)]
        print(f"Filtered to {len(examples)} incorrect examples")

    # Compute stats and score each example
    scored_examples = []
    for ex in examples:
        stats = compute_importance_stats(ex)
        score = score_example_interest(ex, stats)
        scored_examples.append((score, ex, stats))

    # Sort by score (descending)
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    # Print aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")

    all_stats = [s for _, _, s in scored_examples]

    correct_count = sum(1 for _, e, _ in scored_examples if e.get("is_correct", False))
    print(f"Correct examples: {correct_count}/{len(examples)} ({100*correct_count/len(examples):.1f}%)")

    avg_spans = np.mean([s.get("num_spans", 0) for s in all_stats])
    print(f"Average spans per example: {avg_spans:.1f}")

    avg_weight_spread = np.mean([s.get("weight_spread", 0) for s in all_stats])
    print(f"Average weight spread: {avg_weight_spread:.3f}")

    avg_importance_spread = np.mean([s.get("importance_spread", 0) for s in all_stats])
    print(f"Average importance spread: {avg_importance_spread:.3f}")

    # Print top examples
    print(f"\n{'='*80}")
    print(f"TOP {args.top_n} MOST INTERESTING EXAMPLES")
    print(f"{'='*80}")

    output_lines = []
    for rank, (score, ex, stats) in enumerate(scored_examples[:args.top_n], 1):
        formatted = format_example_for_paper(ex, rank)
        print(formatted)
        output_lines.append(formatted)
        print(f"\n[Interest Score: {score:.3f}]")
        output_lines.append(f"\n[Interest Score: {score:.3f}]")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Verbose Log Analysis: {args.log_path}\n")
            f.write(f"Total examples: {len(examples)}\n")
            f.write(f"Correct: {correct_count}/{len(examples)}\n\n")
            f.write("\n".join(output_lines))
        print(f"\nSaved formatted examples to {args.output}")

    # Save all examples as JSON for further analysis
    json_output = args.output.replace('.txt', '.json') if args.output else args.log_path.replace('.jsonl', '_analysis.json')
    with open(json_output, 'w') as f:
        analysis = {
            "source": args.log_path,
            "total_examples": len(examples),
            "correct_count": correct_count,
            "aggregate_stats": {
                "avg_spans": avg_spans,
                "avg_weight_spread": avg_weight_spread,
                "avg_importance_spread": avg_importance_spread,
            },
            "top_examples": [
                {"rank": i+1, "score": score, "example": ex, "stats": stats}
                for i, (score, ex, stats) in enumerate(scored_examples[:args.top_n])
            ]
        }
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved full analysis to {json_output}")


if __name__ == "__main__":
    main()
