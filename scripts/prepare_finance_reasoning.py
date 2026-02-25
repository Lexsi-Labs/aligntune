"""Prepare FinanceReasoning dataset for evaluation and training.

Downloads from BUPT-Reasoning-Lab/FinanceReasoning, bypassing the broken
load_dataset (schema mismatch in financial_documents.json).

Usage:
    python scripts/prepare_finance_reasoning.py [--levels medium] [--output_dir DIR]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download


def render_context(context) -> str:
    """Render the context field (dict or string) as readable text."""
    if not context:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        # Context is a nested dict representing a table/data
        # e.g. {"GBP": {"FY 2019": 91.6, "FY 2018": 87.1}, ...}
        lines = []
        for key, val in context.items():
            if isinstance(val, dict):
                row_parts = [f"{k}: {v}" for k, v in val.items()]
                lines.append(f"{key} — {', '.join(row_parts)}")
            else:
                lines.append(f"{key}: {val}")
        return "\n".join(lines)
    return str(context)


def process_rows(raw_data: list) -> list:
    """Convert raw JSON rows into the pipeline's expected format."""
    rows = []
    for r in raw_data:
        question_text = r.get("question", "")
        context = r.get("context", "")
        ground_truth = r.get("ground_truth")

        if ground_truth is None:
            continue

        # Build full prompt
        ctx_str = render_context(context)
        if ctx_str:
            full_question = f"{ctx_str}\n\nQuestion: {question_text}"
        else:
            full_question = question_text

        # Clean answer
        answer_str = str(ground_truth)
        try:
            float(answer_str)
        except ValueError:
            continue

        rows.append({
            "question": full_question,
            "answer": answer_str,
            "python_solution": r.get("python_solution", ""),
            "difficulty": r.get("difficulty", 0.0),
            "level": r.get("level", ""),
            "source": r.get("source", ""),
            "question_id": r.get("question_id", ""),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare FinanceReasoning dataset")
    parser.add_argument("--output_dir", type=str, default="data/finance_reasoning",
                        help="Output directory for the processed dataset")
    parser.add_argument("--levels", type=str, nargs="+", default=["medium"],
                        help="Difficulty levels to include (easy, medium, hard)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Filter out prompts longer than this many tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading FinanceReasoning dataset...")

    # Load raw JSON files directly (bypasses broken load_dataset)
    level_files = {
        "easy": "FinanceReasoning/easy.json",
        "medium": "FinanceReasoning/medium.json",
        "hard": "FinanceReasoning/hard.json",
    }

    all_rows = []
    for level in args.levels:
        if level not in level_files:
            print(f"  Warning: unknown level '{level}', skipping")
            continue
        try:
            path = hf_hub_download(
                repo_id="BUPT-Reasoning-Lab/FinanceReasoning",
                filename=level_files[level],
                repo_type="dataset",
            )
            with open(path) as f:
                raw_data = json.load(f)
            rows = process_rows(raw_data)
            all_rows.extend(rows)
            print(f"  Loaded {level}: {len(rows)} examples")
        except Exception as e:
            print(f"  Warning: Could not load {level}: {e}")

    print(f"Total: {len(all_rows)} examples")

    # Token length filtering
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    before = len(all_rows)
    all_rows = [r for r in all_rows if len(tok.encode(r["question"])) <= args.max_tokens]
    print(f"Filtered to <= {args.max_tokens} tokens: {len(all_rows)} ({before - len(all_rows)} removed)")

    # Save as a single test split (this is an eval dataset)
    dataset = Dataset.from_list(all_rows)
    dataset_dict = DatasetDict({"test": dataset})

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"\nSaved to {output_path}/")

    # Stats
    import numpy as np
    lengths = [len(tok.encode(r["question"])) for r in all_rows]
    lengths = np.array(lengths)
    print(f"\ntest split ({len(all_rows)} examples):")
    print(f"  Token lengths — min: {lengths.min()}, median: {np.median(lengths):.0f}, "
          f"mean: {lengths.mean():.0f}, max: {lengths.max()}")

    # Samples
    print(f"\n--- Sample Examples ---")
    for i in range(min(3, len(all_rows))):
        r = all_rows[i]
        print(f"\nExample {i+1}:")
        print(f"  Question: {r['question'][:300]}")
        print(f"  Answer: {r['answer']}")
        print(f"  Level: {r['level']}, Difficulty: {r['difficulty']:.2f}")


if __name__ == "__main__":
    main()
