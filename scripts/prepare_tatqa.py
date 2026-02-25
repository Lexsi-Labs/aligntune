"""Prepare TAT-QA arithmetic subset for counterfactual GRPO training.

Converts the nested TAT-QA dataset into a flat HuggingFace dataset
with `question` and `answer` fields matching the pipeline's expected format.

Usage:
    python scripts/prepare_tatqa.py [--max_samples N] [--output_dir DIR]
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download


def render_table(table_data: list) -> str:
    """Render a 2D table as a pipe-separated markdown table."""
    if not table_data:
        return ""

    # Compute column widths
    n_cols = max(len(row) for row in table_data)
    col_widths = [0] * n_cols
    for row in table_data:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(str(cell)))

    lines = []
    for i, row in enumerate(table_data):
        cells = [str(cell).ljust(col_widths[j]) for j, cell in enumerate(row)]
        # Pad if row has fewer columns
        while len(cells) < n_cols:
            cells.append(" " * col_widths[len(cells)])
        lines.append("| " + " | ".join(cells) + " |")
        # Add separator after header
        if i == 0:
            sep = ["-" * w for w in col_widths]
            lines.append("| " + " | ".join(sep) + " |")

    return "\n".join(lines)


def build_context(table_dict: dict, paragraphs: list) -> str:
    """Combine table and paragraphs into a single context string."""
    parts = []

    # Paragraphs (sorted by order)
    sorted_paras = sorted(paragraphs, key=lambda p: p.get("order", 0))
    para_texts = [p["text"] for p in sorted_paras if p.get("text", "").strip()]
    if para_texts:
        parts.append("\n".join(para_texts))

    # Table
    table_rows = table_dict.get("table", [])
    if table_rows:
        parts.append(render_table(table_rows))

    return "\n\n".join(parts)


def process_tatqa(raw_data: list) -> list:
    """Flatten nested TAT-QA docs into individual arithmetic QA rows."""
    rows = []

    for doc in raw_data:
        table_dict = doc.get("table", {})
        paragraphs = doc.get("paragraphs", [])
        questions = doc.get("questions", [])

        context = build_context(table_dict, paragraphs)

        for q in questions:
            if q.get("answer_type") != "arithmetic":
                continue

            answer_raw = q["answer"]
            # Handle list answers (rare for arithmetic, but be safe)
            if isinstance(answer_raw, list):
                if len(answer_raw) == 1:
                    answer_raw = answer_raw[0]
                else:
                    continue  # Skip multi-value arithmetic answers

            # Clean the numeric answer
            answer_str = str(answer_raw).replace(",", "").strip()

            # Verify it's actually numeric
            try:
                float(answer_str)
            except ValueError:
                continue

            question_text = q["question"]
            full_question = f"{context}\n\nQuestion: {question_text}"

            rows.append({
                "question": full_question,
                "answer": answer_str,
                "derivation": q.get("derivation", ""),
                "scale": q.get("scale", ""),
                "answer_from": q.get("answer_from", ""),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare TAT-QA for counterfactual GRPO")
    parser.add_argument("--output_dir", type=str, default="data/tatqa_arithmetic",
                        help="Output directory for the processed dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max train samples to keep (randomly sampled)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Filter out prompts longer than this many tokens")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    print("Loading TAT-QA dataset...")

    # Load raw JSON files from HuggingFace (avoids Arrow mixed-type issues)
    splits_raw = {}
    split_files = {
        "train": "tatqa_dataset_train.json",
        "test": "tatqa_dataset_dev.json",
    }
    for split_name, split_file in split_files.items():
        try:
            path = hf_hub_download(
                repo_id="next-tat/TAT-QA",
                filename=split_file,
                repo_type="dataset",
            )
            with open(path) as f:
                splits_raw[split_name] = json.load(f)
            print(f"  Loaded {split_file} -> {split_name}: {len(splits_raw[split_name])} documents")
        except Exception as e:
            print(f"  Warning: Could not load {split_file}: {e}")

    # Process each split independently
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    split_datasets = {}
    for split_name, raw_data in splits_raw.items():
        rows = process_tatqa(raw_data)
        print(f"\n{split_name}: {len(rows)} arithmetic questions extracted")

        # Filter out prompts > max_tokens
        before = len(rows)
        rows = [r for r in rows if len(tok.encode(r["question"])) <= args.max_tokens]
        print(f"  Filtered to <= {args.max_tokens} tokens: {len(rows)} ({before - len(rows)} removed)")

        # Optionally limit train samples
        if split_name == "train" and args.max_samples and len(rows) > args.max_samples:
            import random
            random.seed(args.seed)
            rows = random.sample(rows, args.max_samples)
            print(f"  Sampled down to {args.max_samples} examples")

        split_datasets[split_name] = Dataset.from_list(rows)

    # Save as DatasetDict with train/test splits
    dataset_dict = DatasetDict(split_datasets)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"\nSaved to {output_path}/")

    # Print stats per split
    import numpy as np
    for split_name, ds in dataset_dict.items():
        lengths = [len(tok.encode(row["question"])) for row in ds]
        lengths = np.array(lengths)
        print(f"\n{split_name} split ({len(ds)} examples):")
        print(f"  Token lengths — min: {lengths.min()}, median: {np.median(lengths):.0f}, "
              f"mean: {lengths.mean():.0f}, max: {lengths.max()}")

    # Print sample examples from train
    train_ds = dataset_dict["train"]
    print(f"\n--- Sample Train Examples ---")
    for i in range(min(3, len(train_ds))):
        row = train_ds[i]
        q = row["question"]
        print(f"\nExample {i+1}:")
        print(f"  Question (last 300 chars): ...{q[-300:]}")
        print(f"  Answer: {row['answer']}")
        print(f"  Derivation: {row['derivation']}")
        print(f"  Scale: {row['scale']}")


if __name__ == "__main__":
    main()
