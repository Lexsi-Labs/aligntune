#!/usr/bin/env python3
"""
Extended GSM8k evaluation with pass@k and maj@k metrics.
Uses vLLM for fast inference with multiple samples per problem.
"""

import argparse
import json
import os
import re
import math
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from transformers import AutoTokenizer


def extract_answer(text: str) -> str:
    """Extract numeric answer from model output."""
    # Try #### format first
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in text
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    return nums[-1].replace(",", "") if nums else ""


def parse_gold_answer(answer: str) -> str:
    """Parse gold answer from GSM8k format."""
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer.replace(",", ""))
    return nums[-1].replace(",", "") if nums else ""


def numeric_equal(a: str, b: str, rtol=1e-4) -> bool:
    """Check if two numeric strings are approximately equal."""
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=rtol)
    except:
        return a.strip() == b.strip()


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def majority_vote(answers: list) -> str:
    """Return most common answer."""
    valid = [a for a in answers if a]
    if not valid:
        return ""
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load vLLM with LoRA
    print(f"Loading base model with vLLM...")
    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
    )

    lora_request = LoRARequest(
        lora_name=Path(args.checkpoint).name,
        lora_int_id=1,
        lora_path=args.checkpoint,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        n=args.num_samples,  # Generate multiple samples per prompt
    )

    # Load GSM8k test set
    print("Loading GSM8k test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} problems")

    # Create prompts
    system = "You are a careful math tutor. Solve step by step and provide the final numeric answer after ####."
    prompts = []
    gold_answers = []

    for item in dataset:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Problem:\n{item['question']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        gold_answers.append(parse_gold_answer(item['answer']))

    # Generate
    print(f"Generating {args.num_samples} samples per problem...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Evaluate
    results = []
    for idx, (output, gold) in enumerate(tqdm(zip(outputs, gold_answers), total=len(outputs), desc="Evaluating")):
        sample_answers = []
        sample_correct = []

        for completion in output.outputs:
            pred = extract_answer(completion.text)
            correct = numeric_equal(pred, gold)
            sample_answers.append(pred)
            sample_correct.append(correct)

        n_correct = sum(sample_correct)
        n_total = len(sample_correct)

        # Majority vote
        maj_answer = majority_vote(sample_answers)
        maj_correct = numeric_equal(maj_answer, gold)

        results.append({
            "idx": idx,
            "gold": gold,
            "n_correct": n_correct,
            "n_total": n_total,
            "maj_answer": maj_answer,
            "maj_correct": maj_correct,
            "sample_answers": sample_answers[:5],  # Save first 5 for inspection
        })

    # Compute metrics
    n = args.num_samples

    # Pass@k
    pass_at_1 = np.mean([compute_pass_at_k(n, r["n_correct"], 1) for r in results])
    pass_at_8 = np.mean([compute_pass_at_k(n, r["n_correct"], 8) for r in results]) if n >= 8 else 0
    pass_at_16 = np.mean([compute_pass_at_k(n, r["n_correct"], 16) for r in results]) if n >= 16 else 0
    pass_at_32 = np.mean([compute_pass_at_k(n, r["n_correct"], 32) for r in results]) if n >= 32 else 0

    # Avg@k (average accuracy across k samples)
    avg_at_1 = np.mean([r["n_correct"] / r["n_total"] for r in results])
    avg_at_8 = np.mean([min(r["n_correct"], 8) / min(r["n_total"], 8) for r in results]) if n >= 8 else 0
    avg_at_16 = np.mean([min(r["n_correct"], 16) / min(r["n_total"], 16) for r in results]) if n >= 16 else 0
    avg_at_32 = np.mean([min(r["n_correct"], 32) / min(r["n_total"], 32) for r in results]) if n >= 32 else 0

    # Maj@k (majority vote accuracy using k samples)
    def maj_at_k(results, k):
        correct = 0
        for r in results:
            answers_k = r["sample_answers"][:k] if len(r["sample_answers"]) >= k else r["sample_answers"]
            maj = majority_vote(answers_k)
            if numeric_equal(maj, r["gold"]):
                correct += 1
        return correct / len(results)

    maj_at_8 = maj_at_k(results, 8) if n >= 8 else 0
    maj_at_16 = maj_at_k(results, 16) if n >= 16 else 0
    maj_at_32 = maj_at_k(results, 32) if n >= 32 else 0

    # Wait, I stored only first 5 answers. Let me fix this by recomputing from the full data
    # Actually for maj@k we need all answers. Let me store them all.

    # Recompute with full answers
    for idx, (output, gold) in enumerate(zip(outputs, gold_answers)):
        all_answers = [extract_answer(c.text) for c in output.outputs]
        results[idx]["all_answers"] = all_answers

    def maj_at_k_full(results, k):
        correct = 0
        for r in results:
            answers_k = r["all_answers"][:k]
            maj = majority_vote(answers_k)
            if numeric_equal(maj, r["gold"]):
                correct += 1
        return correct / len(results)

    maj_at_8 = maj_at_k_full(results, 8) if n >= 8 else 0
    maj_at_16 = maj_at_k_full(results, 16) if n >= 16 else 0
    maj_at_32 = maj_at_k_full(results, 32) if n >= 32 else 0

    # Summary
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "total_problems": len(results),
        "pass@1": pass_at_1,
        "pass@8": pass_at_8,
        "pass@16": pass_at_16,
        "pass@32": pass_at_32,
        "avg@1": avg_at_1,
        "avg@8": avg_at_8,
        "avg@16": avg_at_16,
        "avg@32": avg_at_32,
        "maj@8": maj_at_8,
        "maj@16": maj_at_16,
        "maj@32": maj_at_32,
    }

    # Print results
    print("\n" + "=" * 60)
    print("EXTENDED EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples per problem: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print("-" * 60)
    print(f"Pass@1:  {pass_at_1:.4f}")
    print(f"Pass@8:  {pass_at_8:.4f}  |  Avg@8:  {avg_at_8:.4f}  |  Maj@8:  {maj_at_8:.4f}")
    print(f"Pass@16: {pass_at_16:.4f}  |  Avg@16: {avg_at_16:.4f}  |  Maj@16: {maj_at_16:.4f}")
    print(f"Pass@32: {pass_at_32:.4f}  |  Avg@32: {avg_at_32:.4f}  |  Maj@32: {maj_at_32:.4f}")
    print("=" * 60)

    # Save
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results (without all_answers to save space)
    for r in results:
        del r["all_answers"]
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
