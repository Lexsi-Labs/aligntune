#!/usr/bin/env python3
"""
Evaluate GSM8K using vLLM with LoRA support.
Uses the same approach as eval_mbpp_plus.py but for math problems.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# Try vLLM first, fall back to transformers
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, using transformers (slower)")


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from model output."""
    # Try to find \boxed{...} first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try #### format (GSM8K standard)
    hash_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        return hash_match.group(1).strip()

    # Try "Final Answer: X" format
    final_match = re.search(r'[Ff]inal [Aa]nswer[:\s]+(.+?)(?:\n|$)', text)
    if final_match:
        return final_match.group(1).strip()

    # Try "The answer is X" format
    answer_match = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)', text)
    if answer_match:
        return answer_match.group(1).strip()

    # Last resort: find the last number in the text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove commas, $, %, spaces
    answer = answer.replace(',', '').replace('$', '').replace('%', '').strip()
    # Remove trailing period
    answer = answer.rstrip('.')
    # Try to convert to number for comparison
    try:
        return str(float(answer))
    except ValueError:
        return answer.lower()


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Direct string match
    if pred_norm == gold_norm:
        return True

    # Numeric comparison with tolerance
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        return abs(pred_num - gold_num) < 1e-6
    except ValueError:
        return False


def create_prompt(question: str, system_prompt: str = None) -> list:
    """Create chat messages for the model."""
    if system_prompt is None:
        system_prompt = "You are an expert mathematician. Solve the problem step by step and put your final answer in \\boxed{}."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


def run_eval_vllm(model_path: str, lora_path: str = None, output_dir: str = None,
                   temperature: float = 0.1, max_tokens: int = 1024,
                   limit: int = None) -> dict:
    """Run GSM8K evaluation using vLLM."""

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Evaluating {len(dataset)} problems")

    # Initialize vLLM
    print(f"Loading model: {model_path}")
    if lora_path:
        print(f"With LoRA adapter: {lora_path}")

    llm = LLM(
        model=model_path,
        enable_lora=lora_path is not None,
        max_lora_rank=64,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.95,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
        seed=42 if temperature == 0.0 else None,
    )

    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter", 1, lora_path)

    # Prepare prompts
    prompts = []
    for item in dataset:
        messages = create_prompt(item["question"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)

    # Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Evaluate
    correct = 0
    results = []

    for i, (item, output) in enumerate(zip(dataset, outputs)):
        response = output.outputs[0].text

        # Extract gold answer from GSM8K format (after ####)
        gold_full = item["answer"]
        gold_match = re.search(r'####\s*(.+?)$', gold_full)
        gold = gold_match.group(1).strip() if gold_match else gold_full

        predicted = extract_answer(response)
        is_correct = check_answer(predicted, gold)

        if is_correct:
            correct += 1

        results.append({
            "question": item["question"],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500]  # Truncate for storage
        })

    accuracy = correct / len(dataset)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": len(dataset),
                "model": model_path,
                "lora": lora_path,
            }, f, indent=2)

        with open(os.path.join(output_dir, "details.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {"accuracy": accuracy, "correct": correct, "total": len(dataset)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_results/gsm8k")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("ERROR: vLLM is required for this script")
        sys.exit(1)

    results = run_eval_vllm(
        model_path=args.model_path,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )

    print("\n" + "=" * 60)
    print("GSM8K EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print("=" * 60)

    if args.output_dir:
        print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()