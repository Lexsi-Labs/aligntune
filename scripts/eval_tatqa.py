#!/usr/bin/env python3
"""
Evaluate TAT-QA arithmetic subset using vLLM or transformers with LoRA support.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def extract_answer(text: str) -> str:
    """Extract the final numerical answer from model output."""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    hash_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        return hash_match.group(1).strip()

    final_match = re.search(r'[Ff]inal [Aa]nswer[:\s]+(.+?)(?:\n|$)', text)
    if final_match:
        return final_match.group(1).strip()

    answer_match = re.search(r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)', text)
    if answer_match:
        return answer_match.group(1).strip()

    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.replace(',', '').replace('$', '').replace('%', '').strip()
    answer = answer.rstrip('.')
    try:
        return str(float(answer))
    except ValueError:
        return answer.lower()


def check_answer(predicted: str, gold: str, rtol: float = 1e-3) -> bool:
    """Check if predicted answer matches gold answer with tolerance."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        if gold_num == 0:
            return abs(pred_num) < 1e-6
        return math.isclose(pred_num, gold_num, rel_tol=rtol, abs_tol=1e-6)
    except ValueError:
        return False


def create_prompt(question: str, system_prompt: str = None) -> list:
    """Create chat messages for the model."""
    if system_prompt is None:
        system_prompt = (
            "You are an expert financial analyst. Solve the problem step by step, "
            "showing your work clearly. Put your final numeric answer in \\boxed{}."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


def load_tatqa_test(data_dir: str):
    """Load TAT-QA test split."""
    test_path = os.path.join(data_dir, "test")
    if os.path.isdir(test_path):
        return load_from_disk(test_path)
    ds = load_from_disk(data_dir)
    if hasattr(ds, 'keys') and 'test' in ds:
        return ds["test"]
    return ds


def evaluate_results(dataset, responses):
    """Score predictions against gold answers."""
    correct = 0
    results = []
    scale_stats = {}

    for i, (item, response) in enumerate(zip(dataset, responses)):
        gold = item["answer"]
        predicted = extract_answer(response)
        is_correct = check_answer(predicted, gold)
        scale = item.get("scale", "")

        if is_correct:
            correct += 1

        if scale not in scale_stats:
            scale_stats[scale] = {"correct": 0, "total": 0}
        scale_stats[scale]["total"] += 1
        if is_correct:
            scale_stats[scale]["correct"] += 1

        results.append({
            "question": item["question"][:300],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "scale": scale,
            "derivation": item.get("derivation", ""),
            "answer_from": item.get("answer_from", ""),
            "response": response[:500],
        })

    accuracy = correct / len(dataset) if dataset else 0
    return accuracy, correct, results, scale_stats


def run_eval_vllm(model_path: str, lora_path: str = None, output_dir: str = None,
                   data_dir: str = "data/tatqa_arithmetic",
                   temperature: float = 0.0, max_tokens: int = 1024,
                   limit: int = None) -> dict:
    """Run TAT-QA evaluation using vLLM."""

    print(f"Loading dataset from {data_dir}...")
    dataset = load_tatqa_test(data_dir)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Evaluating {len(dataset)} problems")

    print(f"Loading model: {model_path}")
    if lora_path:
        print(f"With LoRA adapter: {lora_path}")

    llm = LLM(
        model=model_path,
        enable_lora=lora_path is not None,
        max_lora_rank=64,
        max_model_len=8192,
        gpu_memory_utilization=0.80,
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

    prompts = []
    for item in dataset:
        messages = create_prompt(item["question"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)

    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    responses = [o.outputs[0].text for o in outputs]

    accuracy, correct, results, scale_stats = evaluate_results(dataset, responses)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({
                "accuracy": accuracy, "correct": correct, "total": len(dataset),
                "model": model_path, "lora": lora_path,
                "scale_breakdown": {
                    k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                    for k, v in scale_stats.items()
                },
            }, f, indent=2)
        with open(os.path.join(output_dir, "details.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {"accuracy": accuracy, "correct": correct, "total": len(dataset), "scale_breakdown": scale_stats}


def run_eval_transformers(model_path: str, lora_path: str = None, output_dir: str = None,
                          data_dir: str = "data/tatqa_arithmetic",
                          temperature: float = 0.0, max_tokens: int = 1024,
                          limit: int = None, batch_size: int = 8) -> dict:
    """Run TAT-QA evaluation using HF transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading dataset from {data_dir}...")
    dataset = load_tatqa_test(data_dir)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print(f"Evaluating {len(dataset)} problems")

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    all_prompts = []
    for item in dataset:
        messages = create_prompt(item["question"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        all_prompts.append(prompt)

    print(f"Generating responses (batch_size={batch_size})...")
    all_responses = []
    for batch_start in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(model.device)
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                              pad_token_id=tokenizer.pad_token_id)
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            output_ids = model.generate(**inputs, **gen_kwargs)

        for j, out in enumerate(output_ids):
            input_len = inputs["input_ids"][j].shape[0]
            all_responses.append(tokenizer.decode(out[input_len:], skip_special_tokens=True))

    accuracy, correct, results, scale_stats = evaluate_results(dataset, all_responses)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({
                "accuracy": accuracy, "correct": correct, "total": len(dataset),
                "model": model_path, "lora": lora_path,
                "scale_breakdown": {
                    k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                    for k, v in scale_stats.items()
                },
            }, f, indent=2)
        with open(os.path.join(output_dir, "details.json"), "w") as f:
            json.dump(results, f, indent=2)

    return {"accuracy": accuracy, "correct": correct, "total": len(dataset), "scale_breakdown": scale_stats}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TAT-QA arithmetic")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/tatqa_arithmetic")
    parser.add_argument("--output_dir", type=str, default="./eval_results/tatqa")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if VLLM_AVAILABLE:
        run_fn = run_eval_vllm
        extra_kwargs = {}
    else:
        print("vLLM not available, using transformers backend")
        run_fn = run_eval_transformers
        extra_kwargs = {"batch_size": args.batch_size}

    results = run_fn(
        model_path=args.model_path, lora_path=args.lora_path,
        output_dir=args.output_dir, data_dir=args.data_dir,
        temperature=args.temperature, max_tokens=args.max_tokens,
        limit=args.limit, **extra_kwargs,
    )

    print("\n" + "=" * 60)
    print("TAT-QA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print()
    print("Per-scale breakdown:")
    for scale, stats in sorted(results["scale_breakdown"].items(), key=lambda x: x[0]):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        label = scale if scale else "(none)"
        print(f"  {label:25s} {acc:.4f} ({stats['correct']}/{stats['total']})")
    print("=" * 60)

    if args.output_dir:
        print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
