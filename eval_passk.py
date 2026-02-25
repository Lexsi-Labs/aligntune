#!/usr/bin/env python3
"""Evaluate Pass@k and Majority@k for k=8,16,32 on MBPP+ and HumanEval+."""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from collections import Counter
from itertools import combinations
import math

import numpy as np
import torch

from evalplus.data import get_mbpp_plus, get_human_eval_plus
from evalplus.eval import untrusted_check
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    patterns = [
        r'```python\s*(.*?)```',
        r'```\s*(.*?)```',
    ]

    all_blocks = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        all_blocks.extend(matches)

    if all_blocks:
        for block in all_blocks:
            if 'def ' in block:
                return block.strip()
        return all_blocks[0].strip()

    lines = response.split('\n')
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
        if in_function:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines)

    return response.strip()


def create_prompt(problem: dict, tokenizer, benchmark: str) -> str:
    """Create prompt for problem."""
    prompt_text = problem['prompt']

    if benchmark == "mbpp":
        user_content = f"""{prompt_text}

Write a Python function that solves the above problem.
Output ONLY the Python code, no markdown, no explanations."""
    else:
        user_content = f"""{prompt_text}

Complete the function. Output ONLY the Python code, no markdown, no explanations."""

    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    return user_content


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k given n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def majority_at_k(results: list, k: int, n_trials: int = 100) -> float:
    """Compute majority@k by sampling k results and taking majority vote."""
    if len(results) < k:
        return 0.0

    correct_count = 0
    for _ in range(n_trials):
        sample = random.sample(results, k)
        # Majority vote: if more than half are correct
        if sum(sample) > k // 2:
            correct_count += 1

    return correct_count / n_trials


def evaluate_problem(problem, completions, entry_point, benchmark):
    """Evaluate multiple completions for a single problem."""
    base_in = problem.get("base_input", [])
    plus_in_only = problem.get("plus_input", [])
    atol = problem.get("atol", 0.0)

    # Compute expected outputs
    try:
        canonical_code = problem["prompt"] + problem["canonical_solution"]
        canonical_globals = {}
        exec(canonical_code, canonical_globals)
        entry_func = canonical_globals[entry_point]

        def safe_call(fn, args):
            try:
                if isinstance(args, (list, tuple)):
                    return fn(*args)
                return fn(args)
            except:
                return None

        base_out = [safe_call(entry_func, args) for args in base_in]
        plus_out_only = [safe_call(entry_func, args) for args in plus_in_only]
    except Exception as e:
        return [False] * len(completions)

    plus_in = list(base_in) + list(plus_in_only)
    plus_out = list(base_out) + list(plus_out_only)

    results = []
    for completion in completions:
        try:
            result = untrusted_check(
                benchmark,
                completion,
                plus_in,
                entry_point,
                plus_out,
                atol,
                [1.0] * len(plus_in),
                fast_check=True,
            )
            results.append(result[0] == "pass")
        except:
            results.append(False)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", default="eval_results/passk")
    parser.add_argument("--benchmark", choices=["mbpp", "humaneval", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_path}")

    llm_kwargs = {
        "model": args.model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
    }

    if args.lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        lora_request = LoRARequest("adapter", 1, args.lora_path)
    else:
        lora_request = None

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=2048,
        n=args.n_samples,
    )

    benchmarks = []
    if args.benchmark in ["mbpp", "both"]:
        benchmarks.append(("mbpp", get_mbpp_plus()))
    if args.benchmark in ["humaneval", "both"]:
        benchmarks.append(("humaneval", get_human_eval_plus()))

    all_results = {}

    for benchmark_name, problems in benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating {benchmark_name.upper()}+ ({len(problems)} problems)")
        print(f"{'='*60}")

        task_ids = list(problems.keys())
        prompts = [create_prompt(problems[tid], tokenizer, benchmark_name) for tid in task_ids]

        print(f"Generating {args.n_samples} samples per problem...")
        start_time = time.time()

        if lora_request:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params)

        gen_time = time.time() - start_time
        print(f"Generation took {gen_time:.1f}s")

        print("Evaluating...")
        problem_results = []

        for i, (task_id, output) in enumerate(zip(task_ids, outputs)):
            problem = problems[task_id]
            completions = [extract_code(o.text) for o in output.outputs]

            results = evaluate_problem(
                problem, completions, problem["entry_point"], benchmark_name
            )
            problem_results.append(results)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(task_ids)} problems evaluated")

        # Compute metrics
        n = args.n_samples
        ks = [8, 16, 32]

        metrics = {}
        for k in ks:
            if k > n:
                continue

            # Pass@k
            pass_k_scores = [pass_at_k(n, sum(r), k) for r in problem_results]
            metrics[f"pass@{k}"] = np.mean(pass_k_scores)

            # Majority@k
            maj_k_scores = [majority_at_k(r, k) for r in problem_results]
            metrics[f"maj@{k}"] = np.mean(maj_k_scores)

        # Also compute pass@1 for reference
        pass_1_scores = [pass_at_k(n, sum(r), 1) for r in problem_results]
        metrics["pass@1"] = np.mean(pass_1_scores)

        all_results[benchmark_name] = metrics

        print(f"\n{benchmark_name.upper()}+ Results:")
        print(f"  Pass@1:  {metrics['pass@1']:.4f}")
        for k in ks:
            if k <= n:
                print(f"  Pass@{k}: {metrics[f'pass@{k}']:.4f}  |  Maj@{k}: {metrics[f'maj@{k}']:.4f}")

    # Save results
    summary = {
        "model": args.model_path,
        "lora_path": args.lora_path,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "results": all_results
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
