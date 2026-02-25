#!/usr/bin/env python3
"""Quick HumanEval+ evaluation script using evalplus dataset and vLLM inference."""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from evalplus.data import get_human_eval_plus
from evalplus.eval import untrusted_check
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try markdown code blocks first
    patterns = [
        r'```python\s*(.*?)```',
        r'```\s*(.*?)```',
    ]

    all_blocks = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        all_blocks.extend(matches)

    if all_blocks:
        # Return FIRST block containing a function definition
        for block in all_blocks:
            if 'def ' in block:
                return block.strip()
        return all_blocks[0].strip()

    # Try to find function definition
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


def create_prompt(problem: dict, tokenizer) -> str:
    """Create prompt for HumanEval+ problem."""
    prompt_text = problem['prompt']

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output_dir", default="eval_results/humaneval_plus")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
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
        "gpu_memory_utilization": 0.8,
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
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_p=0.95,
        max_tokens=2048,
    )

    # Load HumanEval+
    problems = get_human_eval_plus()
    task_ids = list(problems.keys())

    if args.max_samples:
        task_ids = task_ids[:args.max_samples]

    print(f"Evaluating {len(task_ids)} HumanEval+ problems")

    # Generate completions
    prompts = [create_prompt(problems[tid], tokenizer) for tid in task_ids]

    print("Generating completions...")
    start_time = time.time()

    if lora_request:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    gen_time = time.time() - start_time
    print(f"Generation took {gen_time:.1f}s")

    # Evaluate
    print("Evaluating with evalplus...")
    results = []
    passed_base = 0
    passed_plus = 0

    def safe_call(fn, args):
        try:
            if isinstance(args, (list, tuple)):
                return fn(*args)
            return fn(args)
        except:
            return None

    for i, (task_id, output) in enumerate(zip(task_ids, outputs)):
        problem = problems[task_id]
        completion = extract_code(output.outputs[0].text)

        # Get inputs
        base_in = problem.get("base_input", [])
        plus_in_only = problem.get("plus_input", [])
        atol = problem.get("atol", 0.0)

        # Compute expected outputs using canonical solution
        try:
            canonical_code = problem["prompt"] + problem["canonical_solution"]
            canonical_globals = {}
            exec(canonical_code, canonical_globals)
            entry_func = canonical_globals[problem["entry_point"]]
            base_out = [safe_call(entry_func, args) for args in base_in]
            plus_out_only = [safe_call(entry_func, args) for args in plus_in_only]
        except Exception as e:
            results.append({
                "task_id": task_id,
                "completion": completion,
                "base_passed": False,
                "plus_passed": False,
                "error": str(e)
            })
            continue

        plus_in = list(base_in) + list(plus_in_only)
        plus_out = list(base_out) + list(plus_out_only)

        # Test with base inputs
        try:
            base_result = untrusted_check(
                "humaneval",
                completion,
                base_in,
                problem["entry_point"],
                base_out,
                atol,
                [1.0] * len(base_in),
                fast_check=True,
            )
            base_passed = base_result[0] == "pass"
        except Exception as e:
            base_passed = False

        # Test with plus inputs
        try:
            plus_result = untrusted_check(
                "humaneval",
                completion,
                plus_in,
                problem["entry_point"],
                plus_out,
                atol,
                [1.0] * len(plus_in),
                fast_check=True,
            )
            plus_passed = plus_result[0] == "pass"
        except Exception as e:
            plus_passed = False

        if base_passed:
            passed_base += 1
        if plus_passed:
            passed_plus += 1

        results.append({
            "task_id": task_id,
            "completion": completion,
            "base_passed": base_passed,
            "plus_passed": plus_passed,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(task_ids)}: base={passed_base}, plus={passed_plus}")

    # Calculate metrics
    total = len(task_ids)
    humaneval_pass1 = passed_base / total
    humaneval_plus_pass1 = passed_plus / total

    print("\n" + "="*60)
    print("HUMANEVAL+ EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Total problems: {total}")
    print(f"HumanEval Pass@1: {humaneval_pass1:.4f} ({passed_base}/{total})")
    print(f"HumanEval+ Pass@1: {humaneval_plus_pass1:.4f} ({passed_plus}/{total})")
    print("="*60)

    # Save results
    summary = {
        "model": args.model_path,
        "total_problems": total,
        "humaneval_pass@1": humaneval_pass1,
        "humaneval_plus_pass@1": humaneval_plus_pass1,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
