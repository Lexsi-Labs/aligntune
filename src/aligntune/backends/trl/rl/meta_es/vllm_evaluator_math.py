"""
vLLM-based Fast Evaluator for ES Meta-Learning (Math)

This module provides fast value function V(π) computation using vLLM batched inference.
Evaluates trained policy on GSM8K validation set with batched generation.
"""

import os
import sys
import re
import math
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


# ============================================================================
# GSM8K ANSWER PARSING (from train_nmdrgrpo_math.py)
# ============================================================================

ANS_RE = re.compile(r"####\s*(.+)$", flags=re.MULTILINE)


def parse_gsm8k_gold(ans: str) -> str:
    """Extract the numeric target from GSM8K 'answer' field (after '####')."""
    m = ANS_RE.search(ans)
    if not m:
        # Fallback: last number in the string
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        return nums[-1] if nums else ""
    s = m.group(1).strip()
    s = s.replace(",", "")
    # Handle simple fractions like "3/4"
    if re.fullmatch(r"-?\d+/\d+", s):
        num, den = s.split("/")
        try:
            return str(float(num) / float(den))
        except Exception:
            return s
    return s


def parse_pred_number(text: str) -> str:
    """Extract the last number from the output."""
    # Look for the last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    candidate = nums[-1] if nums else ""
    # Normalize simple fractions
    if re.fullmatch(r"-?\d+/\d+", candidate):
        num, den = candidate.split("/")
        try:
            return str(float(num) / float(den))
        except Exception:
            return candidate
    return candidate


def numeric_equal(a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
    """Check if two answer strings are numerically equal."""
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except Exception:
        # Pure string fallback
        return a.strip() == b.strip()


# ============================================================================
# VLLM CONFIGURATION AND UTILITIES
# ============================================================================

DEFAULT_SYSTEM_MATH = "You are a careful math tutor. " "Solve step by step and provide the final numeric answer."


def _get_adaptive_memory_settings():
    """Auto-detect GPU memory and return adaptive settings."""
    if torch and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # Get GPU memory in GB
        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_mem_gb = gpu_mem_bytes / (1024**3)

        print(f"Detected {gpu_mem_gb:.1f}GB GPU memory")

        # Adaptive memory utilization based on available memory
        if gpu_mem_gb >= 40:  # High-end GPUs (A100, H100, L40S, etc.)
            gpu_utilization = 0.9
            max_model_len = 32768
        elif gpu_mem_gb >= 20:  # Mid-range GPUs (RTX 3090, 4090, etc.)
            gpu_utilization = 0.75
            max_model_len = 16384
        elif gpu_mem_gb >= 10:  # Lower-end GPUs
            gpu_utilization = 0.7
            max_model_len = 8192
        else:  # Very limited memory
            gpu_utilization = 0.6
            max_model_len = 4096

        print(
            f"Using {
                gpu_utilization *
                100:.0f}% GPU memory, max context: {max_model_len}")
        return gpu_utilization, max_model_len
    else:
        print("No CUDA GPU detected, using conservative defaults")
        return 0.7, 8192


@dataclass
class VLLMConfig:
    model: str
    max_new_tokens: int = 256  # Math needs less tokens than code
    temperature: float = 0.0  # Greedy decoding for evaluation
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1  # Single GPU to avoid distributed communication issues
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    lora_path: Optional[str] = None
    do_sample: bool = False  # Greedy decoding for evaluation

    def __post_init__(self):
        """Auto-detect memory settings if not specified."""
        if self.gpu_memory_utilization is None or self.max_model_len is None:
            auto_utilization, auto_max_len = _get_adaptive_memory_settings()
            if self.gpu_memory_utilization is None:
                self.gpu_memory_utilization = auto_utilization
            if self.max_model_len is None:
                self.max_model_len = auto_max_len


def _is_lora_checkpoint(path: str) -> bool:
    """Check if a path contains LoRA checkpoint files."""
    path_obj = Path(path)
    if not path_obj.exists():
        return False

    # Check for key LoRA files
    lora_files = ["adapter_config.json", "adapter_model.safetensors"]

    return all((path_obj / file).exists() for file in lora_files)


class VLLMRunner:
    def __init__(self, cfg: VLLMConfig):
        from vllm import LLM, SamplingParams

        # Suppress VLLM progress bars and verbose output
        os.environ["VLLM_LOG_LEVEL"] = "WARNING"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Handle LoRA checkpoint loading
        if cfg.lora_path:
            if not _is_lora_checkpoint(cfg.lora_path):
                raise ValueError(
                    f"❌ LoRA path specified but invalid: {cfg.lora_path}\n"
                    f"   Path must exist and contain adapter_config.json and adapter_model.safetensors"
                )

            print(
                f"Loading model {
                    cfg.model} with LoRA adapter {
                    cfg.lora_path} using VLLM...")

            # Use vLLM's native LoRA support with LoRARequest
            self.llm = LLM(
                model=cfg.model,
                dtype=cfg.dtype,
                trust_remote_code=cfg.trust_remote_code,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                seed=cfg.seed,
                max_model_len=cfg.max_model_len,
                enable_lora=True,
                max_loras=1,
                max_lora_rank=64,
            )

            # Store LoRA path for use in generation
            self.lora_path = cfg.lora_path
            print(f"✅ LoRA support enabled. LoRA adapter: {cfg.lora_path}")
        else:
            # Standard model loading (no LoRA specified)
            self.lora_path = None
            print(f"Loading model {cfg.model} with VLLM (no LoRA adapter)...")
            self.llm = LLM(
                model=cfg.model,
                dtype=cfg.dtype,
                trust_remote_code=cfg.trust_remote_code,
                tensor_parallel_size=cfg.tensor_parallel_size,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                seed=cfg.seed,
                max_model_len=cfg.max_model_len,
            )

        self.sampling_params = SamplingParams(
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p if cfg.do_sample else 1.0,
            top_k=cfg.top_k if cfg.do_sample else -1,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )

        self.cfg = cfg
        print("VLLM model loaded successfully!")

    def cleanup(self):
        """Clean up VLLM resources and GPU memory aggressively."""
        try:
            import gc
            import torch

            print("🧹 Cleaning up VLLM model...")

            # More aggressive VLLM engine cleanup
            if hasattr(self, "llm"):
                # Try to shutdown VLLM engine properly
                try:
                    if hasattr(self.llm, "engine"):
                        print("  🔄 Shutting down VLLM engine workers...")
                        # VLLM engines may have worker processes
                        if hasattr(self.llm.engine, "workers"):
                            for worker in self.llm.engine.workers:
                                if hasattr(worker, "shutdown"):
                                    worker.shutdown()
                except Exception as e:
                    print(f"  ⚠️ Engine shutdown warning: {e}")

                del self.llm

            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()

            # Aggressive GPU memory cleanup
            if torch.cuda.is_available():
                print("  🧹 Clearing CUDA memory...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Reset memory stats to get clean slate
                torch.cuda.reset_peak_memory_stats()

                # Clear cache on all available devices
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()

            print("✅ VLLM cleanup completed")

        except Exception as e:
            print(f"⚠️ Warning during VLLM cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except BaseException:
            pass

    def generate_batch(self,
                       prompts: List[str],
                       n: int = 1,
                       system: Optional[str] = None) -> List[List[str]]:
        """Generate responses for multiple prompts in a single batch for fast evaluation."""
        from vllm import SamplingParams

        if not prompts:
            return []

        # Format as chat messages for all prompts
        if system is None:
            system = DEFAULT_SYSTEM_MATH

        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            formatted_prompts.append(formatted_prompt)

        # Update sampling params for this generation
        sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=self.sampling_params.max_tokens,
            n=n,
            seed=self.cfg.seed,
        )

        # Create LoRA request if LoRA path is available
        lora_request = None
        if hasattr(self, "lora_path") and self.lora_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("lora_adapter", 1, self.lora_path)

        # Generate for all prompts in one batch - THIS IS THE SPEED BOOST!
        print(
            f"🚀 Batch generating {
                len(formatted_prompts)} prompts × {n} samples = {
                len(formatted_prompts) *
                n} total generations")
        outputs = self.llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_request)

        # Extract generated text for each prompt
        all_results = []
        for output in outputs:
            prompt_results = []
            for completion in output.outputs:
                generated_text = completion.text
                prompt_results.append(generated_text)
            all_results.append(prompt_results)

        return all_results


# ============================================================================
# GSM8K DATASET UTILITIES
# ============================================================================


def get_gsm8k_splits(test_size: float = 0.2,
                     seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split GSM8K train (7473) into inner_train (5978) + validation (1495).

    Args:
        test_size: fraction for validation split (default: 0.2 for ~1495 samples)
        seed: random seed for reproducibility

    Returns:
        Tuple of (inner_train_dataset, validation_dataset)
    """
    # Load GSM8K train split
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    # Split into inner_train and validation
    indices = list(range(len(gsm8k)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True)

    inner_train = gsm8k.select(train_indices)
    validation = gsm8k.select(val_indices)

    print(
        f"GSM8K split: {len(inner_train)} inner_train + {len(validation)} validation")

    return inner_train, validation


def build_gsm8k_prompt(problem: dict) -> str:
    """Build prompt for GSM8K problem."""
    return f"Problem:\n{problem['question']}"


# ============================================================================
# VALUE FUNCTION COMPUTATION
# ============================================================================


def compute_value_function_gsm8k(
    checkpoint_dir: str,
    validation_dataset: List[dict],
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_new_tokens: int = 256,
    k: int = 1,
    temperature: float = 0.8,
    verbose: bool = False,
) -> float:
    """
    Compute value function V(π) = GSM8K accuracy@k on validation set using vLLM.

    This is the fitness function for ES meta-learning.
    Uses vLLM batched inference for fast evaluation.

    Args:
        checkpoint_dir: Path to LoRA checkpoint directory
        validation_dataset: List of GSM8K problem dicts (typically ~1500 problems or subset)
        base_model: Base model identifier
        max_new_tokens: Maximum tokens to generate per problem
        k: Number of samples per problem for pass@k metric (default: 1)
        temperature: Sampling temperature for k>1 (default: 0.8, ignored when k=1)
        verbose: Print detailed evaluation progress

    Returns:
        accuracy@k metric (fraction of problems with at least 1 correct answer in k attempts)
    """
    print(f"\n{'=' * 60}")
    print(
        f"🔍 Computing V(π) = GSM8K accuracy@{k} for checkpoint: {checkpoint_dir}")
    print(
        f"📊 Validation set: {
            len(validation_dataset)} problems × {k} samples")
    print(f"{'=' * 60}\n")

    # Initialize vLLM with LoRA checkpoint
    # For accuracy@k with k>1, use specified temperature for diversity
    # For accuracy@1, use greedy decoding (temperature=0.0)
    eval_temperature = temperature if k > 1 else 0.0

    cfg = VLLMConfig(
        model=base_model,
        max_new_tokens=max_new_tokens,
        lora_path=checkpoint_dir,
        temperature=eval_temperature,
        do_sample=k > 1,  # Enable sampling only when k > 1
        seed=42,
    )

    if k > 1:
        print(
            f"🎲 Using sampling with temperature={eval_temperature} for accuracy@{k}")

    runner = VLLMRunner(cfg)

    try:
        # Build prompts for all validation problems
        prompts = [build_gsm8k_prompt(problem)
                   for problem in validation_dataset]

        # Extract gold answers
        gold_answers = [
            parse_gsm8k_gold(
                problem["answer"]) for problem in validation_dataset]

        # Batch generate k solutions per problem
        print(
            f"⚡ Generating {
                len(prompts)} prompts × {k} samples = {
                len(prompts) *
                k} total generations")
        all_completions = runner.generate_batch(prompts, n=k)

        # Evaluate each problem with k attempts
        problems_solved = 0
        total_problems = len(validation_dataset)

        for i, (problem, completions, gold) in enumerate(
                zip(validation_dataset, all_completions, gold_answers)):
            # Try all k completions, check if ANY matches the gold answer
            problem_solved = False

            for attempt_idx, response in enumerate(completions):
                # Extract predicted answer
                predicted = parse_pred_number(response)

                # Check if correct
                if numeric_equal(predicted, gold):
                    problem_solved = True
                    if verbose:
                        print(f"  ✅ Solved on attempt {attempt_idx + 1}/{k}")
                    break  # Found a correct answer, stop checking

            if problem_solved:
                problems_solved += 1

            if verbose:
                status = "✅ PASS" if problem_solved else "❌ FAIL"
                print(
                    f"  [{i + 1}/{total_problems}] {status} - Problem: {problem['question'][:50]}...")

        # Compute accuracy@k metric
        accuracy_at_k = problems_solved / total_problems

        print(f"\n{'=' * 60}")
        print(f"📈 V(π) = GSM8K accuracy@{k} = {accuracy_at_k:.4f}")
        print(
            f"   ({problems_solved}/{total_problems} problems solved with {k} attempts)")
        print(f"{'=' * 60}\n")

        return accuracy_at_k

    finally:
        # Clean up vLLM resources
        runner.cleanup()
