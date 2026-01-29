#!/usr/bin/env python3
"""
Precompute baseline v̂ values for BOLT warm-start.

Generates baseline success rates for each prompt in a dataset by:
1. Loading a model (base or checkpoint)
2. Generating N completions per prompt
3. Scoring correctness of each completion
4. Computing v̂(x) = correct/N for each prompt

Output JSON can be used for warm-start in BOLT training:
  baseline_warm_start: "./baselines/gsm8k_qwen25.json"

Usage examples:
    # GSM8K with base model
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --dataset gsm8k \
        --num_samples 8 \
        --output baselines/gsm8k_qwen25.json

    # MATH with checkpoint
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --lora_path ./checkpoint-500 \
        --dataset math \
        --num_samples 8 \
        --output baselines/math_step500.json

    # Custom dataset with custom prompt formatter
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --dataset_path ./my_dataset.jsonl \
        --prompt_template "Solve: {question}\nAnswer:" \
        --answer_key "solution" \
        --num_samples 8 \
        --output baselines/custom.json

Output format:
{
    "metadata": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset": "gsm8k",
        "num_samples": 8,
        "n_prompts": 7473,
        "mean_v_hat": 0.423,
        "timestamp": "2024-12-01T12:00:00"
    },
    "baselines": {
        "prompt_key_1": 0.75,
        "prompt_key_2": 0.125,
        ...
    }
}
"""

import os
import re
import json
import math
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple

import numpy as np
from tqdm import tqdm

# Optional sympy for robust math validation
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import simplify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Optional evalplus for MBPP+ dataset
try:
    from evalplus.data import get_mbpp_plus
    EVALPLUS_AVAILABLE = True
except ImportError:
    EVALPLUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import robust math grading from shared module
from aligntune.utils.math_grading import (
    grade_math_answer,
    grade_gsm8k_answer,
    extract_math_answer as shared_extract_math_answer,
    extract_math_gold,
    extract_gsm8k_answer as shared_extract_gsm8k_answer,
    parse_gsm8k_gold,
)


# ============================================================================
# Robust answer extraction and validation (uses shared math_grading module)
# ============================================================================

def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} format with proper brace matching.

    Handles nested braces like \\boxed{\\frac{1}{2}} correctly.
    """
    idx = text.find('\\boxed')
    if idx == -1:
        return ""
    idx += len('\\boxed')
    # Skip whitespace
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != '{':
        return ""

    # Match braces properly
    brace_count = 1
    idx += 1
    start = idx
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return text[start:idx-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize LaTeX answer for comparison."""
    answer = answer.strip()
    # Remove common LaTeX spacing commands
    answer = re.sub(r'\\,|\\;|\\:|\\!|\\quad|\\qquad', ' ', answer)
    # Remove \text{} wrapper
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    # Remove \left and \right
    answer = re.sub(r'\\left|\\right', '', answer)
    # Normalize operators
    answer = re.sub(r'\\cdot', '*', answer)
    answer = re.sub(r'\\times', '*', answer)
    answer = re.sub(r'\\div', '/', answer)
    # Remove $ signs
    answer = answer.replace('$', '').strip()
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer


def latex_to_sympy(latex_str: str):
    """Convert LaTeX string to sympy expression."""
    if not SYMPY_AVAILABLE:
        return None
    try:
        s = latex_str.strip()
        # Clean up LaTeX for parsing
        s = re.sub(r'\\text\{([^}]+)\}', '', s)
        s = re.sub(r'\\,|\\;|\\:|\\!', '', s)
        s = re.sub(r'\\left|\\right', '', s)
        s = s.replace('\\cdot', '*')
        s = s.replace('\\times', '*')
        s = s.replace('\\div', '/')
        expr = parse_latex(s)
        return expr
    except Exception:
        return None


def sympy_equal(pred_str: str, gold_str: str) -> bool:
    """Check if two expressions are mathematically equal using sympy."""
    if not SYMPY_AVAILABLE:
        return False
    try:
        pred_expr = latex_to_sympy(pred_str)
        gold_expr = latex_to_sympy(gold_str)
        if pred_expr is None or gold_expr is None:
            return False

        # Try symbolic equality
        if simplify(pred_expr - gold_expr) == 0:
            return True

        # Try numeric evaluation
        try:
            pred_val = complex(N(pred_expr, 10))
            gold_val = complex(N(gold_expr, 10))
            if abs(pred_val - gold_val) < 1e-6:
                return True
            # Relative tolerance for larger numbers
            if abs(gold_val) > 1e-10:
                if abs(pred_val - gold_val) / abs(gold_val) < 1e-4:
                    return True
        except:
            pass
        return False
    except Exception:
        return False


def parse_coordinate(s: str):
    """Parse coordinate pair like (1, 2) or [1, 2]."""
    s = s.strip()
    match = re.match(r'[\[\(]?\s*([^,\[\]\(\)]+)\s*,\s*([^,\[\]\(\)]+)\s*[\]\)]?', s)
    if match:
        try:
            a = float(match.group(1).strip())
            b = float(match.group(2).strip())
            return (a, b)
        except:
            return None
    return None


# Tolerance constants - MUST match rewards/core.py MathReasoningReward
# Training uses rel_tol=1e-5, abs_tol=1e-9
NUMERIC_REL_TOL = 1e-5
NUMERIC_ABS_TOL = 1e-9


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.

    Matches the logic from rewards/core.py MathReasoningReward._extract_numbers
    for consistency between training and baseline computation.

    Supports:
    - Standard decimal numbers: 123, -45.67
    - Fractions: 1/2, 3/4
    - Scientific notation: 1.5e-3, 2E10
    """
    numbers = []

    # Standard decimal numbers
    pattern1 = r'-?\d+\.?\d*'
    for match in re.findall(pattern1, text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    # Fractions (e.g., "1/2", "3/4")
    pattern2 = r'(\d+)\s*/\s*(\d+)'
    for num, den in re.findall(pattern2, text):
        try:
            numbers.append(float(num) / float(den))
        except (ValueError, ZeroDivisionError):
            continue

    # Scientific notation (e.g., "1.5e-3", "2E10")
    pattern3 = r'-?\d+\.?\d*[eE][+-]?\d+'
    for match in re.findall(pattern3, text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def numbers_match(num1: float, num2: float) -> bool:
    """Check if two numbers match within tolerance.

    Uses same tolerance as rewards/core.py MathReasoningReward._numbers_match
    """
    return math.isclose(num1, num2, rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)


def any_number_matches(pred_text: str, gold_text: str) -> bool:
    """
    Check if ANY number in prediction matches ANY number in gold.

    This matches the training reward logic from MathReasoningReward._evaluate_correctness:
    - Extract all numbers from both texts
    - Return True if any pair matches within tolerance

    This is important for consistency: during training, the model gets reward=1.0
    if any extracted number matches, so baseline computation should use the same logic.
    """
    pred_nums = extract_numbers(pred_text)
    gold_nums = extract_numbers(gold_text)

    if not pred_nums or not gold_nums:
        return False

    for pred_num in pred_nums:
        for gold_num in gold_nums:
            if numbers_match(pred_num, gold_num):
                return True

    return False


def answers_equal(pred: str, gold: str) -> bool:
    """
    Comprehensive answer comparison for MATH-style problems.

    Tries multiple matching strategies (in order):
    1. Exact string match (normalized)
    2. Numeric comparison with tolerance (matches training)
    3. ANY number in pred matches ANY number in gold (matches training reward)
    4. Coordinate pair comparison
    5. Sympy symbolic/numeric equality (bonus robustness)

    The tolerance and number extraction logic MUST match rewards/core.py
    MathReasoningReward for consistency between training and baseline computation.
    """
    pred = pred.strip()
    gold = gold.strip()

    # Normalize and compare
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Case-insensitive, whitespace-normalized
    pred_clean = re.sub(r'\s+', '', pred_norm.lower())
    gold_clean = re.sub(r'\s+', '', gold_norm.lower())
    if pred_clean == gold_clean:
        return True

    # Numeric comparison (direct)
    try:
        pred_num = float(pred_norm.replace(',', ''))
        gold_num = float(gold_norm.replace(',', ''))
        if numbers_match(pred_num, gold_num):
            return True
    except:
        pass

    # ANY number matches (training reward logic)
    # This is critical for matching training behavior
    if any_number_matches(pred, gold):
        return True

    # Coordinate comparison
    pred_coord = parse_coordinate(pred_norm)
    gold_coord = parse_coordinate(gold_norm)
    if pred_coord is not None and gold_coord is not None:
        if (numbers_match(pred_coord[0], gold_coord[0]) and
            numbers_match(pred_coord[1], gold_coord[1])):
            return True

    # Sympy comparison (most robust but slowest)
    if sympy_equal(pred, gold):
        return True
    if sympy_equal(pred_norm, gold_norm):
        return True

    return False


# ============================================================================
# Dataset formatters
# ============================================================================

def format_gsm8k(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format GSM8K example."""
    question = example["question"]
    answer = example["answer"]

    # Extract numeric answer
    if "####" in answer:
        gold = answer.split("####")[-1].strip().replace(",", "")
    else:
        gold = answer.strip().replace(",", "")

    # Format prompt
    # Note: enable_thinking=False disables Qwen3's thinking mode
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
            {"role": "user", "content": f"Solve this problem and put your final numeric answer after '####':\n\n{question}"}
        ]
        # Disable thinking mode for Qwen3 models (no <think> tags)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    key = f"gsm8k_{idx}"
    return prompt, gold, key


def format_math(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format MATH/MATH-500 example."""
    problem = example.get("problem", example.get("question", ""))
    solution = example.get("solution", example.get("answer", ""))

    # Extract answer from \boxed{} if present
    match = re.search(r"\\boxed\{([^}]+)\}", solution)
    if match:
        gold = match.group(1).strip()
    else:
        gold = solution.strip()

    # Format prompt
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
            {"role": "user", "content": f"Solve this problem and put your final answer within \\boxed{{}}:\n\n{problem}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    key = f"math_{idx}"
    return prompt, gold, key


def format_mbpp(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format MBPP example with explicit function signature."""
    prompt_text = example.get("prompt", example.get("text", ""))
    test_cases = example.get("test_list", example.get("test", []))

    # Extract expected function name from test assertions
    # e.g., "assert first_repeated_char(...)" -> "first_repeated_char"
    func_name = "solution"
    if test_cases:
        first_test = test_cases[0] if isinstance(test_cases, list) else str(test_cases)
        match = re.search(r'assert\s+(\w+)\s*\(', first_test)
        if match:
            func_name = match.group(1)

    # Format prompt with explicit function signature (like bolt script)
    user_content = f"""{prompt_text}

IMPLEMENT THE FUNCTION:
def {func_name}(...):
    # Your implementation here

REQUIREMENTS:
- Use this exact function name: {func_name}
- Return a value, do not print
- Use only ASCII characters
- Output ONLY Python code
- No markdown, no explanations, no examples"""

    # Note: enable_thinking=False disables Qwen3's thinking mode (no <think> tags)
    if use_chat:
        messages = [
            {"role": "user", "content": user_content}
        ]
        # Disable thinking mode for Qwen3 models (no <think> tags)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Write a Python function:\n{prompt_text}"

    # Gold is the test cases (for reference)
    gold = json.dumps(test_cases) if isinstance(test_cases, list) else str(test_cases)

    # Use task_id if available (like bolt baseline format: mbpp_train_Mbpp/{task_id})
    task_id = example.get("task_id", idx)
    # evalplus task_id is already "Mbpp/102", HF uses int like 601
    if isinstance(task_id, str) and task_id.startswith("Mbpp/"):
        key = f"mbpp_train_{task_id}"  # -> mbpp_train_Mbpp/102
    else:
        key = f"mbpp_train_Mbpp/{task_id}"  # -> mbpp_train_Mbpp/601
    return prompt, gold, key


DATASET_FORMATTERS = {
    "gsm8k": format_gsm8k,
    "math": format_math,
    "mbpp": format_mbpp,
}


# ============================================================================
# Answer extractors/validators
# ============================================================================

def extract_gsm8k_answer(text: str) -> str:
    """Extract answer from GSM8K-style response.

    Tries multiple formats:
    1. #### format (standard GSM8K)
    2. \\boxed{} format (Qwen3 and other models) - uses robust brace matching
    3. Last number in response (fallback)
    """
    # Try #### format first
    if "####" in text:
        parts = text.split("####")
        answer = parts[-1].strip().split()[0] if parts[-1].strip() else ""
        return answer.replace(",", "")

    # Try \boxed{} format with robust extraction (handles nested braces)
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.replace(",", "")

    # Fallback: try to find last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_math_answer(text: str) -> str:
    """Extract answer from MATH-style response.

    Uses robust brace-matching extract_boxed_answer, with fallback to last number.
    """
    # Try \boxed{} with robust extraction
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Fallback: try to find last number in text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return ""


def validate_gsm8k(pred: str, gold: str) -> bool:
    """Validate GSM8K answer with numeric tolerance.

    Uses shared math_grading module for consistency across eval/training/baseline.
    """
    return grade_gsm8k_answer(pred, gold)


def validate_math(pred: str, gold: str) -> bool:
    """Validate MATH answer using three-tier grading.

    Uses shared math_grading module with:
    1. Official MATH dataset normalization
    2. PRM800K normalization
    3. SymPy symbolic equivalence
    """
    return grade_math_answer(pred, gold)


def _strip_markdown(code: str) -> str:
    """Extract code from markdown code blocks."""
    # Remove ```python ... ``` blocks
    code = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), code, flags=re.S)
    code = re.sub(r"^```(\w+)?\s*|\s*```$", "", code.strip(), flags=re.M)
    return code


def _ensure_entrypoint(code: str, expected_name: str) -> str:
    """Ensure the expected function name exists, adding alias if needed."""
    # If expected function already exists, return as-is
    if re.search(rf"\bdef\s+{re.escape(expected_name)}\s*\(", code):
        return code

    # Find any function definition and alias it
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        actual = m.group(1)
        return code + f"\n\n# alias for test\n{expected_name} = {actual}\n"

    # No function found - add stub that raises
    return f"def {expected_name}(*args, **kwargs):\n    raise NotImplementedError\n\n" + code


def _flexible_signature_shim(expected_name: str) -> str:
    """Handle argument count mismatches gracefully.

    Note: Disabled for now as SafeCodeExecutor runs in restricted namespace.
    """
    # Return empty string - the entrypoint aliasing should be sufficient
    return ""


def validate_mbpp_with_execution(pred: str, gold: str) -> float:
    """
    Validate MBPP code by executing against test cases.

    Args:
        pred: Generated code
        gold: JSON string containing test cases

    Returns:
        Float score: fraction of tests passed (0.0 to 1.0)
    """
    try:
        from aligntune.eval.safe_executor import SafeCodeExecutor

        # Parse test cases from gold
        test_cases_raw = json.loads(gold) if isinstance(gold, str) else gold

        if not test_cases_raw:
            return 0.0

        # Extract expected function name from first test
        expected_name = "solution"
        first_test = test_cases_raw[0] if test_cases_raw else ""
        match = re.search(r'assert\s+(\w+)\s*\(', first_test)
        if match:
            expected_name = match.group(1)

        # Execute code with test assertions
        executor = SafeCodeExecutor(timeout=5)

        # Extract and prepare code
        code = executor.extract_code(pred)
        if not code:
            code = _strip_markdown(pred)

        # Ensure function name matches
        code = _ensure_entrypoint(code, expected_name)

        # Run each test case separately to count passes
        tests_passed = 0
        for tc in test_cases_raw:
            if isinstance(tc, str):
                test_code = code + "\n\n" + tc + "\n"
                result = executor.execute(test_code, test_cases=[])
                if result.success and not result.error:
                    tests_passed += 1

        # Return fractional score
        return float(tests_passed / len(test_cases_raw)) if test_cases_raw else 0.0

    except Exception as e:
        return 0.0


def validate_mbpp_stub(pred: str, gold: str) -> bool:
    """
    STUB: MBPP validation requires code execution.

    WARNING: This returns False always. For actual MBPP baseline computation,
    use evalplus or another code execution framework.
    """
    logger.warning(
        "MBPP validation requires code execution. "
        "Use evalplus for accurate baselines. Returning False."
    )
    return False


ANSWER_EXTRACTORS = {
    "gsm8k": extract_gsm8k_answer,
    "math": extract_math_answer,
    "mbpp": lambda x: x,  # MBPP needs code execution, not extraction
}

def validate_exact_match(pred: str, gold: str) -> bool:
    """Simple exact match validator (fallback)."""
    return pred.strip().lower() == gold.strip().lower()


VALIDATORS = {
    "gsm8k": validate_gsm8k,
    "math": validate_math,
    "mbpp": validate_mbpp_with_execution,  # Uses SafeCodeExecutor
}


# ============================================================================
# Prompt key generation
# ============================================================================

def make_prompt_key(prompt: str) -> str:
    """Create stable hash-based key for prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


# ============================================================================
# Main computation
# ============================================================================

def compute_baselines(
    llm,
    prompts: List[str],
    golds: List[str],
    keys: List[str],
    dataset_type: str,
    num_samples: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Dict[str, float]:
    """
    Compute v̂(x) for each prompt by generating samples and scoring.

    Returns:
        Dictionary mapping prompt keys to v̂ values
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        n=num_samples,
    )

    logger.info(f"Generating {num_samples} samples for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Get extractor and validator for dataset
    extractor = ANSWER_EXTRACTORS.get(dataset_type, lambda x: x.strip())
    validator = VALIDATORS.get(dataset_type, validate_exact_match)

    baselines = {}

    for i, output in enumerate(tqdm(outputs, desc="Scoring")):
        key = keys[i]
        gold = golds[i]

        # Score each sample
        total_score = 0.0
        for completion in output.outputs:
            pred = extractor(completion.text)
            result = validator(pred, gold)
            # Handle both boolean (gsm8k, math) and float (mbpp) returns
            if isinstance(result, bool):
                total_score += 1.0 if result else 0.0
            else:
                total_score += float(result)

        # Compute v̂ = mean score across samples
        v_hat = total_score / num_samples
        baselines[key] = v_hat

    return baselines


def main():
    parser = argparse.ArgumentParser(
        description="Precompute baseline v̂ values for BOLT warm-start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--lora_path", help="Optional LoRA adapter path")
    parser.add_argument("--lora_name", default="adapter", help="Name for LoRA adapter")

    # Dataset arguments
    parser.add_argument("--dataset", choices=["gsm8k", "math", "mbpp", "custom"],
                        default="gsm8k", help="Dataset type")
    parser.add_argument("--dataset_path", help="Path for custom dataset (JSONL)")
    parser.add_argument("--dataset_split", default="train", help="Dataset split")
    parser.add_argument("--max_prompts", type=int, help="Limit number of prompts")

    # Prompt formatting
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                        help="Use chat template formatting")
    parser.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")

    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")

    # Output
    parser.add_argument("--output", required=True, help="Output JSON path")

    # Runtime
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", "-y", action="store_true",
                        help="Skip confirmation prompts")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BOLT Baseline Precomputation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    if args.lora_path:
        logger.info(f"LoRA: {args.lora_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Output: {args.output}")
    if SYMPY_AVAILABLE:
        logger.info("Sympy: available (robust math validation enabled)")
    else:
        logger.info("Sympy: not available (using basic validation)")
    logger.info("=" * 60)

    # MBPP info - code execution enabled
    if args.dataset == "mbpp":
        logger.info("=" * 60)
        logger.info("MBPP baseline computation with SafeCodeExecutor")
        logger.info("Generated code will be executed against test assertions")
        logger.info("Timeout per execution: 5 seconds")
        logger.info("=" * 60)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.dataset == "custom":
        if not args.dataset_path:
            raise ValueError("--dataset_path required for custom dataset")
        with open(args.dataset_path) as f:
            if args.dataset_path.endswith(".jsonl"):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)
    else:
        from datasets import load_dataset
        if args.dataset == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split=args.dataset_split)
        elif args.dataset == "math":
            dataset = load_dataset("hendrycks/competition_math", split=args.dataset_split)
        elif args.dataset == "mbpp":
            if EVALPLUS_AVAILABLE:
                logger.info("Using evalplus MBPP+ dataset (better test coverage)")
                mbpp_plus = get_mbpp_plus()
                # Convert evalplus format to list of dicts
                dataset = []
                for task_id, prob in mbpp_plus.items():
                    # Extract test assertions from prompt
                    tests = []
                    assert_matches = re.findall(r'(assert\s+.+?)(?:\n|$)', prob["prompt"])
                    for assert_stmt in assert_matches:
                        tests.append(assert_stmt.strip())
                    dataset.append({
                        "task_id": task_id,  # e.g., "Mbpp/2"
                        "text": prob["prompt"],
                        "test_list": tests if tests else ["pass"],
                        "entry_point": prob.get("entry_point", ""),
                    })
                logger.info(f"Loaded {len(dataset)} problems from evalplus MBPP+")
            else:
                logger.info("evalplus not available, using HuggingFace MBPP dataset")
                dataset = load_dataset("mbpp", split=args.dataset_split)

    # Limit prompts if specified
    if args.max_prompts and len(dataset) > args.max_prompts:
        indices = np.random.choice(len(dataset), size=args.max_prompts, replace=False)
        if hasattr(dataset, "select"):
            dataset = dataset.select(sorted(indices))
        else:
            dataset = [dataset[i] for i in sorted(indices)]

    # Format prompts
    logger.info(f"Formatting {len(dataset)} prompts...")
    formatter = DATASET_FORMATTERS.get(args.dataset)

    prompts, golds, keys = [], [], []
    for i, example in enumerate(tqdm(dataset, desc="Formatting")):
        prompt, gold, key = formatter(example, i, tokenizer, args.use_chat_template)
        prompts.append(prompt)
        golds.append(gold)
        keys.append(key)

    logger.info(f"Prepared {len(prompts)} prompts")

    # Load model with vLLM
    from vllm import LLM

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }

    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64  # Accommodate larger LoRA ranks

    logger.info("Loading model with vLLM...")
    llm = LLM(**llm_kwargs)

    # Load LoRA if specified
    if args.lora_path:
        logger.info(f"Loading LoRA adapter from {args.lora_path}")
        llm.load_lora_adapter(
            lora_name=args.lora_name,
            lora_path=args.lora_path
        )
        llm.set_lora_adapter(args.lora_name)

    # Compute baselines
    baselines = compute_baselines(
        llm=llm,
        prompts=prompts,
        golds=golds,
        keys=keys,
        dataset_type=args.dataset,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Compute statistics
    v_hats = list(baselines.values())
    mean_v = np.mean(v_hats)
    std_v = np.std(v_hats)

    # Create output
    output = {
        "metadata": {
            "model": args.model,
            "lora_path": args.lora_path,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "num_samples": args.num_samples,
            "n_prompts": len(baselines),
            "mean_v_hat": float(mean_v),
            "std_v_hat": float(std_v),
            "pct_easy": float(100 * sum(1 for v in v_hats if v > 0.8) / len(v_hats)),
            "pct_hard": float(100 * sum(1 for v in v_hats if v < 0.2) / len(v_hats)),
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
        },
        "baselines": baselines,
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("Baseline Computation Complete")
    logger.info("=" * 60)
    logger.info(f"Prompts: {len(baselines)}")
    logger.info(f"Mean v̂: {mean_v:.3f} ± {std_v:.3f}")
    logger.info(f"Easy (v̂ > 0.8): {output['metadata']['pct_easy']:.1f}%")
    logger.info(f"Hard (v̂ < 0.2): {output['metadata']['pct_hard']:.1f}%")
    logger.info(f"Saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Precompute baseline v̂ values for BOLT warm-start.

Generates baseline success rates for each prompt in a dataset by:
1. Loading a model (base or checkpoint)
2. Generating N completions per prompt
3. Scoring correctness of each completion
4. Computing v̂(x) = correct/N for each prompt

Output JSON can be used for warm-start in BOLT training:
  baseline_warm_start: "./baselines/gsm8k_qwen25.json"

Usage examples:
    # GSM8K with base model
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --dataset gsm8k \
        --num_samples 8 \
        --output baselines/gsm8k_qwen25.json

    # MATH with checkpoint
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --lora_path ./checkpoint-500 \
        --dataset math \
        --num_samples 8 \
        --output baselines/math_step500.json

    # Custom dataset with custom prompt formatter
    python precompute_baseline.py \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        --dataset_path ./my_dataset.jsonl \
        --prompt_template "Solve: {question}\nAnswer:" \
        --answer_key "solution" \
        --num_samples 8 \
        --output baselines/custom.json

Output format:
{
    "metadata": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset": "gsm8k",
        "num_samples": 8,
        "n_prompts": 7473,
        "mean_v_hat": 0.423,
        "timestamp": "2024-12-01T12:00:00"
    },
    "baselines": {
        "prompt_key_1": 0.75,
        "prompt_key_2": 0.125,
        ...
    }
}
"""

import os
import re
import json
import math
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple

import numpy as np
from tqdm import tqdm

# Optional sympy for robust math validation
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import simplify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Optional evalplus for MBPP+ dataset
try:
    from evalplus.data import get_mbpp_plus
    EVALPLUS_AVAILABLE = True
except ImportError:
    EVALPLUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import robust math grading from shared module
from aligntune.utils.math_grading import (
    grade_math_answer,
    grade_gsm8k_answer,
    extract_math_answer as shared_extract_math_answer,
    extract_math_gold,
    extract_gsm8k_answer as shared_extract_gsm8k_answer,
    parse_gsm8k_gold,
)


# ============================================================================
# Robust answer extraction and validation (uses shared math_grading module)
# ============================================================================

def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} format with proper brace matching.

    Handles nested braces like \\boxed{\\frac{1}{2}} correctly.
    """
    idx = text.find('\\boxed')
    if idx == -1:
        return ""
    idx += len('\\boxed')
    # Skip whitespace
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != '{':
        return ""

    # Match braces properly
    brace_count = 1
    idx += 1
    start = idx
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return text[start:idx-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize LaTeX answer for comparison."""
    answer = answer.strip()
    # Remove common LaTeX spacing commands
    answer = re.sub(r'\\,|\\;|\\:|\\!|\\quad|\\qquad', ' ', answer)
    # Remove \text{} wrapper
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    # Remove \left and \right
    answer = re.sub(r'\\left|\\right', '', answer)
    # Normalize operators
    answer = re.sub(r'\\cdot', '*', answer)
    answer = re.sub(r'\\times', '*', answer)
    answer = re.sub(r'\\div', '/', answer)
    # Remove $ signs
    answer = answer.replace('$', '').strip()
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer


def latex_to_sympy(latex_str: str):
    """Convert LaTeX string to sympy expression."""
    if not SYMPY_AVAILABLE:
        return None
    try:
        s = latex_str.strip()
        # Clean up LaTeX for parsing
        s = re.sub(r'\\text\{([^}]+)\}', '', s)
        s = re.sub(r'\\,|\\;|\\:|\\!', '', s)
        s = re.sub(r'\\left|\\right', '', s)
        s = s.replace('\\cdot', '*')
        s = s.replace('\\times', '*')
        s = s.replace('\\div', '/')
        expr = parse_latex(s)
        return expr
    except Exception:
        return None


def sympy_equal(pred_str: str, gold_str: str) -> bool:
    """Check if two expressions are mathematically equal using sympy."""
    if not SYMPY_AVAILABLE:
        return False
    try:
        pred_expr = latex_to_sympy(pred_str)
        gold_expr = latex_to_sympy(gold_str)
        if pred_expr is None or gold_expr is None:
            return False

        # Try symbolic equality
        if simplify(pred_expr - gold_expr) == 0:
            return True

        # Try numeric evaluation
        try:
            pred_val = complex(N(pred_expr, 10))
            gold_val = complex(N(gold_expr, 10))
            if abs(pred_val - gold_val) < 1e-6:
                return True
            # Relative tolerance for larger numbers
            if abs(gold_val) > 1e-10:
                if abs(pred_val - gold_val) / abs(gold_val) < 1e-4:
                    return True
        except:
            pass
        return False
    except Exception:
        return False


def parse_coordinate(s: str):
    """Parse coordinate pair like (1, 2) or [1, 2]."""
    s = s.strip()
    match = re.match(r'[\[\(]?\s*([^,\[\]\(\)]+)\s*,\s*([^,\[\]\(\)]+)\s*[\]\)]?', s)
    if match:
        try:
            a = float(match.group(1).strip())
            b = float(match.group(2).strip())
            return (a, b)
        except:
            return None
    return None


# Tolerance constants - MUST match rewards/core.py MathReasoningReward
# Training uses rel_tol=1e-5, abs_tol=1e-9
NUMERIC_REL_TOL = 1e-5
NUMERIC_ABS_TOL = 1e-9


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.

    Matches the logic from rewards/core.py MathReasoningReward._extract_numbers
    for consistency between training and baseline computation.

    Supports:
    - Standard decimal numbers: 123, -45.67
    - Fractions: 1/2, 3/4
    - Scientific notation: 1.5e-3, 2E10
    """
    numbers = []

    # Standard decimal numbers
    pattern1 = r'-?\d+\.?\d*'
    for match in re.findall(pattern1, text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    # Fractions (e.g., "1/2", "3/4")
    pattern2 = r'(\d+)\s*/\s*(\d+)'
    for num, den in re.findall(pattern2, text):
        try:
            numbers.append(float(num) / float(den))
        except (ValueError, ZeroDivisionError):
            continue

    # Scientific notation (e.g., "1.5e-3", "2E10")
    pattern3 = r'-?\d+\.?\d*[eE][+-]?\d+'
    for match in re.findall(pattern3, text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def numbers_match(num1: float, num2: float) -> bool:
    """Check if two numbers match within tolerance.

    Uses same tolerance as rewards/core.py MathReasoningReward._numbers_match
    """
    return math.isclose(num1, num2, rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)


def any_number_matches(pred_text: str, gold_text: str) -> bool:
    """
    Check if ANY number in prediction matches ANY number in gold.

    This matches the training reward logic from MathReasoningReward._evaluate_correctness:
    - Extract all numbers from both texts
    - Return True if any pair matches within tolerance

    This is important for consistency: during training, the model gets reward=1.0
    if any extracted number matches, so baseline computation should use the same logic.
    """
    pred_nums = extract_numbers(pred_text)
    gold_nums = extract_numbers(gold_text)

    if not pred_nums or not gold_nums:
        return False

    for pred_num in pred_nums:
        for gold_num in gold_nums:
            if numbers_match(pred_num, gold_num):
                return True

    return False


def answers_equal(pred: str, gold: str) -> bool:
    """
    Comprehensive answer comparison for MATH-style problems.

    Tries multiple matching strategies (in order):
    1. Exact string match (normalized)
    2. Numeric comparison with tolerance (matches training)
    3. ANY number in pred matches ANY number in gold (matches training reward)
    4. Coordinate pair comparison
    5. Sympy symbolic/numeric equality (bonus robustness)

    The tolerance and number extraction logic MUST match rewards/core.py
    MathReasoningReward for consistency between training and baseline computation.
    """
    pred = pred.strip()
    gold = gold.strip()

    # Normalize and compare
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Case-insensitive, whitespace-normalized
    pred_clean = re.sub(r'\s+', '', pred_norm.lower())
    gold_clean = re.sub(r'\s+', '', gold_norm.lower())
    if pred_clean == gold_clean:
        return True

    # Numeric comparison (direct)
    try:
        pred_num = float(pred_norm.replace(',', ''))
        gold_num = float(gold_norm.replace(',', ''))
        if numbers_match(pred_num, gold_num):
            return True
    except:
        pass

    # ANY number matches (training reward logic)
    # This is critical for matching training behavior
    if any_number_matches(pred, gold):
        return True

    # Coordinate comparison
    pred_coord = parse_coordinate(pred_norm)
    gold_coord = parse_coordinate(gold_norm)
    if pred_coord is not None and gold_coord is not None:
        if (numbers_match(pred_coord[0], gold_coord[0]) and
            numbers_match(pred_coord[1], gold_coord[1])):
            return True

    # Sympy comparison (most robust but slowest)
    if sympy_equal(pred, gold):
        return True
    if sympy_equal(pred_norm, gold_norm):
        return True

    return False


# ============================================================================
# Dataset formatters
# ============================================================================

def format_gsm8k(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format GSM8K example."""
    question = example["question"]
    answer = example["answer"]

    # Extract numeric answer
    if "####" in answer:
        gold = answer.split("####")[-1].strip().replace(",", "")
    else:
        gold = answer.strip().replace(",", "")

    # Format prompt
    # Note: enable_thinking=False disables Qwen3's thinking mode
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
            {"role": "user", "content": f"Solve this problem and put your final numeric answer after '####':\n\n{question}"}
        ]
        # Disable thinking mode for Qwen3 models (no <think> tags)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    key = f"gsm8k_{idx}"
    return prompt, gold, key


def format_math(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format MATH/MATH-500 example."""
    problem = example.get("problem", example.get("question", ""))
    solution = example.get("solution", example.get("answer", ""))

    # Extract answer from \boxed{} if present
    match = re.search(r"\\boxed\{([^}]+)\}", solution)
    if match:
        gold = match.group(1).strip()
    else:
        gold = solution.strip()

    # Format prompt
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step."},
            {"role": "user", "content": f"Solve this problem and put your final answer within \\boxed{{}}:\n\n{problem}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    key = f"math_{idx}"
    return prompt, gold, key


def format_mbpp(example: Dict, idx: int, tokenizer, use_chat: bool = True) -> Tuple[str, str, str]:
    """Format MBPP example with explicit function signature."""
    prompt_text = example.get("prompt", example.get("text", ""))
    test_cases = example.get("test_list", example.get("test", []))

    # Extract expected function name from test assertions
    # e.g., "assert first_repeated_char(...)" -> "first_repeated_char"
    func_name = "solution"
    if test_cases:
        first_test = test_cases[0] if isinstance(test_cases, list) else str(test_cases)
        match = re.search(r'assert\s+(\w+)\s*\(', first_test)
        if match:
            func_name = match.group(1)

    # Format prompt with explicit function signature (like bolt script)
    user_content = f"""{prompt_text}

IMPLEMENT THE FUNCTION:
def {func_name}(...):
    # Your implementation here

REQUIREMENTS:
- Use this exact function name: {func_name}
- Return a value, do not print
- Use only ASCII characters
- Output ONLY Python code
- No markdown, no explanations, no examples"""

    # Note: enable_thinking=False disables Qwen3's thinking mode (no <think> tags)
    if use_chat:
        messages = [
            {"role": "user", "content": user_content}
        ]
        # Disable thinking mode for Qwen3 models (no <think> tags)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback for tokenizers that don't support enable_thinking
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Write a Python function:\n{prompt_text}"

    # Gold is the test cases (for reference)
    gold = json.dumps(test_cases) if isinstance(test_cases, list) else str(test_cases)

    # Use task_id if available (like bolt baseline format: mbpp_train_Mbpp/{task_id})
    task_id = example.get("task_id", idx)
    # evalplus task_id is already "Mbpp/102", HF uses int like 601
    if isinstance(task_id, str) and task_id.startswith("Mbpp/"):
        key = f"mbpp_train_{task_id}"  # -> mbpp_train_Mbpp/102
    else:
        key = f"mbpp_train_Mbpp/{task_id}"  # -> mbpp_train_Mbpp/601
    return prompt, gold, key


DATASET_FORMATTERS = {
    "gsm8k": format_gsm8k,
    "math": format_math,
    "mbpp": format_mbpp,
}


# ============================================================================
# Answer extractors/validators
# ============================================================================

def extract_gsm8k_answer(text: str) -> str:
    """Extract answer from GSM8K-style response.

    Tries multiple formats:
    1. #### format (standard GSM8K)
    2. \\boxed{} format (Qwen3 and other models) - uses robust brace matching
    3. Last number in response (fallback)
    """
    # Try #### format first
    if "####" in text:
        parts = text.split("####")
        answer = parts[-1].strip().split()[0] if parts[-1].strip() else ""
        return answer.replace(",", "")

    # Try \boxed{} format with robust extraction (handles nested braces)
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.replace(",", "")

    # Fallback: try to find last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_math_answer(text: str) -> str:
    """Extract answer from MATH-style response.

    Uses robust brace-matching extract_boxed_answer, with fallback to last number.
    """
    # Try \boxed{} with robust extraction
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # Fallback: try to find last number in text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return ""


def validate_gsm8k(pred: str, gold: str) -> bool:
    """Validate GSM8K answer with numeric tolerance.

    Uses shared math_grading module for consistency across eval/training/baseline.
    """
    return grade_gsm8k_answer(pred, gold)


def validate_math(pred: str, gold: str) -> bool:
    """Validate MATH answer using three-tier grading.

    Uses shared math_grading module with:
    1. Official MATH dataset normalization
    2. PRM800K normalization
    3. SymPy symbolic equivalence
    """
    return grade_math_answer(pred, gold)


def _strip_markdown(code: str) -> str:
    """Extract code from markdown code blocks."""
    # Remove ```python ... ``` blocks
    code = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), code, flags=re.S)
    code = re.sub(r"^```(\w+)?\s*|\s*```$", "", code.strip(), flags=re.M)
    return code


def _ensure_entrypoint(code: str, expected_name: str) -> str:
    """Ensure the expected function name exists, adding alias if needed."""
    # If expected function already exists, return as-is
    if re.search(rf"\bdef\s+{re.escape(expected_name)}\s*\(", code):
        return code

    # Find any function definition and alias it
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        actual = m.group(1)
        return code + f"\n\n# alias for test\n{expected_name} = {actual}\n"

    # No function found - add stub that raises
    return f"def {expected_name}(*args, **kwargs):\n    raise NotImplementedError\n\n" + code


def _flexible_signature_shim(expected_name: str) -> str:
    """Handle argument count mismatches gracefully.

    Note: Disabled for now as SafeCodeExecutor runs in restricted namespace.
    """
    # Return empty string - the entrypoint aliasing should be sufficient
    return ""


def validate_mbpp_with_execution(pred: str, gold: str) -> float:
    """
    Validate MBPP code by executing against test cases.

    Args:
        pred: Generated code
        gold: JSON string containing test cases

    Returns:
        Float score: fraction of tests passed (0.0 to 1.0)
    """
    try:
        from aligntune.eval.safe_executor import SafeCodeExecutor

        # Parse test cases from gold
        test_cases_raw = json.loads(gold) if isinstance(gold, str) else gold

        if not test_cases_raw:
            return 0.0

        # Extract expected function name from first test
        expected_name = "solution"
        first_test = test_cases_raw[0] if test_cases_raw else ""
        match = re.search(r'assert\s+(\w+)\s*\(', first_test)
        if match:
            expected_name = match.group(1)

        # Execute code with test assertions
        executor = SafeCodeExecutor(timeout=5)

        # Extract and prepare code
        code = executor.extract_code(pred)
        if not code:
            code = _strip_markdown(pred)

        # Ensure function name matches
        code = _ensure_entrypoint(code, expected_name)

        # Run each test case separately to count passes
        tests_passed = 0
        for tc in test_cases_raw:
            if isinstance(tc, str):
                test_code = code + "\n\n" + tc + "\n"
                result = executor.execute(test_code, test_cases=[])
                if result.success and not result.error:
                    tests_passed += 1

        # Return fractional score
        return float(tests_passed / len(test_cases_raw)) if test_cases_raw else 0.0

    except Exception as e:
        return 0.0


def validate_mbpp_stub(pred: str, gold: str) -> bool:
    """
    STUB: MBPP validation requires code execution.

    WARNING: This returns False always. For actual MBPP baseline computation,
    use evalplus or another code execution framework.
    """
    logger.warning(
        "MBPP validation requires code execution. "
        "Use evalplus for accurate baselines. Returning False."
    )
    return False


ANSWER_EXTRACTORS = {
    "gsm8k": extract_gsm8k_answer,
    "math": extract_math_answer,
    "mbpp": lambda x: x,  # MBPP needs code execution, not extraction
}

def validate_exact_match(pred: str, gold: str) -> bool:
    """Simple exact match validator (fallback)."""
    return pred.strip().lower() == gold.strip().lower()


VALIDATORS = {
    "gsm8k": validate_gsm8k,
    "math": validate_math,
    "mbpp": validate_mbpp_with_execution,  # Uses SafeCodeExecutor
}


# ============================================================================
# Prompt key generation
# ============================================================================

def make_prompt_key(prompt: str) -> str:
    """Create stable hash-based key for prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


# ============================================================================
# Main computation
# ============================================================================

def compute_baselines(
    llm,
    prompts: List[str],
    golds: List[str],
    keys: List[str],
    dataset_type: str,
    num_samples: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Dict[str, float]:
    """
    Compute v̂(x) for each prompt by generating samples and scoring.

    Returns:
        Dictionary mapping prompt keys to v̂ values
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        n=num_samples,
    )

    logger.info(f"Generating {num_samples} samples for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Get extractor and validator for dataset
    extractor = ANSWER_EXTRACTORS.get(dataset_type, lambda x: x.strip())
    validator = VALIDATORS.get(dataset_type, validate_exact_match)

    baselines = {}

    for i, output in enumerate(tqdm(outputs, desc="Scoring")):
        key = keys[i]
        gold = golds[i]

        # Score each sample
        total_score = 0.0
        for completion in output.outputs:
            pred = extractor(completion.text)
            result = validator(pred, gold)
            # Handle both boolean (gsm8k, math) and float (mbpp) returns
            if isinstance(result, bool):
                total_score += 1.0 if result else 0.0
            else:
                total_score += float(result)

        # Compute v̂ = mean score across samples
        v_hat = total_score / num_samples
        baselines[key] = v_hat

    return baselines


def main():
    parser = argparse.ArgumentParser(
        description="Precompute baseline v̂ values for BOLT warm-start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--lora_path", help="Optional LoRA adapter path")
    parser.add_argument("--lora_name", default="adapter", help="Name for LoRA adapter")

    # Dataset arguments
    parser.add_argument("--dataset", choices=["gsm8k", "math", "mbpp", "custom"],
                        default="gsm8k", help="Dataset type")
    parser.add_argument("--dataset_path", help="Path for custom dataset (JSONL)")
    parser.add_argument("--dataset_split", default="train", help="Dataset split")
    parser.add_argument("--max_prompts", type=int, help="Limit number of prompts")

    # Prompt formatting
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                        help="Use chat template formatting")
    parser.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")

    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")

    # Output
    parser.add_argument("--output", required=True, help="Output JSON path")

    # Runtime
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", "-y", action="store_true",
                        help="Skip confirmation prompts")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BOLT Baseline Precomputation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    if args.lora_path:
        logger.info(f"LoRA: {args.lora_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Output: {args.output}")
    if SYMPY_AVAILABLE:
        logger.info("Sympy: available (robust math validation enabled)")
    else:
        logger.info("Sympy: not available (using basic validation)")
    logger.info("=" * 60)

    # MBPP info - code execution enabled
    if args.dataset == "mbpp":
        logger.info("=" * 60)
        logger.info("MBPP baseline computation with SafeCodeExecutor")
        logger.info("Generated code will be executed against test assertions")
        logger.info("Timeout per execution: 5 seconds")
        logger.info("=" * 60)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.dataset == "custom":
        if not args.dataset_path:
            raise ValueError("--dataset_path required for custom dataset")
        with open(args.dataset_path) as f:
            if args.dataset_path.endswith(".jsonl"):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)
    else:
        from datasets import load_dataset
        if args.dataset == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split=args.dataset_split)
        elif args.dataset == "math":
            dataset = load_dataset("hendrycks/competition_math", split=args.dataset_split)
        elif args.dataset == "mbpp":
            if EVALPLUS_AVAILABLE:
                logger.info("Using evalplus MBPP+ dataset (better test coverage)")
                mbpp_plus = get_mbpp_plus()
                # Convert evalplus format to list of dicts
                dataset = []
                for task_id, prob in mbpp_plus.items():
                    # Extract test assertions from prompt
                    tests = []
                    assert_matches = re.findall(r'(assert\s+.+?)(?:\n|$)', prob["prompt"])
                    for assert_stmt in assert_matches:
                        tests.append(assert_stmt.strip())
                    dataset.append({
                        "task_id": task_id,  # e.g., "Mbpp/2"
                        "text": prob["prompt"],
                        "test_list": tests if tests else ["pass"],
                        "entry_point": prob.get("entry_point", ""),
                    })
                logger.info(f"Loaded {len(dataset)} problems from evalplus MBPP+")
            else:
                logger.info("evalplus not available, using HuggingFace MBPP dataset")
                dataset = load_dataset("mbpp", split=args.dataset_split)

    # Limit prompts if specified
    if args.max_prompts and len(dataset) > args.max_prompts:
        indices = np.random.choice(len(dataset), size=args.max_prompts, replace=False)
        if hasattr(dataset, "select"):
            dataset = dataset.select(sorted(indices))
        else:
            dataset = [dataset[i] for i in sorted(indices)]

    # Format prompts
    logger.info(f"Formatting {len(dataset)} prompts...")
    formatter = DATASET_FORMATTERS.get(args.dataset)

    prompts, golds, keys = [], [], []
    for i, example in enumerate(tqdm(dataset, desc="Formatting")):
        prompt, gold, key = formatter(example, i, tokenizer, args.use_chat_template)
        prompts.append(prompt)
        golds.append(gold)
        keys.append(key)

    logger.info(f"Prepared {len(prompts)} prompts")

    # Load model with vLLM
    from vllm import LLM

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }

    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64  # Accommodate larger LoRA ranks

    logger.info("Loading model with vLLM...")
    llm = LLM(**llm_kwargs)

    # Load LoRA if specified
    if args.lora_path:
        logger.info(f"Loading LoRA adapter from {args.lora_path}")
        llm.load_lora_adapter(
            lora_name=args.lora_name,
            lora_path=args.lora_path
        )
        llm.set_lora_adapter(args.lora_name)

    # Compute baselines
    baselines = compute_baselines(
        llm=llm,
        prompts=prompts,
        golds=golds,
        keys=keys,
        dataset_type=args.dataset,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Compute statistics
    v_hats = list(baselines.values())
    mean_v = np.mean(v_hats)
    std_v = np.std(v_hats)

    # Create output
    output = {
        "metadata": {
            "model": args.model,
            "lora_path": args.lora_path,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "num_samples": args.num_samples,
            "n_prompts": len(baselines),
            "mean_v_hat": float(mean_v),
            "std_v_hat": float(std_v),
            "pct_easy": float(100 * sum(1 for v in v_hats if v > 0.8) / len(v_hats)),
            "pct_hard": float(100 * sum(1 for v in v_hats if v < 0.2) / len(v_hats)),
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
        },
        "baselines": baselines,
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("Baseline Computation Complete")
    logger.info("=" * 60)
    logger.info(f"Prompts: {len(baselines)}")
    logger.info(f"Mean v̂: {mean_v:.3f} ± {std_v:.3f}")
    logger.info(f"Easy (v̂ > 0.8): {output['metadata']['pct_easy']:.1f}%")
    logger.info(f"Hard (v̂ < 0.2): {output['metadata']['pct_hard']:.1f}%")
    logger.info(f"Saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
