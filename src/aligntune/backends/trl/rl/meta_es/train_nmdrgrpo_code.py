# train_nmdrgrpo_code.py – Fine-tune language models on MBPP with Neural Mirror Dr GRPO
# Neural Mirror Dr GRPO: Bregman divergence regularization with neural network mirror map + Dr GRPO optimizations
# Dr GRPO improvements: loss_type="dr_grpo" (eliminates length bias), scale_rewards=False (eliminates difficulty bias)
# Supports any causal language model (Qwen, Llama, Mistral, etc.)
# Requirements: pip install transformers accelerate trl peft datasets wandb

import unicodedata
import sys
import tempfile
import subprocess
import time
import re
import inspect
import textwrap
import ast
import wandb
from logging_config import get_adaptive_logging_config, print_logging_config
from neural_mirror_grpo import NeuralMirrorGRPOConfig, NeuralMirrorGRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_dataset, Dataset
from pathlib import Path
from datetime import datetime
import logging
import json
import torch
import os
import argparse

# --- GRADIENT DEBUG ENVIRONMENT SETUP ---
os.environ.pop("UNSLOTH_VLLM_STANDBY", None)  # Remove any Unsloth env vars
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FiniteGradCallback(TrainerCallback):
    """Callback to detect NaN/Inf gradients and stop training"""

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(
                    p.grad).all():
                raise RuntimeError(f"NaN/Inf in grad of {n}")


class NeuralMirrorRegularizationCallback(TrainerCallback):
    """Callback to capture and log Neural Mirror Bregman divergence from training"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Extract Bregman divergence metrics from Neural Mirror GRPO logs
        mirror_metrics = {}
        found_bregman = False

        for key, value in logs.items():
            if "bregman" in key.lower():
                mirror_metrics[f"neural_mirror/{key}"] = value
                found_bregman = True
                # Print Bregman divergence values
                print(f"🔍 BREGMAN DIVERGENCE: {key} = {value:.6f}")
            elif "divergence" in key.lower():
                mirror_metrics[f"neural_mirror/{key}"] = value
                found_bregman = True
                print(f"🔍 NEURAL MIRROR: {key} = {value:.6f}")
            elif "policy" in key.lower() and "loss" in key.lower():
                mirror_metrics[f"policy/{key}"] = value
                print(f"📊 POLICY LOSS: {key} = {value:.6f}")
            elif "entropy" in key.lower():
                mirror_metrics[f"policy/{key}"] = value
                print(f"🎲 ENTROPY: {key} = {value:.6f}")

        # Also check for any metrics that might contain mirror map information
        for key, value in logs.items():
            if any(substr in key.lower()
                   for substr in ["ref", "reference", "baseline", "ratio"]):
                if isinstance(value, (int, float)):
                    mirror_metrics[f"reference/{key}"] = value
                    print(f"📌 REFERENCE METRIC: {key} = {value:.6f}")

        # Print step info when Bregman metrics are found
        if found_bregman:
            print(
                f"⏱️  Step {
                    state.global_step}: Bregman divergence logged to wandb")

        # Log to wandb if we found Bregman metrics
        if mirror_metrics:
            try:
                wandb.log(mirror_metrics)
            except BaseException:
                pass  # wandb might not be available


def strip_markdown(code: str) -> str:
    """Remove markdown code fences, headings, and prose lines"""
    # Remove common code fences and headings
    code = re.sub(
        r"```.*?```",
        lambda m: m.group(0).strip("`"),
        code,
        flags=re.S)
    code = re.sub(r"^```(\w+)?\s*|\s*```$", "", code.strip(), flags=re.M)
    # Drop leading markdown/prose lines
    lines = code.splitlines()
    keep = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith(
                ("#", "def ", "class ", "import ", "from ")) or (
                keep and stripped):
            keep.append(ln)
        elif keep:  # already started code region
            keep.append(ln)
    return "\n".join(keep) if keep else code


# Single-codepoint to ASCII (values may be multi-char; that's fine)
_ASCII_TRANS = {
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",  # dashes/minus
    ord("×"): "*",
    ord("·"): "*",
    ord("“"): '"',
    ord("”"): '"',
    ord("„"): '"',
    ord("’"): "'",
    ord("‘"): "'",
    ord("²"): "**2",
    ord("³"): "**3",
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u200b"): "",  # zero-width space
}


def ascii_only(s: str) -> str:
    # 1) Direct codepoint translation (safe for multi-char replacements)
    s = s.translate(_ASCII_TRANS)
    # 2) Normalize compatibility forms (e.g., weird digits/symbols)
    s = unicodedata.normalize("NFKC", s)
    # 3) Clamp to ASCII so anything unmapped can't poison the parser
    return s.encode("ascii", "ignore").decode("ascii")


def ensure_entrypoint(code: str, expected_name: str) -> str:
    """Ensure the expected function name is defined"""
    if re.search(rf"\bdef\s+{re.escape(expected_name)}\s*\(", code):
        return code
    # alias the first defined function to expected name
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        actual = m.group(1)
        return code + f"\n\n# alias for grader\n{expected_name} = {actual}\n"
    # last resort: stub to avoid NameError (will likely fail tests but not
    # crash)
    return f"def {expected_name}(*args, **kwargs):\n    raise NotImplementedError\n\n" + code


def run_code_in_sandbox(
    code: str, setup_code: str, tests: list, timeout_seconds: int = 5
) -> tuple[bool, int, list[str]]:
    """
    Run code in a subprocess sandbox with timeout.
    Returns: (execution_success, tests_passed, error_messages)
    """
    error_messages = []

    # Block input() to prevent hanging
    input_blocker = "import builtins; builtins.input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no input'))"

    # Create the test script
    test_script = f"""
import sys
import traceback

# Block input
{input_blocker}

# Setup code
{setup_code}

# Generated code
{code}

# Test execution
results = []
for i, test_code in enumerate({repr(tests)}):
    try:
        exec(test_code)
        results.append(f"PASS:{{i}}")
    except Exception as e:
        results.append(f"FAIL:{{i}}:{{type(e).__name__}}:{{str(e)}}")

# Output results
for result in results:
    print(result)
"""

    try:
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script)
            temp_file = f.name

        # Run in subprocess with timeout
        result = subprocess.run([sys.executable,
                                 temp_file],
                                capture_output=True,
                                text=True,
                                timeout=timeout_seconds)

        # Parse output
        tests_passed = 0
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("PASS:"):
                    tests_passed += 1
                elif line.startswith("FAIL:"):
                    parts = line.split(":", 3)
                    if len(parts) >= 4:
                        error_messages.append(
                            f"Test {parts[1]}: {parts[2]} - {parts[3]}")
            execution_success = True
        else:
            execution_success = False
            if result.stderr:
                error_messages.append(
                    f"Execution error: {result.stderr[:200]}")

        return execution_success, tests_passed, error_messages

    except subprocess.TimeoutExpired:
        error_messages.append("Code execution timed out (infinite loop)")
        return False, 0, error_messages
    except Exception as e:
        error_messages.append(f"Sandbox error: {str(e)}")
        return False, 0, error_messages
    finally:
        # Clean up temp file
        try:
            if "temp_file" in locals():
                os.unlink(temp_file)
        except BaseException:
            pass


def flexible_signature_shim(expected_name: str) -> str:
    """Allow function to accept superfluous args that some tests pass"""
    return f"""
try:
    _orig = {expected_name}
    import inspect
    def {expected_name}(*args, **kwargs):
        sig = inspect.signature(_orig)
        params = list(sig.parameters.values())
        # Trim extra positional args if grader passes more than we use
        trimmed = args[:len(params)]
        return _orig(*trimmed, **kwargs)
except Exception:
    pass
"""


def prepare_for_exec(generated: str, expected_name: str) -> str:
    """Sanitize and prepare generated code for execution"""
    # Use extract_python_code instead of strip_markdown (more reliable)
    code = extract_python_code(generated)
    code = ascii_only(code)
    code = ensure_entrypoint(code, expected_name)
    code = code + "\n" + flexible_signature_shim(expected_name)
    # compile gate
    try:
        ast.parse(code)
    except SyntaxError:
        # fall back to just a stub to avoid poisoning reward with SyntaxError
        # noise
        code = f"def {expected_name}(*args, **kwargs):\n    raise SyntaxError('bad submission')\n"
    return textwrap.dedent(code)


def extract_function_info_from_test(test_case: str) -> tuple[str, int]:
    """Extract function name and arity from test assertion"""
    # Look for function calls in assert statements
    match = re.search(r"(\w+)\s*\([^)]*\)", test_case)
    if match:
        func_name = match.group(1)
        # Count arguments by counting commas at the top level only
        call_match = re.search(
            rf"{re.escape(func_name)}\s*\(([^)]*)\)", test_case)
        if call_match:
            args_str = call_match.group(1).strip()
            if not args_str:
                return func_name, 0

            # Count top-level commas (not inside brackets/parens)
            paren_depth = 0
            bracket_depth = 0
            comma_count = 0

            for char in args_str:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "," and paren_depth == 0 and bracket_depth == 0:
                    comma_count += 1

            arg_count = comma_count + 1  # commas + 1 = number of arguments
            return func_name, arg_count
    return None, 0


def extract_python_code(text):
    """Extract clean Python code from text that may contain thinking mode artifacts"""
    # Remove thinking tags
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove /think and /no_think commands
    text = re.sub(r"/think|/no_think", "", text)

    # Look for Python code blocks (```python ... ```)
    python_blocks = re.findall(
        r"```python\s*(.*?)\s*```",
        text,
        flags=re.DOTALL)
    if python_blocks:
        return python_blocks[0].strip()

    # Look for general code blocks (``` ... ```)
    code_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # If no code blocks, return the text stripped of thinking artifacts
    cleaned = text.strip()

    # Remove common non-code prefixes
    prefixes_to_remove = [
        "Here's the solution:",
        "Here's a Python function:",
        "The Python code is:",
        "Solution:",
    ]

    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    return cleaned


def setup_debug_logging(debug_mode: bool, debug_log_dir: str):
    """Setup detailed debug logging for Neural Mirror Dr GRPO training"""
    if not debug_mode:
        return None, None

    # Create debug log directory
    debug_dir = Path(debug_log_dir)
    debug_dir.mkdir(exist_ok=True)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = debug_dir / f"nmdrgrpo_debug_{timestamp}.log"

    logger = logging.getLogger("nmdrgrpo_debug")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, debug_dir


def parse_args():
    """Parse command line arguments for Neural Mirror Dr GRPO training"""
    parser = argparse.ArgumentParser(
        description="Train language models on MBPP using Neural Mirror Dr GRPO")

    # Model arguments
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        type=str,
        help="Model to use")
    parser.add_argument(
        "--output_dir",
        default="./Qwen2-0.5B-NeuralMirrorDrGRPO-MBPP",
        type=str,
        help="Output directory")

    # Training hyperparameters
    parser.add_argument(
        "--max_steps",
        default=1600,
        type=int,
        help="Maximum training steps")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
        help="Batch size per device")
    parser.add_argument(
        "--num_generations",
        default=32,
        type=int,
        help="Generations per prompt")
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate")
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=8,
        type=int,
        help="Gradient accumulation steps")
    parser.add_argument(
        "--max_completion_length",
        default=256,
        type=int,
        help="Max completion length")
    parser.add_argument(
        "--max_prompt_length",
        default=768,
        type=int,
        help="Max prompt length")
    parser.add_argument(
        "--temperature",
        default=0.8,
        type=float,
        help="Generation temperature")
    parser.add_argument(
        "--top_p",
        default=0.9,
        type=float,
        help="Nucleus sampling top_p")

    # Neural Mirror GRPO specific
    parser.add_argument(
        "--mirror_coefficient",
        default=0.0001,
        type=float,
        help="Bregman divergence regularization coefficient")
    parser.add_argument(
        "--mirror_init_scale",
        default=0.01,
        type=float,
        help="Initialization scale for mirror map parameters")
    parser.add_argument(
        "--mirror_seed",
        default=42,
        type=int,
        help="Random seed for mirror map initialization")

    # Mixed precision
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision")

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for training")
    parser.add_argument("--lora_r", default=64, type=int, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha",
        default=64,
        type=int,
        help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout",
        default=0.05,
        type=float,
        help="LoRA dropout")

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        default="nmdrgrpo-code-training",
        type=str,
        help="Wandb project name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name (auto-generated if not provided)")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug logging")
    parser.add_argument(
        "--debug_log_dir",
        default="./debug_logs",
        type=str,
        help="Debug log directory")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Setup debug logging
    debug_logger, debug_dir = setup_debug_logging(
        args.debug_mode, args.debug_log_dir)
    DEBUG_MODE = args.debug_mode

    if DEBUG_MODE and debug_logger:
        debug_logger.info(
            "=== Neural Mirror Dr GRPO Debug Training Started ===")
        debug_logger.info(f"Debug logs will be saved to: {debug_dir}")

    # Configure run naming
    run_name = (
        args.wandb_run_name
        or f"{args.model.split('/')[-1]}-nmdrgrpo-mbpp-mc{args.mirror_coefficient}-seed{args.mirror_seed}-lr{args.learning_rate}-steps{args.max_steps}"
    )

    # Initialize Weights & Biases logging
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                "algorithm": "NeuralMirrorDrGRPO",
                "divergence_type": "neural_mirror",
                "loss_type": "dr_grpo",
                "scale_rewards": False,
                "dataset": "mbpp",
                "mirror_map_neurons": 126,
            },
        )

    print("=" * 80)
    print("Neural Mirror Dr GRPO Code Training on MBPP")
    print("=" * 80)
    print(f"Algorithm: Neural Mirror Dr GRPO (Bregman divergence + GRPO Done Right)")
    print(f"Dr GRPO optimizations:")
    print(f"  - loss_type='dr_grpo': eliminates response-level length bias")
    print(f"  - scale_rewards=False: eliminates question-level difficulty bias")
    print(f"Model: {args.model}")
    print(f"Dataset: MBPP (train split)")
    print(f"Mirror coefficient: {args.mirror_coefficient}")
    print(f"Mirror init scale: {args.mirror_init_scale}")
    print(f"Mirror seed: {args.mirror_seed}")
    print(f"Mirror map: 126 neurons (6 activation types)")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(
        f"Batch size: {
            args.per_device_train_batch_size} x {
            args.gradient_accumulation_steps} = {
                args.per_device_train_batch_size *
            args.gradient_accumulation_steps}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"LoRA: {'Yes' if args.use_lora else 'No'}")
    print(f"Debug mode: {'Yes' if DEBUG_MODE else 'No'}")
    print("=" * 80)

    # Load the MBPP dataset (Mostly Basic Python Problems)
    if debug_logger:
        debug_logger.info("Loading MBPP dataset...")

    # Load MBPP with proper train/test splits to avoid data leakage
    raw_train_dataset = load_dataset(
        "mbpp", split="train")  # 374 problems for training
    # 500 problems for evaluation
    test_dataset = load_dataset("mbpp", split="test")

    if debug_logger:
        debug_logger.info(f"Train split: {len(raw_train_dataset)} problems")
        debug_logger.info(f"Test split: {len(test_dataset)} problems")
        debug_logger.info(
            "Using train split for Neural Mirror Dr GRPO training (avoiding data leakage)")

    if debug_logger:
        debug_logger.info(f"Dataset loaded: {len(raw_train_dataset)} samples")
        debug_logger.info(f"Dataset columns: {raw_train_dataset.column_names}")
        # Log a sample problem
        sample = raw_train_dataset[0]
        debug_logger.info(
            f"Sample problem text: {sample.get('text', 'N/A')[:200]}...")
        debug_logger.info(
            f"Sample has {len(sample.get('test_list', []))} test cases")

    # Build a mapping from index to test data for robust lookup
    test_data_by_id = {}

    for i, item in enumerate(raw_train_dataset):
        # Store test data by index ID
        setup_code = item.get("test_setup_code", "")
        tests = item.get("test_list", [])
        test_data_by_id[i] = {"setup": setup_code or "", "tests": tests or []}

    # Now prepare the dataset for training: rename 'text' column to 'prompt'
    # and add ID
    train_dataset = raw_train_dataset.rename_column("text", "prompt")
    # Add index IDs for robust test data lookup
    train_dataset = train_dataset.add_column(
        "id", list(range(len(train_dataset))))
    # Remove unnecessary columns (keep only 'prompt' and 'id')
    cols_to_keep = ["prompt", "id"]
    cols_to_remove = [
        col for col in train_dataset.column_names if col not in cols_to_keep]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)

    # Reward step counter for progress tracking
    reward_step = 0
    start_time = time.time()

    # Enhanced reward function with negative rewards and variance fallback
    def mbpp_reward_function(completions, **kwargs):
        nonlocal reward_step, start_time
        reward_step += 1

        # Get prompts from kwargs (TRL passes them here)
        prompts = kwargs.get("prompts", [""] * len(completions))
        queries = kwargs.get("queries", [""] * len(completions))

        # Neural Mirror GRPO Fix: The prompts list may be duplicated per generation
        # Try to deduplicate prompts to find actual batch size
        unique_prompts = []
        seen_prompts = set()
        for p in prompts:
            p_str = str(p)
            if p_str not in seen_prompts:
                unique_prompts.append(p)
                seen_prompts.add(p_str)

        # Calculate group size based on unique prompts
        if len(unique_prompts) > 0:
            group_size = len(completions) // len(unique_prompts)
        else:
            group_size = len(completions)  # degenerate case
        group_size = max(1, group_size)

        if DEBUG_MODE:
            print(
                f"🔍 Grouping debug: {
                    len(completions)} completions, {
                    len(prompts)} prompts ({
                    len(unique_prompts)} unique), group_size={group_size}")
            # Debug the first few unique prompts to understand structure
            for i in range(min(3, len(unique_prompts))):
                prompt_str = (unique_prompts[i][:100] if isinstance(
                    unique_prompts[i], str) else str(unique_prompts[i])[:100])
                print(f"  Unique prompt {i}: {prompt_str}...")
            if len(unique_prompts) > 3:
                print(
                    f"  ... and {
                        len(unique_prompts) -
                        3} more unique prompts")

        # Helper: map completion idx -> prompt group index
        def group_of(idx: int) -> int:
            return idx // group_size

        if DEBUG_MODE:
            print(
                f"🎯 Reward function called with {
                    len(completions)} completions")

        rewards = []
        completion_lengths = []  # For ranking fallback

        brevity_bonuses_applied = 0  # Track how many brevity bonuses we give

        for i, completion in enumerate(completions):
            # Extract text content of the completion
            if isinstance(completion, dict) and "content" in completion:
                completion_code = completion["content"]
            else:
                completion_code = str(completion)

            completion_lengths.append(len(completion_code))

            # Extract sample_id using proper per-prompt grouping
            g = group_of(i)  # prompt index within the batch
            prompt_text = unique_prompts[g] if g < len(unique_prompts) else ""
            # Prefer explicit ID from the prompt; else use g
            sample_id = g
            id_match = re.search(r"__SAMPLE_ID:(\d+)__", prompt_text or "")
            if id_match:
                sample_id = int(id_match.group(1))

            if DEBUG_MODE:
                print(
                    f"Sample {
                        i +
                        1}: Generated {
                        len(completion_code)} characters (ID: {sample_id})")
                print(f"🔤 Full prompt sent to model:")
                print(f"{prompt_text}")
                print("=" * 80)
                print(f"💻 Complete model output:")
                print(f"{completion_code}")
                print("=" * 80)

                # Log prompt function signature extraction
                if prompt_text:
                    prompt_func_match = re.search(
                        r"def\s+([A-Za-z_]\w*)\s*\([^)]*\):", prompt_text)
                    if prompt_func_match:
                        prompt_func_name = prompt_func_match.group(1)
                        print(
                            f"🎯 Function name from prompt: {prompt_func_name}")
                    else:
                        print(
                            f"⚠️  Could not extract function name from prompt")

            # Get tests for this sample ID
            tests_info = test_data_by_id.get(sample_id, {})
            setup_code = tests_info.get("setup", "")
            tests = tests_info.get("tests", [])

            if not tests:
                if DEBUG_MODE:
                    print(f"⚠️  No tests found for sample {i + 1}")
                rewards.append(-0.10)  # Negative reward for no tests
                continue

            # Extract expected function name and arity from first test (using
            # proper extraction)
            expected_func, expected_arity = extract_function_info_from_test(
                tests[0])

            if not expected_func:
                if DEBUG_MODE:
                    print(
                        f"⚠️  Could not extract function name from tests for sample {
                            i + 1}")
                rewards.append(-0.10)  # Negative reward for parsing failure
                continue

            if DEBUG_MODE:
                print(
                    f"🎯 Expected function from test: {expected_func} (arity: {expected_arity})")
                print(f"📋 First test string: {tests[0] if tests else 'None'}")
                print(f"🔧 Setup code: {repr(setup_code)}")

                # Check for sample ID mismatch issues
                if prompt_text:
                    prompt_func_match = re.search(
                        r"def\s+([A-Za-z_]\w*)\s*\([^)]*\):", prompt_text)
                    if prompt_func_match:
                        prompt_func_name = prompt_func_match.group(1)
                        if prompt_func_name != expected_func:
                            print(
                                f"🚨 MISMATCH: Prompt has '{prompt_func_name}' but test expects '{expected_func}'")
                        else:
                            print(f"✅ Function names match: {expected_func}")

                # Log the actual dataset sample for this ID
                print(f"📊 Sample ID {sample_id} from test_data_by_id:")
                if sample_id < len(test_data_by_id):
                    print(f"   Has {len(tests)} tests")
                    print(f"   Setup: {repr(setup_code[:100])}...")
                else:
                    print(
                        f"⚠️  Sample ID {sample_id} >= dataset size {
                            len(test_data_by_id)}")

            # Step 1: Contract enforcement with sanitization
            sanitized_code = prepare_for_exec(completion_code, expected_func)
            if DEBUG_MODE:
                print(f"🧹 Sanitized code: {sanitized_code[:200]}...")

            # Step 2: Execute and test with sanitized code using robust sandbox
            execution_success, passed, error_messages = run_code_in_sandbox(
                code=sanitized_code, setup_code=setup_code, tests=tests, timeout_seconds=5)

            # Step 3: Calculate base reward from correctness
            base_reward = float(passed / len(tests)) if tests else 0.0

            # Apply negative rewards for failure cases
            if passed == 0:
                base_reward -= 0.10  # Penalty for zero tests passed
                if DEBUG_MODE:
                    print("❌ No tests passed - applying penalty")

            # Heavy penalty for timeout/crash
            if not execution_success:
                timeout_crash = any(
                    "timed out" in msg for msg in error_messages)
                if timeout_crash:
                    base_reward = -0.25  # Heavy penalty for timeout
                    if DEBUG_MODE:
                        print("❌ Code timed out - heavy penalty")
                else:
                    base_reward = -0.15  # Penalty for other execution failures
                    if DEBUG_MODE:
                        print("❌ Code execution failed - penalty")

            if DEBUG_MODE:
                if execution_success:
                    print("✅ Code executed successfully")
                    print(f"Running {len(tests)} tests...")
                    for j in range(len(tests)):
                        if j < passed:
                            print(f"✅ Test {j + 1} passed")
                        else:
                            print(f"❌ Test {j + 1} failed")

                    # Show error details
                    for error_msg in error_messages:
                        print(f"❌ {error_msg}")
                else:
                    for error_msg in error_messages:
                        print(f"❌ {error_msg}")

                # Log what we're actually executing
                print(f"🔬 What we executed:")
                print(f"   Setup: {repr(setup_code)}")
                print(f"   Code: {repr(sanitized_code[:200])}...")
                print(f"   Tests: {tests[:2]}...")  # First 2 tests only

            # Step 4: Small format penalties (negative-only)
            format_penalties = 0.0

            # Check for banned tokens in raw completion
            banned_tokens = [
                "```",
                "print(",
                "input(",
                "Example:",
                "Explanation:",
                "# Example",
                "# Test"]
            for token in banned_tokens:
                if token in completion_code:
                    format_penalties -= 0.02
                    if DEBUG_MODE:
                        print(
                            f"⚠️  Found banned token '{token}' - small penalty")

            # Check for non-ASCII characters
            try:
                completion_code.encode("ascii")
            except UnicodeEncodeError:
                format_penalties -= 0.03
                if DEBUG_MODE:
                    print("⚠️  Non-ASCII characters found - small penalty")

            # Step 5: Brevity bonus (only when tests pass)
            brevity_bonus = 0.0
            if passed > 0 and execution_success:  # Only reward brevity for working solutions
                if len(completion_code) < 100:  # Very short solutions
                    brevity_bonus = 0.02
                    brevity_bonuses_applied += 1
                    if DEBUG_MODE:
                        print(
                            f"✅ Brief working solution ({
                                len(completion_code)} chars) - tiny bonus")
                elif len(completion_code) < 150:
                    brevity_bonus = 0.01
                    brevity_bonuses_applied += 1
                    if DEBUG_MODE:
                        print(
                            f"✅ Concise working solution ({
                                len(completion_code)} chars) - tiny bonus")
            elif DEBUG_MODE and len(completion_code) < 150:
                print(
                    f"⚠️  Short solution ({
                        len(completion_code)} chars) but no tests passed - no brevity bonus")

            # Step 6: Combine rewards (correctness dominates)
            total_reward = base_reward + format_penalties + brevity_bonus

            rewards.append(total_reward)
            if DEBUG_MODE:
                brevity_str = f"brevity: {
                    brevity_bonus:+.4f}" if brevity_bonus > 0 else "no brevity bonus"
                print(
                    f"🏆 Sample {
                        i + 1} reward: {
                        total_reward:.4f} (base: {
                        base_reward:.4f}, format: {
                        format_penalties:+.4f}, {brevity_str}, {passed}/{
                        len(tests)} tests)")

        # Step 7: Group completions by sample_id for per-group analysis
        # sample_id -> [(idx, completion_text, reward, length), ...]
        groups = {}
        sample_ids = []

        for i in range(len(completions)):
            # Extract sample_id for grouping
            sample_id = i % len(test_data_by_id)  # Fallback
            if i < len(prompts) and prompts[i]:
                id_match = re.search(r"__SAMPLE_ID:(\d+)__", prompts[i])
                if id_match:
                    sample_id = int(id_match.group(1))

            completion_text = str(completions[i])
            if isinstance(
                    completions[i],
                    dict) and "content" in completions[i]:
                completion_text = completions[i]["content"]

            if sample_id not in groups:
                groups[sample_id] = []
                sample_ids.append(sample_id)

            groups[sample_id].append(
                (i, completion_text, rewards[i], completion_lengths[i]))

        # Step 8: Apply duplicate penalty within each group
        duplicate_penalties = [0.0] * len(rewards)

        for sample_id, group_items in groups.items():
            seen_texts = {}  # text -> first_occurrence_idx
            for idx, text, reward, length in group_items:
                if text in seen_texts:
                    # This is a duplicate
                    duplicate_penalties[idx] = -0.02
                    if DEBUG_MODE:
                        first_idx = seen_texts[text]
                        print(
                            f"⚠️  Sample {
                                idx +
                                1} is duplicate of sample {
                                first_idx +
                                1} - penalty -0.02")
                else:
                    seen_texts[text] = idx

        # Apply duplicate penalties
        for i in range(len(rewards)):
            rewards[i] += duplicate_penalties[i]

        # Step 9: Per-group variance analysis and tie-breaking
        per_group_stats = {}
        groups_with_zero_variance = 0
        total_groups = len(groups)

        for sample_id, group_items in groups.items():
            # Extract rewards after duplicate penalty
            group_rewards = [item[2] for item in group_items]
            group_mean = sum(group_rewards) / len(group_rewards)
            group_variance = sum(
                (r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)
            group_std = group_variance**0.5

            per_group_stats[sample_id] = {
                "size": len(group_items),
                "mean": group_mean,
                "std": group_std,
                "unique_texts": len(set(item[1] for item in group_items)),
            }

            if group_std < 1e-6:  # Zero variance in this group
                groups_with_zero_variance += 1
                if DEBUG_MODE:
                    print(
                        f"⚠️  Group {sample_id} has zero reward variance - applying tie-breaker")

                # Apply tie-breaking by code length ranking
                sorted_items = sorted(
                    group_items, key=lambda x: x[3])  # Sort by length
                for rank, (orig_idx, text, reward,
                           length) in enumerate(sorted_items):
                    if len(sorted_items) > 1:
                        tie_breaker = 0.03 * \
                            (len(sorted_items) - rank - 1) / (len(sorted_items) - 1) - 0.015
                        rewards[orig_idx] += tie_breaker
                        if DEBUG_MODE:
                            print(
                                f"Sample {orig_idx + 1}: Added tie-breaker {tie_breaker:+.4f} (rank {rank + 1}/{len(sorted_items)})"
                            )

        # Step 10: Soft clip to [-0.5, 1.0] (preserve negative signals)
        rewards = [max(-0.5, min(1.0, r)) for r in rewards]

        # Calculate overall metrics
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        overall_std = (sum((r - avg_reward) ** 2 for r in rewards) /
                       len(rewards)) ** 0.5 if rewards else 0
        frac_groups_zero_std = groups_with_zero_variance / \
            total_groups if total_groups > 0 else 0

        # Calculate within-group uniqueness
        total_unique = sum(stats["unique_texts"]
                           for stats in per_group_stats.values())
        total_completions = len(completions)
        within_group_unique_frac = total_unique / \
            total_completions if total_completions > 0 else 0.0

        if DEBUG_MODE:
            print(f"📊 Batch metrics:")
            print(f"    Average reward: {avg_reward:.4f}")
            print(f"    Overall reward std: {overall_std:.6f}")
            print(f"    Total groups: {total_groups}")
            print(
                f"    Groups with zero variance: {groups_with_zero_variance} ({
                    frac_groups_zero_std:.2f})")
            print(
                f"    Within-group unique fraction: {within_group_unique_frac:.3f}")
            print(
                f"    Duplicate penalties applied: {sum(1 for p in duplicate_penalties if p < 0)}")
            print(
                f"    Brevity bonuses applied: {brevity_bonuses_applied}/{
                    len(completions)} ({
                    brevity_bonuses_applied / len(completions) * 100:.1f}%)")

            # Log per-group details
            for sample_id, stats in per_group_stats.items():
                print(
                    f"    Group {sample_id}: {
                        stats['size']} completions, {
                        stats['unique_texts']} unique, std={
                        stats['std']:.4f}")

        # Finite reward sanity check
        import math

        for r in rewards:
            if not math.isfinite(r):
                raise RuntimeError(f"Non-finite reward: {r}")

        # Report metrics to wandb if available
        try:
            reward_metrics = {
                "reward/avg": avg_reward,
                "reward/std": overall_std,
                "reward/min": min(rewards) if rewards else 0,
                "reward/max": max(rewards) if rewards else 0,
                "training_health/frac_groups_zero_std": frac_groups_zero_std,
                "training_health/within_group_unique_frac": within_group_unique_frac,
                "training_health/total_groups": total_groups,
                "training_health/groups_with_zero_variance": groups_with_zero_variance,
                "training_health/duplicate_penalties_applied": sum(1 for p in duplicate_penalties if p < 0),
                "training_health/brevity_bonuses_applied": brevity_bonuses_applied,
                "training_health/brevity_bonus_rate": brevity_bonuses_applied / len(completions) if completions else 0,
                "training_health/reward_step": reward_step,
                # Log average per-group stats
                "per_group/avg_group_size": (
                    sum(stats["size"]
                        for stats in per_group_stats.values()) / len(per_group_stats)
                    if per_group_stats
                    else 0
                ),
                "per_group/avg_group_std": (
                    sum(stats["std"] for stats in per_group_stats.values()
                        ) / len(per_group_stats)
                    if per_group_stats
                    else 0
                ),
                "per_group/avg_unique_per_group": (
                    sum(stats["unique_texts"]
                        for stats in per_group_stats.values()) / len(per_group_stats)
                    if per_group_stats
                    else 0
                ),
            }

            # Add timestamp for correlation with Bregman metrics
            reward_metrics["reward/timestamp"] = time.time() - start_time

            wandb.log(reward_metrics)
        except BaseException:
            pass  # wandb might not be available

        return rewards

    # Load the language model with standard transformers
    if debug_logger:
        debug_logger.info(f"Loading model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure PAD token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(
            torch.bfloat16
            if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        ),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Use eager attention for training stability
    )

    # Critical: Enable gradient checkpointing and disable cache for RL training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Configure LoRA using PEFT (optional)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        print(
            f"✅ LoRA applied with rank={
                args.lora_r}, alpha={
                args.lora_alpha}")
        model.print_trainable_parameters()

    # Enable training mode
    model.train()

    if debug_logger:
        debug_logger.info(f"Model loaded successfully")
        debug_logger.info(
            f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        debug_logger.info(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        debug_logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        debug_logger.info(
            f"Chat template available: {
                tokenizer.chat_template is not None}")

    # Apply chat template formatting for instruction-tuned model
    if tokenizer.chat_template is not None:
        if debug_logger:
            debug_logger.info("Applying chat template formatting...")
            debug_logger.info(
                f"Original prompt example: {train_dataset['prompt'][0][:150]}...")

        formatted_prompts = []

        for i, prompt in enumerate(train_dataset["prompt"]):
            # Extract exact function signature from tests (more reliable than
            # code)
            tests = raw_train_dataset[i].get("test_list", [])
            function_name = "your_function"  # fallback
            function_args = []

            if tests:
                # Extract function name and signature from first test
                func_name, arity = extract_function_info_from_test(tests[0])
                if func_name:
                    function_name = func_name

                    # Extract actual parameter names from reference code if
                    # available
                    ref_code = raw_train_dataset[i].get("code", "")
                    func_match = re.search(
                        rf"def\s+{re.escape(function_name)}\s*\(([^)]*)\)", ref_code)
                    if func_match:
                        params_str = func_match.group(1)
                        function_args = [p.strip().split("=")[0].strip()
                                         for p in params_str.split(",") if p.strip()]
                    else:
                        # Generate generic parameter names based on arity
                        function_args = [f"arg{j + 1}" for j in range(arity)]

            # Create precise function signature
            signature = f"def {function_name}({', '.join(function_args)}):"

            # Enhanced prompt with exact signature and constraints
            enhanced_prompt = f"""__SAMPLE_ID:{i}__ {prompt}

IMPLEMENT EXACTLY THIS FUNCTION:

{signature}
    # Your implementation here

REQUIREMENTS:
- Use ONLY this function name: {function_name}
- Accept exactly these parameters: {', '.join(function_args) if function_args else 'none'}
- Return a value, do not print
- Use only ASCII characters
- Output ONLY Python code
- No markdown, no explanations, no examples"""

            # Debug: Print the enhanced prompt for all samples in debug mode
            if debug_logger and i < 2:
                debug_logger.info(f"🔧 Enhanced prompt {i + 1}:")
                debug_logger.info(f"   Function signature: {signature}")
                debug_logger.info(f"   Full prompt: {enhanced_prompt}")
                debug_logger.info("-" * 50)

            messages = [{"role": "user", "content": enhanced_prompt}]
            # Apply chat template (disable thinking mode if supported)
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    # Disable thinking mode
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                # Fallback if enable_thinking parameter not supported
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_prompt)

            # Log first few formatted examples
            if debug_logger and i < 2:
                debug_logger.info(
                    f"Formatted prompt example {i + 1}: {formatted_prompt[:200]}...")

        # Update the dataset with formatted prompts
        train_dataset = Dataset.from_dict({"prompt": formatted_prompts})

        if debug_logger:
            debug_logger.info(
                f"Chat template formatting complete. Dataset now has {
                    len(train_dataset)} formatted prompts.")
    else:
        if debug_logger:
            debug_logger.warning("No chat template found! Using raw prompts.")

    # Adaptive logging configuration based on training length
    logging_steps, save_steps, save_total_limit = get_adaptive_logging_config(
        args.max_steps)
    print_logging_config(
        args.max_steps,
        logging_steps,
        save_steps,
        save_total_limit)

    # Configure Neural Mirror Dr GRPO training parameters
    if debug_logger:
        debug_logger.info(
            "Configuring Neural Mirror Dr GRPO training parameters...")

    generation_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )  # must be divisible by num_generations

    nmgrpo_config = NeuralMirrorGRPOConfig(
        # Output directory for logs and model checkpoints
        output_dir=args.output_dir,
        run_name=run_name,
        # Neural Mirror GRPO specific settings
        divergence_type="neural_mirror",
        mirror_coefficient=args.mirror_coefficient,
        mirror_init_scale=args.mirror_init_scale,
        mirror_seed=args.mirror_seed,
        beta=0.0,  # Disable KL regularization
        # Dr GRPO specific settings
        loss_type="dr_grpo",  # Use Dr GRPO loss (eliminates length bias)
        scale_rewards=False,
        # Do not scale rewards (eliminates difficulty bias)
        # Training hyperparameters
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Improved numerics for gradient stability
        max_grad_norm=1.0,  # Gradient clipping for stability
        adam_epsilon=1e-6,  # Stable Adam epsilon
        adam_beta1=0.9,
        adam_beta2=0.95,
        # Mixed precision and acceleration
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available()
        # Only enable fp16 on CUDA
        and not (args.bf16 and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        # Logging settings
        logging_steps=logging_steps,
        # Empty list disables all reporting
        report_to=["wandb"] if not args.no_wandb else [],
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy="steps",
        # Memory optimization settings
        sync_ref_model=False,  # Disable reference model syncing to save memory
        remove_unused_columns=False,
        # Generation parameters for guaranteed diversity
        temperature=args.temperature,
        top_p=args.top_p,
        # Neural Mirror GRPO specifics
        epsilon=0.2,  # PPO-style clip
        tf32=torch.cuda.is_available()
        and not (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    )

    print(f"\n🚀 Neural Mirror Dr GRPO Training Configuration:")
    print(f"   Divergence type: {nmgrpo_config.divergence_type}")
    print(f"   Mirror coefficient: {nmgrpo_config.mirror_coefficient}")
    print(f"   Mirror init scale: {nmgrpo_config.mirror_init_scale}")
    print(f"   Mirror seed: {nmgrpo_config.mirror_seed}")
    print(f"   Mirror map neurons: 126 (21 per activation type)")
    print(f"   Beta (KL): {nmgrpo_config.beta} (disabled)")
    print(
        f"   Loss type: {
            nmgrpo_config.loss_type} (Dr GRPO - eliminates length bias)")
    print(
        f"   Scale rewards: {
            nmgrpo_config.scale_rewards} (eliminates difficulty bias)")
    print(f"   Max grad norm: {nmgrpo_config.max_grad_norm}")
    print(
        f"   Expected debug: [NEURAL MIRROR] + [MIRROR DEBUG] messages during training")
    print()

    if debug_logger:
        debug_logger.info("Neural Mirror Dr GRPO Configuration:")
        debug_logger.info(f"  Output directory: {nmgrpo_config.output_dir}")
        debug_logger.info(
            f"  Batch size per device: {
                nmgrpo_config.per_device_train_batch_size}")
        debug_logger.info(
            f"  Gradient accumulation steps: {
                nmgrpo_config.gradient_accumulation_steps}")
        debug_logger.info(
            f"  Effective batch size: {
                nmgrpo_config.per_device_train_batch_size *
                nmgrpo_config.gradient_accumulation_steps}")
        debug_logger.info(f"  Learning rate: {nmgrpo_config.learning_rate}")
        debug_logger.info(f"  Max steps: {nmgrpo_config.max_steps}")
        debug_logger.info(
            f"  Generations per prompt: {
                nmgrpo_config.num_generations}")
        debug_logger.info(
            f"  Max completion length: {
                nmgrpo_config.max_completion_length}")
        debug_logger.info(
            f"  Max prompt length: {
                nmgrpo_config.max_prompt_length}")
        debug_logger.info(
            f"  Mixed precision: BF16={
                nmgrpo_config.bf16}, FP16={
                nmgrpo_config.fp16}")
        debug_logger.info(
            f"  Gradient checkpointing: {
                nmgrpo_config.gradient_checkpointing}")
        debug_logger.info(f"  Loss type: {nmgrpo_config.loss_type}")
        debug_logger.info(f"  Scale rewards: {nmgrpo_config.scale_rewards}")

    # Initialize the Neural Mirror Dr GRPO trainer with model, dataset, and
    # reward function
    if debug_logger:
        debug_logger.info("Initializing Neural Mirror Dr GRPO trainer...")

    # Use NeuralMirrorGRPOTrainer
    trainer = NeuralMirrorGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=mbpp_reward_function,
        args=nmgrpo_config,
        train_dataset=train_dataset,
    )

    # Add NaN detection callback
    trainer.add_callback(FiniteGradCallback())

    # Add Neural Mirror regularization tracking callback
    trainer.add_callback(NeuralMirrorRegularizationCallback())

    # Configure additional generation parameters for diversity
    generation_config_applied = False

    if hasattr(
            trainer,
            "generation_config") and trainer.generation_config is not None:
        trainer.generation_config.do_sample = True
        trainer.generation_config.top_k = 50
        trainer.generation_config.repetition_penalty = 1.05
        trainer.generation_config.min_new_tokens = 32  # avoid micro-stubs
        generation_config_applied = True
        if debug_logger:
            debug_logger.info(
                "Applied generation parameters to trainer.generation_config")

    if hasattr(
            trainer.model,
            "generation_config") and trainer.model.generation_config is not None:
        trainer.model.generation_config.do_sample = True
        trainer.model.generation_config.top_k = 50
        trainer.model.generation_config.repetition_penalty = 1.05
        trainer.model.generation_config.min_new_tokens = 32  # avoid micro-stubs
        generation_config_applied = True

    # Also try to set on the base model
    if (
        hasattr(trainer.model, "model")
        and hasattr(trainer.model.model, "generation_config")
        and trainer.model.model.generation_config is not None
    ):
        trainer.model.model.generation_config.do_sample = True
        trainer.model.model.generation_config.top_k = 50
        trainer.model.model.generation_config.repetition_penalty = 1.05
        trainer.model.model.generation_config.min_new_tokens = 32
        generation_config_applied = True

    if not generation_config_applied and debug_logger:
        debug_logger.warning(
            "⚠️  Could not find generation_config to apply diversity parameters!")

    if debug_logger:
        debug_logger.info(
            "Neural Mirror Dr GRPO trainer initialized successfully")
        debug_logger.info(f"Training dataset size: {len(train_dataset)}")
        debug_logger.info(
            f"Mirror module parameters: {sum(p.numel() for p in trainer.mirror_module.buffers())}")
        debug_logger.info("Starting Neural Mirror Dr GRPO training...")
        debug_logger.info(f"{'=' * 80}")
        debug_logger.info(
            "TRAINING BEGINS - Watch for reward evaluations and [MIRROR DEBUG] messages!")
        debug_logger.info(f"{'=' * 80}")

    print("🚀 Starting Neural Mirror Dr GRPO training on MBPP...")
    print(
        "Expected output: Code accuracy + [NEURAL MIRROR] initialization + [MIRROR DEBUG] Bregman divergence analysis"
    )

    # Start training
    try:
        trainer.train()
        if debug_logger:
            debug_logger.info("Training completed successfully!")
    except Exception as e:
        if debug_logger:
            debug_logger.error(f"Training failed with error: {e}")
        raise

    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save mirror map parameters for analysis/ES
    mirror_params_path = os.path.join(args.output_dir, "mirror_map_params.pt")
    torch.save(
        {
            "v": trainer.mirror_module.v,
            "w": trainer.mirror_module.w,
            "b": trainer.mirror_module.b,
            "a": trainer.mirror_module.a,
            "c": trainer.mirror_module.c,
            "seed": args.mirror_seed,
            "init_scale": args.mirror_init_scale,
        },
        mirror_params_path,
    )
    print(
        f"✅ Done. Neural Mirror Dr GRPO model and mirror map parameters saved to: {
            args.output_dir}")

    # End the W&B run after training
    if not args.no_wandb:
        wandb.finish()

    if debug_logger:
        debug_logger.info(
            "=== Neural Mirror Dr GRPO Debug Training Completed ===")
        debug_logger.info(f"All debug logs saved to: {debug_dir}")

    # Final summary
    print("\n" + "=" * 80)
    print("NEURAL MIRROR DR GRPO CODE TRAINING COMPLETED")
    print("=" * 80)
    print(f"Algorithm: Neural Mirror Dr GRPO with 126-neuron mirror map")
    print(f"Dr GRPO optimizations:")
    print(f"  - loss_type='dr_grpo': constant normalization (eliminates length bias)")
    print(f"  - scale_rewards=False: no std scaling (eliminates difficulty bias)")
    print(f"Mirror coefficient: {args.mirror_coefficient}")
    print(f"Training completed: {args.max_steps} steps")
    print(f"Model saved to: {args.output_dir}")
    print(f"Mirror map parameters saved to: {mirror_params_path}")
    print(f"Check logs above for:")
    print(f"  - Code accuracy progress")
    print(f"  - [NEURAL MIRROR] initialization messages")
    print(f"  - [MIRROR DEBUG] Bregman divergence analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()
