"""
vLLM-based Fast Evaluator for ES Meta-Learning

This module provides fast value function V(π) computation using vLLM batched inference.
Evaluates trained policy on MBPP validation set (75 problems) with 5× speedup.
"""

import os
import sys
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

# Import execution functions from train_nmgrpo_code.py
# Add these functions to vllm_evaluator.py to make it self-contained
# Remove the import line and add these implementations instead

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
import unicodedata


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
    """Convert string to ASCII-only, replacing common Unicode characters."""
    # 1) Direct codepoint translation (safe for multi-char replacements)
    s = s.translate(_ASCII_TRANS)
    # 2) Normalize compatibility forms (e.g., weird digits/symbols)
    s = unicodedata.normalize("NFKC", s)
    # 3) Clamp to ASCII so anything unmapped can't poison the parser
    return s.encode("ascii", "ignore").decode("ascii")


def ensure_entrypoint(code: str, expected_name: str) -> str:
    """Ensure the expected function name is defined."""
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


def flexible_signature_shim(expected_name: str) -> str:
    """Allow function to accept superfluous args that some tests pass."""
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


def extract_python_code(text: str) -> str:
    """Extract clean Python code from text that may contain thinking mode artifacts."""
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


def prepare_for_exec(generated: str, expected_name: str) -> str:
    """Sanitize and prepare generated code for execution."""
    # Use extract_python_code to strip markdown/prose
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
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

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


# ============================================================================
# VLLM CONFIGURATION AND UTILITIES (copied from llm-test-bench)
# ============================================================================

DEFAULT_SYSTEM_NO_REASONING = (
    "You are a helpful Python coding assistant. "
    "Output ONLY the final code solution.\n"
    "Rules:\n"
    "1) Do NOT include explanations or thinking steps.\n"
    "2) Return Python code in a single fenced block like:\n"
    "```python\n<code here>\n```\n"
    "3) If tests or function signature are given, satisfy them exactly.\n"
)


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
    max_new_tokens: int = 512
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

    def _strip_code(self, text: str) -> str:
        """Extract code block from generated text."""
        # Use the same extraction logic as training
        return extract_python_code(text)

    def generate_code_batch(self,
                            prompts: List[str],
                            n: int = 1,
                            system: Optional[str] = None) -> List[List[str]]:
        """Generate code for multiple prompts in a single batch for much faster evaluation."""
        from vllm import SamplingParams

        if not prompts:
            return []

        # Format as chat messages for all prompts
        if system is None:
            system = DEFAULT_SYSTEM_NO_REASONING

        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
                prompt_results.append(self._strip_code(generated_text))
            all_results.append(prompt_results)

        return all_results


# ============================================================================
# VALUE FUNCTION COMPUTATION
# ============================================================================


def build_mbpp_prompt(problem: dict) -> str:
    """Build prompt for MBPP problem."""
    prompt = f"{problem['text']}\n\nYour code should pass these tests:\n"
    for test in problem["test_list"]:
        prompt += f"{test}\n"
    return prompt.strip()


def evaluate_mbpp_solution(code: str, problem: dict,
                           timeout: int = 5) -> tuple[bool, int, int, List[str]]:
    """
    Evaluate a single MBPP solution using sandbox execution.

    Args:
        code: Generated code to evaluate
        problem: MBPP problem dict with keys: task_id, text, test_list, test_setup_code
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (execution_success, tests_passed, total_tests, error_messages)
    """
    # Extract expected function name from first test
    test_list = problem.get("test_list", [])
    if not test_list:
        return False, 0, 0, ["No tests available"]

    # Simple regex to extract function name from assert statement
    first_test = test_list[0]
    match = re.search(r"assert\s+(\w+)\s*\(", first_test)
    if not match:
        return False, 0, 0, ["Could not extract function name from test"]

    expected_name = match.group(1)

    # Prepare code for execution
    prepared_code = prepare_for_exec(code, expected_name)

    # Get setup code
    setup_code = problem.get("test_setup_code", "")

    # Run in sandbox
    execution_success, tests_passed, error_messages = run_code_in_sandbox(
        prepared_code, setup_code, test_list, timeout_seconds=timeout
    )

    total_tests = len(test_list)

    return execution_success, tests_passed, total_tests, error_messages


def compute_value_function_vllm(
    checkpoint_dir: str,
    validation_dataset: List[dict],
    base_model: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 512,
    timeout: int = 5,
    k: int = 1,
    temperature: float = 0.8,
    verbose: bool = False,
) -> float:
    """
    Compute value function V(π) = mbpp_pass@k on validation set using vLLM.

    This is the fitness function for ES meta-learning.
    Uses vLLM batched inference for fast evaluation.

    Args:
        checkpoint_dir: Path to LoRA checkpoint directory
        validation_dataset: List of MBPP problem dicts (typically 75 problems)
        base_model: Base model identifier
        max_new_tokens: Maximum tokens to generate per problem
        timeout: Execution timeout per problem in seconds
        k: Number of samples per problem for pass@k metric (default: 1)
        temperature: Sampling temperature for k>1 (default: 0.8, ignored when k=1)
        verbose: Print detailed evaluation progress

    Returns:
        pass@k metric (fraction of problems with at least 1 passing solution in k attempts)
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 Computing V(π) = pass@{k} for checkpoint: {checkpoint_dir}")
    print(
        f"📊 Validation set: {
            len(validation_dataset)} problems × {k} samples")
    print(f"{'=' * 60}\n")

    # Initialize vLLM with LoRA checkpoint
    # For pass@k with k>1, use specified temperature for diversity
    # For pass@1, use greedy decoding (temperature=0.0)
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
            f"🎲 Using sampling with temperature={eval_temperature} for pass@{k}")

    runner = VLLMRunner(cfg)

    try:
        # Build prompts for all validation problems
        prompts = [build_mbpp_prompt(problem)
                   for problem in validation_dataset]

        # Batch generate k solutions per problem
        print(
            f"⚡ Generating {
                len(prompts)} prompts × {k} samples = {
                len(prompts) *
                k} total generations")
        all_completions = runner.generate_code_batch(prompts, n=k)

        # Evaluate each problem with k attempts
        problems_solved = 0
        total_problems = len(validation_dataset)

        for i, (problem, completions) in enumerate(
                zip(validation_dataset, all_completions)):
            # Try all k completions, check if ANY passes all tests
            problem_solved = False

            for attempt_idx, code in enumerate(completions):
                # Evaluate with sandbox execution
                exec_success, tests_passed, total_tests, errors = evaluate_mbpp_solution(
                    code, problem, timeout=timeout)

                # Problem is solved if all tests pass
                if exec_success and tests_passed == total_tests:
                    problem_solved = True
                    if verbose:
                        print(f"  ✅ Solved on attempt {attempt_idx + 1}/{k}")
                    break  # Found a passing solution, stop checking

            if problem_solved:
                problems_solved += 1

            if verbose:
                status = "✅ PASS" if problem_solved else "❌ FAIL"
                print(
                    f"  [{i + 1}/{total_problems}] {status} - " f"task_id={problem.get('task_id', 'N/A')}")

        # Compute pass@k metric
        pass_at_k = problems_solved / total_problems

        print(f"\n{'=' * 60}")
        print(f"📈 V(π) = mbpp_pass@{k} = {pass_at_k:.4f}")
        print(
            f"   ({problems_solved}/{total_problems} problems solved with {k} attempts)")
        print(f"{'=' * 60}\n")

        return pass_at_k

    finally:
        # Clean up vLLM resources
        runner.cleanup()
