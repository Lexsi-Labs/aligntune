"""
TRL Counterfactual GRPO Trainer with Custom Importance Weighting

This module provides a TRL backend for Counterfactual Group Relative Policy Optimization,
integrating the custom CounterfactualGRPOTrainer with reward structures for both
math (GSM8K) and code (MBPP) training.
"""

import logging
import time
import yaml
import math
import ast
import re
import os
import sys
import subprocess
import tempfile
import textwrap
import unicodedata
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np

from finetunehub.core.rl.trainer_base import TrainerBase
from finetunehub.core.rl.config import UnifiedConfig
from finetunehub.core.rl.registries import DatasetRegistry, RewardRegistry
from finetunehub.core.rl.caching import DatasetCache
from finetunehub.core.rl.sample_logger import generate_and_log_samples
from finetunehub.core.sft.evaluator import EnhancedEvaluator
from finetunehub.utils.math_grading import extract_math_gold, extract_math_answer, grade_math_answer

logger = logging.getLogger(__name__)


# =============================================================================
# CODE EXECUTION HELPERS (for MBPP)
# Aligned with vanilla MBPP training script (train_grpo.py)
# =============================================================================

# Single-codepoint to ASCII translation
# Using Unicode escapes to avoid linter corruption of special characters
_ASCII_TRANS = {
    0x2013: '-',   # en-dash
    0x2014: '-',   # em-dash
    0x2212: '-',   # minus sign
    0x00D7: '*',   # multiplication sign
    0x00B7: '*',   # middle dot
    0x201C: '"',   # left double quotation
    0x201D: '"',   # right double quotation
    0x201E: '"',   # double low-9 quotation
    0x2018: "'",   # left single quotation
    0x2019: "'",   # right single quotation
    0x00B2: '**2', # superscript 2
    0x00B3: '**3', # superscript 3
    0x00A0: ' ',   # non-breaking space
    0x200B: '',    # zero-width space
}


def _ascii_only(s: str) -> str:
    """Normalize to ASCII, replacing common Unicode chars."""
    s = s.translate(_ASCII_TRANS)
    s = unicodedata.normalize('NFKC', s)
    return s.encode('ascii', 'ignore').decode('ascii')


def _strip_markdown(code: str) -> str:
    """Remove markdown code fences, headings, and prose lines."""
    # First, try to extract code from markdown code blocks
    code_block_match = re.search(r'```(?:python)?\s*\n(.*?)```', code, flags=re.S)
    if code_block_match:
        return code_block_match.group(1).strip()

    # If no code block, try to extract just the function definitions
    lines = code.splitlines()
    keep = []
    in_code = False
    for ln in lines:
        stripped = ln.lstrip()
        # Start of code
        if stripped.startswith(("def ", "class ", "import ", "from ", "@")):
            in_code = True
        # End of code (prose line after code)
        if in_code and stripped and not stripped.startswith(("#", "def ", "class ", "import ", "from ", "@", "return", "if ", "elif ", "else:", "for ", "while ", "try:", "except", "finally:", "with ", "    ", "\t", ")", "]", "}", "pass", "raise", "yield", "break", "continue", "global", "nonlocal", "assert", "lambda")):
            # Check if it looks like prose (contains common prose patterns)
            if re.match(r'^[A-Z][a-z].*[.!?]', stripped) or stripped.startswith(("This ", "The ", "Here ", "Note", "Example")):
                break  # Stop at prose
        if in_code:
            keep.append(ln)

    return "\n".join(keep) if keep else code


def _ensure_entrypoint(code: str, expected_name: str) -> str:
    """Ensure the expected function name is defined."""
    if re.search(rf"\bdef\s+{re.escape(expected_name)}\s*\(", code):
        return code
    # alias the first defined function to expected name
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    if m:
        actual = m.group(1)
        return code + f"\n\n# alias for grader\n{expected_name} = {actual}\n"
    # last resort: stub to avoid NameError (will likely fail tests but not crash)
    return f"def {expected_name}(*args, **kwargs):\n    raise NotImplementedError\n\n" + code


def _flexible_signature_shim(expected_name: str) -> str:
    """Allow function to accept superfluous args that some tests pass."""
    return f"""
try:
    _orig = {expected_name}
    import inspect
    def {expected_name}(*args, **kwargs):
        sig = inspect.signature(_orig)
        params = list(sig.parameters.values())
        trimmed = args[:len(params)]
        return _orig(*trimmed, **kwargs)
except Exception:
    pass
"""


def _prepare_code_for_exec(generated: str, expected_name: str) -> str:
    """Sanitize and prepare generated code for execution."""
    code = _strip_markdown(generated)
    code = _ascii_only(code)
    code = _ensure_entrypoint(code, expected_name)
    code = code + "\n" + _flexible_signature_shim(expected_name)
    # compile gate
    try:
        ast.parse(code)
    except SyntaxError:
        # fall back to just a stub to avoid poisoning reward with SyntaxError noise
        code = f"def {expected_name}(*args, **kwargs):\n    raise SyntaxError('bad submission')\n"
    return textwrap.dedent(code)


def _extract_function_name_from_test(test_case) -> Optional[str]:
    """Extract function name from test assertion."""
    # Handle case where test_case is a list
    if isinstance(test_case, list):
        test_case = test_case[0] if test_case else ""
    test_case = str(test_case)
    match = re.search(r'(\w+)\s*\([^)]*\)', test_case)
    if match:
        return match.group(1)
    return None


def _run_code_in_sandbox(
    code: str,
    setup_code: str,
    tests: List[str],
    timeout_seconds: int = 5
) -> Tuple[bool, int, List[str]]:
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

    temp_file = None
    try:
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
            for line in result.stdout.strip().split('\n'):
                if line.startswith('PASS:'):
                    tests_passed += 1
                elif line.startswith('FAIL:'):
                    parts = line.split(':', 3)
                    if len(parts) >= 4:
                        error_messages.append(f"Test {parts[1]}: {parts[2]} - {parts[3]}")
            execution_success = True
        else:
            execution_success = False
            if result.stderr:
                error_messages.append(f"Execution error: {result.stderr[:200]}")

        return execution_success, tests_passed, error_messages

    except subprocess.TimeoutExpired:
        error_messages.append("Code execution timed out (infinite loop)")
        return False, 0, error_messages
    except Exception as e:
        error_messages.append(f"Sandbox error: {str(e)}")
        return False, 0, error_messages
    finally:
        # Clean up temp file
        if temp_file:
            try:
                os.unlink(temp_file)
            except:
                pass


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class CounterfactGRPOTrainer(TrainerBase):
    """
    Counterfactual GRPO backend using TRL's GRPOTrainer with custom importance weighting.

    Supports both math (GSM8K) and code (MBPP) training with unified structure.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if TRL and required dependencies are available."""
        try:
            from trl import GRPOTrainer, GRPOConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import numpy as np
            return True
        except ImportError:
            return False

    def __init__(self, config: UnifiedConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.reward_functions = []
        self.prompt_to_data: Dict[str, Dict] = {}  # Maps prompts to test data (for MBPP)
        self.dataset_type: str = "math"  # "math" or "code"
        self.reward_step = 0

    # -------------------------------------------------------------------------
    # Setup methods
    # -------------------------------------------------------------------------

    def _get_cfg(self, obj, attr: str, default=None):
        """Get attribute from config object or dict with default fallback."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def setup_model(self) -> None:
        """Load and configure the model with optional LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        model_cfg = self.config.model
        model_name = self._get_cfg(model_cfg, "name_or_path")
        precision = self._get_cfg(model_cfg, "precision", "bf16")
        # Handle PrecisionType enum
        if hasattr(precision, 'value'):
            precision = precision.value
        use_peft = self._get_cfg(model_cfg, "use_peft", False)

        logger.info(f"Loading model: {model_name}")

        # Configure dtype
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "auto": torch.bfloat16}
        torch_dtype = dtype_map.get(precision, torch.bfloat16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }

        # Quantization
        quantization = self._get_cfg(model_cfg, "quantization", {})
        load_in_8bit = quantization.get("load_in_8bit", False) if isinstance(quantization, dict) else False
        load_in_4bit = quantization.get("load_in_4bit", False) if isinstance(quantization, dict) else False
        if load_in_8bit or load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Apply LoRA
        if use_peft:
            lora_config = LoraConfig(
                r=self._get_cfg(model_cfg, "lora_r", 16),
                lora_alpha=self._get_cfg(model_cfg, "lora_alpha", 32),
                lora_dropout=self._get_cfg(model_cfg, "lora_dropout", 0.05),
                target_modules=self._get_cfg(model_cfg, "lora_target_modules", ["q_proj", "v_proj"]),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        logger.info(f"Model loaded with precision: {precision}, LoRA: {use_peft}")

    def setup_data(self) -> None:
        """Load and prepare training dataset."""
        from datasets import load_dataset

        datasets_cfg = self.config.datasets
        if not datasets_cfg:
            raise ValueError("No datasets configured")

        ds_cfg = datasets_cfg[0]
        ds_name = self._get_cfg(ds_cfg, "name", "").lower()
        split = self._get_cfg(ds_cfg, "split", "train")
        ds_config = self._get_cfg(ds_cfg, "config", None)  # e.g. "main" for gsm8k
        system_prompt = self._get_cfg(ds_cfg, "system_prompt", "")

        # Detect dataset type
        if "mbpp" in ds_name:
            self.dataset_type = "code"
            self._setup_mbpp_data(ds_name, split, system_prompt)
        elif "gsm8k" in ds_name or "math" in ds_name:
            self.dataset_type = "math"
            self._setup_math_data(ds_name, split, system_prompt, ds_config=ds_config)
        else:
            # Default to math
            self.dataset_type = "math"
            self._setup_math_data(ds_name, split, system_prompt, ds_config=ds_config)

        logger.info(f"Dataset type: {self.dataset_type}, samples: {len(self.train_dataset)}")

    def _setup_mbpp_data(self, ds_name: str, split: str, system_prompt: str) -> None:
        """Setup MBPP dataset for code training."""
        from datasets import load_dataset

        # Try to use evalplus MBPP+ if available
        try:
            from evalplus.data import get_mbpp_plus
            mbpp_plus = get_mbpp_plus()
            logger.info(f"Using evalplus MBPP+ with {len(mbpp_plus)} problems")

            # Build dataset from evalplus format
            formatted_data = []
            for task_id, item in mbpp_plus.items():
                prompt_text = item.get("prompt", "")
                entry_point = item.get("entry_point", "")  # Function name from evalplus
                base_inputs = item.get("base_input", [])
                canonical_solution = item.get("canonical_solution", "")

                # Build assertion-style tests from base_input for display
                test_display = []
                for inp in base_inputs[:3]:  # Show first 3
                    test_display.append(f"{entry_point}({inp})")

                # Format prompt with test examples
                formatted_prompt = self._format_code_prompt(prompt_text, test_display, system_prompt)

                # Store test data for reward computation
                self.prompt_to_data[formatted_prompt] = {
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "base_inputs": base_inputs,
                    "canonical_solution": canonical_solution,
                    "expected_func": entry_point,  # Use entry_point directly
                }

                formatted_data.append({"prompt": formatted_prompt})

            from datasets import Dataset
            self.train_dataset = Dataset.from_list(formatted_data)

        except ImportError:
            logger.info("evalplus not available, using HuggingFace MBPP")
            # Fallback to HuggingFace MBPP
            raw_dataset = load_dataset("mbpp", split=split)

            formatted_data = []
            for idx, item in enumerate(raw_dataset):
                prompt_text = item.get("text", "")
                test_list = item.get("test_list", [])
                setup_code = item.get("test_setup_code", "")

                # Format prompt
                formatted_prompt = self._format_code_prompt(prompt_text, test_list, system_prompt)

                # Store test data
                self.prompt_to_data[formatted_prompt] = {
                    "idx": idx,
                    "setup": setup_code or "",
                    "tests": test_list or [],
                    "expected_func": _extract_function_name_from_test(test_list[0]) if test_list else None
                }

                formatted_data.append({"prompt": formatted_prompt})

            from datasets import Dataset
            self.train_dataset = Dataset.from_list(formatted_data)

    def _format_code_prompt(self, problem_text: str, test_list: List[str], system_prompt: str) -> str:
        """Format a code problem as a prompt."""
        # Build message
        user_content = f"{problem_text}\n\nTest cases:\n"
        for test in test_list[:3]:  # Show first 3 test cases
            user_content += f"- {test}\n"
        user_content += "\nWrite the Python function to solve this problem."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # Apply chat template if available
        enable_thinking = self._get_cfg(self.config.train, "enable_thinking", False)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except:
            # Fallback for tokenizers without enable_thinking
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _setup_math_data(self, ds_name: str, split: str, system_prompt: str, ds_config: str = None) -> None:
        """Setup math dataset (GSM8K or similar)."""
        from datasets import load_dataset

        # GSM8K requires config="main" to be specified
        if ds_config:
            raw_dataset = load_dataset(ds_name, ds_config, split=split)
        elif "gsm8k" in ds_name:
            raw_dataset = load_dataset(ds_name, "main", split=split)
        else:
            raw_dataset = load_dataset(ds_name, split=split)

        # Detect format
        if "question" in raw_dataset.column_names:
            question_col = "question"
            answer_col = "answer"
        else:
            # Handle various dataset formats
            question_col = "problem" if "problem" in raw_dataset.column_names else raw_dataset.column_names[0]
            # Look for answer/solution column
            if "answer" in raw_dataset.column_names:
                answer_col = "answer"
            elif "solution" in raw_dataset.column_names:
                answer_col = "solution"
            else:
                answer_col = raw_dataset.column_names[1] if len(raw_dataset.column_names) > 1 else None

        formatted_data = []
        for item in raw_dataset:
            question = item.get(question_col, "")
            answer = item.get(answer_col, "") if answer_col else ""

            # Extract gold answer for math
            gold_answer = extract_math_gold(answer) if answer else None

            # Format prompt
            formatted_prompt = self._format_math_prompt(question, system_prompt)

            # Store answer data
            self.prompt_to_data[formatted_prompt] = {
                "gold_answer": gold_answer,
                "full_answer": answer
            }

            formatted_data.append({"prompt": formatted_prompt})

        from datasets import Dataset
        self.train_dataset = Dataset.from_list(formatted_data)

    def _format_math_prompt(self, question: str, system_prompt: str) -> str:
        """Format a math question as a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        enable_thinking = self._get_cfg(self.config.train, "enable_thinking", False)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def setup_rewards(self) -> None:
        """Setup reward functions based on dataset type."""
        if self.dataset_type == "code":
            logger.info("Setting up code execution rewards for MBPP")
        else:
            logger.info("Setting up math reasoning rewards")

    # -------------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------------

    def _combined_reward_function(self, completions: List, **kwargs) -> List[float]:
        """Combined reward function that routes to appropriate reward based on dataset type."""
        self.reward_step += 1

        if self.dataset_type == "code":
            rewards = self._compute_code_rewards(completions, **kwargs)
        else:
            rewards = self._compute_math_rewards(completions, **kwargs)

        # Print reward summary
        self._print_reward_summary(rewards)
        return rewards

    def _print_reward_summary(self, rewards: List[float]) -> None:
        """Print summary of reward distribution."""
        if not rewards:
            return

        rewards_arr = np.array(rewards)
        n_positive = np.sum(rewards_arr > 0)
        n_zero = np.sum(rewards_arr == 0)
        n_negative = np.sum(rewards_arr < 0)
        n_perfect = np.sum(rewards_arr >= 1.0)

        print(f"\n{'='*60}")
        print(f"[Reward Step {self.reward_step}] Dataset: {self.dataset_type.upper()}")
        print(f"{'='*60}")
        print(f"  Batch size: {len(rewards)}")
        print(f"  Reward stats: min={rewards_arr.min():.3f}, max={rewards_arr.max():.3f}, "
              f"mean={rewards_arr.mean():.3f}, std={rewards_arr.std():.3f}")
        print(f"  Distribution: +ve={n_positive}, zero={n_zero}, -ve={n_negative}, perfect={n_perfect}")
        print(f"  Success rate: {(n_positive / len(rewards)) * 100:.1f}%")
        print(f"{'='*60}\n")

    def _run_evalplus_tests(
        self,
        code: str,
        canonical_solution: str,
        entry_point: str,
        base_inputs: List,
        timeout_seconds: int = 5
    ) -> Tuple[bool, int, List[str]]:
        """
        Run evalplus-style tests by comparing outputs to canonical solution.
        Returns: (execution_success, tests_passed, error_messages)
        """
        error_messages = []
        passed = 0

        # Block input() to prevent hanging
        input_blocker = "import builtins; builtins.input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no input'))"

        # Create test script that compares generated vs canonical outputs
        test_script = f'''
import sys
import traceback

# Block input
{input_blocker}

# Canonical solution
{canonical_solution}
_canonical_func = {entry_point}

# Generated solution
{code}
_generated_func = {entry_point}

# Test inputs
base_inputs = {repr(base_inputs)}

# Run tests
results = []
for i, inp in enumerate(base_inputs):
    try:
        if isinstance(inp, (list, tuple)):
            expected = _canonical_func(*inp)
            actual = _generated_func(*inp)
        else:
            expected = _canonical_func(inp)
            actual = _generated_func(inp)

        if expected == actual:
            results.append(f"PASS:{{i}}")
        else:
            results.append(f"FAIL:{{i}}:expected={{expected}},got={{actual}}")
    except Exception as e:
        results.append(f"ERROR:{{i}}:{{type(e).__name__}}:{{str(e)[:50]}}")

for r in results:
    print(r)
'''

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_file = f.name

            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('PASS:'):
                        passed += 1
                    elif line.startswith('FAIL:') or line.startswith('ERROR:'):
                        error_messages.append(line)
                return True, passed, error_messages
            else:
                error_messages.append(f"Execution error: {result.stderr[:200]}")
                return False, 0, error_messages

        except subprocess.TimeoutExpired:
            error_messages.append("Code execution timed out")
            return False, 0, error_messages
        except Exception as e:
            error_messages.append(f"Sandbox error: {str(e)}")
            return False, 0, error_messages
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _compute_code_rewards(self, completions: List, **kwargs) -> List[float]:
        """Compute rewards for code generation (MBPP)."""
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        # Track statistics
        stats = {
            "no_tests": 0,
            "no_func": 0,
            "timeout": 0,
            "exec_fail": 0,
            "zero_pass": 0,
            "partial_pass": 0,
            "full_pass": 0,
        }

        print(f"\n[Code Rewards] Processing {len(completions)} completions...")

        for i, completion in enumerate(completions):
            # Extract completion text
            if isinstance(completion, dict) and "content" in completion:
                completion_code = completion["content"]
            elif isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict):
                    completion_code = completion[0].get("content", str(completion))
                else:
                    completion_code = str(completion[0])
            else:
                completion_code = str(completion)

            # Get test data for this prompt
            prompt = prompts[i] if i < len(prompts) else ""
            test_data = self.prompt_to_data.get(prompt, {})

            # Get expected function name and test inputs
            expected_func = test_data.get("expected_func") or test_data.get("entry_point")
            base_inputs = test_data.get("base_inputs", [])
            canonical_solution = test_data.get("canonical_solution", "")

            # Fallback for HF MBPP format
            if not base_inputs:
                tests = test_data.get("tests", [])
                if tests:
                    # Use old assertion-based testing
                    if not expected_func:
                        expected_func = _extract_function_name_from_test(tests[0])
                else:
                    rewards.append(-0.10)
                    stats["no_tests"] += 1
                    if i < 3:
                        print(f"  [{i}] No tests found -> reward=-0.10")
                    continue

            if not expected_func:
                rewards.append(-0.10)
                stats["no_func"] += 1
                if i < 3:
                    print(f"  [{i}] Cannot extract function name -> reward=-0.10")
                continue

            # Sanitize and prepare code
            sanitized_code = _prepare_code_for_exec(completion_code, expected_func)

            # Execute with evalplus-style testing (compare outputs)
            if base_inputs and canonical_solution:
                execution_success, passed, error_messages = self._run_evalplus_tests(
                    code=sanitized_code,
                    canonical_solution=canonical_solution,
                    entry_point=expected_func,
                    base_inputs=base_inputs,
                    timeout_seconds=5
                )
            else:
                # Fallback to assertion-based testing
                execution_success, passed, error_messages = _run_code_in_sandbox(
                    code=sanitized_code,
                    setup_code=test_data.get("setup", ""),
                    tests=test_data.get("tests", []),
                    timeout_seconds=5
                )

            # Compute reward (use base_inputs or tests depending on format)
            total_tests = len(base_inputs) if base_inputs else len(test_data.get("tests", [1]))
            if total_tests > 0:
                base_reward = float(passed) / total_tests
            else:
                base_reward = 0.0

            # Apply penalties and track stats
            if not execution_success:
                timeout_crash = any("timed out" in msg for msg in error_messages)
                if timeout_crash:
                    base_reward = -0.25
                    stats["timeout"] += 1
                    if i < 3:
                        print(f"  [{i}] func={expected_func} TIMEOUT -> reward=-0.25")
                else:
                    base_reward = -0.15
                    stats["exec_fail"] += 1
                    if i < 3:
                        print(f"  [{i}] func={expected_func} EXEC_FAIL -> reward=-0.15")
            elif passed == 0:
                base_reward -= 0.10
                stats["zero_pass"] += 1
                if i < 3:
                    print(f"  [{i}] func={expected_func} 0/{total_tests} tests -> reward={base_reward:.2f}")
            elif passed == total_tests:
                stats["full_pass"] += 1
                if i < 3:
                    print(f"  [{i}] func={expected_func} {passed}/{total_tests} tests -> reward={base_reward:.2f} (PASS)")
            else:
                stats["partial_pass"] += 1
                if i < 3:
                    print(f"  [{i}] func={expected_func} {passed}/{total_tests} tests -> reward={base_reward:.2f}")

            rewards.append(base_reward)

        # Print code-specific stats
        print(f"\n[Code Stats] no_tests={stats['no_tests']}, no_func={stats['no_func']}, "
              f"timeout={stats['timeout']}, exec_fail={stats['exec_fail']}")
        print(f"[Code Stats] zero_pass={stats['zero_pass']}, partial={stats['partial_pass']}, "
              f"full_pass={stats['full_pass']}")

        return rewards

    def _compute_math_rewards(self, completions: List, **kwargs) -> List[float]:
        """Compute rewards for math reasoning (GSM8K)."""
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        # Track statistics
        n_correct = 0
        n_incorrect = 0
        n_no_gold = 0
        n_no_pred = 0

        print(f"\n[Math Rewards] Processing {len(completions)} completions...")

        for i, completion in enumerate(completions):
            # Extract completion text
            if isinstance(completion, dict) and "content" in completion:
                completion_text = completion["content"]
            elif isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict):
                    completion_text = completion[0].get("content", str(completion))
                else:
                    completion_text = str(completion[0])
            else:
                completion_text = str(completion)

            # Get gold answer
            prompt = prompts[i] if i < len(prompts) else ""
            data = self.prompt_to_data.get(prompt, {})
            gold_answer = data.get("gold_answer")

            if gold_answer is None:
                rewards.append(0.0)
                n_no_gold += 1
                if i < 3:
                    print(f"  [{i}] No gold answer -> reward=0.0")
                continue

            # Extract predicted answer and grade
            predicted_answer = extract_math_answer(completion_text)
            is_correct = grade_math_answer(predicted_answer, gold_answer)

            if predicted_answer is None:
                n_no_pred += 1

            if is_correct:
                rewards.append(1.0)
                n_correct += 1
                if i < 3:
                    print(f"  [{i}] gold={gold_answer}, pred={predicted_answer} -> CORRECT (reward=1.0)")
            else:
                rewards.append(0.0)
                n_incorrect += 1
                if i < 3:
                    print(f"  [{i}] gold={gold_answer}, pred={predicted_answer} -> WRONG (reward=0.0)")

        # Print math-specific stats
        print(f"\n[Math Stats] correct={n_correct}, incorrect={n_incorrect}, "
              f"no_gold={n_no_gold}, no_pred={n_no_pred}")
        print(f"[Math Stats] Accuracy: {(n_correct / len(completions)) * 100:.1f}%")

        return rewards

    # -------------------------------------------------------------------------
    # Abstract method implementations (TRL handles these internally)
    # -------------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step (stub - TRL handles this internally)."""
        logger.debug("train_step() called but TRL GRPO uses TRL's internal training loop")
        return {"loss": 0.0}

    def create_data_loader(self) -> Optional[DataLoader]:
        """Create data loader (stub - TRL handles this internally)."""
        logger.debug("create_data_loader() called but TRL GRPO uses TRL's internal data loading")
        return None

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run counterfactual GRPO training."""
        from trl import GRPOConfig
        from .custom_trainer import CounterfactualGRPOTrainer as CustomTrainer

        # Setup components if not already done
        if self.model is None:
            self.setup_model()
        if self.train_dataset is None:
            self.setup_data()
        self.setup_rewards()

        train_cfg = self.config.train
        log_cfg = self.config.logging

        # Build GRPOConfig
        grpo_config = GRPOConfig(
            output_dir=self._get_cfg(log_cfg, "output_dir", "./output"),
            run_name=self._get_cfg(log_cfg, "run_name", "counterfact_grpo"),
            per_device_train_batch_size=self._get_cfg(train_cfg, "per_device_batch_size", 4),
            gradient_accumulation_steps=self._get_cfg(train_cfg, "gradient_accumulation_steps", 4),
            num_generations=self._get_cfg(train_cfg, "num_generations", 8),
            max_steps=self._get_cfg(train_cfg, "max_steps", 500),
            learning_rate=self._get_cfg(train_cfg, "learning_rate", 2.5e-5),
            max_prompt_length=self._get_cfg(train_cfg, "max_prompt_length", 512),
            max_completion_length=self._get_cfg(train_cfg, "max_completion_length", 512),
            temperature=self._get_cfg(train_cfg, "temperature", 0.7),
            top_p=self._get_cfg(train_cfg, "top_p", 0.95),
            beta=self._get_cfg(train_cfg, "beta", 0.005),
            save_steps=self._get_cfg(train_cfg, "save_steps", 100),
            save_strategy=self._get_cfg(train_cfg, "save_strategy", "steps"),
            save_total_limit=self._get_cfg(train_cfg, "save_total_limit", 3),
            logging_steps=self._get_cfg(train_cfg, "logging_steps", 10),
            warmup_ratio=self._get_cfg(train_cfg, "warmup_ratio", 0.05),
            seed=self._get_cfg(train_cfg, "seed", 42),
            report_to=self._get_cfg(log_cfg, "loggers", []),
        )

        # Gradient checkpointing
        if self._get_cfg(train_cfg, "use_gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()

        # Counterfactual params
        counterfactual_params = {
            "boost_factor": self._get_cfg(train_cfg, "boost_factor", 4.0),
            "min_weight": self._get_cfg(train_cfg, "min_weight", 0.5),
            "max_spans": self._get_cfg(train_cfg, "max_spans", 10),
            "answer_weight": self._get_cfg(train_cfg, "answer_weight", 1.5),
            "enable_gradient_conservation": self._get_cfg(train_cfg, "enable_gradient_conservation", True),
            "weighting_mode": self._get_cfg(train_cfg, "weighting_mode", "counterfactual"),
            "weight_debug": self._get_cfg(train_cfg, "weight_debug", False),
            "enable_thinking": self._get_cfg(train_cfg, "enable_thinking", False),
            "dataset_type": self.dataset_type,
            # Extra verbose mode for detailed logging (paper examples)
            "extra_verbose": self._get_cfg(train_cfg, "extra_verbose", False),
            "extra_verbose_log_path": self._get_cfg(train_cfg, "extra_verbose_log_path", None),
            "extra_verbose_sample_rate": self._get_cfg(train_cfg, "extra_verbose_sample_rate", 0.1),
            # Ultraverbose: print detailed spans/histogram/tokens to console
            "ultraverbose": self._get_cfg(train_cfg, "ultraverbose", False),
            # Ablation controls for drop sign and length normalization
            "invert_drop_sign": self._get_cfg(train_cfg, "invert_drop_sign", False),
            "normalize_by_span_length": self._get_cfg(train_cfg, "normalize_by_span_length", True),
        }

        # Create custom trainer
        self.trainer = CustomTrainer(
            model=self.model,
            args=grpo_config,
            processing_class=self.tokenizer,  # TRL uses processing_class instead of tokenizer
            train_dataset=self.train_dataset,
            reward_funcs=[self._combined_reward_function],
            # Counterfactual params (passed as individual args)
            weighting_mode=counterfactual_params["weighting_mode"],
            boost_factor=counterfactual_params["boost_factor"],
            min_weight=counterfactual_params["min_weight"],
            max_spans=counterfactual_params["max_spans"],
            answer_weight=counterfactual_params["answer_weight"],
            enable_gradient_conservation=counterfactual_params["enable_gradient_conservation"],
            weight_debug=counterfactual_params["weight_debug"],
            # Extra verbose logging for paper examples
            extra_verbose=counterfactual_params["extra_verbose"],
            extra_verbose_log_path=counterfactual_params["extra_verbose_log_path"],
            extra_verbose_sample_rate=counterfactual_params["extra_verbose_sample_rate"],
            # Ultraverbose: print detailed debug to console
            ultraverbose=counterfactual_params["ultraverbose"],
            # Ablation controls
            invert_drop_sign=counterfactual_params["invert_drop_sign"],
            normalize_by_span_length=counterfactual_params["normalize_by_span_length"],
        )

        # Setup wandb if configured
        loggers = self._get_cfg(log_cfg, "loggers", [])
        if "wandb" in loggers:
            try:
                import wandb
                wandb_cfg = self._get_cfg(log_cfg, "wandb", {})
                if isinstance(wandb_cfg, dict):
                    project = wandb_cfg.get("project", "counterfact-grpo")
                else:
                    project = self._get_cfg(wandb_cfg, "project", "counterfact-grpo")
                wandb.init(
                    project=project,
                    name=self._get_cfg(log_cfg, "run_name"),
                    config={
                        "model": self._get_cfg(self.config.model, "name_or_path"),
                        "dataset_type": self.dataset_type,
                    }
                )
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")

        # Train
        logger.info("Starting counterfactual GRPO training...")
        train_result = self.trainer.train()

        return {
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "train_runtime": train_result.metrics.get("train_runtime") if hasattr(train_result, 'metrics') else None,
            "dataset_type": self.dataset_type,
        }

    def save(self, output_path: str) -> None:
        """Save the trained model."""
        if self.trainer:
            self.trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def create_trainer(config: UnifiedConfig, **kwargs) -> CounterfactGRPOTrainer:
    """Factory function to create the trainer."""
    return CounterfactGRPOTrainer(config, **kwargs)
