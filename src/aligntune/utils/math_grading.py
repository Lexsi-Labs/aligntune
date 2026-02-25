"""
Robust MATH dataset answer parsing and grading.

Ported from bolt/benchmark_math_competition.py - combines:
1. Official MATH dataset normalization (hendrycks/math)
2. PRM800K grader (OpenAI) with SymPy symbolic equivalence
3. Qwen2.5-Math extraction strategy

This module provides a single source of truth for MATH grading used by:
- Evaluation (eval/eval2.py)
- Baseline computation (scripts/precompute_baseline.py)
- Training rewards (rewards/core.py)

Usage:
    from aligntune.utils.math_grading import grade_math_answer, extract_math_answer

    # Grade a prediction against ground truth
    is_correct = grade_math_answer(prediction, gold_answer)

    # Extract answer from model output
    pred_answer = extract_math_answer(model_output)

    # Extract gold answer from MATH dataset
    gold_answer = extract_math_gold(solution_field)
"""

import re
from typing import Optional

# Optional dependencies - graceful degradation
try:
    import sympy
    from sympy.parsing import sympy_parser
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympy = None
    sympy_parser = None

try:
    from pylatexenc import latex2text
    PYLATEXENC_AVAILABLE = True
except ImportError:
    PYLATEXENC_AVAILABLE = False
    latex2text = None


# =============================================================================
# Official MATH Dataset Normalization (hendrycks/math)
# From: https://github.com/hendrycks/math
# =============================================================================

def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from answer."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]

    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]

    return s


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} or \\fbox{...} from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def fix_fracs(string: str) -> str:
    """Fix \\frac without braces: \\frac12 -> \\frac{1}{2}"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b}"""
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        if string == "{}/{}".format(a, b):
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        pass
    return string


def remove_right_units(string: str) -> str:
    """Remove units from the right side."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string: str) -> str:
    """Fix \\sqrt without braces: \\sqrt3 -> \\sqrt{3}"""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    """
    Official MATH dataset normalization function.
    Normalizes LaTeX strings for comparison.
    """
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\\\ with \\
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \\left and \\right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \\frac1b or \\frac12 --> \\frac{1}{b} and \\frac{1}{2}
    string = fix_fracs(string)

    # manually change 0.5 --> \\frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # X/Y changed to \\frac{X}{Y}
    string = fix_a_slash_b(string)

    return string


# =============================================================================
# PRM800K Grader (OpenAI)
# From: https://github.com/openai/prm800k
# =============================================================================

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    if not SYMPY_AVAILABLE:
        return None
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers

    # Add parentheses to fractions to preserve grouping
    def _fix_frac_precedence(latex_expr: str) -> str:
        pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
        def replace_func(match):
            num, den = match.group(1), match.group(2)
            return f"({num})/({den})"
        return re.sub(pattern, replace_func, latex_expr)

    expr = _fix_frac_precedence(expr)

    if PYLATEXENC_AVAILABLE:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    else:
        # Fallback: basic LaTeX removal
        expr = re.sub(r'\\[a-zA-Z]+', '', expr)
        expr = expr.replace('{', '').replace('}', '')

    # Replace special characters
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    """Make mixed numbers evaluable: 7 3/4 => 7+3/4"""
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step


def _strip_properly_formatted_commas(expr: str) -> str:
    """Remove thousand separators while preserving tuple commas."""
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize_prm(expr: str) -> Optional[str]:
    """Normalize answer expressions (PRM800K style)."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile",
        "second", "minute", "hour", "day", "week", "month", "year",
        "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub(r"- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # drop any remaining latex braces
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # case insensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str) -> int:
    """Count number of unknown variables in expression."""
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str) -> bool:
    """Check if expression is safe to evaluate with sympy."""
    if count_unknown_letters_in_expr(expr) > 5:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    """Check if two normalized expressions are equal using SymPy."""
    if not SYMPY_AVAILABLE:
        return False

    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                return True
    except:
        pass
    return False


def split_tuple(expr: str) -> list:
    """Split elements in a tuple/interval, handling commas in large numbers."""
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def normalize_answer_mathd(answer: Optional[str]) -> Optional[str]:
    """Normalize using official MATH dataset normalization."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return strip_string(answer)
    except:
        return answer


# =============================================================================
# Main Grading Function (Three-tier)
# =============================================================================

def grade_math_answer(given_answer: str, ground_truth: str) -> bool:
    """
    Grade a math answer using three-tier comparison:

    1. Official MATH dataset normalization
    2. PRM800K normalization
    3. SymPy symbolic equivalence

    Args:
        given_answer: The predicted answer
        ground_truth: The gold/correct answer

    Returns:
        True if the answer is correct, False otherwise
    """
    if given_answer is None:
        return False

    # Tier 1: Official MATH normalization
    ground_truth_normalized_mathd = normalize_answer_mathd(ground_truth)
    given_answer_normalized_mathd = normalize_answer_mathd(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    # Tier 2: PRM800K normalization
    ground_truth_normalized = _normalize_prm(ground_truth)
    given_normalized = _normalize_prm(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if not given_normalized or len(given_normalized) == 0:
        return False

    # Tier 3: SymPy symbolic equivalence (handles tuples/intervals)
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        return False

    if len(ground_truth_elems) != len(given_elems):
        return False

    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
        if _is_frac(ground_truth_elem) and _is_frac(given_elem):
            # Fractions must match exactly (no unreduced fractions)
            if ground_truth_elem != given_elem:
                return False
        else:
            if not are_equal_under_sympy(ground_truth_elem, given_elem):
                return False

    return True


# Alias for backwards compatibility
grade_answer = grade_math_answer


# =============================================================================
# Answer Extraction Functions
# =============================================================================

def extract_math_answer(text: str) -> str:
    """
    Extract the predicted answer from model output.
    Uses Qwen2.5-Math extraction strategy for robustness.

    Handles:
    - \\boxed{...} format (most common)
    - "final answer is $...$" format (Minerva)
    - "the answer is ..." format
    - Last number fallback
    """
    pred_str = text

    # Check for Minerva math format
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
        return pred

    # Check for boxed answer (most common)
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            # Stack-based brace matching
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            return a
        else:
            # No braces, take until first $
            a = ans.split("$")[0].strip()
            return a

    # Check for "answer is" phrases
    if "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
        return pred
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
        return pred

    # Last resort: extract last number
    pattern = r"-?\d*\.?\d+"
    matches = re.findall(pattern, pred_str.replace(",", ""))
    if len(matches) >= 1:
        return matches[-1]

    return ""


def extract_math_gold(answer_data) -> str:
    """
    Extract the gold answer from MATH dataset solution field.
    Uses official MATH approach: find last \\boxed{} and extract its content.

    Args:
        answer_data: Either a string (solution) or dict with 'answer'/'solution' key

    Returns:
        The extracted gold answer
    """
    if isinstance(answer_data, dict):
        solution = answer_data.get('answer', answer_data.get('solution', ''))
    else:
        solution = str(answer_data)

    # Use official MATH extraction: last_boxed_only_string
    boxed_str = last_boxed_only_string(solution)
    if boxed_str:
        try:
            return remove_boxed(boxed_str)
        except:
            return boxed_str

    return ""


# =============================================================================
# GSM8K Support (simpler format)
# =============================================================================

def extract_gsm8k_answer(text: str) -> str:
    """
    Extract answer from GSM8K-style response.

    Handles:
    1. #### format (standard GSM8K)
    2. \\boxed{} format
    3. Last number fallback
    """
    # Try #### format first
    if "####" in text:
        answer = text.split("####")[-1].strip()
        # Clean up common artifacts
        answer = answer.split("\n")[0].strip()
        answer = answer.replace(",", "")
        return answer

    # Try boxed format
    boxed = extract_math_answer(text)
    if boxed:
        return boxed

    # Fallback: last number
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text.replace(",", ""))
    if matches:
        return matches[-1]

    return ""


def parse_gsm8k_gold(ans: str) -> str:
    """Parse gold answer from GSM8K format (#### answer)."""
    # Standard GSM8K format: "#### 42"
    ans_re = re.compile(r"####\s*(.+)$", flags=re.MULTILINE)
    m = ans_re.search(ans)
    if not m:
        # Fallback: extract last number
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        return nums[-1] if nums else ""
    s = m.group(1).strip().replace(",", "")
    # Handle fractions like "1/2"
    if re.fullmatch(r"-?\d+/\d+", s):
        try:
            num, den = s.split("/")
            return str(float(num) / float(den))
        except:
            pass
    return s


def grade_gsm8k_answer(pred: str, gold: str, rel_tol: float = 1e-5, abs_tol: float = 1e-9) -> bool:
    """
    Grade GSM8K answer with numeric tolerance.

    Args:
        pred: Predicted answer string
        gold: Gold answer string
        rel_tol: Relative tolerance for numeric comparison
        abs_tol: Absolute tolerance for numeric comparison

    Returns:
        True if correct, False otherwise
    """
    import math

    # Try exact string match first
    pred_clean = pred.strip().replace(",", "").lower()
    gold_clean = gold.strip().replace(",", "").lower()

    if pred_clean == gold_clean:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_clean)
        gold_num = float(gold_clean)
        return math.isclose(pred_num, gold_num, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError):
        pass

    return False


# =============================================================================
# Unified Grading Interface
# =============================================================================

def grade_answer_auto(pred: str, gold: str, dataset_type: str = "math") -> bool:
    """
    Automatically select the right grading function based on dataset type.

    Args:
        pred: Predicted answer
        gold: Gold answer
        dataset_type: "math" for MATH dataset, "gsm8k" for GSM8K

    Returns:
        True if correct, False otherwise
    """
    if dataset_type.lower() == "gsm8k":
        pred_answer = extract_gsm8k_answer(pred)
        gold_answer = parse_gsm8k_gold(gold) if "####" in gold else gold
        return grade_gsm8k_answer(pred_answer, gold_answer)
    else:
        # MATH dataset or similar
        pred_answer = extract_math_answer(pred)
        gold_answer = extract_math_gold(gold) if "\\boxed" in gold else gold
        return grade_math_answer(pred_answer, gold_answer)
