"""Gold-answer extraction for Counterfactual GRPO datasets.

Keep GSM8K explicit answer markers separate from free-form MATH expressions.
"""

import re
from typing import Any, Mapping

from aligntune.utils.math_grading import extract_math_gold

_MATH_SYNTAX = re.compile(r"[\\^_{}=+*/()\[\]<>|]|\w-\w|\w\s+-\s+\w")
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")


def _lone_number(text: str) -> str:
    """The number in plain text with exactly one ("72 eggs"), else "" ("x^2+1", "(1,2)", "3x").

    Known limit: "2 x" and "2 pi" read as 2, as they did before.
    """
    if _MATH_SYNTAX.search(text):
        return ""
    numbers = _NUMBER.findall(text)
    return numbers[0].replace(",", "") if len(numbers) == 1 else ""


def extract_counterfact_gold(example: Mapping[str, Any], answer_col: str = "answer") -> str:
    """Extract a gold answer without treating arbitrary expressions as GSM8K prose.

    ``answer_col`` is the column holding the answer, for datasets that do not call it "answer".
    """
    raw_answer = example.get(answer_col)
    raw_solution = example.get("solution")
    answer = "" if raw_answer is None else str(raw_answer).strip()
    solution = "" if raw_solution is None else str(raw_solution).strip()

    # GSM8K's explicit marker is the reliable signal for its prose-style format.
    if "####" in answer:
        return extract_math_gold(answer)
    if "####" in solution:
        return extract_math_gold(solution)

    # MATH datasets commonly put the canonical boxed answer in the solution.
    solution_gold = extract_math_gold(solution) if solution else ""
    if solution_gold:
        return solution_gold

    # Some datasets store a bare LaTeX expression directly in answer.
    answer_gold = extract_math_gold(answer) if answer else ""
    if answer_gold:
        return answer_gold

    # Preserve unboxed values, including expressions and numeric zero. If the
    # answer column is absent/empty, retain a non-empty solution rather than
    # silently replacing it with an empty gold answer.
    text = answer or solution
    return _lone_number(text) or text
