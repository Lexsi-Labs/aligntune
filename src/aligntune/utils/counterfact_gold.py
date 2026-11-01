"""Gold-answer extraction for Counterfactual GRPO datasets.

Keep GSM8K explicit answer markers separate from free-form MATH expressions.
"""

from typing import Any, Mapping

from aligntune.utils.math_grading import extract_math_gold, parse_gsm8k_gold


def extract_counterfact_gold(example: Mapping[str, Any]) -> str:
    """Extract a gold answer without treating arbitrary expressions as GSM8K prose."""
    raw_answer = example.get("answer")
    raw_solution = example.get("solution")
    answer = "" if raw_answer is None else str(raw_answer).strip()
    solution = "" if raw_solution is None else str(raw_solution).strip()

    # GSM8K's explicit marker is the reliable signal for its prose-style format.
    if "####" in answer:
        return parse_gsm8k_gold(answer)
    if "####" in solution:
        return parse_gsm8k_gold(solution)

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
    return answer or solution
