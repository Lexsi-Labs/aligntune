"""Gold-answer extraction for Counterfactual GRPO datasets.

Keep GSM8K explicit answer markers separate from free-form MATH expressions.
"""

from typing import Any, Mapping

from aligntune.utils.math_grading import extract_math_gold, parse_gsm8k_gold


def extract_counterfact_gold(example: Mapping[str, Any]) -> str:
    """Extract a gold answer without treating arbitrary expressions as GSM8K prose."""
    answer = str(example.get("answer") or "").strip()
    solution = str(example.get("solution") or "").strip()

    if "####" in answer:
        return parse_gsm8k_gold(answer)
    if "####" in solution:
        return parse_gsm8k_gold(solution)

    solution_gold = extract_math_gold(solution) if solution else ""
    if solution_gold:
        return solution_gold

    answer_gold = extract_math_gold(answer) if answer else ""
    if answer_gold:
        return answer_gold

    # Preserve an unboxed expression; never guess its value from its last digit.
    return answer
