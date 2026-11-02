"""Regression tests for Counterfactual GRPO gold-answer extraction."""

import pytest

from aligntune.utils.counterfact_gold import extract_counterfact_gold


@pytest.mark.parametrize("answer", [
    r"\frac{14}{3}",
    r"x^2+1",
    r"\sqrt{2}",
    r"(1,2)",
    r"2\pi",
])
def test_preserves_unboxed_math_expressions(answer):
    assert extract_counterfact_gold({"answer": answer}) == answer


def test_preserves_unboxed_expression_from_solution_when_answer_is_empty():
    expression = r"\frac{14}{3}"
    assert extract_counterfact_gold({"answer": "", "solution": expression}) == expression


def test_preserves_numeric_zero():
    assert extract_counterfact_gold({"answer": 0}) == "0"


def test_prefers_boxed_answer_from_math_solution():
    example = {
        "answer": r"\frac{14}{3}",
        "solution": r"Work shown. Final answer: \boxed{\frac{14}{3}}.",
    }
    assert extract_counterfact_gold(example) == r"\frac{14}{3}"


def test_parses_gsm8k_answer_using_explicit_marker():
    assert extract_counterfact_gold({"answer": "Work: 6 times 7.\n#### 42"}) == "42"


def test_does_not_extract_last_digit_from_unmarked_text():
    assert extract_counterfact_gold({"answer": "The result is x^2+1"}) == "The result is x^2+1"


def test_returns_empty_when_no_answer_is_available():
    assert extract_counterfact_gold({}) == ""
