"""Regression tests: GSM8K grading must compare values, not number formatting.

``grade_gsm8k_answer`` only tried ``float()``, so a currency sign, a percent sign
or a fraction made a correct answer grade as wrong. No model is loaded.
"""

import pytest

from aligntune.utils.math_grading import grade_answer_auto, grade_gsm8k_answer


@pytest.mark.parametrize(
    "response, gold",
    [
        ("#### $18", "#### 18"),
        ("#### $18.00", "#### 18"),
        ("The answer is $18.", "#### 18"),
        ("\\boxed{$18}", "#### 18"),
        ("\\boxed{\\$18}", "#### 18"),
        ("\\boxed{18\\%}", "#### 18"),
        ("#### 1/2", "#### 1/2"),
    ],
)
def test_correct_answers_grade_as_correct(response, gold):
    assert grade_answer_auto(response, gold, "gsm8k")


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("$1,200.50", "1200.5"),
        ("-$5", "-5"),
        ("18%", "18"),
        ("3/4", "0.75"),
    ],
)
def test_numeric_formats_compare_by_value(pred, gold):
    assert grade_gsm8k_answer(pred, gold)


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ("$19", "18", False),
        ("1/3", "0.5", False),
        ("1/0", "18", False),
        pytest.param("9" * 400 + "/1", "18", False, id="fraction-too-large-for-float"),
        ("18 dollars", "18", False),
        ("18.", "18", True),
    ],
)
def test_grading_that_must_not_change(pred, gold, expected):
    assert grade_gsm8k_answer(pred, gold) is expected
