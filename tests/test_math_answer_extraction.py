"""Regression tests: an answer in a sentence must grade like the bare answer.

``extract_math_answer`` returned everything after "the answer is", so a colon,
a full stop or a closing paragraph reached the graders and a correct response
got reward 0.0. No model is loaded.
"""

import pytest

from aligntune.utils.math_grading import extract_math_answer, grade_answer_auto


@pytest.mark.parametrize(
    "response, expected",
    [
        ("The answer is: 42", "42"),
        ("The answer is: 42.", "42"),
        ("The answer is 3/4.", "3/4"),
        ("So the answer is 7.5.", "7.5"),
        ("The answer is: .5.", ".5"),
        ("The final answer is: 42", "42"),
        ("The answer is 42\n\nHope this helps", "42"),
        ("The answer is 42.\n\nHope this helps.", "42"),
        ("The answer is:\n\n42", "42"),
    ],
)
def test_phrase_answer_keeps_only_the_answer(response, expected):
    assert extract_math_answer(response) == expected


@pytest.mark.parametrize(
    "response, gold",
    [
        ("The answer is: 42", "42"),
        ("So the answer is 7.5.", "7.5"),
        ("The answer is 3/4.", "3/4"),
        ("The answer is \\frac{3}{4}.", "\\frac{3}{4}"),
        ("The answer is $\\frac{3}{4}$.", "\\frac{3}{4}"),
        ("The answer is 42\n\nHope this helps", "42"),
        (
            "The answer is\n\\begin{pmatrix}\n1 \\\\\n2\n\\end{pmatrix}.",
            "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
        ),
    ],
)
def test_math_grading_accepts_sentence_answers(response, gold):
    assert grade_answer_auto(response, gold, "math")


def test_gsm8k_grading_gets_the_same_cleanup():
    assert grade_answer_auto("The answer is: 18", "#### 18", "gsm8k")


@pytest.mark.parametrize(
    "response, gold, expected",
    [
        ("The answer is: 41.", "42", False),
        ("The answer is 42. Hope this helps", "42", False),
        (
            "The answer is\n\\begin{pmatrix}\n1 \\\\\n2\n\\end{pmatrix}",
            "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
            True,
        ),
    ],
)
def test_grading_that_must_not_change(response, gold, expected):
    assert grade_answer_auto(response, gold, "math") is expected
