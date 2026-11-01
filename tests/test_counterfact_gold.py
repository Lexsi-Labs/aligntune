"""Regression tests for Counterfactual GRPO gold-answer extraction."""

import pytest

from aligntune.utils.counterfact_gold import extract_counterfact_gold
from aligntune.utils.math_grading import extract_math_answer, grade_math_answer


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


def trl_golds(rows):
    """Run the real TRL _setup_math_data over rows; return the gold of each, in order."""
    from unittest import mock

    from datasets import Dataset

    from aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo import (
        TRLCounterFactGRPOTrainer,
    )

    trainer = object.__new__(TRLCounterFactGRPOTrainer)
    trainer.prompt_to_data = {}
    trainer._format_math_prompt = lambda question, system_prompt: question
    with mock.patch("datasets.load_dataset", return_value=Dataset.from_list(rows)):
        trainer._setup_math_data("toy/math", "test", "")
    return [trainer.prompt_to_data[row["problem"]]["gold_answer"] for row in rows]


def test_trl_reads_the_column_it_chose():
    assert trl_golds([{"problem": "p", "final_answer": "42"}]) == ["42"]
    assert trl_golds([{"problem": "p", "final_answer": r"\frac{1}{3}"}]) == [r"\frac{1}{3}"]


def test_trl_rows_with_no_gold_stay_none():
    assert trl_golds([{"problem": "a", "answer": ""}, {"problem": "b", "answer": None}]) == [
        None,
        None,
    ]
    assert trl_golds([{"problem": "a", "final_answer": ""}]) == [None]
    assert trl_golds([{"problem": "a", "answer": "0"}]) == ["0"]


@pytest.mark.parametrize("answer, prediction, gold", [
    ("72 eggs", "The answer is 72.", "72"),
    ("The answer is 42", "The answer is 42.", "42"),
    ("$1,234", "The answer is 1234.", "1234"),
    ("-5 degrees", "The answer is -5.", "-5"),
    ("It is -5", "The answer is -5.", "-5"),
    ("3.5 km", "The answer is 3.5.", "3.5"),
])
def test_one_number_in_plain_text_is_the_gold(answer, prediction, gold):
    assert extract_counterfact_gold({"answer": answer}) == gold
    assert grade_math_answer(extract_math_answer(prediction), gold)


@pytest.mark.parametrize("answer", [
    "x^2+1",
    "3x",
    "x - 1",
    "x = 5",
    "1, 2",
    "1,2 or 3",
    "3 apples and 4 pears",
    "Even",
])
def test_text_that_is_not_one_plain_number_stays_whole(answer):
    assert extract_counterfact_gold({"answer": answer}) == answer


@pytest.mark.parametrize("row", [{"answer": "#### 1/3"}, {"answer": "", "solution": "#### 1/3"}])
def test_gsm8k_marker_keeps_a_fraction(row):
    gold = extract_counterfact_gold(row)
    assert gold == "1/3"
    assert grade_math_answer(extract_math_answer("The answer is 1/3."), gold)


@pytest.mark.parametrize("completion", ["so 0.33333", "so 1/3"])
def test_unsloth_reward_reads_a_fraction_gold(completion):
    from aligntune.backends.unsloth.rl.counterfact_grpo.counterfact_grpo import (
        UnslothCounterFactGRPOTrainer,
    )

    trainer = object.__new__(UnslothCounterFactGRPOTrainer)
    trainer.prompt_to_answer = {"p": extract_counterfact_gold({"answer": "#### 1/3"})}
    assert trainer._combined_reward_function([completion], prompts=["p"]) == [1.0]
    assert trainer._combined_reward_function(["so 0.5"], prompts=["p"]) == [0.0]
