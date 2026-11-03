"""ORPO: a tokenized row whose chosen or rejected completion has no tokens makes the loss NaN.

TRL averages a completion's log-probabilities over its label tokens, so a row with none
is 0/0. drop_rows_without_completion_tokens removes such rows. No model is loaded.
"""

from datasets import Dataset

from aligntune.backends.trl.rl.orpo.orpo import drop_rows_without_completion_tokens


def _row(chosen_tokens, rejected_tokens):
    # labels as TRL builds them: -100 over the prompt, token ids over the completion
    return {
        "chosen_labels": [-100, -100] + [7] * chosen_tokens,
        "rejected_labels": [-100, -100] + [8] * rejected_tokens,
    }


def test_rows_without_a_completion_on_either_side_are_dropped():
    data = Dataset.from_list([_row(3, 4), _row(0, 4), _row(3, 0), _row(1, 1)])

    kept, dropped = drop_rows_without_completion_tokens(data)

    assert dropped == 2
    assert len(kept) == 2
    assert all(any(x != -100 for x in row["chosen_labels"]) for row in kept)


def test_a_clean_dataset_is_returned_untouched():
    data = Dataset.from_list([_row(2, 2), _row(5, 1)])

    kept, dropped = drop_rows_without_completion_tokens(data)

    assert dropped == 0 and kept is data


def test_data_without_label_columns_is_left_alone():
    data = Dataset.from_list([{"prompt": "a", "chosen": "b", "rejected": "c"}])

    assert drop_rows_without_completion_tokens(data) == (data, 0)
