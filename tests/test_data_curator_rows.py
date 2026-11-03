"""Regression tests: rows that come back from CuratorKIT must keep their data.

Each case loads a small local file through ``DataManager``; nothing is downloaded.
"""

import csv
import json

import pytest

from aligntune.data.manager import DataManager


def _orca_rows(n=12):
    # Shaped like argilla/distilabel-intel-orca-dpo-pairs: ``input`` is the prompt,
    # and ``rating`` is a list on most rows and null on ties.
    return [
        {
            "system": "",
            "input": f"Question {i}: what is {i} plus {i}?",
            "chosen": f"{i} plus {i} is {2 * i}.",
            "rejected": "I don't know.",
            "rating": None if i % 4 == 0 else [10.0, 2.0],
            "status": "tie" if i % 4 == 0 else "changed",
        }
        for i in range(n)
    ]


@pytest.fixture
def orca_file(tmp_path):
    path = tmp_path / "orca.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _orca_rows()) + "\n")
    return str(path)


def _blank(dataset, column):
    return sum(1 for value in dataset[column] if not str(value or "").strip())


@pytest.mark.parametrize("task_type", ["grpo", "dpo"])
def test_keep_columns_keeps_the_input_column_and_null_metadata(orca_file, task_type):
    # The canonical row's blank ``input`` erased the dataset's own ``input`` (the
    # prompt), and null ``rating`` cells came back as "" beside lists (Arrow error).
    train = DataManager(task_type=task_type, keep_columns=True).load_dataset(orca_file)["train"]

    assert len(train) == 12
    assert _blank(train, "prompt") == 0
    assert _blank(train, "input") == 0


def test_sft_with_input_mapped_to_prompt_has_real_prompts(orca_file):
    # CuratorKIT puts the mapped text under ``instruction`` and leaves ``prompt``
    # blank, so every user message was empty.
    train = DataManager(task_type="sft", column_mapping={"input": "prompt"}).load_dataset(orca_file)["train"]

    users = [next(m["content"] for m in row if m["role"] == "user") for row in train["messages"]]
    assert all(user.startswith("Question ") for user in users)


@pytest.mark.parametrize("name", ["supervised_fine_tuning", "text_generation", "chat_completion"])
def test_trainer_level_sft_task_names_are_accepted(name):
    assert DataManager(task_type=name).task_type.value == "sft"


def test_csv_chat_lists_are_read_as_text(tmp_path):
    # A CSV stores a chat list as its Python repr; KTO saw "[{'content': ...".
    path = tmp_path / "kto.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "completion", "label"])
        writer.writeheader()
        for i in range(8):
            writer.writerow({
                "prompt": str([{"content": f"What is {i} plus {i}?", "role": "user"}]),
                "completion": str([{"content": f"It is {2 * i}.", "role": "assistant"}]),
                "label": i % 2 == 0,
            })

    train = DataManager(task_type="kto").load_dataset(str(path))["train"]

    assert train[0]["prompt"].startswith("What is 0 plus 0?")
    assert not any(str(value).startswith("[{") for value in train["prompt"])
