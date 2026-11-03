"""Regression test: Unsloth SFT dataset formatting must not pickle the trainer.

`datasets` fingerprints every `Dataset.map` transform by pickling it, closure
included. When the formatting function closed over the trainer, that pickled
`self.model` too, copying every parameter into host RAM (tens of GB for a 4B
model). Runs on CPU: the GPU-only pieces of setup_trainer() are stubbed.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

datasets = pytest.importorskip("datasets")
pytest.importorskip("trl")

from aligntune.backends.unsloth.sft.sft import UnslothSFTTrainer  # noqa: E402
from aligntune.core.registry import TaskType  # noqa: E402


class _PickleTripwire:
    """Stands in for the model; records whether anything tried to pickle it."""

    pickled = False

    def __reduce__(self):
        type(self).pickled = True
        return (_PickleTripwire, ())


class _ChatTokenizer:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)


def _make_trainer():
    trainer = UnslothSFTTrainer.__new__(UnslothSFTTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(max_seq_length=128),
        logging=SimpleNamespace(output_dir="unused"),
    )
    trainer.task_type = TaskType.SFT
    trainer.tokenizer = _ChatTokenizer()
    trainer.model = trainer.unsloth_model = _PickleTripwire()
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"say {i}"},
                {"role": "assistant", "content": str(i)},
            ]
        }
        for i in range(4)
    ]
    trainer.train_dataset = datasets.Dataset.from_list(rows)
    trainer.eval_dataset = None
    trainer.get_hf_callbacks = lambda: []
    trainer._setup_optimizer_scheduler = lambda **_: {
        "optimizer_name": "adamw_torch",
        "scheduler_name": "cosine",
        "optim_args_str": "",
        "scheduler_config": {"lr_scheduler_kwargs": {}},
    }
    return trainer


def test_message_formatting_does_not_pickle_the_model():
    _PickleTripwire.pickled = False
    trainer = _make_trainer()

    with (
        mock.patch(
            "aligntune.core.sft.training_args_builder.build_sft_config",
            return_value=SimpleNamespace(dataset_text_field="text"),
        ),
        mock.patch("trl.SFTTrainer") as sft_trainer,
    ):
        trainer.setup_trainer()

    train_dataset = sft_trainer.call_args.kwargs["train_dataset"]
    assert train_dataset["text"][1] == "user: say 1\nassistant: 1"
    assert not _PickleTripwire.pickled, "dataset.map() pickled the model"
