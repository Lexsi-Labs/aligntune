"""Class names reach the model config, numeric text labels sort as numbers, and a head too
small for the labels fails with a clear error."""
from types import SimpleNamespace

import pytest
from datasets import Dataset

from aligntune.data.classification import (
    ClassificationLabelEncoder,
    attach_label_names,
    normalize_text_classification,
)


def _model(num_labels):
    return SimpleNamespace(config=SimpleNamespace(num_labels=num_labels))


def test_numeric_text_labels_sort_as_numbers():
    enc = ClassificationLabelEncoder().fit(["10", "2", "1"])
    assert enc.label_names() == {0: "1", 1: "2", 2: "10"}


def test_integer_labels_have_no_names():
    assert ClassificationLabelEncoder().fit([0, 1, 2]).label_names() is None


def test_text_label_names_are_kept_on_the_model():
    ds, enc = normalize_text_classification(Dataset.from_dict({"text": ["a", "b", "c"], "label": ["pos", "neg", "pos"]}))
    model = _model(2)
    attach_label_names(model, enc, ds, None)
    assert model.config.id2label == {0: "neg", 1: "pos"}
    assert model.config.label2id == {"neg": 0, "pos": 1}


def test_more_classes_than_the_head_raises():
    ds, enc = normalize_text_classification(Dataset.from_dict({"text": ["a", "b", "c"], "label": ["x", "y", "z"]}))
    with pytest.raises(ValueError, match="need 3 classes .* num_labels=2"):
        attach_label_names(_model(2), enc, ds)


def test_eval_split_labels_count_too():
    train, enc = normalize_text_classification(Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}))
    evaluation, _ = normalize_text_classification(Dataset.from_dict({"text": ["c"], "label": [2]}), encoder=enc)
    with pytest.raises(ValueError):
        attach_label_names(_model(2), enc, train, evaluation)


def test_no_model_or_encoder_is_a_no_op():
    ds, _ = normalize_text_classification(Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}))
    attach_label_names(None, None, ds)


def test_numeric_sort_does_not_crash_on_digit_like_text():
    enc = ClassificationLabelEncoder().fit(["--1", "²", "3", "-2"])
    assert enc.label_names() == {0: "-2", 1: "3", 2: "--1", 3: "²"}


def test_fewer_names_than_outputs_keeps_num_labels_and_reloads(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32, num_labels=5)
    model = transformers.BertForSequenceClassification(config)
    ds, enc = normalize_text_classification(Dataset.from_dict({"text": ["a", "b", "c"], "label": ["x", "y", "z"]}))
    attach_label_names(model, enc, ds)
    assert model.config.num_labels == 5
    assert model.config.id2label == {0: "x", 1: "y", 2: "z", 3: "LABEL_3", 4: "LABEL_4"}
    model.save_pretrained(tmp_path)
    reloaded = transformers.BertForSequenceClassification.from_pretrained(tmp_path)
    assert reloaded.config.id2label[2] == "z" and reloaded.config.num_labels == 5


def test_text_classification_setup_data_attaches_names(monkeypatch):
    """Through the trainer: deleting the attach_label_names call in setup_data fails this."""
    import aligntune.core.sft.dataset_builder as builder
    from aligntune.backends.trl.sft.sft_classification import TextClassificationTrainer

    train, enc = normalize_text_classification(Dataset.from_dict({"text": ["a b", "c"], "label": ["neg", "pos"]}))
    monkeypatch.setattr(builder, "build_sft_datasets",
                        lambda config, tokenizer, task_type, return_label_encoder=False: (train, None, enc))

    class Tok:
        pad_token = "[PAD]"
        pad_token_id = 0

        def __call__(self, texts, **_):
            return {"input_ids": [[1] * len(t.split()) for t in texts]}

    trainer = TextClassificationTrainer.__new__(TextClassificationTrainer)
    trainer.config = SimpleNamespace(model=SimpleNamespace(max_seq_length=8))
    trainer.task_type = None
    trainer.tokenizer = Tok()
    trainer.model = _model(4)
    trainer.setup_data()
    assert trainer.model.config.id2label == {0: "neg", 1: "pos", 2: "LABEL_2", 3: "LABEL_3"}
    assert trainer.train_dataset["labels"] == [0, 1]


@pytest.mark.parametrize("module", [
    "src/aligntune/backends/trl/sft/sft_classification.py",
    "src/aligntune/backends/trl/sft/sft_token_classification.py",
    "src/aligntune/backends/unsloth/sft/sft_classification.py",
    "src/aligntune/backends/unsloth/sft/sft_token_classification.py",
])
def test_every_classification_trainer_attaches_names(module):
    # The Unsloth trainers can't be imported without unsloth, so check their source.
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / module).read_text()
    assert "attach_label_names(self.model, label_encoder" in source
