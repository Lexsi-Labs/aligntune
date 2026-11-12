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
    # "--1" and "²" are not numbers, so the whole set sorts as plain text.
    enc = ClassificationLabelEncoder().fit(["--1", "²", "3", "-2"])
    assert enc.label_names() == {0: "--1", 1: "-2", 2: "3", 3: "²"}


def test_float_labels_sort_as_numbers():
    enc = ClassificationLabelEncoder().fit([10.0, 2.0, 1.0])
    assert enc.label_names() == {0: "1.0", 1: "2.0", 2: "10.0"}
    assert [enc.encode(v) for v in (1.0, 2.0, 10.0)] == [0, 1, 2]


def test_mixed_int_text_and_float_labels_sort_as_numbers():
    enc = ClassificationLabelEncoder().fit([1, "2", 10.0, "-3", "1e2"])
    assert enc.label_names() == {0: "-3", 1: "1", 2: "2", 3: "10.0", 4: "1e2"}


def test_one_non_numeric_label_sorts_everything_as_text():
    enc = ClassificationLabelEncoder().fit(["10", "2", "pos"])
    assert enc.label_names() == {0: "10", 1: "2", 2: "pos"}


def test_nan_and_inf_are_names_not_numbers():
    enc = ClassificationLabelEncoder().fit(["10", "2", "nan"])
    assert enc.label_names() == {0: "10", 1: "2", 2: "nan"}


def test_integer_labels_keep_their_ids():
    enc = ClassificationLabelEncoder().fit([10, 2, 1, 2])
    assert enc.label_names() is None
    assert [enc.encode(v) for v in (10, 2, 1)] == [10, 2, 1]  # sparse integer ids are unchanged
    assert ClassificationLabelEncoder().fit([1, 2, 3]).encode(3) == 2  # contiguous 1..N compacts


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
    "src/aligntune/backends/trl/sft/Classification_trainer.py",
    "src/aligntune/backends/unsloth/sft/Classification_trainer.py",
])
def test_every_classification_trainer_attaches_names(module):
    # The Unsloth trainers can't be imported without unsloth, so check their source.
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / module).read_text()
    assert "attach_label_names(self.model, label_encoder" in source or "attach_label_names(self.model, encoder" in source


def _classification_config(label_column="label"):
    import aligntune.core.sft.config as sft_config

    dataset = SimpleNamespace(
        name="x", subset=None, split="train", max_samples=None, column_mapping=None,
        label_column=label_column, text_column="text", tokens_column="tokens", tags_column="ner_tags",
        task_type=sft_config.TaskType.TEXT_CLASSIFICATION,
    )
    return SimpleNamespace(dataset=dataset, model=SimpleNamespace(max_seq_length=8))


class _Tok:
    def __call__(self, texts, **_):
        return {"input_ids": [[1] * len(t.split()) for t in texts]}


def _exported_trainer(module_name, class_name, monkeypatch, rows):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: rows)
    trainer = getattr(module, class_name)(_classification_config())
    trainer.tokenizer = _Tok()
    return trainer


# Only the TRL class is called for real: importing the Unsloth module needs a GPU (it imports
# unsloth), so the Unsloth copy is covered by the source check below.
EXPORTED = [("aligntune.backends.trl.sft.Classification_trainer", "ClassificationTrainer")]


@pytest.mark.parametrize("module_name,class_name", EXPORTED)
def test_exported_classification_trainer_attaches_names(module_name, class_name, monkeypatch):
    """Deleting the attach_label_names call from ClassificationTrainer.setup_data fails this."""
    rows = Dataset.from_dict({"text": ["a b"] * 20, "label": ["neg", "pos"] * 10})
    trainer = _exported_trainer(module_name, class_name, monkeypatch, rows)
    trainer.model = _model(3)
    trainer.setup_data()
    assert trainer.model.config.id2label == {0: "neg", 1: "pos", 2: "LABEL_2"}
    assert trainer.model.config.label2id == {"neg": 0, "pos": 1, "LABEL_2": 2}


@pytest.mark.parametrize("module_name,class_name", EXPORTED)
def test_exported_classification_trainer_rejects_a_small_head(module_name, class_name, monkeypatch):
    rows = Dataset.from_dict({"text": ["a b"] * 21, "label": ["x", "y", "z"] * 7})
    trainer = _exported_trainer(module_name, class_name, monkeypatch, rows)
    trainer.model = _model(2)
    with pytest.raises(ValueError, match="need 3 classes .* num_labels=2"):
        trainer.setup_data()
