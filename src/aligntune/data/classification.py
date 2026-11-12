"""Shared dataset normalization for sequence and token classification.

Classification trainers use ``transformers.Trainer`` rather than TRL's
causal-language-model SFT trainer.  They therefore need a small, explicit
normalization step that produces ``text``/``labels`` or ``tokens``/``labels``
before tokenization.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Dict, Iterable, Optional, Tuple

from datasets import Dataset


def _canonical_mapping(column_mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return destination -> source mappings, accepting the old inverse form."""
    mapping = dict(column_mapping or {})
    canonical = {"text", "label", "labels", "tokens", "ner_tags", "tags"}
    normalized: Dict[str, str] = {}
    for key, value in mapping.items():
        if key in canonical:
            normalized[key] = value
        elif value in canonical:
            # Legacy classification code accepted {source: destination}.
            normalized[value] = key
        else:
            normalized[key] = value
    return normalized


def _source_column(
    columns: Iterable[str],
    mapping: Dict[str, str],
    names: tuple[str, ...],
    default: str,
) -> str:
    columns = set(columns)
    for name in names:
        source = mapping.get(name)
        if source:
            if source not in columns:
                raise ValueError(
                    f"Column mapping for '{name}' points to missing column '{source}'. "
                    f"Available columns: {sorted(columns)}"
                )
            return source
    if default in columns:
        return default
    raise ValueError(
        f"Required classification column '{default}' was not found. "
        f"Available columns: {sorted(columns)}"
    )


def _parse_number(name: str) -> Optional[Any]:
    """Return the number ``name`` spells ("2", "-1", "10.0", "1e3"), or None for any other text."""
    text = name.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None  # "nan" and "inf" are names, not numbers


def _order_labels(names: Iterable[str]) -> list[str]:
    """One rule for every label type (int, float, text): if every label spells a number, sort by
    value ("1" < "2" < "10", "1.0" < "2.0" < "10.0"); otherwise sort as plain text."""
    names = list(names)
    parsed = {name: _parse_number(name) for name in names}
    if all(value is not None for value in parsed.values()):
        return sorted(names, key=lambda name: (parsed[name], name))
    return sorted(names)


class ClassificationLabelEncoder:
    """Stable encoder shared by train and evaluation splits."""

    def __init__(self) -> None:
        self.mapping: Optional[Dict[str, int]] = None
        self.offset = 0
        self.fitted = False

    @staticmethod
    def _flatten(values: Iterable[Any]) -> list[Any]:
        flat: list[Any] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                flat.extend(ClassificationLabelEncoder._flatten(value))
            elif value != -100:
                flat.append(value)
        return flat

    def fit(self, values: Iterable[Any]) -> "ClassificationLabelEncoder":
        flat = self._flatten(values)
        if not flat:
            raise ValueError("Classification dataset contains no valid labels.")

        numeric = all(isinstance(value, Integral) and not isinstance(value, bool) for value in flat)
        if numeric:
            # Preserve normal 0..N-1 labels; compact contiguous 1..N (or
            # k..k+N) datasets without changing sparse IDs.
            unique = sorted(set(int(value) for value in flat))
            contiguous = unique == list(range(unique[0], unique[-1] + 1))
            self.offset = unique[0] if unique[0] > 0 and contiguous else 0
            self.mapping = None
        else:
            # Sort names so label IDs do not depend on dataset shuffle order;
            # numeric labels (float 10.0, text "10") sort by value, not as text.
            classes = _order_labels({str(value) for value in flat})
            self.mapping = {name: index for index, name in enumerate(classes)}
        self.fitted = True
        return self

    def label_names(self) -> Optional[Dict[int, str]]:
        """id -> name for text labels (for model.config.id2label); None for integer labels."""
        if self.mapping is None:
            return None
        return {index: name for name, index in self.mapping.items()}

    def encode(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        if value == -100:
            return -100
        if self.mapping is not None:
            key = str(value)
            if key not in self.mapping:
                raise ValueError(f"Evaluation label '{value}' was not present in the training labels.")
            return self.mapping[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, Integral):
            encoded = int(value) - self.offset
            if encoded < 0:
                raise ValueError(f"Label '{value}' becomes negative after normalization.")
            return encoded
        raise ValueError(f"Unsupported classification label value: {value!r}")


def normalize_text_classification(
    dataset: Dataset,
    *,
    column_mapping: Optional[Dict[str, str]] = None,
    text_column: str = "text",
    label_column: str = "label",
    encoder: Optional[ClassificationLabelEncoder] = None,
) -> Tuple[Dataset, ClassificationLabelEncoder]:
    """Normalize arbitrary scalar-label data to ``text`` and ``labels``."""
    mapping = _canonical_mapping(column_mapping)
    text_source = _source_column(dataset.column_names, mapping, ("text",), text_column)
    label_source = _source_column(dataset.column_names, mapping, ("labels", "label"), label_column)
    encoder = encoder or ClassificationLabelEncoder()
    raw_labels = dataset[label_source]
    if not encoder.fitted:
        encoder.fit(raw_labels)

    def convert(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": str(example[text_source]), "labels": encoder.encode(example[label_source])}

    return dataset.map(convert, remove_columns=dataset.column_names), encoder


def normalize_token_classification(
    dataset: Dataset,
    *,
    column_mapping: Optional[Dict[str, str]] = None,
    tokens_column: str = "tokens",
    tags_column: str = "ner_tags",
    encoder: Optional[ClassificationLabelEncoder] = None,
) -> Tuple[Dataset, ClassificationLabelEncoder]:
    """Normalize token/tag data to ``tokens`` and aligned ``labels``."""
    mapping = _canonical_mapping(column_mapping)
    tokens_source = _source_column(dataset.column_names, mapping, ("tokens",), tokens_column)
    labels_source = _source_column(
        dataset.column_names, mapping, ("labels", "ner_tags", "tags"), tags_column
    )
    encoder = encoder or ClassificationLabelEncoder()
    raw_labels = dataset[labels_source]
    if not encoder.fitted:
        encoder.fit(raw_labels)

    def convert(example: Dict[str, Any]) -> Dict[str, Any]:
        tokens = example[tokens_source]
        labels = encoder.encode(example[labels_source])
        if len(tokens) != len(labels):
            raise ValueError(
                f"Token/label length mismatch: {len(tokens)} tokens vs {len(labels)} labels."
            )
        return {"tokens": tokens, "labels": labels}

    return dataset.map(convert, remove_columns=dataset.column_names), encoder


def attach_label_names(model: Any, encoder: Optional[ClassificationLabelEncoder], *datasets: Optional[Dataset]) -> None:
    """Check the encoded labels fit the model's head and keep the class names on the model.

    ``datasets`` are the already-normalized splits (``labels`` column). More classes than
    ``model.config.num_labels`` raises ``ValueError`` (training would otherwise fail later with
    an index error, or silently drop classes). Text labels' id -> name mapping is saved to
    ``model.config.id2label``/``label2id`` so predictions can be read back.
    """
    labels = [
        value
        for dataset in datasets
        if dataset is not None
        for value in ClassificationLabelEncoder._flatten(dataset["labels"])
    ]
    config = getattr(model, "config", None)
    num_labels = getattr(config, "num_labels", None)
    if labels and num_labels:
        num_classes = max(int(value) for value in labels) + 1
        if num_classes > num_labels:
            raise ValueError(
                f"The labels need {num_classes} classes but the model has num_labels={num_labels}. "
                "Set model.num_labels to at least the number of classes."
            )
    names = encoder.label_names() if encoder is not None else None
    if names and config is not None:
        # transformers derives num_labels from len(id2label), so pad to the head's size:
        # fewer names than outputs must not shrink num_labels, or the saved config no
        # longer matches the classifier weights and the model won't reload.
        size = max(num_labels or 0, len(names))
        config.id2label = {index: names.get(index, f"LABEL_{index}") for index in range(size)}
        config.label2id = {name: index for index, name in config.id2label.items()}
