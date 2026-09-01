"""Shared dataset normalization for sequence and token classification.

Classification trainers use ``transformers.Trainer`` rather than TRL's
causal-language-model SFT trainer.  They therefore need a small, explicit
normalization step that produces ``text``/``labels`` or ``tokens``/``labels``
before tokenization.
"""

from __future__ import annotations

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
            # Sort names so label IDs do not depend on dataset shuffle order.
            classes = sorted({str(value) for value in flat})
            self.mapping = {name: index for index, name in enumerate(classes)}
        self.fitted = True
        return self

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
