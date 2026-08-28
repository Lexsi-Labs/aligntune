"""Small adapter between Hugging Face datasets and CuratorKIT.

CuratorKIT currently reads dataset names or files rather than an in-memory
``datasets.Dataset``.  This module keeps that implementation detail isolated
from ``DataManager`` by using a temporary Parquet file and returning the IDs
of rows that CuratorKIT accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Optional

from datasets import Dataset
from aligntune.data.column_schemas import find_task_fallback


@dataclass
class CuratorAdapterResult:
    """CuratorKIT output needed to rebuild the original HF split."""

    accepted_ids: set[str]
    rejected: list[Any]
    passed: list[Any]


def _with_stable_ids(dataset: Dataset) -> Dataset:
    """Return a copy with a unique ``id`` column required by CuratorKIT."""
    if "id" in dataset.column_names:
        existing_ids = [str(value) for value in dataset["id"]]
        if len(existing_ids) == len(set(existing_ids)):
            return dataset

    if "_aligntune_original_id" in dataset.column_names:
        raise ValueError(
            "Dataset already contains the reserved column "
            "'_aligntune_original_id'"
        )

    columns = dataset.column_names
    if "id" in columns:
        dataset = dataset.rename_column("id", "_aligntune_original_id")

    return dataset.add_column("id", [str(index) for index in range(len(dataset))])


def curate_split(
    dataset: Dataset,
    *,
    task_type: Optional[str] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    preprocessing_fn: Optional[Callable[..., Any]] = None,
    preprocessing_batched: bool = False,
    format: str = "auto",
    output_dir: str | Path | None = None,
    schema_gate: bool = True,
    clean: bool = False,
    dedup: str = "none",
    use_tiktoken: bool = False,
    max_tokens: int = 1_000_000,
) -> CuratorAdapterResult:
    """Run CuratorKIT on one in-memory HF split.

    The original dataset is not replaced.  The caller should use
    ``accepted_ids`` to filter its original split, which preserves custom RL
    and distillation columns that are not part of CuratorKIT's canonical
    ``DataSample``.

    Args:
        dataset: In-memory Hugging Face dataset split.
        task_type: AlignTune task used to scope fallback schemas.
        field_mapping: Raw-column to CuratorKIT canonical-field mapping.
        preprocessing_fn: Optional row preprocessing function. Returning
            ``None`` rejects that row in CuratorKIT.
        preprocessing_batched: Whether ``preprocessing_fn`` expects batches.
        format: CuratorKIT format or ``"auto"``.
        output_dir: Optional CuratorKIT output directory. A temporary
            directory is used when omitted.
        schema_gate: Enable CuratorKIT schema validation.
        clean: Enable CuratorKIT cleaning.
        dedup: CuratorKIT deduplication mode; ``"none"`` disables it.
        use_tiktoken: Use CuratorKIT's tiktoken counter when enabled.
        max_tokens: Maximum CuratorKIT schema length.

    Returns:
        Accepted row IDs and CuratorKIT's passed/rejected samples.

    Raises:
        ImportError: If CuratorKIT is not installed.
    """
    try:
        from curatorkit import Curator, CuratorConfig
    except ImportError as exc:
        raise ImportError(
            "CuratorKIT is required for dataset curation. "
            "Install it with `pip install curatorkit`."
        ) from exc

    if preprocessing_batched:
        # TODO(CuratorKIT): add buffered batch preprocessing in BaseConnector
        # while preserving row IDs, source line numbers, and rejection reasons.
        raise ValueError(
            "CuratorKIT preprocessing_fn receives one row at a time; "
            "batch preprocessing must be applied before curate_split."
        )

    working_dataset = _with_stable_ids(dataset)
    target_dir = Path(output_dir) if output_dir is not None else None

    with TemporaryDirectory() as temporary_dir:
        parquet_path = Path(temporary_dir) / "input.parquet"
        working_dataset.to_parquet(str(parquet_path))

        def run_once(
            selected_format: str, selected_mapping: Optional[Dict[str, str]]
        ) -> Any:
            """Run CuratorKIT once with a specific format and mapping."""
            config = CuratorConfig(
                dataset=str(parquet_path),
                split="train",
                format=selected_format,
                field_mapping=selected_mapping or {},
                preprocessing_fn=preprocessing_fn,
                schema_gate=schema_gate,
                min_tokens=0,
                max_tokens=max_tokens,
                schema_use_tiktoken=use_tiktoken,
                clean=clean,
                dedup=dedup,
                embedding_dedup=False,
                output_split=None,
                export_formats=[],
                output_dir=str(target_dir or Path(temporary_dir) / "curated"),
            )
            return Curator(config).run()

        result = run_once(format, field_mapping)

        # Auto-detection is attempted first.  A fallback is used only when no
        # row was accepted and the caller did not provide an explicit mapping.
        if not field_mapping and not result.passed and task_type:
            fallback = find_task_fallback(task_type, dataset.column_names)
            if fallback is not None:
                fallback_format, fallback_mapping = fallback
                result = run_once(fallback_format, fallback_mapping)

    accepted_ids = {str(sample.id) for sample in result.passed}
    return CuratorAdapterResult(
        accepted_ids=accepted_ids,
        rejected=result.rejected,
        passed=result.passed,
    )


def restore_accepted_rows(
    dataset: Dataset, accepted_ids: set[str]
) -> Dataset:
    """Filter the original split using CuratorKIT-accepted row IDs."""
    had_user_id = "id" in dataset.column_names
    working_dataset = _with_stable_ids(dataset)
    restored = working_dataset.filter(
        lambda row: str(row["id"]) in accepted_ids
    )

    if "_aligntune_original_id" in restored.column_names:
        restored = restored.remove_columns("id")
        return restored.rename_column("_aligntune_original_id", "id")

    if not had_user_id:
        return restored.remove_columns("id")

    return restored
