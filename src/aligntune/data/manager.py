import json
import logging
from typing import Optional, Dict, Union, Any, Callable
from datasets import Dataset, DatasetDict
from aligntune.data.loaders.resolver import LoaderResolver
from aligntune.data.schemas import TaskType
from aligntune.data.processors import (
    ColumnMapper,
    SystemPromptInjector,
)
from aligntune.data.curator_adapter import curate_split, restore_accepted_rows

logger = logging.getLogger(__name__)

class DataManager:
    """
    High-level dataset orchestration around CuratorKIT and trainer formatting.

    CuratorKIT owns format detection and row preprocessing.  AlignTune applies
    the explicit column mapping, final trainer normalization, and system prompt
    after CuratorKIT returns its canonical samples.
    """
    
    def __init__(
        self,
        task_type: Union[str, TaskType],
        column_mapping: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        tokenizer=None,  # NEW: Add tokenizer parameter
        enable_thinking: bool = False,  # NEW: Add enable_thinking parameter
        processing_fn: Optional[Callable] = None,
        processing_batched: bool = False,
        val_split_ratio: Optional[float] = None,
        test_split_ratio: Optional[float] = None,
        seed: int = 42,
        max_samples: int = None,
        max_length: int = None,
        auto_detect: bool = True,  # Retained for backwards-compatible callers
        lmbda: float = 0.0,  # Online (1.0) vs offline (0.0) distillation schema selection
        curator_schema_gate: bool = True,
        curator_clean: bool = False,
        curator_dedup: str = "none",
        curator_use_tiktoken: bool = False,
        curator_max_tokens: int = 1_000_000,
        expected_format: Optional[str] = None,
        privileged_context_column: Optional[str] = None,
        keep_columns: Optional[bool] = None,
    ):
        # Map preference algorithms to DPO schema (except KTO which has its own schema)
        if isinstance(task_type, str):
            task_type = task_type.lower()

            # Offline Preference Optimization variants (require: prompt, chosen, rejected)
            if task_type in ["orpo", "simpo", "spin", "cgpo", "online_dpo"]:
                task_type = "dpo"

            # Online Actor-Critic / Baseline variants (require: prompt)
            elif task_type in ["rloo"]:
                task_type = "ppo"

            # Online Generative / Reward-based variants (require: prompt, optionally response/solution)
            elif task_type in ["dapo", "gspo", "bolt", "counterfact_grpo", "dr_grpo", "gbmpo", "meta_es", "neural_mirror_grpo", "pace", "online_dpo"]:
                task_type = "grpo"

            # NEW: Distillation variants
            elif task_type in ["gold", "distillation", "standard_distillation", "standard"]:
                # trl's GOLDTrainer/DistillationTrainer both do their own
                # online/offline mixing internally via `args.lmbda` (a random
                # per-example draw at train time - see self.lmbda in trl's
                # gold_trainer.py/distillation_trainer.py), and both backends
                # already forward config.train.lmbda straight into the trl
                # config. They always need the full prompt+completion (user+
                # assistant) shape to know the prompt boundary, even when
                # lmbda=1.0 causes every example to be generated on-policy -
                # the completion is only ever discarded, never the prompt.
                # Selecting a prompt-only "distillation_online" schema here
                # produced datasets with an empty prompt, which crashed
                # generation (`IndexError: index -1 is out of bounds for
                # dimension 1 with size 0` in trl's own GOLDTrainer) or
                # produced garbage logits (`probability tensor contains inf/
                # nan`) in standard distillation's online path.
                task_type = "distillation_offline"

            elif task_type in ["sdft", "self_distillation_ft"]:
                task_type = "distillation_sdft"

            elif task_type in ["sdpo", "self_distillation_po"]:
                task_type = "distillation_sdpo"
                
        # Handle if task_type is already a TaskType enum
        if isinstance(task_type, TaskType):
            self.task_type = task_type
        elif hasattr(task_type, 'value'):
            # Handle enum from different import location - extract string value
            self.task_type = TaskType(task_type.value)
        else:
            self.task_type = TaskType(task_type)

        self.user_processing_fn = processing_fn  # Store user function
        self.processing_batched = processing_batched
        self.auto_detect = auto_detect
        self.curator_schema_gate = curator_schema_gate
        self.curator_clean = curator_clean
        if not isinstance(curator_dedup, str):
            raise TypeError("curator_dedup must be a string: 'none', 'exact', or 'minhash'")
        curator_dedup = curator_dedup.strip().lower()
        if curator_dedup not in {"none", "exact", "minhash"}:
            raise ValueError(
                "curator_dedup must be one of: 'none', 'exact', or 'minhash'"
            )
        self.curator_dedup = curator_dedup
        self.curator_use_tiktoken = curator_use_tiktoken
        self.curator_max_tokens = curator_max_tokens
        self.expected_format = expected_format
        self.keep_columns = (
            self._keep_original_columns()
            if keep_columns is None
            else keep_columns
        )
        self.privileged_context_column = privileged_context_column
        self.tokenizer = tokenizer  # NEW: Store tokenizer
        self.enable_thinking = enable_thinking  # NEW: Store enable_thinking

        # Initialize Processors Pipeline
        self.mapper = ColumnMapper(self.task_type, column_mapping)
        
        # NEW: Pass tokenizer and enable_thinking to SystemPromptInjector
        self.injector = SystemPromptInjector(
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            task_type=self.task_type,
        )
        
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.max_samples = max_samples

        self._validate_split_ratios()

    def _keep_original_columns(self) -> bool:
        """Return whether task-specific source columns must be preserved."""
        return self.task_type.value in {
            "grpo",
            "distillation_sdft",
            "distillation_sdpo",
        }

    def _apply_privileged_context(self, dataset: Dataset) -> Dataset:
        """Add the canonical privileged-context field for self-distillation."""
        if self.task_type.value not in {
            "distillation_sdft",
            "distillation_sdpo",
        }:
            return dataset

        aliases = (
            "privileged_context",
            "context",
            "demonstration",
            "hint",
            "feedback",
            "few_shot",
            "example",
            "reference",
            "environment_feedback",
            "explanation",
            "error_message",
        )
        source = next(
            (
                name
                for name in (self.privileged_context_column, *aliases)
                if name and name in dataset.column_names
            ),
            None,
        )
        if source is None or source == "privileged_context":
            return dataset

        return dataset.add_column(
            "privileged_context",
            dataset[source],
        ) if "privileged_context" not in dataset.column_names else dataset

    @staticmethod
    def _canonical_dataset(samples: list[Any]) -> Dataset:
        """Convert CuratorKIT DataSamples into an HF Dataset."""
        rows = []
        for sample in samples:
            if hasattr(sample, "model_dump"):
                rows.append(sample.model_dump())
            else:
                rows.append(dict(sample))
        DataManager._stabilize_metadata_types(rows)
        return Dataset.from_list(rows)

    @staticmethod
    def _stabilize_metadata_types(rows: list[dict]) -> None:
        """Normalize per-row ``metadata`` values that vary in type across rows.

        CuratorKIT carries source-dataset columns it doesn't otherwise map
        straight into each sample's ``metadata`` dict. For some source
        datasets (e.g. argilla/distilabel-intel-orca-dpo-pairs's ``rating``
        column) the same key arrives as a list for some rows and a string/
        float for others, which pyarrow's schema inference in
        ``Dataset.from_list`` rejects outright (``cannot mix list and
        non-list, non-null values``). ``metadata`` is auxiliary/debug
        information only, never read by trainers, so once a key's type is
        inconsistent across the batch, flatten every value for that key to
        a JSON string instead of letting pyarrow fail the whole load.
        """
        type_by_key: Dict[str, set] = {}
        for row in rows:
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    type_by_key.setdefault(key, set()).add(type(value))

        unstable_keys = {key for key, types in type_by_key.items() if len(types) > 1}
        if not unstable_keys:
            return

        for row in rows:
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                for key in unstable_keys:
                    if key in metadata:
                        metadata[key] = json.dumps(metadata[key])

    # Columns CuratorKIT's format detectors expect as plain text. Some HF
    # datasets (e.g. trl-lib/kto-mix-14k) ship these as chat-format
    # `[{"role": ..., "content": ...}]` lists instead - CuratorKIT's layer-2
    # value validation rejects non-string values outright, so flatten them
    # to plain text before curation ever sees the dataset.
    _CURATOR_TEXT_COLUMNS = ("prompt", "completion", "chosen", "rejected", "instruction", "output", "input")

    @classmethod
    def _flatten_chat_format_columns(cls, dataset: Dataset) -> Dataset:
        """Flatten `[{"role": ..., "content": ...}]`-style columns to plain text."""
        if len(dataset) == 0:
            return dataset

        def is_chat_list(value: Any) -> bool:
            return (
                isinstance(value, list) and len(value) > 0
                and all(isinstance(turn, dict) and "content" in turn for turn in value)
            )

        columns_to_flatten = [
            column for column in cls._CURATOR_TEXT_COLUMNS
            if column in dataset.column_names and is_chat_list(dataset[0].get(column))
        ]
        if not columns_to_flatten:
            return dataset

        def flatten_row(example: Dict[str, Any]) -> Dict[str, Any]:
            for column in columns_to_flatten:
                value = example.get(column)
                if is_chat_list(value):
                    example[column] = "\n\n".join(str(turn.get("content", "")) for turn in value)
            return example

        return dataset.map(flatten_row, desc="Flattening chat-format columns for curation")

    def _run_curator(self, dataset: Dataset) -> Dataset:
        """Run CuratorKIT and return canonical or source-preserving rows."""
        # CuratorKIT's format validators require plain text, but trainers such
        # as SDPO/GRPO can consume conversational prompts as message lists.
        # Curate a flattened copy while retaining the original dataset for
        # accepted-row restoration; otherwise the message structure would be
        # irreversibly converted to a string before the trainer sees it.
        source_dataset = dataset
        curator_dataset = self._flatten_chat_format_columns(dataset)
        curator_format = self.expected_format or (
            "alpaca"
            if self.task_type == TaskType.DISTILLATION_OFFLINE
            else "auto"
        )
        result = curate_split(
            curator_dataset,
            task_type=self.task_type.value,
            field_mapping=self.mapper.user_mapping,
            preprocessing_fn=self.user_processing_fn,
            preprocessing_batched=self.processing_batched,
            format=curator_format,
            schema_gate=self.curator_schema_gate,
            clean=self.curator_clean,
            dedup=self.curator_dedup,
            use_tiktoken=self.curator_use_tiktoken,
            max_tokens=self.curator_max_tokens,
        )

        # CuratorKIT's canonical DataSample flattens ShareGPT conversations to
        # prompt/completion. Preserve accepted source rows long enough for the
        # SFT normalizer to rebuild their complete ``messages`` sequence.
        preserve_sft_conversations = (
            self.task_type == TaskType.SFT
            and any(
                column in dataset.column_names
                for column in ("messages", "conversations", "conversation")
            )
        )
        if not self.keep_columns and not preserve_sft_conversations:
            return self._canonical_dataset(result.passed)

        # Keep original requested columns, then add Curator's canonical fields
        # so ColumnMapper can still produce the trainer schema.
        restored = restore_accepted_rows(source_dataset, result.accepted_ids)
        canonical_rows = [sample.model_dump() for sample in result.passed]
        if len(restored) != len(canonical_rows):
            raise RuntimeError(
                "CuratorKIT accepted-row count does not match restored-row count"
            )

        def add_canonical_fields(
            row: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            row.update(canonical_rows[index])
            return row

        return restored.map(add_canonical_fields, with_indices=True)

    def _validate_split_ratios(self) -> None:
        """Validate optional validation and test split ratios."""
        for name, ratio in (
            ("val_split_ratio", self.val_split_ratio),
            ("test_split_ratio", self.test_split_ratio),
        ):
            if ratio is not None and not 0 < ratio < 1:
                raise ValueError(f"{name} must be between 0 and 1, got {ratio!r}")

        if (
            self.val_split_ratio is not None
            and self.test_split_ratio is not None
            and self.val_split_ratio + self.test_split_ratio >= 1
        ):
            raise ValueError("val_split_ratio + test_split_ratio must be less than 1")

    @staticmethod
    def _as_dataset_dict(
        raw_data: Union[Dataset, DatasetDict], single_split_name: str = "train"
    ) -> DatasetDict:
        """Normalize a single HF Dataset or DatasetDict to DatasetDict."""
        if isinstance(raw_data, Dataset):
            return DatasetDict({single_split_name: raw_data})
        if isinstance(raw_data, DatasetDict):
            return raw_data
        raise TypeError(
            "Dataset loader must return datasets.Dataset or datasets.DatasetDict, "
            f"got {type(raw_data).__name__}"
        )

    @staticmethod
    def _normalize_split_names(dataset_dict: DatasetDict) -> DatasetDict:
        """Normalize common train/validation/test split aliases."""
        aliases = {
            "train": {"train", "training"},
            "validation": {"validation", "valid", "val", "dev", "development"},
            "test": {"test", "testing", "eval", "evaluation"},
        }
        normalized: Dict[str, Dataset] = {}
        consumed = set()

        for target, names in aliases.items():
            for name in dataset_dict:
                if name.lower() in names and target not in normalized:
                    normalized[target] = dataset_dict[name]
                    consumed.add(name)
                    break

        for name, dataset in dataset_dict.items():
            if name not in consumed:
                normalized[name] = dataset

        return DatasetDict(normalized)

    def _limit_split(self, dataset: Dataset, split_name: str) -> Dataset:
        """Apply the configured maximum sample count to one split."""
        if self.max_samples is None or len(dataset) <= self.max_samples:
            return dataset
        logger.info(
            "Limiting %s split from %s to %s samples",
            split_name,
            len(dataset),
            self.max_samples,
        )
        return dataset.select(range(self.max_samples))

    def _split_once(
        self, dataset: Dataset, test_size: float, split_name: str
    ) -> tuple[Dataset, Dataset]:
        """Split one dataset deterministically and provide a useful small-data error."""
        if len(dataset) < 2:
            raise ValueError(
                f"Cannot create {split_name} split from a dataset with {len(dataset)} row"
            )
        parts = dataset.train_test_split(test_size=test_size, seed=self.seed)
        return parts["train"], parts["test"]

    def _apply_split_policy(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Apply explicit train/validation/test ratio rules before training."""
        dataset_dict = self._normalize_split_names(dataset_dict)
        dataset_dict = DatasetDict(
            {
                name: self._limit_split(dataset, name)
                for name, dataset in dataset_dict.items()
            }
        )

        has_train = "train" in dataset_dict
        has_validation = "validation" in dataset_dict
        has_test = "test" in dataset_dict

        if not has_train:
            return dataset_dict

        splits = dict(dataset_dict)

        # Existing train/validation/test splits are authoritative.
        if has_validation and has_test:
            return DatasetDict(splits)

        # If validation exists and only a test ratio is requested, split validation.
        if has_validation and not has_test and self.test_split_ratio is not None:
            validation, test = self._split_once(
                splits["validation"], self.test_split_ratio, "test"
            )
            splits["validation"] = validation
            splits["test"] = test
            return DatasetDict(splits)

        # If test exists and only a validation ratio is requested, split train.
        if has_test and not has_validation and self.val_split_ratio is not None:
            train, validation = self._split_once(
                splits["train"], self.val_split_ratio, "validation"
            )
            splits["train"] = train
            splits["validation"] = validation
            return DatasetDict(splits)

        # With only train, create whichever explicitly requested splits exist.
        if not has_validation and not has_test:
            train = splits["train"]
            if self.val_split_ratio is not None:
                train, validation = self._split_once(
                    train, self.val_split_ratio, "validation"
                )
                splits["train"] = train
                splits["validation"] = validation

            if self.test_split_ratio is not None:
                remaining_test_ratio = self.test_split_ratio
                if self.val_split_ratio is not None:
                    remaining_test_ratio = self.test_split_ratio / (
                        1 - self.val_split_ratio
                    )
                train, test = self._split_once(train, remaining_test_ratio, "test")
                splits["train"] = train
                splits["test"] = test

        return DatasetDict(splits)
    
    def load_dataset(
        self,
        dataset_name_or_path: Union[str, Dataset, DatasetDict],
        split: Optional[str] = None,
        **loader_kwargs,
    ) -> DatasetDict:
        """Load and process a dataset, optionally selecting one split first."""
        # 1. Load
        # Map 'subset' to 'config_name' so HFLoader receives the correct argument
        if 'subset' in loader_kwargs and 'config_name' not in loader_kwargs:
            loader_kwargs['config_name'] = loader_kwargs.pop('subset')
        if isinstance(dataset_name_or_path, (Dataset, DatasetDict)):
            raw_data = dataset_name_or_path
        else:
            loader = LoaderResolver.resolve(
                dataset_name_or_path, split=split, **loader_kwargs
            )
            raw_data = loader.load()

        # A loader returns a single Dataset when ``split`` is specified. Keep
        # that name instead of silently relabeling it as ``train``.
        raw_data = self._as_dataset_dict(raw_data, single_split_name=split or "train")

        if split is not None:
            raw_data = self._normalize_split_names(raw_data)
            if split not in raw_data:
                raise ValueError(
                    f"Requested split {split!r} was not found. "
                    f"Available splits: {list(raw_data.keys())}"
                )
            raw_data = DatasetDict({split: raw_data[split]})

        # Establish the final train/validation/test splits before any
        # CuratorKIT filtering. Each final split must be curated independently.
        raw_data = self._apply_split_policy(raw_data)

        # CuratorKIT owns format detection and row preprocessing.  AlignTune
        # only performs the final trainer normalization and prompt injection.
        processed_splits = {}
        for split_name, dataset in raw_data.items():
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split ({len(dataset)} examples)")
            print(f"{'='*60}")

            dataset = self._run_curator(dataset)
            dataset = self._apply_privileged_context(dataset)

            # Normalize CuratorKIT canonical fields for the trainer.
            dataset = self.mapper.process(dataset)

            if self.task_type == TaskType.PRETRAINING:
                dataset = dataset.select_columns(["text"])
            
            # Pretraining consumes raw causal-LM text. It must never receive a
            # system message, conversation wrapper, or chat-template kwargs.
            if self.task_type != TaskType.PRETRAINING:
                dataset = self.injector.process(dataset)
            
            print(f"Final columns: {dataset.column_names}")
            processed_splits[split_name] = dataset
        
        dataset_dict = DatasetDict(processed_splits)
        return dataset_dict
