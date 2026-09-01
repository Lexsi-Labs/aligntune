"""
Abstract base trainer for Knowledge Distillation, inheriting from UnifiedTrainerBase.

Supports:
- Standard Distillation (offline/online with external teacher)
- SDFT (self-distillation fine-tuning)
"""

import logging
from pathlib import Path
from abc import abstractmethod
from typing import Dict, Any, Optional, List

from .config import DistillConfig
from ..trainer_base import UnifiedTrainerBase, TrainingState
from ..callbacks import TrainerCallback

logger = logging.getLogger(__name__)


class DistillTrainerBase(UnifiedTrainerBase):
    """
    Distillation-specific base trainer. Inherits lifecycle, saving, and callbacks from UnifiedTrainerBase.
    Adds support for student/teacher model management and distillation-specific setup.
    """

    # Child classes should define the task type for data loading
    TASK_TYPE: str = "distillation"
    KEEP_COLUMNS: bool = False

    def __init__(self, config: DistillConfig, callbacks: Optional[List[TrainerCallback]] = None):
        """Initialize distillation trainer."""
        super().__init__(config, callbacks)

        self.config = config

        # Distillation-specific models
        self.student_model = None
        self.teacher_model = None
        self.tokenizer = None

        # Dataset attributes
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset_dict = None

        logger.info(f"Initialized {self.__class__.__name__} with TASK_TYPE={self.TASK_TYPE}")

    def _get_student_for_io(self):
        """Return the trained student model used by save/Hub operations."""
        model = self.student_model if self.student_model is not None else self.model
        trainer = getattr(self, "trainer", None)
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            raise RuntimeError("Student model is not loaded.")
        return model

    def save_model(self, output_dir: Optional[str] = None) -> str:
        """Save the distilled student model and tokenizer."""
        logging_cfg = getattr(self.config, "logging", None)
        default_dir = getattr(logging_cfg, "output_dir", "./distill_output")
        save_path = Path(output_dir or default_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model = self._get_student_for_io()
        model.save_pretrained(str(save_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Distilled student saved to {save_path}")
        return str(save_path)

    def save_checkpoint(self) -> None:
        """Save a standard checkpoint using the distilled student model."""
        student = self._get_student_for_io()
        original_model = self.model
        self.model = student
        try:
            super().save_checkpoint()
        finally:
            self.model = original_model

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload distilled student model",
    ) -> str:
        """Push the distilled student model and tokenizer to the Hub."""
        student = self._get_student_for_io()
        original_model = self.model
        self.model = student
        try:
            return super().push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token,
                commit_message=commit_message,
            )
        finally:
            self.model = original_model

    @abstractmethod
    def setup_model(self) -> None:
        """
        Setup student and teacher models.
        Implemented by specific distillation trainer subclasses.
        """
        pass

    def setup_data(self) -> None:
        """
        Unified setup_data implementation for distillation trainers.
        Uses DataManager to load and process datasets based on TASK_TYPE.
        Override this method only if you need custom dataset handling.
        """
        logger.info("Setting up distillation datasets with unified DataManager...")

        dataset_config = self._extract_dataset_config()
        task_type = self._get_task_type(dataset_config)
        params = self._extract_dataset_params(dataset_config)
        keep_columns = (
            params['keep_columns']
            if params['keep_columns'] is not None
            else self.KEEP_COLUMNS
        )
        # Offline distillation accepts both ShareGPT conversations and plain
        # instruction/output pairs. Let CuratorKIT detect the source format
        # unless the caller explicitly selects one.
        expected_format = params['format_type']

        logger.info(f"Loading dataset: {params['dataset_name']} (split: {params['split']}, task: {task_type})")

        from aligntune.data.manager import DataManager

        manager = DataManager(
            task_type=task_type,
            system_prompt=params['system_prompt'],
            tokenizer=self.tokenizer,
            enable_thinking=params['enable_thinking'],
            column_mapping=params['column_mapping'],
            processing_fn=params['processing_fn'],
            processing_batched=params['processing_batched'],
            max_samples=params['max_samples'],
            max_length=getattr(self.config.model, 'max_seq_length', 1024),
            lmbda=getattr(self.config.train, 'lmbda', 0.0),  # Pass lmbda for online/offline detection
            expected_format=expected_format,
            keep_columns=keep_columns,
            val_split_ratio=params['val_split_ratio'],
            test_split_ratio=params['test_split_ratio'],
            seed=params['split_seed'],
            curator_schema_gate=params['curator_schema_gate'],
            curator_clean=params['curator_clean'],
            curator_dedup=params['curator_dedup'],
            curator_use_tiktoken=params['curator_use_tiktoken'],
            curator_max_tokens=params['curator_max_tokens'],
            privileged_context_column=params['privileged_context_column'],
        )

        dataset_dict = manager.load_dataset(
            params['dataset_name'],
            config_name=params['config_name'],
            split=params['split'],
        )

        self.train_dataset = dataset_dict.get("train", None)
        self.eval_dataset = dataset_dict.get("validation", None)
        self.dataset_dict = dataset_dict

        if self.train_dataset is None:
            raise ValueError("No training dataset loaded")

        logger.info(f"Dataset loaded: {len(self.train_dataset)} train samples")
        if self.eval_dataset:
            logger.info(f"Evaluation dataset: {len(self.eval_dataset)} samples")

    def _get_task_type(self, dataset_config=None) -> str:
        """Resolve the user override or the trainer's DataManager task type."""
        configured_task_type = self._get_config_value(
            dataset_config, 'task_type', default=None
        ) if dataset_config is not None else None
        if configured_task_type is not None:
            return configured_task_type

        task_type = getattr(self, 'TASK_TYPE', "distillation")
        if task_type == "distillation":
            # Always route through "distillation_offline" regardless of on_policy/lmbda -
            # see manager.py's task_type remap table: TRL's DistillationTrainer/GOLDTrainer
            # need the full prompt+completion "messages" shape to know the prompt boundary
            # even when on-policy, since the completion is only ever discarded, never the
            # prompt. Routing on-policy runs to "ppo" here produces a prompt-only dataset
            # with no "messages" column, crashing with `KeyError: 'messages'`.
            return "distillation_offline"
        if task_type:
            return task_type
        return "distillation"

    def _extract_dataset_config(self):
        """Extract dataset configuration from config."""
        if hasattr(self.config, 'dataset'):
            return self.config.dataset
        elif hasattr(self.config, 'datasets') and len(self.config.datasets) > 0:
            return self.config.datasets[0]
        else:
            raise ValueError("No dataset configuration found in DistillConfig")

    def _extract_dataset_params(self, dataset_config) -> Dict[str, Any]:
        """Extract all dataset parameters from config."""
        return {
            'dataset_name': self._get_config_value(dataset_config, 'name', 'dataset_name', default='wikitext'),
            'split': self._get_config_value(dataset_config, 'split', default=None),
            'config_name': self._get_config_value(dataset_config, 'config_name', default=None),
            'system_prompt': self._get_config_value(dataset_config, 'system_prompt', default=None),
            'enable_thinking': self._get_config_value(dataset_config, 'enable_thinking', default=False),
            'column_mapping': self._get_config_value(dataset_config, 'column_mapping', default=None),
            'processing_fn': self._get_config_value(dataset_config, 'processing_fn', default=None),
            'processing_batched': self._get_config_value(dataset_config, 'processing_batched', default=False),
            'max_samples': self._get_config_value(dataset_config, 'max_samples', default=None),
            'format_type': self._get_config_value(dataset_config, 'format_type', default=None),
            'keep_columns': self._get_config_value(dataset_config, 'keep_columns', default=None),
            'val_split_ratio': self._get_config_value(dataset_config, 'val_split_ratio', default=None),
            'test_split_ratio': self._get_config_value(dataset_config, 'test_split_ratio', default=None),
            'split_seed': self._get_config_value(dataset_config, 'split_seed', default=42),
            'curator_schema_gate': self._get_config_value(dataset_config, 'curator_schema_gate', default=True),
            'curator_clean': self._get_config_value(dataset_config, 'curator_clean', default=False),
            'curator_dedup': self._get_config_value(dataset_config, 'curator_dedup', default='none'),
            'curator_use_tiktoken': self._get_config_value(dataset_config, 'curator_use_tiktoken', default=False),
            'curator_max_tokens': self._get_config_value(dataset_config, 'curator_max_tokens', default=1_000_000),
            'privileged_context_column': self._get_config_value(
                dataset_config, 'privileged_context_column', default=None
            ),
        }

    def _get_config_value(self, config_obj, *keys, default=None):
        """Get config value, checking multiple possible key names."""
        for key in keys:
            if hasattr(config_obj, key):
                value = getattr(config_obj, key)
                if value is not None:
                    return value
        return default

    @abstractmethod
    def train(self) -> None:
        """
        Train the distillation model.
        Implemented by specific distillation trainer subclasses.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Evaluate the distillation model.
        Implemented by specific distillation trainer subclasses.
        """
        pass
