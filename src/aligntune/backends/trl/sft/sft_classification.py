"""Text Classification Trainer (sentiment, topic, etc.)."""

import logging
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ....core.sft.trainer_base import SFTTrainerBase
from ....core.registry import TaskType

logger = logging.getLogger(__name__)


class TextClassificationTrainer(SFTTrainerBase):
    """Trainer for text/sequence classification. Uses HF Trainer with AutoModelForSequenceClassification."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.task_type = TaskType.TEXT_CLASSIFICATION
        logger.info("TextClassificationTrainer initialized")

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def setup_model(self):
        from ....core.model_loader import build_model
        self.model, self.tokenizer = build_model(
            config=self.config,
            task_type=self.task_type,
            use_unsloth=False
        )
        if getattr(getattr(self.config.model, 'peft', None), 'enabled', False):
            from ....core.peft.factory import PEFTFactory
            adapter = PEFTFactory.get_adapter(self.config)
            self.model = adapter.apply_to_transformers(self.model, task_type="SEQ_CLS")

    def setup_data(self):
        from ....core.sft.dataset_builder import build_sft_datasets
        raw_train, raw_eval = build_sft_datasets(
            config=self.config, tokenizer=self.tokenizer, task_type=self.task_type
        )
        self._ensure_pad_token()

        def tokenize(examples):
            result = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.model.max_seq_length,
            )
            labels = examples['labels']
            # Remap 1-indexed labels to 0-indexed
            if isinstance(labels, list) and labels and isinstance(labels[0], int) and min(labels) > 0:
                labels = [l - 1 for l in labels]
            result['labels'] = labels
            return result

        self.train_dataset = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
        if raw_eval:
            self.eval_dataset = raw_eval.map(tokenize, batched=True, remove_columns=raw_eval.column_names)

    def setup_training(self):
        self._ensure_pad_token()

        args_dict = {
            "output_dir": self.config.logging.output_dir,
            "run_name": self.config.logging.run_name,
            "per_device_train_batch_size": self.config.train.per_device_batch_size,
            "per_device_eval_batch_size": (
                self.config.train.per_device_eval_batch_size
                or self.config.train.per_device_batch_size
            ),
            "gradient_accumulation_steps": self.config.train.gradient_accumulation_steps,
            "learning_rate": self.config.train.learning_rate,
            "weight_decay": self.config.train.weight_decay,
            "warmup_ratio": self.config.train.warmup_ratio,
            "eval_strategy": "steps",
            "eval_steps": self.config.train.eval_interval,
            "save_strategy": "steps",
            "save_steps": self.config.train.save_interval,
            "load_best_model_at_end": self.config.train.load_best_model_at_end,
            "metric_for_best_model": self.config.train.metric_for_best_model,
            "greater_is_better": self.config.train.greater_is_better,
            "logging_steps": 10,
            "fp16": getattr(self.config.model.precision, 'value', '') == 'fp16',
            "bf16": getattr(self.config.model.precision, 'value', '') == 'bf16',
            "report_to": "none",
            "save_total_limit": 2,
            "push_to_hub": False,
        }
        if self.config.train.max_steps:
            args_dict["max_steps"] = self.config.train.max_steps
        else:
            args_dict["num_train_epochs"] = self.config.train.epochs

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**args_dict),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8),
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
        )

    def train(self):
        self.setup_model()
        self.setup_data()
        self.setup_training()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for cb in self.get_hf_callbacks():
            self.trainer.add_callback(cb)
        try:
            return self.trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.error(f"CUDA OOM during training: {e}")
            raise

    def evaluate(self):
        self._ensure_pad_token()
        result = self.trainer.evaluate()
        if "eval_loss" in result:
            try:
                result["eval_perplexity"] = float(torch.exp(torch.tensor(float(result["eval_loss"]))).item())
            except Exception:
                pass
        return result

    def run_experiment(self):
        logger.info("Starting text classification experiment")
        try:
            train_result = self.train()
            eval_result = self.evaluate()
            model_path = self.save_model()
            return {
                'task_type': self.task_type.value,
                'train_loss': train_result.training_loss,
                'eval_results': eval_result,
                'model_path': model_path,
                'steps': train_result.global_step,
            }
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
