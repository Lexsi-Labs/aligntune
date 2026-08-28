"""TRL SFT backend for causal LM / generation tasks."""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any
import torch

from ....core.sft.trainer_base import SFTTrainerBase
from ....core.registry import TaskType
from ....core.logging import WandBLogger

logger = logging.getLogger(__name__)


class TRLSFTTrainer(SFTTrainerBase):
    """TRL SFTTrainer for supervised fine-tuning / generation tasks (causal LM)."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_type = self._get_task_type()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_cache = None
        self.dataset = None
        self.training_history = []
        self.logging_manager = None
        self.evaluator = None
        self.eval_dataset = None
        self.custom_evaluator = None
        self.wandb_logger = None
        logger.info(f"Initialized TRLSFTTrainer for task: {self.task_type.value}")

    def _get_task_type(self) -> TaskType:
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'task_type'):
            task_type = self.config.dataset.task_type
        elif hasattr(self.config, 'train') and hasattr(self.config.train, 'task_type'):
            task_type = self.config.train.task_type
        else:
            task_type = TaskType.SFT
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        return task_type

    def _initialize_wandb_logger(self) -> None:
        try:
            logging_config = self.config.logging
            if hasattr(logging_config, 'wandb_project') and logging_config.wandb_project:
                self.wandb_logger = WandBLogger(
                    project=logging_config.wandb_project,
                    entity=getattr(logging_config, 'wandb_entity', None),
                    config=self._prepare_wandb_config(),
                    tags=getattr(logging_config, 'wandb_tags', None),
                    notes=getattr(logging_config, 'wandb_notes', None),
                    name=getattr(logging_config, 'wandb_name', None),
                )
                logger.info(f"WandB logger initialized for project: {logging_config.wandb_project}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb logger: {str(e)}")

    def _prepare_wandb_config(self) -> Dict[str, Any]:
        return {
            'model_name': self.config.model.name_or_path,
            'task_type': self.task_type.value,
            'learning_rate': self.config.train.learning_rate,
            'batch_size': self.config.train.per_device_batch_size,
            'max_steps': self.config.train.max_steps,
            'epochs': getattr(self.config.train, 'epochs', None),
            'gradient_accumulation_steps': self.config.train.gradient_accumulation_steps,
        }

    @classmethod
    def is_available(cls) -> bool:
        try:
            from trl import SFTTrainer, SFTConfig, ModelConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        from ....core.model_loader import build_model
        apply_peft = getattr(getattr(self.config.model, 'peft', None), 'enabled', False)
        self.model, self.tokenizer = build_model(
            config=self.config,
            task_type=self.task_type,
            use_unsloth=False,
            apply_peft=apply_peft
        )

    # setup_data() inherited from SFTTrainerBase — uses unified DataManager

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting TRL SFT training for task: {self.task_type.value}")
        self._initialize_wandb_logger()

        if self.model is None:
            self.setup_model()
        if self.dataset is None:
            self.setup_data()
            self.dataset = self.train_dataset

        try:
            from trl import SFTTrainer
        except ImportError as e:
            raise ImportError("TRL required. Install with: pip install trl") from e

        from ....core.sft.training_args_builder import build_sft_config

        optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.dataset)
        filtered_scheduler_kwargs = {
            k: v for k, v in optim_scheduler['scheduler_config']['lr_scheduler_kwargs'].items()
            if k not in ['num_training_steps', 'num_warmup_steps']
        }
        extra_kwargs = {
            "optim": optim_scheduler['optimizer_name'],
            "lr_scheduler_type": optim_scheduler['scheduler_name'],
            "lr_scheduler_kwargs": filtered_scheduler_kwargs,
        }
        optim_args_str = optim_scheduler['optim_args_str']
        should_skip_optim_args = (
            not optim_args_str or
            not optim_args_str.strip() or
            "=" not in optim_args_str or
            any(x in optim_scheduler['optimizer_name'].lower() for x in ["8bit", "8_bit", "bnb", "paged"])
        )
        if not should_skip_optim_args:
            extra_kwargs["optim_args"] = optim_args_str
        if getattr(self.config.train, 'torch_compile', False):
            extra_kwargs["torch_compile"] = True
        if getattr(self.config.train, 'use_liger_kernel', False):
            try:
                import liger_kernel
                extra_kwargs["use_liger_kernel"] = True
            except ImportError:
                logger.warning("Liger kernel requested but not installed.")
        accelerate_config_path = getattr(self.config.train, 'accelerate_config_path', None)
        if accelerate_config_path:
            import os
            os.environ['ACCELERATE_CONFIG_FILE'] = str(Path(accelerate_config_path).resolve())

        sft_config = build_sft_config(
            config=self.config,
            train_dataset=self.dataset,
            eval_dataset=getattr(self, 'eval_dataset', None),
            output_dir=self.config.logging.output_dir,
            **extra_kwargs
        )

        train_dataset = self.dataset
        eval_dataset = self.eval_dataset

        def select_trl_columns(dataset):
            """Pass TRL only the fields required by the resolved SFT task."""
            if self.task_type.value == "pretraining":
                names = ["text"]
            elif "messages" in dataset.column_names:
                names = ["messages", "chat_template_kwargs"]
            else:
                names = ["prompt", "completion"]
            return dataset.select_columns(
                [name for name in names if name in dataset.column_names]
            )

        train_dataset = select_trl_columns(train_dataset)
        if eval_dataset is not None:
            eval_dataset = select_trl_columns(eval_dataset)

        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=None,
            callbacks=self.get_hf_callbacks(),
        )

        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"Starting TRL SFT Training - Task: {self.task_type.value}")
        logger.info("=" * 80)

        train_result = self.trainer.train()
        training_duration = time.time() - start_time

        logger.info(f"Training completed in {training_duration:.2f} seconds")

        if self.logging_manager and hasattr(train_result, 'metrics'):
            self.logging_manager.log_metrics(train_result.metrics)

        self.trainer.save_model()
        self._auto_export()

        self.training_history.append({
            'timestamp': time.time(),
            'duration': training_duration,
            'steps': getattr(train_result, 'global_step', 0),
            'task_type': self.task_type.value,
            'model_path': self.config.logging.output_dir,
        })

        logger.info("=" * 80)
        logger.info("TRL SFT training completed successfully!")
        logger.info("=" * 80)

        if self.wandb_logger:
            try:
                final_metrics = {
                    'training_time': training_duration,
                    'final_loss': getattr(train_result, 'train_loss', 0.0),
                    'epochs': getattr(train_result, 'epoch', 0),
                }
                self.wandb_logger.log_metrics(final_metrics, step=getattr(train_result, 'global_step', 0))
                checkpoint_path = Path(self.config.logging.output_dir)
                if checkpoint_path.exists():
                    self.wandb_logger.log_artifact(
                        str(checkpoint_path),
                        artifact_type="model",
                        name=f"final-checkpoint-{getattr(train_result, 'global_step', 0)}"
                    )
                self.wandb_logger.finalize()
            except Exception as e:
                logger.warning(f"Error logging to wandb: {str(e)}")

        return {
            'training_time': training_duration,
            'final_loss': getattr(train_result, 'train_loss', 0.0),
            'model_path': self.config.logging.output_dir,
            'steps': getattr(train_result, 'global_step', 0),
            'epochs': getattr(train_result, 'epoch', 0),
            'metrics': getattr(train_result, 'metrics', {}),
            'task_type': self.task_type.value,
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {"loss": 0.0}

    def get_sample_outputs(self) -> list:
        if not self.dataset:
            return []
        sample_outputs = []
        for i in range(min(3, len(self.dataset))):
            sample = self.dataset[i]
            if "text" in sample:
                text = sample["text"]
            elif "messages" in sample:
                text = self.tokenizer.apply_chat_template(sample["messages"], tokenize=False)
            else:
                text = str(sample.get("input", ""))
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=getattr(self.config.train, 'temperature', 0.7),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sample_outputs.append({
                "input": text[:100] + "..." if len(text) > 100 else text,
                "output": response,
                "task_type": self.task_type.value,
            })
        return sample_outputs

    def cleanup(self) -> None:
        super().cleanup()

    def get_training_stats(self) -> Dict[str, Any]:
        return {
            'config': {
                'model_name': self.config.model.name_or_path,
                'task_type': self.task_type.value,
                'dataset_name': self.config.dataset.name,
                'epochs': getattr(self.config.train, 'epochs', None),
                'max_steps': self.config.train.max_steps,
                'learning_rate': self.config.train.learning_rate,
                'batch_size': self.config.train.per_device_batch_size,
            },
            'dataset_info': {
                'train_size': len(self.dataset) if self.dataset else 0,
                'val_size': len(self.eval_dataset) if self.eval_dataset else 0,
            },
            'model_info': {
                'loaded': self.model is not None,
                'device': str(next(self.model.parameters()).device) if self.model else 'unknown',
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0,
                'has_peft': hasattr(self.model, 'peft_config') if self.model else False,
            },
            'training_history': self.training_history,
        }

    def save_config(self, path: str):
        config_dict = {
            'model_name': self.config.model.name_or_path,
            'task_type': self.task_type.value,
            'max_seq_length': self.config.model.max_seq_length,
            'learning_rate': self.config.train.learning_rate,
            'epochs': getattr(self.config.train, 'epochs', None),
            'max_steps': self.config.train.max_steps,
            'batch_size': self.config.train.per_device_batch_size,
            'dataset_name': self.config.dataset.name,
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Config saved to {path}")

    def run_experiment(self):
        logger.info(f"Starting TRL SFT experiment: {self.config.logging.run_name}")
        logger.info(f"Task Type: {self.task_type.value}")
        try:
            self.setup_model()
            self.setup_data()
            zero_shot_results = None
            if hasattr(self.config, 'evaluation') and getattr(self.config.evaluation, 'run_zero_shot', False):
                logger.info("Running pre-training zero-shot evaluation...")
                zero_shot_results = self.run_zero_shot_evaluation()
            train_results = self.train()
            eval_results = self.evaluate()
            results = {
                'mode': 'train_and_eval',
                'task_type': self.task_type.value,
                'train_results': train_results,
                'eval_results': eval_results,
                'training_stats': self.get_training_stats(),
            }
            if zero_shot_results:
                results['zero_shot_evaluation'] = zero_shot_results
            output_dir = Path(self.config.logging.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "experiment_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            logger.info(f"Experiment completed. Results saved to {results_path}")
            return results
        except Exception as e:
            logger.error(f"TRL SFT experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.logging_manager:
                self.logging_manager.finish()
