"""
Core evaluation system for AlignTune.


This module provides the foundation for comprehensive evaluation capabilities
covering all types of tasks and training scenarios. This is separate from
reward functions which are used for RL training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
import logging
import json
import time
from pathlib import Path
import torch
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

try:
    # tqdm is an optional dependency; fall back gracefully if unavailable
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - defensive import
    tqdm = None  # type: ignore


class EvalType(Enum):
    """Types of evaluation."""
    TRAINING = "training"  # During training
    STANDALONE = "standalone"  # Standalone model evaluation
    BENCHMARK = "benchmark"  # Standard benchmarks
    CUSTOM = "custom"  # Custom evaluation tasks


class TaskCategory(Enum):
    """Categories of tasks for evaluation."""
    # Language Understanding
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    QUESTION_ANSWERING = "question_answering"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    
    # Language Generation
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    DIALOGUE = "dialogue"
    STORY_GENERATION = "story_generation"
    
    # Code
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUGGING = "code_debugging"
    
    # Math and Reasoning
    MATH_REASONING = "math_reasoning"
    LOGICAL_REASONING = "logical_reasoning"
    COMMONSENSE_REASONING = "commonsense_reasoning"
    
    # Specialized
    SAFETY = "safety"
    BIAS_DETECTION = "bias_detection"
    FACTUAL_ACCURACY = "factual_accuracy"
    HALLUCINATION_DETECTION = "hallucination_detection"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    eval_type: EvalType
    task_categories: List[TaskCategory]
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "bleu", "rouge"])
    batch_size: int = 32
    max_samples: Optional[int] = None
    device: str = "auto"
    precision: str = "bf16"
    use_cache: bool = True
    cache_dir: Optional[str] = None
    output_dir: str = "./eval_results"
    save_predictions: bool = True
    save_metrics: bool = True
    verbose: bool = False


@dataclass
class EvalTask:
    """Definition of an evaluation task."""
    name: str
    category: TaskCategory
    description: str
    dataset_name: str
    dataset_config: Optional[str] = None
    input_column: str = "text"
    target_column: str = "label"
    prompt_template: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    max_length: int = 512
    few_shot_examples: int = 0
    custom_metrics: List[Callable] = field(default_factory=list)


@dataclass
class EvalResult:
    """Results from evaluation."""
    task_name: str
    category: TaskCategory
    metrics: Dict[str, float]
    predictions: List[Any] = field(default_factory=list)
    targets: List[Any] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    eval_time: float = 0.0
    num_samples: int = 0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "category": self.category.value,
            "metrics": self.metrics,
            "eval_time": self.eval_time,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp
        }


class EvalLogger:
    """Logger for evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EvalResult] = []
    
    def log_result(self, result: EvalResult):
        """Log an evaluation result."""
        self.results.append(result)
        
        # Save individual result
        result_file = self.output_dir / f"{result.task_name}_{result.timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Logged evaluation result for {result.task_name}")
    
    def save_summary(self):
        """Save summary of all results."""
        summary = {
            "total_tasks": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "summary_metrics": self._compute_summary_metrics()
        }
        
        summary_file = self.output_dir / "eval_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved evaluation summary to {summary_file}")
    
    def _compute_summary_metrics(self) -> Dict[str, float]:
        """Compute summary metrics across all tasks."""
        if not self.results:
            return {}
        
        # Average metrics across all tasks
        all_metrics = {}
        for result in self.results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        summary = {}
        for metric, values in all_metrics.items():
            summary[f"{metric}_mean"] = np.mean(values)
            summary[f"{metric}_std"] = np.std(values)
            summary[f"{metric}_min"] = np.min(values)
            summary[f"{metric}_max"] = np.max(values)
        
        return summary


class EvalRegistry:
    """Registry for evaluation tasks and metrics."""
    
    _tasks: Dict[str, EvalTask] = {}
    _metrics: Dict[str, Callable] = {}
    _datasets: Dict[str, Callable] = {}
    
    @classmethod
    def register_task(cls, task: EvalTask):
        """Register an evaluation task."""
        cls._tasks[task.name] = task
        logger.info(f"Registered evaluation task: {task.name}")
    
    @classmethod
    def register_metric(cls, name: str, metric_func: Callable):
        """Register a metric function."""
        cls._metrics[name] = metric_func
        logger.info(f"Registered metric: {name}")
    
    @classmethod
    def register_dataset(cls, name: str, dataset_func: Callable):
        """Register a dataset loader."""
        cls._datasets[name] = dataset_func
        logger.info(f"Registered dataset: {name}")
    
    @classmethod
    def get_task(cls, name: str) -> EvalTask:
        """Get a registered task."""
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name]
    
    @classmethod
    def get_metric(cls, name: str) -> Callable:
        """Get a registered metric."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name]
    
    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks."""
        return list(cls._tasks.keys())
    
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics."""
        return list(cls._metrics.keys())


class EvalRunner:
    """Main evaluation runner."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.logger = EvalLogger(config.output_dir)
        self.device = self._setup_device()
    
    def _setup_device(self) -> str:
        """Setup evaluation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def evaluate_task(self, task: EvalTask, model, tokenizer) -> EvalResult:
        """Evaluate a single task."""
        logger.info(f"Evaluating task: {task.name}")
        start_time = time.time()
        
        # Load dataset
        dataset = self._load_dataset(task)
        
        # Prepare data
        eval_data = self._prepare_data(dataset, task)
        
        # Run evaluation
        predictions, targets = self._run_evaluation(eval_data, task, model, tokenizer)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, task)
        
        # Create result
        result = EvalResult(
            task_name=task.name,
            category=task.category,
            metrics=metrics,
            predictions=predictions,
            targets=targets,
            samples=eval_data[:self.config.max_samples] if self.config.max_samples else eval_data,
            eval_time=time.time() - start_time,
            num_samples=len(eval_data),
            timestamp=time.strftime("%Y%m%d_%H%M%S")
        )
        
        # Log result
        self.logger.log_result(result)
        
        return result
    
    def evaluate_tasks(self, task_names: List[str], model, tokenizer) -> List[EvalResult]:
        """Evaluate multiple tasks."""
        results: List[EvalResult] = []
        iterable = task_names
        if tqdm is not None:
            iterable = tqdm(task_names, desc="Evaluating tasks", leave=False)
        for task_name in iterable:
            task = EvalRegistry.get_task(task_name)
            result = self.evaluate_task(task, model, tokenizer)
            results.append(result)
        
        # Save summary
        self.logger.save_summary()
        
        return results
    
    def _load_dataset(self, task: EvalTask) -> Dataset:
        """Load dataset for evaluation."""
        if task.dataset_name in EvalRegistry._datasets:
            return EvalRegistry._datasets[task.dataset_name](
                config=task.dataset_config,
                split="test" if task.dataset_config else "validation"
            )
        else:
            # Use HuggingFace datasets as fallback
            from datasets import load_dataset
            return load_dataset(
                task.dataset_name,
                task.dataset_config,
                split="test" if task.dataset_config else "validation"
            )
    
    def _prepare_data(self, dataset: Dataset, task: EvalTask) -> List[Dict[str, Any]]:
        """Prepare data for evaluation."""
        data = []
        for i, sample in enumerate(dataset):
            if self.config.max_samples and i >= self.config.max_samples:
                break
            
            # Apply prompt template if provided
            if task.prompt_template:
                text = task.prompt_template.format(
                    input=sample[task.input_column],
                    target=sample.get(task.target_column, "")
                )
            else:
                text = sample[task.input_column]
            
            data.append({
                "input": text,
                "target": sample.get(task.target_column, ""),
                "sample": sample
            })
        
        return data
    
    def _run_evaluation(self, data: List[Dict[str, Any]], task: EvalTask, model, tokenizer) -> Tuple[List[Any], List[Any]]:
        """Run the actual evaluation."""
        predictions: List[Any] = []
        targets: List[Any] = []
        
        model.eval()
        iterable = range(0, len(data), self.config.batch_size)
        if tqdm is not None:
            iterable = tqdm(iterable, desc="Evaluating (batches)", leave=False)
        with torch.no_grad():
            for i in iterable:
                batch = data[i:i + self.config.batch_size]
                
                # Tokenize inputs
                inputs = tokenizer(
                    [item["input"] for item in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=task.max_length
                ).to(self.device)
                
                # Generate predictions
                if task.category in [TaskCategory.TEXT_GENERATION, TaskCategory.SUMMARIZATION, 
                                   TaskCategory.DIALOGUE, TaskCategory.CODE_GENERATION]:
                    # Generation task
                    outputs = model.generate(
                        **inputs,
                        max_length=task.max_length,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    predictions.extend(batch_predictions)
                else:
                    # Classification/understanding task
                    outputs = model(**inputs)
                    if hasattr(outputs, 'logits'):
                        batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                        predictions.extend(batch_predictions.tolist())
                
                targets.extend([item["target"] for item in batch])
        
        return predictions, targets
    
    def _compute_metrics(self, predictions: List[Any], targets: List[Any], task: EvalTask) -> Dict[str, float]:
        """Compute metrics for the evaluation."""
        metrics = {}
        
        for metric_name in task.metrics:
            if metric_name in EvalRegistry._metrics:
                metric_func = EvalRegistry._metrics[metric_name]
                try:
                    score = metric_func(predictions, targets)
                    metrics[metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Failed to compute metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        # Compute custom metrics
        for custom_metric in task.custom_metrics:
            try:
                score = custom_metric(predictions, targets)
                metrics[f"custom_{custom_metric.__name__}"] = float(score)
            except Exception as e:
                logger.warning(f"Failed to compute custom metric {custom_metric.__name__}: {e}")
        
        return metrics
