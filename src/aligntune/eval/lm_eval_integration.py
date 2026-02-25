"""
Integration with lm-eval library for standardized evaluation.

This module provides integration with the lm-eval library for running
standardized benchmarks and evaluations.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass

from .core import EvalConfig, EvalTask, EvalResult, TaskCategory

logger = logging.getLogger(__name__)


class TaskCategory:
    """Categories for evaluation tasks."""
    COMMONSENSE_REASONING = "commonsense_reasoning"
    QUESTION_ANSWERING = "question_answering"
    FACTUAL_ACCURACY = "factual_accuracy"
    MATH_REASONING = "math_reasoning"
    CODE_GENERATION = "code_generation"
    BIAS_DETECTION = "bias_detection"


@dataclass
class EvalResult:
    """Result of an evaluation task."""
    task_name: str
    category: str
    metrics: Dict[str, float]
    eval_time: float
    num_samples: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "category": self.category,
            "metrics": self.metrics,
            "eval_time": self.eval_time,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp
        }


@dataclass
class LMEvalConfig:
    """Configuration for lm-eval integration."""
    model_name: str
    model_args: str = "pretrained={model_name}"
    batch_size: int = 1
    device: str = "auto"
    limit: Optional[int] = None
    output_dir: str = "./lm_eval_results"
    save_results: bool = True
    verbose: bool = False


@dataclass
class LMEvalTask:
    """Definition of an lm-eval task."""
    name: str
    category: str # changed from TaskCategory
    description: str
    lm_eval_task_name: str
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["acc", "acc_norm", "bleu", "rouge1", "rouge2", "rougeL"]


class LMEvalRunner:
    """Runner for lm-eval evaluations."""
    
    def __init__(self, config: LMEvalConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_task(self, task: LMEvalTask) -> EvalResult:
        """Evaluate a single lm-eval task."""
        logger.info(f"Running lm-eval for task: {task.name}")
        
        # Prepare command
        cmd = self._build_command(task)
        
        # Run evaluation
        result = self._run_evaluation(cmd, task)
        
        return result
    
    def evaluate_tasks(self, tasks: List[LMEvalTask]) -> List[EvalResult]:
        """Evaluate multiple lm-eval tasks."""
        results = []
        for task in tasks:
            result = self.evaluate_task(task)
            results.append(result)
        
        # Save combined results
        if self.config.save_results:
            self._save_combined_results(results)
        
        return results
    
    def _build_command(self, task: LMEvalTask) -> List[str]:
        """Build lm-eval command."""
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", self.config.model_args.format(model_name=self.config.model_name),
            "--tasks", task.lm_eval_task_name,
            "--batch_size", str(self.config.batch_size),
            "--output_path", str(self.output_dir / f"{task.name}_results.json"),
            "--log_samples"
        ]
        
        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])
        
        # Note: --verbose flag not supported by lm-eval, using --log_samples instead
        
        return cmd
    
    def _run_evaluation(self, cmd: List[str], task: LMEvalTask) -> EvalResult:
        """Run the lm-eval command and parse results."""
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run command
            # Changed to capture_output=False to allow streaming logs/progress to console
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            
            # Parse results
            results_file = self.output_dir / f"{task.name}_results.json"

            # FIX: Fallback for timestamped files (lm_eval v0.4+)
            if not results_file.exists():
                # Search for any json file containing the task name
                candidates = list(self.output_dir.glob(f"*{task.lm_eval_task_name}*.json"))
                # Exclude summary/samples files to avoid reading the wrong data
                candidates = [f for f in candidates if "summary" not in f.name and "samples" not in f.name]
                
                if candidates:
                    # Sort by modification time to get the newest file
                    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    results_file = candidates[0]
                    logger.info(f"Found timestamped results file: {results_file.name}")

            if results_file.exists():
                with open(results_file, 'r') as f:
                    lm_eval_results = json.load(f)
                
                # Extract metrics
                metrics = self._extract_metrics(lm_eval_results, task)
                
                return EvalResult(
                    task_name=task.name,
                    category=task.category,
                    metrics=metrics,
                    eval_time=0.0,  # lm-eval doesn't provide timing
                    num_samples=lm_eval_results.get("num_samples", 0),
                    timestamp=""
                )
            else:
                logger.error(f"Results file not found: {results_file}")
                return EvalResult(
                    task_name=task.name,
                    category=task.category,
                    metrics={"error": 1.0},
                    eval_time=0.0,
                    num_samples=0,
                    timestamp=""
                )
        
        except subprocess.CalledProcessError as e:
            # Since output is not captured, stderr is already printed to console
            logger.error(f"lm-eval failed with exit code {e.returncode}")
            return EvalResult(
                task_name=task.name,
                category=task.category,
                metrics={"error": 1.0},
                eval_time=0.0,
                num_samples=0,
                timestamp=""
            )
        except Exception as e:
            logger.error(f"Unexpected error in lm-eval: {e}")
            return EvalResult(
                task_name=task.name,
                category=task.category,
                metrics={"error": 1.0},
                eval_time=0.0,
                num_samples=0,
                timestamp=""
            )
    
    def _extract_metrics(self, lm_eval_results: Dict[str, Any], task: LMEvalTask) -> Dict[str, float]:
        """
        Extract metrics from lm-eval results.
        Handles both v0.3 (simple keys) and v0.4+ (comma-separated keys like 'exact_match,flexible-extract').
        """
        metrics = {}
        
        # Get results for the task
        task_results = lm_eval_results.get("results", {}).get(task.lm_eval_task_name, {})


        # Fallback: if specific task key not found, look for alias or partial match
        if not task_results:
             keys = list(lm_eval_results.get("results", {}).keys())
             if len(keys) == 1:
                 task_results = lm_eval_results["results"][keys[0]]
             else:
                 for k in keys:
                     if task.lm_eval_task_name in k:
                         task_results = lm_eval_results["results"][k]
                         break
        
        # Helper to find a value for a metric name, trying exact and suffix matches
        def find_metric_value(target_name, results_dict):
            # 1. Exact match
            if target_name in results_dict:
                return results_dict[target_name]
            
            # 2. Suffix match (e.g. "exact_match,flexible-extract")
            # We prioritize "flexible-extract" if multiple exist
            best_val = None
            for key in results_dict.keys():
                if key.startswith(target_name + ","):
                    val = results_dict[key]
                    if "flexible-extract" in key:
                        return val # Priority return
                    best_val = val # Keep looking but store fallback
            return best_val
        
        # Extract requested metrics
        for metric in task.metrics:
            val = find_metric_value(metric, task_results)
            
            # If not found, try alternative names
            if val is None:
                alt_names = self._get_alternative_metric_names(metric)
                for alt_name in alt_names:
                    val = find_metric_value(alt_name, task_results)
                    if val is not None:
                        break
            
            if val is not None:
                # Handle formatted strings like "13.0%"
                if isinstance(val, str) and "%" in val:
                    try:
                        val = float(val.strip("%")) / 100.0
                    except: pass
                
                try:
                    metrics[metric] = float(val)
                except:
                    logger.warning(f"Could not convert metric {metric}={val} to float")
                    metrics[metric] = 0.0
            else:
                metrics[metric] = 0.0
                # Only warn if it's a genuine failure (not expected 0.0 placeholders)
                # logger.warning(f"Metric {metric} not found in results. Keys: {list(task_results.keys())}")
        
        return metrics
    
    def _get_alternative_metric_names(self, metric: str) -> List[str]:
        """Get alternative names for metrics."""
        alternatives = {
            "acc": ["accuracy", "acc_norm", "exact_match", "mean"],
            "acc_norm": ["accuracy_norm", "acc_norm,none"],
            "bleu": ["bleu_score", "bleu-1", "bleu-4"],
            "rouge1": ["rouge-1", "rouge1_f", "rouge1_precision", "rouge1_recall"],
            "rouge2": ["rouge-2", "rouge2_f", "rouge2_precision", "rouge2_recall"],
            "rougeL": ["rouge-l", "rougeL_f", "rougeL_precision", "rougeL_recall"],
            "f1": ["f1_score", "f1_macro", "f1_micro"],
            "perplexity": ["ppl", "perplexity"],
            "exact_match": ["acc", "accuracy"] # Bi-directional help
        }
        return alternatives.get(metric, [])
    
    def _save_combined_results(self, results: List[EvalResult]):
        """Save combined results from multiple tasks."""
        combined_results = {
            "model": self.config.model_name,
            "total_tasks": len(results),
            "results": [r.to_dict() for r in results],
            "summary": self._compute_summary(results)
        }
        
        summary_file = self.output_dir / "lm_eval_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"Saved combined lm-eval results to {summary_file}")
    
    def _compute_summary(self, results: List[EvalResult]) -> Dict[str, float]:
        """Compute summary statistics."""
        if not results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Compute statistics
        summary = {}
        for metric, values in all_metrics.items():
            if values:
                summary[f"{metric}_mean"] = sum(values) / len(values)
                summary[f"{metric}_max"] = max(values)
                summary[f"{metric}_min"] = min(values)
        
        return summary


# Predefined lm-eval tasks
LMEVAL_TASKS = {
    # Language Understanding
    "hellaswag": LMEvalTask(
        name="hellaswag",
        category=TaskCategory.COMMONSENSE_REASONING,
        description="Commonsense reasoning task",
        lm_eval_task_name="hellaswag",
        metrics=["acc", "acc_norm"]
    ),
    
    "arc_challenge": LMEvalTask(
        name="arc_challenge",
        category=TaskCategory.COMMONSENSE_REASONING,
        description="ARC Challenge dataset",
        lm_eval_task_name="arc_challenge",
        metrics=["acc", "acc_norm"]
    ),
    
    "arc_easy": LMEvalTask(
        name="arc_easy",
        category=TaskCategory.COMMONSENSE_REASONING,
        description="ARC Easy dataset",
        lm_eval_task_name="arc_easy",
        metrics=["acc", "acc_norm"]
    ),
    
    "mmlu": LMEvalTask(
        name="mmlu",
        category=TaskCategory.QUESTION_ANSWERING,
        description="Massive Multitask Language Understanding",
        lm_eval_task_name="mmlu",
        metrics=["acc", "acc_norm"]
    ),
    
    "truthfulqa": LMEvalTask(
        name="truthfulqa",
        category=TaskCategory.FACTUAL_ACCURACY,
        description="TruthfulQA benchmark",
        lm_eval_task_name="truthfulqa_mc",
        metrics=["acc", "acc_norm"]
    ),
    
    # Language Generation
    "gsm8k": LMEvalTask(
        name="gsm8k",
        category=TaskCategory.MATH_REASONING,
        description="Grade School Math 8K",
        lm_eval_task_name="gsm8k",
        metrics=["exact_match"]  # Updated to match lm_eval v0.4+ output
    ),
    
    "human_eval": LMEvalTask(
        name="human_eval",
        category=TaskCategory.CODE_GENERATION,
        description="HumanEval code generation",
        lm_eval_task_name="human_eval",
        metrics=["pass@1", "pass@10", "pass@100"]
    ),
    
    "mbpp": LMEvalTask(
        name="mbpp",
        category=TaskCategory.CODE_GENERATION,
        description="Mostly Basic Python Problems",
        lm_eval_task_name="mbpp",
        metrics=["pass@1", "pass@10", "pass@100"]
    ),
    
    # Safety and Bias
    "crows_pairs": LMEvalTask(
        name="crows_pairs",
        category=TaskCategory.BIAS_DETECTION,
        description="CrowS-Pairs bias detection",
        lm_eval_task_name="crows_pairs",
        metrics=["acc", "acc_norm"]
    ),
    
    "winogender": LMEvalTask(
        name="winogender",
        category=TaskCategory.BIAS_DETECTION,
        description="WinoGender bias detection",
        lm_eval_task_name="winogender",
        metrics=["acc", "acc_norm"]
    ),
}


def get_available_lm_eval_tasks() -> List[str]:
    """Get list of available lm-eval tasks."""
    return list(LMEVAL_TASKS.keys())


def get_lm_eval_task(task_name: str) -> LMEvalTask:
    """Get a specific lm-eval task."""
    if task_name not in LMEVAL_TASKS:
        raise ValueError(f"Unknown lm-eval task: {task_name}. Available: {list(LMEVAL_TASKS.keys())}")
    return LMEVAL_TASKS[task_name]


def run_standard_benchmark(model_name: str, tasks: Optional[List[str]] = None, **kwargs) -> List[EvalResult]:
    """Run standard benchmark evaluation."""
    if tasks is None:
        tasks = ["hellaswag", "arc_challenge", "mmlu", "gsm8k", "human_eval"]
    
    config = LMEvalConfig(model_name=model_name, **kwargs)
    runner = LMEvalRunner(config)
    
    lm_eval_tasks = [get_lm_eval_task(task_name) for task_name in tasks]
    results = runner.evaluate_tasks(lm_eval_tasks)
    
    return results
