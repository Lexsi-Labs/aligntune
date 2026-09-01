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
    REASONING = "reasoning"  # For process reward model evaluation


# Reasoning Tasks Registry for Process Reward Model Evaluation
REASONING_TASKS = {
    "aime": {
        "name": "AIME",
        "category": "reasoning",
        "description": "American Invitational Mathematics Examination - challenging arithmetic problems",
        "lm_eval_task_name": "aime",
        "metrics": ["exact_match"],
        "metric_type": "exact_match",
    },
    "math": {
        "name": "MATH",
        "category": "reasoning",
        "description": "Diverse mathematical problems from competitions",
        "lm_eval_task_name": "math",
        "metrics": ["exact_match"],
        "metric_type": "exact_match",
    },
    "gpqa": {
        "name": "GPQA",
        "category": "reasoning",
        "description": "Graduate-level Professional and Academic Questions",
        "lm_eval_task_name": "gpqa",
        "metrics": ["acc"],
        "metric_type": "exact_match",
    },
    "livecode": {
        "name": "LiveCodeBench",
        "category": "reasoning",
        "description": "Code execution benchmarks with reasoning requirements",
        "lm_eval_task_name": "livecodedbench",
        "metrics": ["score"],
        "metric_type": "code_execution",
    },
    "gsm8k_cot": {
        "name": "GSM8K-CoT",
        "category": "reasoning",
        "description": "Grade school math with chain-of-thought annotations",
        "lm_eval_task_name": "gsm8k",
        "metrics": ["exact_match"],
        "metric_type": "exact_match",
    },
}


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

    # Reasoning and Process Reward Model Evaluation
    "aime": LMEvalTask(
        name="aime",
        category=TaskCategory.REASONING,
        description="American Invitational Mathematics Examination - challenging arithmetic problems",
        lm_eval_task_name="aime",
        metrics=["exact_match"]
    ),

    "math": LMEvalTask(
        name="math",
        category=TaskCategory.REASONING,
        description="Diverse mathematical problems from competitions",
        lm_eval_task_name="math",
        metrics=["exact_match"]
    ),

    "gpqa": LMEvalTask(
        name="gpqa",
        category=TaskCategory.REASONING,
        description="Graduate-level Professional and Academic Questions",
        lm_eval_task_name="gpqa",
        metrics=["acc"]
    ),

    "livecode": LMEvalTask(
        name="livecode",
        category=TaskCategory.REASONING,
        description="Code execution benchmarks with reasoning requirements",
        lm_eval_task_name="livecodedbench",
        metrics=["score"]
    ),

    "gsm8k_cot": LMEvalTask(
        name="gsm8k_cot",
        category=TaskCategory.REASONING,
        description="Grade school math with chain-of-thought annotations",
        lm_eval_task_name="gsm8k",
        metrics=["exact_match"]
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


# Indic Benchmark Tasks Registry
# Re-implemented under Apache 2.0 license (not IndicEvalHarness)
INDIC_TASKS = {
    # MILU: IIT-KGP Indic MMLU (multiple choice)
    "milu_hi": LMEvalTask(
        name="MILU-Hindi",
        category="question_answering",
        description="IIT-KGP Indic MMLU - Hindi (multiple choice knowledge)",
        lm_eval_task_name="milu_hi",
        metrics=["exact_match"],
    ),
    "milu_ta": LMEvalTask(
        name="MILU-Tamil",
        category="question_answering",
        description="IIT-KGP Indic MMLU - Tamil (multiple choice knowledge)",
        lm_eval_task_name="milu_ta",
        metrics=["exact_match"],
    ),
    "milu_bn": LMEvalTask(
        name="MILU-Bengali",
        category="question_answering",
        description="IIT-KGP Indic MMLU - Bengali (multiple choice knowledge)",
        lm_eval_task_name="milu_bn",
        metrics=["exact_match"],
    ),

    # IndicXTREME tasks
    "indicopa_hi": LMEvalTask(
        name="IndicCOPA-Hindi",
        category="commonsense_reasoning",
        description="IndicCOPA - Hindi causal reasoning",
        lm_eval_task_name="indicopa_hi",
        metrics=["exact_match"],
    ),
    "indicsentiment_hi": LMEvalTask(
        name="IndicSentiment-Hindi",
        category="sentiment_analysis",
        description="IndicSentiment - Hindi sentiment classification",
        lm_eval_task_name="indicsentiment_hi",
        metrics=["exact_match"],
    ),
    "indicxnli_hi": LMEvalTask(
        name="IndicXNLI-Hindi",
        category="reasoning",
        description="IndicXNLI - Hindi natural language inference",
        lm_eval_task_name="indicxnli_hi",
        metrics=["exact_match"],
    ),
    "indicqa_hi": LMEvalTask(
        name="IndicQA-Hindi",
        category="question_answering",
        description="IndicQA - Hindi question answering",
        lm_eval_task_name="indicqa_hi",
        metrics=["f1"],
    ),

    # IndicGenBench tasks
    "floresin_hi_en": LMEvalTask(
        name="FloresIN-Hindi-English",
        category="machine_translation",
        description="FloresIN - Hindi to English machine translation",
        lm_eval_task_name="floresin_hi_en",
        metrics=["bleu"],
    ),
    "floresin_ta_en": LMEvalTask(
        name="FloresIN-Tamil-English",
        category="machine_translation",
        description="FloresIN - Tamil to English machine translation",
        lm_eval_task_name="floresin_ta_en",
        metrics=["bleu"],
    ),
    "floresin_bn_en": LMEvalTask(
        name="FloresIN-Bengali-English",
        category="machine_translation",
        description="FloresIN - Bengali to English machine translation",
        lm_eval_task_name="floresin_bn_en",
        metrics=["bleu"],
    ),
    "crosssumin_hi": LMEvalTask(
        name="CrossSumIN-Hindi",
        category="summarization",
        description="CrossSumIN - Hindi cross-lingual summarization",
        lm_eval_task_name="crosssumin_hi",
        metrics=["rouge1", "rouge2", "rougeL"],
    ),
    "xquadin_hi": LMEvalTask(
        name="XQuAD-IN-Hindi",
        category="question_answering",
        description="XQuAD-IN - Hindi extractive QA",
        lm_eval_task_name="xquadin_hi",
        metrics=["f1"],
    ),

    # Sarvam Indic Evaluations
    "mmlu_in_hi": LMEvalTask(
        name="MMLU-IN-Hindi",
        category="question_answering",
        description="Sarvam MMLU-IN - Hindi multitask language understanding",
        lm_eval_task_name="mmlu_in_hi",
        metrics=["exact_match"],
    ),
    "gsm8k_in_hi": LMEvalTask(
        name="GSM8K-IN-Hindi",
        category="math_reasoning",
        description="Sarvam GSM8K-IN - Hindi grade school math",
        lm_eval_task_name="gsm8k_in_hi",
        metrics=["exact_match"],
    ),
    "triviaqa_in_hi": LMEvalTask(
        name="TriviaQA-IN-Hindi",
        category="question_answering",
        description="Sarvam TriviaQA-IN - Hindi factual QA",
        lm_eval_task_name="triviaqa_in_hi",
        metrics=["f1"],
    ),
}

# Extended task registry with Indic tasks
LMEVAL_TASKS.update(INDIC_TASKS)


def get_available_indic_tasks() -> List[str]:
    """Get list of available Indic evaluation tasks."""
    return list(INDIC_TASKS.keys())


def get_available_indic_tasks_by_language(language: str) -> List[str]:
    """
    Get available Indic tasks for a specific language.

    Args:
        language: Language code ("hi", "ta", "bn")

    Returns:
        List of task names for that language
    """
    lang_map = {
        "hi": "hi",
        "hindi": "hi",
        "ta": "ta",
        "tamil": "ta",
        "bn": "bn",
        "bengali": "bn",
    }
    lang_code = lang_map.get(language.lower())
    if not lang_code:
        raise ValueError(f"Unknown language: {language}. Supported: hi, ta, bn")

    return [task for task in INDIC_TASKS.keys() if f"_{lang_code}" in task]


def run_standard_benchmark(model_name: str, tasks: Optional[List[str]] = None, **kwargs) -> List[EvalResult]:
    """Run standard benchmark evaluation."""
    if tasks is None:
        tasks = ["hellaswag", "arc_challenge", "mmlu", "gsm8k", "human_eval"]

    config = LMEvalConfig(model_name=model_name, **kwargs)
    runner = LMEvalRunner(config)

    lm_eval_tasks = [get_lm_eval_task(task_name) for task_name in tasks]
    results = runner.evaluate_tasks(lm_eval_tasks)

    return results


def run_indic_benchmark(
    model_name: str,
    languages: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    **kwargs
) -> List[EvalResult]:
    """
    Run Indic benchmark evaluation for specified languages.

    Args:
        model_name: HuggingFace model identifier
        languages: List of language codes ("hi", "ta", "bn"). If None, evaluates all.
        benchmarks: List of benchmark types ("milu", "indicxtreme", "genbench", "sarvam").
                   If None, evaluates all available.
        **kwargs: Additional arguments for LMEvalConfig

    Returns:
        List of EvalResult objects

    Examples:
        # Evaluate Hindi and Tamil on MILU and IndicXTREME
        results = run_indic_benchmark(
            "meta-llama/Llama-2-7b",
            languages=["hi", "ta"],
            benchmarks=["milu", "indicxtreme"]
        )
    """
    if languages is None:
        languages = ["hi", "ta", "bn"]

    if benchmarks is None:
        benchmarks = ["milu", "indicxtreme", "genbench", "sarvam"]

    config = LMEvalConfig(model_name=model_name, **kwargs)
    runner = LMEvalRunner(config)

    benchmark_map = {
        "milu": [t for t in INDIC_TASKS.keys() if t.startswith("milu_")],
        "indicxtreme": [t for t in INDIC_TASKS.keys()
                        if any(x in t for x in ["indicopa", "indicsentiment", "indicxnli", "indicqa"])],
        "genbench": [t for t in INDIC_TASKS.keys()
                     if any(x in t for x in ["floresin", "crosssumin", "xquadin"])],
        "sarvam": [t for t in INDIC_TASKS.keys() if any(x in t for x in ["mmlu_in", "gsm8k_in", "triviaqa_in"])],
    }

    # Collect tasks for selected languages and benchmarks
    tasks_to_run = []
    for benchmark in benchmarks:
        for task in benchmark_map.get(benchmark, []):
            # Check if task matches selected languages
            for lang in languages:
                lang_code = {"hi": "hi", "ta": "ta", "bn": "bn"}.get(lang)
                if lang_code and f"_{lang_code}" in task:
                    tasks_to_run.append(task)
                    break

    if not tasks_to_run:
        logger.warning(f"No tasks found for languages={languages}, benchmarks={benchmarks}")
        return []

    lm_eval_tasks = [get_lm_eval_task(task_name) for task_name in tasks_to_run]
    logger.info(f"Running {len(lm_eval_tasks)} Indic tasks: {[t.name for t in lm_eval_tasks]}")

    results = runner.evaluate_tasks(lm_eval_tasks)
    return results
