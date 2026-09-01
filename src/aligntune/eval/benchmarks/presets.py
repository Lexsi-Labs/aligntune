"""
Preset benchmark bundles for comprehensive model evaluation.

This module defines curated collections of evaluation tasks grouped by theme:
- alignment_core: Core alignment and safety benchmarks
- safety: Safety-focused benchmarks
- reasoning: Reasoning and knowledge benchmarks
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# Define benchmark bundles as lists of task names compatible with lm-eval
BENCHMARK_BUNDLES: Dict[str, List[str]] = {
    "alignment_core": [
        "truthfulqa_mc",      # Measure truthfulness
        "toxigen",            # Toxicity generation
        "bbq",                # Bias in closed QA
        "mmlu_safety",        # Safety-relevant MMLU subset
        "hellaswag",          # Commonsense reasoning baseline
    ],
    "safety": [
        "harmbench",          # Comprehensive harm evaluation
        "xstest",             # Cross-topic safety test
        "advbench",           # Adversarial benchmarks
    ],
    "reasoning": [
        "gsm8k",              # Grade school math
        "math_algebra",       # Mathematical reasoning
        "bbh",                # Big Bench Hard reasoning tasks
    ],
}


def get_bundle(name: str) -> List[str]:
    """
    Retrieve a benchmark bundle by name.

    Args:
        name: Name of the bundle ('alignment_core', 'safety', or 'reasoning')

    Returns:
        List of task names in the bundle

    Raises:
        ValueError: If bundle name not found
    """
    if name not in BENCHMARK_BUNDLES:
        available = ", ".join(BENCHMARK_BUNDLES.keys())
        raise ValueError(
            f"Unknown benchmark bundle: {name}. "
            f"Available bundles: {available}"
        )
    return BENCHMARK_BUNDLES[name]


def list_bundles() -> List[str]:
    """
    List all available benchmark bundle names.

    Returns:
        List of bundle names
    """
    return list(BENCHMARK_BUNDLES.keys())
