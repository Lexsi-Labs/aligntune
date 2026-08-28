"""
Cost estimator and algorithm recommendation engine for AlignTune.

This module provides heuristic-based resource/cost estimation and algorithm
recommendations based on model size, dataset characteristics, and hardware profiles.

All estimates are deterministic and use simple heuristics (no ML models).
Estimates should be treated as approximate with ±30% uncertainty bounds.
"""

import logging
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Carbon / Energy Constants
# ============================================================================

# Carbon intensity by cloud region in gCO2eq/kWh (static, no external API).
# Sources: regional grid emission factors from public cloud provider reports.
REGION_CARBON_INTENSITY: Dict[str, float] = {
    # AWS
    "us-east-1": 380.0,
    "us-west-2": 136.0,
    "eu-west-1": 316.0,
    "ap-southeast-1": 493.0,
    # GCP
    "us-central1": 395.0,
    "europe-west4": 284.0,
    "asia-east1": 543.0,
    # Azure
    "eastus": 385.0,
    "westeurope": 232.0,
    # Fallback
    "default": 475.0,  # global average
}

# Canonical GPU power draw in watts (used for carbon estimation).
# Values represent typical sustained training workload wattage.
GPU_POWER_WATTS: Dict[str, float] = {
    "a100-40gb": 300.0,
    "a100-80gb": 400.0,
    "h100": 700.0,
    "l4": 72.0,
    "t4": 70.0,
    "rtx3090": 350.0,
    "rtx4090": 450.0,
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GPUProfile:
    """GPU specifications and pricing information."""
    name: str
    vram_gb: float
    tflops_fp16: float
    price_per_hour_usd: float
    power_consumption_watts: float = 400.0

    def __repr__(self) -> str:
        return f"GPUProfile({self.name}, {self.vram_gb}GB, ${self.price_per_hour_usd}/h)"


@dataclass
class CarbonEstimate:
    """Carbon and energy estimate for a training run."""
    co2_grams: float
    kwh: float
    region: str
    intensity: float  # gCO2eq/kWh used for the calculation

    def __repr__(self) -> str:
        return (
            f"CarbonEstimate(co2={self.co2_grams:.1f}g, "
            f"kwh={self.kwh:.3f}, region={self.region}, "
            f"intensity={self.intensity}gCO2/kWh)"
        )


@dataclass
class Estimate:
    """Resource estimation result."""
    vram_gb: float
    wallclock_hours: float
    cost_usd: float
    vram_uncertainty_pct: float = 30.0
    time_uncertainty_pct: float = 30.0
    carbon: Optional["CarbonEstimate"] = None

    def __repr__(self) -> str:
        carbon_str = f", {self.carbon!r}" if self.carbon else ""
        return (
            f"Estimate(vram={self.vram_gb:.1f}GB ±{self.vram_uncertainty_pct}%, "
            f"time={self.wallclock_hours:.2f}h ±{self.time_uncertainty_pct}%, "
            f"cost=${self.cost_usd:.2f}{carbon_str})"
        )


@dataclass
class Recommendation:
    """Algorithm recommendation with scoring."""
    algorithm: str
    score: float
    reason: str

    def __repr__(self) -> str:
        return f"Recommendation({self.algorithm}, score={self.score:.2f}, reason={self.reason})"


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for training."""
    optimization: str
    benefit: str
    impact: str

    def __repr__(self) -> str:
        return f"OptimizationSuggestion({self.optimization}, benefit={self.benefit}, impact={self.impact})"


# ============================================================================
# GPU Price & Specs Table (Constants)
# ============================================================================

GPU_PROFILES: Dict[str, GPUProfile] = {
    "a100-40gb": GPUProfile(
        name="A100-40GB",
        vram_gb=40.0,
        tflops_fp16=312.0,
        price_per_hour_usd=1.93,
        power_consumption_watts=250.0,
    ),
    "a100-80gb": GPUProfile(
        name="A100-80GB",
        vram_gb=80.0,
        tflops_fp16=312.0,
        price_per_hour_usd=3.26,
        power_consumption_watts=250.0,
    ),
    "h100": GPUProfile(
        name="H100",
        vram_gb=80.0,
        tflops_fp16=756.0,
        price_per_hour_usd=4.00,
        power_consumption_watts=350.0,
    ),
    "l4": GPUProfile(
        name="L4",
        vram_gb=24.0,
        tflops_fp16=120.0,
        price_per_hour_usd=0.35,
        power_consumption_watts=70.0,
    ),
    "t4": GPUProfile(
        name="T4",
        vram_gb=16.0,
        tflops_fp16=65.0,
        price_per_hour_usd=0.35,
        power_consumption_watts=70.0,
    ),
    "rtx3090": GPUProfile(
        name="RTX3090",
        vram_gb=24.0,
        tflops_fp16=142.0,
        price_per_hour_usd=0.25,
        power_consumption_watts=320.0,
    ),
    "rtx4090": GPUProfile(
        name="RTX4090",
        vram_gb=24.0,
        tflops_fp16=330.0,
        price_per_hour_usd=0.30,
        power_consumption_watts=450.0,
    ),
}

# Model size lookup table (infer parameters from model name)
MODEL_SIZES_PARAMS: Dict[str, int] = {
    "0.5b": 500,
    "1b": 1000,
    "2b": 2000,
    "3b": 3000,
    "7b": 7000,
    "13b": 13000,
    "34b": 34000,
    "70b": 70000,
    "110b": 110000,
}

# Algorithm multipliers for resource estimation
ALGORITHM_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "sft": {"vram": 1.0, "throughput": 1.0},
    "dpo": {"vram": 1.3, "throughput": 0.8},
    "ppo": {"vram": 2.5, "throughput": 0.4},
    "grpo": {"vram": 2.0, "throughput": 0.5},
    "gspo": {"vram": 1.8, "throughput": 0.55},
    "lora": {"vram": 0.3, "throughput": 1.1},
    "qlora": {"vram": 0.15, "throughput": 1.0},
    "full_tune": {"vram": 1.0, "throughput": 1.0},
}

# ============================================================================
# Utility Functions
# ============================================================================

def infer_model_size(model_name: str) -> Optional[int]:
    """
    Infer model size (in millions of parameters) from model name.

    Examples: "Qwen/Qwen2.5-7B" -> 7000, "mistral-7b" -> 7000, "llama-70b" -> 70000

    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen2.5-7B" or "7b")

    Returns:
        Model size in millions of parameters, or None if unable to infer
    """
    model_lower = model_name.lower()

    # Check for common patterns: 7B, 70B, etc.
    for size_str, size_val in MODEL_SIZES_PARAMS.items():
        # Match patterns like "7b", "7B", "-7b", "-7B"
        if size_str[:-1] in model_lower or f"-{size_str[:-1]}" in model_lower:
            return size_val

    return None


def round_to_significant_figures(value: float, sig_figs: int = 2) -> float:
    """Round value to N significant figures to avoid false precision."""
    if value == 0:
        return 0
    return round(value, -int(math.floor(math.log10(abs(value)))) + (sig_figs - 1))


# ============================================================================
# Carbon Estimation
# ============================================================================

def estimate_carbon(
    wallclock_hours: float,
    gpu_type: str,
    num_gpus: int = 1,
    region: str = "default",
) -> CarbonEstimate:
    """
    Estimate carbon emissions and energy consumption for a training run.

    Formula:
        kwh = (gpu_power_watts * num_gpus * wallclock_hours) / 1000
        co2_grams = kwh * region_carbon_intensity_gCO2_per_kwh

    Args:
        wallclock_hours: Total wall-clock training time in hours.
        gpu_type: GPU type key (e.g., "a100-40gb", "h100").  Falls back to
                  GPU_PROFILES power_consumption_watts if not found in
                  GPU_POWER_WATTS.
        num_gpus: Number of GPUs used in parallel.
        region: Cloud region identifier (e.g., "us-east-1").  Unknown regions
                fall back to "default" (global average).

    Returns:
        CarbonEstimate with co2_grams, kwh, region, and intensity used.
    """
    gpu_lower = gpu_type.lower()

    # Resolve power draw
    if gpu_lower in GPU_POWER_WATTS:
        power_watts = GPU_POWER_WATTS[gpu_lower]
    elif gpu_lower in GPU_PROFILES:
        power_watts = GPU_PROFILES[gpu_lower].power_consumption_watts
        logger.debug(f"GPU '{gpu_type}' not in GPU_POWER_WATTS, using profile power: {power_watts}W")
    else:
        power_watts = 400.0  # conservative generic default
        logger.warning(f"Unknown GPU '{gpu_type}' for carbon estimation, assuming {power_watts}W")

    # Resolve carbon intensity
    region_key = region if region in REGION_CARBON_INTENSITY else "default"
    if region_key == "default" and region != "default":
        logger.warning(f"Unknown region '{region}', using global average carbon intensity")
    intensity = REGION_CARBON_INTENSITY[region_key]

    kwh = (power_watts * num_gpus * wallclock_hours) / 1000.0
    co2_grams = kwh * intensity

    logger.info(
        f"Carbon estimate: {co2_grams:.1f}g CO2, {kwh:.3f} kWh "
        f"(GPU={gpu_type} {power_watts}W x{num_gpus}, "
        f"region={region} @ {intensity}gCO2/kWh)"
    )

    return CarbonEstimate(
        co2_grams=round(co2_grams, 2),
        kwh=round(kwh, 4),
        region=region,
        intensity=intensity,
    )


# ============================================================================
# Core Estimation Functions
# ============================================================================

def estimate_resources(
    model_name: str,
    dataset_size: int,
    algorithm: str = "sft",
    hardware_profile: str = "a100-40gb",
    batch_size: int = 4,
    seq_len: int = 512,
    num_epochs: int = 3,
    gradient_accumulation: int = 1,
    region: str = "default",
    num_gpus: int = 1,
) -> Estimate:
    """
    Estimate resource requirements for training.

    Heuristics:
    - Model size: inferred from name or model_name lookup
    - Dataset tokens: dataset_size * avg_tokens_per_sample (assume 512)
    - VRAM: (model_params + adapter_params + batch_size * seq_len) / 1B * 4 bytes
    - Training throughput: (hardware_tflops * 0.5) / model_size_tflops
    - Wallclock: total_tokens / (throughput * 1e12) — in hours
    - Cost: wallclock_hours * gpu_price_per_hour
    - Carbon: kwh * region_carbon_intensity

    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen2.5-7B")
        dataset_size: Number of samples in dataset
        algorithm: Training algorithm (sft, dpo, ppo, lora, qlora, etc.)
        hardware_profile: GPU type (a100-40gb, h100, l4, t4, rtx3090, rtx4090)
        batch_size: Training batch size per device
        seq_len: Sequence length in tokens
        num_epochs: Number of training epochs
        gradient_accumulation: Gradient accumulation steps
        region: Cloud region for carbon intensity lookup (e.g., "us-east-1")
        num_gpus: Number of GPUs (used for carbon estimation)

    Returns:
        Estimate object with vram_gb, wallclock_hours, cost_usd, carbon
    """
    algorithm_lower = algorithm.lower()
    hardware_lower = hardware_profile.lower()

    # Get GPU profile
    if hardware_lower not in GPU_PROFILES:
        logger.warning(f"Unknown GPU: {hardware_profile}, using A100-40GB as default")
        gpu = GPU_PROFILES["a100-40gb"]
    else:
        gpu = GPU_PROFILES[hardware_lower]

    # Infer model size
    model_params_millions = infer_model_size(model_name)
    if model_params_millions is None:
        logger.warning(f"Could not infer model size from {model_name}, assuming 7B")
        model_params_millions = 7000

    logger.info(f"Model size inferred: {model_params_millions}M parameters")

    # Get algorithm multipliers
    if algorithm_lower not in ALGORITHM_MULTIPLIERS:
        logger.warning(f"Unknown algorithm: {algorithm}, using SFT multipliers")
        algo_mult = ALGORITHM_MULTIPLIERS["sft"]
    else:
        algo_mult = ALGORITHM_MULTIPLIERS[algorithm_lower]

    # Estimate VRAM
    # Base formula: (model_params + batch_size * seq_len) in GB * 4 bytes (fp32)
    # For fp16/bf16 activations + gradients, assume ~2x model params
    model_params_gb = model_params_millions / 250.0  # 4 bytes per param
    activation_memory_gb = (batch_size * seq_len * model_params_millions) / (250.0 * 1000.0)

    # Adapter memory (LoRA rank=8, alpha=16 overhead ~1-5% of model)
    if algorithm_lower in ["lora", "qlora"]:
        adapter_memory_gb = model_params_gb * 0.05
    else:
        adapter_memory_gb = 0

    vram_gb = (model_params_gb + activation_memory_gb + adapter_memory_gb) * algo_mult["vram"]
    vram_gb = max(vram_gb, 0.5)  # Floor at 0.5 GB

    logger.info(f"VRAM estimate: {vram_gb:.2f} GB (model={model_params_gb:.2f}GB, "
                f"activation={activation_memory_gb:.2f}GB, adapter={adapter_memory_gb:.2f}GB)")

    # Estimate training throughput
    # Rough heuristic: (GPU_TFLOPS * 0.5 utilization) / model_TFLOPS_to_train
    # Assume 0.5 TFLOPS for training (conservative, includes memory stalls)
    # Model TFLOPS ≈ params * 2 (multiply-accumulate)
    model_tflops_to_train = model_params_millions * 2.0 / 1000.0  # in TFLOPS
    utilization = 0.5  # Conservative hardware utilization
    training_tflops = (gpu.tflops_fp16 * utilization) / max(model_tflops_to_train, 1.0)
    training_tflops *= algo_mult["throughput"]

    # Tokens per second: training_tflops / params (rough approximation)
    tokens_per_second = training_tflops * 1e12 / (model_params_millions * 1e6)
    tokens_per_second = max(tokens_per_second, 1.0)  # Floor at 1 tok/s

    logger.info(f"Training throughput: {tokens_per_second:.2f} tokens/second")

    # Total tokens to process
    avg_tokens_per_sample = seq_len  # Assume samples fill the context
    total_tokens = dataset_size * avg_tokens_per_sample * num_epochs

    # Wallclock hours
    wallclock_hours = (total_tokens / tokens_per_second) / 3600.0

    logger.info(f"Total tokens: {total_tokens:.2e}, wallclock hours: {wallclock_hours:.2f}h")

    # Cost
    cost_usd = wallclock_hours * gpu.price_per_hour_usd

    # Round to 2 significant figures to avoid false precision
    vram_gb = round_to_significant_figures(vram_gb, 2)
    wallclock_hours = round_to_significant_figures(wallclock_hours, 2)
    cost_usd = round_to_significant_figures(cost_usd, 2)

    logger.info(f"Final estimate: {vram_gb:.1f}GB, {wallclock_hours:.2f}h, ${cost_usd:.2f}")

    # Carbon estimate
    carbon = estimate_carbon(
        wallclock_hours=wallclock_hours,
        gpu_type=hardware_lower,
        num_gpus=num_gpus,
        region=region,
    )

    return Estimate(
        vram_gb=vram_gb,
        wallclock_hours=wallclock_hours,
        cost_usd=cost_usd,
        vram_uncertainty_pct=30.0,
        time_uncertainty_pct=30.0,
        carbon=carbon,
    )


def recommend_algorithm(
    task_description: str,
    dataset_size: int,
    budget_usd: Optional[float] = None,
    model_size: Optional[str] = None,
) -> List[Recommendation]:
    """
    Recommend training algorithms based on task and constraints.

    Scoring heuristics:
    - If budget_usd is set, filter out algos that exceed it
    - If task has "alignment" → rank DPO/RLHF high
    - If task has "speed" → rank LoRA/QLoRA high
    - If task has "distill" → rank KD high
    - If dataset < 10k samples → rank LoRA high
    - If dataset > 100k samples → rank full-tune/RL higher
    - Default: DPO for alignment, SFT for general, PPO for complex RL

    Args:
        task_description: Description of the task (e.g., "alignment", "speed", "distill")
        dataset_size: Number of training samples
        budget_usd: Optional budget constraint in USD
        model_size: Optional model size hint (e.g., "7b", "70b")

    Returns:
        List of Recommendation objects sorted by score descending
    """
    task_lower = task_description.lower()
    recommendations = []

    # Score each algorithm
    scores = {
        "sft": 0.70,
        "dpo": 0.75,
        "ppo": 0.65,
        "grpo": 0.70,
        "gspo": 0.68,
        "lora": 0.80,
        "qlora": 0.78,
    }

    reasons = {
        "sft": "General-purpose fine-tuning",
        "dpo": "Best for alignment, direct preference optimization",
        "ppo": "Powerful RL approach, requires more compute",
        "grpo": "Group-relative policy optimization, good efficiency",
        "gspo": "Sequence-level policy optimization, balanced alignment method",
        "lora": "Memory-efficient, works with any model",
        "qlora": "Ultra-efficient, quantized LoRA",
    }

    # Boost scores based on task keywords
    if "alignment" in task_lower or "dpo" in task_lower:
        scores["dpo"] += 0.10

    if "speed" in task_lower or "fast" in task_lower:
        scores["lora"] += 0.10
        scores["qlora"] += 0.10
        scores["sft"] += 0.05

    if "distill" in task_lower or "knowledge" in task_lower:
        scores["sft"] += 0.08
        scores["lora"] += 0.05

    # Adjust for dataset size
    if dataset_size < 10000:
        scores["lora"] += 0.15
        scores["qlora"] += 0.15
        scores["dpo"] += 0.05
        scores["ppo"] -= 0.10
        reasons["lora"] = "Best for small datasets, memory-efficient"
        reasons["qlora"] = "Ultra-efficient for small datasets"
    elif dataset_size > 100000:
        scores["ppo"] += 0.10
        scores["grpo"] += 0.08
        scores["sft"] += 0.05
        scores["lora"] -= 0.05
        reasons["ppo"] = "Large dataset, powerful RL approach"

    # Budget filtering (quick estimate: assume A100-40GB, SFT baseline)
    if budget_usd is not None:
        baseline_cost_per_hour = GPU_PROFILES["a100-40gb"].price_per_hour_usd
        max_hours = budget_usd / baseline_cost_per_hour

        # Filter algorithms by estimated cost (rough heuristic)
        if max_hours < 1:
            scores["qlora"] = max(0, scores["qlora"] + 0.20)
            for algo in ["ppo", "grpo"]:
                scores[algo] -= 0.50
        elif max_hours < 5:
            scores["lora"] += 0.10
            scores["qlora"] += 0.10

    # Normalize scores to [0.5, 1.0] range
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score > min_score:
        for algo in scores:
            scores[algo] = 0.5 + 0.5 * (scores[algo] - min_score) / (max_score - min_score)

    # Build recommendations
    for algo in sorted(scores.keys(), key=lambda x: scores[x], reverse=True):
        recommendations.append(Recommendation(
            algorithm=algo,
            score=round(scores[algo], 2),
            reason=reasons[algo],
        ))

    logger.info(f"Generated {len(recommendations)} algorithm recommendations")
    return recommendations


def suggest_optimizations(
    model_size: str,
    precision: str = "fp32",
    hardware: str = "a100-40gb",
    dataset_size: int = 10000,
    vram_tight: bool = False,
    carbon: Optional["CarbonEstimate"] = None,
    carbon_threshold_grams: float = 1000.0,
    current_region: str = "default",
) -> List[OptimizationSuggestion]:
    """
    Suggest optimizations for training based on configuration.

    Rules:
    - If model > 7B and precision != "quant" → suggest QLoRA
    - If dataset > 50k → suggest gradient_accumulation
    - If hardware is Ampere+ → suggest FlashAttention-2
    - If hardware has Tensor cores and precision == "fp32" → suggest bf16
    - If model in ["llama", "mistral"] → suggest LoRA
    - If dataset_size < 1000 → warn "too small for RL"
    - If carbon > carbon_threshold_grams → suggest greener region

    Args:
        model_size: Model size string (e.g., "7b", "70b")
        precision: Training precision (fp32, fp16, bf16, int8, int4)
        hardware: GPU hardware type
        dataset_size: Number of training samples
        vram_tight: Whether VRAM is constrained
        carbon: Optional CarbonEstimate from estimate_carbon()
        carbon_threshold_grams: CO2 threshold in grams above which a region
            suggestion is emitted (default 1000g = 1 kg CO2)
        current_region: The region currently in use (for comparison advice)

    Returns:
        List of OptimizationSuggestion objects
    """
    suggestions = []
    model_size_lower = model_size.lower()
    precision_lower = precision.lower()
    hardware_lower = hardware.lower()

    # Parse model size
    model_params = infer_model_size(model_size_lower)
    if model_params is None:
        # Try parsing direct number like "13b"
        try:
            model_params = int(model_size_lower.replace("b", "").replace("B", ""))
        except ValueError:
            model_params = 7000  # Default

    # QLoRA for large models
    if model_params > 7000 and "quant" not in precision_lower and vram_tight:
        suggestions.append(OptimizationSuggestion(
            optimization="QLoRA (4-bit quantization)",
            benefit="2-4x VRAM savings, minimal quality loss",
            impact="Slightly slower training (~10% throughput reduction)",
        ))
    elif model_params > 13000 and "quant" not in precision_lower:
        suggestions.append(OptimizationSuggestion(
            optimization="QLoRA (4-bit quantization)",
            benefit="2-4x VRAM savings",
            impact="Reduced throughput, improved memory efficiency",
        ))

    # Gradient accumulation for large datasets
    if dataset_size > 50000:
        suggestions.append(OptimizationSuggestion(
            optimization="Gradient accumulation (steps=4-8)",
            benefit="More stable gradients, better convergence",
            impact="Requires more training steps, slightly longer wallclock time",
        ))

    # FlashAttention-2 for modern GPUs
    if any(x in hardware_lower for x in ["a100", "h100", "rtx4090", "l4"]):
        suggestions.append(OptimizationSuggestion(
            optimization="FlashAttention-2 (fa2)",
            benefit="3-5x attention layer speedup",
            impact="No quality impact, requires hardware support",
        ))

    # Precision tuning
    if precision_lower == "fp32" and any(x in hardware_lower for x in ["a100", "h100", "rtx"]):
        suggestions.append(OptimizationSuggestion(
            optimization="Switch to BF16 precision",
            benefit="2x training speed, same accuracy as fp32",
            impact="Minimal overhead, supported on modern Ampere+ GPUs",
        ))

    # LoRA for specific models
    if any(x in model_size_lower for x in ["llama", "mistral", "qwen", "phi"]):
        suggestions.append(OptimizationSuggestion(
            optimization="LoRA / QLoRA fine-tuning",
            benefit="Works with any model size, memory efficient",
            impact="Slightly reduced model capacity, widely compatible",
        ))

    # Dataset size warnings
    if dataset_size < 1000:
        suggestions.append(OptimizationSuggestion(
            optimization="Dataset is very small (< 1k samples)",
            benefit="Use SFT + distillation, avoid RL",
            impact="RL algorithms require more diverse training signal",
        ))

    # VRAM-constrained suggestions
    if vram_tight:
        suggestions.append(OptimizationSuggestion(
            optimization="Enable gradient checkpointing",
            benefit="Reduce peak memory by ~40%",
            impact="Moderate slowdown (~20%) due to recomputation",
        ))

    # Carbon / green region suggestion
    if carbon is not None and carbon.co2_grams > carbon_threshold_grams:
        # Find the greenest region by intensity
        greenest_region = min(REGION_CARBON_INTENSITY.items(), key=lambda x: x[1])
        greenest_name, greenest_intensity = greenest_region

        current_intensity = REGION_CARBON_INTENSITY.get(
            current_region, REGION_CARBON_INTENSITY["default"]
        )

        if greenest_intensity < current_intensity:
            pct_reduction = int((1.0 - greenest_intensity / current_intensity) * 100)
            suggestions.append(OptimizationSuggestion(
                optimization=f"Use a greener cloud region (e.g., {greenest_name})",
                benefit=(
                    f"{pct_reduction}% lower carbon intensity than {current_region} "
                    f"({greenest_intensity} vs {current_intensity} gCO2eq/kWh)"
                ),
                impact=f"Could reduce CO2 by ~{pct_reduction}% with no quality impact",
            ))

    logger.info(f"Generated {len(suggestions)} optimization suggestions")
    return suggestions


# ============================================================================
# Utility for formatting results
# ============================================================================

def format_estimate_table(
    model_name: str,
    dataset_size: int,
    algorithm: str,
    hardware: str,
    estimate: Estimate,
    region: str = "default",
) -> str:
    """Format estimate as a readable table string."""
    gpu = GPU_PROFILES.get(hardware.lower(), GPU_PROFILES["a100-40gb"])
    fits = "✓" if estimate.vram_gb <= gpu.vram_gb else "✗"

    lines = [
        f"Model: {model_name} | Dataset: {dataset_size:,} samples | GPU: {gpu.name}",
        "",
        f"Estimated Resources ({algorithm.upper()}):",
        f"  - VRAM: {estimate.vram_gb:.1f} GB (±{estimate.vram_uncertainty_pct}%) [{fits}]",
        f"  - Wallclock: ~{estimate.wallclock_hours:.2f} hours (±{estimate.time_uncertainty_pct}%)",
        f"  - Cost: ~${estimate.cost_usd:.2f}",
    ]

    if estimate.carbon is not None:
        c = estimate.carbon
        lines.append(
            f"  - Carbon: ~{c.co2_grams:.1f}g CO2 (~{c.kwh:.3f} kWh)"
            f" [region: {region}, {c.intensity} gCO2eq/kWh]"
        )

    lines.append("")
    return "\n".join(lines)


def format_recommendations(recommendations: List[Recommendation]) -> str:
    """Format recommendations as readable list."""
    lines = ["Recommended Algorithms:"]
    for i, rec in enumerate(recommendations[:5], 1):
        lines.append(f"{i}. {rec.algorithm.upper()} (score {rec.score:.2f}) — {rec.reason}")
    lines.append("")
    return "\n".join(lines)


def format_optimizations(suggestions: List[OptimizationSuggestion]) -> str:
    """Format optimization suggestions as readable list."""
    if not suggestions:
        return ""

    lines = ["Optimization Tips:"]
    for sug in suggestions:
        lines.append(f"  - {sug.optimization}")
        lines.append(f"    Benefit: {sug.benefit}")
        lines.append(f"    Impact: {sug.impact}")
    lines.append("")
    return "\n".join(lines)
