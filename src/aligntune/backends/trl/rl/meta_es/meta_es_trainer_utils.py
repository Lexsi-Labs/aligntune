"""
Meta-Learning ES Training for Mirror Maps

This is the main coordinator for Evolution Strategies (ES) meta-learning.
Optimizes 380 mirror map parameters to maximize E[V(π_T^{h_φ})].
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
import wandb

# Import ES utilities
from .es_utils import (
    ESState,
    phi_to_mirror_params,
    save_phi_for_training,
    sample_perturbations,
    compute_es_gradient,
    save_es_checkpoint,
    load_es_checkpoint,
    check_early_stopping,
    initialize_es_state,
    get_mbpp_splits,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


class ESConfig:
    """Configuration for ES meta-learning."""

    def __init__(self, args):
        # ES hyperparameters
        self.N = args.N  # Population size (must be even)
        self.T = args.T  # Training steps per evaluation
        self.meta_iterations = args.meta_iterations
        self.sigma = args.sigma  # ES noise std
        self.sigma_decay = args.sigma_decay  # ES sigma decay per iteration
        self.alpha = args.alpha  # ES learning rate
        self.patience = args.patience
        self.min_delta = args.min_delta

        # Training configuration
        self.base_model = args.base_model
        self.per_device_batch_size = args.per_device_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.num_generations = args.num_generations
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.mirror_coefficient = args.mirror_coefficient
        self.use_lora = args.use_lora
        self.lora_r = args.lora_r
        self.bf16 = args.bf16
        self.debug_mode = args.debug_mode

        # Parallelization
        self.num_workers = args.num_workers

        # Paths
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialization
        self.init_scale = args.init_scale
        self.seed = args.seed

        # Evaluation
        self.eval_timeout = args.eval_timeout
        self.eval_max_tokens = args.eval_max_tokens
        self.eval_k = args.eval_k  # Number of samples for pass@k
        self.eval_temperature = args.eval_temperature  # Sampling temperature for pass@k

        # Logging
        self.use_wandb = not args.no_wandb
        self.wandb_project = args.wandb_project

    def __repr__(self):
        return (
            f"ESConfig(\n" f"  N={
                self.N}, T={
                self.T}, meta_iterations={
                self.meta_iterations}\n" f"  sigma={
                    self.sigma}, sigma_decay={
                        self.sigma_decay}, alpha={
                            self.alpha}\n" f"  patience={
                                self.patience}, min_delta={
                                    self.min_delta}\n" f"  num_workers={
                                        self.num_workers}\n" f"  output_dir={
                                            self.output_dir}\n" f")")


# ============================================================================
# WORKER FUNCTIONS
# ============================================================================


def _train_worker_wrapper(args):
    """
    Module-level wrapper for multiprocessing (required for pickling).

    This wrapper must be at module level because multiprocessing with 'spawn'
    method can't pickle local functions defined inside other functions.

    Args:
        args: Tuple of (rank, phi, run_id, config, results_dict, show_output)

    Returns:
        None (results stored in results_dict)
    """
    rank, phi, run_id, config, results_dict, show_output = args
    checkpoint_dir = train_worker(rank, phi, run_id, config, show_output)
    results_dict[rank] = checkpoint_dir


def train_worker(
        rank: int,
        phi_perturbed: np.ndarray,
        run_id: str,
        config: ESConfig,
        show_output: bool = True) -> Optional[str]:
    """
    Worker function to train a policy with a perturbed mirror map.

    This function ONLY trains the policy and returns the checkpoint path.
    Evaluation is done separately in the main process to avoid vLLM conflicts.

    Args:
        rank: Worker rank for GPU memory management
        phi_perturbed: Perturbed mirror map parameters [380]
        run_id: Unique identifier for this run
        config: ES configuration
        show_output: If True, stream training output to console. If False, capture output (silent).

    Returns:
        checkpoint_dir: Path to trained model checkpoint, or None if training failed
    """
    print(f"🔧 Worker {rank}: Starting training with run_id={run_id}")
    sys.stdout.flush()

    # GPU setup and memory management
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus  # Distribute workers across GPUs

        # Set this worker to use specific GPU
        torch.cuda.set_device(gpu_id)

        # Clear GPU cache before training (only affects this process)
        torch.cuda.empty_cache()
        print(f"🔧 Worker {rank}: Using GPU {gpu_id}/{num_gpus - 1}")
        print(f"🧹 Worker {rank}: Cleared GPU {gpu_id} cache (process-local)")
        sys.stdout.flush()

        # Set CUDA memory allocation config
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Let PyTorch manage GPU memory automatically (no hard limits)
        # This prevents workers from blocking when soft limits are exceeded
        workers_per_gpu = (config.train.num_workers + num_gpus - 1) // num_gpus
        print(
            f"💾 Worker {rank}: Using GPU {gpu_id} ({workers_per_gpu} workers sharing this GPU)")

    # Create temporary directory for this run
    run_dir = Path(config.logging.output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save perturbed mirror map parameters for training
    mirror_params_path = run_dir / "mirror_map_params.pt"
    save_phi_for_training(phi_perturbed, str(mirror_params_path))

    if config.datasets[0].name.lower() == 'gsm8k':
        file = "train_nmdrgrpo_es_math.py"
    elif config.datasets[0].name.lower() == 'mbpp':
        file = "train_nmdrgrpo_es_code.py"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, file)

    # Build training command
    train_cmd = [
        sys.executable,
        # "train_nmdrgrpo_es_code.py",
        script_path,
        "--model",
        # config.model.base_model,
        config.model.name_or_path,
        "--max_steps",
        str(config.train.T),
        "--per_device_train_batch_size",
        str(config.train.per_device_batch_size),
        "--grad_accum",
        str(config.train.gradient_accumulation_steps),
        "--num_generations",
        str(config.train.num_generations),
        "--max_prompt_length",
        str(config.train.max_prompt_length),
        "--max_completion_length",
        str(config.train.max_completion_length),
        "--mirror_coefficient",
        str(config.train.mirror_coefficient),
        "--mirror_init_scale",
        str(config.train.init_scale),
        "--output_dir",
        str(run_dir),
        "--es_mode",  # Enable ES mode (uses 80% train split)
        "--mirror_params_path",
        str(mirror_params_path),
        "--no_wandb",  # Disable wandb for parallel runs
        "--mirror_seed",
        # Use same seed for all runs (isolates mirror map effect)
        str(config.train.seed),
    ]

    if config.model.use_peft:
        # Create LoRA config from your model config
        train_cmd.extend(["--use_lora", "--lora_r", str(config.model.lora_r)])

    # if config.bf16:
    if config.model.precision == "bf16":
        train_cmd.append("--bf16")

    if config.train.debug_mode:
        train_cmd.append("--debug_mode")

    # Run training subprocess
    print(f"🚀 Worker {rank}: Running training for {config.train.T} steps...")
    if show_output:
        print(f"{'=' * 80}")

    try:
        if show_output:
            # Show training output in real-time
            result = subprocess.run(
                train_cmd, timeout=43200)  # 12 hour timeout
        else:
            # Capture output (silent mode)
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=43200)

        if show_output:
            print(f"{'=' * 80}")

        if result.returncode != 0:
            print(
                f"❌ Worker {rank}: Training failed with return code {
                    result.returncode}")
            if not show_output and hasattr(result, "stderr"):
                print(f"   stderr: {result.stderr[-500:]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"⏱️  Worker {rank}: Training timed out!")
        return None
    except Exception as e:
        print(f"💥 Worker {rank}: Training crashed: {e}")
        return None

    # Find checkpoint directory (should be final checkpoint)
    checkpoint_dir = run_dir / f"checkpoint-{config.train.T}"
    if not checkpoint_dir.exists():
        # Try to find any checkpoint
        checkpoints = sorted(run_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
        else:
            print(f"❌ Worker {rank}: No checkpoint found!")
            sys.stdout.flush()
            return None

    print(
        f"✅ Worker {rank}: Training completed, checkpoint saved to {checkpoint_dir}")
    sys.stdout.flush()

    return str(checkpoint_dir)


def train_parallel_evaluate_sequential(
    phi_list: List[np.ndarray], config: ESConfig, validation_dataset: List[dict]
) -> List[float]:
    """
    Train in parallel, then evaluate sequentially.

    Phase 1: Train N policies in parallel (PyTorch only, no vLLM conflicts)
    Phase 2: Evaluate N checkpoints sequentially (vLLM in main process)

    This separation avoids vLLM multiprocessing conflicts while preserving
    parallelization for the expensive training phase.

    Args:
        phi_list: List of N perturbed mirror maps, each [380]
        config: ES configuration
        validation_dataset: Validation dataset

    Returns:
        List of N fitness values (mbpp_pass@1)
    """
    N = len(phi_list)
    print(f"\n{'=' * 60}")
    print(
        f"🔄 Phase 1: Parallel training ({N} models, {
            config.train.num_workers} workers)")
    print(f"{'=' * 60}\n")

    # ========================================================================
    # PHASE 1: TRAINING (no vLLM)
    # ========================================================================

    if config.train.num_workers == 1:
        # Sequential training - show output for all runs
        checkpoint_dirs = []
        for i, phi in enumerate(phi_list):
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            checkpoint_dir = train_worker(
                0, phi, run_id, config, show_output=True)
            checkpoint_dirs.append(checkpoint_dir)
    else:
        # Parallel training with multiprocessing
        mp.set_start_method("spawn", force=True)

        # Create batches for parallel execution
        batches = [phi_list[i: i + config.train.num_workers]
                   for i in range(0, N, config.train.num_workers)]

        checkpoint_dirs = []
        for batch_idx, batch in enumerate(batches):
            print(
                f"\n📦 Training batch {
                    batch_idx + 1}/{
                    len(batches)} ({
                    len(batch)} models in parallel)")

            # Create processes for this batch
            processes = []
            results = mp.Manager().dict()

            for local_rank, phi in enumerate(batch):
                global_idx = batch_idx * config.train.num_workers + local_rank
                run_id = f"{
                    datetime.now().strftime('%Y%m%d_%H%M%S')}_{global_idx}"

                # Use module-level wrapper for pickling compatibility
                # Show output only for first worker (local_rank==0) to avoid
                # messy parallel logs
                show_output = local_rank == 0
                args = (local_rank, phi, run_id, config, results, show_output)
                p = mp.Process(target=_train_worker_wrapper, args=(args,))
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect checkpoint paths
            batch_checkpoints = [results[i] for i in range(len(batch))]
            checkpoint_dirs.extend(batch_checkpoints)

            print(f"✅ Training batch {batch_idx + 1} completed")

    print(f"\n{'=' * 60}")
    print(f"✅ Phase 1 complete: {len(checkpoint_dirs)} models trained")
    print(f"{'=' * 60}\n")

    # ========================================================================
    # PHASE 2: SEQUENTIAL EVALUATION (vLLM in main process, zero conflicts)
    # ========================================================================

    print(f"{'=' * 60}")
    print(f"📊 Phase 2: Sequential evaluation with vLLM ({N} models)")
    print(f"{'=' * 60}\n")

    fitnesses = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        if checkpoint_dir is None:
            # Training failed for this model
            print(
                f"⚠️  Model {
                    i + 1}/{N}: Training failed, assigning fitness=0.0")
            fitnesses.append(0.0)
            continue

        print(f"\n🔍 Evaluating model {i + 1}/{N}: {checkpoint_dir}")

        try:
            fitness = compute_value_function_vllm(
                checkpoint_dir=checkpoint_dir,
                validation_dataset=validation_dataset,
                base_model=config.base_model,
                max_new_tokens=config.eval_max_tokens,
                timeout=config.eval_timeout,
                k=config.eval_k,
                temperature=config.eval_temperature,
                verbose=False,
            )
            fitnesses.append(fitness)
            print(f"✅ Model {i + 1}/{N}: fitness={fitness:.4f}")

            # Clean up training checkpoint to save disk space
            # Only keep ES checkpoints (best_mirror_map_params.pt,
            # es_checkpoint_iter*.json)
            import shutil

            run_dir = Path(checkpoint_dir).parent
            print(f"🧹 Cleaning up {run_dir.name} to save disk space...")
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"✅ Cleanup complete\n")

        except Exception as e:
            print(f"💥 Model {i + 1}/{N}: Evaluation crashed: {e}")
            fitnesses.append(0.0)

    print(f"{'=' * 60}")
    print(f"✅ Phase 2 complete: All evaluations done")
    print(f"   Fitness range: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]")
    print(f"   Mean fitness: {sum(fitnesses) / len(fitnesses):.4f}")
    print(f"{'=' * 60}\n")

    return fitnesses


# ============================================================================
# ES ITERATION
# ============================================================================


def run_es_iteration(
        state: ESState,
        config: ESConfig,
        validation_dataset: List[dict]) -> ESState:
    """
    Run one ES iteration.

    Steps:
    1. Sample N perturbations with antithetic sampling
    2. Train N policies in parallel
    3. Evaluate each policy on validation set
    4. Compute ES gradient
    5. Update phi
    6. Update state and check early stopping

    Args:
        state: Current ES state
        config: ES configuration
        validation_dataset: Validation dataset

    Returns:
        Updated ES state
    """
    print(f"\n{'#' * 60}")
    print(
        f"# ES ITERATION {state.meta_iteration + 1}/{config.train.meta_iterations}")
    print(f"{'#' * 60}\n")

    # Step 1: Calculate current sigma with decay
    current_sigma = config.train.sigma * \
        (config.train.sigma_decay**state.meta_iteration)

    # Step 1.5: Handle elitism - reuse elite samples from previous rejected
    # iteration
    elite_count = 0
    elite_epsilons = []
    elite_fitnesses = []

    if state.elite_samples and len(state.elite_samples) > 0:
        elite_count = len(state.elite_samples)

        # CRITICAL: Ensure elite_count is EVEN for antithetic sampling
        # If odd (from old checkpoint), drop the worst elite sample
        if elite_count % 2 != 0:
            print(
                f"⚠️  WARNING: Found {elite_count} elite samples (odd) - adjusting to even")
            elite_count = elite_count - 1  # Drop worst sample
            # Sort by fitness descending and keep top elite_count
            sorted_elites = sorted(
                state.elite_samples,
                key=lambda x: x[1],
                reverse=True)
            adjusted_elites = sorted_elites[:elite_count]
            elite_epsilons = [eps for eps, _ in adjusted_elites]
            elite_fitnesses = [fit for _, fit in adjusted_elites]
            print(
                f"   Adjusted to {elite_count} elite samples (dropped worst)")
        else:
            elite_epsilons = [eps for eps, _ in state.elite_samples]
            elite_fitnesses = [fit for _, fit in state.elite_samples]

        print(
            f"♻️  Reusing {elite_count} elite samples from previous iteration")
        print(f"   Elite fitnesses: {[f'{f:.4f}' for f in elite_fitnesses]}")

    # Sample only remaining perturbations (N - elite_count)
    n_new = config.train.N - elite_count
    print(f"🎲 Sampling {n_new} new perturbations (antithetic)...")
    print(
        f"   Sigma: {
            current_sigma:.6f} (initial={
            config.train.sigma}, decay={
                config.train.sigma_decay}, iteration={
                    state.meta_iteration})")
    seed = config.train.seed + state.meta_iteration
    new_epsilons = sample_perturbations(n_new, dim=380, seed=seed)

    # Combine elite and new perturbations
    epsilons = elite_epsilons + new_epsilons

    # Step 2: Create perturbed mirror maps with decayed sigma
    # Elite samples already have their phi from previous iteration, but we recreate them
    # This is fine because phi hasn't changed (Restart from Best)
    new_phi_list = [state.phi + current_sigma * eps for eps in new_epsilons]

    # Step 3: Train and evaluate ONLY new samples (reuse elite fitnesses)
    print(f"\n🏋️  Training and evaluating {n_new} new models...")
    new_fitnesses = train_parallel_evaluate_sequential(
        new_phi_list, config, validation_dataset)

    # Combine elite and new fitnesses
    fitnesses = elite_fitnesses + new_fitnesses
    print(
        f"\n📊 Total population: {elite_count} elite + {n_new} new = {len(fitnesses)} samples")

    # Step 4: Compute mean fitness for acceptance decision
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    min_fitness = np.min(fitnesses)
    max_fitness = np.max(fitnesses)

    print(
        f"\n📊 Iteration results: mean={mean_fitness:.4f}, "
        f"std={std_fitness:.4f}, "
        f"min={min_fitness:.4f}, "
        f"max={max_fitness:.4f}"
    )
    print(f"   Current best: {state.best_fitness:.4f}")

    # Step 5: ACCEPTANCE DECISION (before computing gradient!)
    # Convert numpy.bool_ to Python bool for JSON serialization
    accepted = bool(mean_fitness > state.best_fitness)

    if accepted:
        # ACCEPT: Compute gradient and update phi
        print(f"\n✅ ACCEPT: {mean_fitness:.4f} > {state.best_fitness:.4f}")
        print(f"   Improvement: +{mean_fitness - state.best_fitness:.4f}")
        print(f"   Computing gradient and updating phi...")

        gradient, stats = compute_es_gradient(
            fitnesses, epsilons, current_sigma)
        new_phi = state.phi + config.alpha * gradient
        new_best_fitness = mean_fitness

        print(f"   Gradient norm: {stats['gradient_norm']:.6f}")
        if "success_rate" in stats:
            print(f"   Success rate: {stats['success_rate']:.1%}")

        # Clear elite samples on success (no longer needed)
        new_elite_samples = []

    else:
        # REJECT: Skip gradient computation, keep current phi
        print(f"\n❌ REJECT: {mean_fitness:.4f} ≤ {state.best_fitness:.4f}")
        print(
            f"   No improvement (gap: {
                state.best_fitness -
                mean_fitness:.4f})")
        print(f"   Keeping current phi (no gradient computation)")

        new_phi = state.phi  # No change
        new_best_fitness = state.best_fitness  # No change
        stats = {
            "gradient_norm": 0.0,  # Not computed
            "mean_fitness": mean_fitness,
            "std_fitness": std_fitness,
            "min_fitness": min_fitness,
            "max_fitness": max_fitness,
        }

        # ELITISM: Save top 25% samples for next iteration
        # Ensure elite_k is EVEN for antithetic sampling compatibility
        # (so that n_new = N - elite_k is also even)
        elite_k_target = config.train.N // 4  # Target: 25%
        # Round to nearest even, at least 2
        elite_k = max(2, (elite_k_target // 2) * 2)
        elite_indices = np.argsort(fitnesses)[-elite_k:]  # Indices of top K
        new_elite_samples = [(epsilons[i], fitnesses[i])
                             for i in elite_indices]

        print(f"\n📦 ELITISM: Keeping top {elite_k} samples for next iteration")
        elite_fits = [fitnesses[i] for i in elite_indices]
        print(
            f"   Elite fitnesses: {[f'{f:.4f}' for f in sorted(elite_fits, reverse=True)]}")

    # Step 6: Update state
    new_meta_iteration = state.meta_iteration + 1
    new_best_phi = new_phi  # Always equals new_phi (invariant)

    # Update history
    new_fitness_history = state.fitness_history + [mean_fitness]
    new_gradient_history = state.gradient_history + [stats["gradient_norm"]]
    new_fitness_std_history = state.fitness_std_history + \
        [stats["std_fitness"]]
    new_fitness_min_history = state.fitness_min_history + \
        [stats["min_fitness"]]
    new_fitness_max_history = state.fitness_max_history + \
        [stats["max_fitness"]]
    new_accept_history = (state.accept_history or []) + [accepted]

    # Create new state
    new_state = ESState(
        phi=new_phi,
        meta_iteration=new_meta_iteration,
        best_fitness=new_best_fitness,
        best_phi=new_best_phi,
        fitness_history=new_fitness_history,
        gradient_history=new_gradient_history,
        fitness_std_history=new_fitness_std_history,
        fitness_min_history=new_fitness_min_history,
        fitness_max_history=new_fitness_max_history,
        accept_history=new_accept_history,
        elite_samples=new_elite_samples,
    )

    # Log to wandb
    if config.use_wandb:
        # Calculate accept rate
        accept_rate = sum(new_accept_history) / \
            len(new_accept_history) if new_accept_history else 0.0

        wandb_log = {
            "es_iteration": new_meta_iteration,
            "fitness/mean": mean_fitness,
            "fitness/std": stats["std_fitness"],
            "fitness/min": stats["min_fitness"],
            "fitness/max": stats["max_fitness"],
            "fitness/best": new_best_fitness,
            "gradient/norm": stats["gradient_norm"],
            "fitness/improved": 1.0 if accepted else 0.0,
            "es/accept_rate": accept_rate,
            "es/sigma": current_sigma,
        }

        # Add success rate if available
        if "success_rate" in stats:
            wandb_log["training/success_rate"] = stats["success_rate"]

        wandb.log(wandb_log, step=new_meta_iteration)

    # Save checkpoint
    checkpoint_path = config.logging.output_dir / \
        f"es_checkpoint_iter{new_meta_iteration}.json"
    save_es_checkpoint(new_state, str(checkpoint_path))

    # Save best mirror map
    best_mirror_path = config.logging.output_dir / "best_mirror_map_params.pt"
    save_phi_for_training(new_best_phi, str(best_mirror_path))
    print(f"💾 Saved best mirror map to {best_mirror_path}")

    return new_state
