"""
ES Utilities for Meta-Learning Mirror Maps

This module provides core utilities for Evolution Strategies (ES) optimization:
1. Conversions between numpy phi and torch mirror_params
2. MBPP dataset splitting
3. ES algorithms (perturbations, gradient computation)
4. State management (checkpointing, early stopping)
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


# ============================================================================
# SECTION 1: CONVERSIONS
# ============================================================================


def phi_to_mirror_params(phi: np.ndarray) -> dict:
    """
    Convert ES phi (numpy[380]) to mirror params (torch dict) for training.

    phi layout: [v[126], w[126], b[126], a[1], c[1]]

    Args:
        phi: numpy array of shape (380,) with mirror map parameters

    Returns:
        dict with keys {v, w, b, a, c} containing torch.Tensor values
    """
    if phi.shape != (380,):
        raise ValueError(f"Expected phi shape (380,), got {phi.shape}")

    return {
        "v": torch.tensor(phi[:126], dtype=torch.float32),
        "w": torch.tensor(phi[126:252], dtype=torch.float32),
        "b": torch.tensor(phi[252:378], dtype=torch.float32),
        "a": torch.tensor(phi[378:379], dtype=torch.float32),
        "c": torch.tensor(phi[379:380], dtype=torch.float32),
    }


def mirror_params_to_phi(mirror_params: dict) -> np.ndarray:
    """
    Convert mirror params (torch dict) to ES phi (numpy[380]).

    Args:
        mirror_params: dict with keys {v, w, b, a, c} containing torch.Tensor values

    Returns:
        numpy array of shape (380,)
    """
    required_keys = {"v", "w", "b", "a", "c"}
    if not required_keys.issubset(mirror_params.keys()):
        raise ValueError(
            f"mirror_params must contain keys {required_keys}, got {
                mirror_params.keys()}")

    return np.concatenate(
        [
            mirror_params["v"].cpu().detach().numpy().flatten(),
            mirror_params["w"].cpu().detach().numpy().flatten(),
            mirror_params["b"].cpu().detach().numpy().flatten(),
            mirror_params["a"].cpu().detach().numpy().flatten(),
            mirror_params["c"].cpu().detach().numpy().flatten(),
        ]
    )


def save_phi_for_training(phi: np.ndarray, path: str):
    """
    Save ES phi as torch mirror params checkpoint for training.

    Args:
        phi: numpy array of shape (380,)
        path: checkpoint file path (e.g., 'mirror_map_params.pt')
    """
    mirror_params = phi_to_mirror_params(phi)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(mirror_params, path)


def load_phi_from_training(path: str) -> np.ndarray:
    """
    Load training checkpoint mirror params as ES phi.

    Args:
        path: checkpoint file path (e.g., 'mirror_map_params.pt')

    Returns:
        numpy array of shape (380,)
    """
    mirror_params = torch.load(path, map_location="cpu")
    return mirror_params_to_phi(mirror_params)


# ============================================================================
# SECTION 2: DATA PREPARATION
# ============================================================================


def get_mbpp_splits(test_size: float = 0.2,
                    seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split MBPP train (374) into inner_train (299) + validation (75).

    Args:
        test_size: fraction for validation split (default: 0.2 for 75 samples)
        seed: random seed for reproducibility

    Returns:
        Tuple of (inner_train_dataset, validation_dataset)
    """
    # Load MBPP train split
    mbpp = load_dataset("mbpp", split="train")

    # Split into inner_train and validation
    indices = list(range(len(mbpp)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True)

    inner_train = mbpp.select(train_indices)
    validation = mbpp.select(val_indices)

    print(
        f"MBPP split: {len(inner_train)} inner_train + {len(validation)} validation")

    return inner_train, validation


# ============================================================================
# SECTION 3: ES ALGORITHMS
# ============================================================================


def sample_perturbations(N: int, dim: int,
                         seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Sample N perturbations with antithetic pairs for variance reduction.

    Uses antithetic sampling: [ε₁, -ε₁, ε₂, -ε₂, ...]
    N must be even for proper pairing.

    Args:
        N: number of perturbations (must be even)
        dim: dimensionality (380 for mirror maps)
        seed: random seed for reproducibility

    Returns:
        List of N perturbations, each shape (dim,)
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even for antithetic sampling, got {N}")

    rng = np.random.RandomState(seed)

    # Sample N/2 base perturbations
    base_perturbations = rng.randn(N // 2, dim)

    # Create antithetic pairs
    perturbations = []
    for eps in base_perturbations:
        perturbations.append(eps)
        perturbations.append(-eps)

    return perturbations


def compute_es_gradient(fitnesses: List[float],
                        epsilons: List[np.ndarray],
                        sigma: float) -> Tuple[np.ndarray,
                                               dict]:
    """
    Compute ES gradient using raw fitness (not normalized).

    Only includes successful training runs (non-zero fitness).
    Failed runs (fitness=0) are excluded from gradient computation.

    Formula: ∇J ≈ (1/(N_success·σ)) · Σ [f_i · ε_i] for f_i > 0

    We use RAW fitness because:
    - Optimizing expected value function E[V(π)]
    - Normalization changes the objective
    - Antithetic sampling handles variance reduction

    Args:
        fitnesses: List of N fitness values (mbpp_pass@1)
        epsilons: List of N perturbations, each shape (dim,)
        sigma: ES noise standard deviation

    Returns:
        Tuple of (gradient, stats_dict)
        - gradient: shape (dim,) ES gradient estimate
        - stats_dict: {'mean_fitness', 'std_fitness', 'min_fitness', 'max_fitness', 'success_rate'}
    """
    N_total = len(fitnesses)
    if len(epsilons) != N_total:
        raise ValueError(
            f"fitnesses and epsilons must have same length, got {
                len(fitnesses)} vs {
                len(epsilons)}")

    # Convert to numpy arrays
    fitnesses = np.array(fitnesses)
    epsilons = np.array(epsilons)  # shape: (N, dim)

    # Filter out failed runs (fitness=0)
    success_mask = fitnesses > 0
    success_fitnesses = fitnesses[success_mask]
    success_epsilons = epsilons[success_mask]

    N_success = len(success_fitnesses)

    if N_success == 0:
        # All training runs failed
        print("⚠️  WARNING: All training runs failed! Returning zero gradient.")
        return np.zeros_like(epsilons[0]), {
            "mean_fitness": 0.0,
            "std_fitness": 0.0,
            "min_fitness": 0.0,
            "max_fitness": 0.0,
            "gradient_norm": 0.0,
            "success_rate": 0.0,
        }

    # Compute gradient using only successful runs
    # ∇J ≈ (1/(N_success·σ)) · Σ [f_i · ε_i] for f_i > 0
    gradient = (1.0 / (N_success * sigma)) * \
        np.sum(success_fitnesses[:, np.newaxis] * success_epsilons, axis=0)

    # Compute statistics (using only successful runs)
    success_rate = N_success / N_total
    stats = {
        "mean_fitness": float(np.mean(success_fitnesses)),
        "std_fitness": float(np.std(success_fitnesses)),
        "min_fitness": float(np.min(success_fitnesses)),
        "max_fitness": float(np.max(success_fitnesses)),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "success_rate": float(success_rate),
    }

    print(
        f"📊 ES Gradient: {N_success}/{N_total} successful runs ({success_rate:.1%})")

    return gradient, stats


# ============================================================================
# SECTION 4: STATE MANAGEMENT
# ============================================================================


@dataclass
class ESState:
    """
    Evolution Strategies state for checkpointing and resumption.

    Attributes:
        phi: Current mirror map parameters [380] as numpy array (always the best!)
        meta_iteration: Current ES iteration number
        best_fitness: Best fitness seen so far
        best_phi: Mirror map parameters that achieved best_fitness
                  NOTE: With "Restart from Best", phi == best_phi always.
                  Kept for backward compatibility with old checkpoints.
        fitness_history: List of mean fitness per iteration
        gradient_history: List of gradient norms per iteration
        fitness_std_history: List of fitness std per iteration
        fitness_min_history: List of min fitness per iteration
        fitness_max_history: List of max fitness per iteration
        accept_history: Track accept/reject decisions (True=accept, False=reject)
        elite_samples: Elite samples from rejected iterations [(epsilon, fitness), ...]
    """

    phi: np.ndarray
    meta_iteration: int
    best_fitness: float
    best_phi: np.ndarray
    fitness_history: List[float]
    gradient_history: List[float]
    fitness_std_history: List[float]
    fitness_min_history: List[float]
    fitness_max_history: List[float]
    # Track accept/reject decisions (True=accept, False=reject)
    accept_history: List[bool] = None
    # Elite samples: [(epsilon, fitness), ...]
    elite_samples: List[Tuple[np.ndarray, float]] = None

    def to_dict(self) -> dict:
        """Convert to serializable dict for JSON."""
        # Serialize elite_samples: [(epsilon, fitness), ...] -> [{"epsilon":
        # [...], "fitness": ...}, ...]
        elite_serialized = None
        if self.elite_samples is not None and len(self.elite_samples) > 0:
            elite_serialized = [{"epsilon": eps.tolist(), "fitness": float(
                fit)} for eps, fit in self.elite_samples]

        return {
            "phi": self.phi.tolist(),
            "meta_iteration": self.meta_iteration,
            "best_fitness": self.best_fitness,
            "best_phi": self.best_phi.tolist(),
            "fitness_history": self.fitness_history,
            "gradient_history": self.gradient_history,
            "fitness_std_history": self.fitness_std_history,
            "fitness_min_history": self.fitness_min_history,
            "fitness_max_history": self.fitness_max_history,
            "accept_history": self.accept_history if self.accept_history is not None else [],
            "elite_samples": elite_serialized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ESState":
        """Create from dict loaded from JSON."""
        # Deserialize elite_samples: [{"epsilon": [...], "fitness": ...}, ...]
        # -> [(epsilon, fitness), ...]
        elite_deserialized = None
        if "elite_samples" in data and data["elite_samples"] is not None:
            elite_deserialized = [
                (np.array(
                    item["epsilon"]),
                    item["fitness"]) for item in data["elite_samples"]]

        return cls(
            phi=np.array(data["phi"]),
            meta_iteration=data["meta_iteration"],
            best_fitness=data["best_fitness"],
            best_phi=np.array(data["best_phi"]),
            fitness_history=data["fitness_history"],
            gradient_history=data["gradient_history"],
            fitness_std_history=data.get("fitness_std_history", []),
            fitness_min_history=data.get("fitness_min_history", []),
            fitness_max_history=data.get("fitness_max_history", []),
            accept_history=data.get("accept_history", []),
            elite_samples=elite_deserialized,
        )


def save_es_checkpoint(state: ESState, path: str):
    """
    Save ES state to checkpoint file.

    Args:
        state: ESState instance to save
        path: checkpoint file path (e.g., 'es_checkpoint.json')
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)

    print(f"💾 Saved ES checkpoint to {path}")


def load_es_checkpoint(path: str) -> ESState:
    """
    Load ES state from checkpoint file.

    Args:
        path: checkpoint file path

    Returns:
        ESState instance
    """
    with open(path, "r") as f:
        data = json.load(f)

    state = ESState.from_dict(data)
    print(
        f"📂 Loaded ES checkpoint from {path} (iteration {
            state.meta_iteration})")

    return state


def check_early_stopping(
        fitness_history: List[float],
        best_fitness: float,
        patience: int = 5,
        min_delta: float = 0.001) -> bool:
    """
    Check if training should stop early due to no improvement.

    Stops if no improvement > min_delta for patience iterations.

    Args:
        fitness_history: List of mean fitness values per iteration
        best_fitness: Best fitness seen so far
        patience: Number of iterations without improvement before stopping
        min_delta: Minimum improvement threshold (absolute, e.g., 0.001 = 0.1%)

    Returns:
        True if should stop, False otherwise
    """
    if len(fitness_history) < patience:
        return False

    # Check last 'patience' iterations
    recent_history = fitness_history[-patience:]

    # Check if any recent fitness improved by min_delta
    for fitness in recent_history:
        if fitness > best_fitness + min_delta:
            return False  # Found improvement, don't stop

    # No significant improvement in last 'patience' iterations
    return True


def initialize_es_state(
        dim: int = 380,
        init_scale: float = 0.01,
        seed: int = 42) -> ESState:
    """
    Initialize ES state with random mirror map parameters.

    Args:
        dim: Dimensionality of mirror map (default: 380)
        init_scale: Initialization scale for parameters
        seed: Random seed for reproducibility

    Returns:
        ESState instance with initialized phi
    """
    rng = np.random.RandomState(seed)
    phi = rng.randn(dim) * init_scale

    return ESState(
        phi=phi,
        meta_iteration=0,
        best_fitness=-np.inf,
        best_phi=phi.copy(),
        fitness_history=[],
        gradient_history=[],
        fitness_std_history=[],
        fitness_min_history=[],
        fitness_max_history=[],
        accept_history=[],
        elite_samples=[],
    )
