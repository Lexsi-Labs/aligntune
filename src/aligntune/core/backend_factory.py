"""
Backend Factory for AlignTune - FIXED VERSION

This module provides a clean backend selection system where users can choose:
1. Training Type: SFT or RL
2. Backend: Unsloth or TRL
3. Algorithm: (for RL) DPO, PPO, GRPO, GSPO

The factory pattern ensures proper backend selection and fallback handling.
"""

import logging
import os
from typing import Dict, Any, Optional, Type, List, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import yaml
import json

# Import colored logging
try:
    from ..utils.colored_logging import (
        print_section_banner,
        aligntune_info,
        aligntune_success,
        aligntune_warning,
        Fore,
    )
    COLORED_LOGGING_AVAILABLE = True
except ImportError:
    COLORED_LOGGING_AVAILABLE = False
    # Fallback functions
    def print_section_banner(title, char="=", width=80, color=""):
        print("\n" + char * width)
        print(f"  {title}".center(width))
        print(char * width + "\n")
    
    def aligntune_info(msg, prefix="[aligntune]"):
        print(f"{prefix} INFO - {msg}")
    
    def aligntune_success(msg, prefix="[aligntune]"):
        print(f"{prefix} ✓ {msg}")
    
    def aligntune_warning(msg, prefix="[aligntune]"):
        print(f"{prefix} WARNING - {msg}")
    
    class Fore:
        CYAN = ""
        RESET = ""


# CRITICAL: Only set PURE_TRL_MODE when TRL is being used to prevent Unsloth interference
# This ensures TRL backends work without Unsloth patches, but allows Unsloth when requested
# We'll set this conditionally based on the backend being used


from .rl.config import (
    UnifiedConfig,
    ModelConfig as RLModelConfig,
    DatasetConfig as RLDatasetConfig,
    TrainingConfig as RLTrainingConfig,
    LoggingConfig as RLLoggingConfig,
    SampleLoggingConfig,
)
from .sft.config import SFTConfig, ModelConfig as SFTModelConfig, DatasetConfig as SFTDatasetConfig, TrainingConfig as SFTTrainingConfig, LoggingConfig as SFTLoggingConfig, EvaluationConfig as SFTEvaluationConfig
from .long_context.rope_config import RopeConfig

# Distillation config - optional dependency
try:
    from .distill.config import UnifiedDistillConfig, DistillationType
    DISTILL_CONFIG_AVAILABLE = True
except ImportError:
    DISTILL_CONFIG_AVAILABLE = False
    UnifiedDistillConfig = None
    DistillationType = None

from .rl.trainer_base import TrainerBase as RLTrainerBase
from .sft.trainer_base import SFTTrainerBase
from ..utils.config_utils import parse_config_to_unified,  load_config
from ..utils.environment import set_seed  # Import seed utility

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Import backend trainers conditionally to avoid import errors

try:
    import vllm.sampling_params as _vsp
    if not hasattr(_vsp, 'GuidedDecodingParams') and hasattr(_vsp, 'StructuredOutputsParams'):
        _vsp.GuidedDecodingParams = _vsp.StructuredOutputsParams
except Exception:
    pass


logger.debug("Attempting to import TRL backends...")

TRL_AVAILABLE = False
TRL_SFT_AVAILABLE = False
TRL_RL_AVAILABLE = False
TRL_DISTILL_AVAILABLE = False

# SFT Trainer
try:
    from ..backends.trl.sft.sft_generation import TRLSFTTrainer
    logger.debug("TRLSFTTrainer imported")
    TRL_SFT_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TRLSFTTrainer import failed: {e}")

# RL Trainers
try:
    from ..backends.trl.rl.dpo.dpo import TRLDPOTrainer
    from ..backends.trl.rl.online_dpo.online_dpo import TRLOnlineDPOTrainer
    from ..backends.trl.rl.ppo.ppo import TRLPPOTrainer
    from ..backends.trl.rl.grpo.grpo import TRLGRPOTrainer
    from ..backends.trl.rl.gspo.gspo import TRLGSPOTrainer
    from ..backends.trl.rl.counterfact_grpo.counterfact_grpo import TRLCounterFactGRPOTrainer
    from ..backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
    from ..backends.trl.rl.dr_grpo.drgrpo import TRLDRGRPOTrainer
    from ..backends.trl.rl.dapo.dapo import TRLDAPOTrainer
    from ..backends.trl.rl.pace.pace import TRLPaceTrainer
    from ..backends.trl.rl.orpo.orpo import TRLORPOTrainer
    from ..backends.trl.rl.spin.spin import TRLSPINTrainer
    logger.debug("TRL RL trainers imported")
    TRL_RL_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TRL RL trainers import failed: {e}")

# Distillation Trainers
try:
    from ..backends.trl.distill.distillation.distillation import TRLDistillationTrainer
    from ..backends.trl.distill.sdft.sdft import TRLSDFTTrainer
    logger.debug("TRL Distillation trainers imported")
    TRL_DISTILL_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TRL Distillation trainers import failed: {e}")

# Set TRL_AVAILABLE if at least SFT or RL trainers work
TRL_AVAILABLE = TRL_SFT_AVAILABLE or TRL_RL_AVAILABLE
logger.debug(f"TRL_AVAILABLE = {TRL_AVAILABLE}")

# Evolution Strategies backend
try:
    from ..backends.es.trainer import ESTrainerBackend
    ES_AVAILABLE = True
except ImportError:
    logger.debug("ES backend not available")
    ES_AVAILABLE = False
    ESTrainerBackend = None

def _check_unsloth_available():
    """Check if Unsloth is available without importing backends."""
    try:
        from .._imports import _check_unsloth_available as _lazy_check
        return _lazy_check()
    except ImportError:
        return False

# Don't check Unsloth availability at import time to avoid global patching
UNSLOTH_AVAILABLE = None

# Backend management functions will be defined after imports

# Legacy backend removed - all functionality now uses TRL or Unsloth backends


def _lazy_import_unsloth_trainer(algorithm: str, training_type: str):
    """Lazy import Unsloth trainers to ensure proper import order."""
    # Force check here before importing
    global UNSLOTH_AVAILABLE
    if UNSLOTH_AVAILABLE is None:
        UNSLOTH_AVAILABLE = _check_unsloth_available()
    
    if not UNSLOTH_AVAILABLE:
        # Get detailed error information
        from .._imports import UNSLOTH_ERROR_INFO
        
        if UNSLOTH_ERROR_INFO:
            error_msg = f"Unsloth not available: {UNSLOTH_ERROR_INFO['error_type']}\n"
            error_msg += f"Error: {UNSLOTH_ERROR_INFO['error']}\n"
            error_msg += f"Environment: PyTorch {UNSLOTH_ERROR_INFO['environment'].get('pytorch_version', 'unknown')}, CUDA {UNSLOTH_ERROR_INFO['environment'].get('cuda_version', 'unknown')}\n"
            error_msg += "Suggestions:\n"
            for suggestion in UNSLOTH_ERROR_INFO['suggestion']:
                error_msg += f"  - {suggestion}\n"
            error_msg += "\nAlternatively, use TRL backends instead: --backend trl"
        else:
            error_msg = "Unsloth not available. Install with: pip install unsloth\nAlternatively, use TRL backends instead: --backend trl"
        
        raise ImportError(error_msg)
    
    # Import unsloth FIRST
    try:
        import unsloth
    except Exception as e:
        # This is where the actual error happens
        from .._imports import _categorize_unsloth_error, _get_unsloth_fix_suggestion
        import torch
        
        env_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        error_type = _categorize_unsloth_error(e)
        error_msg = f"Unsloth not available: {error_type}\n"
        error_msg += f"Error: {e}\n"
        error_msg += f"Environment: PyTorch {env_info['pytorch_version']}, CUDA {env_info['cuda_version']}\n"
        error_msg += "Suggestions:\n"
        for suggestion in _get_unsloth_fix_suggestion(error_type, env_info):
            error_msg += f"  - {suggestion}\n"
        error_msg += "\nAlternatively, use TRL backends instead: --backend trl"
        
        raise ImportError(error_msg)
    
    # Now safe to import backends
    try:
        if training_type == "sft":
            from ..backends.unsloth.sft.sft import UnslothSFTTrainer
            return UnslothSFTTrainer
        elif algorithm == "dpo":
            from ..backends.unsloth.rl.dpo.dpo import UnslothDPOTrainer
            return UnslothDPOTrainer
        elif algorithm == "ppo":
            from ..backends.unsloth.rl.ppo.ppo import UnslothPPOTrainer
            return UnslothPPOTrainer
        elif algorithm == "grpo":
            from ..backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
            return UnslothGRPOTrainer
        elif algorithm == "drgrpo":
            from ..backends.unsloth.rl.dr_grpo.drgrpo import UnslothDRGRPOTrainer
            return UnslothDRGRPOTrainer
        elif algorithm == "dapo":
            from ..backends.unsloth.rl.dapo.dapo import UnslothDAPOTrainer
            return UnslothDAPOTrainer
        elif algorithm == "gspo":
            from ..backends.unsloth.rl.gspo.gspo import UnslothGSPOTrainer
            return UnslothGSPOTrainer
        elif algorithm == "pace":
            from ..backends.unsloth.rl.pace.pace import UnslothPaceTrainer
            return UnslothPaceTrainer
        elif algorithm == "counterfact_grpo":
            from ..backends.unsloth.rl.counterfact_grpo.counterfact_grpo import UnslothCounterFactGRPOTrainer
            return UnslothCounterFactGRPOTrainer
        elif algorithm == "gbmpo":
            from ..backends.unsloth.rl.gbmpo.gbmpo import UnslothGBMPOTrainer
            return UnslothGBMPOTrainer
        elif algorithm == "online_dpo":
            from ..backends.unsloth.rl.online_dpo.online_dpo import UnslothOnlineDPOTrainer
            return UnslothOnlineDPOTrainer
        elif algorithm == "spin":
            from ..backends.unsloth.rl.spin.spin import UnslothSPINTrainer
            return UnslothSPINTrainer
        elif algorithm == "orpo":
            from ..backends.unsloth.rl.orpo.orpo import UnslothORPOTrainer
            return UnslothORPOTrainer
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    except ImportError as e:
        raise ImportError(f"Failed to import Unsloth {training_type}/{algorithm} trainer: {e}") from e


def _lazy_import_unsloth_distill_trainer(distill_type: str):
    """Lazy import Unsloth distillation trainers to ensure proper import order."""
    global UNSLOTH_AVAILABLE
    if UNSLOTH_AVAILABLE is None:
        UNSLOTH_AVAILABLE = _check_unsloth_available()

    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "Unsloth not available. Install with: pip install unsloth\n"
            "Alternatively, use TRL backend instead: --backend trl"
        )

    import unsloth  # noqa: F401  (must import before transformers/trl)

    if distill_type == "distillation":
        from ..backends.unsloth.distill.distillation.distillation import UnslothDistillationTrainer
        return UnslothDistillationTrainer
    elif distill_type == "sdft":
        from ..backends.unsloth.distill.sdft.sdft import UnslothSDFTTrainer
        return UnslothSDFTTrainer
    else:
        raise ValueError(f"Unknown distillation type: {distill_type}")


def _unsloth_placeholder(algorithm: str, training_type: str):
    """Build a lazy-loading placeholder class for an Unsloth SFT/RL trainer."""
    class _UnslothPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer(algorithm, training_type)(config)
    return _UnslothPlaceholder


def _unsloth_distill_placeholder(distill_type: str):
    """Build a lazy-loading placeholder class for an Unsloth distillation trainer."""
    class _UnslothDistillPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_distill_trainer(distill_type)(config)
    return _UnslothDistillPlaceholder


def _lazy_import_verl_trainer(algorithm: str):
    """Lazy import veRL trainers to ensure proper import order.

    veRL is an optional dependency. This function returns None if veRL is not available.
    """
    try:
        if algorithm == "ppo":
            from ..backends.verl.rl.ppo.ppo import VerlPPOTrainer
            return VerlPPOTrainer
        elif algorithm == "grpo":
            from ..backends.verl.rl.grpo.grpo import VerlGRPOTrainer
            return VerlGRPOTrainer
        else:
            # veRL only supports PPO and GRPO in this version
            logger.warning(f"Algorithm {algorithm} not supported by veRL backend")
            return None
    except ImportError:
        # veRL not available
        return None


class TrainingType(Enum):
    """Training types supported by AlignTune."""

    SFT = "sft"  # Supervised Fine-Tuning
    RL = "rl"    # Reinforcement Learning
    ES = "es"    # Evolutionary Strategies
    DISTILL = "distill"  # Knowledge Distillation
    RAFT = "raft"  # Retrieval Augmented Fine-Tuning (v3.10)


class BackendType(Enum):
    """Backend types for training."""
    UNSLOTH = "unsloth"  # Unsloth backend (fast, memory efficient)
    TRL = "trl"          # TRL backend (standard, reliable)
    VERL = "verl"        # veRL backend (high-throughput RLHF via HybridFlow)
    ES = "es"            # Evolution Strategies backend (gradient-free optimizer)
    REGISTER = "register"  # Register/Formality Control trainer (v3.10)
    EXTRACTION = "extraction"  # Schema-guided JSON extraction trainer (v3.10)


class RLAlgorithm(Enum):
    """RL algorithms supported."""
    DPO = "dpo"    # Direct Preference Optimization
    ONLINE_DPO = "online_dpo"  # Online Iterative DPO
    PPO = "ppo"    # Proximal Policy Optimization
    GRPO = "grpo"  # Group Relative Policy Optimization
    GSPO = "gspo"  # Generalized Scoring Proximal Objective
    # New
    COUNTERFACT_GRPO = "counterfact_grpo"
    # NEW: Add GBMPO
    GBMPO = "gbmpo"
    DRGRPO = "drgrpo"
    DAPO = "dapo"
    PACE = "pace"  # Baseline-Optimized Learning Technique
    ORPO = "orpo"  # Odds Ratio Preference Optimization
    # Self-Play Fine-Tuning (SPIN)
    SPIN = "spin"  # Self-Play Improvement through No-regret learning


# Import TaskType from your SFT config
from .sft.config import TaskType

@dataclass
class BackendConfig:
    """Configuration for backend selection with task type support."""
    training_type: TrainingType
    backend: BackendType
    algorithm: Optional[RLAlgorithm] = None  # Only for RL
    task_type: Optional[TaskType] = None  # NEW: For SFT tasks
    fallback_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.training_type == TrainingType.RL and self.algorithm is None:
            raise ValueError("RL training requires algorithm specification")
        
        if self.training_type == TrainingType.SFT and self.algorithm is not None:
            logger.warning("Algorithm specified for SFT training, ignoring")
        
        # NEW: Set default task type for SFT if not specified
        if self.training_type == TrainingType.SFT and self.task_type is None:
            logger.info("No task type specified for SFT, using default SUPERVISED_FINE_TUNING")
            self.task_type = TaskType.SFT


# Backend Management Functions
def _enable_unsloth_backend():
    """Enable Unsloth backend by clearing TRL-only mode."""
    os.environ.pop('PURE_TRL_MODE', None)
    logger.info("🦥 Unsloth backend enabled - cleared TRL-only mode")

def _disable_unsloth_backend():
    """Disable Unsloth backend by setting TRL-only mode."""
    os.environ['PURE_TRL_MODE'] = '1'
    logger.info("🚫 Unsloth backend disabled - set TRL-only mode")

def _check_backend_availability(backend_type: BackendType) -> bool:
    """Check if a specific backend is available."""
    if backend_type == BackendType.TRL:
        return TRL_AVAILABLE
    elif backend_type == BackendType.UNSLOTH:
        # Temporarily enable Unsloth to check availability
        _enable_unsloth_backend()
        try:
            return _check_unsloth_available()
        finally:
            # Restore previous state
            if os.environ.get('PURE_TRL_MODE') == '1':
                _disable_unsloth_backend()
    elif backend_type == BackendType.ES:
        return ES_AVAILABLE
    return False

def validate_backend_selection(backend: str, training_type: str = "RL") -> BackendType:
    """Validate backend selection and provide helpful error messages."""
    try:
        if hasattr(backend, 'value'):  # BackendType enum
            backend_type = backend
        else:  # string
            backend_type = BackendType(backend.lower())
    except ValueError:
        available_backends = [bt.value for bt in BackendType]
        raise ValueError(f"Invalid backend '{backend}'. Available backends: {available_backends}")
    
    # Check availability
    if not _check_backend_availability(backend_type):
        if backend_type == BackendType.TRL:
            raise ImportError(
                "TRL backend not available. Install with: pip install trl\n"
                "Alternatively, use Unsloth backend: --backend unsloth"
            )
        elif backend_type == BackendType.UNSLOTH:
            raise ImportError(
                "Unsloth backend not available. Install with: pip install unsloth\n"
                "Alternatively, use TRL backend: --backend trl"
            )
    
    return backend_type

def get_backend_status() -> Dict[str, Any]:
    """Get current backend status and environment variables."""
    return {
        "pure_trl_mode": os.environ.get('PURE_TRL_MODE', '0'),
        "trl_available": TRL_AVAILABLE,
        "unsloth_available": _check_backend_availability(BackendType.UNSLOTH),
        "es_available": _check_backend_availability(BackendType.ES),
        "current_mode": "TRL-ONLY" if os.environ.get('PURE_TRL_MODE') == '1' else "UNSLOTH-ENABLED"
    }


class BackendFactory:
    """Factory for creating training backends."""
    
    # Registry of available backends
    _backends: Dict[tuple, Type[Union[SFTTrainerBase, RLTrainerBase]]] = {}

    @classmethod
    def register_backend(
        cls,
        training_type: TrainingType,
        backend: BackendType,
        algorithm: Optional[RLAlgorithm] = None,
        trainer_class: Type[Union[SFTTrainerBase, RLTrainerBase]] = None
    ):
        """Register a backend trainer class."""
        key = (training_type, backend, algorithm)
        cls._backends[key] = trainer_class
        logger.debug(f"Registered backend: {key} -> {trainer_class.__name__}")
    
    @classmethod
    def _select_best_backend(cls, backend_config: BackendConfig) -> BackendConfig:
        """Select the best available backend with fallback."""
        # Check if requested backend is available
        if cls._is_backend_available(backend_config):
            return backend_config
        
        # Try fallback backends
        if backend_config.fallback_enabled:
            fallback_order = cls._get_fallback_order(backend_config)
            
            for fallback_backend in fallback_order:
                if cls._is_backend_available(fallback_backend):
                    logger.warning(f"Requested backend {backend_config.backend} not available, "
                                 f"falling back to {fallback_backend.backend}")
                    return fallback_backend
        
        raise RuntimeError(f"No available backend for {backend_config}")
    
    @classmethod
    def _is_backend_available(cls, backend_config: BackendConfig) -> bool:
        """Check if a backend is available."""
        key = (backend_config.training_type, backend_config.backend, backend_config.algorithm)
        trainer_class = cls._backends.get(key)
        
        if trainer_class is None:
            return False
        
        # Check if the trainer class can be instantiated
        try:
            return trainer_class.is_available()
        except Exception:
            return False
    
    @classmethod
    def _get_fallback_order(cls, backend_config: BackendConfig) -> list:
        """Get fallback order for backend selection."""
        if backend_config.training_type == TrainingType.SFT:
            # SFT fallback order: Unsloth -> TRL
            fallbacks = [
                BackendConfig(TrainingType.SFT, BackendType.UNSLOTH),
                BackendConfig(TrainingType.SFT, BackendType.TRL),
            ]
        else:  # RL
            # RL fallback order: Unsloth -> TRL
            fallbacks = [
                BackendConfig(TrainingType.RL, BackendType.UNSLOTH, backend_config.algorithm),
                BackendConfig(TrainingType.RL, BackendType.TRL, backend_config.algorithm),
            ]
        
        # Remove the original backend from fallbacks
        fallbacks = [fb for fb in fallbacks if fb.backend != backend_config.backend]
        return fallbacks
    
    @classmethod
    def list_available_backends(cls) -> Dict[str, Any]:
        """List all available backends."""
        available = {
            "SFT": [],
            "RL": {}
        }
        
        for (training_type, backend, algorithm), trainer_class in cls._backends.items():
            try:
                if trainer_class.is_available():
                    if training_type == TrainingType.SFT:
                        available["SFT"].append(backend.value)
                    else:  # RL
                        if algorithm.value not in available["RL"]:
                            available["RL"][algorithm.value] = []
                        available["RL"][algorithm.value].append(backend.value)
            except Exception:
                continue
        
        return available
    
    def is_backend_available(self, backend_type: BackendType) -> bool:
        """Check if a specific backend type is available."""
        try:
            if backend_type == BackendType.TRL:
                return TRL_AVAILABLE
            elif backend_type == BackendType.UNSLOTH:
                return UNSLOTH_AVAILABLE
            else:
                return False
        except Exception:
            return False
    
    @classmethod
    def get_recommended_backend(cls, training_type: TrainingType, algorithm: Optional[RLAlgorithm] = None) -> BackendConfig:
        """Get recommended backend for given training type and algorithm."""
        # print(cls)
        if training_type == TrainingType.SFT:
            # For SFT: Unsloth is best, then TRL
            for backend in [BackendType.UNSLOTH, BackendType.TRL]:
                config = BackendConfig(training_type, backend)
                if cls._is_backend_available(config):
                    return config
        else:  # RL
            # For RL: Unsloth+TRL is best, then TRL
            for backend in [BackendType.UNSLOTH, BackendType.TRL]:
                config = BackendConfig(training_type, backend, algorithm)
                if cls._is_backend_available(config):
                    return config
        
        raise RuntimeError(f"No available backend for {training_type} {algorithm}")
    
    
    @classmethod
    def create_trainer(cls, config, backend_config: BackendConfig) -> Union[SFTTrainerBase, RLTrainerBase]:
        """Create a trainer instance based on backend configuration."""

        # ==================== CLASSIFICATION ROUTING ====================
        # Check for classification tasks - route to ClassificationTrainer
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'task_type'):
            task_type = config.dataset.task_type

            # For classification, route to the TRL or Unsloth classification trainer
            # based on the requested backend (mirrors normal RL/SFT backend selection).
            use_unsloth_classification = (
                backend_config.backend == BackendType.UNSLOTH and _check_unsloth_available()
            )

            if task_type == TaskType.TEXT_CLASSIFICATION:
                if use_unsloth_classification:
                    from aligntune.backends.unsloth.sft.sft_classification import UnslothTextClassificationTrainer
                    logger.info(f"Using UnslothTextClassificationTrainer for {task_type.value}")
                    return UnslothTextClassificationTrainer(config)
                from aligntune.backends.trl.sft.sft_classification import TextClassificationTrainer
                logger.info(f"Using TextClassificationTrainer for {task_type.value}")
                return TextClassificationTrainer(config)

            if task_type == TaskType.TOKEN_CLASSIFICATION:
                if use_unsloth_classification:
                    from aligntune.backends.unsloth.sft.sft_token_classification import UnslothTokenClassificationTrainer
                    logger.info(f"Using UnslothTokenClassificationTrainer for {task_type.value}")
                    return UnslothTokenClassificationTrainer(config)
                from aligntune.backends.trl.sft.sft_token_classification import TokenClassificationTrainer
                logger.info(f"Using TokenClassificationTrainer for {task_type.value}")
                return TokenClassificationTrainer(config)
        # ==================== END CLASSIFICATION ROUTING ====================
        
        # Normal backend routing for non-classification tasks
        key = (backend_config.training_type, backend_config.backend, backend_config.algorithm)
        
        if key not in cls._backends:
            raise ValueError(f"No backend registered for {key}")
        
        trainer_class = cls._backends[key]
        
        # Check if backend is available
        if not trainer_class.is_available():
            if backend_config.fallback_enabled:
                # Try fallback backends
                fallback_config = cls.get_recommended_backend(backend_config.training_type, backend_config.algorithm)
                if fallback_config != backend_config:
                    logger.warning(f"Backend {backend_config.backend} not available, falling back to {fallback_config.backend}")
                    return cls.create_trainer(config, fallback_config)
            raise RuntimeError(f"Backend {backend_config.backend} is not available")
        
        return trainer_class(config)


# Register all available backends
def _register_backends():
    """Register all available backend trainers."""
    logger.debug(f"_register_backends() called, TRL_AVAILABLE = {TRL_AVAILABLE}")

    # TRL Backends
    if TRL_AVAILABLE:
        BackendFactory.register_backend(TrainingType.SFT, BackendType.TRL, None, TRLSFTTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DPO, TRLDPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.ONLINE_DPO, TRLOnlineDPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.PPO, TRLPPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GRPO, TRLGRPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GSPO, TRLGSPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.COUNTERFACT_GRPO, TRLCounterFactGRPOTrainer)  # NEW
        # NEW: Register GBMPO 
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GBMPO, TRLGBMPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DRGRPO, TRLDRGRPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DAPO, TRLDAPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.PACE, TRLPaceTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.ORPO, TRLORPOTrainer)
        # Self-Play Fine-Tuning (SPIN)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.SPIN, TRLSPINTrainer)
        # Knowledge Distillation (optional - only if available)
        if DISTILL_CONFIG_AVAILABLE and TRL_DISTILL_AVAILABLE:
            BackendFactory.register_backend(TrainingType.DISTILL, BackendType.TRL, DistillationType.STANDARD.value, TRLDistillationTrainer)
            BackendFactory.register_backend(TrainingType.DISTILL, BackendType.TRL, DistillationType.SDFT.value, TRLSDFTTrainer)

            # Unsloth distillation backends (lazy registration - use placeholder classes)
            BackendFactory.register_backend(TrainingType.DISTILL, BackendType.UNSLOTH, DistillationType.STANDARD.value, _unsloth_distill_placeholder("distillation"))
            BackendFactory.register_backend(TrainingType.DISTILL, BackendType.UNSLOTH, DistillationType.SDFT.value, _unsloth_distill_placeholder("sdft"))
            logger.info("Unsloth distillation backends registered (lazy loading)")
        logger.info("TRL backends registered")
    else:
        logger.debug("Skipping TRL backend registration because TRL_AVAILABLE = False")

    # Unsloth Backends (lazy registration - use placeholder classes)
    # Always register Unsloth backends as placeholders for lazy loading
    BackendFactory.register_backend(TrainingType.SFT, BackendType.UNSLOTH, None, _unsloth_placeholder("sft", "sft"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DPO, _unsloth_placeholder("dpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.PPO, _unsloth_placeholder("ppo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GRPO, _unsloth_placeholder("grpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DRGRPO, _unsloth_placeholder("drgrpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DAPO, _unsloth_placeholder("dapo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.PACE, _unsloth_placeholder("pace", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.COUNTERFACT_GRPO, _unsloth_placeholder("counterfact_grpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GBMPO, _unsloth_placeholder("gbmpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GSPO, _unsloth_placeholder("gspo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.ONLINE_DPO, _unsloth_placeholder("online_dpo", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.SPIN, _unsloth_placeholder("spin", "rl"))
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.ORPO, _unsloth_placeholder("orpo", "rl"))
    logger.info("Unsloth backends registered (lazy loading)")

    # veRL Backends (lazy registration - optional dependency)
    # veRL only supports PPO and GRPO
    class VerlPPOPlaceholder:
        @classmethod
        def is_available(cls):
            try:
                import verl
                return True
            except ImportError:
                return False

        def __new__(cls, config):
            trainer_class = _lazy_import_verl_trainer("ppo")
            if trainer_class is None:
                raise ImportError("veRL not available or does not support PPO")
            return trainer_class(config)

    class VerlGRPOPlaceholder:
        @classmethod
        def is_available(cls):
            try:
                import verl
                return True
            except ImportError:
                return False

        def __new__(cls, config):
            trainer_class = _lazy_import_verl_trainer("grpo")
            if trainer_class is None:
                raise ImportError("veRL not available or does not support GRPO")
            return trainer_class(config)

    # Register veRL backends (always as placeholders for lazy loading)
    BackendFactory.register_backend(TrainingType.RL, BackendType.VERL, RLAlgorithm.PPO, VerlPPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.VERL, RLAlgorithm.GRPO, VerlGRPOPlaceholder)
    logger.info("veRL backends registered (lazy loading, optional dependency)")

    # Evolution Strategies Backend
    if ES_AVAILABLE:
        class ESPlaceholder:
            @classmethod
            def is_available(cls):
                return ES_AVAILABLE

            def __new__(cls, config):
                return ESTrainerBackend(config)

        # ES backend registration
        BackendFactory.register_backend(TrainingType.ES, BackendType.ES, None, ESPlaceholder)

        logger.info("ES backend registered (Evolution Strategies optimization)")


# Register backends when module is imported
_register_backends()




# Convenience functions for easy backend selection
def create_sft_trainer(
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    backend: str = "auto",
    output_dir: str = "./output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    system_prompt: Optional[str] = None,  #
    config: Optional[Union[str, Path, Dict]] = None,
    rope_type: Optional[str] = None,
    rope_target_max_seq_length: Optional[int] = None,
    rope_factor: Optional[float] = None,
    attn_implementation: str = "auto",
    sliding_window: Optional[int] = None,
    **kwargs
) -> Union[SFTTrainerBase, RLTrainerBase]:
    """Create an SFT trainer with optional long-context configuration."""

    # Set seed globally at the start of trainer creation
    seed = kwargs.get('seed', 42)
    set_seed(seed)

    if config is not None:
        if isinstance(config, (str, Path)):
            config_dict = load_config(config)  # ← Use load_config from config_utils
        else:
            config_dict = dict(config)
        
        # Parse config to unified format
        parsed_config = parse_config_to_unified(config_dict, training_type="sft")  # ← Use the function
        
        # Merge: parsed_config as base, kwargs override
        merged = {**parsed_config, **kwargs}
        kwargs = merged
        
        # Extract required parameters
        model_name = model_name or kwargs.pop('model_name', None)
        dataset_name = dataset_name or kwargs.pop('dataset_name', None)
        backend = kwargs.pop('backend', backend)

        # Update seed if it was in config but not kwargs
        if 'seed' in kwargs:
             seed = kwargs['seed']
             set_seed(seed)

    # Validate required parameters
    if model_name is None:
        raise ValueError("model_name must be provided either as argument or in config")
    if dataset_name is None:
        raise ValueError("dataset_name must be provided either as argument or in config")

    # Explicit function arguments take precedence over values loaded from a
    # configuration file.
    resolved_rope_values = {
        "rope_type": (
            rope_type if rope_type is not None else kwargs.get("rope_type")
        ),
        "target_max_seq_length": (
            rope_target_max_seq_length
            if rope_target_max_seq_length is not None
            else kwargs.get("rope_target_max_seq_length")
        ),
        "factor": (
            rope_factor if rope_factor is not None else kwargs.get("rope_factor")
        ),
        # Advanced, strategy-specific options remain available through **kwargs
        # without expanding the factory's primary public interface.
        "rope_theta": kwargs.get("rope_theta"),
        "original_max_position_embeddings": kwargs.get(
            "rope_original_max_position_embeddings"
        ),
        "partial_rotary_factor": kwargs.get("rope_partial_rotary_factor"),
        "attention_factor": kwargs.get("rope_attention_factor"),
        "beta_fast": kwargs.get("rope_beta_fast"),
        "beta_slow": kwargs.get("rope_beta_slow"),
        "short_factor": kwargs.get("rope_short_factor"),
        "long_factor": kwargs.get("rope_long_factor"),
        "low_freq_factor": kwargs.get("rope_low_freq_factor"),
        "high_freq_factor": kwargs.get("rope_high_freq_factor"),
    }

    resolved_rope_type = resolved_rope_values["rope_type"]
    additional_rope_values = {
        key: value
        for key, value in resolved_rope_values.items()
        if key != "rope_type"
    }

    if resolved_rope_type is None:
        provided_fields = [
            key
            for key, value in additional_rope_values.items()
            if value is not None
        ]
        if provided_fields:
            raise ValueError(
                "rope_type must be provided when setting other RoPE parameters: "
                + ", ".join(provided_fields)
            )
        rope_config = None
    else:
        rope_config = RopeConfig(**resolved_rope_values)

    resolved_attn_implementation = kwargs.get(
        "attn_implementation",
        attn_implementation,
    )
    resolved_sliding_window = (
        sliding_window
        if sliding_window is not None
        else kwargs.get("sliding_window")
    )

    # Create configuration
    # Build model config kwargs, only including precision if explicitly provided
    from .sft.config import PeftConfigData
    model_config_kwargs = {
        "name_or_path": model_name,
        "tokenizer_name_or_path": kwargs.get('tokenizer_name_or_path'),
        "max_seq_length": max_seq_length,
        "quantization": kwargs.get('quantization', {}),
        "use_unsloth": kwargs.get('use_unsloth', False),
        "peft": PeftConfigData(
            enabled=kwargs.get('use_peft', kwargs.get('peft_enabled', kwargs.get('use_lora', False))),
            variant=kwargs.get('lora_variant', kwargs.get('peft_variant', 'standard')),
            rank=kwargs.get('lora_r', 16),
            alpha=kwargs.get('lora_alpha', 32),
            dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=kwargs.get('lora_target_modules', None),
            bias=kwargs.get('lora_bias', 'none'),
        ),
        "rope": rope_config,
        "attn_implementation": resolved_attn_implementation,
        "sliding_window": resolved_sliding_window,
        "s2_group_size_ratio": kwargs.get('s2_group_size_ratio', 0.25),
        "s2_min_seq_length": kwargs.get('s2_min_seq_length', 64),
        "s2_shift_ratio": kwargs.get('s2_shift_ratio', 0.5),
        "max_memory": kwargs.get('max_memory'),
        "use_gradient_checkpointing": kwargs.get('use_gradient_checkpointing', True),
        "num_labels": kwargs.get('num_labels'),  # For classification tasks
        "model_init_kwargs": kwargs.get('model_init_kwargs', {}),
        "device_map":  kwargs.get('device_map', 'auto'),
        "trust_remote_code": kwargs.get('trust_remote_code', False),
        "embedding_init_method": kwargs.get('embedding_init_method'),
        "embedding_pad_to_multiple_of": kwargs.get('embedding_pad_to_multiple_of'),
        "train_embeddings": kwargs.get('train_embeddings', False),
    }
    # Only add precision if explicitly provided (otherwise use default from ModelConfig)
    if 'precision' in kwargs and kwargs.get('precision') is not None:
        # Validate precision value
        precision_value = kwargs.get('precision')
        if isinstance(precision_value, str):
            # Convert string to PrecisionType enum
            from .sft.config import PrecisionType
            try:
                precision_value = PrecisionType(precision_value.lower())
            except ValueError:
                logger.warning(f"Invalid precision '{precision_value}', using default")
                precision_value = None
        
        if precision_value is not None:
            model_config_kwargs['precision'] = precision_value
    
    config = SFTConfig(
        model=SFTModelConfig(**model_config_kwargs),
        dataset=SFTDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split'),
            config_name=kwargs.get('config_name'),
            subset=kwargs.get('subset'),  # Support dataset config/subset (e.g., for financial_phrasebank)
            config=kwargs.get('config'),  # Alternative name for subset
            max_samples=max_samples,
            percent=kwargs.get('percent'),
            column_mapping=kwargs.get('column_mapping', {}),
            format_type=kwargs.get('format_type'),
            task_type=kwargs.get('task_type', 'sft'),
            system_prompt=system_prompt,
            # dataset_kwargs=kwargs.get('dataset_kwargs', {}),
            dataset_num_proc=kwargs.get('dataset_num_proc'),
            dataset_text_field=kwargs.get('dataset_text_field', 'text'),
            text_column=kwargs.get('text_column', kwargs.get('dataset_text_field', 'text')),  # For classification tasks
            label_column=kwargs.get('label_column', 'label'),  # For classification tasks
            # eos_token=kwargs.get('eos_token'),
            pad_token=kwargs.get('pad_token'),
            chat_template=kwargs.get('chat_template'),


            preserve_columns=kwargs.get('preserve_columns'),
            keep_columns=kwargs.get('keep_columns', False),
            processing_fn=kwargs.get('processing_fn'),
            processing_batched=kwargs.get('processing_batched', False),
            processing_fn_kwargs=kwargs.get('processing_fn_kwargs', {}),
            val_split_ratio=kwargs.get('val_split_ratio'),
            test_split_ratio=kwargs.get('test_split_ratio'),
            split_seed=kwargs.get('split_seed', 42),
            curator_schema_gate=kwargs.get('curator_schema_gate', True),
            curator_clean=kwargs.get('curator_clean', False),
            curator_dedup=kwargs.get('curator_dedup', 'none'),
            curator_use_tiktoken=kwargs.get('curator_use_tiktoken', False),
            curator_max_tokens=kwargs.get('curator_max_tokens', 1_000_000),
        ),
        train=SFTTrainingConfig(
            epochs=num_epochs,
            max_steps=kwargs.get('max_steps'),
            per_device_batch_size=batch_size,
            per_device_eval_batch_size=kwargs.get('eval_batch_size', batch_size),
            learning_rate=learning_rate,
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
            warmup_steps=kwargs.get('warmup_steps', 0),
            warmup_ratio=kwargs.get('warmup_ratio', 0.1),
            weight_decay=kwargs.get('weight_decay', 0.01),
            eval_interval=kwargs.get('eval_interval', 100),
            save_interval=kwargs.get('save_steps', kwargs.get('save_interval', 500)),
            max_grad_norm=kwargs.get('max_grad_norm', 1.0),
            fp16=kwargs.get('fp16', False),
            bf16=kwargs.get('bf16', False),
            dataloader_num_workers=kwargs.get('dataloader_num_workers', 0),
            remove_unused_columns=kwargs.get('remove_unused_columns', False),
            optimizer=kwargs.get('optimizer', kwargs.get('optim', 'adamw_torch')),
            lr_scheduler=kwargs.get('lr_scheduler', kwargs.get('lr_scheduler_type', 'cosine')),
            group_by_length=kwargs.get('group_by_length', True),
            dataloader_drop_last=kwargs.get('dataloader_drop_last', False),
            eval_accumulation_steps=kwargs.get('eval_accumulation_steps'),
            label_smoothing_factor=kwargs.get('label_smoothing_factor', 0.0),
            early_stopping_patience=kwargs.get('early_stopping_patience'),
            early_stopping_threshold=kwargs.get('early_stopping_threshold', 0.0),
            load_best_model_at_end=kwargs.get('load_best_model_at_end', True),
            metric_for_best_model=kwargs.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=kwargs.get('greater_is_better', False),
            use_trl=kwargs.get('use_trl', False),
            dataset_num_proc=kwargs.get('train_dataset_num_proc'),
            dataset_kwargs=kwargs.get('train_dataset_kwargs', {}),
            packing=kwargs.get('packing', False),
            packing_strategy=kwargs.get('packing_strategy', 'bfd'),
            eval_packing=kwargs.get('eval_packing'),
            padding_free=kwargs.get('padding_free', False),
            pad_to_multiple_of=kwargs.get('pad_to_multiple_of'),
            completion_only_loss=kwargs.get('completion_only_loss'),
            assistant_only_loss=kwargs.get('assistant_only_loss', False),
            loss_type=kwargs.get('loss_type', 'nll'),
            activation_offloading=kwargs.get('activation_offloading', False),
            use_flash_attention_2=kwargs.get('use_flash_attention_2'),
            enable_thinking=kwargs.get('enable_thinking', False),
            gradient_checkpointing=kwargs.get('gradient_checkpointing', False),
            gradient_checkpointing_kwargs=kwargs.get('gradient_checkpointing_kwargs', {"use_reentrant": False}),
            use_liger_kernel=kwargs.get('use_liger_kernel', False),
            extra_params=kwargs,
            seed=seed,
            data_seed=kwargs.get('data_seed'),
        ),
        logging=SFTLoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name'),
            loggers=kwargs.get('loggers', ["tensorboard"]),
            log_level=kwargs.get('log_level', 'INFO'),
            log_interval=kwargs.get('logging_steps', kwargs.get('log_interval', 10)),
            save_strategy=kwargs.get('save_strategy', 'steps'),
            eval_strategy=kwargs.get('eval_strategy', 'steps'),
            report_to=kwargs.get('report_to', "none"),
        ),
        evaluation=SFTEvaluationConfig(
            compute_perplexity=kwargs.get('compute_perplexity', True),
            compute_rouge=kwargs.get('compute_rouge', True),
            compute_bleu=kwargs.get('compute_bleu', True),
            compute_meteor=kwargs.get('compute_meteor', False),
            compute_bertscore=kwargs.get('compute_bertscore', False),
            compute_semantic_similarity=kwargs.get('compute_semantic_similarity', False),
            compute_codebleu=kwargs.get('compute_codebleu', False),
            max_samples_for_quality_metrics=kwargs.get('max_samples_for_quality_metrics', 50),
            bertscore_model=kwargs.get('bertscore_model', 'microsoft/deberta-xlarge-mnli'),
            semantic_similarity_model=kwargs.get('semantic_similarity_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
    )
    
    # Create backend config
    if backend == "auto":
        backend_config = BackendFactory.get_recommended_backend(TrainingType.SFT)
    else:
        if hasattr(backend, 'value'):  # BackendType enum
            backend_type = backend
        else:  # string
            backend_type = BackendType(backend.lower())
        backend_config = BackendConfig(TrainingType.SFT, backend_type)
    
    # Set PURE_TRL_MODE only when TRL backend is being used
    if backend_config.backend == BackendType.TRL:
        _disable_unsloth_backend()
    elif backend_config.backend == BackendType.VERL:
        _disable_unsloth_backend()
    else:
        # Clear PURE_TRL_MODE for other backends (especially Unsloth)
        _enable_unsloth_backend()

    return BackendFactory.create_trainer(config, backend_config)


def create_rl_trainer(
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    algorithm: Optional[str] = None,
    backend: str = "auto",
    output_dir: str = "./output",
    num_epochs: int = 3,
    max_steps: Optional[int] = -1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    reward_value_model: Optional[str] = None,
    reward_model_name: Optional[str] = None,  # NEW
    reward_model_path: Optional[str] = None,  # NEW: Local reward model path
    train_custom_reward_model: bool = False,  # NEW: Train custom reward model
    reward_training_texts: Optional[List[str]] = None,  # NEW: Training texts for custom model
    reward_functions: Optional[List[str]] = None,  # NEW: Reward functions for custom model
    reward_function_weights: Optional[List[float]] = None,  # NEW: Weights for reward functions
    reward_training_base_model: Optional[str] = None,  # NEW: Base model for custom training
    reward_training_output_dir: Optional[str] = None,  # NEW: Output dir for custom training
    reward_value_loading_type: Optional[str] = None,
    reward_model_quantization: Optional[Dict] = None,
    value_model_quantization: Optional[Dict] = None,
    reward_training: Optional[Dict[str, Any]] = None,  # NEW: Flexible reward training config
    reward_device: str = "auto",
    sample_logging: Optional[Dict[str, Any]] = None,
    # NEW: Counterfactual GRPO specific parameters
    boost_factor: float = 2.0,
    min_weight: float = 0.5,
    max_spans: int = 10,
    answer_weight: float = 1.5,
    method_name: str = "counterfactual",
    random_importance: bool = False,
    invert_importance: bool = False,
    enable_gradient_conservation: bool = True,
    weight_debug: bool = False,
    system_prompt: Optional[str] = None,  #
    # NEW: GBMPO-specific parameters
    gbmpo_divergence_type: Optional[str] = None,
    config: Optional[Union[str, Path, Dict]] = None,
    **kwargs
) -> Union[SFTTrainerBase, RLTrainerBase]:
    """Create RL trainer with specified algorithm and backend."""

    # Set seed globally at the start of trainer creation
    seed = kwargs.get('seed', 42)
    set_seed(seed)

    # NEW: Load config if provided# Load and parse config if provided
    if config is not None:
        if isinstance(config, (str, Path)):
            config_dict = load_config(config)  # ← Use load_config from config_utils
        else:
            config_dict = dict(config)
        
        # Parse config to unified format
        parsed_config = parse_config_to_unified(config_dict, training_type="rl")  # ← Use the function
        
        # Merge: parsed_config as base, kwargs override
        merged = {**parsed_config, **kwargs}
        kwargs = merged
        
        # Extract required parameters from merged config
        model_name = model_name or kwargs.pop('model_name', None)
        dataset_name = dataset_name or kwargs.pop('dataset_name', None)
        algorithm = algorithm or kwargs.pop('algorithm', None)
        backend = kwargs.pop('backend', backend)

        # Update seed if it was in config
        if 'seed' in kwargs:
             seed = kwargs['seed']
             set_seed(seed)
    
    # Validate required parameters
    if model_name is None:
        raise ValueError("model_name must be provided either as argument or in config")
    if dataset_name is None:
        raise ValueError("dataset_name must be provided either as argument or in config")
    if algorithm is None:
        raise ValueError("algorithm must be provided either as argument or in config")

    reward_device = kwargs.pop('reward_device', reward_device or "auto")
    reward_device = reward_device or "auto"
    
    sample_logging_dict: Dict[str, Any] = {}
    base_sample_logging = sample_logging or kwargs.get('sample_logging_config')
    if base_sample_logging:
        sample_logging_dict.update(base_sample_logging)
    inline_sample_logging = {
        "enabled": kwargs.get('enable_sample_logging'),
        "prompts": kwargs.get('sample_logging_prompts'),
        "interval_steps": kwargs.get('sample_logging_interval_steps'),
        "percent_of_max_steps": kwargs.get('sample_logging_percent_of_max_steps', kwargs.get('sample_logging_percent')),
        "max_new_tokens": kwargs.get('sample_logging_max_new_tokens'),
        "temperature": kwargs.get('sample_logging_temperature'),
        "top_p": kwargs.get('sample_logging_top_p'),
        "num_samples": kwargs.get('sample_logging_num_samples'),
    }
    for key, value in inline_sample_logging.items():
        if value is not None:
            sample_logging_dict[key] = value
    cleaned_sample_logging = {k: v for k, v in sample_logging_dict.items() if v is not None}
    if cleaned_sample_logging:
        sample_logging_config = SampleLoggingConfig(**cleaned_sample_logging)
    else:
        sample_logging_config = SampleLoggingConfig()
    if algorithm.lower() == "online_dpo":
        # Online-DPO must not silently reuse the policy as its reward model.
        # A reward model is optional only when callable/registry rewards are
        # explicitly supplied; otherwise Online-DPO will fail with its clear
        # missing-reward error during setup.
        needs_neural_reward = False
    else:
        needs_neural_reward = not any([
            reward_model_path,
            train_custom_reward_model,
            reward_functions,
            kwargs.get('rewards'),
        ])
    if reward_model_name is None and needs_neural_reward:
        reward_model_name = model_name

    # ============================================
    # PPO MODEL CONSISTENCY WARNING
    # ============================================
    if algorithm.lower() == "ppo":
        print("⚠️  NOTE: For optimal PPO performance, ensure policy_model, reward_model, and value_model are from the same model family")
    
    # STRICT VALIDATION: Validate reward model configuration
    reward_model_source = None
    source_count = sum([
        reward_model_name is not None,
        reward_model_path is not None,
        train_custom_reward_model
    ])

    if source_count > 1:
        raise ValueError(
            f"Specify exactly ONE reward source. Found {source_count}: "
            f"reward_model_name={reward_model_name}, "
            f"reward_model_path={reward_model_path}, "
            f"train_custom_reward_model={train_custom_reward_model}"
        )

    if train_custom_reward_model:
        # Validate ALL required fields for custom training
        if not reward_training_texts:
            raise ValueError("reward_training_texts required when train_custom_reward_model=True")
        if not reward_functions:
            raise ValueError("reward_functions required when train_custom_reward_model=True")
        if not reward_training_base_model:
            raise ValueError("reward_training_base_model required when train_custom_reward_model=True")
        if not reward_training_output_dir:
            raise ValueError("reward_training_output_dir required when train_custom_reward_model=True")
        
        # Import required classes
        from .rl.config import RewardModelTrainingConfig, RewardModelSourceConfig
        
        training_config = RewardModelTrainingConfig(
            base_model_name=reward_training_base_model,
            training_texts=reward_training_texts,
            reward_functions=reward_functions,
            output_dir=reward_training_output_dir,
            # Use kwargs for optional training params
            num_epochs=kwargs.get('reward_training_epochs', 3),
            learning_rate=kwargs.get('reward_training_lr', 1e-5),
            batch_size=kwargs.get('reward_training_batch_size', 8),
            reward_weights = reward_function_weights
        )
        reward_model_source = RewardModelSourceConfig(
            source_type="custom_trained",
            training_config=training_config
        )
    elif reward_model_name:
        from .rl.config import RewardModelSourceConfig
        reward_model_source = RewardModelSourceConfig(
            source_type="pretrained_hf",
            model_name=reward_model_name
        )
    elif reward_model_path:
        # Validate path exists
        if not Path(reward_model_path).exists():
            raise FileNotFoundError(f"reward_model_path does not exist: {reward_model_path}")
        from .rl.config import RewardModelSourceConfig
        reward_model_source = RewardModelSourceConfig(
            source_type="pretrained_local",
            model_path=reward_model_path
        )
    
    # Create reward training config if provided
    reward_training_config = None
    if reward_training:
        from .rl.config import RewardModelTrainingConfig
        reward_training_config = RewardModelTrainingConfig(**reward_training)
    
    # Construct rewards config if reward_functions provided but not in kwargs
    # rewards_config = kwargs.get('rewards', [])
    rewards_config = kwargs.get('rewards') or []
    if not rewards_config and reward_functions and not train_custom_reward_model:
        weights = reward_function_weights or [1.0] * len(reward_functions)
        if len(weights) != len(reward_functions):
            logger.warning("reward_function_weights length mismatch, using default 1.0")
            weights = [1.0] * len(reward_functions)
            
        for func_name, weight in zip(reward_functions, weights):
            rewards_config.append({
                "type": func_name,
                "weight": weight,
                "params": {}
            })
        logger.info(f"Constructed rewards config from function list: {len(rewards_config)} functions")

    # FIXED: Added required 'algo' parameter and used 'train' not 'training'
    if algorithm.lower() == "counterfact_grpo":
        # Add counterfactual-specific params to kwargs for config creation
        kwargs.update({
            'boost_factor': boost_factor,
            'min_weight': min_weight,
            'max_spans': max_spans,
            'answer_weight': answer_weight,
            'method_name': method_name,
            'random_importance': random_importance,
            'invert_importance': invert_importance,
            'enable_gradient_conservation': enable_gradient_conservation,
            'weight_debug': weight_debug,
        })
    
    # NEW: Handle GBMPO algorithms - set divergence_type based on algorithm
    # NEW: Handle GBMPO algorithm variants
    original_algorithm = algorithm.lower()
    
    # Map GBMPO variants to base algorithm and extract divergence type
    if original_algorithm.startswith("gbmpo"):
        # Extract divergence type from algorithm name if not explicitly provided
        if gbmpo_divergence_type is None:
            if original_algorithm == "gbmpo":
                # Default to l2kl if no variant specified
                gbmpo_divergence_type = "l2kl"
            else:
                # Extract from algorithm name: gbmpo_l2kl -> l2kl
                suffix = original_algorithm.replace("gbmpo_", "")
                divergence_map = {
                    "l2": "l2",
                    "l2kl": "l2kl",
                    "probl2": "prob_l2",
                    "probl2kl": "prob_l2kl"
                }
                gbmpo_divergence_type = divergence_map.get(suffix, "l2kl")
        
        # Normalize to base "gbmpo" algorithm
        algorithm = "gbmpo"
        kwargs['gbmpo_divergence_type'] = gbmpo_divergence_type
        
        logger.info(f"GBMPO variant detected: {original_algorithm} -> divergence_type={gbmpo_divergence_type}")
    
    config = UnifiedConfig(
        algo=algorithm.lower(),
        model=RLModelConfig(
            name_or_path=model_name,
            backend=backend if backend != "auto" else "trl",
            max_seq_length=max_seq_length,
            quantization=kwargs.get('quantization', {}),
            gradient_checkpointing=kwargs.get('use_gradient_checkpointing', False),
            precision=kwargs.get('precision', 'auto'),
            reward_value_model=reward_value_model or kwargs.get('reward_value_model', None),
            reward_model_name=reward_model_name,
            reward_model_source=reward_model_source,
            reward_value_loading_type=reward_value_loading_type,
            reward_model_quantization=reward_model_quantization or {},
            value_model_quantization=value_model_quantization or {},
            use_peft=kwargs.get('use_peft', kwargs.get('use_lora', True)),
            lora_r=kwargs.get('lora_r', 16),
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.05),
            lora_target_modules=kwargs.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            dora_enabled=kwargs.get('dora_enabled', False),
            rslora_enabled=kwargs.get('rslora_enabled', False),
            loftq_init=kwargs.get('loftq_init', False),
            pissa_init=kwargs.get('pissa_init', False),
            model_init_kwargs=kwargs.get('model_init_kwargs', {}),
            ref_model_init_kwargs=kwargs.get('ref_model_init_kwargs', {}),
            model_adapter_name=kwargs.get('model_adapter_name'),
            ref_adapter_name=kwargs.get('ref_adapter_name'),
            force_use_ref_model=kwargs.get('force_use_ref_model', False),
            disable_dropout=kwargs.get('disable_dropout', True),
            use_logits_to_keep=kwargs.get('use_logits_to_keep', False),
            reward_device=reward_device,
            device_map = kwargs.get('device_map', 'auto'),
            trust_remote_code=kwargs.get('trust_remote_code', False),
            
        ),
        datasets=[
            RLDatasetConfig(
                name=dataset_name,
                split=kwargs.get('split'),
                max_samples=max_samples,
                percent=kwargs.get('percent'),
                max_eval_samples=kwargs.get('max_eval_samples', None),
                field_mappings=kwargs.get('field_mappings', {}),
                column_mapping=kwargs.get('column_mapping', {}),
                weight=kwargs.get('dataset_weight', 1.0),
                dataset_num_proc=kwargs.get('dataset_num_proc'),
                pad_token=kwargs.get('pad_token'),
                label_pad_token_id=kwargs.get('label_pad_token_id', -100),
                truncation_mode=kwargs.get('truncation_mode', 'keep_end'),
                padding_free=kwargs.get('padding_free', False),
                precompute_ref_log_probs=kwargs.get('precompute_ref_log_probs', False),
                precompute_ref_batch_size=kwargs.get('precompute_ref_batch_size'),
                tools=kwargs.get('tools'),
                system_prompt=system_prompt,
                
                preserve_columns=kwargs.get('preserve_columns'),
                processing_fn=kwargs.get('processing_fn'),
                processing_batched=kwargs.get('processing_batched', False),
                processing_fn_kwargs=kwargs.get('processing_fn_kwargs', {}),
                format_type=kwargs.get('format_type'),
                keep_columns=kwargs.get('keep_columns'),
                val_split_ratio=kwargs.get('val_split_ratio'),
                test_split_ratio=kwargs.get('test_split_ratio'),
                split_seed=kwargs.get('split_seed', 42),
                curator_schema_gate=kwargs.get('curator_schema_gate', True),
                curator_clean=kwargs.get('curator_clean', False),
                curator_dedup=kwargs.get('curator_dedup', 'none'),
                curator_use_tiktoken=kwargs.get('curator_use_tiktoken', False),
                curator_max_tokens=kwargs.get('curator_max_tokens', 1_000_000),
                
                config_name=kwargs.get('config_name',None),

            )
        ],
        train=RLTrainingConfig(
            epochs=num_epochs,
            max_steps=max_steps,
            per_device_batch_size=batch_size,
            per_device_eval_batch_size= kwargs.get('eval_batch_size', batch_size),
            learning_rate=learning_rate,
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
            beta=kwargs.get('beta', 0.1),  # YAML beta -> config.train.beta
            kl_coef=kwargs.get('kl_coef', 0.1),
            num_generations=kwargs.get('num_generations', batch_size),
            cliprange=kwargs.get('cliprange', 0.2),
            max_length=max_seq_length,
            use_cache=kwargs.get('use_cache', True),
            eval_interval=kwargs.get('eval_interval', 100),
            save_interval=kwargs.get('save_interval', 100),
            save_steps=kwargs.get('save_steps', 500),
            save_total_limit=kwargs.get('save_total_limit'),
            save_strategy=kwargs.get('save_strategy', 'steps'),
            logging_steps=kwargs.get('logging_steps', 10),
            # Keep the public aliases consistent: eval_steps wins when
            # supplied, otherwise eval_interval controls evaluation cadence.
            eval_steps=kwargs.get('eval_steps', kwargs.get('eval_interval', 100)),
            seed=seed, # did not use kwargs here because it was already replaced acc previosuly
            data_seed=kwargs.get('data_seed', 47),  # Match training_script.py
            mask_truncated_completions=kwargs.get('mask_truncated_completions', False),
            rollout_batch_size=kwargs.get('rollout_batch_size', 1),
            num_ppo_epochs=kwargs.get('num_ppo_epochs'),
            temperature=kwargs.get('temperature', 0.6),
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', 0),
            max_new_tokens=kwargs.get('max_new_tokens', 64),
            repetition_penalty=kwargs.get('repetition_penalty', 1.0),
            max_grad_norm=kwargs.get('max_grad_norm', 1.0),
            whiten_rewards=kwargs.get('whiten_rewards', False),
            kl_estimator=kwargs.get('kl_estimator', 'k1'),
            vf_coef=kwargs.get('vf_coef', 0.1),
            cliprange_value=kwargs.get('cliprange_value', 0.2),
            gamma=kwargs.get('gamma', 1.0),
            lam=kwargs.get('lam', 0.95),
            response_length=kwargs.get('response_length', 128),
            stop_token=kwargs.get('stop_token', 'eos'),
            missing_eos_penalty=kwargs.get('missing_eos_penalty', 1.0),
            ds3_gather_for_generation=kwargs.get('ds3_gather_for_generation', True),
            generation_kwargs=kwargs.get('generation_kwargs', {}),
            max_prompt_length=kwargs.get('max_prompt_length', 512),
            max_target_length=kwargs.get('max_target_length'),
            max_completion_length=kwargs.get('max_completion_length', 256),
            padding_free=kwargs.get('padding_free', False),
            truncation_mode=kwargs.get('truncation_mode', 'keep_end'),
            # Keep this unset globally; DPO and GRPO-family trainers have
            # different valid loss types and apply their own defaults.
            loss_type=kwargs.get('loss_type'),
            loss_weights=kwargs.get('loss_weights'),
            f_divergence_type=kwargs.get('f_divergence_type', 'reverse_kl'),
            f_alpha_divergence_coef=kwargs.get('f_alpha_divergence_coef', 1.0),
            reference_free=kwargs.get('reference_free', False),
            label_smoothing=kwargs.get('label_smoothing', 0.0),
            use_weighting=kwargs.get('use_weighting', False),
            rpo_alpha=kwargs.get('rpo_alpha'),
            ld_alpha=kwargs.get('ld_alpha'),
            discopop_tau=kwargs.get('discopop_tau', 0.05),
            sync_ref_model=kwargs.get('sync_ref_model', False),
            ref_model_mixup_alpha=kwargs.get('ref_model_mixup_alpha', 0.6),
            ref_model_sync_steps=kwargs.get('ref_model_sync_steps', 512),
            use_liger_kernel=kwargs.get('use_liger_kernel', False),
            use_liger_loss=kwargs.get('use_liger_loss'),
            # NEW: Counterfactual GRPO specific
            boost_factor=kwargs.get('boost_factor', 2.0),
            min_weight=kwargs.get('min_weight', 0.5),
            max_spans=kwargs.get('max_spans', 10),
            answer_weight=kwargs.get('answer_weight', 1.5),
            weighting_mode=kwargs.get('weighting_mode'),
            method_name=kwargs.get('method_name', 'counterfactual'),
            random_importance=kwargs.get('random_importance', False),
            invert_importance=kwargs.get('invert_importance', False),
            enable_gradient_conservation=kwargs.get('enable_gradient_conservation', True),
            weight_debug=kwargs.get('weight_debug', False),
            scale_rewards=kwargs.get('scale_rewards', 'group'),
            enable_thinking=kwargs.get('enable_thinking', False),
            fast_inference=kwargs.get('fast_inference', False),  # Unsloth vLLM fast inference
            vllm_gpu_memory_utilization=kwargs.get('vllm_gpu_memory_utilization', 0.7),  # vLLM GPU memory (0.95 for max speed)
            gbmpo_l2_coefficient=kwargs.get('gbmpo_l2_coefficient', 0.0001),
            gbmpo_divergence_type=kwargs.get('gbmpo_divergence_type'),
            optimizer=kwargs.get('optimizer', kwargs.get('optim', 'adamw_torch')),
            lr_scheduler=kwargs.get('lr_scheduler', kwargs.get('lr_scheduler_type', 'cosine')),
            warmup_steps=kwargs.get('warmup_steps', 0),
            warmup_ratio=kwargs.get('warmup_ratio', 0.0),
            eval_strategy=kwargs.get('eval_strategy', 'no'),
            logging_strategy=kwargs.get('logging_strategy', 'steps'),
            
            # PACE curriculum/baseline parameters
            curriculum_enabled=kwargs.get('curriculum_enabled', False),
            curriculum_epsilon=kwargs.get('curriculum_epsilon', 0.05),
            curriculum_update_freq=kwargs.get('curriculum_update_freq', 10),
            baseline_enabled=kwargs.get('baseline_enabled', False),
            baseline_rho_min=kwargs.get('baseline_rho_min', 0.875),
            baseline_rho_max=kwargs.get('baseline_rho_max', 0.96),
            baseline_D_half=kwargs.get('baseline_D_half', 0.5),
            baseline_warm_start=kwargs.get('baseline_warm_start'),
            use_baseline_advantages=kwargs.get('use_baseline_advantages', False),

            # SPIN-specific parameters
            num_rounds=kwargs.get('num_rounds', 2),
            dpo_steps_per_round=kwargs.get('dpo_steps_per_round', 100),
            generation_temperature=kwargs.get('generation_temperature', 0.7),
            generation_max_length=kwargs.get('generation_max_length', 512),
            generation_batch_size=kwargs.get('generation_batch_size'),
            samples_per_round=kwargs.get('samples_per_round'),
            eval_samples=kwargs.get('eval_samples'),

            # Meta-ES specific parameters
            meta_iterations=kwargs.get('meta_iterations', 15),
            patience=kwargs.get('patience', 5),
            min_delta=kwargs.get('min_delta', 0.001),
            init_scale=kwargs.get('init_scale', 0.01),
            N=kwargs.get('N', 10),
            T=kwargs.get('T', 100),
            sigma=kwargs.get('sigma', 0.01),
            sigma_decay=kwargs.get('sigma_decay', 0.99),
            alpha=kwargs.get('alpha', 0.01),
            mirror_coefficient=kwargs.get('mirror_coefficient', 0.0001),
            debug_mode=kwargs.get('debug_mode', False),
            eval_timeout=kwargs.get('eval_timeout', 5),
            eval_max_tokens=kwargs.get('eval_max_tokens', 512),
            eval_k=kwargs.get('eval_k', 1),
            eval_temperature=kwargs.get('eval_temperature', 0.8),
            num_workers=kwargs.get('num_workers', 1),
            no_wandb=kwargs.get('no_wandb', False),
            wandb_project=kwargs.get('wandb_project', 'neural-mirror-es'),
            resume=kwargs.get('resume'),

            # DPO evaluation parameters
            dpo_eval_enabled=kwargs.get('dpo_eval_enabled', False),
            dpo_eval_max_samples=kwargs.get('dpo_eval_max_samples'),
            dpo_zero_shot_max_samples=kwargs.get('dpo_zero_shot_max_samples', 50),
            dpo_few_shot_max_samples=kwargs.get('dpo_few_shot_max_samples', 30),
            dpo_few_shot_examples_text=kwargs.get('dpo_few_shot_examples_text'),

            # Additional checkpoint parameters
            load_best_model_at_end=kwargs.get('load_best_model_at_end', False),
            metric_for_best_model=kwargs.get('metric_for_best_model'),
            greater_is_better=kwargs.get('greater_is_better', False),

            # Additional training parameters
            weight_decay=kwargs.get('weight_decay', 0.01),
            reward_weights=kwargs.get('reward_weights'),

            # GRPO/GSPO specific (if not already present)
            grpo_alpha=kwargs.get('grpo_alpha', 0.1),
            grpo_beta=kwargs.get('grpo_beta', 0.1),
            gspo_gamma=kwargs.get('gspo_gamma', 0.1),
            gspo_delta=kwargs.get('gspo_delta', 0.1),
            group_by_length=kwargs.get('group_by_length', False),
            extra_params=kwargs, 
            use_rewards_directly=kwargs.get('use_rewards_directly', None),

        ),
        logging=RLLoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name'),
            loggers=kwargs.get('loggers', ["tensorboard"]),
            sample_logging=sample_logging_config,
            report_to=kwargs.get('report_to', "none"),
        ),
        rewards=rewards_config,
        chat_template=kwargs.get('chat_template', 'auto'),
        caching={
            'root': kwargs.get('cache_dir', 'cache'),
            'enabled': kwargs.get('caching_enabled', True)
        },
        reward_training=reward_training_config
    )
    
    # Create backend config
    algorithm_enum = RLAlgorithm(algorithm.lower())

    if backend == "auto":
        backend_config = BackendFactory.get_recommended_backend(TrainingType.RL, algorithm_enum)
    else:
        if hasattr(backend, 'value'):  # BackendType enum
            backend_type = backend
        else:  # string
            backend_type = BackendType(backend.lower())
        backend_config = BackendConfig(TrainingType.RL, backend_type, algorithm_enum)

    # Sync model.max_seq_length from train.max_length (train is source of truth)
    config.model.max_seq_length = config.train.max_length

    # Normalize max_prompt_length to be strictly less than max_seq_length
    # This is required by preference-based trainers (DPO, ORPO, SimPO, KTO)
    if config.train.max_prompt_length >= config.train.max_length:
        config.train.max_prompt_length = config.train.max_length // 2
        logger.info(
            f"Adjusted max_prompt_length to {config.train.max_prompt_length} "
            f"(must be strictly less than max_seq_length={config.train.max_length})"
        )

    # Set PURE_TRL_MODE only when TRL backend is being used
    if backend_config.backend == BackendType.TRL:
        _disable_unsloth_backend()
    elif backend_config.backend == BackendType.VERL:
        _disable_unsloth_backend()
    else:
        # Clear PURE_TRL_MODE for other backends (especially Unsloth)
        _enable_unsloth_backend()

    return BackendFactory.create_trainer(config, backend_config)


def create_distill_trainer(
    student_model: str,
    teacher_model: Optional[str] = None,  # Optional for SDFT (self-distillation)
    dataset_name: str = "wikitext",
    backend: str = "trl",
    output_dir: str = "./distill_output",
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    temperature: float = 3.0,
    alpha: float = 0.5,
    loss_type: str = "kl",
    max_steps: Optional[int] = None,
    config: Optional[Union[str, Path, Dict]] = None,
    **kwargs
) -> Union[SFTTrainerBase, RLTrainerBase]:
    """Create knowledge distillation trainer with specified backend and parameters."""
    try:
        from aligntune.core.distill.config import (
            UnifiedDistillConfig,
            DistillModelConfig,
            DistillDatasetConfig,
            DistillTrainingConfig,
            DistillLoggingConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Distillation module not available. Please add the distillation branch/module. "
            f"Error: {e}"
        ) from e

    # Set seed
    seed = kwargs.get('seed', 42)
    set_seed(seed)

    # Create UnifiedDistillConfig with proper nested structures
    distill_extra_params = dict(kwargs)
    # ``loss_type`` is an explicit factory argument, so it is not present in
    # ``kwargs``. Preserve a non-default value for SDPO's config backfill.
    if loss_type != "kl":
        distill_extra_params["loss_type"] = loss_type

    config = UnifiedDistillConfig(
        model=DistillModelConfig(
            student_model=student_model,
            teacher_model=teacher_model,
            teacher_tokenizer_name_or_path=kwargs.get('teacher_tokenizer_name_or_path', None),  # GOLD cross-tokenizer
            teacher_model_kind=kwargs.get('teacher_model_kind', None),  # SDFT: "base", "live", "ema"
            teacher_use_unsloth=kwargs.get('teacher_use_unsloth', None),  # standard distillation, backend="unsloth" only; None = auto-detect
            max_seq_length=kwargs.get('max_seq_length', 512),
            use_peft=kwargs.get('use_peft', False),
            lora_r=kwargs.get('lora_r', 16),
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.05),
            precision=kwargs.get('student_dtype', 'auto'),
            quantization=kwargs.get('quantization', {}),
        ),
        dataset=DistillDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split'),
            max_samples=kwargs.get('max_samples', None),
            max_eval_samples=kwargs.get('max_eval_samples'),
            percent=kwargs.get('percent'),
            field_mappings=kwargs.get('field_mappings', {}),
            format_type=kwargs.get('format_type'),
            column_mapping=kwargs.get('column_mapping', {}),
            task_type=kwargs.get('task_type'),
            system_prompt=kwargs.get('system_prompt'),
            preserve_columns=kwargs.get('preserve_columns'),
            keep_columns=kwargs.get('keep_columns'),
            processing_fn=kwargs.get('processing_fn'),
            processing_batched=kwargs.get('processing_batched', False),
            processing_fn_kwargs=kwargs.get('processing_fn_kwargs', {}),
            config_name=kwargs.get('config_name', kwargs.get('subset')),
            val_split_ratio=kwargs.get('val_split_ratio'),
            test_split_ratio=kwargs.get('test_split_ratio'),
            split_seed=kwargs.get('split_seed', 42),
            curator_schema_gate=kwargs.get('curator_schema_gate', True),
            curator_clean=kwargs.get('curator_clean', False),
            curator_dedup=kwargs.get('curator_dedup', 'none'),
            curator_use_tiktoken=kwargs.get('curator_use_tiktoken', False),
            curator_max_tokens=kwargs.get('curator_max_tokens', 1_000_000),
            privileged_context_column=kwargs.get('privileged_context_column'),
        ),
        train=DistillTrainingConfig(
            per_device_batch_size=batch_size,
            per_device_eval_batch_size=kwargs.get('eval_batch_size', batch_size),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
            epochs=num_epochs if max_steps is None else None,
            max_steps=max_steps,
            learning_rate=learning_rate,
            weight_decay=kwargs.get('weight_decay', 0.01),
            warmup_ratio=kwargs.get('warmup_ratio', 0.1),
            temperature=temperature,
            alpha=alpha,
            on_policy=kwargs.get('on_policy', False),
            lmbda=kwargs.get('lmbda'),
            beta=kwargs.get('beta', 1.0),
            distillation_mode=kwargs.get('distillation_mode'),
            distillation_alpha=kwargs.get('distillation_alpha', 0.5),
            use_uld_loss=kwargs.get('use_uld_loss', False),  # GOLD cross-tokenizer
            uld_use_hybrid_loss=kwargs.get('uld_use_hybrid_loss', False),
            logging_steps=kwargs.get('logging_steps', kwargs.get('log_interval', 10)),
            save_steps=kwargs.get('save_steps', kwargs.get('save_interval', 100)),
            eval_steps=kwargs.get('eval_steps', kwargs.get('eval_interval', 100)),
            # SDFT generation parameters
            num_generations=kwargs.get('num_generations', 8),
            max_completion_length=kwargs.get('max_completion_length', 256),
            use_successful_as_teacher=kwargs.get('use_successful_as_teacher', True),
            include_environment_feedback=kwargs.get('include_environment_feedback', False),
            extra_params=distill_extra_params,
        ),
        logging=DistillLoggingConfig(
            output_dir=output_dir,
            loggers=kwargs.get('loggers', ['tensorboard']),
            log_level=kwargs.get('log_level', 'INFO'),
        ),
        # SDPO: teacher_model_kind + non-empty rewards is what get_distillation_type()
        # checks to select SDPO over SDFT - without forwarding this, SDPO could only
        # be reached by hand-building UnifiedDistillConfig directly.
        rewards=kwargs.get('rewards', []),
    )

    # Detect distillation type from config
    distill_type = config.get_distillation_type()

    # DEBUG
    logger.info(f"[create_distill_trainer] Detected distillation type: {distill_type.value}")
    logger.info(f"[create_distill_trainer] teacher_model_kind: {config.model.teacher_model_kind}")
    logger.info(f"[create_distill_trainer] use_uld_loss: {config.train.use_uld_loss}")

    # Create backend config
    if backend == "auto":
        backend_config = BackendFactory.get_recommended_backend(TrainingType.DISTILL)
    else:
        if hasattr(backend, 'value'):  # BackendType enum
            backend_type = backend
        else:  # string
            backend_type = BackendType(backend.lower())
        backend_config = BackendConfig(TrainingType.DISTILL, backend_type, algorithm=distill_type.value)

    logger.info(f"[create_distill_trainer] Backend config algorithm: {backend_config.algorithm}")

    # Set PURE_TRL_MODE only when TRL backend is being used
    if backend_config.backend == BackendType.TRL:
        _disable_unsloth_backend()
    elif backend_config.backend == BackendType.VERL:
        _disable_unsloth_backend()
    else:
        _enable_unsloth_backend()

    return BackendFactory.create_trainer(config, backend_config)


def create_es_trainer(
    model_name: str,
    dataset_name: str,
    backend: str = "es",
    output_dir: str = "./es_output",
    max_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    max_seq_length: int = 512,
    population_size: int = 64,
    sigma: float = 0.5,
    num_iterations: int = 1000,
    reward_type: str = "math_correctness",
    rewards: Optional[List[Dict]] = None,
    use_peft: bool = True,
    use_unsloth: bool = False,
    quantization: Optional[Dict] = None,
    peft_config: Optional[Dict] = None,
    config: Optional[Union[str, Path, Dict]] = None,
    **kwargs
) -> Union[SFTTrainerBase, RLTrainerBase]:
    """Create Evolution Strategies trainer for population-based black-box optimization.

    Evolution Strategies trains by:
    1. Creating a population of perturbed LoRA adapters
    2. Generating solutions with each adapter and scoring them
    3. Computing ES gradient from fitness scores
    4. Updating base LoRA weights

    Args:
        model_name: Model to optimize (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        dataset_name: Dataset for evaluation (e.g., "openai/gsm8k")
        backend: Backend type ("es" only, currently)
        output_dir: Directory to save checkpoints
        max_steps: Maximum training steps
        batch_size: Batch size for evaluation
        learning_rate: ES learning rate (typical: 0.01-0.1)
        max_seq_length: Maximum sequence length for model
        population_size: Population size for each iteration (typical: 32-128)
        sigma: Mutation standard deviation (typical: 0.1-1.0)
        num_iterations: Number of ES iterations
        reward_type: Default reward function (e.g., "math_correctness")
        rewards: List of reward configs [{"type": "custom", "params": {"function": fn}}]
        use_peft: Enable PEFT (LoRA) adapters
        use_unsloth: Use Unsloth acceleration for the base model backbone
        quantization: Quantization config for LoRA
        peft_config: PEFT configuration
        **kwargs: Additional parameters (dtype, system_prompt, column_mapping, etc.)

    Returns:
        Initialized ES trainer ready for training
    """
    from aligntune.core.es.config import ESConfig

    # Set seed
    seed = kwargs.get('seed', 42)
    set_seed(seed)

    # Create ES config
    es_config = ESConfig(
        population_size=population_size,
        sigma=sigma,
        learning_rate=learning_rate,
        prompt_batch_size=batch_size,
        max_new_tokens=kwargs.get('max_new_tokens', 256),
        temperature=kwargs.get('temperature', 0.7),
        top_p=kwargs.get('top_p', 0.95),
        top_k=kwargs.get('top_k', -1),
        num_iterations=num_iterations,
        save_freq=kwargs.get('save_freq', 100),
        log_freq=kwargs.get('log_freq', 10),
        tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
        dtype=kwargs.get('dtype', 'auto'),
        seed=seed,
        reward_type=reward_type,
        use_unsloth=use_unsloth,
    )

    # Build PEFT config if not provided
    from aligntune.core.es.config import ESModelConfig, ESDatasetConfig, ESPeftConfig, UnifiedESConfig

    if peft_config is None and use_peft:
        peft_config_obj = ESPeftConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
        )
    elif isinstance(peft_config, dict):
        peft_config_obj = ESPeftConfig(
            rank=peft_config.get('rank', peft_config.get('r', 8)),
            alpha=peft_config.get('alpha', peft_config.get('lora_alpha', 16)),
            dropout=peft_config.get('dropout', peft_config.get('lora_dropout', 0.05)),
            target_modules=peft_config.get('target_modules'),
            peft_type=peft_config.get('peft_type', 'lora'),
            variant=peft_config.get('variant', 'standard'),
            use_dora=peft_config.get('use_dora', False),
            use_rslora=peft_config.get('use_rslora', False),
            init_lora_weights=peft_config.get('init_lora_weights', True),
            bias=peft_config.get('bias', 'none'),
        )
    else:
        peft_config_obj = peft_config

    # Create model config
    model_config = ESModelConfig(
        name_or_path=model_name,
        dtype=kwargs.get('dtype', 'auto'),
        max_seq_length=max_seq_length,
        peft=peft_config_obj,
        quantization=quantization or {},
        use_peft=use_peft,
        device_map=kwargs.get('device_map', 'auto')
    )

    # Create dataset config
    dataset_config = ESDatasetConfig(
        name=dataset_name,
        split=kwargs.get('split'),
        subset=kwargs.get('subset') or kwargs.get('config'),  # Support both 'subset' and 'config' keys
        column_mapping=kwargs.get('column_mapping', {}),
        system_prompt=kwargs.get('system_prompt'),
        task_type=kwargs.get('task_type'),
        format_type=kwargs.get('format_type'),
        keep_columns=kwargs.get('keep_columns'),
        max_samples=kwargs.get('max_samples'),
        processing_fn=kwargs.get('processing_fn'),
        processing_batched=kwargs.get('processing_batched', False),
        val_split_ratio=kwargs.get('val_split_ratio'),
        test_split_ratio=kwargs.get('test_split_ratio'),
        split_seed=kwargs.get('split_seed', 42),
        curator_schema_gate=kwargs.get('curator_schema_gate', True),
        curator_clean=kwargs.get('curator_clean', False),
        curator_dedup=kwargs.get('curator_dedup', 'none'),
        curator_use_tiktoken=kwargs.get('curator_use_tiktoken', False),
        curator_max_tokens=kwargs.get('curator_max_tokens', 1_000_000),
        loader_kwargs=kwargs.get('loader_kwargs', {})
    )

    # Create unified ES config
    config = UnifiedESConfig(
        es=es_config,
        model=model_config,
        dataset=dataset_config,
        rewards=rewards or [],
        output_dir=output_dir
    )

    parsed_config = config

    # Create backend config - ES only uses ES backend
    if backend.lower() != 'es':
        aligntune_warning(f"ES trainer only supports 'es' backend, got '{backend}'. Using 'es'.")

    backend_config = BackendConfig(TrainingType.ES, BackendType.ES, None)

    # ES uses the standard Transformers/PEFT model path. Keep Unsloth patches
    # disabled because the rollout loop manages adapters through vLLM.
    _disable_unsloth_backend()

    aligntune_success(f"Creating ES trainer: {model_name} with population_size={population_size}, sigma={sigma}")

    return BackendFactory.create_trainer(parsed_config, backend_config)


def create_raft_trainer(
    model_name: str,
    train_examples: List[Dict],
    eval_examples: Optional[List[Dict]] = None,
    output_dir: str = "./raft_output",
    backend: str = "trl",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_golden_docs: int = 3,
    max_distractor_docs: int = 5,
    use_citation_loss: bool = True,
    citation_loss_weight: float = 0.1,
    config: Optional[Union[str, Path, Dict]] = None,
    **kwargs
):
    """Create RAFT (Retrieval Augmented Fine-Tuning) trainer.

    Handles model/tokenizer loading, document-context formatting, and
    tokenization (which trl's SFTTrainer performs eagerly in __init__, before
    a RaftTrainer even exists) so callers only need to supply raw
    question/answer/document examples, mirroring create_es_trainer/
    create_distill_trainer instead of hand-writing the doc-formatting +
    TrainingArguments boilerplate the notebook previously repeated.

    Args:
        model_name: Base model to fine-tune (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        train_examples: List of dicts with keys question, answer, golden_docs,
            distractor_docs (each doc is {"title": str, "text": str})
        eval_examples: Optional list in the same format for evaluation
        output_dir: Directory to save checkpoints
        backend: "trl" (plain transformers AutoModelForCausalLM, default) or
            "unsloth" (loads the backbone via unsloth.FastLanguageModel for
            faster/lower-memory training) - mirrors the backend= switch on
            create_rl_trainer/create_sft_trainer.
        num_epochs: Number of training epochs
        batch_size: Per-device train batch size
        learning_rate: Learning rate
        max_golden_docs: Max golden documents included in context per example
        max_distractor_docs: Max distractor documents included in context per example
        use_citation_loss: Enable citation-quality tracking
        citation_loss_weight: Weight for the (currently metrics-only) citation loss
        config: Unused, kept for interface parity with other create_* functions
        **kwargs: Forwarded to transformers.TrainingArguments (e.g. max_steps,
            warmup_steps, save_strategy, eval_strategy, report_to)

    Returns:
        Initialized RaftTrainer ready for training via trainer.train()
    """
    from datasets import Dataset

    backend_type = BackendType(backend.lower()) if not hasattr(backend, 'value') else backend
    if backend_type == BackendType.UNSLOTH:
        if not _check_unsloth_available():
            raise ImportError(
                "Unsloth not available. Install with: pip install unsloth\n"
                "Alternatively, use backend='trl' instead."
            )
        import unsloth  # noqa: F401  (must import before transformers/trl)
        from unsloth import FastLanguageModel
        from aligntune.backends.unsloth.raft.raft_trainer import (
            RaftTrainerConfig,
            format_raft_example,
            unsloth_raft_trainer_from_config as raft_trainer_from_config,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from aligntune.backends.trl.raft.raft_trainer import (
            RaftTrainerConfig,
            format_raft_example,
            raft_trainer_from_config,
        )

    seed = kwargs.pop('seed', 42)
    set_seed(seed)

    raft_config = RaftTrainerConfig(
        max_golden_docs=max_golden_docs,
        max_distractor_docs=max_distractor_docs,
        use_citation_loss=use_citation_loss,
        citation_loss_weight=citation_loss_weight,
    )

    if backend_type == BackendType.UNSLOTH:
        aligntune_info(f"Loading RAFT backbone via Unsloth: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=kwargs.get('max_seq_length', 2048),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(train_examples).map(
        lambda ex: format_raft_example(ex, raft_config)
    )
    eval_dataset = (
        Dataset.from_list(eval_examples).map(lambda ex: format_raft_example(ex, raft_config))
        if eval_examples
        else None
    )

    training_config = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_golden_docs": max_golden_docs,
        "max_distractor_docs": max_distractor_docs,
        "use_citation_loss": use_citation_loss,
        "citation_loss_weight": citation_loss_weight,
        # `seed` was popped off kwargs above for the global set_seed() call --
        # it must also land on TrainingArguments.seed itself, since HF's
        # Trainer re-seeds via set_seed(self.args.seed) internally and would
        # otherwise silently reset everything back to args' own default (42).
        "seed": seed,
    }

    # raft_trainer_from_config() reads these specific keys out of `config` and
    # passes the *rest* of its **kwargs straight into TrainingArguments(...)
    # alongside them - anything in `kwargs` here that also names one of these
    # would otherwise reach TrainingArguments twice ("multiple values for
    # keyword argument"). Route recognized keys into training_config instead
    # of leaving them for blind kwargs passthrough.
    _CONFIG_DICT_KEYS = {
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "per_device_eval_batch_size", "learning_rate", "warmup_steps",
        "weight_decay", "logging_steps", "eval_strategy", "save_strategy",
        "save_steps", "gradient_accumulation_steps", "max_grad_norm",
        "max_golden_docs", "max_distractor_docs", "doc_context_template",
        "use_citation_loss", "citation_loss_weight",
    }
    extra_training_args = {}
    for k, v in kwargs.items():
        if k in _CONFIG_DICT_KEYS:
            training_config[k] = v
        else:
            extra_training_args[k] = v

    aligntune_success(f"Creating RAFT trainer: {model_name} with {len(train_dataset)} training examples")

    return raft_trainer_from_config(
        config=training_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **extra_training_args
    )


def create_tokenization_trainer(
    base_model: str,
    target_languages: List[str],
    dataset_name: str,
    output_dir: str = "./tokenization_output",
    num_new_tokens: int = 20000,
    extension_method: str = "continued_bpe",
    config: Optional[Union[str, Path, Dict]] = None,
    **kwargs
):
    """
    Create tokenization trainer for vocabulary adaptation.

    This trainer extends a base tokenizer for new languages/domains using
    continued BPE training or naive extension, with optional pruning.

    Args:
        base_model: Base model to adapt (e.g., "meta-llama/Llama-2-7b-hf")
        target_languages: List of target language codes (e.g., ["hi", "zh"])
        dataset_name: Dataset for tokenization training (e.g., "wikimedia/wikipedia")
        output_dir: Directory to save adapted tokenizer
        num_new_tokens: Number of new tokens to add
        extension_method: "continued_bpe" (recommended) or "naive_extension"
        config: Optional config file/dict
        **kwargs: Additional arguments:
            - config_name: Dataset config (e.g., "20231101.hi")
            - split: Dataset split (default: "train")
            - max_samples: Max samples to use (default: None)
            - text_column: Text column name (default: "text")
            - prune: Enable pruning (default: False)
            - pruning_ratio: Ratio to prune (default: 0.1)
            - pruning_method: "leaf_frequency", "frequency", or "last_n"
            - hub_model_id: Push to HF Hub (e.g., "username/model-name")

    Returns:
        TokenizationTrainer instance

    Examples:
        >>> # Basic Hindi adaptation
        >>> trainer = create_tokenization_trainer(
        ...     base_model="meta-llama/Llama-2-7b-hf",
        ...     target_languages=["hi"],
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.hi",
        ...     num_new_tokens=20000,
        ... )
        >>> result = trainer.train()

        >>> # With pruning
        >>> trainer = create_tokenization_trainer(
        ...     base_model="gpt2",
        ...     target_languages=["zh"],
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.zh",
        ...     num_new_tokens=10000,
        ...     prune=True,
        ...     pruning_ratio=0.1,
        ... )

        >>> # Push to HuggingFace Hub
        >>> trainer = create_tokenization_trainer(
        ...     base_model="gpt2",
        ...     target_languages=["hi"],
        ...     dataset_name="wikimedia/wikipedia",
        ...     output_dir="./gpt2-hindi",
        ...     hub_model_id="myusername/gpt2-hindi",  # Auto-push to hub
        ... )
    """
    try:
        from .tokenization.config import (
            UnifiedTokenizationConfig,
            TokenizationModelConfig,
            VocabExtensionConfig,
            TokenizationDatasetConfig,
            TokenizationLoggingConfig,
            PruningConfig,
            VocabExtensionMethod,
            PruningMethod,
        )
        from .tokenization.trainer import TokenizationTrainer
    except ImportError as e:
        raise ImportError(
            f"Tokenization module not available. Error: {e}"
        ) from e

    # Set seed
    seed = kwargs.get('seed', 42)
    set_seed(seed)

    # Convert extension method to enum
    if extension_method == "continued_bpe":
        method = VocabExtensionMethod.CONTINUED_BPE
    elif extension_method == "naive_extension":
        method = VocabExtensionMethod.NAIVE_EXTENSION
    else:
        raise ValueError(f"Unknown extension_method: {extension_method}")

    # Create pruning config
    prune = kwargs.get('prune', False)
    pruning_config = PruningConfig(
        enabled=prune,
        pruning_ratio=kwargs.get('pruning_ratio', 0.1),
        method=PruningMethod(kwargs.get('pruning_method', 'leaf_frequency')),
        eval_corpus_dataset=kwargs.get('pruning_dataset', None),
        eval_corpus_split=kwargs.get('pruning_split', 'train'),
        eval_corpus_samples=kwargs.get('pruning_samples', 100000),
    )

    # Create configuration
    tokenization_config = UnifiedTokenizationConfig(
        model=TokenizationModelConfig(
            base_model=base_model,
            new_tokens_count=num_new_tokens,
            base_tokenizer=kwargs.get('base_tokenizer', None),
            precision=kwargs.get('precision', "bf16"),
            device_map=kwargs.get('device_map', "auto"),
            trust_remote_code=kwargs.get('trust_remote_code', False),
        ),
        vocab_extension=VocabExtensionConfig(
            target_languages=target_languages,
            method=method,
        ),
        dataset=TokenizationDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            config_name=kwargs.get('config_name', None),
            max_samples=kwargs.get('max_samples', None),
            text_column=kwargs.get('text_column', 'text'),
            streaming=kwargs.get('streaming', False),
        ),
        pruning=pruning_config,
        logging=TokenizationLoggingConfig(
            output_dir=output_dir,
            save_intermediate=kwargs.get('save_intermediate', False),
            log_level=kwargs.get('log_level', 'info'),
            hub_model_id=kwargs.get('hub_model_id', None),
        ),
    )

    aligntune_info("="*80)
    aligntune_info("ALIGNTUNE - TOKENIZATION TRAINER")
    aligntune_info("="*80)
    aligntune_info(f"Base Model: {base_model}")
    aligntune_info(f"Target Languages: {target_languages}")
    aligntune_info(f"Dataset: {dataset_name}")
    aligntune_info(f"Extension Method: {extension_method}")
    aligntune_info(f"New Tokens: {num_new_tokens}")
    aligntune_info("="*80)

    # Create trainer
    trainer = TokenizationTrainer(tokenization_config)
    return trainer


def list_backends() -> Dict[str, list]:
    backends = {
        "TRL": {
            "available": TRL_AVAILABLE,
            "description": "HuggingFace Transformers Reinforcement Learning library",
            "status": "✅ Available" if TRL_AVAILABLE else "❌ Not Available"
        },
        "UNSLOTH": {
            "available": _check_backend_availability(BackendType.UNSLOTH),
            "description": "Unsloth optimized training with memory efficiency",
            "status": "✅ Available" if _check_backend_availability(BackendType.UNSLOTH) else "❌ Not Available"
        },
        "ES": {
            "available": _check_backend_availability(BackendType.ES),
            "description": "Evolution Strategies - gradient-free LoRA optimization",
            "status": "✅ Available" if _check_backend_availability(BackendType.ES) else "❌ Not Available"
        }
    }
    
    # Print availability status
    print("\n" + "="*60)
    print("ALIGNTUNE - BACKEND AVAILABILITY")
    print("="*60)
    for backend_name, info in backends.items():
        print(f"{backend_name:10s}: {info['status']}")
        print(f"             {info['description']}")
    print("="*60)
    print("Note: When TRL is selected, Unsloth is disabled to prevent interference.")
    print("      When Unsloth is selected, TRL-only mode is cleared.")
    print("="*60 + "\n")
    
    return backends
# ============================================================
# MODEL MERGING
# ============================================================

def merge_models(
    models: List[str],
    output_path: str,
    method: str = "linear",
    base_model: Optional[str] = None,
    weights: Optional[List[float]] = None,
    density: Optional[float] = None,
    epsilon: Optional[float] = None,
    t: Optional[float] = None,
    dtype: str = "bfloat16",
    global_params: Optional[Dict[str, Any]] = None,
    lora_adapters: Optional[List[Optional[str]]] = None,
    **kwargs
) -> str:
    """
    Merge multiple models or LoRA adapters using mergekit.

    Supports 3 merge methods:
    - Basic: linear
    - Task vectors: task_arithmetic
    - RL-optimized: ram

    Args:
        models: List of model paths or HuggingFace IDs to merge. If lora_adapters provided, these are base models.
        output_path: Directory where merged model will be saved.
        method: Merge method. Options:
            - "linear": Simple weighted average
            - "task_arithmetic": Basic task vector merging
            - "ram": Reinforced Agent Merging (for RL tasks)
        base_model: Base model (required for all methods except linear).
        weights: Per-model weights (defaults to equal weights).
        density: Sparsity density, unused by the supported methods (kept for API compatibility).
        epsilon: Unused by the supported methods (kept for API compatibility).
        t: Unused by the supported methods (kept for API compatibility).
        dtype: Output model dtype (bfloat16, float16, float32).
        global_params: Method-specific global parameters:
            - RAM: {"epsilon": 1e-5}
        lora_adapters: Optional list of LoRA adapter paths to merge (one per model).
        **kwargs: Additional args passed to mergekit.

    Returns:
        Path to merged model directory.

    Examples:
        # Linear merge (simple weighted average)
        >>> merge_models(
        ...     models=["model1", "model2"],
        ...     method="linear",
        ...     weights=[0.7, 0.3],
        ...     output_path="./merged"
        ... )

        # Task arithmetic merge
        >>> merge_models(
        ...     models=["finetuned_model1", "finetuned_model2"],
        ...     method="task_arithmetic",
        ...     base_model="base_pretrained_model",
        ...     weights=[0.5, 0.5],
        ...     output_path="./merged"
        ... )

        # Multi-LoRA merge via task arithmetic
        >>> merge_models(
        ...     models=["base_model", "base_model", "base_model"],
        ...     lora_adapters=["./lora_en", "./lora_hi", "./lora_zh"],
        ...     method="task_arithmetic",
        ...     base_model="base_model",
        ...     weights=[0.33, 0.33, 0.34],
        ...     output_path="./multilingual_merged"
        ... )

        # RAM merge (for RL-trained agents)
        >>> merge_models(
        ...     models=["base"] * 3,
        ...     lora_adapters=["rl_task1", "rl_task2", "rl_task3"],
        ...     method="ram",
        ...     base_model="base",
        ...     global_params={"epsilon": 1e-5},
        ...     output_path="./rl_merged"
        ... )
    """
    from .merge.mergekit_merger import MergekitMerger

    merger = MergekitMerger()
    return merger.merge(
        models=models,
        output_path=output_path,
        method=method,
        base_model=base_model,
        weights=weights,
        density=density,
        epsilon=epsilon,
        t=t,
        dtype=dtype,
        global_params=global_params,
        lora_adapters=lora_adapters,
        **kwargs
    )


def merge_models_from_yaml(
    yaml_path: str,
    output_path: str,
    **kwargs
) -> str:
    """
    Merge models using an existing mergekit YAML config file.

    This enables advanced features not supported by the basic merge_models() API:
    - Weight gradients: weight: [0.3, 0.5, 0.7, 0.9, 1.0]
    - Filters: module-specific parameters for mlp, self_attn, etc.
    - Layer ranges: layer_range: [0, 40]

    Only methods in MergekitMerger.SUPPORTED_METHODS are accepted; the YAML's
    `merge_method` is validated before mergekit is invoked.

    Args:
        yaml_path: Path to mergekit YAML config file.
        output_path: Directory where merged model will be saved.
        **kwargs: Additional args passed to mergekit.

    Returns:
        Path to merged model directory.

    Example:
        # Create advanced YAML config
        >>> yaml_config = '''
        ... models:
        ...   - model: model1
        ...     parameters:
        ...       weight: 0.5
        ...   - model: model2
        ...     parameters:
        ...       weight: 0.5
        ... merge_method: task_arithmetic
        ... base_model: base_model
        ... dtype: bfloat16
        ... '''
        >>> with open("merge_config.yaml", "w") as f:
        ...     f.write(yaml_config)
        >>> merge_models_from_yaml(
        ...     yaml_path="merge_config.yaml",
        ...     output_path="./merged"
        ... )
    """
    from .merge.mergekit_merger import MergekitMerger

    merger = MergekitMerger()
    return merger.merge_from_yaml(
        yaml_path=yaml_path,
        output_path=output_path,
        **kwargs
    )
