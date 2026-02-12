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
from .rl.trainer_base import TrainerBase
from ..utils.config_utils import parse_config_to_unified,  load_config
from ..utils.environment import set_seed  # Import seed utility

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Import backend trainers conditionally to avoid import errors
try:
    from ..backends.trl.sft.sft import TRLSFTTrainer
    from ..backends.trl.rl.dpo.dpo import TRLDPOTrainer
    from ..backends.trl.rl.ppo.ppo import TRLPPOTrainer
    from ..backends.trl.rl.grpo.grpo import TRLGRPOTrainer
    from ..backends.trl.rl.gspo.gspo import TRLGSPOTrainer
    # NEW
    from ..backends.trl.rl.counterfact_grpo.counterfact_grpo import TRLCounterFactGRPOTrainer
    from ..backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
    from ..backends.trl.rl.dr_grpo.drgrpo import TRLDRGRPOTrainer
    from ..backends.trl.rl.dapo.dapo import TRLDAPOTrainer
    from ..backends.trl.rl.pace.pace import TRLPaceTrainer
    from ..backends.trl.rl.neural_mirror_grpo.NMGrpo import TRLNeuralMirrorGRPOTrainer
    from ..backends.trl.rl.meta_es.meta_es_trainer import TRLMetaEsTrainer
    TRL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TRL backends not available: {e}")
    TRL_AVAILABLE = False

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
        # GSPO is only supported by TRL, not Unsloth
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
    elif algorithm == "nmgrpo":
        from ..backends.unsloth.rl.neural_mirror_grpo.NMGrpo import UnslothNeuralMirrorGRPOTrainer
        return UnslothNeuralMirrorGRPOTrainer
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


class TrainingType(Enum):
    """Training types supported by AlignTune."""

    SFT = "sft"  # Supervised Fine-Tuning
    RL = "rl"    # Reinforcement Learning


class BackendType(Enum):
    """Backend types for training."""
    UNSLOTH = "unsloth"  # Unsloth backend (fast, memory efficient)
    TRL = "trl"          # TRL backend (standard, reliable)


class RLAlgorithm(Enum):
    """RL algorithms supported."""
    DPO = "dpo"    # Direct Preference Optimization
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
    NMGRPO = "nmgrpo"
    METAES = "metaes"


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
            self.task_type = TaskType.SUPERVISED_FINE_TUNING


# Backend Management Functions
def _enable_unsloth_backend():
    """Enable Unsloth backend by clearing TRL-only mode."""
    os.environ.pop('PURE_TRL_MODE', None)
    os.environ.pop('TRL_ONLY_MODE', None)
    os.environ.pop('DISABLE_UNSLOTH_FOR_TRL', None)
    logger.info("🦥 Unsloth backend enabled - cleared TRL-only mode")

def _disable_unsloth_backend():
    """Disable Unsloth backend by setting TRL-only mode."""
    os.environ['PURE_TRL_MODE'] = '1'
    os.environ['TRL_ONLY_MODE'] = '1'
    os.environ['DISABLE_UNSLOTH_FOR_TRL'] = '1'
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
        "trl_only_mode": os.environ.get('TRL_ONLY_MODE', '0'),
        "disable_unsloth_for_trl": os.environ.get('DISABLE_UNSLOTH_FOR_TRL', '0'),
        "trl_available": TRL_AVAILABLE,
        "unsloth_available": _check_backend_availability(BackendType.UNSLOTH),
        "current_mode": "TRL-ONLY" if os.environ.get('PURE_TRL_MODE') == '1' else "UNSLOTH-ENABLED"
    }


class BackendFactory:
    """Factory for creating training backends."""
    
    # Registry of available backends
    _backends: Dict[tuple, Type[TrainerBase]] = {}
    
    @classmethod
    def register_backend(
        cls, 
        training_type: TrainingType, 
        backend: BackendType, 
        algorithm: Optional[RLAlgorithm] = None,
        trainer_class: Type[TrainerBase] = None
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
    
    def get_backend_capabilities(self, backend_type: BackendType) -> List[str]:
        """Get capabilities for a specific backend type."""
        capabilities = []
        
        if backend_type == BackendType.TRL and TRL_AVAILABLE:
            capabilities = ['sft', 'dpo', 'ppo', 'grpo', 'gspo']
        elif backend_type == BackendType.UNSLOTH and (UNSLOTH_AVAILABLE or _check_unsloth_available()):
            capabilities = ['sft', 'dpo', 'ppo', 'grpo', 'gspo']
        
        return capabilities
    
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
    def create_trainer(cls, config, backend_config: BackendConfig) -> TrainerBase:
        """Create a trainer instance based on backend configuration."""
        
        # ==================== CLASSIFICATION ROUTING ====================
        # Check for classification tasks - route to ClassificationTrainer
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'task_type'):
            task_type = config.dataset.task_type
            
            # For classification, use ClassificationTrainer (NOT TRL)
            if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
                try:
                    from aligntune.backends.trl.sft.Classification_trainer import ClassificationTrainer
                    logger.info(f"✓ Using ClassificationTrainer for {task_type.value}")
                    return ClassificationTrainer(config)
                except ImportError as e:
                    logger.error(f"❌ ClassificationTrainer import failed: {e}")
                    logger.error("Make sure the file exists at:")
                    logger.error("  backends/trl/sft/Classification_trainer.py")
                    raise ImportError(
                        f"ClassificationTrainer not found. "
                        f"Error: {e}"
                    )
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
    # TRL Backends
    if TRL_AVAILABLE:
        BackendFactory.register_backend(TrainingType.SFT, BackendType.TRL, None, TRLSFTTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DPO, TRLDPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.PPO, TRLPPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GRPO, TRLGRPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GSPO, TRLGSPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.COUNTERFACT_GRPO, TRLCounterFactGRPOTrainer)  # NEW
        # NEW: Register GBMPO 
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.GBMPO, TRLGBMPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DRGRPO, TRLDRGRPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.DAPO, TRLDAPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.PACE, TRLPaceTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.NMGRPO,  TRLNeuralMirrorGRPOTrainer)
        BackendFactory.register_backend(TrainingType.RL, BackendType.TRL, RLAlgorithm.METAES,  TRLMetaEsTrainer)
        logger.info("TRL backends registered")
    
    # Unsloth Backends (lazy registration - use placeholder classes)
    # Always register Unsloth backends as placeholders for lazy loading
    # Create placeholder classes for lazy loading
    class UnslothSFTPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("sft", "sft")(config)
    
    class UnslothDPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("dpo", "rl")(config)
    
    class UnslothPPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("ppo", "rl")(config)
    
    class UnslothGRPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("grpo", "rl")(config)
    class UnslothDRGRPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("drgrpo", "rl")(config)
    
    class UnslothDAPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("dapo", "rl")(config)
    
    # GSPO is only supported by TRL, not Unsloth
    # No UnslothGSPOPlaceholder - GSPO backend registration skipped for Unsloth

    class UnslothPacePlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("pace", "rl")(config)

    class UnslothCounterFactGRPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("counterfact_grpo", "rl")(config)
    
    # ADD THESE TWO:
    class UnslothGBMPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("gbmpo", "rl")(config)
    
    class UnslothNeuralMirrorGRPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("nmgrpo", "rl")(config)
            
    class UnslothGSPOPlaceholder:
        @classmethod
        def is_available(cls):
            return _check_unsloth_available()
        def __new__(cls, config):
            return _lazy_import_unsloth_trainer("gspo", "rl")(config)

    BackendFactory.register_backend(TrainingType.SFT, BackendType.UNSLOTH, None, UnslothSFTPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DPO, UnslothDPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.PPO, UnslothPPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GRPO, UnslothGRPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DRGRPO, UnslothDRGRPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.DAPO, UnslothDAPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.PACE, UnslothPacePlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.COUNTERFACT_GRPO, UnslothCounterFactGRPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GBMPO, UnslothGBMPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.NMGRPO, UnslothNeuralMirrorGRPOPlaceholder)
    BackendFactory.register_backend(TrainingType.RL, BackendType.UNSLOTH, RLAlgorithm.GSPO, UnslothGSPOPlaceholder)
    logger.info("Unsloth backends registered (lazy loading)")


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
    **kwargs
) -> TrainerBase:
    """Create SFT trainer with specified backend and parameters."""

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
    # Create configuration
    # Build model config kwargs, only including precision if explicitly provided
    model_config_kwargs = {
        "name_or_path": model_name,
        "max_seq_length": max_seq_length,
        "quantization": kwargs.get('quantization', {}),
        "use_unsloth": kwargs.get('use_unsloth', False),
        "peft_enabled": kwargs.get('use_peft', kwargs.get('peft_enabled', False)),
        "lora_rank": kwargs.get('lora_r', 16),
        "lora_alpha": kwargs.get('lora_alpha', 32),
        "lora_dropout": kwargs.get('lora_dropout', 0.1),
        "target_modules": kwargs.get('lora_target_modules', None),
        "bias": kwargs.get('lora_bias', 'none'),
        "attn_implementation": kwargs.get('attn_implementation', 'auto'),
        "max_memory": kwargs.get('max_memory'),
        "use_gradient_checkpointing": kwargs.get('use_gradient_checkpointing', True),
        "num_labels": kwargs.get('num_labels'),  # For classification tasks
        "model_init_kwargs": kwargs.get('model_init_kwargs', {}),
        "device_map":  kwargs.get('device_map', 'auto'),
        "trust_remote_code": kwargs.get('trust_remote_code', False),
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
            split=kwargs.get('split', 'train'),
            subset=kwargs.get('subset'),  # Support dataset config/subset (e.g., for financial_phrasebank)
            config=kwargs.get('config'),  # Alternative name for subset
            max_samples=max_samples,
            percent=kwargs.get('percent'),
            column_mapping=kwargs.get('column_mapping', {}),
            task_type=kwargs.get('task_type', 'supervised_fine_tuning'),
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
            processing_fn=kwargs.get('processing_fn'),
            processing_batched=kwargs.get('processing_batched', False),
            processing_fn_kwargs=kwargs.get('processing_fn_kwargs', {}),
        ),
        train=SFTTrainingConfig(
            epochs=num_epochs,
            max_steps=kwargs.get('max_steps'),
            per_device_batch_size=batch_size,
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
            optimizer=kwargs.get('optimizer', 'adamw_torch'),
            lr_scheduler=kwargs.get('lr_scheduler', 'cosine'),
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
            gradient_checkpointing=kwargs.get('gradient_checkpointing', False),
            gradient_checkpointing_kwargs=kwargs.get('gradient_checkpointing_kwargs', {"use_reentrant": False}),
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
    max_steps: Optional[int] = 100,  # Add max_steps parameter
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
) -> TrainerBase:
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
    needs_neural_reward = not any([reward_model_path, train_custom_reward_model, reward_functions])
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
            use_peft=kwargs.get('use_peft', True),
            lora_r=kwargs.get('lora_r', 16),
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.05),
            lora_target_modules=kwargs.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
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
                split=kwargs.get('split', 'train'),
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
            eval_steps=kwargs.get('eval_steps', 100),
            seed=seed, # did not use kwargs here because it was already replaced acc previosuly
            data_seed=kwargs.get('data_seed', 47),  # Match training_script.py
            mask_truncated_completions=kwargs.get('mask_truncated_completions', True),
            rollout_batch_size=kwargs.get('rollout_batch_size', 1),
            num_ppo_epochs=kwargs.get('num_ppo_epochs'),
            temperature=kwargs.get('temperature', 0.6),
            top_p=kwargs.get('top_p', 0.95),
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
            loss_type=kwargs.get('loss_type', 'sigmoid'),
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
            gbmpo_epsilon=kwargs.get('gbmpo_epsilon', 0.2),
            optimizer=kwargs.get('optimizer', 'adamw_torch'),
            lr_scheduler=kwargs.get('lr_scheduler', 'cosine'),
            warmup_steps=kwargs.get('warmup_steps', 0),
            warmup_ratio=kwargs.get('warmup_ratio', 0.0),
            eval_strategy=kwargs.get('eval_strategy', 'no'),
            logging_strategy=kwargs.get('logging_strategy', 'steps'),
            
            # BOLT-specific parameters
            curriculum_enabled=kwargs.get('curriculum_enabled', False),
            curriculum_epsilon=kwargs.get('curriculum_epsilon', 0.05),
            curriculum_update_freq=kwargs.get('curriculum_update_freq', 10),
            baseline_enabled=kwargs.get('baseline_enabled', False),
            baseline_rho_min=kwargs.get('baseline_rho_min', 0.875),
            baseline_rho_max=kwargs.get('baseline_rho_max', 0.96),
            baseline_D_half=kwargs.get('baseline_D_half', 0.5),
            baseline_warm_start=kwargs.get('baseline_warm_start'),
            use_baseline_advantages=kwargs.get('use_baseline_advantages', False),

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
            group_by_length=kwargs.get('group_by_length', True),
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
    
    # Set PURE_TRL_MODE only when TRL backend is being used
    if backend_config.backend == BackendType.TRL:
        _disable_unsloth_backend()
    else:
        # Clear PURE_TRL_MODE for other backends (especially Unsloth)
        _enable_unsloth_backend()
    
    return BackendFactory.create_trainer(config, backend_config)


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
        }
    }
    
    # Print availability status
    print("\n" + "="*60)
    print("FINETUNEHUB - BACKEND AVAILABILITY")
    print("="*60)
    for backend_name, info in backends.items():
        print(f"{backend_name:10s}: {info['status']}")
        print(f"             {info['description']}")
    print("="*60)
    print("Note: When TRL is selected, Unsloth is disabled to prevent interference.")
    print("      When Unsloth is selected, TRL-only mode is cleared.")
    print("="*60 + "\n")
    
    return backends