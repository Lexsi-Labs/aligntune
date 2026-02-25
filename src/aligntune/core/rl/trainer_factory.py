"""

Trainer factory for automatically selecting the appropriate trainer based on config.

"""

import logging
from typing import Any

from .config import UnifiedConfig, AlgorithmType


logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory for creating RLHF trainers - delegates to Backend Factory."""
    
    @classmethod
    def create_trainer(cls, config: UnifiedConfig) -> Any:
        """
        Create a trainer by delegating to Backend Factory.
        
        Args:
            config: Unified configuration
            
        Returns:
            Trainer instance from Backend Factory
        """
        # Import here to avoid circular imports
        from ..backend_factory import create_rl_trainer, BackendType
        
        # Extract backend from config - default to TRL for safety
        backend = BackendType.TRL
        if hasattr(config, 'backend'):
            backend = config.backend
        elif hasattr(config.model, 'backend'):
            backend = config.model.backend
        
        backend_value = backend.value if hasattr(backend, 'value') else backend
        
        # Delegate to Backend Factory
        return create_rl_trainer(
            model_name=config.model.name_or_path,
            dataset_name=config.datasets[0].name,  # Use first dataset
            algorithm=config.algo.value,
            backend=backend_value,
            output_dir=config.logging.output_dir,
            num_epochs=config.train.epochs,
            max_steps=config.train.max_steps,  # Add max_steps
            batch_size=config.train.per_device_batch_size,
            learning_rate=config.train.learning_rate,
            max_seq_length=config.model.max_seq_length,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            # Use available attributes from TrainingConfig
            eval_interval=config.train.eval_interval,
            save_interval=config.train.save_interval,
            # Add reward model configuration
            reward_model_name=config.model.reward_model_name if hasattr(config.model, 'reward_model_name') else None,
        )

def create_trainer_from_config(config: UnifiedConfig) -> Any:
    """
    Create appropriate trainer based on algorithm and backend in config.
    
    Args:
        config: Unified configuration object
        
    Returns:
        Trainer instance (TRL or Unsloth based)
        
    Raises:
        ValueError: If algorithm/backend combination is not supported
    """
    algorithm = config.algo
    backend = config.model.backend
    
    logger.info(f"Creating trainer: algorithm={algorithm.value}, backend={backend}")
    
    # Map of (algorithm, backend) -> trainer class
    trainer_map = {
        # PPO trainers
        (AlgorithmType.PPO, "trl"): ("aligntune.backends.trl.rl.ppo.ppo", "TRLPPOTrainer"),
        (AlgorithmType.PPO, "unsloth"): ("aligntune.backends.unsloth.rl.ppo.ppo", "UnslothPPOTrainer"),
        
        # DPO trainers
        (AlgorithmType.DPO, "trl"): ("aligntune.backends.trl.rl.dpo.dpo", "TRLDPOTrainer"),
        (AlgorithmType.DPO, "unsloth"): ("aligntune.backends.unsloth.rl.dpo.dpo", "UnslothDPOTrainer"),
        
        # GRPO trainers
        (AlgorithmType.GRPO, "trl"): ("aligntune.backends.trl.rl.grpo.grpo", "TRLGRPOTrainer"),
        (AlgorithmType.GRPO, "unsloth"): ("aligntune.backends.unsloth.rl.grpo.grpo", "UnslothGRPOTrainer"),
        
        # GSPO trainers
        (AlgorithmType.GSPO, "trl"): ("aligntune.backends.trl.rl.gspo.gspo", "TRLGSPOTrainer"),
        (AlgorithmType.GSPO, "unsloth"): ("aligntune.backends.unsloth.rl.gspo.gspo", "UnslothGSPOTrainer"),
        
        # Counterfactual GRPO trainers
        (AlgorithmType.COUNTERFACT_GRPO, "trl"): ("aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo", "TRLCounterFactGRPOTrainer"),
        (AlgorithmType.COUNTERFACT_GRPO, "unsloth"): ("aligntune.backends.unsloth.rl.counterfact_grpo.counterfact_grpo", "UnslothCounterFactGRPOTrainer"),
        
        # GBMPO trainers
        (AlgorithmType.GBMPO, "trl"): ("aligntune.backends.trl.rl.gbmpo.gbmpo", "TRLGBMPOTrainer"),
        (AlgorithmType.GBMPO, "unsloth"): ("aligntune.backends.unsloth.rl.gbmpo.gbmpo", "UnslothGBMPOTrainer"),
        
        # DR-GRPO trainers
        (AlgorithmType.DRGRPO, "trl"): ("aligntune.backends.trl.rl.dr_grpo.drgrpo", "TRLDRGRPOTrainer"),
        (AlgorithmType.DRGRPO, "unsloth"): ("aligntune.backends.unsloth.rl.dr_grpo.drgrpo", "UnslothDRGRPOTrainer"),
        
        # DAPO trainers
        (AlgorithmType.DAPO, "trl"): ("aligntune.backends.trl.rl.dapo.dapo", "TRLDAPOTrainer"),
        (AlgorithmType.DAPO, "unsloth"): ("aligntune.backends.unsloth.rl.dapo.dapo", "UnslothDAPOTrainer"),
        
        # BOLT trainers
        (AlgorithmType.BOLT, "trl"): ("aligntune.backends.trl.rl.bolt.bolt", "TRLBoltTrainer"),
        (AlgorithmType.BOLT, "unsloth"): ("aligntune.backends.unsloth.rl.bolt.bolt", "UnslothBoltTrainer"),
        
        # Neural Mirror GRPO trainers
        (AlgorithmType.NMGRPO, "trl"): ("aligntune.backends.trl.rl.neural_mirror_grpo.NMGrpo", "TRLNeuralMirrorGRPOTrainer"),
        (AlgorithmType.NMGRPO, "unsloth"): ("aligntune.backends.unsloth.rl.neural_mirror_grpo.NMGrpo", "UnslothNeuralMirrorGRPOTrainer"),
        
        # Meta ES trainers
        (AlgorithmType.METAES, "trl"): ("aligntune.backends.trl.rl.meta_es.meta_es_trainer", "TRLMetaEsTrainer"),
    }
    
    key = (algorithm, backend)
    if key not in trainer_map:
        available_combos = "\n".join([f"  - {algo.value} + {bknd}" for (algo, bknd) in trainer_map.keys()])
        raise ValueError(
            f"Unsupported algorithm/backend combination: {algorithm.value}/{backend}\n"
            f"Available combinations:\n{available_combos}"
        )
    
    module_path, class_name = trainer_map[key]
    
    # Dynamically import the trainer class
    try:
        import importlib
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        logger.info(f"Successfully loaded trainer: {class_name}")
        return trainer_class(config)
    except Exception as e:
        logger.error(f"Failed to load trainer {class_name} from {module_path}: {e}")
        raise ImportError(
            f"Could not load trainer {class_name} from {module_path}. "
            f"Error: {e}"
        )


__all__ = ["create_trainer_from_config"]

