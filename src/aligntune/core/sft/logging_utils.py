"""
Logging system for unified SFT training.

This module provides a unified logging system that supports multiple backends
including TensorBoard and WandB.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .config import LoggingConfig

logger = logging.getLogger(__name__)


class SFTLogger:
    """Unified logger for SFT training."""
    
    def __init__(self, config: LoggingConfig):
        """Initialize logger with configuration."""
        self.config = config
        self.loggers = {}
        
        # Setup logging directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        logger.info(f"Initialized SFT logging: {list(self.loggers.keys())}")
    
    def _setup_loggers(self) -> None:
        """Setup logging backends."""
        for logger_name in self.config.loggers:
            if logger_name == "tensorboard":
                self._setup_tensorboard()
            elif logger_name == "wandb":
                self._setup_wandb()
            else:
                logger.warning(f"Unknown logger: {logger_name}")
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_dir = self.output_dir / "tensorboard"
            self.loggers["tensorboard"] = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"Initialized TensorBoard logging: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
    
    def _setup_wandb(self) -> None:
        """Setup WandB logging."""
        try:
            import wandb
            
            run_name = self.config.run_name or "sft_training"
            wandb.init(
                project="aligntune-sft",
                name=run_name,
                dir=str(self.output_dir)
            )
            self.loggers["wandb"] = wandb
            logger.info(f"Initialized WandB logging: {run_name}")
        except ImportError:
            logger.warning("WandB not available")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log metrics to all configured backends."""
        for logger_name, logger_instance in self.loggers.items():
            try:
                if logger_name == "tensorboard":
                    for key, value in metrics.items():
                        full_key = f"{prefix}{key}" if prefix else key
                        logger_instance.add_scalar(full_key, value, step)
                
                elif logger_name == "wandb":
                    wandb_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
                    wandb_metrics["step"] = step
                    logger_instance.log(wandb_metrics)
            
            except Exception as e:
                logger.warning(f"Failed to log to {logger_name}: {e}")
    
    def log_config(self, config: Any) -> None:
        """Log configuration to all configured backends."""
        for logger_name, logger_instance in self.loggers.items():
            try:
                if logger_name == "tensorboard":
                    # TensorBoard doesn't have a direct config logging method
                    # We'll log it as text
                    config_text = str(config.to_dict() if hasattr(config, 'to_dict') else config)
                    logger_instance.add_text("config", config_text, 0)
                
                elif logger_name == "wandb":
                    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
                    logger_instance.config.update(config_dict)
            
            except Exception as e:
                logger.warning(f"Failed to log config to {logger_name}: {e}")
    
    def close(self) -> None:
        """Close all loggers."""
        for logger_name, logger_instance in self.loggers.items():
            try:
                if logger_name == "tensorboard":
                    logger_instance.close()
                elif logger_name == "wandb":
                    logger_instance.finish()
            except Exception as e:
                logger.warning(f"Failed to close {logger_name}: {e}")
        
        self.loggers.clear()
