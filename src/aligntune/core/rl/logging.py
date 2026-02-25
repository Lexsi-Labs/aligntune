"""
Unified logging system for RLHF training.

This module provides rank-0 only logging to TensorBoard and/or WandB
with throttled metrics and comprehensive environment tracking.
"""

import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("Pandas not available. WandB sample logging will be limited.")

from .config import LoggingConfig, UnifiedConfig

logger = logging.getLogger(__name__)


class UnifiedLogger:
    """Unified logging system with rank-0 only operations."""
    
    def __init__(self, config: LoggingConfig, backend=None):
        """Initialize unified logger."""
        self.config = config
        self.backend = backend
        self.is_rank_0 = backend.is_rank_0() if backend else True
        
        # Initialize loggers
        self.tensorboard_logger = None
        self.wandb_logger = None
        
        # Metrics tracking
        self.metrics_history = []
        self.last_log_time = 0
        self.log_interval = 10  # Log every 10 seconds
        
        # Environment info
        self.env_info = self._get_environment_info()
        
        if self.is_rank_0:
            self._initialize_loggers()
    
    def _initialize_loggers(self):
        """Initialize logging backends."""
        # TensorBoard
        if "tensorboard" in self.config.loggers:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                log_dir = Path(self.config.output_dir) / "tensorboard"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                self.tensorboard_logger = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"Initialized TensorBoard logging: {log_dir}")
                
            except ImportError:
                logger.warning("TensorBoard not available, skipping TensorBoard logging")
        
        # WandB
        if "wandb" in self.config.loggers:
            try:
                import wandb
                
                # Initialize WandB
                wandb.init(
                    project=self.config.run_name or "aligntune",
                    dir=self.config.output_dir,
                    config={}
                )
                
                self.wandb_logger = wandb
                logger.info("Initialized WandB logging")
                
            except ImportError:
                logger.warning("WandB not available, skipping WandB logging")
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for logging."""
        env_info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }
        
        # GPU information
        if torch.cuda.is_available():
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env_info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        
        # Check for flash attention
        try:
            import flash_attn
            env_info["flash_attention_version"] = flash_attn.__version__
        except ImportError:
            env_info["flash_attention_version"] = None
        
        return env_info
    
    def log_config(self, config: UnifiedConfig) -> None:
        """Log configuration (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        config_dict = config.to_dict()
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.add_text("config", json.dumps(config_dict, indent=2), 0)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.config.update(config_dict)
        
        # Log environment info
        if self.tensorboard_logger:
            self.tensorboard_logger.add_text("environment", json.dumps(self.env_info, indent=2), 0)
        
        if self.wandb_logger:
            self.wandb_logger.config.update(self.env_info)
        
        logger.info("Logged configuration and environment info")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log metrics with throttling (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        current_time = time.time()
        
        # Throttle logging
        if current_time - self.last_log_time < self.log_interval:
            return
        
        self.last_log_time = current_time
        
        # Add prefix to metrics
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Store in history
        self.metrics_history.append({
            "step": step,
            "timestamp": current_time,
            **prefixed_metrics
        })
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for key, value in prefixed_metrics.items():
                self.tensorboard_logger.add_scalar(key, value, step)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log(prefixed_metrics, step=step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in prefixed_metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")
    
    def log_samples(self, samples: List[Dict[str, Any]], step: int) -> None:
        """Log sample outputs (rank-0 only)."""
        if not self.is_rank_0 or not samples:
            return
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for i, sample in enumerate(samples[:5]):  # Log first 5 samples
                sample_text = json.dumps(sample, indent=2)
                self.tensorboard_logger.add_text(f"sample_{i}", sample_text, step)
        
        # Log to WandB
        if self.wandb_logger and HAS_PANDAS:
            try:
                # Get first 10 samples
                table_data = samples[:10]
                
                if table_data:
                    # Convert list of dicts to pandas DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Create WandB table from DataFrame
                    table = self.wandb_logger.Table(dataframe=df)
                    self.wandb_logger.log({"samples": table}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log samples to WandB: {e}")
    
    def log_model_info(self, model: torch.nn.Module, step: int = 0) -> None:
        """Log model information (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for key, value in model_info.items():
                self.tensorboard_logger.add_scalar(f"model/{key}", value, step)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log(model_info, step=step)
        
        logger.info(f"Model info: {model_info}")
    
    def log_memory_usage(self, step: int) -> None:
        """Log memory usage (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        memory_info = {}
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info[f"gpu_{i}_memory_allocated_mb"] = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_info[f"gpu_{i}_memory_reserved_mb"] = torch.cuda.memory_reserved(i) / (1024 * 1024)
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for key, value in memory_info.items():
                self.tensorboard_logger.add_scalar(f"memory/{key}", value, step)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log(memory_info, step=step)
    
    def log_throughput(self, step: int, tokens_per_second: float, samples_per_second: float) -> None:
        """Log throughput metrics (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        throughput_info = {
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second,
        }
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for key, value in throughput_info.items():
                self.tensorboard_logger.add_scalar(f"throughput/{key}", value, step)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log(throughput_info, step=step)
    
    def log_gradient_norm(self, step: int, grad_norm: float) -> None:
        """Log gradient norm (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar("training/gradient_norm", grad_norm, step)
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log({"training/gradient_norm": grad_norm}, step=step)
    
    def save_metrics_history(self, output_dir: Union[str, Path]) -> None:
        """Save metrics history to file (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Saved metrics history to {metrics_file}")
    
    def close(self) -> None:
        """Close all loggers (rank-0 only)."""
        if not self.is_rank_0:
            return
        
        # Close TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
            logger.info("Closed TensorBoard logger")
        
        # Close WandB
        if self.wandb_logger:
            self.wandb_logger.finish()
            logger.info("Closed WandB logger")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
