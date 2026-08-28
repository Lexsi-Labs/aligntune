"""
Enhanced logging utilities with WandB and TensorBoard support
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class LoggingManager:
    """Unified logging manager for WandB, TensorBoard, and console logging"""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_project: str = "aligntune",
        wandb_entity: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize logging manager
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for logs
            use_wandb: Whether to use WandB logging
            use_tensorboard: Whether to use TensorBoard logging
            wandb_config: Configuration for WandB
            wandb_project: WandB project name
            wandb_entity: WandB entity/team name
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup console logging
        self._setup_console_logging(log_level)
        
        # Initialize WandB
        self.wandb_run = None
        if self.use_wandb:
            self._setup_wandb(wandb_config, wandb_project, wandb_entity)
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        if self.use_tensorboard:
            self._setup_tensorboard()
        
        logger.info(f"Logging manager initialized: WandB={self.use_wandb}, TensorBoard={self.use_tensorboard}")
    
    def _setup_console_logging(self, log_level: str):
        """Setup console logging"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / f"{self.experiment_name}.log")
            ]
        )
    
    def _setup_wandb(self, config: Optional[Dict], project: str, entity: Optional[str]):
        """Setup WandB logging"""
        try:
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=self.experiment_name,
                config=config or {},
                dir=str(self.output_dir),
                resume="allow"
            )
            logger.info(f"WandB initialized: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.use_wandb = False
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            tb_log_dir = self.output_dir / "tensorboard" / self.experiment_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard initialized: {tb_log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.use_tensorboard = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to all enabled loggers"""
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")
        
        # WandB logging
        if self.use_wandb and self.wandb_run:
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")
        
        # TensorBoard logging
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step or 0)
                self.tensorboard_writer.flush()
            except Exception as e:
                logger.warning(f"TensorBoard logging failed: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        logger.info(f"Hyperparameters: {hparams}")
        
        if self.use_wandb and self.wandb_run:
            try:
                wandb.config.update(hparams)
            except Exception as e:
                logger.warning(f"WandB hyperparameter logging failed: {e}")
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                # Convert values to scalars for TensorBoard
                scalar_hparams = {}
                for k, v in hparams.items():
                    if isinstance(v, (int, float, bool)):
                        scalar_hparams[k] = v
                    elif isinstance(v, str):
                        try:
                            scalar_hparams[k] = float(v)
                        except ValueError:
                            continue
                
                if scalar_hparams:
                    self.tensorboard_writer.add_hparams(scalar_hparams, {})
            except Exception as e:
                logger.warning(f"TensorBoard hyperparameter logging failed: {e}")
    
    def log_model_graph(self, model, input_sample):
        """Log model graph to TensorBoard"""
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Model graph logging failed: {e}")
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text samples"""
        logger.info(f"{tag}: {text[:200]}...")
        
        if self.use_wandb and self.wandb_run:
            try:
                wandb.log({tag: wandb.Html(text)}, step=step)
            except Exception as e:
                logger.warning(f"WandB text logging failed: {e}")
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_text(tag, text, step or 0)
            except Exception as e:
                logger.warning(f"TensorBoard text logging failed: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model for gradient and parameter logging"""
        if self.use_wandb and self.wandb_run:
            try:
                wandb.watch(model, log_freq=log_freq, log="all")
                logger.info("Model watching enabled in WandB")
            except Exception as e:
                logger.warning(f"WandB model watching failed: {e}")
    
    def save_model_artifact(self, model_path: str, artifact_name: str, artifact_type: str = "model"):
        """Save model as WandB artifact"""
        if self.use_wandb and self.wandb_run:
            try:
                artifact = wandb.Artifact(artifact_name, type=artifact_type)
                artifact.add_dir(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Model artifact saved: {artifact_name}")
            except Exception as e:
                logger.warning(f"WandB artifact saving failed: {e}")
    
    def finish(self):
        """Clean up logging resources"""
        if self.use_wandb and self.wandb_run:
            try:
                wandb.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.warning(f"WandB finish failed: {e}")
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"TensorBoard close failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_logging_manager(
    experiment_name: str,
    logging_config: Dict[str, Any]
) -> LoggingManager:
    """Factory function to create logging manager from config"""
    return LoggingManager(
        experiment_name=experiment_name,
        output_dir=logging_config.get("output_dir", "./logs"),
        use_wandb=logging_config.get("use_wandb", False),
        use_tensorboard=logging_config.get("use_tensorboard", False),
        wandb_config=logging_config.get("wandb_config"),
        wandb_project=logging_config.get("wandb_project", "aligntune"),
        wandb_entity=logging_config.get("wandb_entity"),
        log_level=logging_config.get("log_level", "INFO")
    )


# Logging configuration helpers
def create_wandb_config(
    project: str = "aligntune",
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Create WandB configuration"""
    config = {
        "use_wandb": True,
        "wandb_project": project,
        "wandb_config": {
            "tags": tags or [],
            "notes": notes or ""
        }
    }
    if entity:
        config["wandb_entity"] = entity
    return config


def create_tensorboard_config(log_dir: str = "./logs/tensorboard") -> Dict[str, Any]:
    """Create TensorBoard configuration"""
    return {
        "use_tensorboard": True,
        "output_dir": log_dir
    }


def create_full_logging_config(
    wandb_project: str = "aligntune",
    wandb_entity: Optional[str] = None,
    tensorboard_dir: str = "./logs",
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """Create complete logging configuration"""
    return {
        "use_wandb": True,
        "use_tensorboard": True,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "output_dir": tensorboard_dir,
        "log_level": log_level,
        "wandb_config": {
            "framework": "aligntune",
        }
    }