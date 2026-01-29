"""
Enhanced checkpointing utilities for training resumption and model saving.
"""

import os
import json
import pickle
import logging
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Enhanced checkpoint manager with automatic saving and loading capabilities."""
    
    def __init__(
        self, 
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_every_n_steps: int = 500,
        save_every_n_epochs: int = 1
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_every_n_steps: Save checkpoint every N steps
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints = []
        self._load_checkpoint_list()
        
        logger.info(f"CheckpointManager initialized with directory: {self.checkpoint_dir}")
    
    def _load_checkpoint_list(self):
        """Load existing checkpoint list."""
        checkpoint_list_file = self.checkpoint_dir / "checkpoint_list.json"
        if checkpoint_list_file.exists():
            try:
                with open(checkpoint_list_file, 'r') as f:
                    self.checkpoints = json.load(f)
                logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
            except Exception as e:
                logger.warning(f"Could not load checkpoint list: {e}")
                self.checkpoints = []
    
    def _save_checkpoint_list(self):
        """Save checkpoint list to file."""
        checkpoint_list_file = self.checkpoint_dir / "checkpoint_list.json"
        try:
            with open(checkpoint_list_file, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint list: {e}")
    
    def save_checkpoint(
        self,
        model: Any,
        tokenizer: Any,
        trainer_state: Dict[str, Any],
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            trainer_state: Training state dictionary
            step: Current training step
            epoch: Current epoch
            metrics: Training metrics
            config: Training configuration
            
        Returns:
            str: Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # Create checkpoint directory
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path / "model")
            tokenizer.save_pretrained(checkpoint_path / "tokenizer")
            
            # Save training state
            trainer_state_file = checkpoint_path / "trainer_state.json"
            with open(trainer_state_file, 'w') as f:
                json.dump(trainer_state, f, indent=2, default=str)
            
            # Save metrics
            if metrics:
                metrics_file = checkpoint_path / "metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            # Save config
            if config:
                config_file = checkpoint_path / "training_config.json"
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
            
            # Save checkpoint metadata
            metadata = {
                "checkpoint_name": checkpoint_name,
                "step": step,
                "epoch": epoch,
                "timestamp": timestamp,
                "model_path": str(checkpoint_path / "model"),
                "tokenizer_path": str(checkpoint_path / "tokenizer"),
                "metrics": metrics or {},
                "created_at": datetime.now().isoformat()
            }
            
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to checkpoint list
            self.checkpoints.append(metadata)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save updated checkpoint list
            self._save_checkpoint_list()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Clean up failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[Union[str, Path]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path to load
            step: Load checkpoint from specific step
            epoch: Load checkpoint from specific epoch
            
        Returns:
            Dictionary with loaded checkpoint data
        """
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
        else:
            # Find checkpoint by step or epoch
            checkpoint_path = self._find_checkpoint(step, epoch)
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load metadata
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Load training state
            trainer_state_file = checkpoint_path / "trainer_state.json"
            trainer_state = {}
            if trainer_state_file.exists():
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
            
            # Load metrics
            metrics_file = checkpoint_path / "metrics.json"
            metrics = {}
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            
            # Load config
            config_file = checkpoint_path / "training_config.json"
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            checkpoint_data = {
                "checkpoint_path": str(checkpoint_path),
                "model_path": str(checkpoint_path / "model"),
                "tokenizer_path": str(checkpoint_path / "tokenizer"),
                "metadata": metadata,
                "trainer_state": trainer_state,
                "metrics": metrics,
                "config": config
            }
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def _find_checkpoint(
        self, 
        step: Optional[int] = None, 
        epoch: Optional[int] = None
    ) -> Optional[Path]:
        """Find checkpoint by step or epoch."""
        if not self.checkpoints:
            return None
        
        if step is not None:
            # Find checkpoint with exact or closest step
            candidates = [cp for cp in self.checkpoints if cp.get("step", 0) <= step]
            if candidates:
                best_checkpoint = max(candidates, key=lambda x: x.get("step", 0))
                return Path(best_checkpoint["checkpoint_name"])
        
        if epoch is not None:
            # Find checkpoint with exact or closest epoch
            candidates = [cp for cp in self.checkpoints if cp.get("epoch", 0) <= epoch]
            if candidates:
                best_checkpoint = max(candidates, key=lambda x: x.get("epoch", 0))
                return Path(best_checkpoint["checkpoint_name"])
        
        # Return latest checkpoint
        if self.checkpoints:
            latest = max(self.checkpoints, key=lambda x: x.get("step", 0))
            return self.checkpoint_dir / latest["checkpoint_name"]
        
        return None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint path."""
        return self._find_checkpoint()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoints.copy()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step (or epoch if step is same)
        sorted_checkpoints = sorted(
            self.checkpoints, 
            key=lambda x: (x.get("step", 0), x.get("epoch", 0))
        )
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = self.checkpoint_dir / checkpoint["checkpoint_name"]
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {checkpoint['checkpoint_name']}: {e}")
        
        # Update checkpoint list
        self.checkpoints = sorted_checkpoints[-self.max_checkpoints:]
    
    def should_save_checkpoint(self, step: int, epoch: int) -> bool:
        """Check if a checkpoint should be saved at current step/epoch."""
        save_by_step = (self.save_every_n_steps > 0 and 
                       step % self.save_every_n_steps == 0)
        save_by_epoch = (self.save_every_n_epochs > 0 and 
                        epoch % self.save_every_n_epochs == 0)
        
        return save_by_step or save_by_epoch
    
    def remove_checkpoint(self, checkpoint_name: str):
        """Remove a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed checkpoint: {checkpoint_path}")
            
            # Remove from checkpoint list
            self.checkpoints = [cp for cp in self.checkpoints 
                              if cp["checkpoint_name"] != checkpoint_name]
            self._save_checkpoint_list()
            
        except Exception as e:
            logger.error(f"Error removing checkpoint {checkpoint_name}: {e}")
            raise
    
    def export_checkpoint(
        self, 
        checkpoint_name: str, 
        export_path: Union[str, Path],
        format: str = "safetensors"
    ):
        """
        Export checkpoint to a different format or location.
        
        Args:
            checkpoint_name: Name of checkpoint to export
            export_path: Export destination path
            format: Export format ('safetensors', 'pytorch', 'onnx')
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        export_path = Path(export_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            from .model_loader import ModelLoader
            
            # Load model and tokenizer
            loader = ModelLoader()
            model, tokenizer = loader.load_local_weights(
                checkpoint_path / "model",
                checkpoint_path / "tokenizer"
            )
            
            # Create export directory
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Save in target format
            if format == "safetensors":
                model.save_pretrained(export_path, safe_serialization=True)
            elif format == "pytorch":
                model.save_pretrained(export_path, safe_serialization=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Save tokenizer
            tokenizer.save_pretrained(export_path)
            
            # Copy metadata and config
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            if metadata_file.exists():
                shutil.copy2(metadata_file, export_path / "checkpoint_metadata.json")
            
            config_file = checkpoint_path / "training_config.json"
            if config_file.exists():
                shutil.copy2(config_file, export_path / "training_config.json")
            
            logger.info(f"Checkpoint exported to: {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting checkpoint: {e}")
            raise
    
    def get_checkpoint_size(self, checkpoint_name: str) -> int:
        """Get the size of a checkpoint in bytes."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            return 0
        
        total_size = 0
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information for all checkpoints."""
        total_size = 0
        checkpoint_sizes = {}
        
        for checkpoint in self.checkpoints:
            name = checkpoint["checkpoint_name"]
            size = self.get_checkpoint_size(name)
            checkpoint_sizes[name] = size
            total_size += size
        
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "checkpoint_sizes": checkpoint_sizes,
            "checkpoint_dir": str(self.checkpoint_dir)
        }


# Utility functions for backwards compatibility
def save_checkpoint(
    model, tokenizer, checkpoint_dir: str, step: int, epoch: int, **kwargs
) -> str:
    """Convenience function to save a checkpoint."""
    manager = CheckpointManager(checkpoint_dir)
    return manager.save_checkpoint(model, tokenizer, {}, step, epoch, **kwargs)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Convenience function to load a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    manager = CheckpointManager(checkpoint_path.parent)
    return manager.load_checkpoint(checkpoint_path)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Convenience function to get latest checkpoint."""
    manager = CheckpointManager(checkpoint_dir)
    latest = manager.get_latest_checkpoint()
    return str(latest) if latest else None