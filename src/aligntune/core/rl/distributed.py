"""
Distributed training backends for unified RLHF training.

This module provides support for different distributed training backends:
- Single GPU/CPU training
- DistributedDataParallel (DDP) via accelerate
- FullyShardedDataParallel (FSDP)
- DeepSpeed ZeRO-2/3
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.distributed as dist

from .config import DistributedConfig, BackendType

logger = logging.getLogger(__name__)


class DistributedBackend(ABC):
    """Abstract base class for distributed training backends."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the distributed backend."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        pass
    
    @abstractmethod
    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        pass
    
    @abstractmethod
    def broadcast_checkpoint_path(self, path: str) -> None:
        """Broadcast checkpoint path to all ranks."""
        pass
    
    def get_device(self) -> torch.device:
        """Get the device for this rank."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + self.rank)
            torch.cuda.manual_seed_all(seed + self.rank)


class SingleBackend(DistributedBackend):
    """Single GPU/CPU training backend."""
    
    def initialize(self) -> None:
        """Initialize single device training."""
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        
        # Set seed
        self.set_seed(self.config.seed)
        
        logger.info("Initialized single device training")
    
    def cleanup(self) -> None:
        """No cleanup needed for single device."""
        pass
    
    def is_rank_0(self) -> bool:
        """Always rank 0 for single device."""
        return True
    
    def broadcast_checkpoint_path(self, path: str) -> None:
        """No broadcast needed for single device."""
        pass


class DDPBackend(DistributedBackend):
    """DistributedDataParallel backend using accelerate."""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.accelerator = None
    
    def initialize(self) -> None:
        """Initialize DDP training."""
        try:
            from accelerate import Accelerator
            
            self.accelerator = Accelerator()
            self.rank = self.accelerator.process_index
            self.world_size = self.accelerator.num_processes
            self.local_rank = self.accelerator.local_process_index
            
            # Set seed
            self.set_seed(self.config.seed)
            
            logger.info(f"Initialized DDP training: rank {self.rank}/{self.world_size}")
            
        except ImportError:
            raise ImportError("Accelerate is required for DDP training. Install with: pip install accelerate")
    
    def cleanup(self) -> None:
        """Cleanup DDP resources."""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
    
    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        return self.rank == 0
    
    def broadcast_checkpoint_path(self, path: str) -> None:
        """Broadcast checkpoint path to all ranks."""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()


class FSDPBackend(DistributedBackend):
    """FullyShardedDataParallel backend."""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.fsdp_config = config.fsdp_config
    
    def initialize(self) -> None:
        """Initialize FSDP training."""
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA")
        
        # Initialize distributed
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            dist.init_process_group(backend="nccl")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set seed
        self.set_seed(self.config.seed)
        
        logger.info(f"Initialized FSDP training: rank {self.rank}/{self.world_size}")
    
    def cleanup(self) -> None:
        """Cleanup FSDP resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        return self.rank == 0
    
    def broadcast_checkpoint_path(self, path: str) -> None:
        """Broadcast checkpoint path to all ranks."""
        if dist.is_initialized():
            # Convert string to tensor for broadcasting
            path_tensor = torch.tensor([ord(c) for c in path], dtype=torch.long)
            dist.broadcast(path_tensor, src=0)
            
            # Convert back to string on other ranks
            if not self.is_rank_0():
                path = ''.join(chr(c.item()) for c in path_tensor)


class DeepSpeedBackend(DistributedBackend):
    """DeepSpeed ZeRO-2/3 backend."""
    
    def __init__(self, config: DistributedConfig):
        super().__init__(config)
        self.deepspeed_config = config.deepspeed_config
        self.deepspeed_engine = None
    
    def initialize(self) -> None:
        """Initialize DeepSpeed training."""
        try:
            import deepspeed
            
            # Initialize distributed
            if not dist.is_initialized():
                deepspeed.init_distributed()
            
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Set seed
            self.set_seed(self.config.seed)
            
            logger.info(f"Initialized DeepSpeed training: rank {self.rank}/{self.world_size}")
            
        except ImportError:
            raise ImportError("DeepSpeed is required for DeepSpeed training. Install with: pip install deepspeed")
    
    def cleanup(self) -> None:
        """Cleanup DeepSpeed resources."""
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.destroy()
        
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        return self.rank == 0
    
    def broadcast_checkpoint_path(self, path: str) -> None:
        """Broadcast checkpoint path to all ranks."""
        if dist.is_initialized():
            # Convert string to tensor for broadcasting
            path_tensor = torch.tensor([ord(c) for c in path], dtype=torch.long)
            dist.broadcast(path_tensor, src=0)
            
            # Convert back to string on other ranks
            if not self.is_rank_0():
                path = ''.join(chr(c.item()) for c in path_tensor)
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Get DeepSpeed configuration."""
        default_config = {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-5,
                    "warmup_num_steps": 100
                }
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        }
        
        # Merge with user config
        config = default_config.copy()
        config.update(self.deepspeed_config)
        
        return config


class BackendFactory:
    """Factory for creating distributed training backends."""
    
    @staticmethod
    def create(config: DistributedConfig) -> DistributedBackend:
        """Create appropriate backend based on configuration."""
        if config.backend == BackendType.SINGLE:
            return SingleBackend(config)
        elif config.backend == BackendType.DDP:
            return DDPBackend(config)
        elif config.backend == BackendType.FSDP:
            return FSDPBackend(config)
        elif config.backend == BackendType.DEEPSPEED:
            return DeepSpeedBackend(config)
        else:
            raise ValueError(f"Unknown backend type: {config.backend}")
    
    @staticmethod
    def get_available_backends() -> list:
        """Get list of available backends based on installed packages."""
        backends = [BackendType.SINGLE]
        
        try:
            import accelerate
            backends.append(BackendType.DDP)
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            backends.append(BackendType.FSDP)
            
            try:
                import deepspeed
                backends.append(BackendType.DEEPSPEED)
            except ImportError:
                pass
        
        return backends
