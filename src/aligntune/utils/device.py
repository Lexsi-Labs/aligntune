"""
Device management utilities for CPU/GPU/Distributed training
"""

import torch
import logging
from typing import Dict, Any, Optional, Union, List
import psutil
import os

logger = logging.getLogger(__name__)


class DeviceManager:
    """Comprehensive device management for training"""
    
    def __init__(self):
        self.device_info = self._detect_devices()
        self.recommended_config = self._get_recommended_config()
        
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available devices and their capabilities"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "gpu_details": []
        }
        
        if info["cuda_available"]:
            for i in range(info["cuda_device_count"]):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info = {
                    "device_id": i,
                    "name": gpu_props.name,
                    "memory_gb": gpu_memory,
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multi_processor_count
                }
                info["gpu_details"].append(gpu_info)
        
        return info
    
    def _get_recommended_config(self) -> Dict[str, Any]:
        """Get recommended configuration based on available hardware"""
        config = {
            "device": "cpu",
            "precision": "fp32",
            "gradient_checkpointing": False,
            "batch_size": 8,
            "use_quantization": False,
            "multi_gpu": False,
            "dataloader_num_workers": min(4, self.device_info["cpu_count"])
        }
        
        if self.device_info["cuda_available"]:
            config["device"] = "cuda"
            
            # Get primary GPU info
            if self.device_info["gpu_details"]:
                primary_gpu = self.device_info["gpu_details"][0]
                gpu_memory = primary_gpu["memory_gb"]
                
                # Precision recommendations based on GPU
                if "T4" in primary_gpu["name"] or gpu_memory < 16:
                    config["precision"] = "fp16"
                    config["use_quantization"] = True
                    config["gradient_checkpointing"] = True
                    config["batch_size"] = 4
                elif gpu_memory >= 24:
                    config["precision"] = "bf16"
                    config["batch_size"] = 8
                else:
                    config["precision"] = "fp16"
                    config["batch_size"] = 6
                
                # distributed training setup
                if self.device_info["cuda_device_count"] > 1:
                    config["multi_gpu"] = True
                    config["batch_size"] = config["batch_size"] * self.device_info["cuda_device_count"]
        
        elif self.device_info["mps_available"]:
            config["device"] = "mps"
            config["precision"] = "fp32"  # MPS doesn't support all precisions
            config["batch_size"] = 4
        
        return config
    
    def get_device_string(self, device_preference: Optional[str] = None) -> str:
        """Get appropriate device string"""
        if device_preference == "auto" or device_preference is None:
            return self.recommended_config["device"]
        
        if device_preference == "cuda" and not self.device_info["cuda_available"]:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        
        if device_preference == "mps" and not self.device_info["mps_available"]:
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        
        return device_preference
    
    def get_precision_config(self, precision: Optional[str] = None) -> Dict[str, bool]:
        """Get precision configuration"""
        if precision == "auto" or precision is None:
            precision = self.recommended_config["precision"]
        
        precision_map = {
            "fp32": {"fp16": False, "bf16": False},
            "fp16": {"fp16": True, "bf16": False},
            "bf16": {"fp16": False, "bf16": True},
            "mixed": {"fp16": True, "bf16": False}  # Default mixed precision
        }
        
        if precision not in precision_map:
            logger.warning(f"Unknown precision {precision}, using fp32")
            precision = "fp32"
        
        return precision_map[precision]
    
    def setup_distributed_training(self) -> Dict[str, Any]:
        """Setup configuration for distributed training"""
        if not self.device_info["cuda_available"] or self.device_info["cuda_device_count"] <= 1:
            return {"use_distributed": False}
        
        return {
            "use_distributed": True,
            "world_size": self.device_info["cuda_device_count"],
            "distributed_backend": "nccl",
            "find_unused_parameters": False
        }
    
    def get_memory_efficient_config(self, model_size: str = "7b") -> Dict[str, Any]:
        """Get memory-efficient configuration based on model size"""
        # Model size estimates (in GB)
        model_memory_estimates = {
            "small": 1,    # <1B params
            "medium": 4,   # 1-3B params  
            "large": 8,    # 3-7B params
            "7b": 14,      # 7B params
            "13b": 26,     # 13B params
            "30b": 60,     # 30B+ params
            "70b": 140     # 70B+ params
        }
        
        estimated_memory = model_memory_estimates.get(model_size.lower(), 14)
        
        config = {
            "gradient_checkpointing": True,
            "use_quantization": False,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "dataloader_pin_memory": False,
            "offload_optimizer": False,
            "cpu_offload": False
        }
        
        if self.device_info["cuda_available"]:
            available_memory = self.device_info["gpu_details"][0]["memory_gb"]
            
            if estimated_memory > available_memory * 0.8:
                config["use_quantization"] = True
                config["gradient_checkpointing"] = True
                config["batch_size"] = 1
                config["gradient_accumulation_steps"] = 16
                
                if estimated_memory > available_memory * 1.5:
                    config["cpu_offload"] = True
                    config["offload_optimizer"] = True
            
            elif estimated_memory > available_memory * 0.5:
                config["gradient_checkpointing"] = True
                config["batch_size"] = 2
                config["gradient_accumulation_steps"] = 8
            
            else:
                config["gradient_checkpointing"] = False
                config["batch_size"] = 4
                config["gradient_accumulation_steps"] = 4
        
        return config
    
    def optimize_dataloader_config(self, batch_size: int) -> Dict[str, Any]:
        """Optimize dataloader configuration based on hardware"""
        cpu_count = self.device_info["cpu_count"]
        memory_gb = self.device_info["cpu_memory_gb"]
        
        # Conservative defaults
        num_workers = min(4, cpu_count // 2)
        pin_memory = False
        persistent_workers = False
        
        # Adjust based on available resources
        if memory_gb > 16 and cpu_count >= 8:
            num_workers = min(8, cpu_count - 2)
            pin_memory = self.device_info["cuda_available"]
            persistent_workers = True
        
        elif memory_gb > 8 and cpu_count >= 4:
            num_workers = min(4, cpu_count - 1)
            pin_memory = self.device_info["cuda_available"]
        
        # Adjust for large batch sizes
        if batch_size > 8:
            num_workers = max(1, num_workers // 2)
        
        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": 2 if num_workers > 0 else None
        }
    
    def print_device_info(self):
        """Print detailed device information"""
        print("🖥️  Device Information")
        print("=" * 50)
        print(f"CPU Cores: {self.device_info['cpu_count']}")
        print(f"CPU Memory: {self.device_info['cpu_memory_gb']:.1f} GB")
        print(f"CUDA Available: {self.device_info['cuda_available']}")
        
        if self.device_info["cuda_available"]:
            print(f"CUDA Devices: {self.device_info['cuda_device_count']}")
            for gpu in self.device_info["gpu_details"]:
                print(f"  GPU {gpu['device_id']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_gb']:.1f} GB")
                print(f"    Compute: {gpu['compute_capability']}")
        
        if self.device_info["mps_available"]:
            print("MPS Available: Yes")
        
        print("\n🔧 Recommended Configuration")
        print("=" * 50)
        for key, value in self.recommended_config.items():
            print(f"{key}: {value}")

def setup_device(device_config='auto'):
    """Setup computation device and optimize settings"""
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            logger.info(f"Detected GPU: {gpu_name}")
            
            # T4 specific optimizations
            if 't4' in gpu_name or 'tesla' in gpu_name:
                logger.info("Tesla T4 detected - using FP16 instead of BF16")
                return device, {'fp16': True, 'bf16': False}
            else:
                return device, {'fp16': False, 'bf16': True}
        else:
            logger.info("No GPU detected, using CPU")
            return device, {'fp16': False, 'bf16': False}


def setup_device_config(
    device: Optional[str] = None,
    precision: Optional[str] = None,
    model_size: str = "7b",
    enable_gradient_checkpointing: Optional[bool] = None,
    enable_quantization: Optional[bool] = None,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Setup comprehensive device configuration
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        precision: Precision preference ("auto", "fp32", "fp16", "bf16")
        model_size: Model size for memory optimization
        enable_gradient_checkpointing: Override gradient checkpointing
        enable_quantization: Override quantization
        batch_size: Override batch size
    
    Returns:
        Complete device configuration dictionary
    """
    manager = DeviceManager()
    
    # Base configuration
    config = {
        "device": manager.get_device_string(device),
        **manager.get_precision_config(precision),
        **manager.get_memory_efficient_config(model_size),
        **manager.setup_distributed_training()
    }
    
    # Apply overrides
    if enable_gradient_checkpointing is not None:
        config["gradient_checkpointing"] = enable_gradient_checkpointing
    
    if enable_quantization is not None:
        config["use_quantization"] = enable_quantization
    
    if batch_size is not None:
        config["batch_size"] = batch_size
        # Update dataloader config with new batch size
        config.update(manager.optimize_dataloader_config(batch_size))
    else:
        config.update(manager.optimize_dataloader_config(config["batch_size"]))
    
    return config


def get_optimal_batch_size(
    model_size: str = "7b",
    sequence_length: int = 2048,
    device: Optional[str] = None
) -> int:
    """Get optimal batch size based on model and hardware"""
    manager = DeviceManager()
    
    if device is None:
        device = manager.get_device_string("auto")
    
    if device == "cpu":
        return 1
    
    if not manager.device_info["cuda_available"]:
        return 1
    
    gpu_memory = manager.device_info["gpu_details"][0]["memory_gb"]
    
    # Rough estimates based on model size and sequence length
    memory_per_sample = {
        "small": 0.1,
        "medium": 0.3,
        "large": 0.8,
        "7b": 1.5,
        "13b": 3.0,
        "30b": 7.0,
        "70b": 15.0
    }
    
    base_memory = memory_per_sample.get(model_size.lower(), 1.5)
    sequence_factor = sequence_length / 2048  # Baseline at 2048 tokens
    estimated_memory_per_sample = base_memory * sequence_factor
    
    # Use 70% of available memory for safety
    usable_memory = gpu_memory * 0.7
    optimal_batch_size = max(1, int(usable_memory / estimated_memory_per_sample))
    
    # Cap at reasonable limits
    return min(optimal_batch_size, 16)


def check_gpu_compatibility(model_name: str) -> Dict[str, Any]:
    """Check GPU compatibility for specific models"""
    manager = DeviceManager()
    
    if not manager.device_info["cuda_available"]:
        return {
            "compatible": False,
            "reason": "No CUDA device available",
            "recommendations": ["Use CPU training", "Consider cloud GPU instances"]
        }
    
    gpu = manager.device_info["gpu_details"][0]
    compute_capability = float(gpu["compute_capability"])
    
    # Requirements for common model architectures
    requirements = {
        "llama": {"min_compute": 3.5, "min_memory": 8},
        "mistral": {"min_compute": 3.5, "min_memory": 6},
        "phi": {"min_compute": 3.5, "min_memory": 4},
        "gemma": {"min_compute": 3.5, "min_memory": 6},
        "qwen": {"min_compute": 3.5, "min_memory": 8}
    }
    
    # Detect model type
    model_type = "unknown"
    for model in requirements.keys():
        if model in model_name.lower():
            model_type = model
            break
    
    if model_type == "unknown":
        return {
            "compatible": True,
            "reason": "Unknown model, assuming compatibility",
            "recommendations": ["Monitor memory usage during training"]
        }
    
    req = requirements[model_type]
    
    if compute_capability < req["min_compute"]:
        return {
            "compatible": False,
            "reason": f"Compute capability {compute_capability} < required {req['min_compute']}",
            "recommendations": ["Upgrade GPU", "Use quantization", "Use smaller model"]
        }
    
    if gpu["memory_gb"] < req["min_memory"]:
        return {
            "compatible": True,
            "reason": f"Limited memory: {gpu['memory_gb']:.1f}GB < recommended {req['min_memory']}GB",
            "recommendations": [
                "Enable gradient checkpointing",
                "Use quantization (4-bit/8-bit)",
                "Reduce batch size",
                "Use CPU offloading"
            ]
        }
    
    return {
        "compatible": True,
        "reason": "Fully compatible",
        "recommendations": []
    }


# Global device manager instance
_device_manager = None

def get_device_manager() -> DeviceManager:
    """Get global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager