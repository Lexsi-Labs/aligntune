"""
Unified Precision Handler for AlignTune

This module provides consistent precision handling across all backends.
"""

import torch
import logging
from enum import Enum
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"


class PrecisionHandler:
    """Handles precision configuration consistently across all backends."""
    
    @staticmethod
    def get_precision_from_config(
        config: Any, 
        default: str = "auto"
    ) -> str:
        """
        Extract precision from config with multiple fallback strategies.
        
        Args:
            config: Config object (can be dict or object)
            default: Default precision if not found
            
        Returns:
            Precision string: "fp32", "fp16", "bf16", or "auto"
        """
        # Try multiple attribute names for flexibility
        precision_attrs = ['precision', 'dtype', 'torch_dtype', 'model_precision']
        
        for attr in precision_attrs:
            # Try as dict
            if isinstance(config, dict):
                value = config.get(attr)
            # Try as object attribute
            elif hasattr(config, attr):
                value = getattr(config, attr)
            else:
                continue
            
            if value is not None:
                # Handle enum
                if hasattr(value, 'value'):
                    return value.value.lower()
                # Handle string
                elif isinstance(value, str):
                    return value.lower()
        
        # Try nested model config
        if hasattr(config, 'model'):
            return PrecisionHandler.get_precision_from_config(config.model, default)
        
        return default.lower()
    
    @staticmethod
    def get_torch_dtype(precision: str) -> torch.dtype:
        """
        Convert precision string to torch dtype.
        
        Args:
            precision: "fp32", "fp16", "bf16", or "auto"
            
        Returns:
            torch.dtype
        """
        precision = precision.lower()
        
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        elif precision == "fp32":
            return torch.float32
        elif precision == "auto":
            # Auto-detect based on CUDA availability
            if torch.cuda.is_available():
                # Check if bf16 is supported (Ampere+ GPUs)
                if torch.cuda.is_bf16_supported():
                    logger.info("Auto-detected bf16 support, using bfloat16")
                    return torch.bfloat16
                else:
                    logger.info("bf16 not supported, using float16")
                    return torch.float16
            else:
                logger.info("CUDA not available, using float32")
                return torch.float32
        else:
            logger.warning(f"Unknown precision '{precision}', using fp32")
            return torch.float32
    
    @staticmethod
    def get_training_args_precision(precision: str) -> Dict[str, bool]:
        """
        Get training arguments for fp16/bf16 flags.
        
        Args:
            precision: "fp32", "fp16", "bf16", or "auto"
            
        Returns:
            Dict with 'fp16' and 'bf16' boolean flags
        """
        precision = precision.lower()
        
        if precision == "auto":
            # Auto-detect
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return {"fp16": False, "bf16": True}
                else:
                    return {"fp16": True, "bf16": False}
            else:
                return {"fp16": False, "bf16": False}
        
        return {
            "fp16": (precision == "fp16"),
            "bf16": (precision == "bf16")
        }
    
    @staticmethod
    def validate_precision(precision: str) -> str:
        """
        Validate precision string and provide helpful error.
        
        Args:
            precision: Precision string to validate
            
        Returns:
            Validated precision string
            
        Raises:
            ValueError: If precision is invalid
        """
        valid = ["fp32", "fp16", "bf16", "auto"]
        precision = precision.lower()
        
        if precision not in valid:
            raise ValueError(
                f"Invalid precision '{precision}'. "
                f"Valid options: {valid}"
            )
        
        # Warn if bf16 requested but not supported
        if precision == "bf16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                logger.warning(
                    "⚠️  bf16 requested but not supported by GPU. "
                    "Falling back to fp16."
                )
                return "fp16"
        
        return precision
    
    @staticmethod
    def log_precision_info(precision: str, backend: str = "unknown"):
        """Log precision configuration info."""
        dtype = PrecisionHandler.get_torch_dtype(precision)
        logger.info("=" * 60)
        logger.info(f"PRECISION CONFIGURATION - {backend.upper()} Backend")
        logger.info("=" * 60)
        logger.info(f"  Requested: {precision}")
        logger.info(f"  PyTorch dtype: {dtype}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
        logger.info("=" * 60)
    
    @staticmethod
    def get_model_load_kwargs(
        precision: str,
        device_map: Optional[Union[str, Dict]] = "auto",
        quantization_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get complete model loading kwargs with precision.
        
        Args:
            precision: Precision string
            device_map: Device map for model
            quantization_config: Optional quantization config
            
        Returns:
            Dict of kwargs for model loading
        """
        kwargs = {
            "torch_dtype": PrecisionHandler.get_torch_dtype(precision),
            "device_map": device_map or "auto",
        }
        
        if quantization_config:
            # If quantizing, dtype might be overridden by quantization
            if quantization_config.get("load_in_4bit") or quantization_config.get("load_in_8bit"):
                logger.info("Quantization enabled, precision will be handled by quantization config")
        
        return kwargs