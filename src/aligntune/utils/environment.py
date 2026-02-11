"""
Environment validation and compatibility checking for AlignTune.

This module provides comprehensive environment diagnostics for unsloth and dependencies,
including PyTorch/CUDA compatibility, attention backends, version validation,
and reproducibility utilities.
"""

import logging
import sys
import platform
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentInfo:
    """Container for environment information."""
    python_version: str
    platform: str
    pytorch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cuda_device_count: int = 0
    unsloth_available: bool = False
    unsloth_error: Optional[str] = None
    flash_attention_available: bool = False
    xformers_available: bool = False
    trl_available: bool = False

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across random, numpy, torch, and transformers.
    
    Args:
        seed: Integer seed value.
    """
    import random
    import numpy as np
    import torch
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Transformers
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
        
    logger.info(f"Global seed set to {seed}")

def check_pytorch_cuda_compatibility() -> Dict[str, Any]:
    """Check PyTorch and CUDA compatibility."""
    info = {
        'pytorch_available': False,
        'pytorch_version': None,
        'cuda_available': False,
        'cuda_version': None,
        'cuda_device_count': 0,
        'compatibility_issues': []
    }
    
    try:
        import torch
        info['pytorch_available'] = True
        info['pytorch_version'] = torch.__version__
        
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
            
            # Check for known compatibility issues
            pytorch_version = torch.__version__
            cuda_version = torch.version.cuda
            
            # Known problematic combinations
            if '2.8.0' in pytorch_version and cuda_version:
                info['compatibility_issues'].append(
                    f"PyTorch {pytorch_version} with CUDA {cuda_version} may have compatibility issues with Unsloth"
                )
                
    except ImportError as e:
        info['compatibility_issues'].append(f"PyTorch not available: {e}")
    
    return info

def check_unsloth_compatibility() -> Dict[str, Any]:
    """Check Unsloth installation and compatibility."""
    info = {
        'unsloth_available': False,
        'unsloth_version': None,
        'error_type': None,
        'error_message': None,
        'suggestions': []
    }
    
    try:
        import unsloth
        from unsloth import FastLanguageModel
        
        info['unsloth_available'] = True
        info['unsloth_version'] = getattr(unsloth, '__version__', 'unknown')
        
    except Exception as e:
        error_str = str(e).lower()
        
        if 'undefined symbol' in error_str and 'cuda' in error_str:
            info['error_type'] = 'cuda_symbol_error'
            info['error_message'] = str(e)
            info['suggestions'] = [
                "CUDA symbol errors indicate version incompatibility",
                "Try: pip install --upgrade unsloth",
                "Or: Use TRL backends instead (--backend trl)"
            ]
        elif 'flash_attn' in error_str or 'flash attention' in error_str:
            info['error_type'] = 'flash_attention_error'
            info['error_message'] = str(e)
            info['suggestions'] = [
                "Flash Attention compatibility issue",
                "Try: pip install --upgrade flash-attn",
                "Unsloth will fallback to Xformers automatically"
            ]
        elif 'not found' in error_str or 'no module named' in error_str:
            info['error_type'] = 'missing_dependency'
            info['error_message'] = str(e)
            info['suggestions'] = [
                "Unsloth not installed",
                "Try: pip install unsloth",
                "Or: Use TRL backends instead (--backend trl)"
            ]
        else:
            info['error_type'] = 'unknown_error'
            info['error_message'] = str(e)
            info['suggestions'] = [
                "Unknown error occurred",
                "Try: pip install --upgrade unsloth",
                "Or: Use TRL backends instead (--backend trl)"
            ]
    
    return info

def check_attention_backends() -> Dict[str, Any]:
    """Check availability of attention backends."""
    info = {
        'flash_attention_available': False,
        'flash_attention_version': None,
        'xformers_available': False,
        'xformers_version': None,
        'recommended_backend': None
    }
    
    # Check Flash Attention
    try:
        import flash_attn
        info['flash_attention_available'] = True
        info['flash_attention_version'] = getattr(flash_attn, '__version__', 'unknown')
    except ImportError:
        pass
    
    # Check Xformers
    try:
        import xformers
        info['xformers_available'] = True
        info['xformers_version'] = getattr(xformers, '__version__', 'unknown')
    except ImportError:
        pass
    
    # Determine recommended backend
    if info['flash_attention_available']:
        info['recommended_backend'] = 'flash_attention'
    elif info['xformers_available']:
        info['recommended_backend'] = 'xformers'
    else:
        info['recommended_backend'] = 'pytorch_attention'
    
    return info

def get_recommended_versions() -> Dict[str, str]:
    """Get recommended versions for optimal compatibility."""
    return {
        'python': '3.8-3.12',
        'pytorch': '2.0.0-2.7.0',  # Avoid 2.8.0+ for now due to CUDA issues
        'cuda': '11.8, 12.1, 12.4',
        'unsloth': 'latest',
        'flash_attention': '2.5.0+',
        'xformers': '0.0.22+'
    }

def generate_diagnostic_report() -> Dict[str, Any]:
    """Generate comprehensive diagnostic report."""
    report = {
        'system_info': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.platform(),
            'architecture': platform.architecture()[0]
        },
        'pytorch_cuda': check_pytorch_cuda_compatibility(),
        'unsloth': check_unsloth_compatibility(),
        'attention_backends': check_attention_backends(),
        'recommended_versions': get_recommended_versions(),
        'overall_status': 'unknown'
    }
    
    # Determine overall status
    if report['unsloth']['unsloth_available']:
        report['overall_status'] = 'optimal'
    elif report['pytorch_cuda']['pytorch_available'] and report['pytorch_cuda']['cuda_available']:
        report['overall_status'] = 'compatible_but_issues'
    else:
        report['overall_status'] = 'incompatible'
    
    return report

def print_diagnostic_report(report: Optional[Dict[str, Any]] = None) -> None:
    """Print a formatted diagnostic report."""
    if report is None:
        report = generate_diagnostic_report()
    
    print("\n" + "="*60)
    print("FINETUNEHUB ENVIRONMENT DIAGNOSTICS")
    print("="*60)
    
    # System Information
    print(f"\nSystem Information:")
    print(f"  Python: {report['system_info']['python_version']}")
    print(f"  Platform: {report['system_info']['platform']}")
    print(f"  Architecture: {report['system_info']['architecture']}")
    
    # PyTorch/CUDA Status
    pytorch_info = report['pytorch_cuda']
    print(f"\nPyTorch/CUDA Status:")
    print(f"  PyTorch: {'✓' if pytorch_info['pytorch_available'] else '✗'} {pytorch_info['pytorch_version'] or 'Not available'}")
    print(f"  CUDA: {'✓' if pytorch_info['cuda_available'] else '✗'} {pytorch_info['cuda_version'] or 'Not available'}")
    if pytorch_info['cuda_available']:
        print(f"  CUDA Devices: {pytorch_info['cuda_device_count']}")
    
    # Unsloth Status
    unsloth_info = report['unsloth']
    print(f"\nUnsloth Status:")
    if unsloth_info['unsloth_available']:
        print(f"  Unsloth: ✓ {unsloth_info['unsloth_version']}")
    else:
        print(f"  Unsloth: ✗ Not available")
        if unsloth_info['error_type']:
            print(f"  Error Type: {unsloth_info['error_type']}")
            print(f"  Error: {unsloth_info['error_message']}")
            print(f"  Suggestions:")
            for suggestion in unsloth_info['suggestions']:
                print(f"    - {suggestion}")
    
    # Attention Backends
    attention_info = report['attention_backends']
    print(f"\nAttention Backends:")
    print(f"  Flash Attention: {'✓' if attention_info['flash_attention_available'] else '✗'} {attention_info['flash_attention_version'] or ''}")
    print(f"  Xformers: {'✓' if attention_info['xformers_available'] else '✗'} {attention_info['xformers_version'] or ''}")
    print(f"  Recommended: {attention_info['recommended_backend']}")
    
    # Overall Status
    status_icons = {
        'optimal': '✓',
        'compatible_but_issues': '⚠',
        'incompatible': '✗'
    }
    print(f"\nOverall Status: {status_icons.get(report['overall_status'], '?')} {report['overall_status'].replace('_', ' ').title()}")
    
    # Compatibility Issues
    if pytorch_info['compatibility_issues']:
        print(f"\nCompatibility Issues:")
        for issue in pytorch_info['compatibility_issues']:
            print(f"  - {issue}")
    
    print("\n" + "="*60)

def check_environment_compatibility() -> Tuple[bool, List[str]]:

    """Check if environment is compatible with AlignTune."""

    report = generate_diagnostic_report()
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"Python {python_version.major}.{python_version.minor} is not supported. Required: Python 3.8+")
    
    # Check PyTorch
    if not report['pytorch_cuda']['pytorch_available']:
        issues.append("PyTorch is not available")
    
    # Check CUDA for GPU training
    if not report['pytorch_cuda']['cuda_available']:
        issues.append("CUDA is not available (GPU training will not work)")
    
    # Check for critical compatibility issues
    if report['pytorch_cuda']['compatibility_issues']:
        issues.extend(report['pytorch_cuda']['compatibility_issues'])
    
    return len(issues) == 0, issues
