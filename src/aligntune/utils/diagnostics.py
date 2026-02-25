"""
Training diagnostics utilities for AlignTune.

This module provides comprehensive diagnostics for training runs,
including performance monitoring, memory usage tracking, and issue detection.
"""

import logging
import time
import platform
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import json
from pathlib import Path

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Real-time training metrics."""
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    throughput_samples_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    cpu_memory_used_gb: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingDiagnostics:
    """Comprehensive training diagnostics."""
    config_validation: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: List[TrainingMetrics] = field(default_factory=list)
    issues_detected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_metric(self, metric: TrainingMetrics):
        """Add a training metric."""
        self.performance_metrics.append(metric)

    def detect_issue(self, issue: str):
        """Record a detected issue."""
        self.issues_detected.append(issue)
        logger.warning(f"Training issue detected: {issue}")

    def add_recommendation(self, recommendation: str):
        """Add a recommendation."""
        self.recommendations.append(recommendation)

    def finalize(self):
        """Finalize diagnostics collection."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_validation": self.config_validation,
            "memory_usage": self.memory_usage,
            "performance_metrics": [vars(m) for m in self.performance_metrics],
            "issues_detected": self.issues_detected,
            "recommendations": self.recommendations,
            "duration_seconds": self.end_time - self.start_time if self.end_time else None,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    def save_to_file(self, path: str):
        """Save diagnostics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class TrainingMonitor:
    """Real-time training monitoring."""

    def __init__(self, diagnostics: TrainingDiagnostics):
        self.diagnostics = diagnostics
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start_monitoring(self, interval_seconds: float = 10.0):
        """Start background monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Training monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Training monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while not self.stop_event.wait(interval):
            if not self.monitoring:
                break

            try:
                metrics = self._collect_current_metrics()
                self.diagnostics.add_metric(metrics)
            except Exception as e:
                logger.warning(f"Failed to collect metrics: {e}")

    def _collect_current_metrics(self) -> TrainingMetrics:
        """Collect current training metrics."""
        # Get CPU memory
        cpu_memory_gb = psutil.virtual_memory().used / (1024 ** 3)

        # Get GPU memory if available
        gpu_memory_used_gb = 0.0
        gpu_memory_allocated_gb = 0.0

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_memory_used_gb = gpu.memoryUsed / 1024  # Convert to GB
                gpu_memory_allocated_gb = gpu.memoryTotal / 1024
        except Exception:
            pass  # GPU monitoring not available

        return TrainingMetrics(
            cpu_memory_used_gb=cpu_memory_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_allocated_gb=gpu_memory_allocated_gb
        )


class DiagnosticsCollector:
    """Collect and analyze training diagnostics."""

    def __init__(self):
        self.diagnostics = TrainingDiagnostics()

    @contextmanager
    def monitor_training(self, config):
        """Context manager for monitoring training."""
        monitor = TrainingMonitor(self.diagnostics)

        # Initial setup
        self._collect_initial_diagnostics(config)

        try:
            monitor.start_monitoring()
            yield self.diagnostics
        finally:
            monitor.stop_monitoring()
            self.diagnostics.finalize()
            self._analyze_diagnostics()

    def _collect_initial_diagnostics(self, config):
        """Collect initial diagnostics."""
        from .validation import ConfigValidator

        # Validate configuration
        is_valid, errors = ConfigValidator.validate_config(config)
        self.diagnostics.config_validation = {
            "is_valid": is_valid,
            "errors": errors
        }

        # Estimate memory usage
        memory_estimate = ConfigValidator.estimate_memory_usage(config)
        self.diagnostics.memory_usage = memory_estimate

        # Check environment compatibility
        env_compat = ConfigValidator.check_environment_compatibility(config)
        if not env_compat["compatible"]:
            for issue in env_compat["issues"]:
                self.diagnostics.detect_issue(f"Environment: {issue}")

        logger.info("Initial diagnostics collected")

    def _analyze_diagnostics(self):
        """Analyze collected diagnostics and generate recommendations."""
        # Analyze memory usage
        if self.diagnostics.memory_usage:
            mem_info = self.diagnostics.memory_usage
            if mem_info.get("total_estimated_memory_gb", 0) > 24:  # More than 24GB
                self.diagnostics.add_recommendation(
                    f"High memory usage detected ({mem_info['total_estimated_memory_gb']:.1f}GB). "
                    "Consider using gradient checkpointing or smaller batch sizes."
                )

        # Analyze performance metrics
        if len(self.diagnostics.performance_metrics) > 1:
            # Check for memory leaks
            memory_trend = [
                m.gpu_memory_used_gb for m in self.diagnostics.performance_metrics
                if m.gpu_memory_used_gb > 0
            ]
            if len(memory_trend) > 5:
                # Simple trend analysis
                if memory_trend[-1] > memory_trend[0] * 1.5:  # 50% increase
                    self.diagnostics.detect_issue("Potential memory leak detected")

            # Check throughput consistency
            throughputs = [
                m.throughput_samples_per_sec for m in self.diagnostics.performance_metrics
                if m.throughput_samples_per_sec > 0
            ]
            if len(throughputs) > 3:
                avg_throughput = sum(throughputs) / len(throughputs)
                if throughputs[-1] < avg_throughput * 0.5:  # 50% drop
                    self.diagnostics.detect_issue("Significant throughput degradation detected")

        # Analyze configuration issues
        if not self.diagnostics.config_validation.get("is_valid", False):
            self.diagnostics.add_recommendation(
                "Configuration validation failed. Check the errors above."
            )

    def save_report(self, output_dir: str, filename: str = "diagnostics.json"):
        """Save diagnostics report."""
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostics.save_to_file(str(output_path))
        logger.info(f"Diagnostics report saved to {output_path}")


def generate_training_report(diagnostics: TrainingDiagnostics) -> str:
    """Generate a human-readable training report."""
    report_lines = []

    report_lines.append("# 🚀 AlignTune Training Diagnostics Report")
    report_lines.append("")

    # Configuration validation
    config_valid = diagnostics.config_validation.get("is_valid", False)
    report_lines.append("## Configuration Validation")
    report_lines.append(f"Status: {'✅ Valid' if config_valid else '❌ Invalid'}")
    if not config_valid:
        errors = diagnostics.config_validation.get("errors", [])
        for error in errors:
            report_lines.append(f"  • {error}")
    report_lines.append("")

    # Memory usage
    if diagnostics.memory_usage:
        report_lines.append("## Memory Usage")
        mem = diagnostics.memory_usage
        report_lines.append(f"- Peak GPU Memory: {mem.get('peak_gpu_gb', 0):.1f} GB")
        report_lines.append(f"- Current GPU Memory: {mem.get('current_gpu_gb', 0):.1f} GB")
        report_lines.append(f"- CPU Memory Usage: {mem.get('cpu_percent', 0):.1f}%")
        report_lines.append("")

    # Performance metrics
    if diagnostics.performance_metrics:
        report_lines.append("## Performance Metrics")
        latest = diagnostics.performance_metrics[-1]
        report_lines.append(f"- Throughput: {latest.get('tokens_per_sec', 0):.2f} tokens/sec")
        report_lines.append(f"- GPU Utilization: {latest.get('gpu_utilization', 0):.1f}%")
        report_lines.append(f"- Loss: {latest.get('loss', 0):.4f}")
        report_lines.append("")

    # Issues detected
    if diagnostics.issues_detected:
        report_lines.append("## Issues Detected")
        for issue in diagnostics.issues_detected:
            report_lines.append(f"  ⚠️  {issue}")
        report_lines.append("")

    # Recommendations
    if diagnostics.recommendations:
        report_lines.append("## Recommendations")
        for rec in diagnostics.recommendations:
            report_lines.append(f"  💡 {rec}")
        report_lines.append("")

    # Duration
    if diagnostics.end_time:
        duration = diagnostics.end_time - diagnostics.start_time
        report_lines.append("## Training Duration")
        report_lines.append(f"- Total time: {duration:.2f} seconds")
        report_lines.append("")

    return "\n".join(report_lines)


# CLI integration functions
def run_config_validation(config_path: str, config_type: str = "auto") -> Dict[str, Any]:
    """
    Run configuration validation from CLI.

    Args:
        config_path: Path to configuration file
        config_type: Type of configuration ("sft", "rl", or "auto")

    Returns:
        Validation results
    """
    try:
        if config_type == "sft":
            from ..core.sft.config_loader import SFTConfigLoader
            config = SFTConfigLoader.load_from_yaml(config_path)
        else:
            from ..core.rl.config_loader import ConfigLoader
            config = ConfigLoader.load_from_yaml(config_path)

        from .validation import validate_config
        is_valid, errors = validate_config(config, config_type)

        result = {
            "config_type": config_type,
            "is_valid": is_valid,
            "errors": errors,
            "memory_estimate": {},
            "environment_check": {}
        }

        # Add memory estimate
        from .validation import ConfigValidator
        result["memory_estimate"] = ConfigValidator.estimate_memory_usage(config)

        # Add environment check
        result["environment_check"] = ConfigValidator.check_environment_compatibility(config)

        return result

    except Exception as e:
        return {
            "config_type": config_type,
            "is_valid": False,
            "errors": [f"Failed to load/validate config: {str(e)}"],
            "memory_estimate": {},
            "environment_check": {}
        }


def run_comprehensive_diagnostics() -> Dict[str, Any]:
    """
    Run comprehensive environment diagnostics.

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        "timestamp": time.time(),
        "system_info": {},
        "gpu_info": {},
        "library_versions": {},
        "compatibility_checks": {}
    }

    # System info
    vm = psutil.virtual_memory()
    diagnostics["system_info"] = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": vm.total / (1024 ** 3),
        "memory_available_gb": vm.available / (1024 ** 3),
        "platform": platform.platform()
    }

    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        diagnostics["gpu_info"] = {
            "gpu_count": len(gpus),
            "gpus": [
                {
                    "name": gpu.name,
                    "memory_total_gb": gpu.memoryTotal / 1024,
                    "driver": gpu.driver
                } for gpu in gpus
            ]
        }
    except Exception as e:
        diagnostics["gpu_info"] = {"error": str(e)}

    # Library versions
    library_checks = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("bitsandbytes", "bitsandbytes"),
        ("unsloth", "unsloth"),
    ]

    for lib_name, import_name in library_checks:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            diagnostics["library_versions"][lib_name] = version
        except ImportError:
            diagnostics["library_versions"][lib_name] = "not installed"

    # Compatibility checks
    compatibility = {
        "cuda_available": False,
        "torch_cuda_available": False,
        "flash_attention_2_available": False,
        "bitsandbytes_available": False,
        "unsloth_available": False
    }

    try:
        import torch
        compatibility["torch_cuda_available"] = torch.cuda.is_available()
        compatibility["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import bitsandbytes
        compatibility["bitsandbytes_available"] = True
    except ImportError:
        pass

    try:
        import unsloth
        compatibility["unsloth_available"] = True
    except ImportError:
        pass

    # Flash attention check (simplified)
    try:
        import transformers
        if hasattr(transformers, "modeling_utils") and hasattr(transformers.modeling_utils, "FlashAttention2"):
            compatibility["flash_attention_2_available"] = True
    except Exception:
        pass

    diagnostics["compatibility_checks"] = compatibility

    return diagnostics
