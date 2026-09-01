import logging
import time
import torch
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .callbacks import CallbackHandler, TrainerControl, TrainerCallback

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Training state tracking."""
    step: int = 0
    epoch: int = 0
    best_metric: float = 0.0
    checkpoint_path: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    last_eval_time: float = field(default_factory=time.time)
    last_save_time: float = field(default_factory=time.time)
    
    def update_step(self, step: int):
        self.step = step
    
    def update_epoch(self, epoch: int):
        self.epoch = epoch
    
    def update_best_metric(self, metric: float):
        if metric > self.best_metric:
            self.best_metric = metric
            return True
        return False
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time

class UnifiedTrainerBase(ABC):
    """
    Unified abstract base trainer for SFT and RLHF.
    Provides standard lifecycle, callbacks, checkpointing, and HF Hub logic.
    """
    
    def __init__(self, config: Any, callbacks: Optional[List[TrainerCallback]] = None):
        self.config = config
        self.state = TrainingState()
        self.control = TrainerControl()
        
        # Accelerate config propagation
        train_cfg = getattr(config, 'train', getattr(config, 'training', None))
        accelerate_config_path = getattr(train_cfg, 'accelerate_config_path', None) if train_cfg else None
        
        if accelerate_config_path:
            import os
            resolved = str(Path(accelerate_config_path).resolve())
            os.environ['ACCELERATE_CONFIG_FILE'] = resolved
            logger.info(f"Accelerate config set via ACCELERATE_CONFIG_FILE={resolved}")

        # Core components to be initialized by subclasses
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.eval_dataset = None
        self.data_loader = None
        self.logger = None  # UnifiedLogger or SFTLogger
        self.backend = None # Distributed backend interface
        
        self.callbacks = callbacks or []
        self.callback_handler = None
        
    @abstractmethod
    def setup_model(self) -> None:
        pass
        
    @abstractmethod
    def setup_data(self) -> None:
        pass
        
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        pass

    def _setup_optimizer_scheduler(self, dataset_for_estimation=None) -> Dict[str, Any]:
        """
        Setup optimizer and scheduler configurations for training.

        Args:
            dataset_for_estimation: Dataset to estimate max_steps if not provided in config

        Returns:
            Dictionary containing:
                - optimizer_name: str
                - scheduler_name: str
                - optimizer_config: Dict with optimizer_kwargs
                - scheduler_config: Dict with lr_scheduler_kwargs
                - optim_args_str: str (formatted for TRL)
                - warmup_steps: int
                - max_steps: int
        """
        from aligntune.core.optimization import get_optimizer_for_config, get_scheduler_for_config

        # Get optimizer and scheduler names
        optimizer_name = getattr(self.config.train, 'optimizer', 'adamw_torch')
        scheduler_name = getattr(self.config.train, 'lr_scheduler', 'cosine')

        # Create optimizer configuration
        optimizer_config = get_optimizer_for_config(
            optimizer_name,
            self.config.train.learning_rate,
            self.config.train.weight_decay or 0.01
        )

        # Calculate max_steps
        num_train_epochs = self.config.train.epochs or 1
        max_steps = getattr(self.config.train, 'max_steps', None)

        if max_steps is None:
            # Try to estimate from dataset
            dataset = dataset_for_estimation or getattr(self, 'train_dataset', None) or getattr(self, 'dataset', None)
            if dataset and hasattr(dataset, '__len__'):
                dataset_size = len(dataset)
                batch_size = self.config.train.per_device_batch_size or 1
                accumulation = self.config.train.gradient_accumulation_steps or 1
                effective_batch_size = batch_size * accumulation
                max_steps = int(num_train_epochs * dataset_size / effective_batch_size)
            else:
                max_steps = 1000  # fallback

        # Calculate warmup steps
        warmup_steps = getattr(self.config.train, 'warmup_steps', 0)
        if hasattr(self.config.train, 'warmup_ratio') and self.config.train.warmup_ratio:
            warmup_steps = int(max_steps * self.config.train.warmup_ratio)

        # Create scheduler configuration
        scheduler_config = get_scheduler_for_config(
            scheduler_name,
            max_steps,
            warmup_steps
        )

        # Helper to convert kwargs to string format
        def kwargs_to_str(kwargs_dict):
            """Convert optimizer/scheduler kwargs dict to string format for TRL."""
            if not kwargs_dict:
                return None
            # Filter out lr and weight_decay for optimizer (handled separately)
            filtered = {k: v for k, v in kwargs_dict.items() if k not in ['lr', 'learning_rate', 'weight_decay']}
            if not filtered:
                return None
            def format_value(v):
                if isinstance(v, tuple):
                    return f"({','.join(map(str, v))})"
                return str(v)
            return ",".join(f"{k}={format_value(v)}" for k, v in filtered.items())

        optim_args_str = kwargs_to_str(optimizer_config['optimizer_kwargs'])

        return {
            'optimizer_name': optimizer_name,
            'scheduler_name': scheduler_name,
            'optimizer_config': optimizer_config,
            'scheduler_config': scheduler_config,
            'optim_args_str': optim_args_str,
            'warmup_steps': warmup_steps,
            'max_steps': max_steps,
        }

    def get_hf_callbacks(self) -> List[Any]:
        """Get callback adapters for TRL / HF Trainers."""
        from aligntune.core.callbacks.hf_adapter import HuggingFaceCallbackAdapter
        
        if self.callback_handler is None:
            self.callback_handler = CallbackHandler(
                self.callbacks, self.model, self.tokenizer,
                optimizer=getattr(self, "optimizer", None),
                scheduler=getattr(self, "lr_scheduler", None)
            )
            
        hf_callback = HuggingFaceCallbackAdapter(
            callback_handler=self.callback_handler,
            unified_logger=self.logger,
            aligntune_config=self.config
        )
        return [hf_callback]
        
    def save_checkpoint(self) -> None:
        """Standard HF checkpoint saving."""
        if self.backend and not self.backend.is_rank_0():
            return
            
        logging_cfg = getattr(self.config, 'logging', None)
        out_dir = getattr(logging_cfg, 'output_dir', './output') if logging_cfg else './output'
        checkpoint_dir = Path(out_dir) / f"checkpoint-{self.state.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        if self.model is not None:
            self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
            
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "elapsed_time": self.state.get_elapsed_time()
            }, f, indent=2)
            
        self.state.checkpoint_path = str(checkpoint_dir)
        if self.callback_handler:
            self.callback_handler.on_save(self.config, self.state, self.control)
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load checkpoint state."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if self.model is not None:
            self.model.load_state_dict(torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu"))
        if self.tokenizer is not None:
            self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)
            
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state_data = json.load(f)
                self.state.step = state_data.get("step", 0)
                self.state.epoch = state_data.get("epoch", 0)
                self.state.best_metric = state_data.get("best_metric", 0.0)
                
        if self.backend and hasattr(self.backend, 'broadcast_checkpoint_path'):
            self.backend.broadcast_checkpoint_path(str(checkpoint_path))

    def push_to_hub(self, repo_id: str, private: bool = False, token: Optional[str] = None, commit_message: str = "Upload model") -> str:
        """Standard push to HF Hub."""
        if self.backend and not self.backend.is_rank_0():
            return ""
            
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")
            
        try:
            from huggingface_hub import HfApi, login
        except ImportError:
            raise ImportError("huggingface_hub is required for push_to_hub.")
            
        if token:
            login(token=token)
            
        try:
            repo_url = self.model.push_to_hub(repo_id, private=private, commit_message=commit_message)
            self.tokenizer.push_to_hub(repo_id, private=private, commit_message=commit_message)
            logger.info(f"Model pushed to hub successfully: {repo_url}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

        kind = "adapter" if hasattr(self.model, "peft_config") else "model"
        self._brand_hub_repo(repo_id, kind, private=private, token=token)
        return str(repo_url)

    def _hub_card_metadata(self) -> Dict[str, str]:
        """base_model / algorithm / backend for the model card, read off the config."""
        from ..utils.hf_publish import resolve_hub_base_model

        base_model = getattr(getattr(self.config, 'model', None), 'name_or_path', '') or ''
        resolved = resolve_hub_base_model(str(base_model))
        if not resolved:
            cfg = getattr(getattr(self, 'model', None), 'config', None)
            resolved = resolve_hub_base_model(
                str(getattr(cfg, '_name_or_path', '') or '')
            )
        algo = getattr(self.config, 'algo', None) or getattr(self.config, 'algorithm', None)
        algorithm = getattr(algo, 'value', algo) or 'finetune'
        module = type(self).__module__
        backend = next((b for b in ('unsloth', 'verl', 'trl') if f'.{b}.' in module), 'trl')
        return {
            'base_model': resolved,
            'algorithm': str(algorithm),
            'backend': backend,
        }

    def _brand_hub_repo(
        self,
        repo_id: str,
        kind: str,
        private: bool = False,
        token: Optional[str] = None,
        **kwargs
    ) -> None:
        """Write the AlignTune model card onto an already-pushed repo.

        Branding must never turn a successful weight upload into a failure, so
        every error here is logged and swallowed.
        """
        try:
            from ..utils.hf_publish import brand_hf_repo
            brand_hf_repo(
                repo_id,
                kind=kind,
                private=private,
                token=token,
                **self._hub_card_metadata(),
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Model card branding skipped for {repo_id}: {e}")

    def _merged_checkpoint(self) -> str:
        """Merge adapters once per session and reuse the result.

        The merge is the peak-memory step shared by the merged / quantized /
        GGUF paths, so it is cached and the training model is moved off the
        accelerator afterwards to leave room for the next load.
        """
        cached = getattr(self, '_merged_path', None)
        if cached:
            return cached

        path = self.merge_adapters()
        self._merged_path = path

        try:
            if self.model is not None:
                self.model.to('cpu')
        except Exception as e:
            # bitsandbytes models refuse .to(); nothing to free in that case.
            logger.debug(f"Could not offload training model to CPU: {e}")
        self._free_accelerator_memory()
        return path

    @staticmethod
    def _free_accelerator_memory() -> None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def push_merged_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        max_shard_size: str = "2GB",
    ) -> str:
        """Merge LoRA into the base weights and push a self-contained model.

        `max_shard_size` bounds the host-memory staging buffer used while
        serializing, since shards are written one at a time.
        """
        if self.backend and not self.backend.is_rank_0():
            return ""

        from transformers import AutoModelForCausalLM

        merged_path = self._merged_checkpoint()
        model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype="auto")
        try:
            model.push_to_hub(repo_id, private=private, max_shard_size=max_shard_size)
            self.tokenizer.push_to_hub(repo_id, private=private)
        finally:
            del model
            self._free_accelerator_memory()

        self._brand_hub_repo(repo_id, "model", private=private, token=token)
        return f"https://huggingface.co/{repo_id}"

    def push_quantized_to_hub(
        self,
        repo_id: str,
        quantization: str = "nf4",
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """Push a BitsAndBytes quantized copy of the merged model.

        `quantization` is one of nf4, fp4 (alias bf4) or int8. Each variant
        needs its own repo: the quantization_config lives at the repo root.
        """
        if self.backend and not self.backend.is_rank_0():
            return ""

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        presets = {
            'nf4': dict(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16),
            'fp4': dict(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.bfloat16),
            'bf4': dict(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.bfloat16),
            'int8': dict(load_in_8bit=True),
        }
        key = quantization.lower()
        if key not in presets:
            raise ValueError(f"quantization must be one of {sorted(presets)}, got {quantization!r}")

        merged_path = self._merged_checkpoint()
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            quantization_config=BitsAndBytesConfig(**presets[key]),
            device_map="auto",
        )
        try:
            model.push_to_hub(repo_id, private=private)
            self.tokenizer.push_to_hub(repo_id, private=private)
        finally:
            del model
            self._free_accelerator_memory()

        self._brand_hub_repo(
            repo_id,
            "quant",
            private=private,
            token=token,
            extra_notes=f"BitsAndBytes `{key}` quantization.",
        )
        return f"https://huggingface.co/{repo_id}"

    def push_gguf_to_hub(
        self,
        repo_id: str,
        quantizations: Union[str, List[str]],
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """Export one or more GGUF quantizations and upload them to a single repo.

        Several .gguf files coexist in one repo under different filenames, so
        the adapters are merged once and each quantization reuses that result.
        """
        if self.backend and not self.backend.is_rank_0():
            return ""

        from huggingface_hub import HfApi
        from .export.gguf import GGUFExporter, GGUF_QUANT_PRESETS

        if isinstance(quantizations, str):
            quantizations = [quantizations]
        quants = [q.upper() for q in quantizations]
        unknown = [q for q in quants if q not in GGUF_QUANT_PRESETS]
        if unknown:
            raise ValueError(
                f"Unknown GGUF quantization(s) {unknown}. "
                f"Valid options: {list(GGUF_QUANT_PRESETS)}"
            )

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        merged_path = self._merged_checkpoint()
        uploaded = []
        for quant in quants:
            out = GGUFExporter(output_dir=f"./gguf_{quant}", quantization=quant).export(
                checkpoint_path=merged_path
            )
            if not out:
                raise RuntimeError(f"GGUF export failed for {quant}")
            filename = f"model-{quant.lower()}.gguf"
            api.upload_file(path_or_fileobj=str(out), path_in_repo=filename, repo_id=repo_id)
            uploaded.append(filename)

        self._brand_hub_repo(repo_id, "gguf", private=private, token=token, gguf_files=uploaded)
        return f"https://huggingface.co/{repo_id}"

    def export_model(self, format: str = "gguf", **kwargs) -> Optional[str]:
        """
        Export the model to a deployment format using core.export pipeline.
        
        Args:
            format: "gguf", "ollama", "merge", or "hf_hub"
            **kwargs: Format-specific arguments (e.g. quantization="Q4_K_M")
        """
        if self.backend and not self.backend.is_rank_0():
            return None
            
        logger.info(f"Starting model export to format: {format}")
        
        # Ensure we have a saved checkpoint to export from
        if not self.state.checkpoint_path:
            logger.info("No checkpoint_path found, forcing a save before export.")
            self.save_checkpoint()
            
        checkpoint_path = Path(self.state.checkpoint_path)
        
        # Output directory alongside the checkpoint
        output_dir = checkpoint_path.parent / f"export_{format}"
        
        try:
            if format.lower() == "gguf":
                from aligntune.core.export.gguf import GGUFExporter
                exporter = GGUFExporter(output_dir=output_dir, quantization=kwargs.get("quantization"))
            elif format.lower() == "ollama":
                from aligntune.core.export.ollama import OllamaExporter
                exporter = OllamaExporter(output_dir=output_dir)
            elif format.lower() == "merge":
                from aligntune.core.export.merge_adapter import MergeAdapterExporter
                exporter = MergeAdapterExporter(output_dir=output_dir)
            elif format.lower() == "hf_hub":
                from aligntune.core.export.hf_hub import HFHubExporter
                exporter = HFHubExporter(output_dir=output_dir)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            export_path = exporter.export(checkpoint_path=str(checkpoint_path), **kwargs)
            logger.info(f"Successfully exported model to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def merge_adapters(
        self,
        output_path: Optional[str] = None,
        adapter_path: Optional[str] = None
    ) -> str:
        """
        Merge LoRA adapters into base model.

        Args:
            output_path: Directory to save merged model. Defaults to output_dir/merged
            adapter_path: Path to LoRA adapter checkpoint. If None, merges current model after training

        Returns:
            Path to merged model directory

        Example:
            >>> trainer.train()
            >>> merged_path = trainer.merge_adapters(output_path="./merged_model")
        """
        from .merge.peft_merger import PEFTMerger

        # Determine output path
        if output_path is None:
            base_output = getattr(self.config, 'logging', self.config).output_dir
            output_path = Path(base_output) / "merged"

        # Get base model path
        if adapter_path is not None:
            # Load from checkpoint
            base_model = getattr(self.config.model, 'name_or_path', 'model')
        else:
            # Use current model
            base_model = self.model

        # Use PEFTMerger
        merger = PEFTMerger()
        return merger.merge_lora(
            base_model=base_model,
            output_path=str(output_path),
            adapter_path=adapter_path,
            tokenizer=self.tokenizer
        )

    def _auto_export(self) -> None:
        """Helper to automatically trigger export_model if configured."""
        train_cfg = getattr(self.config, 'train', getattr(self.config, 'training', None))
        export_format = getattr(train_cfg, 'export_format', None)

        if export_format:
            quantization = getattr(train_cfg, 'export_quantization', None)
            self.export_model(format=export_format, quantization=quantization)