import os
import json
import torch
import logging
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Import Eval Components
from .rl_evaluator import RLEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset, DatasetDict

# Import Data Manager
from ..data.manager import DataManager
from ..data.schemas import TASK_SCHEMAS, TaskType

# Import WandB logger
from ..core.logging import WandBLogger

# Import colored logging
try:
    from ..utils.colored_logging import (
        print_section_banner,
        print_subsection,
        aligntune_info,
        aligntune_warning,
        aligntune_success,
        Fore,
    )
    COLORED_LOGGING_AVAILABLE = True
except ImportError:
    COLORED_LOGGING_AVAILABLE = False
    # Fallback functions
    def print_section_banner(title, char="=", width=80, color=""):
        print("\n" + char * width)
        print(f"  {title}".center(width))
        print(char * width + "\n")
    
    def print_subsection(title, char="-", width=80, color=""):
        print(f"{title}".center(width, char) + "\n")
    
    def aligntune_info(msg, prefix="[aligntune]"):
        print(f"{prefix} INFO - {msg}")
    
    def aligntune_warning(msg, prefix="[aligntune]"):
        print(f"{prefix} WARNING - {msg}")
    
    def aligntune_success(msg, prefix="[aligntune]"):
        print(f"{prefix} ✓ {msg}")
    
    class Fore:
        CYAN = ""
        RESET = ""
        GREEN = ""
        YELLOW = ""

# Import Metrics for Dynamic Loading
from .metrics.generic import PerplexityMetric, AccuracyMetric
from .metrics.code import PassAtKMetric
from .metrics.math import MathAccuracyMetric
from .metrics.text import RougeMetric, BleuMetric
from .metrics.rl import KLDivergenceMetric, RewardAccuracyMetric, PolicyEntropyMetric
from .metrics.dpo import (
    WinRateMetric,
    RewardMarginMetric,
    PreferenceAccuracyMetric,
    LogRatioMetric,
    ImplicitRewardMetric,
    CalibrationMetric
)

# Reasoning and Process Reward Model metrics
from .benchmarks.reasoning import ReasoningBenchmark


logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Configuration for running evaluations."""
    model_path: str
    output_dir: str
    dataset_name: str = None
    dataset_config: str = None
    split: str = "test"
    
    # --- Task Configuration ---
    # task_type: Controls METRICS selection (e.g., 'math', 'code', 'text')
    task_type: str = "math" 
    
    # data_task_type: Controls DATA LOADING/SCHEMA (e.g., 'sft', 'dpo', 'grpo', 'qa')
    # If None, it attempts to infer from task_type or defaults to 'sft'
    data_task_type: Optional[str] = None
    
    metrics: Optional[List[str]] = None
    k_list: Optional[List[int]] = None
    
    batch_size: int = 8
    max_samples: Optional[int] = None
    use_lora: bool = False
    use_unsloth: bool = False
    base_model: Optional[str] = None
    reference_model_path: Optional[str] = None
    reward_model: Optional[Any] = None
    dpo_beta: float = 0.1
    """DPO implicit-reward temperature used for preference metrics."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data configuration
    column_mapping: Optional[Dict[str, str]] = None
    trust_remote_code: bool = True

    # --- Advanced Model Loading Parameters ---
    # Replaces 'dtype' with robust precision argument
    # Options: "fp32", "bf16", "fp16", "auto", "float32", "bfloat16", "half"
    precision: str = "bf16"
    
    gpu_memory_utilization: float = 0.90
    load_in_4bit: bool = False
    low_cpu_mem_usage: bool = True
    max_lora_rank: int = 64
    tensor_parallel_size: Optional[int] = None  # Defaults to GPU count if None
    
    # Advanced Data Processing Features
    system_prompt: Optional[str] = None
    enable_thinking: bool = None
    apply_chat_template: bool = True
    processing_fn: Optional[Callable] = None  # User-defined processing function
    processing_batched: bool = False          # Whether processing_fn is batched
    auto_detect: bool = True                  # Enable auto-detection of dataset format
    
    # Prompt template for code generation
    code_prompt_template: Optional[str] = None
    
    # Data Splitting Parameters
    val_split_ratio: float = 0.1
    test_split_ratio: Optional[float] = None
    seed: int = 42

    # CuratorKIT data processing
    curator_schema_gate: bool = True
    curator_clean: bool = False
    curator_dedup: str = "none"
    curator_use_tiktoken: bool = False
    curator_max_tokens: int = 1_000_000

    # Legacy fields
    domain: Optional[str] = None 
    # NOTE: sampling params only matter when do_sample=True.
    # Keep temperature optional so it is only applied when explicitly set.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    
    # --- Generation & Context Control ---
    # max_length: Controls the CONTEXT WINDOW (Input + Output capacity).
    # Passed to max_seq_length (Unsloth) or max_model_len (vLLM).
    max_length: int = 2048 
    
    # max_new_tokens: Controls the GENERATION output length.
    # Passed to gen_kwargs['max_new_tokens'].
    max_new_tokens: int = 512
    
    num_samples: int = 1
    use_vllm: bool = False
    quantization: Optional[Dict[str, Any]] = None
    device_map: Optional[Any] = None


    num_print_samples: int = 3           # Number of samples to print
    print_sample_mode: str = "random"     # Options: "first", "random", "indices"
    print_sample_indices: Optional[List[int]] = None  # Specific indices if mode="indices"

    # WandB integration (optional, graceful degradation)
    wandb_project: Optional[str] = None
    """wandb project name. If None, wandb tracking is disabled."""

    wandb_entity: Optional[str] = None
    """wandb entity (team or username). Optional."""

    wandb_tags: Optional[List[str]] = None
    """List of tags to attach to wandb run for filtering and organization."""

    wandb_notes: Optional[str] = None
    """Optional notes about the evaluation to log to wandb."""


def get_metrics_from_names(metric_names: List[str], k_list: Optional[List[int]] = None) -> List[Any]:
    """Factory to map string names to Metric objects."""
    metrics_map = {
        "perplexity": PerplexityMetric,
        "accuracy": AccuracyMetric,
        "rouge": RougeMetric,
        "bleu": BleuMetric,
        "math_accuracy": MathAccuracyMetric,
        "pass_at_k": PassAtKMetric,
        "kl_divergence": KLDivergenceMetric,
        "reward_accuracy": RewardAccuracyMetric,
        "policy_entropy": PolicyEntropyMetric,
        
        # DPO metrics
        "win_rate": WinRateMetric,
        "reward_margin": RewardMarginMetric,
        "preference_accuracy": PreferenceAccuracyMetric,
        "log_ratio": LogRatioMetric,
        "implicit_reward": ImplicitRewardMetric,
        "calibration": CalibrationMetric,
    }
    
    instantiated_metrics = []
    for name in metric_names:
        key = name.lower()
        if key in metrics_map:
            if key == "pass_at_k":
                k_list_val = k_list or [1]
                instantiated_metrics.append(metrics_map[key](k_list=k_list_val))
            else:
                instantiated_metrics.append(metrics_map[key]())
        else:
            logger.warning(f"Unknown metric name: '{name}'. Skipping.")
    
    return instantiated_metrics


def run_eval(config: EvalConfig, dataset_dict: Optional[DatasetDict] = None) -> Dict[str, Any]:
    """
    Unified entry point for running evaluation using the Config object.
    """
    # Initialize wandb logger if configured
    wandb_logger = None
    if hasattr(config, 'wandb_project') and config.wandb_project:
        wandb_logger = WandBLogger(
            project=config.wandb_project,
            entity=getattr(config, 'wandb_entity', None),
            config={
                'model_path': config.model_path,
                'task_type': config.task_type,
                'dataset': config.dataset_name,
                'batch_size': config.batch_size,
                'precision': config.precision,
            },
            tags=getattr(config, 'wandb_tags', None),
            notes=getattr(config, 'wandb_notes', None),
        )

    # Aggressive memory cleanup before loading new models
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print_section_banner("ALIGNTUNE EVALUATION", color=Fore.CYAN if COLORED_LOGGING_AVAILABLE else "", use_ascii=True)
    aligntune_info(f"Model: {config.model_path}")
    aligntune_info(f"Eval Task (Metrics): {config.task_type}")
    logger.info(f"--- Starting Evaluation ---")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Eval Task (Metrics): {config.task_type}")
    logger.info(f"Precision: {config.precision}")

    # Resolve torch dtype based on config precision
    # Supports flexible naming: fp32, float32, fp16, float16, half, bf16, bfloat16, auto
    p = config.precision.lower() if config.precision else "auto"
    torch_dtype = "auto"
    vllm_dtype = "auto"

    if p in ["fp32", "float32"]:
        torch_dtype = torch.float32
        vllm_dtype = "float32"
    elif p in ["fp16", "float16", "half"]:
        torch_dtype = torch.float16
        vllm_dtype = "float16"
    elif p in ["bf16", "bfloat16"]:
        torch_dtype = torch.bfloat16
        vllm_dtype = "bfloat16"
    elif p == "auto":
        torch_dtype = "auto"
        vllm_dtype = "auto"
    else:
        # Fallback: assume user might pass a custom string supported by libs
        torch_dtype = p if p == "auto" else "auto" # Safe fallback for torch
        vllm_dtype = p
    
    # 1. Load Model & Tokenizer
    model = None
    tokenizer = None
    
    if config.use_vllm:
        # Pre-check for vLLM installation to allow fallback
        try:
            import vllm
        except ImportError:
            logger.warning("vLLM requested but not installed. Falling back to regular inference.")
            config.use_vllm = False

    if config.use_vllm:
        logger.info("Loading model using vLLM...")
        try:
            from vllm import LLM
            
            # Determine correct model path and enable_lora flag
            vllm_model_path = config.model_path
            enable_lora = False
            
            # If using LoRA, vLLM loads the BASE model in the engine, and applies LoRA at runtime
            if config.use_lora and config.base_model:
                logger.info(f"vLLM LoRA Mode: Loading base '{config.base_model}' for adapter '{config.model_path}'")
                vllm_model_path = config.base_model
                enable_lora = True
            
            tp_size = config.tensor_parallel_size or (torch.cuda.device_count() or 1)

            model = LLM(
                model=vllm_model_path,
                trust_remote_code=config.trust_remote_code,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_lora=enable_lora,
                max_lora_rank=config.max_lora_rank if enable_lora else None, 
                dtype=vllm_dtype, 
                max_model_len=config.max_length # Explicitly control context window
            )
            
            # Attach the actual adapter path to the model object so RLEvaluator can access it
            # if it needs to create a LoRARequest
            if enable_lora:
                model.lora_adapter_path = config.model_path
            
            # vLLM manages its own tokenizer internally, but we need an AutoTokenizer instance
            # for DataManager and metric calculations (decoding, etc.)
            logger.info(f"Loading tokenizer for data processing: {config.model_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {config.model_path}, trying base: {e}")
                if config.base_model:
                    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=config.trust_remote_code)
                else:
                    raise e

        # except ImportError:
        #     logger.error("vLLM is not installed. Please install it with `pip install vllm`.")
        #     raise
        except Exception as e:
            logger.error(f"Failed to load with vLLM: {e}")
            raise

    elif config.use_unsloth:
        logger.info("Loading model using Unsloth FastLanguageModel...")
        try:
            from unsloth import FastLanguageModel
            from peft import PeftModel
            
            unsloth_dtype = None if torch_dtype == "auto" else torch_dtype
            
            # Check if this is a LoRA adapter
            if config.use_lora and config.base_model:
                logger.info(f"Loading Unsloth with LoRA: base={config.base_model}, adapter={config.model_path}")
                
                # Load base model with Unsloth
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.base_model,
                    max_seq_length=config.max_length,
                    dtype=unsloth_dtype,
                    load_in_4bit=config.load_in_4bit,
                    trust_remote_code=config.trust_remote_code
                )
                
                # Load and merge LoRA adapter
                model = PeftModel.from_pretrained(model, config.model_path)
                model = model.merge_and_unload()
                
                FastLanguageModel.for_inference(model)
                logger.info("Merged LoRA adapter with Unsloth optimizations")
            else:
                # Standard model loading
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.model_path,
                    max_seq_length=config.max_length,
                    dtype=unsloth_dtype,
                    load_in_4bit=config.load_in_4bit,
                    trust_remote_code=config.trust_remote_code
                )
                
                FastLanguageModel.for_inference(model)
                logger.info("Unsloth inference optimizations enabled.")
                
        except ImportError:
            logger.error("Unsloth is not installed. Please install it or set use_unsloth=False.")
            raise
        except Exception as e:
            logger.error(f"Failed to load with Unsloth: {e}")
            raise
    else:
        # Standard Hugging Face Loading
        is_peft = config.use_lora or os.path.exists(os.path.join(config.model_path, "adapter_config.json"))
        
        if is_peft:
            base_path = config.base_model
            if not base_path:
                try:
                    peft_cfg = PeftConfig.from_pretrained(config.model_path)
                    base_path = peft_cfg.base_model_name_or_path
                except:
                    raise ValueError("Base model path required for LoRA evaluation.")
            
            aligntune_info(f"Loading LoRA Adapter: {config.model_path} on Base: {base_path}")
            logger.info(f"Loading LoRA Adapter: {config.model_path} on Base: {base_path}")
            
            # Load tokenizer first to check if we need to resize embeddings
            # Try adapter path first (may have resized tokenizer), fallback to base
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                logger.info(f"Loaded tokenizer from adapter path: vocab_size={len(tokenizer)}")
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_path)
                logger.info(f"Loaded tokenizer from base model: vocab_size={len(tokenizer)}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                device_map=config.device,
                torch_dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            
            # Check if embeddings need resizing (adapter may have added tokens)
            base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) != base_vocab_size:
                logger.info(f"Resizing embeddings: {base_vocab_size} -> {len(tokenizer)}")
                base_model.resize_token_embeddings(len(tokenizer))

            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, config.model_path)
            except Exception as e:
                from peft import PeftModel
                logger.warning(f"Failed to load adapter, trying with resized embeddings: {e}")
                # Force resize if loading failed
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                base_model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(base_model, config.model_path)
                
            try:
                model = model.merge_and_unload()
                aligntune_success("Successfully merged LoRA adapter into base model.")
                logger.info("Successfully merged LoRA adapter into base model.")
            except Exception as e:
                logger.warning(f"Could not merge LoRA adapter: {e}")
                
            
        else:
            logger.info("Loading Standard Model...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                device_map=config.device,
                torch_dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code,
                low_cpu_mem_usage=config.low_cpu_mem_usage
            )
            # Some training outputs (e.g., checkpoint folders) may not include a full tokenizer
            # bundle (tokenizer_config.json, special_tokens_map.json, etc.). When that happens,
            # Transformers can fail while inferring tokenizer type from local files.
            #
            # Fall back to the base model tokenizer (if provided) to keep eval working.
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            except Exception as e:
                if config.base_model:
                    logger.warning(
                        f"Failed to load tokenizer from '{config.model_path}': {e}. "
                        f"Falling back to base_model tokenizer '{config.base_model}'."
                    )
                    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
                else:
                    raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # FIX: Explicitly enforce the max_length on the tokenizer.
    if hasattr(config, 'max_length') and config.max_length:
        tokenizer.model_max_length = config.max_length
    
    # Enforce Left Padding for Generation
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
        logger.info("Switched tokenizer padding_side to 'left' for generation stability.")
    
    # 2. Load Reference Model (if needed)
    ref_model = None
    if config.reference_model_path:
        logger.info(f"Loading Reference Model: {config.reference_model_path}")
        # Note: Unsloth loading logic for Ref Model also needs to be robust to dtype
        if config.use_unsloth:
            # Re-import unsloth safely
            from unsloth import FastLanguageModel
            unsloth_dtype = None if torch_dtype == "auto" else torch_dtype
            
            # Note: Ref model is usually base model, not lora
            model_to_load = config.reference_model_path
            
            model_ref, _ = FastLanguageModel.from_pretrained(
                model_name=model_to_load,
                max_seq_length=config.max_length or 2048,
                dtype=unsloth_dtype,
                load_in_4bit=config.load_in_4bit,
                trust_remote_code=config.trust_remote_code
            )
            
            FastLanguageModel.for_inference(model_ref)
            ref_model = model_ref
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                config.reference_model_path,
                device_map=config.device,
                torch_dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code
            )
    
    # 3. Load Dataset using DataManager
    print_subsection("Loading Dataset", color=Fore.CYAN if COLORED_LOGGING_AVAILABLE else "")
    aligntune_info(f"Loading Dataset via DataManager...")
    logger.info(f"Loading Dataset via DataManager...")
    
    # Determine Data Task Type (for Schema/Columns)
    final_data_task = "sft"  # Ultimate default

    if config.data_task_type:
        # User explicitly specified data task type
        final_data_task = config.data_task_type
        logger.info(f"Using explicit Data Task Type: '{final_data_task}'")
    else:
        # Infer from Eval Task Type
        eval_task = config.task_type.lower()
        
        # Map eval task to data task
        task_mapping = {
            # Code tasks → CODE schema (needs test_cases)
            "code": "code",
            
            # Math/Text → SFT schema (prompt/completion)
            "math": "sft",
            "text": "sft",
            "text_generation": "sft",
            "generic": "sft",
            
            # RL tasks
            "dpo": "dpo",
            "grpo": "grpo",
            "ppo": "grpo",  # PPO can use GRPO schema
            
            # Other tasks
            "qa": "qa",
            "summarization": "summarization",
        }
        
        final_data_task = task_mapping.get(eval_task, "sft")
        logger.info(f"Inferred Data Task Type: '{final_data_task}' from Eval Task '{eval_task}'")

    # Initialize DataManager
    data_manager = DataManager(
        task_type=final_data_task,
        column_mapping=config.column_mapping,
        tokenizer=tokenizer,
        val_split_ratio=config.val_split_ratio,
        test_split_ratio=config.test_split_ratio,
        seed=config.seed,
        system_prompt=config.system_prompt,
        enable_thinking=config.enable_thinking,
        processing_fn=config.processing_fn,
        processing_batched=config.processing_batched,
        auto_detect=config.auto_detect,
        max_samples=config.max_samples,
        curator_schema_gate=config.curator_schema_gate,
        curator_clean=config.curator_clean,
        curator_dedup=config.curator_dedup,
        curator_use_tiktoken=config.curator_use_tiktoken,
        curator_max_tokens=config.curator_max_tokens,
        # trust_remote_code=config.trust_remote_code
    )
    
    try:
        # A requested test split may be created from train/validation by
        # DataManager. Do not ask the HF loader for a physical ``test`` split
        # before that split policy runs.
        # When a requested validation/test split is configured by ratio, load
        # the source dataset first so DataManager can create that split. Do
        # not ask the HF loader for a physical split that may not exist.
        ratio_split_requested = (
            config.split == "validation" and config.val_split_ratio is not None
        ) or (
            config.split == "test" and config.test_split_ratio is not None
        )
        load_split = None if ratio_split_requested else config.split
        if dataset_dict is not None:
            logger.info("Using provided dataset; processing it through DataManager")
            # Keep in-memory and named datasets on the same CuratorKIT and
            # normalization path. Previously this branch bypassed all data
            # processing and handed raw columns to the evaluator.
            dataset_dict = data_manager.load_dataset(
                dataset_dict, split=load_split
            )
        else:
            dataset_dict = data_manager.load_dataset(
                config.dataset_name,
                config_name=config.dataset_config,
                split=load_split,
            )
        

        dataset = None
        requested_split = config.split

        # Support HF-style slice syntax like "train[2000:2100]" even when
        # DataManager returns a DatasetDict with literal split keys.
        slice_match = re.match(r"^(?P<base>[^\[\]]+)\[(?P<spec>[^\]]+)\]$", requested_split or "")

        if requested_split in dataset_dict:
            dataset = dataset_dict[requested_split]
            logger.info(f"Using requested split '{requested_split}': {len(dataset)} samples")
        elif slice_match:
            base = slice_match.group("base")
            spec = slice_match.group("spec").strip()

            if base in dataset_dict:
                base_ds = dataset_dict[base]
                try:
                    if ":" in spec:
                        start_s, end_s = spec.split(":", 1)
                        start = int(start_s) if start_s.strip() else 0
                        end = int(end_s) if end_s.strip() else len(base_ds)
                        if start < 0 or end < 0:
                            raise ValueError("Negative slice bounds are not supported.")
                        if end < start:
                            raise ValueError("Slice end must be >= start.")
                        end = min(end, len(base_ds))
                        indices = range(start, end)
                    else:
                        idx = int(spec)
                        if idx < 0:
                            raise ValueError("Negative indices are not supported.")
                        indices = [idx]

                    # `datasets.Dataset.select` accepts list-like indices.
                    dataset = base_ds.select(list(indices))
                    logger.info(
                        f"Using requested split slice '{requested_split}' from '{base}': {len(dataset)} samples"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not apply split slice '{requested_split}' from '{base}': {e}. Falling back..."
                    )

        if dataset is None:
            logger.warning(f"Split '{requested_split}' not found. Falling back...")
            if "test" in dataset_dict:
                dataset = dataset_dict["test"]
                logger.info(f"Fallback: Using 'test' split: {len(dataset)} samples")
            elif "validation" in dataset_dict:
                dataset = dataset_dict["validation"]
                logger.info(f"Fallback: Using 'validation' split: {len(dataset)} samples")
            else:
                fallback_split = list(dataset_dict.keys())[0]
                dataset = dataset_dict[fallback_split]
                logger.warning(f"Fallback: Using available split '{fallback_split}': {len(dataset)} samples")

    except Exception as e:
        logger.error(f"DataManager failed to load dataset: {e}")
        raise

    # --- INSPECT DATASET SCHEMA & SAMPLES ---
    print_subsection("DATASET INSPECTION", color=Fore.CYAN if COLORED_LOGGING_AVAILABLE else "")
    try:
        # Resolve Schema based on final_data_task
        task_enum = TaskType(final_data_task)
        schema = TASK_SCHEMAS.get(task_enum)
        
        print(f"  Active Data Task: {final_data_task.upper()}")
        print(f"  Required Columns: {schema.required_columns}")
        print(f"  Actual Columns:   {dataset.column_names}")
        
        # Verify columns
        missing_cols = [c for c in schema.required_columns if c not in dataset.column_names]
        if missing_cols:
            aligntune_warning(f"Missing required columns for {final_data_task}: {missing_cols}")
            print(f"  Existing heuristic mapping may have failed. Available columns: {dataset.column_names}")
        else:
            aligntune_success(f"Schema validation passed for {final_data_task}.")

        # Print One Sample
        if len(dataset) > 0:
            print("\n  Sample Data Row (First Example):")
            sample_row = dataset[0]
            for col in schema.required_columns:
                val = sample_row.get(col, "[MISSING]")
                val_str = str(val)
                # Truncate for display
                if len(val_str) > 150:
                    val_str = val_str[:150] + "... [truncated]"
                print(f"    - {col}: {val_str}")
        print("")
    except Exception as e:
        logger.warning(f"Could not print dataset inspection sample: {e}")
    
    # 4. Initialize Evaluator (Metrics)
    custom_metrics = None
    if config.metrics:
        logger.info(f"Using custom metric list: {config.metrics}")
        custom_metrics = get_metrics_from_names(config.metrics, k_list=config.k_list)
    
    gen_kwargs = {}
    # Use max_new_tokens for generation length
    if config.max_new_tokens:
        gen_kwargs['max_new_tokens'] = config.max_new_tokens
    if config.temperature is not None:
        gen_kwargs['temperature'] = config.temperature
    if config.top_p is not None:
        gen_kwargs['top_p'] = config.top_p
    if config.top_k is not None:
        gen_kwargs['top_k'] = config.top_k
    
    if config.k_list and max(config.k_list) > 1:
        gen_kwargs['do_sample'] = True
        # Only set a default top_p if the user didn't provide one.
        gen_kwargs.setdefault('top_p', 0.95)

    # Respect explicit do_sample override, otherwise auto-enable if sampling knobs are set.
    if config.do_sample is not None:
        gen_kwargs['do_sample'] = config.do_sample
    elif any(k in gen_kwargs for k in ("temperature", "top_p", "top_k")):
        gen_kwargs['do_sample'] = True
    
    evaluator = RLEvaluator(
        metrics=custom_metrics, 
        task_type=config.task_type, # This controls DEFAULT metrics
        batch_size=config.batch_size,
        device=config.device,
        generation_kwargs=gen_kwargs,
        k_list=config.k_list,
        use_unsloth=config.use_unsloth,
        apply_chat_template=config.apply_chat_template,
        enable_thinking=config.enable_thinking,
        dpo_beta=config.dpo_beta,
        data_task_type=final_data_task,
    )
    
    # 5. Run Evaluation
    print_subsection("Running Evaluation", color=Fore.CYAN if COLORED_LOGGING_AVAILABLE else "")
    aligntune_info("Starting evaluation...")
    print(dataset)
    results = evaluator.evaluate_rl(
        policy_model=model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_model=config.reward_model,
        max_samples=config.max_samples,
        # DataManager has normalized all supported datasets to these canonical
        # names. Keep that contract explicit for the evaluator.
        column_mapping={"prompt": "prompt", "completion": "completion"},
    )
    
    # 6. Save & Return
    os.makedirs(config.output_dir, exist_ok=True)
    results_path = os.path.join(config.output_dir, "eval_results.json")
    
    final_output = {
        "metrics": results,
        "config": {k: str(v) for k, v in config.__dict__.items() if k != "reward_model" and k != "processing_fn"}
    }
    
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    aligntune_success(f"Results saved to {results_path}")
    logger.info(f"Results saved to {results_path}")
    
    if "math_accuracy" in results:
        results["pass_at_1"] = results["math_accuracy"]
        
    for k in (config.k_list or [1]):
        key = f"pass@{k}"
        if key in results:
            results[f"pass_at_{k}"] = results[key]
    
    
    ## Print Samples 
    if config.num_print_samples > 0:
        print_section_banner("SAMPLE OUTPUTS", color=Fore.CYAN if COLORED_LOGGING_AVAILABLE else "")
    
    # Determine which samples to print
    total_samples = len(results.get("sample_queries", []))
    if total_samples == 0:
        total_samples = len(results.get("sample_predictions", []))
    
    if total_samples > 0:
        if config.print_sample_mode == "random":
            import random
            sample_indices = random.sample(range(total_samples), min(config.num_print_samples, total_samples))
        elif config.print_sample_mode == "indices" and config.print_sample_indices:
            sample_indices = [i for i in config.print_sample_indices if i < total_samples]
        else:  # "first" or default
            sample_indices = list(range(min(config.num_print_samples, total_samples)))
        
        for idx in sample_indices:
            print(f"\n{'─' * 70}")
            print(f"Sample {idx + 1}")
            print(f"{'─' * 70}")
            
            if "sample_queries" in results and idx < len(results["sample_queries"]):
                print(f"\n📝 Query:\n{results['sample_queries'][idx]}\n")
            
            if "sample_predictions" in results and idx < len(results["sample_predictions"]):
                print(f"🤖 Generated Response:\n{results['sample_predictions'][idx]}\n")
            
            if "sample_references" in results and idx < len(results["sample_references"]):
                print(f"✅ Reference/Ground Truth:\n{results['sample_references'][idx]}\n")
        
        print(f"{'═' * 70}\n")

    results["total"] = config.max_samples if config.max_samples else len(dataset)

    # Log evaluation metrics to wandb
    if wandb_logger:
        try:
            # Extract metric results and log them
            metric_results = {k: v for k, v in results.items()
                            if k not in ['sample_queries', 'sample_predictions', 'sample_references', 'total']}
            if metric_results:
                wandb_logger.log_metrics(metric_results)

            # Log evaluation checkpoint if output_dir exists
            if config.output_dir and Path(config.output_dir).exists():
                wandb_logger.log_artifact(
                    config.output_dir,
                    artifact_type="evaluation",
                    name=f"eval-{config.task_type}"
                )

            wandb_logger.finalize()
        except Exception as e:
            logger.warning(f"Error logging evaluation to wandb: {str(e)}")

    return results


def register_reasoning_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Register reasoning metrics for process reward model evaluation.

    Returns:
        Dictionary mapping benchmark names to their metric configurations:
        {
            "AIME": {"metric": "exact_match", "task": "AIME"},
            "MATH": {"metric": "exact_match", "task": "MATH"},
            ...
        }

    Note:
        This function prepares metric definitions for reasoning benchmarks
        without executing any actual evaluations.
    """
    reasoning_metrics = {
        "AIME": {
            "metric": "exact_match",
            "task": "AIME",
            "description": "American Invitational Mathematics Examination",
            "category": "reasoning",
        },
        "MATH": {
            "metric": "exact_match",
            "task": "MATH",
            "description": "Diverse mathematical problem collection",
            "category": "reasoning",
        },
        "GPQA": {
            "metric": "accuracy",
            "task": "GPQA",
            "description": "Graduate-level Professional and Academic Questions",
            "category": "reasoning",
        },
        "LiveCodeBench": {
            "metric": "code_execution",
            "task": "livecode",
            "description": "Code execution benchmarks",
            "category": "reasoning",
        },
        "GSM8K-CoT": {
            "metric": "exact_match",
            "task": "gsm8k_cot",
            "description": "Grade school math with chain-of-thought",
            "category": "reasoning",
        },
    }

    logger.info(f"Registered {len(reasoning_metrics)} reasoning metrics")
    return reasoning_metrics


def get_reasoning_benchmark(name: str, split: str = "test") -> Optional[Dict[str, Any]]:
    """
    Get a reasoning benchmark for evaluation.

    Args:
        name: Benchmark name (AIME, MATH, GPQA, LiveCodeBench, GSM8K-CoT)
        split: Dataset split (test, val, train)

    Returns:
        Dictionary with benchmark data (questions, solutions, step_labels)
        or None if benchmark cannot be loaded

    Note:
        This is a stub for the actual benchmark loading infrastructure.
    """
    try:
        benchmark = ReasoningBenchmark()
        data = benchmark.load_benchmark(name, split=split)
        return data.to_dict()
    except Exception as e:
        logger.warning(f"Failed to load reasoning benchmark {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Long-Context Benchmark Registry  (v3.9)
# ---------------------------------------------------------------------------

def register_long_context_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Register long-context evaluation benchmarks in the metrics registry.

    Two benchmarks are provided:

    * **needle_in_haystack** — A retrieval probe that embeds a short "needle"
      sentence at a random position inside a long document ("haystack") and
      asks the model to reproduce it.  Scores are broken down by
      ``context_length`` so you can track degradation as the window grows.

    * **longbench** — A multi-task long-document understanding benchmark
      covering multi-hop QA (``2wikimqa``, ``hotpotqa``) and summarisation
      (``gov_report``).  Each task is evaluated separately and then averaged
      into a composite ``longbench_avg`` score.

    Returns:
        A nested dictionary keyed by benchmark name.  Each value is a
        configuration dictionary that downstream runners consume to set up
        the evaluation environment.  Example structure::

            {
                "needle_in_haystack": {
                    "metric": "retrieval_accuracy",
                    "category": "long_context",
                    "context_lengths": [4096, 8192, 32768],
                    ...
                },
                "longbench": {
                    "metric": "f1",
                    "category": "long_context",
                    "tasks": ["2wikimqa", "hotpotqa", "gov_report"],
                    ...
                },
            }

    Note:
        This function only **registers** benchmark metadata; it does not
        download datasets or run inference.  Pair it with a compatible runner
        (e.g. ``LongContextEvaluator``) to execute evaluations.
    """
    long_context_metrics: Dict[str, Dict[str, Any]] = {
        "needle_in_haystack": {
            "metric": "retrieval_accuracy",
            "task": "needle_in_haystack",
            "description": (
                "Retrieval accuracy probe: a needle sentence is inserted at a "
                "random depth inside a long document and the model must reproduce "
                "it verbatim.  Evaluated at multiple context lengths to expose "
                "performance cliffs."
            ),
            "category": "long_context",
            # Context lengths (in tokens) at which the probe is evaluated.
            # Covers the short-context regime (4k), mid-range (8k), and the
            # extended target (32k) relevant to most long-context recipes.
            "context_lengths": [4096, 8192, 32768],
            "scoring": "exact_match_ratio",  # fraction of needles retrieved correctly
            "dataset": "ruler/needle",        # HF dataset identifier (stub)
        },
        "longbench": {
            "metric": "f1",
            "task": "longbench",
            "description": (
                "Multi-task long-document understanding benchmark covering "
                "multi-hop question answering and long-form summarisation."
            ),
            "category": "long_context",
            # Subset of LongBench tasks selected to cover QA and summarisation.
            "tasks": ["2wikimqa", "hotpotqa", "gov_report"],
            "scoring": "per_task_f1_and_avg",  # individual F1 + composite average
            "dataset": "THUDM/LongBench",       # HF dataset identifier
        },
    }

    logger.info(
        f"Registered {len(long_context_metrics)} long-context benchmarks: "
        + ", ".join(long_context_metrics.keys())
    )
    return long_context_metrics


# Merged metrics registry — combines reasoning and long-context benchmarks
# so callers can discover all available benchmarks from a single source.
METRICS_REGISTRY: Dict[str, Dict[str, Any]] = {
    **register_reasoning_metrics(),
    **register_long_context_metrics(),
}
"""
Global registry of all evaluation benchmarks supported by AlignTune.

Structure::

    {
        "<benchmark_name>": {
            "metric":      str,   # primary metric type
            "task":        str,   # internal task identifier
            "description": str,   # human-readable description
            "category":    str,   # "reasoning" | "long_context"
            ...                   # benchmark-specific keys
        },
        ...
    }

Reasoning benchmarks (from :func:`register_reasoning_metrics`):
    AIME, MATH, GPQA, LiveCodeBench, GSM8K-CoT

Long-context benchmarks (from :func:`register_long_context_metrics`):
    needle_in_haystack, longbench
"""
