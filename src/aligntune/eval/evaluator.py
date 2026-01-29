"""
The Main Evaluator Class. 
Merges the new BaseEvaluator architecture with legacy lm-eval capabilities.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np

# Import New Metrics
from .metrics.base import Metric
from .metrics.generic import PerplexityMetric, AccuracyMetric
from .metrics.code import PassAtKMetric
from .metrics.math import MathAccuracyMetric
from .metrics.text import RougeMetric, BleuMetric
from .caching import EvaluationCache

# Schema for robust column detection
from ..data.schemas import TASK_SCHEMAS, TaskType as SchemaTaskType

# Import Legacy Integrations
try:
    from .lm_eval_integration import LMEvalConfig, LMEvalRunner, get_lm_eval_task
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseEvaluator:
    """
    Universal Evaluator.
    - Handles SFT/Generic Tasks via internal loop.
    - Handles Standard Benchmarks via lm-eval integration.
    - Automatically handles specialized logic for Code and Math tasks.
    """
    def __init__(
        self,
        metrics: Optional[List[Metric]] = None,
        batch_size: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_cache: bool = True,
        cache_dir: str = ".eval_cache",
        task_type: str = "generic",
        generation_kwargs: Optional[Dict[str, Any]] = None,  # Added: Accept generation args
        k_list: Optional[List[int]] = None, # Added: Support custom k values for Pass@K
        use_unsloth: bool = False # Added: Explicit Unsloth support
    ):
        """
        Args:
            metrics: List of metrics to evaluate. If None, chosen based on task_type.
            task_type: 'code', 'math', or 'generic'. Determines default metrics and data handling.
            generation_kwargs: Dict of arguments passed to model.generate (e.g. max_new_tokens, temperature)
            k_list: List of integers for Pass@K calculation (e.g., [1, 10]). Default [1].
            use_unsloth: If True, attempts to use model.fast_generate (if available) or optimized paths.
        """
        self.batch_size = batch_size
        self.device = device
        self.cache = EvaluationCache(cache_dir) if use_cache else None
        self.task_type = task_type.lower()
        self.generation_kwargs = generation_kwargs or {}
        self.use_unsloth = use_unsloth
        
        # Ensure generation kwargs support the requested k
        self.k_list = k_list or [1]
        max_k = max(self.k_list)
        if self.task_type == "code" and max_k > 1:
            current_num_return = self.generation_kwargs.get("num_return_sequences", 1)
            if current_num_return < max_k:
                logger.info(f"Boosting num_return_sequences to {max_k} to support Pass@{max_k}")
                self.generation_kwargs["num_return_sequences"] = max_k
                self.generation_kwargs["do_sample"] = True 

        # Smart Metric Initialization
        if metrics is None:
            if self.task_type == "code":
                logger.info(f"Task type 'code' detected. Using PassAtKMetric with k={self.k_list}.")
                self.metrics = [PassAtKMetric(k_list=self.k_list)]
            elif self.task_type == "math":
                logger.info("Task type 'math' detected. Using MathAccuracyMetric.")
                self.metrics = [MathAccuracyMetric()]
            elif self.task_type in ["text", "text_generation", "generation"]:
                logger.info("Task type 'text_generation' detected. Using Perplexity, ROUGE, and BLEU.")
                self.metrics = [PerplexityMetric(), RougeMetric(), BleuMetric()]
            else:
                self.metrics = [PerplexityMetric(), AccuracyMetric()]
        else:
            self.metrics = metrics

    def add_metric(self, metric: Metric):
        """Add a metric to the evaluator."""
        self.metrics.append(metric)

    def evaluate_benchmark(self, model_name_or_path: str, tasks: List[str]) -> Dict[str, Any]:
        if not LM_EVAL_AVAILABLE:
            logger.warning("lm-eval integration not available. Skipping benchmark evaluation.")
            return {}

        logger.info(f"Running benchmarks: {tasks}")
        config = LMEvalConfig(
            model_name=model_name_or_path,
            batch_size=self.batch_size,
            device=self.device
        )
        runner = LMEvalRunner(config)
        
        task_objects = [get_lm_eval_task(t) for t in tasks]
        results = runner.evaluate_tasks(task_objects)
        
        final_metrics = {}
        for res in results:
            for k, v in res.metrics.items():
                final_metrics[f"{res.task_name}/{k}"] = v
        
        return final_metrics

    def _prepare_batch(self, batch, tokenizer):
        if isinstance(batch, list):
            return tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
        if isinstance(batch, dict):
            return {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        return batch

    def _extract_column_data(self, batch: Dict, heuristics: List[str]) -> List[Any]:
        for col in heuristics:
            if col in batch:
                return batch[col]
        return []

    @staticmethod
    def _custom_collate_fn(batch):
        """
        Custom collate function to handle variable length fields (like test cases).
        - Stacks tensors (if possible)
        - Keeps lists/strings as lists of objects
        """
        if not batch:
            return {}
            
        elem = batch[0]
        collated = {}
        
        for key in elem:
            # Gather all values for this key
            values = [d[key] for d in batch]
            
            # If it's a tensor/number, try default collation
            if isinstance(values[0], (int, float, torch.Tensor)):
                try:
                    collated[key] = torch.utils.data.default_collate(values)
                    continue
                except RuntimeError:
                    # Fallback if dimensions mismatch (e.g. variable length sequences not padded)
                    pass
            
            # Default behavior for variable length items (strings, lists of test cases, etc):
            # Just keep them as a list
            collated[key] = values
            
        return collated

    def evaluate(
        self,
        model,
        tokenizer,
        dataset,
        task_name: Optional[str] = None,
        max_samples: Optional[int] = None,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Main evaluation loop for SFT/Generic tasks.
        """
        model.eval()
        if hasattr(model, "to"):
            model.to(self.device)
        
        # --- FIX: Ensure Correct Padding for Generation ---
        original_padding_side = tokenizer.padding_side
        
        is_encoder_decoder = False
        config = getattr(model, "config", None)
        if hasattr(model, "active_peft_config") or hasattr(model, "peft_config"):
             if hasattr(model, "get_base_model"):
                 base_model = model.get_base_model()
                 if hasattr(base_model, "config"):
                     config = base_model.config
             elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
                 config = model.base_model.config

        if config and hasattr(config, "is_encoder_decoder"):
            is_encoder_decoder = config.is_encoder_decoder
            
        if not is_encoder_decoder:
            if tokenizer.padding_side != 'left':
                tokenizer.padding_side = 'left'

        current_task = task_name or self.task_type
        
        if self.cache:
            cache_args = {"task": current_task, "max_samples": max_samples}
            if column_mapping:
                cache_args["column_mapping"] = str(sorted(column_mapping.items()))
                
            cached_result = self.cache.get(
                getattr(model, "name_or_path", "unknown_model"),
                getattr(dataset, "name", "unknown_dataset"),
                cache_args
            )
            if cached_result:
                logger.info("Loaded evaluation results from cache.")
                if "total" not in cached_result:
                    cached_result["total"] = len(dataset) if max_samples is None else min(len(dataset), max_samples)
                return cached_result

        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        if len(dataset) == 0:
            logger.warning("Empty dataset provided for evaluation.")
            return {}

        # Use custom collate to handle variable length columns (like test_cases)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=self._custom_collate_fn
        )
        
        all_predictions = []
        all_references = []
        all_losses = []

        sft_schema = TASK_SCHEMAS.get(SchemaTaskType.SFT)
        rl_schema = TASK_SCHEMAS.get(SchemaTaskType.GRPO)
        
        prompt_candidates = (
            list(sft_schema.column_heuristics.get("prompt", [])) + 
            list(rl_schema.column_heuristics.get("prompt", [])) + 
            ["prompt", "input", "question", "instruction"]
        )
        target_candidates = (
            list(sft_schema.column_heuristics.get("completion", [])) + 
            list(rl_schema.column_heuristics.get("response", [])) + 
            ["completion", "response", "answer", "target", "output", "label", "ground_truth", "answer_clean", "original_answer", "test_cases", "tests"]
        )
        
        prompt_keys = list(dict.fromkeys(prompt_candidates))
        target_keys = list(dict.fromkeys(target_candidates))

        if column_mapping:
            for key in ["prompt", "input", "instruction", "question"]:
                if key in column_mapping: 
                    prompt_keys.insert(0, column_mapping[key])
            for key in ["target", "output", "response", "answer", "completion"]:
                if key in column_mapping: 
                    target_keys.insert(0, column_mapping[key])

        is_code_task = (current_task == "code") or any(isinstance(m, PassAtKMetric) for m in self.metrics)

        logger.info(f"Starting evaluation on {len(dataset)} samples... (Task Type: {current_task})")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(batch, dict):
                    inputs = self._extract_column_data(batch, prompt_keys)
                    targets = self._extract_column_data(batch, target_keys)
                    
                    if not len(inputs) and 'input_ids' in batch:
                        # Only use raw input_ids if they were collated successfully (are tensor)
                        if isinstance(batch['input_ids'], torch.Tensor):
                            input_ids = batch['input_ids']
                            if hasattr(input_ids, 'to'): input_ids = input_ids.to('cpu')
                            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        else:
                            # If collation kept them as list (variable length), manual tokenize
                            inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

                    if not len(targets) and 'labels' in batch:
                        if isinstance(batch['labels'], torch.Tensor):
                            labels = batch['labels']
                            if hasattr(labels, 'to'): labels = labels.to('cpu')
                            clean_labels = []
                            for label_seq in labels:
                                valid_indices = label_seq[label_seq != -100]
                                clean_labels.append(tokenizer.decode(valid_indices, skip_special_tokens=True))
                            targets = clean_labels
                        # If list (variable length), assume already decoded or handle elsewhere
                else:
                    continue

                if not inputs:
                    continue
                
                if targets:
                    if is_code_task:
                        pass 
                    else:
                        targets = [str(t) if t is not None else "" for t in targets]

                # 1. Forward Pass (Perplexity) https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
                if isinstance(inputs[0], str):
                    if original_padding_side == 'right':
                         tokenizer.padding_side = 'right'
                         
                    # tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    # outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
                    # loss = outputs.loss.item() if outputs.loss is not None else 0.0
                    # # all_losses.extend([loss] * len(inputs))
                    # all_losses.extend([loss])
                        
                    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    
                    # Get input_ids and attention_mask
                    encoded_batch = tokenized_inputs["input_ids"]
                    attn_mask = tokenized_inputs["attention_mask"]
                    labels = encoded_batch
                    
                    # Forward pass
                    with torch.no_grad():
                        out_logits = model(encoded_batch, attention_mask=attn_mask).logits
                    
                    # Shift logits and labels (following HF implementation exactly)
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
                    
                    # Calculate per-sample perplexity using CrossEntropyLoss
                    from torch.nn import CrossEntropyLoss
                    loss_fct = CrossEntropyLoss(reduction="none")
                    
                    # Compute perplexity per sample (EXACT HF formula)
                    perplexity_batch = torch.exp(
                        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                        / shift_attention_mask_batch.sum(1)
                    )
                    
                    # Convert to losses (inverse of exp)
                    # loss_batch = torch.log(perplexity_batch)
                    
                    # ✅ Add per-sample losses
                    all_losses.extend(perplexity_batch.cpu().tolist())
                    
                    if not is_encoder_decoder:
                        tokenizer.padding_side = 'left'
                        
                elif 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
                    tokenized_inputs = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    # outputs = model(**tokenized_inputs, labels=tokenized_inputs["input_ids"])
                    # loss = outputs.loss.item() if outputs.loss is not None else 0.0
                    # all_losses.extend([loss] * len(inputs))
                    
                    encoded_batch = tokenized_inputs["input_ids"]
                    attn_mask = tokenized_inputs.get("attention_mask", torch.ones_like(encoded_batch))
                    labels = encoded_batch
                    
                    with torch.no_grad():
                        out_logits = model(encoded_batch, attention_mask=attn_mask).logits
                    
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
                    
                    from torch.nn import CrossEntropyLoss
                    loss_fct = CrossEntropyLoss(reduction="none")
                    
                    perplexity_batch = torch.exp(
                        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                        / shift_attention_mask_batch.sum(1)
                    )
                    
                    # loss_batch = torch.log(perplexity_batch)
                    all_losses.extend( perplexity_batch.cpu().tolist())

                # 2. Generation Pass
                if any(m.requires_generation for m in self.metrics):
                    if len(targets) > 0:
                        try:
                            if isinstance(inputs[0], str):
                                gen_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
                            else:
                                gen_inputs = tokenized_inputs

                            default_max = 512 if is_code_task else 100
                            gen_kwargs = self.generation_kwargs.copy()
                            if 'max_new_tokens' not in gen_kwargs:
                                gen_kwargs['max_new_tokens'] = default_max
                            
                            if self.use_unsloth and hasattr(model, "fast_generate"):
                                gen_outputs = model.fast_generate(
                                    **gen_inputs, 
                                    pad_token_id=tokenizer.pad_token_id,
                                    do_sample=False,
                                    **gen_kwargs 
                                )
                            else:
                                gen_outputs = model.generate(
                                    **gen_inputs, 
                                    pad_token_id=tokenizer.pad_token_id,
                                    do_sample=False,
                                    **gen_kwargs 
                                )
                                
                            decoded_preds = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
                            
                            clean_preds = []
                            num_return_sequences = gen_kwargs.get("num_return_sequences", 1)
                            
                            if num_return_sequences > 1:
                                for i, prompt in enumerate(inputs):
                                    start_idx = i * num_return_sequences
                                    end_idx = start_idx + num_return_sequences
                                    prompt_preds = decoded_preds[start_idx:end_idx]
                                    clean_prompt_preds = []
                                    for pred in prompt_preds:
                                        if isinstance(prompt, str) and pred.startswith(prompt):
                                            clean_prompt_preds.append(pred[len(prompt):].strip())
                                        else:
                                            clean_prompt_preds.append(pred.strip())
                                    all_predictions.append(clean_prompt_preds)
                                    all_references.append(targets[i])
                            else:
                                for prompt, pred in zip(inputs, decoded_preds):
                                    if isinstance(prompt, str) and pred.startswith(prompt):
                                        clean_preds.append(pred[len(prompt):].strip())
                                    else:
                                        clean_preds.append(pred.strip())
                                all_predictions.extend(clean_preds)
                                all_references.extend(targets)
                        except Exception as e:
                            logger.warning(f"Generation failed for batch: {e}")
        
        if original_padding_side != tokenizer.padding_side:
            tokenizer.padding_side = original_padding_side

        # After the evaluation loop, BEFORE metric computation:
        print(f"\nDEBUG: Collected data:")
        print(f"  all_losses: {len(all_losses)} items")
        print(f"  all_predictions: {len(all_predictions)} items")
        print(f"  all_references: {len(all_references)} items")
        if all_losses:
            print(f"  Sample losses: {all_losses[:3]}")

        results = {}
        for metric in self.metrics:
            print(f"\nDEBUG: Computing {metric.name}...")
            print(f"  requires_generation: {metric.requires_generation}")
            
            if metric.name == "perplexity":
                print(f"  Using all_losses: {len(all_losses)} values")
                res = metric.safe_compute(all_losses, [])
                print(f"  Result: {res}")
            elif metric.requires_generation:
                print(f"  Using predictions/references")
                res = metric.safe_compute(all_predictions, all_references)
                print(f"  Result: {res}")
            else:
                res = metric.safe_compute(all_predictions, all_references)
                print(f"  Result: {res}")
            
            results.update(res)
            print(f"  Updated results: {results}")

        # results = {}
        # for metric in self.metrics:
        #     if metric.name == "perplexity":
        #         res = metric.safe_compute(all_losses, [])
        #     elif metric.requires_generation:
        #         res = metric.safe_compute(all_predictions, all_references)
        #     else:
        #         res = metric.safe_compute(all_predictions, all_references)
            
        #     results.update(res)
        
        results["total"] = len(dataset)

        if self.cache:
            cache_args = {"task": current_task, "max_samples": max_samples}
            if column_mapping:
                cache_args["column_mapping"] = str(sorted(column_mapping.items()))

            self.cache.save(
                getattr(model, "name_or_path", "unknown_model"),
                getattr(dataset, "name", "unknown_dataset"),
                cache_args,
                results
            )

        return results