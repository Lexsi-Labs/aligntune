"""
RLEvaluator with Perplexity Support

This is a patched version of rl_evaluator.py that adds loss collection
for perplexity metric support.

CHANGES:
1. Added all_losses list initialization
2. Added forward pass for loss collection when perplexity metric is present
3. Added proper metric computation for perplexity
4. Added debug logging
5. Loss collection for perplexity
6. DPO metric support (Win Rate, Implicit Reward)
7. Fixes for 'DynamicCache' errors on Phi-3.5/Llama-3 models
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional, List, Any
import logging
import numpy as np

from .evaluator import BaseEvaluator
from .metrics.rl import KLDivergenceMetric, RewardAccuracyMetric, PolicyEntropyMetric
from .metrics.code import PassAtKMetric
from ..data.schemas import TASK_SCHEMAS, TaskType as SchemaTaskType

logger = logging.getLogger(__name__)

class RLEvaluator(BaseEvaluator):
    """
    Evaluator specifically for RL tasks (PPO, DPO, GRPO).
    Requires reference model and optionally a reward model.
    """
    
    def __init__(self, *args, **kwargs):
        # Check if metrics were explicitly provided
        metrics_provided = kwargs.get('metrics') is not None
        
        # BaseEvaluator.__init__ will handle task_type, generation_kwargs, use_unsloth
        super().__init__(*args, **kwargs)
        
        # Only add default RL metrics if no explicit metrics list was provided
        if not metrics_provided:
            self.add_metric(KLDivergenceMetric())
            self.add_metric(RewardAccuracyMetric())
            self.add_metric(PolicyEntropyMetric())

    def _extract_column_data(self, batch: Dict, heuristics: List[str]) -> List[Any]:
        """Helper to find data from the first matching column in the batch."""
        for col in heuristics:
            if col in batch:
                return batch[col]
        return []

    def evaluate_rl(
        self,
        policy_model,
        reference_model,
        tokenizer,
        dataset,
        reward_model=None,
        max_samples: Optional[int] = None,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Specialized RL evaluation loop with perplexity and DPO support.
        """
        policy_model.eval()
        if reference_model:
            reference_model.eval()
            reference_model.to(self.device)
        # policy_model.to(self.device) Not needed else some error will happen on ternminal 

        # --- FIX: Ensure Correct Padding for Generation (Decoder-Only Support) ---
        original_padding_side = tokenizer.padding_side
        
        is_encoder_decoder = False
        config = getattr(policy_model, "config", None)
        
        # Check for PEFT wrapper
        if hasattr(policy_model, "active_peft_config") or hasattr(policy_model, "peft_config"):
             if hasattr(policy_model, "get_base_model"):
                 base_model = policy_model.get_base_model()
                 if hasattr(base_model, "config"):
                     config = base_model.config
             elif hasattr(policy_model, "base_model") and hasattr(policy_model.base_model, "config"):
                 config = policy_model.base_model.config

        if config and hasattr(config, "is_encoder_decoder"):
            is_encoder_decoder = config.is_encoder_decoder
            
        # Decoder-only models require Left Padding for batched generation
        if not is_encoder_decoder:
            if tokenizer.padding_side != 'left':
                tokenizer.padding_side = 'left'

        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        # Use custom collate to handle variable length columns (like test_cases)
        # Inherited from BaseEvaluator
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            collate_fn=self._custom_collate_fn 
        )
        
        # ========== MODIFIED: Added all_losses ==========
        all_losses = []  # ← ADD THIS LINE
        kl_divs = []
        entropies = []
        reward_pairs = []
        all_predictions = []
        all_references = []
        all_queries = []
        
        # Identify which metrics require specific RL computations
        has_kl = any(m.name == "kl_divergence" for m in self.metrics)
        has_entropy = any(m.name == "policy_entropy" for m in self.metrics)
        has_reward_acc = any(m.name == "reward_accuracy" for m in self.metrics)
        # ========== MODIFIED: Added perplexity check ==========
        has_perplexity = any(m.name == "perplexity" for m in self.metrics)

        # Identify DPO metrics
        dpo_metric_names = ["win_rate", "reward_margin", "preference_accuracy", 
                            "calibration", "log_ratio", "implicit_reward"]
        has_dpo_metrics = any(m.name in dpo_metric_names for m in self.metrics)

        rl_metric_names = ["kl_divergence", "reward_accuracy", "policy_entropy"]
        generation_metrics = [m for m in self.metrics if m.name not in rl_metric_names and m.requires_generation]
        needs_generation = len(generation_metrics) > 0

        is_code_task = (self.task_type == "code") or any(isinstance(m, PassAtKMetric) for m in self.metrics)

        schema = TASK_SCHEMAS.get(SchemaTaskType.GRPO)
        
        # Priority: input/instruction (actual content) before prompt (often template text)
        prompt_keys = ["input", "instruction", "question", "prompt"] + list(schema.column_heuristics["prompt"])
        target_keys = ["output", "response", "completion"] + list(schema.column_heuristics["response"])

        if column_mapping:
            for key in ["prompt", "input", "instruction", "question"]:
                if key in column_mapping: prompt_keys.insert(0, column_mapping[key])
            for key in ["target", "output", "response", "answer", "completion"]:
                if key in column_mapping: target_keys.insert(0, column_mapping[key])

        #################################################################################
        # --- FIX: Ensure 'chosen' is checked for DPO datasets ---
        target_keys.extend(["chosen", "answer", "solution", "output", "target", "ground_truth", "label", "answer_clean", "original_answer", "test_cases", "tests"])

        logger.info(f"Starting RL evaluation...")
        logger.info(f"  Generation: {needs_generation}")
        logger.info(f"  Code Mode: {is_code_task}")
        logger.info(f"  DPO Metrics: {has_dpo_metrics}")
        if has_dpo_metrics:
            logger.info(f"  DPO metrics to compute: {dpo_metric_names}")

        first_batch = True  # Flag to control debug printing

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="RL Eval"):
                if isinstance(batch, dict):
                    # --- DEBUGGING START ---
                    if first_batch:
                        print(f"DEBUG: Batch Keys: {list(batch.keys())}")
                        print(f"DEBUG: Prompt Candidates: {prompt_keys}")
                        print(f"DEBUG: Target Candidates: {target_keys}")
                    # --- DEBUGGING END ---

                    prompts = self._extract_column_data(batch, prompt_keys)
                    targets = self._extract_column_data(batch, target_keys)
                    
                    if first_batch:
                        print(f"DEBUG: Extracted Prompts Count: {len(prompts)}")
                        print(f"DEBUG: Extracted Targets Count: {len(targets)}")
                        if len(prompts) > 0:
                            sample_prompt = str(prompts[0])
                            # Truncate long prompts/targets to avoid cluttering output
                            max_display_len = 200
                            if len(sample_prompt) > max_display_len:
                                print(f"DEBUG: Sample Prompt (truncated): {sample_prompt[:max_display_len]}...")
                            else:
                                print(f"DEBUG: Sample Prompt: {sample_prompt}")
                        if len(targets) > 0:
                            sample_target = str(targets[0])
                            max_display_len = 200
                            if len(sample_target) > max_display_len:
                                print(f"DEBUG: Sample Target (truncated): {sample_target[:max_display_len]}...")
                            else:
                                print(f"DEBUG: Sample Target: {sample_target}")
                        first_batch = False

                    chosen = batch.get('chosen', [])
                    rejected = batch.get('rejected', [])
                    
                    if not len(prompts) and 'input_ids' in batch:
                        # Only use raw input_ids if they were collated successfully (are tensor)
                        if isinstance(batch['input_ids'], torch.Tensor):
                            input_ids = batch['input_ids']
                            if hasattr(input_ids, 'to'): input_ids = input_ids.to('cpu')
                            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        else:
                            # If collation kept them as list (variable length), manual tokenize
                            prompts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

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

                if not prompts: 
                    # print("DEBUG: skipping batch due to no prompts") # Optional noisy debug
                    continue

                all_queries.extend(prompts)
                if targets:
                    if is_code_task:
                        pass
                    else:
                        targets = [str(t) if t is not None else "" for t in targets]

                if isinstance(prompts[0], str):
                    # Apply chat template if available (for instruct models)
                    # if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                    #     try:
                    #         formatted_prompts = []
                    #         for p in prompts:
                    #             messages = [{"role": "user", "content": p}]
                    #             formatted = tokenizer.apply_chat_template(
                    #                 messages, 
                    #                 tokenize=False, 
                    #                 add_generation_prompt=True
                    #             )
                    #             formatted_prompts.append(formatted)
                    #         inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    #     except Exception as e:
                    #         logger.debug(f"Chat template failed, using raw prompts: {e}")
                    #         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    # else:
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                elif 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
                     inputs = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                else:
                    continue

                # ========== NEW: Forward Pass for Perplexity with use_cache=False fallback========== ##############
                if has_perplexity:
                    try:
                        # Save current padding
                        temp_padding = tokenizer.padding_side
                        
                        # Use right padding for loss computation
                        tokenizer.padding_side = 'right'
                        
                        # Prepare inputs
                        if isinstance(prompts[0], str):
                            loss_inputs = tokenizer(
                                prompts, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True
                            ).to(self.device)
                        else:
                            loss_inputs = inputs
                        
                        # Forward pass - Fallback mechanism for DynamicCache errors
                        try:
                            outputs = policy_model(**loss_inputs, labels=loss_inputs["input_ids"])
                        except (AttributeError, TypeError, RuntimeError) as cache_err:
                            err_str = str(cache_err)
                            if "DynamicCache" in err_str or "seen_tokens" in err_str or "past_key_values" in err_str:
                                # Retry with cache disabled
                                outputs = policy_model(**loss_inputs, labels=loss_inputs["input_ids"], use_cache=False)
                            else:
                                raise cache_err
                        
                        # Collect loss
                        if outputs.loss is not None:
                            batch_loss = outputs.loss.item()
                            all_losses.extend([batch_loss] * len(prompts))
                        
                        # Restore padding
                        tokenizer.padding_side = temp_padding
                        
                    except Exception as e:
                        logger.warning(f"Loss collection failed for batch: {e}")
                # ========== END NEW SECTION ==========

                # RL Metrics (KL / Entropy) with Fallback
                if reference_model and (has_kl or has_entropy):
                    try:
                        def robust_forward(model, inp):
                            try:
                                return model(**inp)
                            except (AttributeError, TypeError, RuntimeError) as cache_err:
                                err_str = str(cache_err)
                                if "DynamicCache" in err_str or "seen_tokens" in err_str or "past_key_values" in err_str:
                                    return model(**inp, use_cache=False)
                                raise cache_err

                        policy_outputs = robust_forward(policy_model, inputs)
                        policy_logits = policy_outputs.logits
                        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                        
                        if has_kl:
                            ref_outputs = robust_forward(reference_model, inputs)
                            ref_logits = ref_outputs.logits
                            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                            kl = F.kl_div(policy_log_probs, ref_log_probs, log_target=True, reduction='none').sum(-1).mean().item()
                            kl_divs.append(kl)

                        if has_entropy:
                            probs = torch.exp(policy_log_probs)
                            entropy = -(probs * policy_log_probs).sum(-1).mean().item()
                            entropies.append(entropy)
                    except Exception as e:
                        logger.warning(f"KL/Entropy calc failed: {e}")

                # 3. --- FIX 2: DPO / Reward Logic ---###################
                # Run this if we have Explicit Reward metrics OR DPO metrics
                if (has_reward_acc or has_dpo_metrics) and len(chosen) > 0 and len(rejected) > 0:
                    try:
                        # Case A: Explicit Reward Model
                        if reward_model is not None:
                            if callable(reward_model) and not isinstance(reward_model, torch.nn.Module):
                                c_scores = [reward_model(c) for c in chosen]
                                r_scores = [reward_model(r) for r in rejected]
                            else:
                                c_inputs = tokenizer(chosen, return_tensors="pt", padding=True, truncation=True).to(self.device)
                                r_inputs = tokenizer(rejected, return_tensors="pt", padding=True, truncation=True).to(self.device)
                                c_scores = reward_model(**c_inputs).logits.squeeze(-1).tolist()
                                r_scores = reward_model(**r_inputs).logits.squeeze(-1).tolist()
                            reward_pairs.extend(list(zip(c_scores, r_scores)))
                        
                        # Case B: Implicit Reward (Policy Model LogProbs)
                        # Used when no external reward model is provided, common for DPO eval
                        else:
                            # Helper to compute logprobs for full sequences with Fallback
                            def get_batch_logprobs(texts, model):
                                enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                                
                                try:
                                    out = model(**enc)
                                except (AttributeError, TypeError, RuntimeError) as cache_err:
                                    err_str = str(cache_err)
                                    if "DynamicCache" in err_str or "seen_tokens" in err_str or "past_key_values" in err_str:
                                        out = model(**enc, use_cache=False)
                                    else:
                                        raise cache_err

                                logits = out.logits[:, :-1, :]
                                labels = enc.input_ids[:, 1:]
                                log_probs = F.log_softmax(logits, dim=-1)
                                token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
                                mask = enc.attention_mask[:, 1:]
                                # Sum log probs over valid tokens
                                return (token_log_probs * mask).sum(dim=-1).tolist()

                            # 1. Policy Scores
                            # Temporarily switch padding to right for loss/logprob calc if needed
                            curr_pad = tokenizer.padding_side
                            tokenizer.padding_side = 'right'
                            
                            policy_chosen_log = get_batch_logprobs(chosen, policy_model)
                            policy_rejected_log = get_batch_logprobs(rejected, policy_model)
                            
                            # 2. Reference Scores (Optional but recommended for DPO)
                            if reference_model:
                                ref_chosen_log = get_batch_logprobs(chosen, reference_model)
                                ref_rejected_log = get_batch_logprobs(rejected, reference_model)
                                beta = 0.1 # Default DPO beta
                                c_scores = [beta * (p - r) for p, r in zip(policy_chosen_log, ref_chosen_log)]
                                r_scores = [beta * (p - r) for p, r in zip(policy_rejected_log, ref_rejected_log)]
                            else:
                                # Fallback: Just use Policy LogProbs (Higher logprob = Preferred)
                                c_scores = policy_chosen_log
                                r_scores = policy_rejected_log
                                
                            tokenizer.padding_side = curr_pad
                            reward_pairs.extend(list(zip(c_scores, r_scores)))

                    except Exception as e:
                        # logger.debug(f"Reward/DPO computation failed: {e}")
                        logger.warning(f"Reward/DPO computation failed: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
                
                # ========== GENERATION (CRITICAL FIX) ==========
                if needs_generation:
                    if len(targets) > 0:
                        try:
                            default_max = 512 if is_code_task else 100
                            gen_kwargs = self.generation_kwargs.copy()
                            if 'max_new_tokens' not in gen_kwargs:
                                gen_kwargs['max_new_tokens'] = default_max
                            if 'do_sample' not in gen_kwargs:
                                gen_kwargs['do_sample'] = False
                            # Enable KV cache for faster generation
                            if 'use_cache' not in gen_kwargs:
                                gen_kwargs['use_cache'] = True

                            # Calculate Input Length for Slicing
                            input_ids_len = inputs['input_ids'].shape[1]

                            # Generate - ATTEMPT 1: Standard
                            try:
                              if self.use_unsloth and hasattr(policy_model, "fast_generate"):
                                  gen_outputs = policy_model.fast_generate(
                                      **inputs,
                                      pad_token_id=tokenizer.pad_token_id,
                                      **gen_kwargs
                                  )
                              else:
                                  gen_outputs = policy_model.generate(
                                      **inputs,
                                      pad_token_id=tokenizer.pad_token_id,
                                      **gen_kwargs
                                  )
                              # Generate - ATTEMPT 2: Fallback if cache error
                            except (AttributeError, TypeError, RuntimeError) as cache_error:
                                err_msg = str(cache_error)
                                if "DynamicCache" in err_msg or "seen_tokens" in err_msg or "past_key_values" in err_msg:
                                    logger.warning(f"Generation failed with cache error. Retrying with use_cache=False. Error: {err_msg}")
                                    gen_kwargs['use_cache'] = False
                                    gen_outputs = policy_model.generate(
                                        **inputs,
                                        pad_token_id=tokenizer.pad_token_id,
                                        **gen_kwargs
                                    )
                                else:
                                    raise cache_error
                                    
                            # --- CRITICAL FIX: Slice off the input tokens ---
                            # HF generate returns [input_ids + generated_tokens]
                            # We assume left-padding, so the suffix is the new content.
                            # Even if num_return_sequences > 1, the prefix length is consistent.
                            
                            # NOTE: If gen_outputs dim 1 is smaller than input_ids_len (rare), don't slice
                            if gen_outputs.shape[1] > input_ids_len:
                                generated_tokens = gen_outputs[:, input_ids_len:]
                            else:
                                generated_tokens = gen_outputs

                            # Decode ONLY the generated tokens
                            decoded_preds = tokenizer.batch_decode(
                                generated_tokens, 
                                skip_special_tokens=True
                            )
                            
                            # Handle Grouping (Pass@K > 1)
                            clean_preds = []
                            num_return_sequences = gen_kwargs.get("num_return_sequences", 1)
                            
                            if num_return_sequences > 1:
                                # Regroup flattened list
                                for i in range(len(prompts)):
                                    start_idx = i * num_return_sequences
                                    end_idx = start_idx + num_return_sequences
                                    # Get candidates for this specific prompt
                                    prompt_preds = decoded_preds[start_idx:end_idx]
                                    
                                    # Just strip whitespace, no need to check startsWith(prompt) 
                                    # because we already sliced the prompt off at tensor level.
                                    clean_prompt_preds = [p.strip() for p in prompt_preds]
                                    
                                    all_predictions.append(clean_prompt_preds)
                                    all_references.append(targets[i])
                            else:
                                # 1-to-1 mapping
                                for pred in decoded_preds:
                                    clean_preds.append(pred.strip())
                                all_predictions.extend(clean_preds)
                                all_references.extend(targets)

                        except Exception as e:
                            logger.warning(f"Generation failed for batch: {e}")
                    else:
                        all_references.extend([""] * len(prompts))
        
        if original_padding_side != tokenizer.padding_side:
            tokenizer.padding_side = original_padding_side

        # ========== Debug Stats (only if logger level is DEBUG) ==========
        if logger.isEnabledFor(logging.DEBUG):
            print(f"\n{'='*60}")
            print(f"DEBUG: Collected data")
            print(f"{'='*60}")
            print(f"  all_losses: {len(all_losses)} items")
            print(f"  all_predictions: {len(all_predictions)} items")
            print(f"  all_references: {len(all_references)} items")
            print(f"  kl_divs: {len(kl_divs)} items")
            print(f"  entropies: {len(entropies)} items")
            print(f"  reward_pairs: {len(reward_pairs)} items")
            
            if all_losses:
                print(f"  Sample losses: {all_losses[:3]}")
            if all_predictions:
                sample_pred = all_predictions[0]
                if isinstance(sample_pred, list):
                    pred_str = str(sample_pred[0])[:100] if len(sample_pred) > 0 else ""
                else:
                    pred_str = str(sample_pred)[:100]
                print(f"  Sample prediction [0]: {pred_str}...")
            if all_references:
                print(f"  Sample reference [0]: {str(all_references[0])[:100]}...")
            print(f"{'='*60}\n")

        results = {}
        for metric in self.metrics:
            # Existing RL metrics
            if metric.name == "kl_divergence" and kl_divs:
                results.update(metric.safe_compute(kl_divs, []))
            elif metric.name == "policy_entropy" and entropies:
                results.update(metric.safe_compute(entropies, []))
            elif metric.name == "reward_accuracy" and reward_pairs:
                results.update(metric.safe_compute(reward_pairs, []))
            elif metric.name == "perplexity":
                if all_losses:
                    results.update(metric.safe_compute(all_losses, []))
                else:
                    results["perplexity"] = float('nan')
            
            # ========== NEW: DPO metrics (uses same reward_pairs data) ==========
            elif metric.name in ["win_rate", "reward_margin", "preference_accuracy", 
                                  "calibration", "log_ratio", "implicit_reward"]:
                if reward_pairs:
                    logger.debug(f"Computing DPO metric '{metric.name}' with {len(reward_pairs)} pairs")
                    results.update(metric.safe_compute(reward_pairs, []))
                else:
                    logger.warning(f"DPO metric '{metric.name}' requires chosen/rejected pairs")
            # ========== END NEW ==========
            
            elif metric.requires_generation:
                results.update(metric.safe_compute(all_predictions, all_references))

        results["total"] = len(dataset)
        
        # Add sample predictions to results for display
        if all_predictions and len(all_predictions) > 0:
            # Store first few predictions for display
            num_samples = len(all_predictions)
            sample_preds = []
            sample_queries = []
            for i in range(num_samples):
                pred = all_predictions[i]
                if isinstance(pred, list):
                    # For Pass@K, take first candidate
                    sample_preds.append(pred[0] if len(pred) > 0 else "")
                else:
                    sample_preds.append(str(pred))
                
                if i < len(all_queries):
                    sample_queries.append(str(all_queries[i]))

            results["sample_predictions"] = sample_preds
            results["sample_queries"] = sample_queries
            
            # Also store corresponding references if available
            if all_references and len(all_references) >= num_samples:
                results["sample_references"] = [str(all_references[i]) for i in range(num_samples)]
        
        return results