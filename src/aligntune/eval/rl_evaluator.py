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
from .metrics.generic import PerplexityMetric, completion_loss_totals
from .metrics.code import PassAtKMetric
from .metrics.text import RougeMetric, BleuMetric
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
        self.data_task_type = kwargs.pop('data_task_type', None)
        self.apply_chat_template = kwargs.pop('apply_chat_template', True)
        self.enable_thinking = kwargs.pop('enable_thinking', False)
        self.dpo_beta = kwargs.pop('dpo_beta', 0.1)

        if self.data_task_type == "sft" and not metrics_provided:
            kwargs['metrics'] = [PerplexityMetric(), RougeMetric(), BleuMetric()]
            metrics_provided = True
        
        # BaseEvaluator.__init__ will handle task_type, generation_kwargs, use_unsloth
        super().__init__(*args, **kwargs)
        
        # Only add default RL metrics if no explicit metrics list was provided
        if not metrics_provided:
            self.add_metric(KLDivergenceMetric())
            self.add_metric(RewardAccuracyMetric())
            self.add_metric(PolicyEntropyMetric())

    @staticmethod
    def _extract_sft_rows(batch: Dict[str, Any], prompts: List[Any], targets: List[Any]):
        """Build evaluation prompt histories and references from SFT conversations."""
        message_rows = batch.get("messages")
        if not message_rows:
            message_rows = [
                [
                    {"role": "user", "content": str(prompt)},
                    {"role": "assistant", "content": str(target)},
                ]
                for prompt, target in zip(prompts, targets)
            ]

        valid_prompts, valid_targets, valid_messages, valid_kwargs = [], [], [], []
        template_kwargs_rows = batch.get("chat_template_kwargs", [])
        for index, messages in enumerate(message_rows):
            if not isinstance(messages, list) or not messages:
                logger.warning("Skipping SFT evaluation row without messages")
                continue

            normalized = [
                message for message in messages
                if isinstance(message, dict)
                and message.get("role") in {"system", "user", "assistant"}
                and "content" in message
            ]
            if not normalized or normalized[-1].get("role") != "assistant":
                logger.warning(
                    "Skipping SFT evaluation row whose final message is not an assistant response"
                )
                continue

            history = normalized[:-1]
            if not history:
                logger.warning("Skipping SFT evaluation row without prompt history")
                continue

            valid_prompts.append(str(history[-1].get("content", "")))
            valid_targets.append(str(normalized[-1]["content"]))
            valid_messages.append(normalized)
            row_kwargs = (
                template_kwargs_rows[index]
                if index < len(template_kwargs_rows)
                else {}
            )
            valid_kwargs.append(row_kwargs if isinstance(row_kwargs, dict) else {})

        return valid_prompts, valid_targets, valid_messages, valid_kwargs

    def _format_generation_prompts(
        self,
        tokenizer,
        prompts: List[str],
        message_rows: Optional[List[Any]] = None,
        template_kwargs_rows: Optional[List[Any]] = None,
    ) -> List[str]:
        """Format generation inputs from conversation history or raw prompts."""
        if not self.apply_chat_template or not getattr(tokenizer, 'chat_template', None):
            return prompts

        formatted_prompts = []
        for index, prompt in enumerate(prompts):
            messages = None
            if message_rows is not None and index < len(message_rows):
                candidate = message_rows[index]
                if isinstance(candidate, list):
                    messages = [
                        message for message in candidate
                        if isinstance(message, dict)
                        and message.get("role") in {"system", "user", "assistant"}
                    ]

            if messages:
                # The final assistant turn is the reference completion. Keep
                # the system prompt and conversation history, but never feed
                # that answer to generation.
                if messages[-1].get("role") == "assistant":
                    messages = messages[:-1]
            else:
                messages = [{"role": "user", "content": prompt}]

            template_kwargs = {"enable_thinking": self.enable_thinking}
            if template_kwargs_rows is not None and index < len(template_kwargs_rows):
                row_kwargs = template_kwargs_rows[index]
                if isinstance(row_kwargs, dict):
                    template_kwargs.update(row_kwargs)

            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except TypeError:
                # Older tokenizers do not expose the Qwen-specific argument.
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted_prompts.append(formatted)
        return formatted_prompts

    @staticmethod
    def _prepare_dpo_batch(
        tokenizer,
        prompts: List[Any],
        chosen: List[Any],
        rejected: List[Any],
        template_kwargs_rows: Optional[List[Any]] = None,
    ) -> tuple[List[str], List[str], List[str]]:
        """Render canonical DPO rows into strings using TRL's own formatter."""
        from trl.data_utils import maybe_apply_chat_template

        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError("DPO prompt, chosen, and rejected batches must have equal lengths")

        rendered_prompts, rendered_chosen, rendered_rejected = [], [], []
        for index, (prompt, chosen_value, rejected_value) in enumerate(
            zip(prompts, chosen, rejected)
        ):
            row = {
                "prompt": prompt,
                "chosen": chosen_value,
                "rejected": rejected_value,
            }
            if template_kwargs_rows and index < len(template_kwargs_rows):
                row_kwargs = template_kwargs_rows[index]
                if isinstance(row_kwargs, dict):
                    row["chat_template_kwargs"] = row_kwargs

            rendered = maybe_apply_chat_template(row, tokenizer)
            values = (
                rendered.get("prompt"),
                rendered.get("chosen"),
                rendered.get("rejected"),
            )
            if not all(isinstance(value, str) for value in values):
                raise ValueError("TRL did not render DPO row to prompt/chosen/rejected strings")

            rendered_prompts.append(values[0])
            rendered_chosen.append(values[1])
            rendered_rejected.append(values[2])

        return rendered_prompts, rendered_chosen, rendered_rejected

    def _extract_column_data(self, batch: Dict, heuristics: List[str]) -> List[Any]:
        """Extract the first non-empty candidate value for each batch row."""
        candidate_columns = [batch[col] for col in heuristics if col in batch]
        if not candidate_columns:
            return []

        values = []
        for row_index in range(len(candidate_columns[0])):
            value = ""
            for column in candidate_columns:
                candidate = column[row_index]
                if candidate is not None and (
                    not isinstance(candidate, str) or candidate.strip()
                ):
                    value = candidate
                    break
            values.append(value)
        return values

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
        
        total_completion_nll = 0.0
        total_completion_tokens = 0
        kl_divs = []
        entropies = []
        reward_pairs = []
        dpo_reward_pairs = []
        policy_logprob_pairs = []
        policy_reference_pairs = []
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
        
        # DataManager's canonical prompt is complete; optional `input` fields
        # are often present but blank after normalization.
        prompt_keys = ["prompt", "instruction", "question", "input"] + list(schema.column_heuristics["prompt"])
        # GRPO's canonical reference field is named ``reference``. Keep the
        # generic output aliases as well because DataManager may normalize
        # SFT/preference data to ``completion`` or ``chosen``.
        target_keys = ["output", "response", "completion"] + list(
            schema.column_heuristics.get("reference", [])
        )

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

                    if self.data_task_type == "sft":
                        prompts, targets, message_rows, template_kwargs_rows = (
                            self._extract_sft_rows(batch, prompts, targets)
                        )
                    else:
                        message_rows = batch.get("messages")
                        template_kwargs_rows = batch.get("chat_template_kwargs")
                    
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

                    if self.data_task_type == "dpo":
                        prompts, chosen, rejected = self._prepare_dpo_batch(
                            tokenizer,
                            batch.get("prompt", []),
                            chosen,
                            rejected,
                            template_kwargs_rows,
                        )
                        targets = chosen
                        message_rows = None
                        template_kwargs_rows = None
                    
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

                    if self.data_task_type == "dpo":
                        # DPO rows are already rendered by TRL above.
                        model_prompts = prompts
                    elif isinstance(prompts[0], str):
                        model_prompts = self._format_generation_prompts(
                            tokenizer,
                            prompts,
                            message_rows=message_rows,
                            template_kwargs_rows=template_kwargs_rows,
                        )
                    inputs = tokenizer(
                        model_prompts, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)
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

                        # FIXED: Compute loss over completions only (same as evaluator.py)
                        if targets and len(targets) > 0 and isinstance(prompts[0], str):
                            # Concatenate prompt + completion
                            full_texts = [p + t for p, t in zip(model_prompts, targets)]

                            # Tokenize full sequences (batched)
                            full_enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                            # Get prompt lengths for masking
                            prompt_enc = tokenizer(model_prompts, return_tensors="pt", padding=True, truncation=True)
                            prompt_lens = prompt_enc.attention_mask.sum(1)

                            # Create labels with prompt tokens masked
                            labels = full_enc.input_ids.clone()
                            for i, prompt_len in enumerate(prompt_lens):
                                labels[i, :prompt_len] = -100  # Mask prompt tokens

                            # Forward pass (batched) with fallback for DynamicCache errors
                            try:
                                with torch.no_grad():
                                    out_logits = policy_model(full_enc.input_ids, attention_mask=full_enc.attention_mask).logits
                            except (AttributeError, TypeError, RuntimeError) as cache_err:
                                err_str = str(cache_err)
                                if "DynamicCache" in err_str or "seen_tokens" in err_str or "past_key_values" in err_str:
                                    with torch.no_grad():
                                        out_logits = policy_model(full_enc.input_ids, attention_mask=full_enc.attention_mask, use_cache=False).logits
                                else:
                                    raise cache_err

                            # Shift for next-token prediction
                            shift_logits = out_logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            shift_mask = full_enc.attention_mask[:, 1:].contiguous()

                            # PPL is token-weighted across the full evaluation
                            # corpus, not an equal-weight average of examples.
                            from torch.nn import CrossEntropyLoss
                            loss_fct = CrossEntropyLoss(reduction="none")
                            per_token_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)

                            # Mask padded and prompt tokens
                            loss_mask = (shift_labels != -100).float() * shift_mask
                            batch_nll, batch_tokens = completion_loss_totals(
                                per_token_loss, loss_mask
                            )
                            total_completion_nll += batch_nll
                            total_completion_tokens += batch_tokens

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
                            def get_batch_logprobs(prompts_list, responses_list, model):
                                """Compute log P(response | prompt) for entire batch."""
                                # Concatenate prompt + response
                                full_texts = [p + r for p, r in zip(prompts_list, responses_list)]

                                # Tokenize full sequences (batched)
                                full_enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                                # Get prompt lengths for masking
                                prompt_enc = tokenizer(prompts_list, return_tensors="pt", padding=True, truncation=True)
                                prompt_lens = prompt_enc.attention_mask.sum(1)

                                # Forward pass with fallback for DynamicCache errors
                                try:
                                    with torch.no_grad():
                                        logits = model(full_enc.input_ids, attention_mask=full_enc.attention_mask).logits
                                except (AttributeError, TypeError, RuntimeError) as cache_err:
                                    err_str = str(cache_err)
                                    if "DynamicCache" in err_str or "seen_tokens" in err_str or "past_key_values" in err_str:
                                        with torch.no_grad():
                                            logits = model(full_enc.input_ids, attention_mask=full_enc.attention_mask, use_cache=False).logits
                                    else:
                                        raise cache_err

                                # Shift for next-token prediction
                                shift_logits = logits[:, :-1, :].contiguous()
                                shift_labels = full_enc.input_ids[:, 1:].contiguous()

                                # Compute log probs (batched)
                                log_probs = F.log_softmax(shift_logits, dim=-1)
                                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                                # Shifted labels predict token i + 1, so the first
                                # response token is at prompt_length - 1.
                                positions = torch.arange(shift_labels.shape[1], device=self.device).unsqueeze(0)
                                prompt_lens_expanded = prompt_lens.unsqueeze(1).to(self.device)
                                response_mask = positions >= (prompt_lens_expanded - 1)

                                # Exclude right-padding tokens.
                                valid_mask = full_enc.attention_mask[:, 1:].bool()
                                final_mask = response_mask & valid_mask

                                # Sum log probs over response tokens only (batched)
                                sample_logprobs = (token_log_probs * final_mask.float()).sum(1)

                                return sample_logprobs.tolist()

                            # 1. Policy Scores
                            # Temporarily switch padding to right for loss/logprob calc
                            curr_pad = tokenizer.padding_side
                            tokenizer.padding_side = 'right'

                            policy_chosen_log = get_batch_logprobs(model_prompts, chosen, policy_model)
                            policy_rejected_log = get_batch_logprobs(model_prompts, rejected, policy_model)

                            batch_policy_pairs = list(
                                zip(policy_chosen_log, policy_rejected_log)
                            )
                            policy_logprob_pairs.extend(batch_policy_pairs)

                            # DPO implicit rewards require a reference model.
                            if reference_model:
                                ref_chosen_log = get_batch_logprobs(model_prompts, chosen, reference_model)
                                ref_rejected_log = get_batch_logprobs(model_prompts, rejected, reference_model)

                                for pc, pr, rc, rr in zip(
                                    policy_chosen_log,
                                    policy_rejected_log,
                                    ref_chosen_log,
                                    ref_rejected_log,
                                ):
                                    chosen_reward = self.dpo_beta * (pc - rc)
                                    rejected_reward = self.dpo_beta * (pr - rr)
                                    dpo_reward_pairs.append(
                                        (chosen_reward, rejected_reward)
                                    )
                                    policy_reference_pairs.append(
                                        ((pc, rc), (pr, rr))
                                    )
                            else:
                                # A policy-only ranking is useful when no reference
                                # model is available, but is not an implicit DPO reward.
                                dpo_reward_pairs.extend(batch_policy_pairs)

                            tokenizer.padding_side = curr_pad

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
            print(f"  completion tokens: {total_completion_tokens}")
            print(f"  all_predictions: {len(all_predictions)} items")
            print(f"  all_references: {len(all_references)} items")
            print(f"  kl_divs: {len(kl_divs)} items")
            print(f"  entropies: {len(entropies)} items")
            print(f"  reward_pairs: {len(reward_pairs)} items")
            
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
                if total_completion_tokens:
                    mean_loss = total_completion_nll / total_completion_tokens
                    results.update(metric.safe_compute([mean_loss], []))
                else:
                    results["perplexity"] = float('nan')
            
            elif metric.name in ["win_rate", "reward_margin", "preference_accuracy", "calibration"]:
                if dpo_reward_pairs:
                    results.update(metric.safe_compute(dpo_reward_pairs, []))
                else:
                    logger.warning(f"DPO metric '{metric.name}' requires chosen/rejected pairs")
            elif metric.name == "log_ratio":
                if policy_logprob_pairs:
                    results.update(metric.safe_compute(policy_logprob_pairs, []))
                else:
                    logger.warning("DPO metric 'log_ratio' requires chosen/rejected pairs")
            elif metric.name == "implicit_reward":
                if policy_reference_pairs:
                    results.update(metric.safe_compute(policy_reference_pairs, []))
                else:
                    logger.warning("DPO metric 'implicit_reward' requires a reference model")
            
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
