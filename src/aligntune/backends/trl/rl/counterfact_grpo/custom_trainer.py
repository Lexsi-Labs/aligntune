"""
GRPO Training with Counterfactual Importance - v2
==================================================

Same as v2 improved, but uses COUNTERFACTUAL MASKING for importance.

Key changes from v1:
- Still uses soft weighting (no zero-sum constraint)
- Integrated with v2's modular structure
- Can combine with process rewards

Comparison point: Counterfactual vs Gradient vs Attention
"""

import re
import math
import argparse
import hashlib
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import nanmin, nanmax
from peft import LoraConfig, get_peft_model
import numpy as np


# =============================================================================
# EXTRA VERBOSE LOGGING FOR PAPER EXAMPLES
# =============================================================================

@dataclass
class SpanDetail:
    """Details about a detected span."""
    start_pos: int
    end_pos: int
    text: str
    span_type: str  # "arithmetic" or "sentence"
    baseline_score: float
    masked_score: float
    importance_drop: float

    def to_dict(self):
        return asdict(self)


@dataclass
class VerboseExample:
    """Complete example for paper with all credit allocation details."""
    # Metadata
    step: int
    sample_idx: int
    timestamp: str

    # Input/Output
    prompt_text: str
    completion_text: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    reward: float
    advantage: float

    # Span Detection
    spans: List[Dict]
    reasoning_region: Tuple[int, int]  # (start, end)
    answer_region: Tuple[int, int]  # (start, end) or None

    # Importance Computation
    baseline_answer_logprob: float
    importance_raw: List[float]  # Per-token raw importance
    importance_normalized: List[float]  # Per-token normalized [0,1]

    # Final Weights
    weights_final: List[float]  # Per-token final weights
    weight_stats: Dict[str, float]  # mean, std, min, max

    # Top tokens for easy inspection
    top_weighted_tokens: List[Dict]  # [{pos, token, weight, importance}, ...]
    bottom_weighted_tokens: List[Dict]

    # Token-level details (for deep analysis)
    tokens: List[str]  # All completion tokens

    def to_dict(self):
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d['reasoning_region'] = list(self.reasoning_region) if self.reasoning_region else None
        d['answer_region'] = list(self.answer_region) if self.answer_region else None
        return d


class VerboseLogger:
    """Logger for extra verbose mode - saves detailed examples to JSONL."""

    def __init__(self, log_path: str, run_name: str = None):
        self.log_path = log_path
        self.run_name = run_name or "unknown"
        self.examples_logged = 0

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

        # Write header info
        with open(log_path, 'w') as f:
            header = {
                "type": "header",
                "run_name": run_name,
                "created_at": datetime.now().isoformat(),
                "description": "Extra verbose counterfactual importance logs for paper examples"
            }
            f.write(json.dumps(header) + "\n")

        print(f"\n[VerboseLogger] Initialized - saving to {log_path}")

    def log_example(self, example: VerboseExample):
        """Append an example to the log file."""
        with open(self.log_path, 'a') as f:
            entry = {"type": "example", **example.to_dict()}
            f.write(json.dumps(entry, default=str) + "\n")
        self.examples_logged += 1

        if self.examples_logged % 50 == 0:
            print(f"[VerboseLogger] Logged {self.examples_logged} examples to {self.log_path}")

    def log_summary(self):
        """Write summary at end of training."""
        with open(self.log_path, 'a') as f:
            summary = {
                "type": "summary",
                "total_examples": self.examples_logged,
                "finished_at": datetime.now().isoformat()
            }
            f.write(json.dumps(summary) + "\n")
        print(f"\n[VerboseLogger] Finished - {self.examples_logged} examples saved to {self.log_path}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_chat_template_safe(tokenizer, messages: list, tokenize: bool = False,
                              add_generation_prompt: bool = True, enable_thinking: bool = False) -> str:
    """Apply chat template with optional enable_thinking support.

    Note: enable_thinking is passed via **kwargs in most tokenizers, so we always
    pass it. Tokenizers that don't use it will simply ignore it.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
    )


# =============================================================================
# PROCESS REWARD MODULE (same as v2)
# =============================================================================

def extract_calculation_steps(text: str) -> List[Tuple[str, str]]:
    """Extract intermediate calculations from reasoning text."""
    steps = []
    calc_pattern = r'(\d+(?:\.\d+)?)\s*([+\-×*/÷])\s*(\d+(?:\.\d+)?)\s*(?:=|is)\s*(\d+(?:\.\d+)?)'
    
    for match in re.finditer(calc_pattern, text.replace(',', '')):
        left, op, right, result = match.groups()
        expr = f"{left}{op}{right}"
        steps.append((expr, result))
    
    return steps


def verify_calculation(expr: str, claimed_result: str) -> bool:
    """Check if a calculation is correct."""
    try:
        parts = re.match(r'(\d+(?:\.\d+)?)([+\-×*/÷])(\d+(?:\.\d+)?)', expr)
        if not parts:
            return False
        
        left, op, right = parts.groups()
        left, right = float(left), float(right)
        claimed = float(claimed_result)
        
        op_map = {'+': left + right, '-': left - right, 
                  '×': left * right, '*': left * right,
                  '/': left / right if right != 0 else float('inf'),
                  '÷': left / right if right != 0 else float('inf')}
        
        if op not in op_map:
            return False
        
        actual = op_map[op]
        return math.isclose(actual, claimed, rel_tol=1e-4, abs_tol=1e-8)
    except:
        return False


def compute_process_reward(completion: str, final_answer: str, gold_answer: str) -> Dict[str, float]:
    """Compute reward with partial credit for intermediate steps."""
    pred = parse_pred_number(completion)
    final_correct = 1.0 if numeric_equal(pred, gold_answer) else 0.0
    
    steps = extract_calculation_steps(completion)
    if len(steps) == 0:
        process_score = 0.0
    else:
        correct_steps = sum(verify_calculation(expr, res) for expr, res in steps)
        process_score = correct_steps / len(steps)
    
    total_reward = 0.7 * final_correct + 0.3 * process_score
    
    return {
        'final_correct': final_correct,
        'process_score': process_score,
        'total_reward': total_reward,
        'num_steps': len(steps)
    }


# =============================================================================
# LIGHTWEIGHT SPAN DETECTION (for counterfactual)
# =============================================================================

def detect_arithmetic_spans(
    tokens: List[str],
    prompt_len: int,
    reasoning_end: int,
    max_spans: int = 10
) -> List[Tuple[int, int]]:
    """
    Detect fine-grained arithmetic calculations using regex patterns.
    Returns list of (start, end) tuples for spans like "7 × 7 = 49".

    FIX: Filters out LaTeX formatting spans (e.g., $...$ or \text{...}).
    """
    reasoning_text = ''.join(tokens[prompt_len:reasoning_end])

    # Pattern: number operator(s) number = result
    # Handles: "7 × 7 = 49", "12 + 24 + 72 = 108", etc.
    pattern = r'\d+(?:\.\d+)?(?:\s*[+\-×*/÷\\times\\div]\s*\d+(?:\.\d+)?)+\s*=\s*\d+(?:\.\d+)?'

    spans = []
    for match in re.finditer(pattern, reasoning_text):
        # Convert character positions to token positions
        char_start = match.start()
        char_end = match.end()

        # Find corresponding token indices
        curr_pos = 0
        token_start = None
        token_end = None

        for i in range(prompt_len, reasoning_end):
            token_len = len(tokens[i])
            if curr_pos <= char_start < curr_pos + token_len and token_start is None:
                token_start = i
            if curr_pos < char_end <= curr_pos + token_len and token_end is None:
                token_end = i + 1
                break
            curr_pos += token_len

        if token_start is not None and token_end is not None:
            # FIX 1: Filter LaTeX formatting
            # Skip spans that are mostly LaTeX commands (>30% backslashes)
            span_text = ''.join(tokens[token_start:token_end])
            if len(span_text) > 0:
                backslash_ratio = span_text.count('\\') / len(span_text)
                if backslash_ratio > 0.3:
                    continue  # Skip LaTeX-heavy spans

            spans.append((token_start, token_end))

        if len(spans) >= max_spans:
            break

    return spans


def detect_sentence_spans(
    tokens: List[str],
    prompt_len: int,
    reasoning_end: int,
    max_spans: int = 20,
    min_length: int = 5
) -> List[Tuple[int, int]]:
    """
    Detect sentence-level spans for broad reasoning coverage.
    Returns list of (start, end) tuples for complete sentences.

    FIX: Filters out LaTeX formatting spans.
    """
    spans = []
    sentence_start = prompt_len

    for i in range(prompt_len, reasoning_end):
        token = tokens[i].strip()

        # Sentence boundary markers
        is_boundary = False
        if token in {'.', '!', '?'}:
            is_boundary = True
        elif '\n\n' in tokens[i]:
            is_boundary = True
        elif token == '\n' and i + 1 < reasoning_end and tokens[i + 1].strip() == '\n':
            is_boundary = True

        if is_boundary:
            sentence_end = i + 1

            # Only keep sentences longer than min_length tokens
            if sentence_end - sentence_start >= min_length:
                # FIX 1: Filter LaTeX formatting
                # Skip spans that are mostly LaTeX commands (>30% backslashes)
                span_text = ''.join(tokens[sentence_start:sentence_end])
                if len(span_text) > 0:
                    backslash_ratio = span_text.count('\\') / len(span_text)
                    if backslash_ratio > 0.3:
                        sentence_start = sentence_end
                        continue  # Skip LaTeX-heavy spans

                spans.append((sentence_start, sentence_end))

            sentence_start = sentence_end

            if len(spans) >= max_spans:
                break

    # Handle last sentence if no terminator
    if sentence_start < reasoning_end and reasoning_end - sentence_start >= min_length:
        # FIX 1: Filter LaTeX formatting for last sentence too
        span_text = ''.join(tokens[sentence_start:reasoning_end])
        if len(span_text) > 0:
            backslash_ratio = span_text.count('\\') / len(span_text)
            if backslash_ratio <= 0.3:  # Only add if not LaTeX-heavy
                spans.append((sentence_start, reasoning_end))

    return spans


def merge_spans(
    arithmetic_spans: List[Tuple[int, int]],
    sentence_spans: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Merge arithmetic and sentence spans, removing overlaps.
    Priority: arithmetic spans (fine-grained) > sentence spans (broad).
    """
    # Start with arithmetic spans (higher priority)
    result = list(arithmetic_spans)

    # Add sentence spans that don't overlap with arithmetic
    for s_start, s_end in sentence_spans:
        overlaps = False
        for a_start, a_end in arithmetic_spans:
            # Check if they overlap
            if not (s_end <= a_start or s_start >= a_end):
                overlaps = True
                break

        if not overlaps:
            result.append((s_start, s_end))

    # Sort by start position
    result = sorted(result, key=lambda x: x[0])

    return result


def detect_calculation_spans(
    tokens: List[str],
    prompt_len: int,
    max_spans: int = 10,
    min_length: int = 5,
    max_length: int = 50,
    stop_at_padding: bool = True,
    pad_token: str = None
) -> List[Tuple[int, int]]:
    """
    HYBRID span detection: Combines arithmetic detection + sentence-level spans.

    Strategy:
    1. Detect fine-grained arithmetic: "7 × 7 = 49"
    2. Detect sentence-level spans: "Lynne bought 7 books..."
    3. Merge them (arithmetic takes priority)

    This captures both precise calculations AND logical reasoning.
    """
    # Find reasoning end (stop at padding/answer markers)
    reasoning_end = len(tokens)
    for i in range(prompt_len, len(tokens)):
        token = tokens[i].strip()

        if stop_at_padding and pad_token and tokens[i] == pad_token:
            reasoning_end = i
            break

        if token in ['Final', 'final'] and i + 1 < len(tokens):
            next_token = tokens[i + 1].strip()
            if next_token in ['Answer', 'answer', 'answer:']:
                reasoning_end = i
                break

        if token == '####':
            reasoning_end = i
            break

    # Detect both types of spans
    arithmetic_spans = detect_arithmetic_spans(tokens, prompt_len, reasoning_end, max_spans=max_spans//2)
    sentence_spans = detect_sentence_spans(tokens, prompt_len, reasoning_end, max_spans=max_spans, min_length=min_length)

    # Merge (arithmetic priority)
    merged = merge_spans(arithmetic_spans, sentence_spans)

    # Limit to max_spans and filter by max_length
    result = []
    for s, e in merged:
        if e - s <= max_length:
            result.append((s, e))
        if len(result) >= max_spans:
            break

    return result


# =============================================================================
# COUNTERFACTUAL IMPORTANCE
# =============================================================================

def compute_counterfactual_importance(
    model: AutoModelForCausalLM,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
    max_spans: int = 10,
    answer_tail: int = 5,
    debug: bool = False,
    random_mode: bool = False,
    invert_mode: bool = False
) -> np.ndarray:
    """
    Compute importance using counterfactual masking.

    Args:
        random_mode: Use random importance (control experiment)
        invert_mode: Invert importance scores (ablation study)

    For each detected span:
    1. Measure baseline probability of reasoning tokens (BEFORE "Final Answer")
    2. Mask the span and remeasure
    3. Drop = baseline - masked gives span importance
    4. Distribute importance uniformly over span tokens

    IMPORTANT: Only measures impact on REASONING region, not answer region.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    # Detect and skip left-padding tokens
    # Find where actual content starts (first non-padding token)
    pad_token_id = tokenizer.pad_token_id if tokenizer else None
    if pad_token_id is None and tokenizer:
        pad_token_id = tokenizer.eos_token_id  # Common fallback for padding

    actual_prompt_start = 0
    if pad_token_id is not None:
        input_list = input_ids[0].tolist()
        for i, tok_id in enumerate(input_list):
            if tok_id != pad_token_id:
                actual_prompt_start = i
                break

    # Adjust prompt_len to be relative to actual content
    left_padding_len = actual_prompt_start
    effective_prompt_len = prompt_len  # Keep original for return array sizing

    if left_padding_len > 0 and debug:
        print(f"[Counterfactual] Detected {left_padding_len} left-padding tokens, skipping them")
        print(f"[Counterfactual] Actual prompt starts at position {actual_prompt_start}")

    # RANDOM MODE: Use random importance (control experiment)
    if random_mode:
        importance = np.random.uniform(0, 1, size=seq_len)
        importance[:prompt_len] = 0.0  # Zero out prompt
        if debug:
            print("[Counterfactual] Using RANDOM importance (control experiment)")
            reasoning_imp = importance[prompt_len:]
            print(f"  Random importance range: [{reasoning_imp.min():.4f}, {reasoning_imp.max():.4f}]")
            print(f"  Random importance mean: {reasoning_imp.mean():.4f}")
        return importance

    # Start with baseline importance for ALL tokens (not zero!)
    # This ensures logical reasoning text also gets credit, not just arithmetic
    importance = np.ones(seq_len) * 0.2

    # FIRST: Find answer marker to determine reasoning region
    answer_marker_pos = None

    if tokenizer is not None:
        gen_ids = input_ids[0, prompt_len:].tolist()
        pad_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}

        # Clean padding first - find where actual content ends
        clean_gen_ids = []
        padding_start_idx = None
        for idx, token_id in enumerate(gen_ids):
            if token_id in pad_ids:
                padding_start_idx = idx
                break
            clean_gen_ids.append(token_id)

        # Default reasoning_end: stop at padding, not seq_len
        if padding_start_idx is not None:
            reasoning_end = prompt_len + padding_start_idx
        else:
            reasoning_end = prompt_len + len(gen_ids)  # No padding, use all generation

        full_text = tokenizer.decode(clean_gen_ids) if clean_gen_ids else ""

        if debug:
            # Print full prompt + generation
            prompt_text = tokenizer.decode(input_ids[0, :prompt_len].tolist())
            print(f"[Counterfactual] Sequence length: {seq_len}, Prompt length: {prompt_len}, Generation length: {len(gen_ids)}")
            print(f"[Counterfactual] Clean generation length (before padding): {len(clean_gen_ids)}")
            if padding_start_idx is not None:
                print(f"[Counterfactual] Padding starts at generation index {padding_start_idx} (absolute pos {prompt_len + padding_start_idx})")
            print(f"[Counterfactual] Full prompt:\n{prompt_text}")
            print(f"[Counterfactual] Full generation:\n{full_text}")

        # Look for answer marker in the clean text (case-insensitive, normalized)
        # Normalize: lowercase, strip extra whitespace
        full_text_normalized = ' '.join(full_text.lower().split())
        for marker in ["final answer", "####"]:
            marker_normalized = ' '.join(marker.lower().split())
            if marker_normalized in full_text_normalized:
                # Find the actual position in the original (non-normalized) text
                marker_pos = full_text.lower().find(marker.lower())

                # Convert text position to token position
                current_pos = 0
                for i, token_id in enumerate(clean_gen_ids):
                    token_text = tokenizer.decode([token_id])
                    if current_pos <= marker_pos < current_pos + len(token_text):
                        answer_marker_pos = prompt_len + i
                        reasoning_end = answer_marker_pos
                        if debug:
                            actual_marker = full_text[marker_pos:marker_pos+len(marker)]
                            print(f"[Counterfactual] Found '{actual_marker}' at position {answer_marker_pos}")
                            print(f"[Counterfactual] Reasoning region: [{prompt_len}, {reasoning_end}) = {reasoning_end - prompt_len} reasoning tokens")
                        break
                    current_pos += len(token_text)

                if answer_marker_pos is not None:
                    break

        # Fallback: look for \boxed{ which typically marks the final answer
        if answer_marker_pos is None and "\\boxed{" in full_text:
            boxed_pos = full_text.find("\\boxed{")
            # Find the token position that corresponds to this text position
            # Approximate by scanning tokens
            if debug:
                print(f"[Counterfactual] Looking for '\\boxed{{' as fallback marker")
            current_pos = 0
            for i, token_id in enumerate(clean_gen_ids):
                token_text = tokenizer.decode([token_id])
                if current_pos <= boxed_pos < current_pos + len(token_text):
                    answer_marker_pos = prompt_len + i
                    reasoning_end = answer_marker_pos
                    if debug:
                        print(f"[Counterfactual] Found '\\boxed{{' at position {answer_marker_pos}")
                        print(f"[Counterfactual] Reasoning region: [{prompt_len}, {reasoning_end}) = {reasoning_end - prompt_len} reasoning tokens")
                    break
                current_pos += len(token_text)

        if debug and answer_marker_pos is None:
            print(f"[Counterfactual] No answer marker found, using clean generation (excluding padding)")
            print(f"[Counterfactual] Reasoning region: [{prompt_len}, {reasoning_end}) = {reasoning_end - prompt_len} reasoning tokens")
    else:
        # No tokenizer, use default
        reasoning_end = seq_len - 1

    # BOUNDS CHECK: Ensure reasoning_end doesn't exceed sequence length
    reasoning_end = min(reasoning_end, seq_len - 1)
    if reasoning_end <= prompt_len:
        if debug:
            print(f"[Counterfactual] WARNING: reasoning_end ({reasoning_end}) <= prompt_len ({prompt_len}), returning zero importance")
        return importance

    # Get tokens
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    # Get pad token for span detection
    pad_token = tokenizer.decode([tokenizer.pad_token_id]) if tokenizer.pad_token_id else None 

    # Detect spans ONLY in reasoning region (will stop at padding and "Final Answer")
    spans = detect_calculation_spans(tokens, prompt_len, max_spans=max_spans, pad_token=pad_token)

    # Filter out spans that extend into answer region or exceed sequence length
    spans = [(s, e) for s, e in spans if e <= reasoning_end and e <= seq_len and s >= prompt_len]

    if len(spans) == 0:
        if debug:
            print("[Counterfactual] No spans detected in reasoning region, returning uniform importance")
        importance[prompt_len:reasoning_end] = 1.0
        return importance

    # Check if we have an answer region
    num_answer_tokens = seq_len - 1 - reasoning_end
    if num_answer_tokens <= 0:
        if debug:
            print("[Counterfactual] No answer tokens found! Returning ZERO importance (no reweighting)")
        # FIX 2: Return zeros instead of ones when no answer region
        # This prevents boosting all tokens uniformly (mean weight ~2.0)
        importance = np.zeros(seq_len)
        importance[:prompt_len] = 0.0
        return importance

    # Compute baseline score over ANSWER region (not reasoning region!)
    # We want to know: which reasoning spans impact the final answer probability?
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits[0]
        logp = F.log_softmax(logits, dim=-1)

        baseline_score = 0.0
        vocab_size = logp.shape[-1]
        # Measure probability of ANSWER tokens (from reasoning_end to seq_len)
        for t in range(reasoning_end, min(seq_len - 1, logp.shape[0] - 1)):
            next_token = input_ids[0, t + 1].item()
            # Bounds check: skip if token ID exceeds vocabulary
            if 0 <= next_token < vocab_size:
                baseline_score += logp[t, next_token].item()

    if debug:
        print(f"\n[Counterfactual] Detected {len(spans)} spans in reasoning region")
        print(f"[Counterfactual] Measuring impact on ANSWER tokens [{reasoning_end}, {seq_len})")
        print(f"[Counterfactual] Baseline score: {baseline_score:.4f}")

    # Mask each span in REASONING region
    mask_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for idx, (s, e) in enumerate(spans):
        # Bounds check for span indices
        s = max(0, min(s, seq_len - 1))
        e = max(s, min(e, seq_len))
        masked_ids = input_ids.clone()
        masked_ids[0, s:e] = mask_token

        with torch.no_grad():
            outputs = model(masked_ids, return_dict=True)
            logits = outputs.logits[0]
            logp = F.log_softmax(logits, dim=-1)

            masked_score = 0.0
            masked_vocab_size = logp.shape[-1]
            # Measure probability of ANSWER tokens (same region)
            for t in range(reasoning_end, min(seq_len - 1, logp.shape[0] - 1)):
                next_token = input_ids[0, t + 1].item()
                # Bounds check: skip if token ID exceeds vocabulary
                if 0 <= next_token < masked_vocab_size:
                    masked_score += logp[t, next_token].item()

        # Drop = how much worse the answer probability got when we masked this span
        # masked_score is MORE negative (worse) → drop should be POSITIVE (important)
        drop = baseline_score - masked_score  # Positive when masking hurts performance

        # Assign drop normalized by span length
        span_len = e - s
        if span_len > 0:
            importance[s:e] += drop / span_len

        if debug:
            text = ''.join(tokens[s:e])[:50]
            span_len = e - s
            print(f"  Span {idx+1} [{s:3d}, {e:3d}) len={span_len:2d} | Drop: {drop:+.4f} | {text}")

    # Zero out prompt and everything after reasoning region
    importance[:prompt_len] = 0.0
    importance[reasoning_end:] = 0.0

    # Normalize ONLY within reasoning region, preserving baseline distinction
    reasoning_importance = importance[prompt_len:reasoning_end]
    if len(reasoning_importance) > 0:
        # Baseline is the minimum (tokens not in any span)
        # Tokens in spans have baseline + drop
        baseline = reasoning_importance.min()
        max_importance = reasoning_importance.max()

        if debug:
            print(f"\n[Normalization] Raw importance range: [{baseline:.4f}, {max_importance:.4f}]")

        if max_importance > baseline:
            # Map [baseline, max] to [0.5, 1.0] instead of [0, 1]
            # This preserves the baseline distinction
            span_importance = reasoning_importance - baseline  # Extra importance beyond baseline
            max_extra = max_importance - baseline

            normalized_extra = span_importance / max_extra  # [0, 1]
            importance[prompt_len:reasoning_end] = 0.5 + 0.5 * normalized_extra  # [0.5, 1.0]
        else:
            # No spans had extra importance (all tokens have same importance)
            importance[prompt_len:reasoning_end] = 0.5

    if debug:
        # Print stats ONLY for reasoning region (not answer/padding which are zeroed)
        reasoning_importance = importance[prompt_len:reasoning_end]
        print(f"\n[Counterfactual] Reasoning region importance:")
        print(f"  Range: [{reasoning_importance.min():.4f}, {reasoning_importance.max():.4f}]")
        print(f"  Mean: {reasoning_importance.mean():.4f}")
        print(f"  Std: {reasoning_importance.std():.4f}")

        # Sanity Check 1.2: Importance Distribution + Histogram
        nonzero_importance = reasoning_importance[reasoning_importance > 0.1]  # Filter out near-zero
        if len(nonzero_importance) > 0:
            unique_values = len(np.unique(np.round(nonzero_importance, 2)))
            importance_spread = nonzero_importance.max() - nonzero_importance.min()

            if unique_values > 10 and importance_spread > 0.3:
                print(f"  ✓ PASS: Varied importance distribution ({unique_values} unique values, spread={importance_spread:.2f})")
            elif importance_spread < 0.1:
                print(f"  ⚠️  WARNING: All tokens have similar importance (spread={importance_spread:.2f})")
            else:
                print(f"  ℹ️  INFO: Limited variation ({unique_values} unique values, spread={importance_spread:.2f})")

            # ASCII Histogram with 10 bins
            print(f"\n  Importance Histogram (10 bins):")
            min_val = reasoning_importance.min()
            max_val = reasoning_importance.max()

            if max_val > min_val:
                # Create 10 bins
                bins = np.linspace(min_val, max_val, 11)
                hist, _ = np.histogram(reasoning_importance, bins=bins)

                # Find max count for scaling
                max_count = hist.max()
                bar_width = 40  # Max bar width in characters

                for i in range(10):
                    bin_start = bins[i]
                    bin_end = bins[i + 1]
                    count = hist[i]

                    # Scale bar length
                    if max_count > 0:
                        bar_len = int((count / max_count) * bar_width)
                    else:
                        bar_len = 0

                    bar = '█' * bar_len
                    pct = (count / len(reasoning_importance)) * 100 if len(reasoning_importance) > 0 else 0

                    print(f"  [{bin_start:.2f}, {bin_end:.2f}): {bar} {count:4d} ({pct:5.1f}%)")

        # Show top 10 most important tokens (ONLY from reasoning region, exclude zeros)
        gen_importance = importance[prompt_len:]
        gen_ids = input_ids[0, prompt_len:].tolist()

        # Filter to only non-zero importance tokens
        nonzero_indices = [(i, gen_importance[i]) for i in range(len(gen_importance)) if gen_importance[i] > 0]

        if len(nonzero_indices) > 0:
            # Sort by importance score
            nonzero_indices.sort(key=lambda x: x[1], reverse=True)
            top_k = min(10, len(nonzero_indices))

            print(f"\n[Top {top_k} Most Important Tokens (from {len(nonzero_indices)} reasoning tokens)]")

            # Sanity Check 1.3: Manual Inspection
            formatting_tokens = ['#', '##', '###', '####', 'Step', ':', '=', '-', '*']
            format_count = 0
            number_count = 0

            for rank in range(top_k):
                idx, score = nonzero_indices[rank]
                token = tokenizer.decode([gen_ids[idx]])
                global_pos = prompt_len + idx
                print(f"  {rank+1}. Pos {global_pos:3d} | Score: {score:.4f} | Token: '{token}'")

                # Count formatting vs meaningful tokens
                token_stripped = token.strip()
                if token_stripped in formatting_tokens:
                    format_count += 1
                elif any(c.isdigit() for c in token_stripped):
                    number_count += 1

            # Report sanity check
            if format_count >= top_k * 0.7:  # 70%+ formatting
                print(f"  ⚠️  WARNING: {format_count}/{top_k} top tokens are formatting (measuring brittleness, not reasoning!)")
            elif number_count >= 3 or format_count <= 2:
                print(f"  ✓ PASS: Top tokens include meaningful content ({number_count} numbers, {format_count} formatting)")
            else:
                print(f"  ℹ️  INFO: {number_count} numbers, {format_count} formatting in top-{top_k}")
        else:
            print(f"\n[Top Tokens] No reasoning tokens with non-zero importance found!")

    # INVERT MODE: Flip importance scores (ablation study)
    if invert_mode:
        reasoning_importance = importance[prompt_len:]
        if len(reasoning_importance) > 0:
            # Invert: high becomes low, low becomes high
            max_imp = reasoning_importance.max()
            min_imp = reasoning_importance.min()
            if max_imp > min_imp:
                # Normalize to [0, 1], then invert, then scale back
                normalized = (reasoning_importance - min_imp) / (max_imp - min_imp)
                inverted = 1.0 - normalized
                importance[prompt_len:] = min_imp + inverted * (max_imp - min_imp)
            if debug:
                print(f"\n[Counterfactual] INVERTED importance (ablation study)")
                print(f"  New range: [{importance[prompt_len:].min():.4f}, {importance[prompt_len:].max():.4f}]")

    return importance


# =============================================================================
# SOFT WEIGHTING (same as v2)
# =============================================================================

def compute_soft_weights(
    importance: np.ndarray,
    prompt_len: int,
    boost_factor: float = 2.0,
    min_weight: float = 0.5,
    answer_weight: float = 1.5,
    enable_gradient_conservation: bool = True,
    tokenizer = None,
    input_ids = None,
    debug: bool = False
) -> np.ndarray:
    """
    Convert importance to soft weights WITH optional gradient conservation.

    Args:
        enable_gradient_conservation: If True, normalize weights to mean=1.0 (gradient conservation).
                                       If False, weights can have mean > 1.0 (effectively boosts LR).

    Strategy:
    1. Map importance [0, 1] to weights [min_weight, boost_factor]
    2. Set answer region to answer_weight (e.g., 1.5x)
    3. Normalize so sum(weights) = num_tokens (gradient conservation!) - OPTIONAL

    This ensures we're REDISTRIBUTING gradient, not just boosting learning rate.
    """
    weights = np.ones_like(importance)

    gen_importance = importance[prompt_len:].copy()
    if len(gen_importance) == 0:
        return weights

    # Optionally downweight formatting tokens
    if tokenizer is not None and input_ids is not None:
        gen_ids = input_ids[0, prompt_len:].tolist()

        for i, token_id in enumerate(gen_ids):
            if i < len(gen_importance) and gen_importance[i] > 0:
                token_text = tokenizer.decode([token_id]).strip()
                if token_text in ['#', '##', '###', '####', '=', '==', '===', '====', '-', '--', '---']:
                    gen_importance[i] *= 0.1

    # STEP 1: Map importance to weights
    gen_weights = min_weight + (boost_factor - min_weight) * gen_importance
    weights[prompt_len:] = gen_weights

    # STEP 2: Find answer and padding regions
    answer_marker_idx = None
    pad_start_idx = None

    if tokenizer is not None and input_ids is not None:
        gen_ids = input_ids[0, prompt_len:].tolist()
        pad_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}

        # Find padding start
        for i, token_id in enumerate(gen_ids):
            if token_id in pad_ids:
                pad_start_idx = i
                break

        # Find answer marker
        clean_gen_ids = gen_ids[:pad_start_idx] if pad_start_idx else gen_ids
        full_text = tokenizer.decode(clean_gen_ids) if clean_gen_ids else ""

        for marker in ["Final Answer", "final answer", "####"]:
            if marker in full_text:
                marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
                for i in range(len(gen_ids) - len(marker_tokens) + 1):
                    if gen_ids[i:i+len(marker_tokens)] == marker_tokens:
                        answer_marker_idx = i
                        break
                if answer_marker_idx is not None:
                    break

        # STEP 3: Set answer region to answer_weight
        if answer_marker_idx is not None:
            answer_end = pad_start_idx if pad_start_idx else len(gen_ids)
            for i in range(answer_marker_idx, answer_end):
                weights[prompt_len + i] = answer_weight

        # STEP 4: Set padding to 1.0 (excluded from normalization)
        if pad_start_idx is not None:
            for i in range(pad_start_idx, len(gen_ids)):
                weights[prompt_len + i] = 1.0

    # STEP 5: GRADIENT CONSERVATION (OPTIONAL) - normalize non-padding weights
    # We want: sum(weights_non_padding) = num_non_padding_tokens
    gen_weights = weights[prompt_len:]

    # Capture BEFORE normalization stats for debug
    if pad_start_idx is not None:
        non_padding_weights_before = gen_weights[:pad_start_idx].copy()
    else:
        non_padding_weights_before = gen_weights.copy()

    mean_before = non_padding_weights_before.mean() if len(non_padding_weights_before) > 0 else 0.0
    sum_before = non_padding_weights_before.sum() if len(non_padding_weights_before) > 0 else 0.0

    # Perform normalization (if enabled)
    if enable_gradient_conservation:
        if pad_start_idx is not None:
            # Only normalize non-padding region
            non_padding_weights = gen_weights[:pad_start_idx]
            if len(non_padding_weights) > 0:
                current_sum = non_padding_weights.sum()
                target_sum = len(non_padding_weights)  # Should sum to N

                if current_sum > 0:
                    scale_factor = target_sum / current_sum
                    weights[prompt_len:prompt_len+pad_start_idx] *= scale_factor
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
        else:
            # No padding, normalize entire generation
            if len(gen_weights) > 0:
                current_sum = gen_weights.sum()
                target_sum = len(gen_weights)

                if current_sum > 0:
                    scale_factor = target_sum / current_sum
                    weights[prompt_len:] *= scale_factor
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
    else:
        # Gradient conservation disabled - skip normalization
        scale_factor = 1.0

    # DEBUG OUTPUT - AFTER normalization so we see the final weights
    if debug:
        gen_weights_final = weights[prompt_len:]

        # Separate padding from non-padding for accurate stats
        if pad_start_idx is not None:
            non_padding_weights = gen_weights_final[:pad_start_idx]
            padding_weights = gen_weights_final[pad_start_idx:]
        else:
            non_padding_weights = gen_weights_final
            padding_weights = np.array([])

        mean_after = non_padding_weights.mean() if len(non_padding_weights) > 0 else 0.0
        sum_after = non_padding_weights.sum() if len(non_padding_weights) > 0 else 0.0

        if enable_gradient_conservation:
            print(f"\n[Gradient Conservation Normalization]")
            print(f"  BEFORE: mean={mean_before:.4f}, sum={sum_before:.2f}")
            print(f"  Scale factor: {scale_factor:.4f}")
            print(f"  AFTER:  mean={mean_after:.4f}, sum={sum_after:.2f} (target: {len(non_padding_weights)})")
        else:
            print(f"\n[Gradient Conservation: DISABLED]")
            print(f"  Mean weight: {mean_after:.4f} (unnormalized)")

        conservation_status = "ON" if enable_gradient_conservation else "OFF"
        print(f"\n[Soft Weighting - Conservation {conservation_status}]")
        print(f"  Boost factor: {boost_factor}x, Min weight: {min_weight}x, Answer weight: {answer_weight}x")
        print(f"  Non-padding tokens: {len(non_padding_weights)}, Padding tokens: {len(padding_weights)}")

        if len(non_padding_weights) > 0:
            print(f"  Weight range (non-padding): [{non_padding_weights.min():.4f}, {non_padding_weights.max():.4f}]")
            print(f"  Weight std (non-padding): {non_padding_weights.std():.4f}")

            # Sanity Check 1.1: Gradient Conservation
            if enable_gradient_conservation:
                if 0.98 <= mean_after <= 1.02:
                    print(f"  ✓ PASS: Gradient conservation (mean = {mean_after:.4f} ≈ 1.0)")
                else:
                    print(f"  ⚠️  ERROR: Gradient NOT conserved! Mean = {mean_after:.4f} (should be ≈1.0)")
            else:
                print(f"  ℹ️  INFO: Gradient conservation disabled (mean = {mean_after:.4f})")

        # Show top 10 most weighted tokens (ONLY from reasoning region, exclude neutral weights)
        if tokenizer is not None and input_ids is not None:
            gen_ids = input_ids[0, prompt_len:].tolist()

            # Filter to tokens with non-neutral weights (weight != 1.0)
            # These are the tokens that will actually affect training
            weighted_indices = [(i, gen_weights_final[i], importance[prompt_len + i])
                               for i in range(len(gen_weights_final))
                               if gen_weights_final[i] != 1.0]

            if len(weighted_indices) > 0:
                # Sort by weight (descending)
                weighted_indices.sort(key=lambda x: x[1], reverse=True)
                top_k = min(10, len(weighted_indices))

                print(f"\n[Top {top_k} Most Weighted Tokens (from {len(weighted_indices)} reasoning tokens)]")
                for rank in range(top_k):
                    idx, weight, imp = weighted_indices[rank]
                    token = tokenizer.decode([gen_ids[idx]])
                    global_pos = prompt_len + idx
                    print(f"  {rank+1}. Pos {global_pos:3d} | Weight: {weight:.4f} | Importance: {imp:.4f} | Token: '{token}'")
            else:
                print(f"\n[Top Weighted Tokens] No reasoning tokens with non-neutral weights found!")

    return weights


# =============================================================================
# DETAILED IMPORTANCE COMPUTATION FOR VERBOSE LOGGING
# =============================================================================

def compute_counterfactual_importance_detailed(
    model: AutoModelForCausalLM,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
    max_spans: int = 10,
) -> Dict[str, Any]:
    """
    Compute counterfactual importance with FULL details for verbose logging.
    Returns everything needed to understand and visualize the credit allocation.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    result = {
        "spans": [],
        "baseline_score": 0.0,
        "importance_raw": [],
        "importance_normalized": [],
        "reasoning_region": (prompt_len, seq_len),
        "answer_region": None,
        "tokens": [],
    }

    # Decode all tokens for logging
    all_tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    result["tokens"] = all_tokens[prompt_len:]  # Only completion tokens

    # Find answer marker and padding
    gen_ids = input_ids[0, prompt_len:].tolist()
    pad_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}

    padding_start_idx = None
    for idx, token_id in enumerate(gen_ids):
        if token_id in pad_ids:
            padding_start_idx = idx
            break

    reasoning_end = prompt_len + (padding_start_idx if padding_start_idx else len(gen_ids))
    reasoning_end = min(reasoning_end, seq_len - 1)

    # Find answer marker
    clean_gen_ids = gen_ids[:padding_start_idx] if padding_start_idx else gen_ids
    full_text = tokenizer.decode(clean_gen_ids) if clean_gen_ids else ""

    answer_marker_pos = None
    for marker in ["final answer", "####", "\\boxed{"]:
        marker_pos = full_text.lower().find(marker.lower())
        if marker_pos >= 0:
            current_pos = 0
            for i, token_id in enumerate(clean_gen_ids):
                token_text = tokenizer.decode([token_id])
                if current_pos <= marker_pos < current_pos + len(token_text):
                    answer_marker_pos = prompt_len + i
                    reasoning_end = answer_marker_pos
                    break
                current_pos += len(token_text)
            break

    result["reasoning_region"] = (prompt_len, reasoning_end)
    if answer_marker_pos:
        answer_end = prompt_len + (padding_start_idx if padding_start_idx else len(gen_ids))
        result["answer_region"] = (answer_marker_pos, answer_end)

    # Initialize importance
    importance = np.ones(seq_len) * 0.2

    # Check if we have answer tokens
    num_answer_tokens = seq_len - 1 - reasoning_end
    if num_answer_tokens <= 0 or reasoning_end <= prompt_len:
        result["importance_raw"] = importance[prompt_len:].tolist()
        result["importance_normalized"] = importance[prompt_len:].tolist()
        return result

    # Get tokens and detect spans
    pad_token = tokenizer.decode([tokenizer.pad_token_id]) if tokenizer.pad_token_id else None

    # Detect arithmetic spans
    arithmetic_spans = detect_arithmetic_spans(all_tokens, prompt_len, reasoning_end, max_spans=max_spans//2)
    # Detect sentence spans
    sentence_spans = detect_sentence_spans(all_tokens, prompt_len, reasoning_end, max_spans=max_spans)
    # Merge
    all_spans = merge_spans(arithmetic_spans, sentence_spans)
    all_spans = [(s, e) for s, e in all_spans if e <= reasoning_end and e <= seq_len and s >= prompt_len][:max_spans]

    if len(all_spans) == 0:
        importance[prompt_len:reasoning_end] = 1.0
        result["importance_raw"] = importance[prompt_len:].tolist()
        result["importance_normalized"] = importance[prompt_len:].tolist()
        return result

    # Compute baseline score
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits[0]
        logp = F.log_softmax(logits, dim=-1)

        baseline_score = 0.0
        vocab_size = logp.shape[-1]
        for t in range(reasoning_end, min(seq_len - 1, logp.shape[0] - 1)):
            next_token = input_ids[0, t + 1].item()
            if 0 <= next_token < vocab_size:
                baseline_score += logp[t, next_token].item()

    result["baseline_score"] = baseline_score

    # Mask each span and measure impact
    mask_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for s, e in all_spans:
        masked_ids = input_ids.clone()
        masked_ids[0, s:e] = mask_token

        with torch.no_grad():
            outputs = model(masked_ids, return_dict=True)
            logits = outputs.logits[0]
            logp = F.log_softmax(logits, dim=-1)

            masked_score = 0.0
            for t in range(reasoning_end, min(seq_len - 1, logp.shape[0] - 1)):
                next_token = input_ids[0, t + 1].item()
                if 0 <= next_token < vocab_size:
                    masked_score += logp[t, next_token].item()

        drop = baseline_score - masked_score
        span_len = e - s
        if span_len > 0:
            importance[s:e] += drop / span_len

        # Determine span type
        span_text = ''.join(all_tokens[s:e])
        is_arithmetic = bool(re.search(r'\d+\s*[+\-×*/÷\\times\\div]\s*\d+', span_text))
        span_type = "arithmetic" if is_arithmetic else "sentence"

        result["spans"].append({
            "start_pos": s - prompt_len,  # Relative to completion start
            "end_pos": e - prompt_len,
            "text": span_text,
            "span_type": span_type,
            "baseline_score": baseline_score,
            "masked_score": masked_score,
            "importance_drop": drop
        })

    # Store raw importance
    result["importance_raw"] = importance[prompt_len:].tolist()

    # Normalize
    importance[:prompt_len] = 0.0
    importance[reasoning_end:] = 0.0

    reasoning_importance = importance[prompt_len:reasoning_end]
    if len(reasoning_importance) > 0:
        baseline = reasoning_importance.min()
        max_importance = reasoning_importance.max()
        if max_importance > baseline:
            span_importance = reasoning_importance - baseline
            max_extra = max_importance - baseline
            normalized_extra = span_importance / max_extra
            importance[prompt_len:reasoning_end] = 0.5 + 0.5 * normalized_extra
        else:
            importance[prompt_len:reasoning_end] = 0.5

    result["importance_normalized"] = importance[prompt_len:].tolist()

    return result


# =============================================================================
# OPTIMIZATION HELPERS FOR CF IMPORTANCE
# =============================================================================

def compute_completion_hash(input_ids: torch.Tensor) -> str:
    """Compute a hash of the completion for deduplication."""
    # Use first 8 bytes of SHA256 for speed
    id_bytes = input_ids.cpu().numpy().tobytes()
    return hashlib.sha256(id_bytes).hexdigest()[:16]


def compute_batched_counterfactual_importance(
    model: AutoModelForCausalLM,
    tokenizer,
    input_ids_batch: torch.Tensor,  # [batch_size, seq_len]
    prompt_len: int,
    max_spans: int = 10,
    debug: bool = False,
    completion_cache: Dict[str, np.ndarray] = None,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """
    Compute counterfactual importance for a batch with optimizations:
    1. Hash-based deduplication: skip recomputation for identical completions
    2. Batched forward passes: compute all baseline + masked sequences together

    Returns:
        importance_list: List of importance arrays for each sample
        stats: Dict with cache_hits, cache_misses, batched_forwards counts
    """
    device = next(model.parameters()).device
    input_ids_batch = input_ids_batch.to(device)
    batch_size, seq_len = input_ids_batch.shape

    stats = {"cache_hits": 0, "cache_misses": 0, "batched_forwards": 0, "skipped_uniform": 0}
    importance_list = []

    # Step 1: Check cache and identify unique completions
    unique_indices = []  # indices that need computation
    hash_to_idx = {}  # map hash -> first index with that hash
    sample_hashes = []

    for b in range(batch_size):
        comp_hash = compute_completion_hash(input_ids_batch[b])
        sample_hashes.append(comp_hash)

        if completion_cache is not None and comp_hash in completion_cache:
            # Cache hit - reuse cached importance
            importance_list.append(completion_cache[comp_hash])
            stats["cache_hits"] += 1
        elif comp_hash in hash_to_idx:
            # Duplicate in batch - will copy from first occurrence
            importance_list.append(None)  # Placeholder, will be filled later
            stats["cache_hits"] += 1
        else:
            # New completion - needs computation
            hash_to_idx[comp_hash] = b
            unique_indices.append(b)
            importance_list.append(None)
            stats["cache_misses"] += 1

    if not unique_indices:
        # All cache hits!
        return importance_list, stats

    # Step 2: Compute spans for unique samples
    # Collect all (sample_idx, spans, reasoning_end) for batched processing
    sample_spans_info = []
    for b in unique_indices:
        input_ids = input_ids_batch[b:b+1]

        # Quick span detection (no forward pass needed)
        tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
        pad_token = tokenizer.decode([tokenizer.pad_token_id]) if tokenizer.pad_token_id else None

        # Find reasoning end
        gen_ids = input_ids[0, prompt_len:].tolist()
        pad_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        padding_start_idx = None
        for idx, token_id in enumerate(gen_ids):
            if token_id in pad_ids:
                padding_start_idx = idx
                break
        reasoning_end = prompt_len + padding_start_idx if padding_start_idx else prompt_len + len(gen_ids)
        reasoning_end = min(reasoning_end, seq_len - 1)

        # Find answer marker
        clean_gen_ids = gen_ids[:padding_start_idx] if padding_start_idx else gen_ids
        full_text = tokenizer.decode(clean_gen_ids) if clean_gen_ids else ""
        for marker in ["final answer", "####", "\\boxed{"]:
            marker_pos = full_text.lower().find(marker.lower())
            if marker_pos >= 0:
                current_pos = 0
                for i, token_id in enumerate(clean_gen_ids):
                    token_text = tokenizer.decode([token_id])
                    if current_pos <= marker_pos < current_pos + len(token_text):
                        reasoning_end = min(prompt_len + i, reasoning_end)
                        break
                    current_pos += len(token_text)
                break

        spans = detect_calculation_spans(tokens, prompt_len, max_spans=max_spans, pad_token=pad_token)
        spans = [(s, e) for s, e in spans if e <= reasoning_end and e <= seq_len and s >= prompt_len]

        sample_spans_info.append({
            "idx": b,
            "spans": spans,
            "reasoning_end": reasoning_end,
            "input_ids": input_ids
        })

    # Step 3: Batched forward pass for all baseline + masked sequences
    # Collect all sequences that need forward pass
    all_sequences = []
    sequence_map = []  # (sample_idx, span_idx or -1 for baseline)

    for info in sample_spans_info:
        b = info["idx"]
        input_ids = info["input_ids"]
        spans = info["spans"]
        reasoning_end = info["reasoning_end"]

        # Check if we have answer tokens
        num_answer_tokens = seq_len - 1 - reasoning_end
        if num_answer_tokens <= 0 or len(spans) == 0:
            # No answer region or no spans - use uniform importance
            importance = np.ones(seq_len) * 0.5
            importance[:prompt_len] = 0.0
            importance[reasoning_end:] = 0.0
            importance_list[b] = importance
            if completion_cache is not None:
                completion_cache[sample_hashes[b]] = importance
            stats["skipped_uniform"] += 1
            continue

        # Add baseline sequence
        all_sequences.append(input_ids[0])
        sequence_map.append((b, -1, reasoning_end))  # -1 = baseline

        # Add masked sequences for each span
        mask_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for span_idx, (s, e) in enumerate(spans):
            masked_ids = input_ids[0].clone()
            masked_ids[s:e] = mask_token
            all_sequences.append(masked_ids)
            sequence_map.append((b, span_idx, reasoning_end))

    if not all_sequences:
        # All samples had no spans/answer region
        return importance_list, stats

    # Step 4: Chunked batched forward pass with IMMEDIATE score computation
    # CRITICAL: Don't store logp tensors - they're huge (seq_len × vocab_size each)
    # Instead, compute scores immediately and only keep scalar scores
    MAX_BATCH_FOR_CF = 8  # Can be higher since we don't store logp tensors
    sample_scores = {}  # sample_idx -> {"baseline": score, "masked": {span_idx: score}}

    with torch.no_grad():
        for chunk_start in range(0, len(all_sequences), MAX_BATCH_FOR_CF):
            chunk_end = min(chunk_start + MAX_BATCH_FOR_CF, len(all_sequences))
            chunk_sequences = all_sequences[chunk_start:chunk_end]
            chunk_map = sequence_map[chunk_start:chunk_end]
            batched_input = torch.stack(chunk_sequences, dim=0)

            outputs = model(batched_input, return_dict=True)
            chunk_logits = outputs.logits
            chunk_logp = F.log_softmax(chunk_logits, dim=-1)
            stats["batched_forwards"] += 1

            # Compute scores IMMEDIATELY for this chunk (don't store logp!)
            vocab_size = chunk_logp.shape[-1]
            for local_idx, (sample_idx, span_idx, reasoning_end) in enumerate(chunk_map):
                logp = chunk_logp[local_idx]
                score = 0.0
                for t in range(reasoning_end, min(seq_len - 1, logp.shape[0] - 1)):
                    next_token = input_ids_batch[sample_idx, t + 1].item()
                    if 0 <= next_token < vocab_size:
                        score += logp[t, next_token].item()
                # Store only scalar scores
                if sample_idx not in sample_scores:
                    sample_scores[sample_idx] = {"baseline": None, "masked": {}}
                if span_idx == -1:
                    sample_scores[sample_idx]["baseline"] = score
                else:
                    sample_scores[sample_idx]["masked"][span_idx] = score

            del outputs, chunk_logits, chunk_logp, batched_input, chunk_sequences
            torch.cuda.empty_cache()

    # Step 6: Convert scores to importance arrays
    for info in sample_spans_info:
        b = info["idx"]
        if importance_list[b] is not None:
            continue  # Already computed (skipped uniform)

        spans = info["spans"]
        reasoning_end = info["reasoning_end"]

        if b not in sample_scores:
            continue

        baseline_score = sample_scores[b]["baseline"]
        masked_scores = sample_scores[b]["masked"]

        # Initialize importance
        importance = np.ones(seq_len) * 0.2

        # Compute drops for each span
        for span_idx, (s, e) in enumerate(spans):
            if span_idx in masked_scores:
                drop = baseline_score - masked_scores[span_idx]
                span_len = e - s
                if span_len > 0:
                    importance[s:e] += drop / span_len

        # Zero out prompt and post-reasoning
        importance[:prompt_len] = 0.0
        importance[reasoning_end:] = 0.0

        # Normalize
        reasoning_importance = importance[prompt_len:reasoning_end]
        if len(reasoning_importance) > 0:
            baseline = reasoning_importance.min()
            max_importance = reasoning_importance.max()
            if max_importance > baseline:
                span_importance = reasoning_importance - baseline
                max_extra = max_importance - baseline
                normalized_extra = span_importance / max_extra
                importance[prompt_len:reasoning_end] = 0.5 + 0.5 * normalized_extra
            else:
                importance[prompt_len:reasoning_end] = 0.5

        importance_list[b] = importance

        # Update cache
        if completion_cache is not None:
            completion_cache[sample_hashes[b]] = importance

    # Step 7: Fill in duplicates from first occurrence
    for b in range(batch_size):
        if importance_list[b] is None:
            first_idx = hash_to_idx.get(sample_hashes[b])
            if first_idx is not None and importance_list[first_idx] is not None:
                importance_list[b] = importance_list[first_idx]
            else:
                # Fallback to uniform
                importance_list[b] = np.ones(seq_len) * 0.5
                importance_list[b][:prompt_len] = 0.0

    return importance_list, stats


# =============================================================================
# CUSTOM GRPO TRAINER (Counterfactual)
# =============================================================================

class CounterfactualGRPOTrainer(GRPOTrainer):
    """GRPO with counterfactual importance."""

    def __init__(
        self,
        *args,
        boost_factor: float = 2.0,
        min_weight: float = 0.5,
        max_spans: int = 10,
        weight_debug: bool = False,
        answer_weight: float = 1.5,
        weighting_mode: str = "counterfactual",  # counterfactual, random, inverted, vanilla
        # Legacy params (deprecated, use weighting_mode instead)
        method_name: str = None,
        random_importance: bool = False,
        invert_importance: bool = False,
        enable_gradient_conservation: bool = True,
        # Extra verbose mode for paper examples
        extra_verbose: bool = False,
        extra_verbose_log_path: str = None,
        extra_verbose_sample_rate: float = 0.1,  # Log 10% of samples by default
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Handle legacy params -> weighting_mode conversion
        if method_name == "baseline" or (boost_factor == 1.0 and min_weight == 1.0):
            weighting_mode = "vanilla"
        elif random_importance:
            weighting_mode = "random"
        elif invert_importance:
            weighting_mode = "inverted"

        self.weighting_mode = weighting_mode
        self.boost_factor = boost_factor
        self.min_weight = min_weight
        self.max_spans = max_spans
        self.weight_debug = weight_debug
        self.answer_weight = answer_weight
        self.enable_gradient_conservation = enable_gradient_conservation
        self.step_count = 0
        self.total_samples_processed = 0

        # Extra verbose logging for paper examples
        self.extra_verbose = extra_verbose
        self.extra_verbose_sample_rate = extra_verbose_sample_rate
        self._verbose_logger = None
        if extra_verbose:
            log_path = extra_verbose_log_path or f"verbose_logs/{self.args.run_name or 'run'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self._verbose_logger = VerboseLogger(log_path, run_name=self.args.run_name)

        # Optimization: Importance cache for deduplication (cleared each step)
        self._importance_cache = {}
        self._optimization_stats = {"cache_hits": 0, "cache_misses": 0, "skipped_uniform": 0}

        # Derive internal flags from weighting_mode
        self.random_importance = (weighting_mode == "random")
        self.invert_importance = (weighting_mode == "inverted")
        self.is_vanilla = (weighting_mode == "vanilla")
        # For backward compat
        self.method_name = "baseline" if self.is_vanilla else weighting_mode

        print(f"\n{'='*100}")
        print(f"COUNTERFACTUAL GRPO TRAINER - Weighting: {weighting_mode.upper()}")
        print(f"{'='*100}")
        if not self.is_vanilla:
            print(f"Boost factor: {boost_factor}x")
            print(f"Min weight: {min_weight}x")
            print(f"Answer weight: {answer_weight}x")
            print(f"Max spans: {max_spans}")
            print(f"Gradient conservation: {enable_gradient_conservation}")
        else:
            print(f"(All tokens weighted equally - vanilla GRPO)")
        print(f"Debug: {weight_debug}")
        if extra_verbose:
            print(f"Extra Verbose: ENABLED (sample rate: {extra_verbose_sample_rate*100:.0f}%)")
            print(f"Verbose Log: {self._verbose_logger.log_path if self._verbose_logger else 'N/A'}")
        print(f"{'='*100}\n")
    
    def _compute_loss_impl(self, model, inputs):
        """Override with counterfactual weighting - named _impl to avoid Unsloth patching."""
        self.step_count += 1

        # ===== Track training samples =====
        # Formula: n_samples_per_step = batch_size_per_gpu * grad_accum * n_gpus / n_generations
        # This gives unique prompts processed per optimizer step
        num_generations = getattr(self.args, 'num_generations', 1)
        grad_accum = getattr(self.args, 'gradient_accumulation_steps', 1)
        batch_size_per_gpu = getattr(self.args, 'per_device_train_batch_size', 16)
        n_gpus = self.accelerator.num_processes if hasattr(self, 'accelerator') else 1

        # Samples per optimizer step (unique prompts)
        samples_per_step = (batch_size_per_gpu * grad_accum * n_gpus) // num_generations

        # Use actual optimizer step from trainer state (self.state.global_step)
        # Note: global_step increments AFTER the optimizer step, so during compute_loss it shows the current step
        optimizer_step = self.state.global_step + 1  # +1 because we're in the middle of this step

        # Total unique prompts processed so far
        self.total_samples_processed = optimizer_step * samples_per_step

        # ===== Standard GRPO forward pass =====
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        # Handle Unsloth's 3D tensor format - Unsloth returns hidden states without gradients
        # We need to do our own forward pass to get logits with proper gradient tracking
        if per_token_logps.dim() == 3:
            if not hasattr(self, '_unsloth_warning_shown'):
                print("\n[INFO] Running forward pass for per-token log probs (counterfactual weighting needs gradients)")
                self._unsloth_warning_shown = True

            # Do our own forward pass to get logits with gradient tracking
            # This bypasses Unsloth's optimization but ensures gradients flow properly
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # [batch, seq, vocab]

            # Take only the completion tokens (last logits_to_keep positions)
            # logits[i] predicts token[i+1], so for completion tokens we need logits[-logits_to_keep-1:-1]
            completion_logits = logits[:, -(logits_to_keep+1):-1, :]  # [batch, completion_len, vocab]

            # Compute log softmax
            log_probs = torch.nn.functional.log_softmax(completion_logits.float(), dim=-1)

            # Gather the log probs for actual completion tokens
            per_token_logps = torch.gather(log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

            # Compute entropy from our completion_logits
            probs = torch.nn.functional.softmax(completion_logits.float(), dim=-1)
            entropies = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)  # [batch, completion_len]

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # ===== Counterfactual Importance Weighting (OPTIMIZED) =====
        batch_size = input_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        token_weights = torch.ones_like(completion_mask, dtype=torch.float32)

        # Skip importance computation for baseline (uniform weights = much faster!)
        if self.method_name == "baseline":
            # Print every 5 optimizer steps for baseline (only on last micro-batch of each step)
            if optimizer_step % 5 == 0 or optimizer_step <= 2:
                if self.step_count % grad_accum == 0:  # Only print once per optimizer step
                    print(f"\n{'='*80}")
                    print(f"[BASELINE MODE - Step {optimizer_step}]")
                    print(f"  Using uniform weights (all tokens weighted equally)")
                    print(f"  Samples processed: {self.total_samples_processed}")
                    print(f"{'='*80}")
        else:
            # OPTIMIZATION #4: Skip CF for uniform advantage groups (0%/100% accuracy)
            # If all advantages in the batch are identical, CF weighting has no effect
            adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            skip_cf_uniform = adv_std < 1e-6  # All advantages are essentially equal

            if skip_cf_uniform:
                self._optimization_stats["skipped_uniform"] += batch_size
                if optimizer_step % 5 == 0 or optimizer_step <= 2:
                    if self.step_count % grad_accum == 0:
                        print(f"\n{'='*80}")
                        print(f"[{self.method_name.upper()} MODE - Step {optimizer_step}] SKIPPED (uniform advantages)")
                        print(f"  Advantage std: {adv_std:.6f} - all samples have same reward")
                        print(f"  Samples processed: {self.total_samples_processed}")
                        print(f"  Optimization stats: {self._optimization_stats}")
                        print(f"{'='*80}")
            else:
                # Compute importance for non-baseline methods
                # Print method info every 5 optimizer steps (only on last micro-batch of each step)
                if optimizer_step % 5 == 0 or optimizer_step <= 2:
                    if self.step_count % grad_accum == 0:  # Only print once per optimizer step
                        print(f"\n{'='*80}")
                        print(f"[{self.method_name.upper()} MODE - Step {optimizer_step}] OPTIMIZED")
                        print(f"  Computing token importance weights (batched)...")
                        print(f"  Boost factor: {self.boost_factor}, Min weight: {self.min_weight}")
                        print(f"  Answer weight: {self.answer_weight}")
                        print(f"  Samples processed: {self.total_samples_processed}")

                # Clear cache at start of each optimizer step (not micro-batch)
                if self.step_count % grad_accum == 1:
                    self._importance_cache.clear()

                # OPTIMIZATION #1 & #2: Use batched computation with caching
                if batch_size > 0 and not self.random_importance and not self.invert_importance:
                    # Use optimized batched function
                    importance_list, stats = compute_batched_counterfactual_importance(
                        model,
                        self.processing_class,
                        input_ids,
                        prompt_len,
                        max_spans=self.max_spans,
                        debug=self.weight_debug,
                        completion_cache=self._importance_cache,
                    )

                    # Update optimization stats
                    self._optimization_stats["cache_hits"] += stats["cache_hits"]
                    self._optimization_stats["cache_misses"] += stats["cache_misses"]
                    self._optimization_stats["skipped_uniform"] += stats["skipped_uniform"]

                    # Apply weights for each sample
                    for b in range(batch_size):
                        importance = importance_list[b]
                        weights_full = compute_soft_weights(
                            importance,
                            prompt_len,
                            boost_factor=self.boost_factor,
                            min_weight=self.min_weight,
                            answer_weight=self.answer_weight,
                            enable_gradient_conservation=self.enable_gradient_conservation,
                            tokenizer=self.processing_class,
                            input_ids=input_ids[b:b+1],
                            debug=self.weight_debug and b == 0  # Only debug first sample
                        )

                        # Apply weights for this sample
                        completion_weights = weights_full[prompt_len:prompt_len+completion_ids.shape[1]]
                        token_weights[b, :len(completion_weights)] = torch.tensor(
                            completion_weights,
                            dtype=torch.float32,
                            device=token_weights.device
                        )

                    # Print optimization stats
                    if optimizer_step % 5 == 0 or optimizer_step <= 2:
                        if self.step_count % grad_accum == 0:
                            print(f"  Batch stats: cache_hits={stats['cache_hits']}, misses={stats['cache_misses']}, skipped={stats['skipped_uniform']}")

                elif batch_size > 0:
                    # Fallback to sequential for random/inverted modes
                    for b in range(batch_size):
                        importance = compute_counterfactual_importance(
                            model,
                            self.processing_class,
                            input_ids[b:b+1],
                            prompt_len,
                            max_spans=self.max_spans,
                            debug=self.weight_debug and b == 0,
                            random_mode=self.random_importance,
                            invert_mode=self.invert_importance
                        )

                        weights_full = compute_soft_weights(
                            importance,
                            prompt_len,
                            boost_factor=self.boost_factor,
                            min_weight=self.min_weight,
                            answer_weight=self.answer_weight,
                            enable_gradient_conservation=self.enable_gradient_conservation,
                            tokenizer=self.processing_class,
                            input_ids=input_ids[b:b+1],
                            debug=self.weight_debug and b == 0
                        )

                        completion_weights = weights_full[prompt_len:prompt_len+completion_ids.shape[1]]
                        token_weights[b, :len(completion_weights)] = torch.tensor(
                            completion_weights,
                            dtype=torch.float32,
                            device=token_weights.device
                        )

                # Print aggregate weight statistics every 5 steps
                if optimizer_step % 5 == 0 or optimizer_step <= 2:
                    if self.step_count % grad_accum == 0:
                        valid_weights = token_weights[token_weights > 0]
                        if len(valid_weights) > 0:
                            print(f"  Weight stats (across {batch_size} samples): mean={valid_weights.mean().item():.3f}, min={valid_weights.min().item():.3f}, max={valid_weights.max().item():.3f}, std={valid_weights.std().item():.3f}")
                        print(f"  Total optimization stats: {self._optimization_stats}")
                        print(f"{'='*80}")

                # ===== Extra Verbose Logging for Paper Examples =====
                if self.extra_verbose and self._verbose_logger and np.random.random() < self.extra_verbose_sample_rate:
                    try:
                        self._log_verbose_examples(
                            model=model,
                            input_ids=input_ids,
                            prompt_ids=prompt_ids,
                            completion_ids=completion_ids,
                            advantages=advantages,
                            token_weights=token_weights,
                            optimizer_step=optimizer_step,
                            batch_size=batch_size,
                            prompt_len=prompt_len,
                        )
                    except Exception as e:
                        print(f"[VerboseLogger] Error logging examples: {e}")

                # Log weight metrics to wandb (every step)
                valid_weights = token_weights[token_weights > 0]
                if len(valid_weights) > 0:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["counterfact/weight_mean"].append(valid_weights.mean().item())
                    self._metrics[mode]["counterfact/weight_std"].append(valid_weights.std().item())
                    self._metrics[mode]["counterfact/weight_min"].append(valid_weights.min().item())
                    self._metrics[mode]["counterfact/weight_max"].append(valid_weights.max().item())
                    # Weight spread: how much variation in token importance
                    self._metrics[mode]["counterfact/weight_spread"].append((valid_weights.max() - valid_weights.min()).item())
                    # Log optimization stats
                    self._metrics[mode]["counterfact/cache_hits"].append(self._optimization_stats["cache_hits"])
                    self._metrics[mode]["counterfact/cache_misses"].append(self._optimization_stats["cache_misses"])

        # Apply weights
        per_token_loss = per_token_loss * token_weights

        # ===== Final aggregation =====
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # ===== Logging =====
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        # ===== Custom Metrics: Sample Counter =====
        # Log total samples processed (only on main process to avoid duplication)
        if self.accelerator.is_main_process:
            self._metrics[mode]["total_samples_processed"] = [self.total_samples_processed]

        # ===== GPU Memory Tracking =====
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            self._metrics[mode]["gpu/memory_allocated_gb"].append(gpu_mem_allocated)
            self._metrics[mode]["gpu/memory_reserved_gb"].append(gpu_mem_reserved)

        return loss

    def _log_verbose_examples(
        self,
        model,
        input_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        advantages: torch.Tensor,
        token_weights: torch.Tensor,
        optimizer_step: int,
        batch_size: int,
        prompt_len: int,
    ):
        """Log detailed examples for paper - captures full credit allocation process."""
        tokenizer = self.processing_class

        # Log 1-3 samples from this batch (mix of correct and incorrect if possible)
        samples_to_log = min(3, batch_size)

        for b in range(samples_to_log):
            try:
                # Decode prompt and completion
                prompt_text = tokenizer.decode(prompt_ids[b], skip_special_tokens=True)
                completion_text = tokenizer.decode(completion_ids[b], skip_special_tokens=True)

                # Get advantage/reward for this sample
                advantage = advantages[b].item()
                is_correct = advantage > 0  # Positive advantage = correct

                # Parse predicted answer
                pred_answer = parse_pred_number(completion_text)

                # Get detailed importance info
                detail = compute_counterfactual_importance_detailed(
                    model,
                    tokenizer,
                    input_ids[b:b+1],
                    prompt_len,
                    max_spans=self.max_spans
                )

                # Get final weights for this sample
                weights = token_weights[b].cpu().numpy()
                valid_mask = weights > 0

                # Compute weight stats
                if valid_mask.sum() > 0:
                    valid_weights = weights[valid_mask]
                    weight_stats = {
                        "mean": float(valid_weights.mean()),
                        "std": float(valid_weights.std()),
                        "min": float(valid_weights.min()),
                        "max": float(valid_weights.max()),
                    }
                else:
                    weight_stats = {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}

                # Get top/bottom weighted tokens
                tokens = detail["tokens"]
                importance_norm = detail["importance_normalized"]

                # Pair tokens with their weights
                token_weight_pairs = []
                for i, (tok, w) in enumerate(zip(tokens, weights[:len(tokens)])):
                    if w > 0:
                        imp = importance_norm[i] if i < len(importance_norm) else 0.0
                        token_weight_pairs.append({
                            "pos": i,
                            "token": tok,
                            "weight": float(w),
                            "importance": float(imp)
                        })

                # Sort by weight
                token_weight_pairs.sort(key=lambda x: x["weight"], reverse=True)
                top_weighted = token_weight_pairs[:10]
                bottom_weighted = sorted(token_weight_pairs, key=lambda x: x["weight"])[:10]

                # Create verbose example
                example = VerboseExample(
                    step=optimizer_step,
                    sample_idx=b,
                    timestamp=datetime.now().isoformat(),
                    prompt_text=prompt_text,
                    completion_text=completion_text,
                    gold_answer="",  # Will be filled if available from dataset
                    predicted_answer=pred_answer,
                    is_correct=is_correct,
                    reward=1.0 if is_correct else 0.0,
                    advantage=advantage,
                    spans=detail["spans"],
                    reasoning_region=detail["reasoning_region"],
                    answer_region=detail["answer_region"],
                    baseline_answer_logprob=detail["baseline_score"],
                    importance_raw=detail["importance_raw"],
                    importance_normalized=importance_norm,
                    weights_final=weights[:len(tokens)].tolist(),
                    weight_stats=weight_stats,
                    top_weighted_tokens=top_weighted,
                    bottom_weighted_tokens=bottom_weighted,
                    tokens=tokens,
                )

                self._verbose_logger.log_example(example)

            except Exception as e:
                print(f"[VerboseLogger] Error logging sample {b}: {e}")
                continue

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use our counterfactual weighting.
        This handles both TRL and Unsloth backends - we compute per-token log probs
        from hidden states if needed.
        """
        if return_outputs:
            raise ValueError("The CounterfactualGRPOTrainer does not support returning outputs")

        # Add num_items_in_batch to inputs if provided (needed for DAPO loss)
        if num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch

        # Call our implementation (handles both TRL and Unsloth tensor formats)
        return self._compute_loss_impl(model, inputs)


# =============================================================================
# ANSWER PARSING
# =============================================================================

ANS_RE = re.compile(r"####\s*(.+)$", flags=re.MULTILINE)

def parse_gsm8k_gold(ans: str) -> str:
    m = ANS_RE.search(ans)
    if not m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        return nums[-1] if nums else ""
    s = m.group(1).strip().replace(",", "")
    if re.fullmatch(r"-?\d+/\d+", s):
        num, den = s.split("/")
        try:
            return str(float(num) / float(den))
        except:
            return s
    return s

def parse_pred_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    candidate = nums[-1] if nums else ""
    if re.fullmatch(r"-?\d+/\d+", candidate):
        num, den = candidate.split("/")
        try:
            return str(float(num) / float(den))
        except:
            return candidate
    return candidate

def numeric_equal(a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except:
        return a.strip() == b.strip()


def process_reward_wrapper(completions: List, answers: List[str], **kwargs) -> List[float]:
    """Reward function with process-level partial credit."""
    if len(answers) == 0:
        return [0.0] * len(completions)
    
    factor = max(1, len(completions) // len(answers))
    expanded = [a for a in answers for _ in range(factor)]
    expanded = expanded[:len(completions)]
    
    rewards = []
    stats = {'correct': 0, 'incorrect': 0, 'process_scores': []}
    
    for comp, gold in zip(completions, expanded):
        # Robust format handling (matches original script)
        if isinstance(comp, str):
            text = comp
        elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
            text = comp[0]["content"]
        else:
            print(f"Warning: Unexpected completion format: {type(comp)}")
            text = str(comp)
        
        result = compute_process_reward(text, text, gold)
        rewards.append(result['total_reward'])
        
        if result['final_correct'] > 0.5:
            stats['correct'] += 1
        else:
            stats['incorrect'] += 1
        stats['process_scores'].append(result['process_score'])
    
    if not hasattr(process_reward_wrapper, 'step'):
        process_reward_wrapper.step = 0
    process_reward_wrapper.step += 1
    
    mean_reward = sum(rewards) / len(rewards)
    mean_process = sum(stats['process_scores']) / len(stats['process_scores'])
    accuracy = stats['correct'] / (stats['correct'] + stats['incorrect'])
    
    print(f"\n🎯 Process Reward (step {process_reward_wrapper.step})")
    print(f"   Accuracy: {accuracy:.2%} ({stats['correct']}/{len(completions)})")
    print(f"   Mean reward: {mean_reward:.4f}, Mean process: {mean_process:.4f}")
    
    sys.stdout.flush()
    return rewards


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

# Chat template format (for instruct models)
SYSTEM_INSTRUCT = "You are a careful math tutor. Solve step by step and provide the final numeric answer."

def to_prompt(question: str, tokenizer, use_base_prompt: bool = False) -> str:
    """Format GSM8K question as prompt."""
    if use_base_prompt:
        # Simple format for base models - no chat template
        return f"{question}\nPlease put your final answer within \\boxed{{}}."
    else:
        # Chat template for instruct models
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCT},
                {"role": "user", "content": f"Problem:\n{question}"}
            ]
            return apply_chat_template_safe(tokenizer, messages, enable_thinking=False)
        else:
            # Fallback
            return f"{SYSTEM_INSTRUCT}\n\nProblem:\n{question}\n\nAnswer:"