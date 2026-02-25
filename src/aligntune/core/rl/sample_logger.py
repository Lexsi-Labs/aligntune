"""
Utility helpers for qualitative sample logging across RL trainers.
"""

import logging
from typing import List, Optional, Dict, Any

import torch

from .config import SampleLoggingConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly,",
    "Write a concise plan for improving developer productivity using",
]


def should_log_samples(
    sample_cfg: SampleLoggingConfig,
    current_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> bool:
    """Return True if qualitative samples should be logged."""
    if not sample_cfg.enabled:
        return False
    if current_step is None:
        return True
    
    if sample_cfg.interval_steps and sample_cfg.interval_steps > 0:
        if current_step % sample_cfg.interval_steps == 0:
            return True
    
    if (
        sample_cfg.percent_of_max_steps
        and max_steps
        and sample_cfg.percent_of_max_steps > 0
    ):
        interval = max(1, int(max_steps * sample_cfg.percent_of_max_steps))
        if current_step % interval == 0:
            return True
    
    return False


def generate_and_log_samples(
    sample_cfg: SampleLoggingConfig,
    model,
    tokenizer,
    reward_functions: Optional[List[Any]] = None,
    stage: str = "train",
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Generate qualitative samples and log them."""
    if not sample_cfg.enabled:
        log.debug("Sample logging disabled; skipping qualitative sample generation.")
        return []
    
    log = log or logger
    prompts = sample_cfg.prompts or DEFAULT_PROMPTS
    if not prompts:
        log.debug("Sample logging enabled but no prompts provided.")
        return []
    
    try:
        model.eval()
    except Exception:
        pass
    
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    
    samples = []
    banner = "=" * 80
    log.info(banner)
    log.info(f"Qualitative samples ({stage}) - prompts: {len(prompts)}")
    print(banner)
    print(f"Qualitative samples ({stage}) - prompts: {len(prompts)}")
    
    num_samples = min(sample_cfg.num_samples, len(prompts))
    for idx, prompt in enumerate(prompts[:num_samples], start=1):
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=sample_cfg.max_new_tokens,
                    do_sample=True,
                    temperature=sample_cfg.temperature,
                    top_p=sample_cfg.top_p,
                    pad_token_id=pad_token_id,
                )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            reward_scores: Dict[str, Any] = {}
            if reward_functions:
                for r_idx, reward_fn in enumerate(reward_functions):
                    try:
                        reward_scores[f"reward_{r_idx}"] = reward_fn(response)
                    except Exception as reward_error:
                        reward_scores[f"reward_{r_idx}"] = f"error: {reward_error}"
            
            prefix = f"[Sample {idx}/{num_samples}]"
            log.info(f"{prefix} Prompt: {prompt}")
            log.info(f"{prefix} Response: {response}")
            print(f"{prefix} Prompt: {prompt}")
            print(f"{prefix} Response: {response}")
            if reward_scores:
                log.info(f"{prefix} Rewards: {reward_scores}")
                print(f"{prefix} Rewards: {reward_scores}")
            
            samples.append({
                "prompt": prompt,
                "response": response,
                "rewards": reward_scores,
            })
        except Exception as generation_error:
            log.warning(f"Failed to generate sample {idx}: {generation_error}")
    
    log.info(banner)
    print(banner)
    return samples

