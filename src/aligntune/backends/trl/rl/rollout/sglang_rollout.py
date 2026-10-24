"""
SGLang rollout backend for TRL's GRPOTrainer.

TRL 1.7.1 only ships vLLM generation, so this plugs SGLang in through TRL's
public ``rollout_func`` hook instead of patching trainer internals:

- Generation: the trainer tokenizes prompts exactly like the HF path, then sends
  the token ids to a running SGLang server (``POST /generate``). GRPO repeats
  each prompt ``num_generations`` times, and SGLang's RadixAttention shares the
  prompt KV cache across those repeats.
- Weight sync: after every optimizer step the policy weights (LoRA adapters
  merged in) are written as one safetensors file and the server reloads them
  with ``POST /update_weights_from_disk``, which also flushes the prefix cache.

The server runs as a separate process (it can live in its own venv), started
with e.g. ``python -m sglang.launch_server --model-path <same model>``. It must
see the trainer's filesystem, so it has to run on the same host.
"""

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "model.safetensors"


class SGLangClient:
    """Minimal HTTP client for the SGLang server endpoints used during RL."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise ConnectionError(
                f"SGLang server at {self.base_url} is unreachable ({e}). Start it with "
                "`python -m sglang.launch_server --model-path <model>` and set "
                "`sglang_server_url`, or use rollout_backend='hf'."
            ) from e
        if response.status_code != 200:
            raise RuntimeError(f"SGLang {path} failed with HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        return self._request("GET", "/get_model_info")

    def generate(self, input_ids: List[List[int]], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = self._request("POST", "/generate", {"input_ids": input_ids, "sampling_params": sampling_params})
        if len(outputs) != len(input_ids):
            raise RuntimeError(f"SGLang returned {len(outputs)} outputs for {len(input_ids)} prompts")
        return outputs

    def update_weights_from_disk(self, model_path: str) -> None:
        result = self._request("POST", "/update_weights_from_disk", {"model_path": model_path})
        if not result.get("success", False):
            raise RuntimeError(f"SGLang weight update from {model_path} failed: {result.get('message')}")


def export_policy_state_dict(model) -> Dict[str, Any]:
    """Return the policy weights under the base model's HF parameter names.

    For a PEFT model the caller must have merged the adapters first; LoRA
    tensors are dropped and wrapper prefixes are stripped (same mapping TRL
    uses for vLLM). ``named_parameters`` skips tied duplicates (e.g. lm_head),
    which safetensors refuses to save twice.
    """
    is_peft = hasattr(model, "peft_config")
    adapter_prefix = getattr(model, "prefix", "lora_")
    state = {}
    for name, param in model.named_parameters():
        if is_peft:
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if adapter_prefix in name or "original_module" in name:
                continue
            name = name.replace("modules_to_save.default.", "")
        state[name] = param.detach().contiguous()
    return state


class SGLangRollout:
    """TRL ``rollout_func`` that generates GRPO completions with an SGLang server.

    Args:
        server_url: Base URL of the SGLang server.
        sync_dir: Directory the weights are written to before each reload. Must
            be readable by the server. Defaults to a fresh dir in /dev/shm.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(self, server_url: str, sync_dir: Optional[str] = None, timeout: float = 600.0):
        self.client = SGLangClient(server_url, timeout=timeout)
        info = self.client.get_model_info()  # fails fast if the server is down
        self.server_model = info.get("model_path")
        shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
        self.sync_dir = sync_dir or tempfile.mkdtemp(prefix="aligntune-sglang-", dir=shm)
        os.makedirs(self.sync_dir, exist_ok=True)
        # None forces a sync before the first rollout, so generation is on-policy
        # even if the server was started from a different checkpoint of the model.
        self._synced_step: Optional[int] = None
        logger.info(f"SGLang rollout backend: server={server_url} model={self.server_model} sync_dir={self.sync_dir}")

    def __call__(self, prompts: list, trainer) -> Dict[str, Any]:
        if self._synced_step != trainer.state.global_step:
            self.sync_weights(trainer)
            self._synced_step = trainer.state.global_step

        prompt_ids, images, _ = trainer._tokenize_prompts(prompts)
        if images is not None:
            raise NotImplementedError("rollout_backend='sglang' does not support multimodal prompts yet")

        start = time.perf_counter()
        outputs = self.client.generate(prompt_ids, self.sampling_params(trainer))
        elapsed = time.perf_counter() - start
        completion_ids = [out["output_ids"] for out in outputs]

        num_tokens = sum(len(ids) for ids in completion_ids)
        trainer._log_metric("rollout/sglang_generate_s", elapsed)
        trainer._log_metric("rollout/sglang_tokens_per_s", num_tokens / max(elapsed, 1e-9))
        # logprobs=None: TRL only consumes rollout logprobs for vLLM importance sampling.
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None}

    @staticmethod
    def sampling_params(trainer) -> Dict[str, Any]:
        """Map the trainer's sampling settings to SGLang's sampling_params."""
        return {
            "max_new_tokens": trainer.max_completion_length,
            "temperature": trainer.temperature,
            "top_p": trainer.top_p,
            "top_k": trainer.top_k if trainer.top_k else -1,  # TRL uses 0 for "disabled", SGLang uses -1
            "min_p": trainer.min_p or 0.0,
            "repetition_penalty": trainer.repetition_penalty,
            # Same stop token as TRL's transformers path; it is kept in output_ids.
            "stop_token_ids": [trainer._tokenizer.eos_token_id],
        }

    def sync_weights(self, trainer) -> None:
        """Push the current policy weights to the SGLang server."""
        from safetensors.torch import save_file

        if trainer.is_fsdp_enabled:
            raise NotImplementedError("rollout_backend='sglang' does not support FSDP yet")

        start = time.perf_counter()
        model = trainer.accelerator.unwrap_model(trainer.model)
        is_peft = hasattr(model, "peft_config")
        # gather_params is a no-op outside DeepSpeed ZeRO-3.
        with trainer._dist.gather_params(list(model.parameters())):
            if trainer.accelerator.is_main_process:
                if is_peft:
                    model.merge_adapter()
                try:
                    save_file(export_policy_state_dict(model), os.path.join(self.sync_dir, WEIGHTS_FILE))
                finally:
                    if is_peft:
                        model.unmerge_adapter()
                self.client.update_weights_from_disk(self.sync_dir)
        trainer.accelerator.wait_for_everyone()
        trainer._log_metric("rollout/sglang_sync_s", time.perf_counter() - start)
