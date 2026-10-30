"""
SGLang rollout backend for TRL's GRPOTrainer.

TRL 1.7.1 only ships vLLM generation, so this plugs SGLang in through TRL's
public ``rollout_func`` hook instead of patching trainer internals:

- Generation: the trainer tokenizes prompts exactly like the HF path, then sends
  the token ids to a running SGLang server (``POST /generate``). Each GRPO
  group goes out as one prompt with ``n=num_generations``, so RadixAttention
  prefills the prompt once and shares its KV cache across the group.
- Weight sync, once per optimizer step, picked from what the server supports:
  * LoRA policy + server started with ``--enable-lora``: only the adapter is
    saved and hot-loaded (``/load_lora_adapter``), then the previous version
    is unloaded. Cheap (adapter-sized I/O) and no base weights touched.
  * Otherwise: full weights (LoRA deltas folded in) are streamed to sharded
    safetensors files with bounded host memory and reloaded via
    ``/update_weights_from_disk``, which also flushes the prefix cache.

The server runs as a separate process (it can live in its own venv), started
with e.g. ``python -m sglang.launch_server --model-path <same model>``. It must
see the trainer's filesystem, so it has to run on the same host. Tested with
SGLang 0.5.10.post1 and 0.5.22; only endpoints present in both are used.
"""

import logging
import os
import shutil
import tempfile
import time
import uuid
import weakref
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

WEIGHTS_PREFIX = "model-"
# Host-RAM bound for full-weight sync: each shard is serialized in memory once.
SHARD_BYTES = 1 << 30


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
        except requests.Timeout as e:
            raise RuntimeError(
                f"SGLang {path} did not answer within {self.timeout:.0f}s; the server is up "
                "but stuck, check its log."
            ) from e
        except requests.RequestException as e:
            raise ConnectionError(
                f"SGLang server at {self.base_url} is unreachable ({e}). Start it with "
                "`python -m sglang.launch_server --model-path <model>` and set "
                "`sglang_server_url`, or use rollout_backend='hf'."
            ) from e
        if response.status_code != 200:
            raise RuntimeError(
                f"SGLang {path} failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        return self._request("GET", "/model_info")

    def generate(
        self,
        input_ids: List[List[int]],
        sampling_params: Dict[str, Any],
        lora_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ``sampling_params.get("n", 1)`` samples per prompt, grouped by prompt."""
        payload = {"input_ids": input_ids, "sampling_params": sampling_params}
        if lora_name is not None:
            payload["lora_path"] = lora_name  # a loaded adapter's name, applied to every prompt
        outputs = self._request("POST", "/generate", payload)
        expected = len(input_ids) * sampling_params.get("n", 1)
        if len(outputs) != expected:
            raise RuntimeError(f"SGLang returned {len(outputs)} outputs, expected {expected}")
        return outputs

    def get_server_info(self) -> Dict[str, Any]:
        return self._request("GET", "/server_info")

    def load_lora_adapter(self, name: str, path: str) -> None:
        self._request("POST", "/load_lora_adapter", {"lora_name": name, "lora_path": path})

    def unload_lora_adapter(self, name: str) -> None:
        self._request("POST", "/unload_lora_adapter", {"lora_name": name})

    def update_weights_from_disk(self, model_path: str) -> None:
        result = self._request("POST", "/update_weights_from_disk", {"model_path": model_path})
        if not result.get("success", False):
            raise RuntimeError(
                f"SGLang weight update from {model_path} failed: {result.get('message')}"
            )


def _base_name(name: str) -> str:
    """Map a PEFT parameter/module name to the base model's HF name."""
    return (
        name.removeprefix("base_model.model.")
        .replace(".base_layer", "")
        .replace("modules_to_save.default.", "")
    )


def _dense(param) -> torch.Tensor:
    """A bitsandbytes 4-bit weight dequantized to its compute dtype; other params as they are."""
    quant_state = getattr(param, "quant_state", None)
    if quant_state is None:
        return param.detach()
    import bitsandbytes.functional as bnb_functional

    return bnb_functional.dequantize_4bit(param.data, quant_state).to(quant_state.dtype)


@torch.no_grad()
def iter_policy_weights(model) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield the policy weights under the base model's HF parameter names.

    LoRA deltas are folded into copies of the base weights one module at a
    time (fp32 math, cast back), so the trainer's model is never merged in
    place (repeated in-place merges would pile bf16 rounding error into the
    frozen base) and at most one merged copy is alive per yielded tensor.
    ``named_parameters`` skips tied duplicates (e.g. lm_head), which
    safetensors refuses to save twice; SGLang re-ties them on load.
    """
    if not hasattr(model, "peft_config"):
        for name, param in model.named_parameters():
            yield name, _dense(param).contiguous()
        return

    adapter_prefix = getattr(model, "prefix", "lora_")
    lora_layers = {
        name + ".base_layer.weight": module
        for name, module in model.named_modules()
        if hasattr(module, "get_delta_weight") and hasattr(module, "base_layer")
    }
    for name, param in model.named_parameters():
        if adapter_prefix in name or "original_module" in name:
            continue
        weight = _dense(param)
        module = lora_layers.get(name)
        if module is not None:
            merged = weight.float()
            for adapter in module.active_adapters:
                if adapter not in module.lora_A and adapter not in getattr(
                    module, "lora_embedding_A", {}
                ):
                    continue
                if getattr(module, "use_dora", {}).get(adapter, False):
                    raise NotImplementedError(
                        "rollout_backend='sglang' does not support DoRA adapters yet"
                    )
                merged = merged + module.get_delta_weight(adapter).float()
            weight = merged.to(weight.dtype)
        yield _base_name(name), weight.contiguous()


def export_policy_state_dict(model) -> Dict[str, torch.Tensor]:
    """Materialize :func:`iter_policy_weights` (tests and small models only)."""
    return dict(iter_policy_weights(model))


def checkpoint_names(model, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename weights to the layout of the hub checkpoint, as ``save_pretrained`` does.

    transformers 5 renames some weights on load (Gemma 3 keeps
    ``language_model.model.*`` on the hub but ``model.language_model.*`` in memory), and
    SGLang loads the hub layout. Older transformers keep the names, so nothing changes there.
    """
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError:
        return weights
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return revert_weight_conversion(base, weights)


def write_sharded_safetensors(
    weights: Iterator[Tuple[str, torch.Tensor]], directory: str, rename=None
) -> List[str]:
    """Write weights as ``model-XXXXX.safetensors`` shards of about ``SHARD_BYTES``.

    Peak host memory is about one shard: tensors stay on their device until
    their shard is serialized. Stale shards from a previous sync are removed.
    ``rename`` maps each shard's name->tensor dict to the names to save under.
    """
    from safetensors.torch import save_file

    for old in os.listdir(directory):
        if old.startswith(WEIGHTS_PREFIX) and old.endswith(".safetensors"):
            os.remove(os.path.join(directory, old))
    paths, shard, size = [], {}, 0

    def flush():
        path = os.path.join(directory, f"{WEIGHTS_PREFIX}{len(paths):05d}.safetensors")
        save_file(rename(shard) if rename else shard, path)
        paths.append(path)

    for name, tensor in weights:
        shard[name] = tensor
        size += tensor.numel() * tensor.element_size()
        if size >= SHARD_BYTES:
            flush()
            shard, size = {}, 0
    if shard:
        flush()
    return paths


def check_adapter_supported(peft_config) -> None:
    """Raise for LoRA features SGLang's adapter loader would silently get wrong."""
    peft_type = str(getattr(peft_config, "peft_type", ""))
    if "LORA" not in peft_type.upper():
        raise NotImplementedError(f"SGLang adapter sync supports LoRA only, got {peft_type}")
    unsupported = {
        "use_rslora": getattr(peft_config, "use_rslora", False),  # SGLang scales by alpha / r
        "use_dora": getattr(peft_config, "use_dora", False),
        "modules_to_save": bool(getattr(peft_config, "modules_to_save", None)),
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad:
        raise NotImplementedError(
            f"SGLang LoRA adapters do not support {bad}. Start the SGLang server without "
            "--enable-lora to sync merged full weights instead."
        )


NON_TEXT_MODULES = ("vision_tower", "vision_model", "visual", "multi_modal_projector", "audio")


def drop_non_text_adapter_weights(adapter_dir: str) -> int:
    """Remove LoRA tensors on vision/audio modules from a saved adapter; return how many.

    A multimodal model (e.g. Gemma 3) matches ``q_proj`` in its vision encoder too, but SGLang
    applies LoRA to the language model only and fails to load adapter tensors of any other
    shape. Text-only rollouts never run those modules, so they are not needed.
    """
    from safetensors.torch import load_file, save_file

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    weights = load_file(path)
    kept = {k: v for k, v in weights.items() if not any(m in k for m in NON_TEXT_MODULES)}
    if len(kept) < len(weights):
        save_file(kept, path, metadata={"format": "pt"})
    return len(weights) - len(kept)


def explicit_target_modules(adapter_dir: str) -> bool:
    """Replace a regex ``target_modules`` in a saved adapter with the module names it covers.

    SGLang accepts a list of module names (or "all"/"all-linear") but not a regex, which is
    what Unsloth saves for some models (e.g. Gemma 3). Returns whether the config changed.
    """
    import json

    from safetensors import safe_open

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    targets = config.get("target_modules")
    if not isinstance(targets, str) or targets in ("all", "all-linear"):
        return False
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        names = {key.split(".lora_")[0].rsplit(".", 1)[-1] for key in f.keys() if ".lora_" in key}
    config["target_modules"] = sorted(names)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return True


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
        # Feature detection: adapter-only sync needs a server launched with --enable-lora.
        self.server_lora = bool(self.client.get_server_info().get("enable_lora"))
        self.lora_name: Optional[str] = None
        self._version = 0
        # Unique per run: a long-lived server may still hold adapters from an earlier run.
        self.adapter_prefix = f"aligntune-{uuid.uuid4().hex[:8]}"
        if sync_dir is None:
            shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
            sync_dir = tempfile.mkdtemp(prefix="aligntune-sglang-", dir=shm)
            # Synced weights live here (RAM-backed in /dev/shm): remove them with the rollout.
            weakref.finalize(self, shutil.rmtree, sync_dir, True)
        self.sync_dir = sync_dir
        os.makedirs(self.sync_dir, exist_ok=True)
        # None forces a sync before the first rollout, so generation is on-policy
        # even if the server was started from a different checkpoint of the model.
        self._synced_step: Optional[int] = None
        logger.info(
            f"SGLang rollout backend: server={server_url} model={self.server_model} "
            f"lora={self.server_lora} sync_dir={self.sync_dir}"
        )

    def __call__(self, prompts: list, trainer) -> Dict[str, Any]:
        if self._synced_step != trainer.state.global_step:
            self.sync_weights(trainer)
            self._synced_step = trainer.state.global_step

        prompt_ids, images, _ = trainer._tokenize_prompts(prompts)
        if images is not None:
            raise NotImplementedError(
                "rollout_backend='sglang' does not support multimodal prompts yet"
            )

        start = time.perf_counter()
        outputs = self._generate_groups(prompt_ids, self.sampling_params(trainer))
        elapsed = time.perf_counter() - start
        completion_ids = [out["output_ids"] for out in outputs]

        num_tokens = sum(len(ids) for ids in completion_ids)
        trainer._log_metric("rollout/sglang_generate_s", elapsed)
        trainer._log_metric("rollout/sglang_tokens_per_s", num_tokens / max(elapsed, 1e-9))
        # logprobs=None: TRL only consumes rollout logprobs for vLLM importance sampling.
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None}

    def _generate_groups(
        self, prompt_ids: List[List[int]], sampling_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate TRL's repeated prompts, prefilling each GRPO group's prompt once.

        TRL repeats every prompt ``num_generations`` times back to back. SGLang
        only shares the prompt prefill across samples requested via ``n``
        (identical prompts in one batch are each prefilled), so collapse equal
        consecutive runs when they all have the same length.
        """
        unique, counts = [], []
        for ids in prompt_ids:
            if unique and ids == unique[-1]:
                counts[-1] += 1
            else:
                unique.append(ids)
                counts.append(1)
        grouped = len(set(counts)) == 1 and counts[0] > 1
        if grouped and self.lora_name is None:
            return self.client.generate(unique, {**sampling_params, "n": counts[0]})
        if grouped:
            # SGLang 0.5.10 never releases a LoRA adapter used with n > 1, so its unload hangs.
            # Prefill each group's prompt once into the prefix cache, then send the duplicates.
            self.client.generate(unique, {**sampling_params, "max_new_tokens": 0}, self.lora_name)
        return self.client.generate(prompt_ids, sampling_params, self.lora_name)

    @staticmethod
    def sampling_params(trainer) -> Dict[str, Any]:
        """Map the trainer's sampling settings to SGLang's sampling_params."""
        return {
            "max_new_tokens": trainer.max_completion_length,
            "temperature": trainer.temperature,
            "top_p": trainer.top_p,
            "top_k": (
                trainer.top_k if trainer.top_k else -1
            ),  # TRL uses 0 for "disabled", SGLang uses -1
            "min_p": trainer.min_p or 0.0,
            "repetition_penalty": trainer.repetition_penalty,
            # Same stop token as TRL's transformers path; it is kept in output_ids.
            "stop_token_ids": [trainer._tokenizer.eos_token_id],
        }

    def sync_weights(self, trainer) -> None:
        """Push the current policy weights to the SGLang server."""
        if trainer.is_fsdp_enabled:
            raise NotImplementedError("rollout_backend='sglang' does not support FSDP yet")

        start = time.perf_counter()
        model = trainer.accelerator.unwrap_model(trainer.model)
        # gather_params is a no-op outside DeepSpeed ZeRO-3.
        with trainer._dist.gather_params(list(model.parameters())):
            if trainer.accelerator.is_main_process:
                if self.server_lora and hasattr(model, "peft_config"):
                    self._sync_adapter(model)
                else:
                    base = model.get_base_model() if hasattr(model, "peft_config") else model
                    if getattr(base, "is_loaded_in_8bit", False):
                        raise NotImplementedError(
                            "Full-weight SGLang sync cannot send 8-bit weights. Start the server "
                            "with --enable-lora to sync the LoRA adapter instead."
                        )
                    write_sharded_safetensors(
                        iter_policy_weights(model),
                        self.sync_dir,
                        rename=lambda shard: checkpoint_names(model, shard),
                    )
                    self.client.update_weights_from_disk(self.sync_dir)
        trainer.accelerator.wait_for_everyone()
        trainer._log_metric("rollout/sglang_sync_s", time.perf_counter() - start)

    def _check_same_base(self, policy_base: Optional[str]) -> None:
        """Adapter sync never touches base weights: the server must serve the policy's base."""

        def short(name: str) -> str:  # "Qwen/Qwen3-4B" and "/models/Qwen3-4B" are the same model
            return name.rstrip("/").split("/")[-1].lower()

        if policy_base and self.server_model and short(policy_base) != short(self.server_model):
            raise ValueError(
                f"SGLang serves '{self.server_model}' but the LoRA policy's base model is "
                f"'{policy_base}'. "
                "Launch the server with --model-path set to the policy's base model."
            )

    def _sync_adapter(self, model) -> None:
        """Hot-load the current LoRA adapter under a new name, then drop the old one."""
        adapter = (
            model.active_adapter
            if isinstance(model.active_adapter, str)
            else model.active_adapter[0]
        )
        check_adapter_supported(model.peft_config[adapter])
        if self.lora_name is None:
            base = model.peft_config[adapter].base_model_name_or_path
            self._check_same_base(base or model.get_base_model().config.name_or_path)
        self._version += 1
        name = f"{self.adapter_prefix}-v{self._version}"
        # Two alternating dirs: the previous adapter's files stay intact until it is unloaded.
        path = os.path.join(self.sync_dir, f"adapter-{self._version % 2}")
        model.save_pretrained(path, selected_adapters=[adapter], safe_serialization=True)
        if adapter != "default":  # PEFT nests non-default adapters in a subdir
            path = os.path.join(path, adapter)
        dropped = drop_non_text_adapter_weights(path)
        explicit_target_modules(path)
        if dropped and self._version == 1:
            logger.warning(
                f"SGLang applies LoRA to the language model only: left out {dropped} adapter "
                "tensors on vision/audio modules (text-only rollouts do not use them)."
            )
        self.client.load_lora_adapter(name, path)
        previous, self.lora_name = self.lora_name, name
        if previous is not None:
            self.client.unload_lora_adapter(previous)

    def close(self) -> None:
        """Unload this run's adapter so a long-lived server does not accumulate them."""
        name, self.lora_name = self.lora_name, None
        self._synced_step = None  # a reused rollout re-syncs before generating again
        if name is None:
            return
        try:
            self.client.unload_lora_adapter(name)
        except (ConnectionError, RuntimeError) as e:  # never fail a finished run over cleanup
            logger.warning(f"Could not unload SGLang adapter {name}: {e}")


class SGLangRolloutCallback(TrainerCallback):
    """Closes the rollout when training ends."""

    def __init__(self, rollout: SGLangRollout):
        self.rollout = rollout

    def on_train_end(self, args, state, control, **kwargs):
        self.rollout.close()
