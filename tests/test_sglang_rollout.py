"""
Unit tests for the SGLang rollout backend (TRL GRPO rollout_func).

A tiny in-process HTTP server stands in for SGLang, so these run on CPU
without sglang installed.
"""

import contextlib
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from aligntune.backends.trl.rl.rollout import sglang_rollout
from aligntune.backends.trl.rl.rollout.sglang_rollout import (
    SGLangClient,
    SGLangRollout,
    export_policy_state_dict,
)

EOS = 2


class FakeSGLang(BaseHTTPRequestHandler):
    """Echo server: each completion is [len(prompt), EOS]."""

    calls = []
    weight_update_ok = True
    enable_lora = False
    model_path = "tiny-model"
    unload_delay_s = 0.0

    def log_message(self, *args):
        pass

    def _reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.calls.append((self.path, None))
        if self.path == "/model_info":
            return self._reply(200, {"model_path": self.model_path})
        if self.path == "/server_info":
            return self._reply(200, {"enable_lora": self.enable_lora})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, payload))
        if self.path == "/generate":
            if payload["sampling_params"].get("max_new_tokens", 1) < 0:
                return self._reply(400, {"error": "bad max_new_tokens"})
            n = payload["sampling_params"].get("n", 1)
            return self._reply(
                200,
                [{"output_ids": [len(ids), EOS]} for ids in payload["input_ids"] for _ in range(n)],
            )
        if self.path == "/update_weights_from_disk":
            ok = self.weight_update_ok and bool(shards(payload["model_path"]))
            return self._reply(200, {"success": ok, "message": "ok" if ok else "load failed"})
        if self.path == "/load_lora_adapter":
            ok = os.path.exists(os.path.join(payload["lora_path"], "adapter_model.safetensors"))
            return self._reply(200 if ok else 400, {"success": ok})
        if self.path == "/unload_lora_adapter":
            time.sleep(self.unload_delay_s)
            return self._reply(200, {"success": True})
        self._reply(404, {"error": "not found"})


def shards(directory):
    return sorted(
        f for f in os.listdir(directory) if f.startswith("model-") and f.endswith(".safetensors")
    )


def load_shards(directory):
    from safetensors.torch import load_file

    state = {}
    for name in shards(directory):
        state.update(load_file(os.path.join(str(directory), name)))
    return state


def calls(path):
    return [p for called, p in FakeSGLang.calls if called == path]


@pytest.fixture
def server():
    FakeSGLang.calls = []
    FakeSGLang.weight_update_ok = True
    FakeSGLang.enable_lora = False
    FakeSGLang.model_path = "tiny-model"
    FakeSGLang.unload_delay_s = 0.0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLang)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def unused_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLang)
    port = httpd.server_address[1]
    httpd.server_close()
    return f"http://127.0.0.1:{port}"


def make_trainer(model, step=0):
    def tokenize(prompts):
        return [[1] * len(p) for p in prompts], None, {}

    metrics = {}
    return SimpleNamespace(
        state=SimpleNamespace(global_step=step),
        _tokenize_prompts=tokenize,
        _tokenizer=SimpleNamespace(eos_token_id=EOS),
        max_completion_length=16,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        model=model,
        is_fsdp_enabled=False,
        accelerator=SimpleNamespace(
            unwrap_model=lambda m: m, is_main_process=True, wait_for_everyone=lambda: None
        ),
        _dist=SimpleNamespace(gather_params=lambda params: nullcontext()),
        _log_metric=lambda name, value: metrics.setdefault(name, []).append(value),
        metrics=metrics,
    )


def tiny_model(tie=True):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    return LlamaForCausalLM(config)


class TestSGLangClient:
    def test_unreachable_server_raises_connection_error(self):
        with pytest.raises(ConnectionError, match="unreachable"):
            SGLangClient(unused_url(), timeout=2).get_model_info()

    def test_http_error_is_raised_not_swallowed(self, server):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            SGLangClient(server).generate([[1, 2]], {"max_new_tokens": -1})

    def test_stuck_server_raises_instead_of_waiting_forever(self, server):
        FakeSGLang.unload_delay_s = 2.0
        with pytest.raises(RuntimeError, match="did not answer within"):
            SGLangClient(server, timeout=0.2).unload_lora_adapter("policy")

    def test_generate_returns_one_output_per_prompt_in_order(self, server):
        outputs = SGLangClient(server).generate([[1], [1, 1, 1]], {"max_new_tokens": 4})
        assert [o["output_ids"] for o in outputs] == [[1, EOS], [3, EOS]]

    def test_failed_weight_update_raises(self, server, tmp_path):
        FakeSGLang.weight_update_ok = False
        with pytest.raises(RuntimeError, match="load failed"):
            SGLangClient(server).update_weights_from_disk(str(tmp_path))


class TestSGLangRollout:
    def test_constructor_fails_fast_when_server_is_down(self, tmp_path):
        with pytest.raises(ConnectionError):
            SGLangRollout(unused_url(), sync_dir=str(tmp_path), timeout=2)

    def test_default_sync_dir_is_removed_with_the_rollout(self, server):
        rollout = SGLangRollout(server)
        sync_dir = rollout.sync_dir
        rollout(["a"], make_trainer(tiny_model()))  # writes the weights
        assert shards(sync_dir)
        del rollout
        assert not os.path.exists(sync_dir)

    def test_rollout_returns_trl_contract(self, server, tmp_path):
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        out = rollout(["ab", "ab", "abcd", "abcd"], make_trainer(tiny_model()))
        assert out["prompt_ids"] == [[1, 1], [1, 1], [1] * 4, [1] * 4]
        assert out["completion_ids"] == [[2, EOS], [2, EOS], [4, EOS], [4, EOS]]
        assert out["logprobs"] is None

    def test_grpo_groups_are_sent_once_with_n(self, server, tmp_path):
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        rollout(["ab", "ab", "abcd", "abcd"], make_trainer(tiny_model()))
        request = [p for path, p in FakeSGLang.calls if path == "/generate"][0]
        assert request["input_ids"] == [[1, 1], [1] * 4]
        assert request["sampling_params"]["n"] == 2

    def test_uneven_groups_fall_back_to_one_sample_per_prompt(self, server, tmp_path):
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        out = rollout(["ab", "ab", "abc"], make_trainer(tiny_model()))
        request = [p for path, p in FakeSGLang.calls if path == "/generate"][0]
        assert request["input_ids"] == [[1, 1], [1, 1], [1] * 3]
        assert "n" not in request["sampling_params"]
        assert out["completion_ids"] == [[2, EOS], [2, EOS], [3, EOS]]

    def test_sampling_params_map_trl_defaults_to_sglang(self, server, tmp_path):
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        rollout(["a"], make_trainer(tiny_model()))
        params = [p for path, p in FakeSGLang.calls if path == "/generate"][0]["sampling_params"]
        assert params == {
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "stop_token_ids": [EOS],
        }

    def test_weights_sync_once_per_optimizer_step(self, server, tmp_path):
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        trainer = make_trainer(tiny_model())
        rollout(["a"], trainer)  # first call syncs (server may hold other weights)
        rollout(["a"], trainer)  # same step (grad accumulation): no sync
        trainer.state.global_step = 1
        rollout(["a"], trainer)
        syncs = [p for path, p in FakeSGLang.calls if path == "/update_weights_from_disk"]
        assert syncs == [{"model_path": str(tmp_path)}] * 2
        assert len(trainer.metrics["rollout/sglang_sync_s"]) == 2
        assert len(trainer.metrics["rollout/sglang_tokens_per_s"]) == 3

    def test_sync_writes_loadable_full_model_weights(self, server, tmp_path):
        model = tiny_model(tie=False)
        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(model))
        saved = load_shards(tmp_path)
        expected = dict(model.named_parameters())
        assert saved.keys() == expected.keys()
        assert all(torch.equal(saved[k], expected[k]) for k in saved)

    def test_full_sync_is_sharded_and_replaces_stale_shards(self, server, tmp_path, monkeypatch):
        monkeypatch.setattr(sglang_rollout, "SHARD_BYTES", 1024)  # force many shards
        (tmp_path / "model-99999.safetensors").write_bytes(b"stale")
        model = tiny_model(tie=False)
        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(model))
        assert len(shards(tmp_path)) > 1
        assert "model-99999.safetensors" not in shards(tmp_path)
        expected = dict(model.named_parameters())
        saved = load_shards(tmp_path)
        assert saved.keys() == expected.keys()
        assert all(torch.equal(saved[k], expected[k]) for k in saved)

    def test_fsdp_is_rejected(self, server, tmp_path):
        trainer = make_trainer(tiny_model())
        trainer.is_fsdp_enabled = True
        with pytest.raises(NotImplementedError, match="FSDP"):
            SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(trainer)


class TestExportPolicyStateDict:
    def test_tied_embeddings_are_exported_once(self):
        state = export_policy_state_dict(tiny_model(tie=True))
        assert "model.embed_tokens.weight" in state
        assert "lm_head.weight" not in state  # SGLang re-ties it on load

    def test_peft_export_uses_base_names_and_merged_weights(self, server, tmp_path):
        peft = pytest.importorskip("peft")
        base = tiny_model(tie=False)
        base_names = {n for n, _ in base.named_parameters()}
        lora = peft.get_peft_model(
            base,
            peft.LoraConfig(
                r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False
            ),
        )
        q_layer = lora.base_model.model.model.layers[0].self_attn.q_proj
        q_before = q_layer.base_layer.weight.detach().clone()
        delta = (
            q_layer.lora_B["default"].weight @ q_layer.lora_A["default"].weight
        ) * q_layer.scaling["default"]

        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(lora))
        saved = load_shards(tmp_path)

        assert set(saved) == base_names
        q = "model.layers.0.self_attn.q_proj.weight"
        assert torch.allclose(saved[q], q_before + delta.detach(), atol=1e-6)  # adapter folded in
        assert torch.equal(q_layer.base_layer.weight, q_before)  # trainer's model is not mutated


def lora_model(**config):
    peft = pytest.importorskip("peft")
    base = tiny_model(tie=False)
    base.config._name_or_path = "org/tiny-model"
    defaults = dict(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    return peft.get_peft_model(base, peft.LoraConfig(**{**defaults, **config}))


def _two_rank_worker(rank, init_file, server_url, out_dir):
    """One training process of a two-process (gloo, CPU) run; records its adapter per step."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        rollout = SGLangRollout(server_url, sync_dir=os.path.join(out_dir, f"sync{rank}"))
        trainer = make_trainer(lora_model())
        trainer.accelerator.is_main_process = rank == 0
        trainer.accelerator.wait_for_everyone = dist.barrier
        names = []
        for step in range(2):
            trainer.state.global_step = step
            rollout(["a"], trainer)
            names.append(rollout.lora_name)
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
            json.dump(names, f)
        rollout.close()  # on_train_end runs on every rank
    finally:
        dist.destroy_process_group()


class TestAdapterSync:
    """Server launched with --enable-lora + LoRA policy: only the adapter is synced."""

    def test_every_rank_generates_with_the_synced_adapter(self, server, tmp_path):
        # Only rank 0 loads the adapter; rank 1 used to send /generate with no lora_path.
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available")
        FakeSGLang.enable_lora = True
        torch.multiprocessing.spawn(
            _two_rank_worker, args=(str(tmp_path / "init"), server, str(tmp_path)), nprocs=2
        )

        v1, v2 = (p["lora_name"] for p in calls("/load_lora_adapter"))
        for rank in (0, 1):
            assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == [v1, v2]
        assert sorted(p["lora_path"] for p in calls("/generate")) == [v1, v1, v2, v2]
        assert [p["lora_name"] for p in calls("/unload_lora_adapter")] == [v1, v2]

    def test_adapter_is_hot_loaded_and_previous_version_unloaded(self, server, tmp_path):
        FakeSGLang.enable_lora = True
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        trainer = make_trainer(lora_model())
        rollout(["a"], trainer)
        trainer.state.global_step = 1
        rollout(["a"], trainer)

        v1, v2 = f"{rollout.adapter_prefix}-v1", f"{rollout.adapter_prefix}-v2"
        assert [p["lora_name"] for p in calls("/load_lora_adapter")] == [v1, v2]
        assert calls("/unload_lora_adapter") == [{"lora_name": v1}]
        assert [p["lora_path"] for p in calls("/generate")] == [v1, v2]
        assert calls("/update_weights_from_disk") == []
        assert shards(tmp_path) == []  # no full-weight copy written

    def test_close_unloads_the_last_adapter(self, server, tmp_path):
        FakeSGLang.enable_lora = True
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        rollout(["a"], make_trainer(lora_model()))
        sglang_rollout.SGLangRolloutCallback(rollout).on_train_end(None, None, None)

        assert calls("/unload_lora_adapter") == [{"lora_name": f"{rollout.adapter_prefix}-v1"}]
        assert rollout.lora_name is None
        rollout.close()  # idempotent
        assert len(calls("/unload_lora_adapter")) == 1

    def test_close_does_not_raise_when_server_is_gone(self, tmp_path):
        rollout = SGLangRollout.__new__(SGLangRollout)
        rollout.client = SGLangClient(unused_url(), timeout=2)
        rollout.lora_name, rollout._synced_step, rollout._version = "aligntune-x-v1", 0, 1
        rollout.close()
        assert rollout.lora_name is None

    def test_runs_sharing_a_server_use_distinct_adapter_names(self, server, tmp_path):
        # Runs on one server must not collide, even if an earlier run did not close cleanly.
        FakeSGLang.enable_lora = True
        for _ in range(2):
            SGLangRollout(server, sync_dir=str(tmp_path))(["a"], make_trainer(lora_model()))
        first, second = (p["lora_name"] for p in calls("/load_lora_adapter"))
        assert first != second

    def test_adapter_mode_prefills_groups_once_without_n(self, server, tmp_path):
        # SGLang 0.5.10 leaks the adapter's ref count on n > 1 requests, so unload would hang.
        FakeSGLang.enable_lora = True
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        out = rollout(["ab", "ab", "abcd", "abcd"], make_trainer(lora_model()))

        warm, sample = calls("/generate")
        assert warm["input_ids"] == [[1, 1], [1] * 4]
        assert warm["sampling_params"]["max_new_tokens"] == 0
        assert sample["input_ids"] == [[1, 1], [1, 1], [1] * 4, [1] * 4]
        assert "n" not in sample["sampling_params"]
        assert warm["lora_path"] == sample["lora_path"] == f"{rollout.adapter_prefix}-v1"
        assert out["completion_ids"] == [[2, EOS], [2, EOS], [4, EOS], [4, EOS]]

    def test_full_model_policy_uses_full_sync_even_on_lora_server(self, server, tmp_path):
        FakeSGLang.enable_lora = True
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        rollout(["a"], make_trainer(tiny_model()))
        assert len(calls("/update_weights_from_disk")) == 1
        assert "lora_path" not in calls("/generate")[0]

    @pytest.mark.parametrize(
        "config", [{"use_rslora": True}, {"use_dora": True}, {"modules_to_save": ["lm_head"]}]
    )
    def test_adapter_features_sglang_cannot_serve_raise(self, server, tmp_path, config):
        FakeSGLang.enable_lora = True
        with pytest.raises(NotImplementedError, match="without --enable-lora"):
            SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(
                make_trainer(lora_model(**config))
            )

    def test_base_model_mismatch_raises(self, server, tmp_path):
        FakeSGLang.enable_lora = True
        FakeSGLang.model_path = "org/other-model"
        with pytest.raises(ValueError, match="base model"):
            SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(lora_model()))

    @pytest.mark.parametrize(
        "policy_base",
        [
            "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
            "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        ],
    )
    def test_prequantized_base_matches_full_precision_server(self, server, tmp_path, policy_base):
        # Unsloth loads load_in_4bit=True from its pre-quantized repo; the adapter still fits
        # the full-precision model the server serves.
        FakeSGLang.enable_lora = True
        FakeSGLang.model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        rollout._check_same_base(policy_base)
        with pytest.raises(ValueError, match="base model"):
            rollout._check_same_base("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")


class TestNoSilentFallback:
    def test_trainers_that_ignore_sglang_raise(self):
        from aligntune.backends.trl.rl.pace.pace import TRLPaceTrainer
        from aligntune.backends.unsloth.rl.pace.pace import UnslothPaceTrainer
        from aligntune.core.rl.trainer_base import RLTrainerBase, check_rollout_backend_supported

        config = SimpleNamespace(train=SimpleNamespace(rollout_backend="sglang"))
        for trainer_cls in (RLTrainerBase, TRLPaceTrainer, UnslothPaceTrainer):
            with pytest.raises(ValueError, match="not supported"):
                check_rollout_backend_supported(trainer_cls, config)

    def test_unsupported_error_does_not_send_users_to_one_backend(self):
        from aligntune.core.rl.trainer_base import RLTrainerBase, check_rollout_backend_supported

        config = SimpleNamespace(train=SimpleNamespace(rollout_backend="sglang"))
        with pytest.raises(ValueError) as error:
            check_rollout_backend_supported(RLTrainerBase, config)
        assert "backend='trl'" not in str(error.value)
        assert "counterfact_grpo" in str(error.value)

    def test_grpo_family_accepts_sglang(self):
        from aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo import (
            TRLCounterFactGRPOTrainer,
        )
        from aligntune.backends.trl.rl.dapo.dapo import TRLDAPOTrainer
        from aligntune.backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
        from aligntune.backends.trl.rl.grpo.grpo import TRLGRPOTrainer
        from aligntune.backends.unsloth.rl.counterfact_grpo.counterfact_grpo import (
            UnslothCounterFactGRPOTrainer,
        )
        from aligntune.backends.unsloth.rl.dapo.dapo import UnslothDAPOTrainer
        from aligntune.backends.unsloth.rl.gbmpo.gbmpo import UnslothGBMPOTrainer
        from aligntune.backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
        from aligntune.core.rl.trainer_base import check_rollout_backend_supported

        config = SimpleNamespace(train=SimpleNamespace(rollout_backend="sglang"))
        for trainer_cls in (
            TRLGRPOTrainer,
            TRLDAPOTrainer,
            TRLGBMPOTrainer,
            UnslothGRPOTrainer,
            UnslothDAPOTrainer,
            UnslothGBMPOTrainer,
            TRLCounterFactGRPOTrainer,
            UnslothCounterFactGRPOTrainer,
        ):
            check_rollout_backend_supported(trainer_cls, config)


def test_checkpoint_names_match_save_pretrained_for_renamed_models(tmp_path):
    """Gemma 3 is renamed on load by transformers 5; SGLang loads the hub names."""
    from safetensors import safe_open
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        SiglipVisionConfig,
    )

    from aligntune.backends.trl.rl.rollout.sglang_rollout import checkpoint_names

    text = Gemma3TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, vocab_size=100,
    )
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14,
    )
    model = Gemma3ForConditionalGeneration(
        Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=1)
    )
    model.save_pretrained(tmp_path)
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as f:
        saved = set(f.keys())

    names = set(checkpoint_names(model, dict(model.named_parameters())))
    assert names - {"lm_head.weight"} == saved  # lm_head is tied to the embeddings


def test_drop_non_text_adapter_weights_keeps_language_model_tensors(tmp_path):
    from safetensors.torch import load_file, save_file

    from aligntune.backends.trl.rl.rollout.sglang_rollout import drop_non_text_adapter_weights

    text = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    vision = "base_model.model.model.vision_tower.encoder.layers.0.q_proj.lora_A.weight"
    save_file(
        {text: torch.zeros(2, 4), vision: torch.zeros(2, 3)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    assert drop_non_text_adapter_weights(str(tmp_path)) == 1
    assert list(load_file(str(tmp_path / "adapter_model.safetensors"))) == [text]
    assert drop_non_text_adapter_weights(str(tmp_path)) == 0


def test_drop_non_text_adapter_weights_matches_whole_module_names(tmp_path):
    from safetensors.torch import load_file, save_file

    from aligntune.backends.trl.rl.rollout.sglang_rollout import drop_non_text_adapter_weights

    keep = {
        f"base_model.model.model.layers.0.{module}.q_proj.lora_A.weight": torch.zeros(2, 4)
        for module in ("visual_attn", "audio_gate", "cross_vision_model_attn")
    }
    drop = {
        f"base_model.model.model.{module}.layers.0.q_proj.lora_A.weight": torch.zeros(2, 3)
        for module in (
            "vision_tower", "vision_model", "vision_encoder", "visual", "multi_modal_projector",
            "audio", "audio_tower", "audio_model", "audio_encoder",
        )
    }
    save_file({**keep, **drop}, str(tmp_path / "adapter_model.safetensors"))
    assert drop_non_text_adapter_weights(str(tmp_path)) == len(drop)
    assert set(load_file(str(tmp_path / "adapter_model.safetensors"))) == set(keep)


def test_full_sync_refuses_8bit_models(server, tmp_path):
    model = tiny_model()
    model.is_loaded_in_8bit = True  # what transformers sets for 8-bit loads
    with pytest.raises(NotImplementedError, match="8-bit"):
        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(model))


def test_regex_target_modules_become_an_explicit_list(tmp_path):
    import json

    from safetensors.torch import save_file

    from aligntune.backends.trl.rl.rollout.sglang_rollout import explicit_target_modules

    prefix = "base_model.model.model.language_model.layers.0"
    save_file(
        {
            f"{prefix}.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
            f"{prefix}.self_attn.q_proj.lora_B.weight": torch.zeros(4, 2),
            f"{prefix}.mlp.down_proj.lora_A.weight": torch.zeros(2, 4),
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    config = tmp_path / "adapter_config.json"
    regex = "(?:.*?language_model.*?(?:q_proj|down_proj))"
    config.write_text(json.dumps({"target_modules": regex}))
    assert explicit_target_modules(str(tmp_path))
    assert json.loads(config.read_text())["target_modules"] == ["down_proj", "q_proj"]
    assert not explicit_target_modules(str(tmp_path))  # a list is left alone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 4-bit needs a GPU")
def test_full_sync_dequantizes_4bit_weights(tmp_path):
    pytest.importorskip("bitsandbytes")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from aligntune.backends.trl.rl.rollout.sglang_rollout import (
        checkpoint_names,
        export_policy_state_dict,
    )

    dense = tiny_model().to(torch.bfloat16)
    dense.save_pretrained(tmp_path / "dense")
    quantized = AutoModelForCausalLM.from_pretrained(
        tmp_path / "dense",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="cuda",
    )
    exported = checkpoint_names(quantized, export_policy_state_dict(quantized))
    for name, tensor in dense.state_dict().items():
        if name in exported:
            assert exported[name].shape == tensor.shape, name
            assert exported[name].dtype == torch.bfloat16, name
            assert (exported[name].float().cpu() - tensor.float()).abs().mean() < 0.05, name


def tiny_gemma3():
    """A multimodal model that transformers 5 renames on load (the Gemma 3 weight-name case)."""
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        SiglipVisionConfig,
    )

    torch.manual_seed(0)
    text = Gemma3TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, vocab_size=100,
    )
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14,
    )
    return Gemma3ForConditionalGeneration(
        Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=1)
    )


class TestSyncWiring:
    """sync_weights and _sync_adapter call the helpers that make a real server accept the files."""

    def test_full_sync_writes_the_names_save_pretrained_writes(self, server, tmp_path):
        from safetensors import safe_open

        model = tiny_gemma3()
        model.save_pretrained(tmp_path / "saved")
        with safe_open(str(tmp_path / "saved" / "model.safetensors"), "pt") as f:
            saved = set(f.keys())

        sync_dir = tmp_path / "sync"
        SGLangRollout(server, sync_dir=str(sync_dir)).sync_weights(make_trainer(model))
        assert set(load_shards(str(sync_dir))) - {"lm_head.weight"} == saved

    def adapter_sent_to_server(self, server, tmp_path, target_modules):
        from peft import LoraConfig, get_peft_model
        from safetensors import safe_open

        FakeSGLang.enable_lora = True
        model = tiny_gemma3()
        model.config._name_or_path = FakeSGLang.model_path
        model = get_peft_model(model, LoraConfig(r=2, target_modules=target_modules))
        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(model))
        path = calls("/load_lora_adapter")[-1]["lora_path"]
        with safe_open(os.path.join(path, "adapter_model.safetensors"), "pt") as f:
            keys = set(f.keys())
        with open(os.path.join(path, "adapter_config.json")) as f:
            return keys, json.load(f)["target_modules"]

    def test_adapter_sync_leaves_out_vision_tensors(self, server, tmp_path):
        keys, _ = self.adapter_sent_to_server(server, tmp_path, ["q_proj"])
        assert keys
        assert not [k for k in keys if "vision_tower" in k]

    def test_adapter_sync_rewrites_a_regex_target_modules(self, server, tmp_path):
        _, target_modules = self.adapter_sent_to_server(
            server, tmp_path, ".*language_model.*q_proj"
        )
        assert target_modules == ["q_proj"]


class _StopAfterSetup(Exception):
    pass


class _RecordingTrainer:
    """Stands in for TRL's trainers: records its kwargs and callbacks."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]
        self.callbacks = []
        type(self).last = self

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def train(self, *args, **kwargs):  # TRLCounterFactGRPOTrainer builds its trainer in train()
        raise _StopAfterSetup

    def compute_loss(self):  # Unsloth counterfactual patches trl.GRPOTrainer with these two
        pass

    _compute_loss_impl = compute_loss


TRAINERS_WITH_SGLANG = [
    "aligntune.backends.trl.rl.grpo.grpo:TRLGRPOTrainer",
    "aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo:TRLCounterFactGRPOTrainer",
    "aligntune.backends.unsloth.rl.grpo.grpo:UnslothGRPOTrainer",
    "aligntune.backends.unsloth.rl.counterfact_grpo.counterfact_grpo:UnslothCounterFactGRPOTrainer",
]


@pytest.mark.parametrize("path", TRAINERS_WITH_SGLANG, ids=lambda p: p.split(":")[1])
def test_setup_trainer_hands_the_rollout_to_trl(path, server, tmp_path, monkeypatch):
    import importlib
    from unittest import mock

    import trl

    from aligntune.backends.trl.rl.counterfact_grpo import custom_trainer as trl_custom
    from aligntune.backends.unsloth.rl.counterfact_grpo import custom_trainer as unsloth_custom

    module, name = path.split(":")
    monkeypatch.setitem(sys.modules, "unsloth", None)  # importing it raises without a GPU
    _RecordingTrainer.last = None
    monkeypatch.setattr(trl, "GRPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(trl_custom, "CounterfactualGRPOTrainer", _RecordingTrainer)
    monkeypatch.setattr(unsloth_custom, "CounterfactualGRPOTrainer", _RecordingTrainer)
    trainer = object.__new__(getattr(importlib.import_module(module), name))
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            rollout_backend="sglang", sglang_server_url=server, sglang_sync_dir=str(tmp_path),
            use_cpu=True, max_steps=1, learning_rate=1e-5, weight_decay=0.0, num_generations=2,
            per_device_batch_size=2,
        ),
        model=SimpleNamespace(),
        logging=SimpleNamespace(output_dir=str(tmp_path), loggers=[]),
        rewards=[],
    )
    for attribute in ("model", "unsloth_model", "tokenizer"):
        setattr(trainer, attribute, mock.MagicMock())
    for step in ("setup_model", "setup_rewards", "setup_data"):  # train() calls these first
        setattr(trainer, step, lambda: None)
    trainer.train_dataset, trainer.eval_dataset = [{"prompt": "a"}], None
    trainer.reward_functions = [lambda completions, **kwargs: [0.0] * len(completions)]
    trainer.prepared_rewards = SimpleNamespace(functions=trainer.reward_functions, weights=[1.0])

    with contextlib.suppress(_StopAfterSetup):
        (trainer.setup_trainer if hasattr(trainer, "setup_trainer") else trainer.train)()

    built = _RecordingTrainer.last
    assert isinstance(built.kwargs["rollout_func"], SGLangRollout)
    assert built.kwargs["rollout_func"].client.base_url == server
    assert [type(c).__name__ for c in built.callbacks].count("SGLangRolloutCallback") == 1
