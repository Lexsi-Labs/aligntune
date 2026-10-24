"""
Unit tests for the SGLang rollout backend (TRL GRPO rollout_func).

A tiny in-process HTTP server stands in for SGLang, so these run on CPU
without sglang installed.
"""

import json
import os
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


class TestAdapterSync:
    """Server launched with --enable-lora + LoRA policy: only the adapter is synced."""

    def test_adapter_is_hot_loaded_and_previous_version_unloaded(self, server, tmp_path):
        FakeSGLang.enable_lora = True
        rollout = SGLangRollout(server, sync_dir=str(tmp_path))
        trainer = make_trainer(lora_model())
        rollout(["a"], trainer)
        trainer.state.global_step = 1
        rollout(["a"], trainer)

        assert [p["lora_name"] for p in calls("/load_lora_adapter")] == [
            "aligntune-policy-v1",
            "aligntune-policy-v2",
        ]
        assert calls("/unload_lora_adapter") == [{"lora_name": "aligntune-policy-v1"}]
        assert [p["lora_path"] for p in calls("/generate")] == [
            "aligntune-policy-v1",
            "aligntune-policy-v2",
        ]
        assert calls("/update_weights_from_disk") == []
        assert shards(tmp_path) == []  # no full-weight copy written

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
        assert warm["lora_path"] == sample["lora_path"] == "aligntune-policy-v1"
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


class TestNoSilentFallback:
    def test_trainers_that_ignore_sglang_raise(self):
        from aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo import (
            TRLCounterFactGRPOTrainer,
        )
        from aligntune.backends.trl.rl.pace.pace import TRLPaceTrainer
        from aligntune.backends.unsloth.rl.dapo.dapo import UnslothDAPOTrainer
        from aligntune.backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
        from aligntune.core.rl.trainer_base import RLTrainerBase, check_rollout_backend_supported

        config = SimpleNamespace(train=SimpleNamespace(rollout_backend="sglang"))
        for trainer_cls in (
            RLTrainerBase,
            TRLPaceTrainer,
            TRLCounterFactGRPOTrainer,
            UnslothGRPOTrainer,
            UnslothDAPOTrainer,
        ):
            with pytest.raises(ValueError, match="not supported"):
                check_rollout_backend_supported(trainer_cls, config)

    def test_trl_grpo_family_accepts_sglang(self):
        from aligntune.backends.trl.rl.dapo.dapo import TRLDAPOTrainer
        from aligntune.backends.trl.rl.gbmpo.gbmpo import TRLGBMPOTrainer
        from aligntune.backends.trl.rl.grpo.grpo import TRLGRPOTrainer
        from aligntune.core.rl.trainer_base import check_rollout_backend_supported

        config = SimpleNamespace(train=SimpleNamespace(rollout_backend="sglang"))
        for trainer_cls in (TRLGRPOTrainer, TRLDAPOTrainer, TRLGBMPOTrainer):
            check_rollout_backend_supported(trainer_cls, config)
