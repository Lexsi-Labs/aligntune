"""
Unit tests for the SGLang rollout backend (TRL GRPO rollout_func).

A tiny in-process HTTP server stands in for SGLang, so these run on CPU
without sglang installed.
"""

import json
import os
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from aligntune.backends.trl.rl.rollout.sglang_rollout import (
    WEIGHTS_FILE,
    SGLangClient,
    SGLangRollout,
    export_policy_state_dict,
)

EOS = 2


class FakeSGLang(BaseHTTPRequestHandler):
    """Echo server: each completion is [len(prompt), EOS]."""

    calls = []
    weight_update_ok = True

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
            return self._reply(200, {"model_path": "tiny-model"})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, payload))
        if self.path == "/generate":
            if payload["sampling_params"].get("max_new_tokens", 1) <= 0:
                return self._reply(400, {"error": "bad max_new_tokens"})
            n = payload["sampling_params"].get("n", 1)
            return self._reply(200, [{"output_ids": [len(ids), EOS]} for ids in payload["input_ids"] for _ in range(n)])
        if self.path == "/update_weights_from_disk":
            ok = self.weight_update_ok and os.path.exists(os.path.join(payload["model_path"], WEIGHTS_FILE))
            return self._reply(200, {"success": ok, "message": "ok" if ok else "load failed"})
        self._reply(404, {"error": "not found"})


@pytest.fixture
def server():
    FakeSGLang.calls = []
    FakeSGLang.weight_update_ok = True
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
    tokenize = lambda prompts: ([[1] * len(p) for p in prompts], None, {})
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
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=tie,
    )
    return LlamaForCausalLM(config)


class TestSGLangClient:
    def test_unreachable_server_raises_connection_error(self):
        with pytest.raises(ConnectionError, match="unreachable"):
            SGLangClient(unused_url(), timeout=2).get_model_info()

    def test_http_error_is_raised_not_swallowed(self, server):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            SGLangClient(server).generate([[1, 2]], {"max_new_tokens": 0})

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
            "max_new_tokens": 16, "temperature": 0.7, "top_p": 0.95, "top_k": -1,
            "min_p": 0.0, "repetition_penalty": 1.0, "stop_token_ids": [EOS],
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
        from safetensors.torch import load_file

        model = tiny_model(tie=False)
        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(model))
        saved = load_file(str(tmp_path / WEIGHTS_FILE))
        expected = dict(model.named_parameters())
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
        from safetensors.torch import load_file

        base = tiny_model(tie=False)
        base_names = {n for n, _ in base.named_parameters()}
        lora = peft.get_peft_model(
            base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        )
        q_layer = lora.base_model.model.model.layers[0].self_attn.q_proj
        q_before = q_layer.base_layer.weight.detach().clone()
        delta = (q_layer.lora_B["default"].weight @ q_layer.lora_A["default"].weight) * q_layer.scaling["default"]

        SGLangRollout(server, sync_dir=str(tmp_path)).sync_weights(make_trainer(lora))
        saved = load_file(str(tmp_path / WEIGHTS_FILE))

        assert set(saved) == base_names
        q = "model.layers.0.self_attn.q_proj.weight"
        assert torch.allclose(saved[q], q_before + delta.detach(), atol=1e-6)  # adapter folded in
        assert torch.equal(q_layer.base_layer.weight, q_before)  # trainer's model is not mutated
