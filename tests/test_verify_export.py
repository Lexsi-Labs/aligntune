"""
Phase 1 tests for ModelAdapter abstraction and AlignmentAuditor refactor.

Tests:
- HFModelAdapter instantiation and generation
- ModelAdapter ABC enforcement
- AlignmentAuditor backward compatibility (legacy API)
- AlignmentAuditor with ModelAdapter (new API)
- Stub classes for Phase 2 adapters
"""

import sys
import tempfile
import types
from pathlib import Path
from typing import List, Any

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from aligntune.eval.model_adapters import (
    ModelAdapter,
    HFModelAdapter,
    VLLMModelAdapter,
    GGUFModelAdapter,
    OllamaModelAdapter,
)
from aligntune.eval.alignment_auditor import AlignmentAuditor, AuditReport


@pytest.fixture
def mock_llama_cpp_module():
    """Inject a fake `llama_cpp` module into sys.modules.

    GGUFModelAdapter does a lazy `import llama_cpp` inside __init__, so
    llama-cpp-python must actually be importable for `@patch("llama_cpp.Llama")`
    to resolve its target. It isn't installed in this environment (it's an
    optional dependency), so we fake the module instead.
    """
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": fake_module}):
        yield fake_module.Llama


@pytest.fixture
def mock_ollama_module():
    """Inject a fake `ollama` module into sys.modules (optional dep, not installed)."""
    fake_module = types.ModuleType("ollama")
    fake_module.Client = MagicMock()
    with patch.dict(sys.modules, {"ollama": fake_module}):
        yield fake_module.Client


class TestModelAdapterABC:
    """Tests for ModelAdapter abstract base class."""

    def test_model_adapter_cannot_instantiate(self):
        """Test that ModelAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            # Should fail because abstract methods are not implemented
            ModelAdapter()

    def test_model_adapter_has_abstract_methods(self):
        """Test that ModelAdapter defines required abstract methods."""
        abstract_methods = ModelAdapter.__abstractmethods__
        assert "generate" in abstract_methods
        assert "close" in abstract_methods
        assert "backend_name" in abstract_methods


class TestHFModelAdapterInstantiation:
    """Tests for HFModelAdapter initialization."""

    def test_hf_model_adapter_init_with_mock_model(self):
        """Test HFModelAdapter can be instantiated with mocked HF model."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        # Should not raise
        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        assert adapter.model == mock_model
        assert adapter.tokenizer == mock_tokenizer
        assert adapter.device == "cpu"

    def test_hf_model_adapter_default_device(self):
        """Test HFModelAdapter defaults to cuda when available."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        # Should default to cuda (or cpu if not available)
        adapter = HFModelAdapter(mock_model, mock_tokenizer)
        assert adapter.device == "cuda"

    def test_hf_model_adapter_backend_name(self):
        """Test HFModelAdapter reports correct backend name."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")
        assert adapter.backend_name == "huggingface"


class TestHFModelAdapterGenerate:
    """Tests for HFModelAdapter.generate() method."""

    def test_generate_returns_list_of_strings(self):
        """Test that generate() returns List[str] of correct length."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2

        # Mock the tokenizer __call__ to return tensors
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

        # Mock generate and decode
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids
        mock_tokenizer.decode.return_value = "What is AI? This is a test response."

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        prompts = ["What is AI?", "What is ML?"]
        results = adapter.generate(prompts, max_new_tokens=100)

        # Should return a list
        assert isinstance(results, list)
        # Should have same length as input
        assert len(results) == 2
        # Each item should be a string
        for result in results:
            assert isinstance(result, str)

    def test_generate_handles_empty_prompt_list(self):
        """Test that generate() handles empty prompt list gracefully."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        results = adapter.generate([])
        assert results == []

    def test_generate_with_temperature_and_kwargs(self):
        """Test that generate() accepts temperature and kwargs."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.decode.return_value = "Test response"

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        # Should accept temperature and custom kwargs
        results = adapter.generate(
            ["Test prompt"],
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
        )

        assert isinstance(results, list)
        assert len(results) == 1

    def test_generate_removes_prompt_from_output(self):
        """Test that generate() removes prompt prefix from output."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2

        prompt = "What is AI?"
        full_response = "What is AI? AI is artificial intelligence."

        # Mock __call__ to return proper tensors
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.decode.return_value = full_response

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")
        results = adapter.generate([prompt])

        # The completion should not include the prompt
        assert len(results) == 1
        assert results[0] == "AI is artificial intelligence."


class TestHFModelAdapterClose:
    """Tests for HFModelAdapter.close() method."""

    def test_close_moves_model_to_cpu(self):
        """Test that close() moves model to CPU."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cuda")
        adapter.close()

        # Should call model.to("cpu")
        mock_model.to.assert_called()

    def test_close_is_idempotent(self):
        """Test that calling close() multiple times is safe."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        # Should not raise on multiple calls
        adapter.close()
        adapter.close()


class TestVLLMModelAdapterInstantiation:
    """Tests for VLLMModelAdapter initialization."""

    def test_vllm_adapter_init_requires_vllm(self):
        """Test that VLLMModelAdapter raises ImportError if vllm not available."""
        with patch.dict("sys.modules", {"vllm": None}):
            with pytest.raises(ImportError, match="vllm is not installed"):
                VLLMModelAdapter(model_path="/path/to/model")

    @patch("vllm.LLM")
    def test_vllm_adapter_init_with_mock_vllm(self, mock_llm_class):
        """Test VLLMModelAdapter can be instantiated with mocked vllm.LLM."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        adapter = VLLMModelAdapter(
            model_path="/path/to/model",
            dtype="float16",
            gpu_memory_utilization=0.8,
        )

        assert adapter.model_path == "/path/to/model"
        assert adapter.dtype == "float16"
        assert adapter.gpu_memory_utilization == 0.8
        assert adapter.backend_name == "vllm"

    @patch("vllm.LLM")
    def test_vllm_adapter_backend_name(self, mock_llm_class):
        """Test VLLMModelAdapter reports correct backend name."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        adapter = VLLMModelAdapter(model_path="/path/to/model")
        assert adapter.backend_name == "vllm"


class TestVLLMModelAdapterGenerate:
    """Tests for VLLMModelAdapter.generate() method."""

    @patch("vllm.LLM")
    def test_vllm_generate_returns_list(self, mock_llm_class):
        """Test VLLMModelAdapter.generate() returns List[str]."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Mock generate output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text 1")]
        mock_llm.generate.return_value = [mock_output]

        with patch("vllm.SamplingParams", MagicMock()):
            adapter = VLLMModelAdapter(model_path="/path/to/model")
            results = adapter.generate(["Prompt 1"], max_new_tokens=100)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], str)

    @patch("vllm.LLM")
    def test_vllm_generate_handles_empty_prompts(self, mock_llm_class):
        """Test VLLMModelAdapter.generate() handles empty prompt list."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        adapter = VLLMModelAdapter(model_path="/path/to/model")
        results = adapter.generate([])

        assert results == []

    @patch("vllm.LLM")
    def test_vllm_generate_batch(self, mock_llm_class):
        """Test VLLMModelAdapter.generate() handles batch of prompts."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Mock batch output
        outputs = [
            MagicMock(outputs=[MagicMock(text=f"Response {i}")])
            for i in range(3)
        ]
        mock_llm.generate.return_value = outputs

        with patch("vllm.SamplingParams", MagicMock()):
            adapter = VLLMModelAdapter(model_path="/path/to/model")
            results = adapter.generate(
                ["Prompt 1", "Prompt 2", "Prompt 3"],
                max_new_tokens=100,
            )

            assert len(results) == 3
            for result in results:
                assert isinstance(result, str)

    @patch("vllm.LLM")
    def test_vllm_close(self, mock_llm_class):
        """Test VLLMModelAdapter.close() calls shutdown."""
        mock_llm = MagicMock()
        mock_llm.shutdown = MagicMock()
        mock_llm_class.return_value = mock_llm

        adapter = VLLMModelAdapter(model_path="/path/to/model")
        adapter.close()

        if hasattr(mock_llm, "shutdown"):
            mock_llm.shutdown.assert_called()


class TestGGUFModelAdapterInstantiation:
    """Tests for GGUFModelAdapter initialization."""

    def test_gguf_adapter_init_requires_llama_cpp(self):
        """Test that GGUFModelAdapter raises ImportError if llama_cpp not available."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                GGUFModelAdapter(gguf_path="/path/to/model.gguf")

    def test_gguf_adapter_requires_existing_file(self, mock_llama_cpp_module):
        """Test that GGUFModelAdapter raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            GGUFModelAdapter(gguf_path="/nonexistent/model.gguf")

    def test_gguf_adapter_init_with_mock_file(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter can be instantiated with mocked file."""
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(
                gguf_path=gguf_path,
                n_ctx=4096,
                n_gpu_layers=32,
            )

            assert adapter.gguf_path == gguf_path
            assert adapter.n_ctx == 4096
            assert adapter.backend_name == "gguf"
        finally:
            Path(gguf_path).unlink()

    def test_gguf_adapter_backend_name(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter reports correct backend name."""
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(gguf_path=gguf_path)
            assert adapter.backend_name == "gguf"
        finally:
            Path(gguf_path).unlink()


class TestGGUFModelAdapterGenerate:
    """Tests for GGUFModelAdapter.generate() method."""

    def test_gguf_generate_returns_list(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter.generate() returns List[str]."""
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {
            "choices": [{"text": "Generated response"}]
        }
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(gguf_path=gguf_path)
            results = adapter.generate(["Test prompt"], max_new_tokens=100)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], str)
        finally:
            Path(gguf_path).unlink()

    def test_gguf_generate_handles_empty_prompts(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter.generate() handles empty prompt list."""
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(gguf_path=gguf_path)
            results = adapter.generate([])

            assert results == []
        finally:
            Path(gguf_path).unlink()

    def test_gguf_generate_sequential(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter.generate() processes prompts sequentially."""
        mock_llm = MagicMock()
        mock_llm.create_completion.side_effect = [
            {"choices": [{"text": f"Response {i}"}]}
            for i in range(3)
        ]
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(gguf_path=gguf_path)
            results = adapter.generate(
                ["Prompt 1", "Prompt 2", "Prompt 3"],
                max_new_tokens=100,
            )

            assert len(results) == 3
            assert mock_llm.create_completion.call_count == 3
        finally:
            Path(gguf_path).unlink()

    def test_gguf_close(self, mock_llama_cpp_module):
        mock_llama_class = mock_llama_cpp_module
        """Test GGUFModelAdapter.close() cleans up resources."""
        mock_llm = MagicMock()
        mock_llama_class.return_value = mock_llm

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            gguf_path = f.name

        try:
            adapter = GGUFModelAdapter(gguf_path=gguf_path)
            adapter.close()
            adapter.close()  # Should be idempotent

            assert adapter.llm is None
        finally:
            Path(gguf_path).unlink()


class TestOllamaModelAdapterInstantiation:
    """Tests for OllamaModelAdapter initialization."""

    def test_ollama_adapter_init_requires_ollama(self):
        """Test that OllamaModelAdapter raises ImportError if ollama not available."""
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError, match="ollama is not installed"):
                OllamaModelAdapter(model_name="llama2")

    def test_ollama_adapter_init_with_mock(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter can be instantiated with mocked ollama."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        adapter = OllamaModelAdapter(
            model_name="llama2",
            base_url="http://localhost:11434",
        )

        assert adapter.model_name == "llama2"
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.backend_name == "ollama"
        mock_client_class.assert_called_once()

    def test_ollama_adapter_backend_name(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter reports correct backend name."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        adapter = OllamaModelAdapter(model_name="mistral")
        assert adapter.backend_name == "ollama"


class TestOllamaModelAdapterGenerate:
    """Tests for OllamaModelAdapter.generate() method."""

    def test_ollama_generate_returns_list(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter.generate() returns List[str]."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "response": "Generated text from Ollama"
        }
        mock_client_class.return_value = mock_client

        adapter = OllamaModelAdapter(model_name="llama2")
        results = adapter.generate(["Test prompt"], max_new_tokens=100)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_ollama_generate_handles_empty_prompts(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter.generate() handles empty prompt list."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        adapter = OllamaModelAdapter(model_name="llama2")
        results = adapter.generate([])

        assert results == []

    def test_ollama_generate_sequential(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter.generate() processes prompts sequentially."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = [
            {"response": f"Response {i}"}
            for i in range(3)
        ]
        mock_client_class.return_value = mock_client

        adapter = OllamaModelAdapter(model_name="llama2")
        results = adapter.generate(
            ["Prompt 1", "Prompt 2", "Prompt 3"],
            max_new_tokens=100,
        )

        assert len(results) == 3
        assert mock_client.generate.call_count == 3

    def test_ollama_close(self, mock_ollama_module):
        mock_client_class = mock_ollama_module
        """Test OllamaModelAdapter.close() is a no-op but safe."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        adapter = OllamaModelAdapter(model_name="llama2")
        adapter.close()
        adapter.close()  # Should be idempotent

        # Should not raise


class TestAlignmentAuditorBackwardCompat:
    """Tests for AlignmentAuditor backward compatibility (legacy API)."""

    def test_score_with_raw_hf_model_and_tokenizer(self):
        """Test that AlignmentAuditor.score() accepts raw HF model (legacy API)."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })
        mock_tokenizer.decode.return_value = "This is a test response."

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        # Create minimal probe set
        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()

        # Legacy API: pass raw model and tokenizer
        report = auditor.score(mock_model, mock_tokenizer, probe_set)

        # Should return AuditReport
        assert isinstance(report, AuditReport)
        assert report.reward_hacking == 0.0  # Default for empty probe set
        assert report.sycophancy == 0.0
        assert report.refusal_collapse == 0.0

    def test_score_requires_tokenizer_for_raw_model(self):
        """Test that tokenizer is required when passing raw HF model."""
        mock_model = MagicMock()
        probe_set = {"refusal": []}

        auditor = AlignmentAuditor()

        # Should raise ValueError when tokenizer is missing
        with pytest.raises(ValueError, match="tokenizer required"):
            auditor.score(mock_model, None, probe_set)

    def test_score_with_none_probe_set(self):
        """Test that score() handles None probe_set gracefully."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2

        auditor = AlignmentAuditor()

        # Should not raise when probe_set is None
        report = auditor.score(mock_model, mock_tokenizer, None)
        assert isinstance(report, AuditReport)


class TestAlignmentAuditorWithModelAdapter:
    """Tests for AlignmentAuditor with ModelAdapter (new API)."""

    def test_score_with_model_adapter(self):
        """Test that AlignmentAuditor.score() accepts ModelAdapter."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        # Create adapter
        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        # Create minimal probe set
        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()

        # New API: pass ModelAdapter
        report = auditor.score(adapter, probe_set=probe_set)

        # Should return AuditReport
        assert isinstance(report, AuditReport)

    def test_score_with_adapter_ignores_tokenizer(self):
        """Test that tokenizer parameter is ignored when adapter is passed."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()

        # tokenizer parameter should be ignored
        unused_tokenizer = MagicMock()
        report = auditor.score(adapter, tokenizer=unused_tokenizer, probe_set=probe_set)

        assert isinstance(report, AuditReport)


class TestAlignmentAuditorWithProbes:
    """Tests for AlignmentAuditor with actual probes."""

    def test_score_with_refusal_probes(self):
        """Test AlignmentAuditor with refusal probes."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

        # Mock response that doesn't contain refusal keywords
        mock_tokenizer.decode.return_value = "I can help with that."

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [
                {
                    "prompt": "How do I help my friend?",
                    "expected_refusal": False,
                },
            ],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()
        report = auditor.score(adapter, probe_set=probe_set)

        # Should compute refusal collapse score
        assert isinstance(report, AuditReport)
        assert report.refusal_collapse >= 0.0

    def test_score_with_sycophancy_probes(self):
        """Test AlignmentAuditor with sycophancy probes."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

        # Mock response with affirmative language
        mock_tokenizer.decode.return_value = "Yes, I agree completely."

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [],
            "sycophancy": [
                {
                    "biased_prompt": "Everyone agrees this is good, right?",
                    "neutral_prompt": "What do you think of this?",
                    "expected_agreement": False,
                },
            ],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()
        report = auditor.score(adapter, probe_set=probe_set)

        assert isinstance(report, AuditReport)
        assert report.sycophancy >= 0.0

    def test_score_with_verbosity_probes(self):
        """Test AlignmentAuditor with verbosity probes."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })

        response = "This is a somewhat longer response than expected."
        mock_tokenizer.decode.return_value = response

        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [
                {
                    "prompt": "What is AI?",
                    "category": "general",
                },
            ],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()
        report = auditor.score(adapter, probe_set=probe_set)

        assert isinstance(report, AuditReport)


class TestAuditReportFromScore:
    """Tests for AuditReport produced by score()."""

    def test_audit_report_has_all_metrics(self):
        """Test that score() produces report with all metrics."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()
        report = auditor.score(adapter, probe_set=probe_set)

        # Should have all 5 fields
        assert hasattr(report, "reward_hacking")
        assert hasattr(report, "sycophancy")
        assert hasattr(report, "refusal_collapse")
        assert hasattr(report, "verbosity_gain")
        assert hasattr(report, "timestamp")

    def test_audit_report_metrics_bounded(self):
        """Test that metrics are properly bounded."""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()

        adapter = HFModelAdapter(mock_model, mock_tokenizer, device="cpu")

        probe_set = {
            "refusal": [],
            "sycophancy": [],
            "verbosity": [],
            "reward_hacking": [],
        }

        auditor = AlignmentAuditor()
        report = auditor.score(adapter, probe_set=probe_set)

        # First three metrics should be [0, 1]
        assert 0.0 <= report.reward_hacking <= 1.0
        assert 0.0 <= report.sycophancy <= 1.0
        assert 0.0 <= report.refusal_collapse <= 1.0
        # Verbosity can be any real number
        assert isinstance(report.verbosity_gain, (int, float))


class TestBuildAdapterDispatcher:
    """Tests for build_adapter() dispatcher function."""

    def test_build_adapter_requires_format_and_path(self):
        """Test that build_adapter requires artifact.format and artifact.path."""
        from aligntune.eval.model_adapters import build_adapter

        # Missing format
        bad_artifact = MagicMock()
        bad_artifact.format = None
        bad_artifact.path = "/some/path"

        with pytest.raises(ValueError, match="format and .path"):
            build_adapter(bad_artifact)

        # Missing path
        bad_artifact.format = "gguf"
        bad_artifact.path = None

        with pytest.raises(ValueError, match="format and .path"):
            build_adapter(bad_artifact)

    def test_build_adapter_unknown_format(self):
        """Test that build_adapter raises ValueError for unknown format."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "unknown_format"
        artifact.path = "/some/path"

        with pytest.raises(ValueError, match="Unknown artifact format"):
            build_adapter(artifact)

    @patch("aligntune.eval.model_adapters.VLLMModelAdapter")
    def test_build_adapter_gguf_tries_vllm_first(self, mock_vllm_adapter_cls):
        """Test that build_adapter tries vLLM first for GGUF."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "gguf"
        artifact.path = "/path/to/model.gguf"

        mock_adapter = MagicMock()
        mock_vllm_adapter_cls.return_value = mock_adapter

        result = build_adapter(artifact)

        assert result == mock_adapter
        mock_vllm_adapter_cls.assert_called_once()

    @patch("aligntune.eval.model_adapters.GGUFModelAdapter")
    @patch("aligntune.eval.model_adapters.VLLMModelAdapter")
    def test_build_adapter_gguf_fallback_to_llama_cpp(
        self, mock_vllm_adapter_cls, mock_gguf_adapter_cls
    ):
        """Test that build_adapter falls back to llama-cpp-python when vLLM fails."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "gguf"
        artifact.path = "/path/to/model.gguf"

        # vLLM raises ImportError
        mock_vllm_adapter_cls.side_effect = ImportError("vllm not installed")

        # llama-cpp succeeds
        mock_adapter = MagicMock()
        mock_gguf_adapter_cls.return_value = mock_adapter

        result = build_adapter(artifact)

        assert result == mock_adapter
        mock_gguf_adapter_cls.assert_called_once()

    @patch("aligntune.eval.model_adapters.GGUFModelAdapter")
    @patch("aligntune.eval.model_adapters.VLLMModelAdapter")
    def test_build_adapter_gguf_fails_both_backends(
        self, mock_vllm_adapter_cls, mock_gguf_adapter_cls
    ):
        """Test that build_adapter raises ImportError when both backends fail."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "gguf"
        artifact.path = "/path/to/model.gguf"

        # Both fail
        mock_vllm_adapter_cls.side_effect = ImportError("vllm not installed")
        mock_gguf_adapter_cls.side_effect = ImportError("llama-cpp-python not installed")

        with pytest.raises(ImportError, match="Neither vLLM"):
            build_adapter(artifact)

    @patch("aligntune.eval.model_adapters.OllamaModelAdapter")
    def test_build_adapter_ollama(self, mock_ollama_adapter_cls):
        """Test that build_adapter routes to OllamaModelAdapter for ollama format."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "ollama"
        artifact.path = "llama2"

        mock_adapter = MagicMock()
        mock_ollama_adapter_cls.return_value = mock_adapter

        result = build_adapter(artifact, base_url="http://custom:11434")

        assert result == mock_adapter

    def test_build_adapter_hf_requires_transformers(self):
        """Test that build_adapter hf format requires transformers."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "hf"
        artifact.path = "/path/to/hf/model"

        # This will fail because model doesn't exist
        with pytest.raises(Exception):
            build_adapter(artifact)

    def test_build_adapter_hf_8bit_requires_transformers(self):
        """Test that build_adapter hf_8bit format requires transformers."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "hf_8bit"
        artifact.path = "/path/to/hf/model"

        # This will fail because model doesn't exist
        with pytest.raises(Exception):
            build_adapter(artifact)


class TestBuildAdapterFallbackChain:
    """Tests for build_adapter fallback chain behavior."""

    @patch("aligntune.eval.model_adapters.GGUFModelAdapter")
    @patch("aligntune.eval.model_adapters.VLLMModelAdapter")
    def test_fallback_chain_order(self, mock_vllm_cls, mock_gguf_cls):
        """Test that fallback chain tries vLLM before llama-cpp-python."""
        from aligntune.eval.model_adapters import build_adapter

        artifact = MagicMock()
        artifact.format = "gguf"
        artifact.path = "/path/to/model.gguf"

        call_order = []

        def vllm_init(*args, **kwargs):
            call_order.append("vllm")
            raise ImportError("vllm not available")

        def gguf_init(*args, **kwargs):
            call_order.append("gguf")
            return MagicMock()

        mock_vllm_cls.side_effect = vllm_init
        mock_gguf_cls.side_effect = gguf_init

        build_adapter(artifact)

        # vLLM should be tried first
        assert call_order == ["vllm", "gguf"]


class TestAdapterBackendNames:
    """Tests for all adapter backend_name properties."""

    def test_all_adapters_have_unique_backend_names(
        self, mock_llama_cpp_module, mock_ollama_module
    ):
        """Test that all adapters report unique backend names."""
        from aligntune.eval.model_adapters import (
            HFModelAdapter,
            VLLMModelAdapter,
            GGUFModelAdapter,
            OllamaModelAdapter,
        )

        # Get class-level backend names
        backend_names = set()

        # HFModelAdapter can be tested directly
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()
        hf_adapter = HFModelAdapter(mock_model, mock_tokenizer)
        backend_names.add(hf_adapter.backend_name)

        # VLLMModelAdapter does `from vllm import LLM` lazily inside
        # __init__, so there's no `aligntune.eval.model_adapters.vllm`/`LLM`
        # module attribute to patch. vllm is actually installed here, so
        # patch the real vllm.LLM class instead.
        with patch("vllm.LLM", MagicMock()):
            vllm_adapter = VLLMModelAdapter("/path/to/model")
            backend_names.add(vllm_adapter.backend_name)

        # GGUFModelAdapter/OllamaModelAdapter also lazily `import llama_cpp`
        # / `import ollama` inside __init__; neither package is installed
        # here, so fake modules are injected into sys.modules by the
        # mock_llama_cpp_module/mock_ollama_module fixtures.
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            gguf_adapter = GGUFModelAdapter(f.name)
            backend_names.add(gguf_adapter.backend_name)

        ollama_adapter = OllamaModelAdapter()
        backend_names.add(ollama_adapter.backend_name)

        # All should be unique
        expected = {"huggingface", "vllm", "gguf", "ollama"}
        assert backend_names == expected


class TestLazyImportGate:
    """Tests for lazy import gates."""

    def test_aligntune_imports_without_vllm(self):
        """Test that AlignTune imports successfully without vllm."""
        # Remove vllm from sys.modules if present
        import sys

        vllm_backup = sys.modules.pop("vllm", None)

        try:
            # Should not raise
            from aligntune.eval.model_adapters import ModelAdapter, HFModelAdapter

            assert ModelAdapter is not None
            assert HFModelAdapter is not None
        finally:
            if vllm_backup is not None:
                sys.modules["vllm"] = vllm_backup

    def test_aligntune_imports_without_llama_cpp(self):
        """Test that AlignTune imports successfully without llama-cpp-python."""
        import sys

        llama_cpp_backup = sys.modules.pop("llama_cpp", None)

        try:
            # Should not raise
            from aligntune.eval.model_adapters import ModelAdapter, HFModelAdapter

            assert ModelAdapter is not None
            assert HFModelAdapter is not None
        finally:
            if llama_cpp_backup is not None:
                sys.modules["llama_cpp"] = llama_cpp_backup

    def test_aligntune_imports_without_ollama(self):
        """Test that AlignTune imports successfully without ollama."""
        import sys

        ollama_backup = sys.modules.pop("ollama", None)

        try:
            # Should not raise
            from aligntune.eval.model_adapters import ModelAdapter, HFModelAdapter

            assert ModelAdapter is not None
            assert HFModelAdapter is not None
        finally:
            if ollama_backup is not None:
                sys.modules["ollama"] = ollama_backup

    def test_vllm_adapter_init_raises_helpful_error(self):
        """Test that VLLMModelAdapter.__init__ raises helpful ImportError message."""
        with patch.dict("sys.modules", {"vllm": None}):
            with pytest.raises(ImportError) as exc_info:
                VLLMModelAdapter(model_path="/path/to/model")

            error_msg = str(exc_info.value)
            assert "pip install vllm" in error_msg

    def test_gguf_adapter_init_raises_helpful_error(self):
        """Test that GGUFModelAdapter.__init__ raises helpful ImportError message."""
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError) as exc_info:
                GGUFModelAdapter(gguf_path="/path/to/model.gguf")

            error_msg = str(exc_info.value)
            assert "pip install llama-cpp-python" in error_msg

    def test_ollama_adapter_init_raises_helpful_error(self):
        """Test that OllamaModelAdapter.__init__ raises helpful ImportError message."""
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError) as exc_info:
                OllamaModelAdapter(model_name="llama2")

            error_msg = str(exc_info.value)
            assert "pip install ollama" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
