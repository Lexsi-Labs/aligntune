"""
Comprehensive tokenization tests covering all methods and tokenizer types.

Tests:
1. Continued BPE (with/without pruning)
2. Naive extension
3. Multiple tokenizer types (BPE, SentencePiece, etc.)
4. Config value verification
5. 10-15 different model tokenizers
"""

import pytest
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

# Check if PyICU is available (required for SentencePiece tokenizers)
try:
    import icu
    HAS_ICU = True
except ImportError:
    HAS_ICU = False

# Models that require PyICU (SentencePiece-based)
REQUIRES_ICU = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "01-ai/Yi-6B",
    "01-ai/Yi-9B",
    "01-ai/Yi-34B",
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "Qwen/Qwen2-0.5B",  # Qwen2 uses tiktoken-style but still needs ICU
]

# Models to test (different tokenizer types)
TEST_MODELS = [
    # ===== GPT FAMILY (Byte-level BPE) =====
    ("gpt2", "BPE"),
    ("openai-community/gpt2-medium", "BPE"),
    ("openai-community/gpt2-large", "BPE"),

    # ===== GPT-NEO/GPT-J FAMILY (Byte-level BPE) =====
    ("EleutherAI/gpt-neo-125M", "BPE"),
    ("EleutherAI/gpt-neo-1.3B", "BPE"),
    ("EleutherAI/gpt-neo-2.7B", "BPE"),
    ("EleutherAI/gpt-j-6B", "BPE"),

    # ===== PYTHIA FAMILY (Byte-level BPE) =====
    ("EleutherAI/pythia-70m", "BPE"),
    ("EleutherAI/pythia-160m", "BPE"),
    ("EleutherAI/pythia-410m", "BPE"),
    ("EleutherAI/pythia-1b", "BPE"),
    ("EleutherAI/pythia-1.4b", "BPE"),
    ("EleutherAI/pythia-2.8b", "BPE"),

    # ===== GPT-NEOX FAMILY (Byte-level BPE) =====
    ("EleutherAI/gpt-neox-20b", "BPE"),

    # ===== PHI FAMILY (Byte-level BPE) =====
    ("microsoft/phi-1", "BPE"),
    ("microsoft/phi-1_5", "BPE"),
    ("microsoft/phi-2", "BPE"),
    ("microsoft/Phi-3-mini-4k-instruct", "BPE"),
    ("microsoft/Phi-3-mini-128k-instruct", "BPE"),

    # ===== LLAMA FAMILY (SentencePiece BPE) =====
    ("meta-llama/Llama-2-7b-hf", "SentencePiece"),
    ("meta-llama/Llama-2-13b-hf", "SentencePiece"),
    ("meta-llama/Llama-2-70b-hf", "SentencePiece"),
    ("meta-llama/Meta-Llama-3-8B", "SentencePiece"),
    ("meta-llama/Meta-Llama-3-70B", "SentencePiece"),
    ("meta-llama/Llama-3.1-8B", "SentencePiece"),
    ("meta-llama/Llama-3.2-1B", "SentencePiece"),
    ("meta-llama/Llama-3.2-3B", "SentencePiece"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "SentencePiece"),

    # ===== MISTRAL FAMILY (SentencePiece BPE) =====
    ("mistralai/Mistral-7B-v0.1", "SentencePiece"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "SentencePiece"),
    ("mistralai/Mixtral-8x7B-v0.1", "SentencePiece"),
    ("mistralai/Mixtral-8x22B-v0.1", "SentencePiece"),

    # ===== QWEN FAMILY (Mixed - BPE/tiktoken) =====
    ("Qwen/Qwen-1_8B", "BPE"),
    ("Qwen/Qwen-7B", "BPE"),
    ("Qwen/Qwen-14B", "BPE"),
    ("Qwen/Qwen1.5-0.5B", "BPE"),
    ("Qwen/Qwen1.5-1.8B", "BPE"),
    ("Qwen/Qwen1.5-7B", "BPE"),
    ("Qwen/Qwen2-0.5B", "BPE"),
    ("Qwen/Qwen2-1.5B", "BPE"),
    ("Qwen/Qwen2-7B", "BPE"),
    ("Qwen/Qwen2.5-0.5B", "BPE"),
    ("Qwen/Qwen2.5-1.5B", "BPE"),
    ("Qwen/Qwen2.5-7B", "BPE"),

    # ===== GEMMA FAMILY (SentencePiece) =====
    ("google/gemma-2b", "SentencePiece"),
    ("google/gemma-7b", "SentencePiece"),
    ("google/gemma-2-2b", "SentencePiece"),
    ("google/gemma-2-9b", "SentencePiece"),

    # ===== YI FAMILY (SentencePiece) =====
    ("01-ai/Yi-6B", "SentencePiece"),
    ("01-ai/Yi-9B", "SentencePiece"),
    ("01-ai/Yi-34B", "SentencePiece"),

    # ===== FALCON FAMILY (Byte-level BPE) =====
    ("tiiuae/falcon-7b", "BPE"),
    ("tiiuae/falcon-40b", "BPE"),
    ("tiiuae/falcon-180B", "BPE"),

    # ===== MPT FAMILY (Byte-level BPE) =====
    ("mosaicml/mpt-7b", "BPE"),
    ("mosaicml/mpt-30b", "BPE"),

    # ===== BLOOM FAMILY (Byte-level BPE) =====
    ("bigscience/bloom-560m", "BPE"),
    ("bigscience/bloom-1b1", "BPE"),
    ("bigscience/bloom-1b7", "BPE"),
    ("bigscience/bloom-3b", "BPE"),
    ("bigscience/bloom-7b1", "BPE"),
    ("bigscience/bloom", "BPE"),  # 176B

    # ===== STABLELM FAMILY (Mixed) =====
    ("stabilityai/stablelm-base-alpha-3b", "BPE"),
    ("stabilityai/stablelm-base-alpha-7b", "BPE"),
    ("stabilityai/stablelm-2-1_6b", "SentencePiece"),
    ("stabilityai/stablelm-2-zephyr-1_6b", "SentencePiece"),

    # ===== OLMO FAMILY (Byte-level BPE) =====
    ("allenai/OLMo-1B", "BPE"),
    ("allenai/OLMo-7B", "BPE"),

    # ===== CODEGEN FAMILY (Byte-level BPE) =====
    ("Salesforce/codegen-350M-mono", "BPE"),
    ("Salesforce/codegen-2B-mono", "BPE"),
    ("Salesforce/codegen-6B-mono", "BPE"),

    # ===== STARCODER FAMILY (Byte-level BPE) =====
    ("bigcode/starcoder", "BPE"),
    ("bigcode/starcoderbase", "BPE"),

    # ===== DEEPSEEK FAMILY (Byte-level BPE) =====
    ("deepseek-ai/deepseek-coder-1.3b-base", "BPE"),
    ("deepseek-ai/deepseek-coder-6.7b-base", "BPE"),
    ("deepseek-ai/deepseek-llm-7b-base", "BPE"),
]

# Smaller models for quick testing (one from each major family, no auth needed)
QUICK_TEST_MODELS = [
    "gpt2",                                          # GPT family (BPE)
    "EleutherAI/gpt-neo-125M",                      # GPT-Neo family (BPE)
    "EleutherAI/pythia-70m",                        # Pythia family (BPE)
    "microsoft/phi-2",                              # Phi family (BPE)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",          # Llama family (SentencePiece)
    "Qwen/Qwen2-0.5B",                              # Qwen family (BPE)
    "google/gemma-2b",                              # Gemma family (SentencePiece)
    "stabilityai/stablelm-base-alpha-3b",          # StableLM family (BPE)
    "bigscience/bloom-560m",                        # BLOOM family (BPE)
    "Salesforce/codegen-350M-mono",                 # CodeGen family (BPE)
]


@pytest.fixture
def hindi_corpus():
    """Small Hindi corpus for testing."""
    return [
        "नमस्ते, आप कैसे हैं?",
        "धन्यवाद, मैं ठीक हूं।",
        "भारत एक सुंदर देश है।",
        "हिंदी भाषा बहुत महत्वपूर्ण है।",
        "मुझे भारतीय संस्कृति पसंद है।",
        "यह एक परीक्षण वाक्य है।",
        "कृपया मेरी मदद करें।",
        "आज मौसम बहुत अच्छा है।",
    ] * 50  # Repeat for sufficient corpus


@pytest.fixture
def chinese_corpus():
    """Small Chinese corpus for testing."""
    return [
        "你好世界",
        "谢谢你的帮助",
        "今天天气很好",
        "我喜欢学习中文",
        "这是一个测试句子",
        "欢迎来到中国",
        "我们一起努力吧",
        "祝你有美好的一天",
    ] * 50


class TestConfigVerification:
    """Test that config values are correctly set."""

    def test_config_values_basic(self, tmp_path):
        """Test basic config values are set correctly."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "test1"),
            num_new_tokens=100,
            extension_method="continued_bpe",
            max_samples=200,  # Increased for 100 tokens (need ~100k corpus tokens)
        )

        # Verify config values
        config = trainer.config
        assert config.model.base_model == "gpt2"
        assert config.model.new_tokens_count == 100
        assert config.vocab_extension.target_languages == ["hi"]
        assert config.vocab_extension.method.value == "continued_bpe"
        assert config.dataset.name == "wikimedia/wikipedia"
        assert config.dataset.config_name == "20231101.simple"
        assert config.dataset.max_samples == 200  # Updated to match increased corpus size
        assert config.logging.output_dir == str(tmp_path / "test1")
        assert config.pruning.enabled == False

    def test_config_values_with_pruning(self, tmp_path):
        """Test config values with pruning enabled."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["zh"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "test2"),
            num_new_tokens=200,
            extension_method="naive_extension",
            max_samples=10,
            prune=True,
            pruning_ratio=0.15,
            pruning_method="frequency",
        )

        config = trainer.config
        assert config.model.new_tokens_count == 200
        assert config.vocab_extension.method.value == "naive_extension"
        assert config.pruning.enabled == True
        assert config.pruning.pruning_ratio == 0.15
        assert config.pruning.method.value == "frequency"

    def test_config_values_advanced(self, tmp_path):
        """Test advanced config values."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["ar", "fa", "ur"],
            dataset_name="wikimedia/wikipedia",
            output_dir=str(tmp_path / "test3"),
            num_new_tokens=500,
            precision="fp16",
            device_map="cpu",
            trust_remote_code=True,
            streaming=True,
            split="validation",
            text_column="content",
        )

        config = trainer.config
        assert config.vocab_extension.target_languages == ["ar", "fa", "ur"]
        assert config.model.precision == "fp16"
        assert config.model.device_map == "cpu"
        assert config.model.trust_remote_code == True
        assert config.dataset.streaming == True
        assert config.dataset.split == "validation"
        assert config.dataset.text_column == "content"


class TestContinuedBPE:
    """Test continued BPE training."""

    @pytest.mark.parametrize("model_name", QUICK_TEST_MODELS)
    def test_continued_bpe_basic(self, model_name, tmp_path):
        """Test continued BPE on different models."""
        # Skip if model requires ICU and ICU is not available
        if model_name in REQUIRES_ICU and not HAS_ICU:
            pytest.skip(f"PyICU not installed, required for {model_name}. Install with: pip install PyICU")

        from aligntune.core.backend_factory import create_tokenization_trainer

        output_dir = tmp_path / f"continued_bpe_{model_name.replace('/', '_')}"

        # Qwen2 needs more samples due to different tokenization and SentencePiece
        max_samples = 400 if "Qwen" in model_name else 150

        trainer = create_tokenization_trainer(
            base_model=model_name,
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(output_dir),
            num_new_tokens=50,
            extension_method="continued_bpe",
            max_samples=max_samples,  # Increased for sufficient corpus size (need ~1M tokens per 1k new tokens, Qwen needs extra)
        )

        # Get original vocab size
        original_vocab = len(AutoTokenizer.from_pretrained(model_name))

        # Train
        result = trainer.train()

        # Verify results
        assert result["original_vocab_size"] == original_vocab
        assert result["vocab_extension"]["method"] == "continued_bpe"
        assert result["vocab_extension"]["num_added_tokens"] == 50

        # Check unreachable tokens
        unreachable = result["vocab_extension"]["unreachable_tokens"]

        # SentencePiece tokenizers have byte fallback tokens that are "unreachable" but expected
        # Allow up to 260 unreachable tokens for SentencePiece (256 byte tokens + special tokens)
        if model_name in REQUIRES_ICU:
            # SentencePiece tokenizer - byte fallback is expected
            assert unreachable <= 260, f"SentencePiece should have ≤260 unreachable (byte fallback), got {unreachable}"
        else:
            # Byte-level BPE - should have 0 unreachable
            assert unreachable == 0, f"Byte-level BPE should have 0 unreachable, got {unreachable}"

        # Verify tokenizer saved
        assert output_dir.exists()
        assert (output_dir / "tokenizer_config.json").exists()

        # Load and verify extended tokenizer
        extended_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        assert len(extended_tokenizer) == original_vocab + 50

    def test_continued_bpe_with_pruning(self, tmp_path):
        """Test continued BPE with pruning."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        output_dir = tmp_path / "continued_bpe_pruned"

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(output_dir),
            num_new_tokens=100,
            extension_method="continued_bpe",
            max_samples=200,  # Increased for 100 tokens (need ~100k corpus tokens)
            prune=True,
            pruning_ratio=0.2,  # Prune 20%
            pruning_method="leaf_frequency",
        )

        original_vocab = len(AutoTokenizer.from_pretrained("gpt2"))
        result = trainer.train()

        # After adding 100 tokens and pruning 20%, should have:
        # original + 100, then prune 20% of that
        extended_size = original_vocab + 100
        pruned_tokens = int(extended_size * 0.2)
        expected_final = extended_size - pruned_tokens

        assert "pruning" in result
        assert result["pruning"]["method"] == "leaf_frequency"
        assert abs(result["pruning"]["n_tokens_pruned"] - pruned_tokens) <= 1  # Allow 1 token difference
        assert abs(result["pruning"]["new_vocab_size"] - expected_final) <= 1  # Allow 1 token difference

    def test_continued_bpe_unreachable_check(self, tmp_path):
        """Test that continued BPE produces no unreachable tokens."""
        from aligntune.core.backend_factory import create_tokenization_trainer
        from aligntune.eval.tokenization import evaluate_unreachable_tokens

        output_dir = tmp_path / "continued_bpe_check"

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(output_dir),
            num_new_tokens=50,
            extension_method="continued_bpe",
            max_samples=150,  # Increased for sufficient corpus size
        )

        result = trainer.train()

        # Load extended tokenizer and evaluate
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        eval_result = evaluate_unreachable_tokens(tokenizer)

        assert eval_result["unreachable_count"] == 0
        assert eval_result["reachable_ratio"] == 100.0


class TestNaiveExtension:
    """Test naive extension method."""

    @pytest.mark.parametrize("model_name", QUICK_TEST_MODELS)
    def test_naive_extension_basic(self, model_name, tmp_path):
        """Test naive extension on different models."""
        # Skip if model requires ICU and ICU is not available
        if model_name in REQUIRES_ICU and not HAS_ICU:
            pytest.skip(f"PyICU not installed, required for {model_name}. Install with: pip install PyICU")

        from aligntune.core.backend_factory import create_tokenization_trainer

        output_dir = tmp_path / f"naive_{model_name.replace('/', '_')}"

        trainer = create_tokenization_trainer(
            base_model=model_name,
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(output_dir),
            num_new_tokens=1000,  # Naive needs larger vocab to train separate tokenizer
            extension_method="naive_extension",
            max_samples=50,
        )

        original_vocab = len(AutoTokenizer.from_pretrained(model_name))
        result = trainer.train()

        # Verify results
        assert result["vocab_extension"]["method"] == "naive_extension"
        assert result["vocab_extension"]["num_added_tokens"] > 0

        # Naive extension MAY have unreachable tokens (that's expected)
        # Just check that we tracked it
        assert "unreachable_tokens" in result["vocab_extension"]

        # Verify tokenizer saved
        assert output_dir.exists()
        extended_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        assert len(extended_tokenizer) > original_vocab

    def test_naive_extension_may_have_unreachable(self, tmp_path):
        """Test that naive extension may produce unreachable tokens (expected behavior)."""
        from aligntune.core.backend_factory import create_tokenization_trainer
        from aligntune.eval.tokenization import evaluate_unreachable_tokens

        output_dir = tmp_path / "naive_unreachable"

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(output_dir),
            num_new_tokens=1000,
            extension_method="naive_extension",
            max_samples=50,
        )

        trainer.train()

        # Load and evaluate
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        eval_result = evaluate_unreachable_tokens(tokenizer)

        # Naive extension often has unreachable tokens
        # We just verify the evaluation ran
        assert "unreachable_count" in eval_result
        assert "vocab_size" in eval_result


class TestMultipleTokenizerTypes:
    """Test on multiple tokenizer types to ensure robustness."""

    def test_gpt2_bpe(self, tmp_path):
        """Test GPT-2 (byte-level BPE)."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "gpt2"),
            num_new_tokens=50,
            max_samples=10,
        )

        result = trainer.train()
        assert result["vocab_extension"]["unreachable_tokens"] == 0

    def test_gpt_neo_bpe(self, tmp_path):
        """Test GPT-Neo (byte-level BPE)."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="EleutherAI/gpt-neo-125M",
            target_languages=["zh"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "gpt_neo"),
            num_new_tokens=50,
            max_samples=10,
        )

        result = trainer.train()
        assert result["vocab_extension"]["unreachable_tokens"] == 0

    def test_llama2_sentencepiece(self, tmp_path):
        """Test Llama-2 (SentencePiece BPE) - requires auth."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="meta-llama/Llama-2-7b-hf",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "llama2"),
            num_new_tokens=50,
            max_samples=10,
        )

        result = trainer.train()
        # Unlike gpt2/gpt-neo's byte-level BPE (which encode every possible
        # byte sequence directly in the base vocab), Llama-2's SentencePiece
        # tokenizer ships with exactly 256 byte-fallback tokens
        # ("<0x00>".."<0xFF>") used only as an encoding fallback for raw
        # bytes - confirmed via tokenizer.get_vocab(). These can never be
        # produced by continued-BPE merges learned from any text corpus (no
        # corpus size fixes this - tried 10 and 50 samples, both report
        # exactly 256), so "0 unreachable tokens" isn't an achievable target
        # for this tokenizer the way it is for byte-level BPE ones.
        assert result["vocab_extension"]["unreachable_tokens"] == 256


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_hindi_adaptation_workflow(self, tmp_path):
        """Test complete Hindi adaptation workflow."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        output_dir = tmp_path / "hindi_complete"

        # Step 1: Extend with continued BPE
        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.hi",
            output_dir=str(output_dir),
            num_new_tokens=100,
            extension_method="continued_bpe",
            max_samples=200,  # Increased for 100 tokens
        )

        result = trainer.train()

        # Verify all outputs
        assert (output_dir / "tokenizer_config.json").exists()
        assert (output_dir / "tokenization_results.json").exists()
        assert result["vocab_extension"]["unreachable_tokens"] == 0

        # Step 2: Load and verify
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))

        # Test tokenization on Hindi text
        hindi_text = "नमस्ते, मैं हिंदी बोल सकता हूं।"
        tokens = tokenizer.encode(hindi_text)
        decoded = tokenizer.decode(tokens)

        # Should be able to encode/decode (may not be perfect but should work)
        assert len(tokens) > 0
        assert len(decoded) > 0

    def test_chinese_adaptation_with_pruning(self, tmp_path):
        """Test Chinese adaptation with pruning."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        output_dir = tmp_path / "chinese_pruned"

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["zh"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.zh",
            output_dir=str(output_dir),
            num_new_tokens=200,
            extension_method="continued_bpe",
            max_samples=400,  # Increased for 200 tokens (need ~200k corpus tokens)
            prune=True,
            pruning_ratio=0.1,
        )

        result = trainer.train()

        # Verify pruning happened
        assert "pruning" in result
        assert result["pruning"]["n_tokens_pruned"] > 0

        # Verify tokenizer works
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        chinese_text = "你好世界"
        tokens = tokenizer.encode(chinese_text)
        assert len(tokens) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_corpus(self, tmp_path):
        """Test with very small corpus."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "small_corpus"),
            num_new_tokens=10,  # Small number of tokens
            max_samples=5,  # Very small corpus
        )

        # Should complete without error
        result = trainer.train()
        assert "vocab_extension" in result

    def test_zero_unreachable_continued_bpe(self, tmp_path):
        """Verify continued BPE always produces 0 unreachable tokens."""
        from aligntune.core.backend_factory import create_tokenization_trainer

        for i in range(3):  # Test multiple times
            output_dir = tmp_path / f"zero_unreachable_{i}"

            trainer = create_tokenization_trainer(
                base_model="gpt2",
                target_languages=["hi"],
                dataset_name="wikimedia/wikipedia",
                config_name="20231101.simple",
                output_dir=str(output_dir),
                num_new_tokens=30,
                extension_method="continued_bpe",
                max_samples=100,  # Increased for 30 tokens (need ~30k corpus tokens)
            )

            result = trainer.train()

            # Critical assertion: continued BPE MUST have 0 unreachable
            assert result["vocab_extension"]["unreachable_tokens"] == 0, \
                f"Continued BPE should NEVER have unreachable tokens (iteration {i})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
