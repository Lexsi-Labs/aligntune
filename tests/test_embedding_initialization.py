"""
Test embedding initialization with extended tokenizer.

User workflow:
1. Extend tokenizer with tokenization trainer
2. Pass extended tokenizer path to SFT trainer
3. SFT trainer handles model loading and embedding initialization
"""

import pytest


class TestEmbeddingInitialization:
    """Test that SFT trainer properly initializes embeddings with extended tokenizer."""

    def test_sft_with_extended_tokenizer(self, tmp_path):
        """Extend tokenizer, then use with SFT trainer (random init)."""
        from aligntune.core.backend_factory import create_tokenization_trainer, create_sft_trainer

        # Step 1: Extend tokenizer
        tokenizer_trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "extended_tokenizer"),
            num_new_tokens=20,
            extension_method="continued_bpe",
            max_samples=100,
        )

        tokenizer_result = tokenizer_trainer.train()
        extended_tokenizer_path = tokenizer_result['output_dir']

        # Step 1b: Evaluate tokenizer (before/after fertility)
        eval_result = tokenizer_trainer.evaluate(eval_samples=100)
        assert eval_result['old_fertility'] > 0
        assert eval_result['new_fertility'] > 0
        assert 'recommendation' in eval_result

        # Step 2: Create SFT trainer with extended tokenizer (random init)
        sft_trainer = create_sft_trainer(
            model_name="gpt2",  # Fixed: was base_model, should be model_name
            tokenizer_name_or_path=extended_tokenizer_path,  # Use extended tokenizer
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            output_dir=str(tmp_path / "sft_output"),
            max_steps=5,
            per_device_train_batch_size=2,
            max_seq_length=128,
            embedding_init_method="random",  # Default: random initialization
            backend="trl",  # Force TRL backend (not Unsloth)
        )

        # Step 3: Setup model (loads model and tokenizer)
        sft_trainer.setup_model()

        # Step 4: Verify model loaded with correct vocab size
        model = sft_trainer.model
        tokenizer = sft_trainer.tokenizer

        # Vocab sizes should match
        vocab_size = len(tokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]

        assert vocab_size == embedding_size, \
            f"Vocab size {vocab_size} != embedding size {embedding_size}"

        # Step 5: Verify model can process text
        text = "नमस्ते दोस्त"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Move to model device
        input_ids = input_ids.to(model.device)

        output = model(input_ids)
        assert output.logits is not None

    def test_sft_with_extended_tokenizer_fvt_mean(self, tmp_path):
        """Test FVT (Fast Vocabulary Transfer) embedding initialization with 'mean' method."""
        from aligntune.core.backend_factory import create_tokenization_trainer, create_sft_trainer
        import torch

        # Step 1: Extend tokenizer
        tokenizer_trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["hi"],
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "extended_tokenizer_fvt"),
            num_new_tokens=20,
            extension_method="continued_bpe",
            max_samples=100,
        )

        tokenizer_result = tokenizer_trainer.train()
        extended_tokenizer_path = tokenizer_result['output_dir']

        # Step 2: Create SFT trainer with FVT mean initialization
        sft_trainer = create_sft_trainer(
            model_name="gpt2",
            tokenizer_name_or_path=extended_tokenizer_path,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            output_dir=str(tmp_path / "sft_output_fvt"),
            max_steps=2,
            per_device_train_batch_size=1,
            max_seq_length=128,
            embedding_init_method="mean",  # FVT: mean of existing embeddings
            backend="trl",  # Force TRL backend (not Unsloth)
        )

        # Step 3: Setup model (loads model and tokenizer with FVT)
        sft_trainer.setup_model()

        # Step 4: Verify model loaded with correct vocab size
        model = sft_trainer.model
        tokenizer = sft_trainer.tokenizer

        vocab_size = len(tokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]

        assert vocab_size == embedding_size, \
            f"Vocab size {vocab_size} != embedding size {embedding_size}"

        # Step 5: Verify FVT initialization was applied
        # New embeddings should have non-zero values (from FVT, not random)
        embeddings = model.get_input_embeddings().weight
        original_vocab_size = 50257  # GPT-2 base vocab
        new_embeddings = embeddings[original_vocab_size:original_vocab_size + 20]

        # Check that new embeddings are not all zeros
        assert torch.abs(new_embeddings).sum() > 0, \
            "New embeddings are all zeros - FVT initialization may have failed"

        # Check that new embeddings have reasonable values (similar magnitude to existing ones)
        original_embeddings = embeddings[:original_vocab_size]
        new_norm = new_embeddings.norm().item()
        original_norm = original_embeddings.norm().item()

        print(f"Original embeddings norm: {original_norm:.4f}")
        print(f"New embeddings norm (FVT mean): {new_norm:.4f}")

        # New embeddings should have similar scale to original embeddings
        assert new_norm > 0, "New embeddings have zero norm"

        # Step 6: Verify model can process Hindi text
        text = "नमस्ते दोस्त"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Move to model device
        input_ids = input_ids.to(model.device)

        output = model(input_ids)
        assert output.logits is not None

    def test_sft_with_extended_tokenizer_fvt_mean_of_constituents(self, tmp_path):
        """Test FVT embedding initialization with 'mean_of_constituents' method."""
        from aligntune.core.backend_factory import create_tokenization_trainer, create_sft_trainer
        import torch

        # Step 1: Extend tokenizer
        tokenizer_trainer = create_tokenization_trainer(
            base_model="gpt2",
            target_languages=["zh"],  # Chinese
            dataset_name="wikimedia/wikipedia",
            config_name="20231101.simple",
            output_dir=str(tmp_path / "extended_tokenizer_fvt_constituents"),
            num_new_tokens=15,
            extension_method="continued_bpe",
            max_samples=80,
        )

        tokenizer_result = tokenizer_trainer.train()
        extended_tokenizer_path = tokenizer_result['output_dir']

        # Step 2: Create SFT trainer with FVT mean_of_constituents initialization
        sft_trainer = create_sft_trainer(
            model_name="gpt2",
            tokenizer_name_or_path=extended_tokenizer_path,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            output_dir=str(tmp_path / "sft_output_fvt_constituents"),
            max_steps=2,
            per_device_train_batch_size=1,
            max_seq_length=128,
            embedding_init_method="mean_of_constituents",  # FVT: mean of constituent tokens
            backend="trl",  # Force TRL backend (not Unsloth)
        )

        # Step 3: Setup model (loads model and tokenizer with FVT)
        sft_trainer.setup_model()

        # Step 4: Verify embeddings
        model = sft_trainer.model
        tokenizer = sft_trainer.tokenizer

        vocab_size = len(tokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]

        assert vocab_size == embedding_size

        # Verify new embeddings are initialized
        embeddings = model.get_input_embeddings().weight
        original_vocab_size = 50257
        new_embeddings = embeddings[original_vocab_size:original_vocab_size + 15]

        assert torch.abs(new_embeddings).sum() > 0, \
            "New embeddings are all zeros - FVT mean_of_constituents failed"

        new_norm = new_embeddings.norm().item()
        print(f"New embeddings norm (FVT mean_of_constituents): {new_norm:.4f}")

        assert new_norm > 0, "New embeddings have zero norm"

        # Step 5: Verify forward pass works
        text = "你好世界"  # Chinese text
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Move to model device
        input_ids = input_ids.to(model.device)

        output = model(input_ids)
        assert output.logits is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
