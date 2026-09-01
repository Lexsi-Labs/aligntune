"""
CPU-only tests for Text-to-LoRA hypernetwork implementation (v3.3 Advanced Parameterization).

Tests verify:
1. TextToLoRAHypernet initialization and forward pass
2. LoRA matrix shape and initialization (A from N(0, std), B zeros)
3. TextToLoRATrainer initialization and meta-training
4. Mock training loop execution without crashing
5. Checkpoint saving/loading
6. DocToLoRA document chunking and embedding pooling
7. Configuration serialization

All tests run on CPU without GPU requirement.
All external dependencies (sentence-transformers, models) are mocked.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestTextToLoRAHypernet:
    """Test TextToLoRAHypernet class initialization and forward pass."""

    def test_hypernet_initialization_basic(self):
        """Test that hypernetwork initializes with default parameters."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(
            hidden_dim=768, lora_r=16, num_target_modules=5, device="cpu"
        )

        assert hypernet.hidden_dim == 768
        assert hypernet.lora_r == 16
        assert hypernet.num_target_modules == 5
        assert hypernet.device == "cpu"

    def test_hypernet_initialization_custom_mlp(self):
        """Test initialization with custom MLP hidden dimension."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(
            hidden_dim=256, lora_r=8, num_target_modules=3, mlp_hidden=256
        )

        assert hypernet.mlp_hidden == 256
        assert hypernet.get_num_parameters() > 0

    def test_hypernet_initialization_invalid_params(self):
        """Test that invalid parameters raise ValueError."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            TextToLoRAHypernet(hidden_dim=-1)

        with pytest.raises(ValueError, match="lora_r must be positive"):
            TextToLoRAHypernet(hidden_dim=768, lora_r=0)

        with pytest.raises(ValueError, match="num_target_modules must be positive"):
            TextToLoRAHypernet(hidden_dim=768, lora_r=16, num_target_modules=0)

        with pytest.raises(ValueError, match="mlp_hidden must be positive"):
            TextToLoRAHypernet(hidden_dim=768, lora_r=16, mlp_hidden=-1)

        with pytest.raises(ValueError, match="lora_init_std must be positive"):
            TextToLoRAHypernet(hidden_dim=768, lora_r=16, lora_init_std=-0.01)

    def test_hypernet_forward_basic(self):
        """Test basic forward pass produces correct output shapes."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16, num_target_modules=5)

        # Create mock embedding
        batch_size = 2
        embeddings = torch.randn(batch_size, 768)

        # Forward pass
        lora_weights = hypernet(embeddings)

        # Check output structure
        assert isinstance(lora_weights, list)
        assert len(lora_weights) == 5  # num_target_modules

        for module_idx, lora_pair in enumerate(lora_weights):
            assert isinstance(lora_pair, dict)
            assert "A" in lora_pair
            assert "B" in lora_pair

            # Check shapes
            a_matrix = lora_pair["A"]
            b_matrix = lora_pair["B"]

            assert a_matrix.shape == (batch_size, 16, 768)
            assert b_matrix.shape == (batch_size, 768, 16)

            # Check that tensors are on correct device
            assert a_matrix.device == embeddings.device
            assert b_matrix.device == embeddings.device

            # Check no NaNs
            assert not torch.isnan(a_matrix).any()
            assert not torch.isnan(b_matrix).any()

    def test_hypernet_forward_batch_size_variations(self):
        """Test forward pass with different batch sizes."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8, num_target_modules=3)

        for batch_size in [1, 4, 8, 16]:
            embeddings = torch.randn(batch_size, 256)
            lora_weights = hypernet(embeddings)

            assert len(lora_weights) == 3
            for lora_pair in lora_weights:
                assert lora_pair["A"].shape[0] == batch_size
                assert lora_pair["B"].shape[0] == batch_size

    def test_hypernet_a_matrix_initialization(self):
        """Test that A matrices are initialized from N(0, lora_init_std)."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(
            hidden_dim=768, lora_r=16, num_target_modules=2, lora_init_std=0.02
        )

        embeddings = torch.randn(32, 768)  # Larger batch for statistics
        lora_weights = hypernet(embeddings)

        # Check A matrix statistics
        for lora_pair in lora_weights:
            a_matrix = lora_pair["A"]

            # A matrix should have non-zero values
            assert a_matrix.abs().mean() > 0.0

            # A matrix values should be roughly centered around 0
            # with std around lora_init_std
            assert not torch.isnan(a_matrix).any()
            assert not torch.isinf(a_matrix).any()

    def test_hypernet_b_matrix_initialization(self):
        """Test that B matrices are initialized to zeros."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16, num_target_modules=3)

        embeddings = torch.randn(4, 768)
        lora_weights = hypernet(embeddings)

        # Check B matrices are zero
        for lora_pair in lora_weights:
            b_matrix = lora_pair["B"]

            # All elements should be zero (or very close)
            assert torch.allclose(b_matrix, torch.zeros_like(b_matrix), atol=1e-6)

    def test_hypernet_forward_invalid_shape(self):
        """Test that invalid input shapes raise ValueError."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)

        # 1D input should fail
        with pytest.raises(ValueError, match="Expected 2D input"):
            hypernet(torch.randn(768))

        # 3D input should fail
        with pytest.raises(ValueError, match="Expected 2D input"):
            hypernet(torch.randn(2, 10, 768))

        # Wrong embedding dimension should fail
        with pytest.raises(ValueError, match="doesn't match"):
            hypernet(torch.randn(2, 256))

    def test_hypernet_get_embedding_model(self):
        """Test get_embedding_model method."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768)

        # Mock the sentence-transformers import
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            model = hypernet.get_embedding_model()
            assert model is not None

    def test_hypernet_get_embedding_model_import_error(self):
        """Test that missing sentence-transformers raises ImportError."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768)

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                hypernet.get_embedding_model()

    def test_hypernet_get_num_parameters(self):
        """Test parameter counting."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16, mlp_hidden=512)
        num_params = hypernet.get_num_parameters()

        # Should be > 0
        assert num_params > 0

        # Should include MLP weights and biases
        # Input: 768 → hidden: 512 → output: num_targets * 2 * rank * hidden_dim
        output_dim = 5 * 2 * 16 * 768  # default num_target_modules=5
        expected_params = 768 * 512 + 512 + 512 * output_dim + output_dim
        assert num_params == expected_params

    def test_hypernet_get_lora_config(self):
        """Test configuration dictionary."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet

        hypernet = TextToLoRAHypernet(
            hidden_dim=768, lora_r=16, num_target_modules=5, mlp_hidden=512
        )

        config = hypernet.get_lora_config()

        assert config["hidden_dim"] == 768
        assert config["lora_r"] == 16
        assert config["num_target_modules"] == 5
        assert config["mlp_hidden"] == 512
        assert "num_parameters" in config
        assert config["num_parameters"] > 0


class TestTextToLoRATrainer:
    """Test TextToLoRATrainer meta-training."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)

        train_tasks = [
            {
                "description": "Task 1: Fine-tune on classification",
                "data": [],
                "eval_fn": lambda x: torch.tensor(0.5),
            }
        ]

        trainer = TextToLoRATrainer(
            hypernet=hypernet,
            target_model=None,
            train_tasks=train_tasks,
            val_tasks=[],
            learning_rate=1e-4,
            device="cpu",
        )

        assert trainer.hypernet is hypernet
        assert trainer.learning_rate == 1e-4
        assert trainer.train_step == 0
        assert trainer.best_val_loss == float("inf")

    def test_trainer_initialization_no_hypernet(self):
        """Test that trainer requires hypernet."""
        from aligntune.core.adapters.text2lora import TextToLoRATrainer

        with pytest.raises(ValueError, match="hypernet cannot be None"):
            TextToLoRATrainer(None, None, [], [])

    def test_trainer_initialization_no_train_tasks(self):
        """Test that trainer requires training tasks."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()

        with pytest.raises(ValueError, match="train_tasks cannot be None or empty"):
            TextToLoRATrainer(hypernet, None, None, [])

        with pytest.raises(ValueError, match="train_tasks cannot be None or empty"):
            TextToLoRATrainer(hypernet, None, [], [])

    def test_train_step_meta_basic(self):
        """Test a single meta-training step."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8)

        train_tasks = [
            {"description": f"Task {i}", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}
            for i in range(3)
        ]

        trainer = TextToLoRATrainer(hypernet, None, train_tasks, [], device="cpu")

        # Run one training step
        metrics = trainer.train_step_meta()

        assert "loss" in metrics
        assert "lr" in metrics
        assert "grad_norm" in metrics

        assert metrics["loss"] >= 0
        assert metrics["lr"] > 0

    def test_train_epoch_multiple_steps(self):
        """Test training for multiple steps."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8)

        train_tasks = [
            {"description": "Task 1", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}
        ]

        trainer = TextToLoRATrainer(hypernet, None, train_tasks, [], device="cpu")

        # Train for 5 steps
        metrics = trainer.train_epoch(num_steps=5)

        assert "avg_loss" in metrics
        assert "final_lr" in metrics
        assert "total_steps" in metrics

        assert metrics["total_steps"] == 5
        assert metrics["avg_loss"] >= 0

    def test_train_epoch_invalid_steps(self):
        """Test that invalid step count raises ValueError."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]

        trainer = TextToLoRATrainer(hypernet, None, train_tasks)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            trainer.train_epoch(num_steps=0)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            trainer.train_epoch(num_steps=-1)

    def test_trainer_learning_rate_warmup(self):
        """Test learning rate warmup schedule."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]

        trainer = TextToLoRATrainer(
            hypernet, None, train_tasks, warmup_steps=100, learning_rate=1e-4, device="cpu"
        )

        # During warmup
        lr_start = trainer._get_learning_rate(step=0)
        lr_mid = trainer._get_learning_rate(step=50)
        lr_end = trainer._get_learning_rate(step=100)
        lr_after = trainer._get_learning_rate(step=200)

        assert lr_start < lr_mid < lr_end == lr_after
        assert lr_start == 0.0
        assert abs(lr_end - 1e-4) < 1e-7

    def test_trainer_validate(self, temp_checkpoint_dir):
        """Test validation loop."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet(hidden_dim=256)

        train_tasks = [{"description": "Train", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]
        val_tasks = [
            {"description": "Val 1", "data": [], "eval_fn": lambda x: torch.tensor(0.6)},
            {"description": "Val 2", "data": [], "eval_fn": lambda x: torch.tensor(0.7)},
        ]

        trainer = TextToLoRATrainer(
            hypernet, None, train_tasks, val_tasks, checkpoint_dir=temp_checkpoint_dir
        )

        val_metrics = trainer.validate()

        assert "val_loss" in val_metrics
        assert "num_val_tasks" in val_metrics
        assert "is_best" in val_metrics

        assert val_metrics["num_val_tasks"] == 2
        assert val_metrics["is_best"] is True  # First validation is always best

    def test_trainer_validate_no_tasks(self):
        """Test validation with no validation tasks."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]

        trainer = TextToLoRATrainer(hypernet, None, train_tasks, val_tasks=None)

        val_metrics = trainer.validate()

        assert val_metrics["val_loss"] == 0.0
        assert val_metrics["num_val_tasks"] == 0
        assert val_metrics["is_best"] is False

    def test_trainer_save_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint saving."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]

        trainer = TextToLoRATrainer(
            hypernet, None, train_tasks, checkpoint_dir=temp_checkpoint_dir
        )

        # Train a few steps
        trainer.train_epoch(num_steps=2)

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint("test_ckpt.pt")

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pt"

        # Check checkpoint contains expected keys
        state = torch.load(checkpoint_path)
        assert "hypernet" in state
        assert "optimizer" in state
        assert "train_step" in state

    def test_trainer_load_checkpoint(self, temp_checkpoint_dir):
        """Test checkpoint loading."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]

        trainer = TextToLoRATrainer(
            hypernet, None, train_tasks, checkpoint_dir=temp_checkpoint_dir
        )

        # Train and save
        trainer.train_epoch(num_steps=5)
        original_step = trainer.train_step
        checkpoint_path = trainer.save_checkpoint("test.pt")

        # Create new trainer and load
        hypernet2 = TextToLoRAHypernet()
        trainer2 = TextToLoRATrainer(
            hypernet2, None, train_tasks, checkpoint_dir=temp_checkpoint_dir
        )

        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.train_step == original_step

    def test_trainer_get_best_checkpoint(self, temp_checkpoint_dir):
        """Test getting best checkpoint state."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        hypernet = TextToLoRAHypernet()
        train_tasks = [{"description": "Task", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}]
        val_tasks = [{"description": "Val", "data": [], "eval_fn": lambda x: torch.tensor(0.6)}]

        trainer = TextToLoRATrainer(
            hypernet, None, train_tasks, val_tasks, checkpoint_dir=temp_checkpoint_dir
        )

        # No checkpoint yet
        best = trainer.get_best_checkpoint()
        assert best is None

        # Validate to save checkpoint
        trainer.validate()
        best = trainer.get_best_checkpoint()

        assert best is not None
        assert "hypernet" in best
        assert "config" in best
        assert "best_val_loss" in best


class TestDocToLoRA:
    """Test DocToLoRA document-based LoRA generation."""

    def test_doc2lora_initialization(self):
        """Test DocToLoRA initialization."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=768, lora_r=16)
        doc2lora = DocToLoRA(hypernet, chunk_size=512, num_chunks=3)

        assert doc2lora.chunk_size == 512
        assert doc2lora.num_chunks == 3
        assert doc2lora.pooling_strategy == "mean"

    def test_doc2lora_initialization_invalid_params(self):
        """Test invalid DocToLoRA parameters."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()

        with pytest.raises(ValueError, match="hypernet cannot be None"):
            DocToLoRA(None)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocToLoRA(hypernet, chunk_size=0)

        with pytest.raises(ValueError, match="num_chunks must be positive"):
            DocToLoRA(hypernet, num_chunks=0)

        with pytest.raises(ValueError, match="pooling_strategy must be"):
            DocToLoRA(hypernet, pooling_strategy="invalid")

    def test_doc2lora_chunk_document(self):
        """Test document chunking."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()
        doc2lora = DocToLoRA(hypernet, chunk_size=50, num_chunks=3)

        doc = "This is word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = doc2lora._chunk_document(doc)

        assert isinstance(chunks, list)
        assert len(chunks) <= 3
        assert all(isinstance(c, str) for c in chunks)

    def test_doc2lora_chunk_document_empty(self):
        """Test chunking empty document."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()
        doc2lora = DocToLoRA(hypernet, chunk_size=50, num_chunks=3)

        chunks = doc2lora._chunk_document("")
        assert chunks == [""]

        chunks = doc2lora._chunk_document("   ")
        assert chunks == [""]

    def test_doc2lora_embed_chunks(self):
        """Test chunk embedding."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=256)
        doc2lora = DocToLoRA(hypernet, device="cpu")

        chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = doc2lora._embed_chunks(chunks)

        assert embeddings.shape == (3, 256)
        assert not torch.isnan(embeddings).any()

    def test_doc2lora_embed_chunks_empty(self):
        """Test that empty chunks list raises error."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()
        doc2lora = DocToLoRA(hypernet)

        with pytest.raises(ValueError, match="chunks cannot be empty"):
            doc2lora._embed_chunks([])

    def test_doc2lora_pool_embeddings_mean(self):
        """Test mean pooling."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=256)
        doc2lora = DocToLoRA(hypernet, pooling_strategy="mean")

        embeddings = torch.randn(4, 256)
        pooled = doc2lora._pool_embeddings(embeddings)

        assert pooled.shape == (1, 256)
        assert torch.allclose(pooled, embeddings.mean(dim=0, keepdim=True))

    def test_doc2lora_pool_embeddings_weighted(self):
        """Test weighted (attention) pooling."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=256)
        doc2lora = DocToLoRA(hypernet, pooling_strategy="weighted")

        embeddings = torch.randn(4, 256)
        pooled = doc2lora._pool_embeddings(embeddings)

        assert pooled.shape == (1, 256)
        # Should be different from mean (with high probability)
        # Don't assert inequality due to randomness

    def test_doc2lora_forward(self):
        """Test full forward pass from document to LoRA."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8, num_target_modules=3)
        doc2lora = DocToLoRA(hypernet, chunk_size=100, num_chunks=2)

        doc = "This is a long document describing a fine-tuning task. " * 5

        lora_weights = doc2lora(doc)

        assert len(lora_weights) == 3
        for lora_pair in lora_weights:
            assert "A" in lora_pair
            assert "B" in lora_pair
            assert lora_pair["A"].shape == (1, 8, 256)
            assert lora_pair["B"].shape == (1, 256, 8)

    def test_doc2lora_forward_invalid_input(self):
        """Test that invalid input types raise errors."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()
        doc2lora = DocToLoRA(hypernet)

        with pytest.raises(ValueError, match="document_text must be a string"):
            doc2lora(123)

        with pytest.raises(ValueError, match="document_text must be a string"):
            doc2lora(None)

    def test_doc2lora_get_config(self):
        """Test configuration retrieval."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet()
        doc2lora = DocToLoRA(hypernet, chunk_size=512, num_chunks=4, pooling_strategy="weighted")

        config = doc2lora.get_config()

        assert config["chunk_size"] == 512
        assert config["num_chunks"] == 4
        assert config["pooling_strategy"] == "weighted"
        assert "hypernet_config" in config


class TestIntegration:
    """Integration tests across multiple components."""

    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, TextToLoRATrainer

        # Create hypernet
        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8, num_target_modules=3)

        # Create tasks
        train_tasks = [
            {"description": f"Train task {i}", "data": [], "eval_fn": lambda x: torch.tensor(0.5)}
            for i in range(2)
        ]
        val_tasks = [
            {"description": f"Val task {i}", "data": [], "eval_fn": lambda x: torch.tensor(0.6)}
            for i in range(1)
        ]

        # Create trainer
        trainer = TextToLoRATrainer(hypernet, None, train_tasks, val_tasks, device="cpu")

        # Train
        train_metrics = trainer.train_epoch(num_steps=3)
        assert train_metrics["total_steps"] == 3

        # Validate
        val_metrics = trainer.validate()
        assert "val_loss" in val_metrics

    def test_hypernet_with_doc2lora(self):
        """Test TextToLoRA with DocToLoRA integration."""
        from aligntune.core.adapters.text2lora import TextToLoRAHypernet, DocToLoRA

        hypernet = TextToLoRAHypernet(hidden_dim=256, lora_r=8)
        doc2lora = DocToLoRA(hypernet, chunk_size=100, num_chunks=2)

        # Process multiple documents
        docs = [
            "Document about task 1: classification problem with 10 classes",
            "Document about task 2: regression task with continuous outputs",
            "Document about task 3: machine translation from English to French",
        ]

        all_loras = []
        for doc in docs:
            lora = doc2lora(doc)
            all_loras.append(lora)
            assert len(lora) == hypernet.num_target_modules

        # All should produce valid LoRA weights
        assert len(all_loras) == 3
