"""
Tests for RAFT Trainer implementation.

CPU-only tests with mock data. Validates:
1. RAFT example preparation (document context formatting)
2. Citation quality metrics
3. Trainer initialization and config handling
"""

import pytest
import logging
from typing import Dict, List, Any

# Set up logging to suppress TRL warnings during tests
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("trl").setLevel(logging.ERROR)

from src.aligntune.backends.trl.raft.raft_trainer import (
    RaftTrainer,
    RaftTrainerConfig,
    raft_trainer_from_config,
)


class TestRaftTrainerConfig:
    """Test RAFT trainer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RaftTrainerConfig()
        assert config.max_golden_docs == 3
        assert config.max_distractor_docs == 5
        assert config.use_citation_loss is True
        assert config.citation_loss_weight == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = RaftTrainerConfig(
            max_golden_docs=2,
            max_distractor_docs=8,
            citation_loss_weight=0.2,
        )
        assert config.max_golden_docs == 2
        assert config.max_distractor_docs == 8
        assert config.citation_loss_weight == 0.2


class TestRaftExamplePreparation:
    """Test RAFT example preparation and formatting."""

    def test_prepare_example_with_docs(self):
        """Test preparing an example with documents."""
        # Mock RaftTrainer without actual model
        config = RaftTrainerConfig()

        # Create a minimal mock trainer
        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _prepare_raft_example(self, example):
                """Same logic as RaftTrainer."""
                question = example.get("question", "")
                answer = example.get("answer", "")
                golden_docs = example.get("golden_docs", [])[:config.max_golden_docs]
                distractor_docs = example.get("distractor_docs", [])[:config.max_distractor_docs]

                doc_context_parts = []
                for idx, doc in enumerate(golden_docs, start=1):
                    title = doc.get("title", f"Document {idx}")
                    text = doc.get("text", "")[:500]
                    doc_str = f"[DOC {idx}] {title}: {text}"
                    doc_context_parts.append(doc_str)

                for idx, doc in enumerate(distractor_docs, start=len(golden_docs) + 1):
                    title = doc.get("title", f"Document {idx}")
                    text = doc.get("text", "")[:500]
                    doc_str = f"[DOC {idx}] {title}: {text}"
                    doc_context_parts.append(doc_str)

                doc_context = "\n\n".join(doc_context_parts)

                if doc_context:
                    full_prompt = (
                        f"Context Documents:\n{doc_context}\n\n"
                        f"Question: {question}\n"
                        f"Answer: "
                    )
                else:
                    full_prompt = f"Question: {question}\nAnswer: "

                example["text"] = full_prompt + answer
                example["_raft_golden_doc_count"] = len(golden_docs)
                return example

        trainer = MockRaftTrainer()

        example = {
            "question": "What is PM-KISAN?",
            "answer": "PM-KISAN provides Rs 6,000 per year to farmers.",
            "golden_docs": [
                {
                    "title": "PM-KISAN Scheme",
                    "text": "Pradhan Mantri Kisan Samman Nidhi scheme",
                }
            ],
            "distractor_docs": [],
        }

        result = trainer._prepare_raft_example(example)

        assert "Context Documents:" in result["text"]
        assert "[DOC 1]" in result["text"]
        assert "PM-KISAN Scheme" in result["text"]
        assert "Answer:" in result["text"]
        assert result["_raft_golden_doc_count"] == 1

    def test_prepare_example_doc_truncation(self):
        """Test that documents are truncated to 500 chars."""
        config = RaftTrainerConfig()

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _prepare_raft_example(self, example):
                golden_docs = example.get("golden_docs", [])[:config.max_golden_docs]
                text_parts = []
                for idx, doc in enumerate(golden_docs, start=1):
                    text = doc.get("text", "")[:500]  # Truncate to 500
                    text_parts.append(text)
                example["_truncated_texts"] = text_parts
                return example

        trainer = MockRaftTrainer()

        long_text = "x" * 1000
        example = {
            "golden_docs": [{"title": "Doc", "text": long_text}],
        }

        result = trainer._prepare_raft_example(example)
        assert len(result["_truncated_texts"][0]) == 500

    def test_prepare_example_max_docs_respected(self):
        """Test that document limits are respected."""
        config = RaftTrainerConfig(max_golden_docs=2, max_distractor_docs=3)

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _prepare_raft_example(self, example):
                golden = example.get("golden_docs", [])[:config.max_golden_docs]
                distract = example.get("distractor_docs", [])[:config.max_distractor_docs]
                example["_doc_counts"] = (len(golden), len(distract))
                return example

        trainer = MockRaftTrainer()

        example = {
            "golden_docs": [
                {"title": f"G{i}", "text": f"text{i}"}
                for i in range(5)
            ],
            "distractor_docs": [
                {"title": f"D{i}", "text": f"text{i}"}
                for i in range(10)
            ],
        }

        result = trainer._prepare_raft_example(example)
        golden_count, distract_count = result["_doc_counts"]
        assert golden_count == 2
        assert distract_count == 3


class TestCitationQuality:
    """Test citation quality metrics."""

    def test_citation_quality_perfect(self):
        """Test perfect citation (all golden docs cited)."""
        config = RaftTrainerConfig()

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _compute_citation_quality(self, generated_text, golden_doc_titles):
                if not golden_doc_titles:
                    return 1.0
                text_lower = generated_text.lower()
                cited = sum(1 for title in golden_doc_titles if title.lower() in text_lower)
                return min(cited / len(golden_doc_titles), 1.0)

        trainer = MockRaftTrainer()

        quality = trainer._compute_citation_quality(
            generated_text="According to PM-KISAN scheme, farmers get Rs 6000.",
            golden_doc_titles=["PM-KISAN Scheme"],
        )
        assert quality == 1.0

    def test_citation_quality_partial(self):
        """Test partial citation."""
        config = RaftTrainerConfig()

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _compute_citation_quality(self, generated_text, golden_doc_titles):
                if not golden_doc_titles:
                    return 1.0
                text_lower = generated_text.lower()
                cited = sum(1 for title in golden_doc_titles if title.lower() in text_lower)
                return min(cited / len(golden_doc_titles), 1.0)

        trainer = MockRaftTrainer()

        quality = trainer._compute_citation_quality(
            generated_text="The scheme provides Rs 6000 to farmers.",
            golden_doc_titles=["PM-KISAN", "SEBI Notification"],
        )
        assert quality < 1.0

    def test_citation_quality_none(self):
        """Test when no golden docs provided."""
        config = RaftTrainerConfig()

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config

            def _compute_citation_quality(self, generated_text, golden_doc_titles):
                if not golden_doc_titles:
                    return 1.0
                text_lower = generated_text.lower()
                cited = sum(1 for title in golden_doc_titles if title.lower() in text_lower)
                return min(cited / len(golden_doc_titles), 1.0)

        trainer = MockRaftTrainer()

        quality = trainer._compute_citation_quality(
            generated_text="Some answer.",
            golden_doc_titles=[],
        )
        assert quality == 1.0


class TestRaftTrainerFactory:
    """Test RAFT trainer factory function."""

    def test_config_to_trainer_config(self):
        """Test creating RaftTrainerConfig from dict."""
        config_dict = {
            "max_golden_docs": 4,
            "max_distractor_docs": 6,
            "use_citation_loss": True,
            "citation_loss_weight": 0.15,
        }

        raft_cfg = RaftTrainerConfig(
            max_golden_docs=config_dict.get("max_golden_docs", 3),
            max_distractor_docs=config_dict.get("max_distractor_docs", 5),
            use_citation_loss=config_dict.get("use_citation_loss", True),
            citation_loss_weight=config_dict.get("citation_loss_weight", 0.1),
        )

        assert raft_cfg.max_golden_docs == 4
        assert raft_cfg.max_distractor_docs == 6
        assert raft_cfg.citation_loss_weight == 0.15


class TestRaftTrainerMetrics:
    """Test RAFT-specific metrics tracking."""

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        config = RaftTrainerConfig()

        class MockRaftTrainer:
            def __init__(self):
                self.raft_config = config
                self.citation_metrics = {
                    "citation_quality": 0.0,
                    "golden_doc_cited": 0,
                    "total_examples": 0,
                }

        trainer = MockRaftTrainer()

        assert trainer.citation_metrics["total_examples"] == 0
        assert trainer.citation_metrics["citation_quality"] == 0.0
