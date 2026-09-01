"""
Tests for RAFT Dataset Builder.

CPU-only tests with mock data. Validates:
1. Document addition and storage
2. QA pair handling
3. Distractor sampling (BM25 and random)
4. Example building and splitting
"""

import pytest
from typing import Dict, List, Any

from src.aligntune.data.raft_dataset_builder import (
    RaftDatasetBuilder,
    RaftExample,
    BM25Ranker,
    build_raft_dataset_from_docs,
)


class TestRaftExample:
    """Test RaftExample dataclass."""

    def test_raft_example_creation(self):
        """Test creating a RAFT example."""
        example = RaftExample(
            question="What is PM-KISAN?",
            answer="PM-KISAN provides Rs 6,000 per year to farmers.",
            golden_docs=[{"title": "PM-KISAN", "text": "Scheme details..."}],
            distractor_docs=[],
        )

        assert example.question == "What is PM-KISAN?"
        assert len(example.golden_docs) == 1
        assert len(example.distractor_docs) == 0

    def test_raft_example_to_dict(self):
        """Test converting example to dict."""
        example = RaftExample(
            question="Test Q",
            answer="Test A",
            golden_docs=[],
            distractor_docs=[],
            metadata={"source": "test"},
        )

        d = example.to_dict()
        assert d["question"] == "Test Q"
        assert d["answer"] == "Test A"
        assert d["source"] == "test"


class TestBM25Ranker:
    """Test BM25 ranker for document ranking."""

    def test_ranker_initialization(self):
        """Test initializing ranker with documents."""
        docs = [
            {"title": "Doc1", "text": "machine learning algorithms"},
            {"title": "Doc2", "text": "deep learning neural networks"},
            {"title": "Doc3", "text": "natural language processing"},
        ]

        ranker = BM25Ranker(docs, use_sklearn=False)
        assert len(ranker.documents) == 3

    def test_ranker_ranking(self):
        """Test that ranker ranks documents."""
        docs = [
            {"title": "Machine Learning", "text": "ML is about learning from data"},
            {"title": "Deep Learning", "text": "DL uses neural networks"},
            {"title": "Cooking", "text": "Recipes and cooking techniques"},
        ]

        ranker = BM25Ranker(docs, use_sklearn=False)
        ranked = ranker.rank("machine learning algorithms", k=2)

        # Should rank ML docs higher than cooking
        assert len(ranked) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)

    def test_ranker_top_k(self):
        """Test k parameter."""
        docs = [{"title": f"Doc{i}", "text": f"Text{i}"} for i in range(10)]

        ranker = BM25Ranker(docs, use_sklearn=False)
        ranked = ranker.rank("text", k=3)

        assert len(ranked) <= 3


class TestRaftDatasetBuilder:
    """Test RaftDatasetBuilder class."""

    def test_builder_initialization(self):
        """Test initializing builder."""
        builder = RaftDatasetBuilder(num_distractors=5)
        assert builder.num_distractors == 5
        assert len(builder.documents) == 0
        assert len(builder.qa_pairs) == 0

    def test_add_documents(self):
        """Test adding documents."""
        builder = RaftDatasetBuilder()
        docs = [
            {"title": "PM-KISAN", "text": "Scheme details"},
            {"title": "MGNREGA", "text": "Employment scheme"},
        ]

        builder.add_documents(docs)

        assert len(builder.documents) == 2
        assert builder.documents[0]["id"] == "doc_0"
        assert builder.documents[1]["id"] == "doc_1"

    def test_add_documents_with_ids(self):
        """Test adding documents with existing IDs."""
        builder = RaftDatasetBuilder()
        docs = [
            {"id": "custom_1", "title": "Doc1", "text": "Text1"},
            {"id": "custom_2", "title": "Doc2", "text": "Text2"},
        ]

        builder.add_documents(docs)

        assert builder.doc_id_to_doc["custom_1"]["title"] == "Doc1"
        assert builder.doc_id_to_doc["custom_2"]["title"] == "Doc2"

    def test_add_qa_pairs(self):
        """Test adding QA pairs."""
        builder = RaftDatasetBuilder()
        qa_pairs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        builder.add_qa_pairs(qa_pairs)

        assert len(builder.qa_pairs) == 2

    def test_method_chaining(self):
        """Test method chaining."""
        docs = [{"title": "Doc1", "text": "Text1"}]
        qa = [{"question": "Q1", "answer": "A1"}]

        builder = (
            RaftDatasetBuilder()
            .add_documents(docs)
            .add_qa_pairs(qa)
        )

        assert len(builder.documents) == 1
        assert len(builder.qa_pairs) == 1

    def test_build_without_documents_raises(self):
        """Test that building without documents raises error."""
        builder = RaftDatasetBuilder()
        builder.add_qa_pairs([{"question": "Q1", "answer": "A1"}])

        with pytest.raises(ValueError, match="No documents"):
            builder.build()

    def test_build_without_qa_raises(self):
        """Test that building without QA pairs raises error."""
        builder = RaftDatasetBuilder()
        builder.add_documents([{"title": "Doc1", "text": "Text1"}])

        with pytest.raises(ValueError, match="No QA pairs"):
            builder.build()

    def test_build_creates_examples(self):
        """Test that build creates valid examples."""
        builder = RaftDatasetBuilder(num_distractors=1)

        docs = [
            {"id": "doc_pm_kisan", "title": "PM-KISAN", "text": "Farmer scheme Rs 6000"},
            {"id": "doc_mgnrega", "title": "MGNREGA", "text": "Employment guarantee act"},
        ]

        qa = [
            {
                "question": "What is PM-KISAN?",
                "answer": "PM-KISAN provides Rs 6000",
                "golden_doc_ids": ["doc_pm_kisan"],
            }
        ]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        examples = builder.build()

        assert len(examples) == 1
        example = examples[0]
        assert isinstance(example, RaftExample)
        assert example.question == "What is PM-KISAN?"
        assert len(example.golden_docs) > 0

    def test_build_and_split(self):
        """Test train/eval splitting."""
        builder = RaftDatasetBuilder()

        docs = [
            {"id": f"doc_{i}", "title": f"Doc{i}", "text": f"Text{i}"}
            for i in range(5)
        ]

        qa = [
            {"question": f"Q{i}", "answer": f"A{i}", "golden_doc_ids": []}
            for i in range(10)
        ]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        train, eval = builder.build_and_split(train_ratio=0.8)

        assert len(train) == 8
        assert len(eval) == 2
        assert len(train) + len(eval) == 10

    def test_distractor_sampling_exclusion(self):
        """Test that golden docs are excluded from distractors."""
        builder = RaftDatasetBuilder(num_distractors=3)

        docs = [
            {"id": "golden_1", "title": "Golden", "text": "This is relevant"},
            {"id": "dist_1", "title": "Dist1", "text": "Irrelevant text 1"},
            {"id": "dist_2", "title": "Dist2", "text": "Irrelevant text 2"},
            {"id": "dist_3", "title": "Dist3", "text": "Irrelevant text 3"},
        ]

        qa = [
            {
                "question": "Test question?",
                "answer": "Test answer",
                "golden_doc_ids": ["golden_1"],
            }
        ]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        examples = builder.build()

        example = examples[0]
        golden_ids = {doc["id"] for doc in example.golden_docs}
        distractor_ids = {doc["id"] for doc in example.distractor_docs}

        # Golden should not appear in distractors
        assert len(golden_ids & distractor_ids) == 0


class TestConvenienceFunction:
    """Test convenience function for quick dataset building."""

    def test_build_raft_dataset_from_docs(self):
        """Test one-call dataset building."""
        docs = [
            {"title": "Doc1", "text": "Text 1"},
            {"title": "Doc2", "text": "Text 2"},
        ]

        qa = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        examples = build_raft_dataset_from_docs(docs, qa, num_distractors=1)

        assert len(examples) == 2
        assert all(isinstance(ex, RaftExample) for ex in examples)


class TestRaftBuilderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_build_with_single_document(self):
        """Test building with only one document."""
        builder = RaftDatasetBuilder(num_distractors=2)

        docs = [{"id": "only_doc", "title": "Only", "text": "Single doc"}]
        qa = [{"question": "Q", "answer": "A"}]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        examples = builder.build()

        assert len(examples) == 1
        # Should handle gracefully with 0 distractors
        assert len(examples[0].distractor_docs) == 0

    def test_build_with_empty_distractor_list(self):
        """Test when distractor list is empty."""
        builder = RaftDatasetBuilder(num_distractors=0)

        docs = [{"id": "d1", "title": "D1", "text": "Text"}]
        qa = [{"question": "Q", "answer": "A"}]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        examples = builder.build()

        assert len(examples[0].distractor_docs) == 0

    def test_metadata_preservation(self):
        """Test that QA metadata is preserved."""
        builder = RaftDatasetBuilder()

        docs = [{"id": "d1", "title": "D", "text": "T"}]
        qa = [
            {
                "question": "Q",
                "answer": "A",
                "metadata": {"difficulty": "hard", "domain": "legal"},
            }
        ]

        builder.add_documents(docs)
        builder.add_qa_pairs(qa)
        examples = builder.build()

        assert examples[0].metadata["difficulty"] == "hard"
        assert examples[0].metadata["domain"] == "legal"
