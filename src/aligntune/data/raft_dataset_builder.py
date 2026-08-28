"""
RAFT Dataset Builder: Creates Retrieval Augmented Fine-Tuning datasets.

This module provides utilities to build RAFT training datasets from documents
and QA pairs. It handles:
1. Distractor sampling (BM25-based hard negative mining)
2. Dataset formatting for RAFT trainer
3. Train/eval splitting with stratification
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RaftExample:
    """A single RAFT training example."""
    question: str
    answer: str
    golden_docs: List[Dict[str, str]]  # [{"title": str, "text": str}, ...]
    distractor_docs: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for dataset."""
        return {
            "question": self.question,
            "answer": self.answer,
            "golden_docs": self.golden_docs,
            "distractor_docs": self.distractor_docs,
            **self.metadata,
        }


class BM25Ranker:
    """
    Simple BM25-like ranker for hard negative sampling.
    Falls back to TF-IDF if sklearn unavailable.
    """

    def __init__(self, documents: List[Dict[str, str]], use_sklearn: bool = True):
        """
        Initialize ranker with corpus.

        Args:
            documents: List of docs with 'title' and 'text' keys
            use_sklearn: Whether to use sklearn (else simple bag-of-words)
        """
        self.documents = documents
        self.use_sklearn = use_sklearn and SKLEARN_AVAILABLE

        if self.use_sklearn:
            self._init_sklearn()
        else:
            self._init_simple()

    def _init_sklearn(self):
        """Initialize sklearn TF-IDF vectorizer."""
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in self.documents]

        # For very small corpora, adjust parameters
        num_docs = len(texts)
        max_df = min(0.95, 1.0 - (1.0 / max(num_docs, 2)))  # Ensure at least 1 doc excluded

        try:
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=1,
                max_df=max_df,
                stop_words="english",
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        except (ValueError, Exception) as e:
            # Fallback to simple method if sklearn fails
            logger.debug(f"TF-IDF initialization failed: {e}. Falling back to simple ranker.")
            self.use_sklearn = False
            self._init_simple()

    def _init_simple(self):
        """Initialize simple bag-of-words ranker."""
        self.vocab = set()
        for doc in self.documents:
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower().split()
            self.vocab.update(text)
        self.vocab = list(self.vocab)

    def rank(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Rank documents by relevance to query.

        Args:
            query: Query text
            k: Number of top results to return

        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        if self.use_sklearn and hasattr(self, 'tfidf_matrix'):
            try:
                query_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            except Exception:
                # Fallback to simple method if sklearn fails
                query_words = set(query.lower().split())
                scores = []
                for doc in self.documents:
                    doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
                    doc_words = set(doc_text.split())
                    overlap = len(query_words & doc_words) / max(len(query_words), 1)
                    scores.append(overlap)
        else:
            # Simple word overlap
            query_words = set(query.lower().split())
            scores = []
            for doc in self.documents:
                doc_text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
                doc_words = set(doc_text.split())
                overlap = len(query_words & doc_words) / max(len(query_words), 1)
                scores.append(overlap)

        # Return top-k indices and scores
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


class RaftDatasetBuilder:
    """
    Builder for RAFT (Retrieval Augmented Fine-Tuning) datasets.

    Creates training examples with golden (relevant) and distractor
    (irrelevant) documents for each QA pair.
    """

    def __init__(
        self,
        num_distractors: int = 5,
        use_bm25: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize builder.

        Args:
            num_distractors: Number of distractor docs per example
            use_bm25: Use BM25/TF-IDF for hard negative mining
            random_seed: Random seed for reproducibility
        """
        self.num_distractors = num_distractors
        self.use_bm25 = use_bm25
        self.random_seed = random_seed
        random.seed(random_seed)

        self.documents = []
        self.qa_pairs = []
        self.doc_id_to_doc = {}  # For quick lookup

    def add_documents(self, documents: List[Dict[str, str]]) -> "RaftDatasetBuilder":
        """
        Add documents to the corpus.

        Args:
            documents: List of docs with 'title', 'text', optionally 'id', 'date', 'domain'

        Returns:
            Self for method chaining
        """
        for idx, doc in enumerate(documents):
            if "id" not in doc:
                doc["id"] = f"doc_{idx}"
            self.documents.append(doc)
            self.doc_id_to_doc[doc["id"]] = doc

        logger.info(f"Added {len(documents)} documents. Total corpus: {len(self.documents)}")
        return self

    def add_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> "RaftDatasetBuilder":
        """
        Add QA pairs to the dataset.

        Args:
            qa_pairs: List of dicts with at minimum 'question' and 'answer'.
                     Optionally: 'golden_doc_ids' (list of doc IDs that are relevant)

        Returns:
            Self for method chaining
        """
        self.qa_pairs.extend(qa_pairs)
        logger.info(f"Added {len(qa_pairs)} QA pairs. Total: {len(self.qa_pairs)}")
        return self

    def build(self) -> List[RaftExample]:
        """
        Build RAFT training examples.

        Returns:
            List of RaftExample objects ready for training
        """
        if not self.documents:
            raise ValueError("No documents added. Call add_documents() first.")
        if not self.qa_pairs:
            raise ValueError("No QA pairs added. Call add_qa_pairs() first.")

        examples = []
        ranker = BM25Ranker(self.documents, use_sklearn=self.use_bm25) if self.use_bm25 else None

        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")

            # Get golden docs (provided explicitly, or use BM25 to find them)
            golden_doc_ids = qa_pair.get("golden_doc_ids", [])
            golden_docs = [
                self.doc_id_to_doc[doc_id]
                for doc_id in golden_doc_ids
                if doc_id in self.doc_id_to_doc
            ]

            # If no golden docs specified, use BM25 to find them
            if not golden_docs and ranker:
                ranked = ranker.rank(question, k=3)
                golden_docs = [self.documents[idx] for idx, _ in ranked[:2]]

            # Sample distractors (hard negatives)
            distractor_docs = self._sample_distractors(
                question=question,
                golden_doc_ids={doc.get("id") for doc in golden_docs},
                ranker=ranker,
            )

            example = RaftExample(
                question=question,
                answer=answer,
                golden_docs=golden_docs,
                distractor_docs=distractor_docs,
                metadata=qa_pair.get("metadata", {}),
            )
            examples.append(example)

        logger.info(f"Built {len(examples)} RAFT examples")
        return examples

    def _sample_distractors(
        self,
        question: str,
        golden_doc_ids: Set[str],
        ranker: Optional[BM25Ranker],
    ) -> List[Dict[str, str]]:
        """
        Sample hard negative documents (distractors).

        Strategy:
        1. If BM25 available: rank all docs, take middle-ranked ones
        2. Else: random sampling
        3. Exclude golden docs

        Args:
            question: Query for ranking
            golden_doc_ids: Set of golden doc IDs to exclude
            ranker: Optional BM25 ranker

        Returns:
            List of distractor documents
        """
        candidate_docs = [
            doc for doc in self.documents
            if doc.get("id") not in golden_doc_ids
        ]

        if not candidate_docs:
            return []

        if ranker:
            # Get ranked list excluding golden docs
            ranked = ranker.rank(question, k=len(candidate_docs))
            distractor_indices = [
                idx for idx, _ in ranked
                if self.documents[idx].get("id") not in golden_doc_ids
            ]
            # Take middle-ranked ones (hard negatives, not random)
            start_idx = len(distractor_indices) // 4
            end_idx = start_idx + self.num_distractors
            distractor_docs = [
                self.documents[idx] for idx in distractor_indices[start_idx:end_idx]
            ]
        else:
            # Random sampling
            distractor_docs = random.sample(
                candidate_docs,
                min(self.num_distractors, len(candidate_docs)),
            )

        return distractor_docs

    def build_and_split(
        self,
        train_ratio: float = 0.8,
    ) -> Tuple[List[RaftExample], List[RaftExample]]:
        """
        Build examples and split into train/eval sets.

        Args:
            train_ratio: Fraction for training (rest goes to eval)

        Returns:
            Tuple of (train_examples, eval_examples)
        """
        examples = self.build()

        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        logger.info(
            f"Split into train ({len(train_examples)}) and eval ({len(eval_examples)})"
        )

        return train_examples, eval_examples


def build_raft_dataset_from_docs(
    documents: List[Dict[str, str]],
    qa_pairs: List[Dict[str, Any]],
    num_distractors: int = 5,
    use_bm25: bool = True,
) -> List[RaftExample]:
    """
    Convenience function to build RAFT dataset in one call.

    Args:
        documents: List of documents with 'title' and 'text'
        qa_pairs: List of QA pairs with 'question' and 'answer'
        num_distractors: Number of distractor docs per example
        use_bm25: Whether to use BM25 for ranking

    Returns:
        List of RAFT examples
    """
    builder = RaftDatasetBuilder(num_distractors=num_distractors, use_bm25=use_bm25)
    builder.add_documents(documents)
    builder.add_qa_pairs(qa_pairs)
    return builder.build()
