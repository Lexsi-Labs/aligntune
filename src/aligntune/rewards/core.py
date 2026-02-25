"""

Comprehensive reward function system for AlignTune.

This module provides a wide range of reward functions covering all types of tasks
and training scenarios for both SFT and RL training.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import logging
import re
import math
import json
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import ast
import time 

# Import robust math grading from shared module
from aligntune.utils.math_grading import (
    grade_math_answer,
    grade_gsm8k_answer,
    extract_math_answer,
    extract_gsm8k_answer,
)

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of reward functions."""
    # Basic rewards
    LENGTH = "length"
    COHERENCE = "coherence"
    
    
    # Task-specific rewards
    SENTIMENT = "sentiment"
    SAFETY = "safety"
    FACTUALITY = "factuality"
    BIAS = "bias"
    
    # Generation quality
    BLEU = "bleu"
    ROUGE = "rouge"
    METEOR = "meteor"
    BERTSCORE = "bertscore"
    
    # Code quality
    CODE_SYNTAX = "code_syntax"
    CODE_EXECUTION = "code_execution"
    CODE_COMPLETENESS = "code_completeness"
    
    # Math and reasoning
    MATH_CORRECTNESS = "math_correctness"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMMONSENSE = "commonsense"
    
    # Specialized
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    POLITENESS = "politeness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    # NEW: Math and reasoning
    MATH_REASONING = "math_reasoning"
    
    # NEW: Code rewards
    CODE_QUALITY = "code_quality"
    CODE_CORRECTNESS = "code_correctness"
    
    # NEW: Enhanced quality rewards
    DIVERSITY = "diversity"
    FLUENCY = "fluency"
    RELEVANCE = "relevance"
    BREVITY = "brevity"
    
    # NEW: Instruction following & alignment
    INSTRUCTION_FOLLOWING = "instruction_following"
    HARMLESSNESS = "harmlessness"
    CONCISENESS = "conciseness"
    
    # NEW: Context & temporal awareness
    CONTEXT_RELEVANCE = "context_relevance"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    
    # NEW: Advanced quality metrics
    SEMANTIC_SIMILARITY = "semantic_similarity"
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    
    # NEW: Domain-specific rewards
    MEDICAL_ACCURACY = "medical_accuracy"
    LEGAL_COMPLIANCE = "legal_compliance"
    FINANCIAL_ACCURACY = "financial_accuracy"
    
    # NEW: Advanced reasoning
    CAUSAL_REASONING = "causal_reasoning"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    
    # Multi-modal (for future)
    IMAGE_RELEVANCE = "image_relevance"
    AUDIO_QUALITY = "audio_quality"
    
    # New Rewards 
    COUNTERFACTUAL_MATH = "counterfactual_math"
    MBPP_REWARD = "mbpp_reward"


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_type: RewardType
    weight: float = 1.0
    params: Dict[str, Any] = None
    model_name: Optional[str] = None
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.device = self._setup_device()
        self._model = None
        self._tokenizer = None
    
    def _setup_device(self) -> str:
        """Setup device for reward computation."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute reward for given text."""
        raise NotImplementedError
    
    def batch_compute(self, texts: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        """Compute rewards for a batch of texts.
        
        For pipeline-based rewards, this uses batch processing for efficiency.
        For other rewards, falls back to sequential computation.
        """
        if references is None:
            references = [None] * len(texts)
        
        # Check if this reward supports batch processing via pipeline
        if hasattr(self, '_batch_compute_pipeline'):
            # Use batch processing for pipeline-based rewards
            return self._batch_compute_pipeline(texts, references, **kwargs)
        
        # Fallback to sequential computation
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references)]


class LengthReward(RewardFunction):
    """Reward based on text length."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        min_length = self.config.params.get("min_length", 10)
        max_length = self.config.params.get("max_length", 500)
        target_length = self.config.params.get("target_length", None)
        
        word_count = len(text.split())
        
        if target_length:
            # Reward closeness to target length
            distance = abs(word_count - target_length)
            max_distance = max(target_length, 100)
            return max(0.0, 1.0 - (distance / max_distance))
        else:
            # Reward within range
            if word_count < min_length or word_count > max_length:
                return 0.0
            return 1.0


class CoherenceReward(RewardFunction):
    """Reward for text coherence."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        # Simple coherence metrics
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        sentence_length_variance = np.var([len(s.split()) for s in sentences])
        
        # Reward moderate sentence length and low variance
        length_score = min(1.0, avg_sentence_length / 20.0)
        variance_penalty = max(0.0, 1.0 - (sentence_length_variance / 100.0))
        
        return (length_score + variance_penalty) / 2.0


class SentimentReward(RewardFunction):
    """Reward based on sentiment alignment."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._sentiment_pipeline = None
        self._max_length = 512  # Most sentiment models have max_length=512
    
    def _get_sentiment_pipeline(self):
        """Get or create sentiment analysis pipeline."""
        if self._sentiment_pipeline is None:
            model_name = self.config.model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            # Use pipeline's tokenizer (exact tokenizer used by pipeline)
            try:
                if hasattr(self._sentiment_pipeline, 'tokenizer') and self._sentiment_pipeline.tokenizer is not None:
                    # Get max_length from tokenizer's model_max_length
                    self._max_length = getattr(self._sentiment_pipeline.tokenizer, 'model_max_length', 512)
                    # Ensure it doesn't exceed 512 (most reward models have this limit)
                    self._max_length = min(self._max_length, 512)
            except Exception:
                pass  # Fallback to default 512
        return self._sentiment_pipeline
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_length tokens to prevent tensor size mismatches.
        
        Uses pipeline's tokenizer with add_special_tokens=True to let tokenizer
        automatically handle truncation accounting for special tokens.
        """
        if not text or len(text) == 0:
            return text
        
        # Use pipeline's tokenizer for accurate truncation
        pipeline = self._get_sentiment_pipeline()
        if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
            try:
                # Encode with add_special_tokens=True and let tokenizer handle truncation
                # This ensures special tokens ([CLS], [SEP]) are accounted for
                tokens = pipeline.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self._max_length,
                    truncation=True
                )
                # Decode back to text (tokenizer will have truncated appropriately)
                truncated_text = pipeline.tokenizer.decode(tokens, skip_special_tokens=True)
                return truncated_text
            except Exception:
                pass  # Fallback to character-based truncation
        
        # Fallback: character-based truncation (conservative estimate: 1 token ≈ 4 chars)
        # Account for special tokens by reducing max_chars slightly
        max_chars = (self._max_length - 2) * 3
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        target_sentiment = self.config.params.get("target_sentiment", "positive")
        
        try:
            # Truncate text to prevent tensor size mismatches
            text = self._truncate_text(text)
            
            pipeline = self._get_sentiment_pipeline()
            result = pipeline(text)[0]
            
            if result['label'].lower() == target_sentiment.lower():
                return result['score']
            else:
                return 1.0 - result['score']
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.5
    
    def _batch_compute_pipeline(self, texts: List[str], references: Optional[List[str]], **kwargs) -> List[float]:
        """Batch compute using pipeline for efficiency."""
        target_sentiment = self.config.params.get("target_sentiment", "positive")
        
        try:
            # Truncate all texts first
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            pipeline = self._get_sentiment_pipeline()
            # Use batch processing if pipeline supports it
            results = pipeline(truncated_texts)
            
            rewards = []
            for result in results:
                if isinstance(result, list):
                    result = result[0]
                if result['label'].lower() == target_sentiment.lower():
                    rewards.append(result['score'] * self.config.weight)
                else:
                    rewards.append((1.0 - result['score']) * self.config.weight)
            return rewards
        except Exception as e:
            logger.warning(f"Batch sentiment analysis failed, falling back to sequential: {e}")
            return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


class SafetyReward(RewardFunction):
    """Reward for safety compliance."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._safety_pipeline = None
        self._max_length = 512  # Most safety models (toxic-bert) have max_length=512
    
    def _get_safety_pipeline(self):
        """Get or create safety classification pipeline."""
        if self._safety_pipeline is None:
            model_name = self.config.model_name or "unitary/toxic-bert"
            self._safety_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            # Use pipeline's tokenizer (exact tokenizer used by pipeline)
            try:
                if hasattr(self._safety_pipeline, 'tokenizer') and self._safety_pipeline.tokenizer is not None:
                    # Get max_length from tokenizer's model_max_length
                    self._max_length = getattr(self._safety_pipeline.tokenizer, 'model_max_length', 512)
                    # Ensure it doesn't exceed 512 (most reward models have this limit)
                    self._max_length = min(self._max_length, 512)
            except Exception:
                pass  # Fallback to default 512
        return self._safety_pipeline
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_length tokens to prevent tensor size mismatches.
        
        Uses pipeline's tokenizer with add_special_tokens=True to let tokenizer
        automatically handle truncation accounting for special tokens.
        """
        if not text or len(text) == 0:
            return text
        
        # Use pipeline's tokenizer for accurate truncation
        pipeline = self._get_safety_pipeline()
        if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
            try:
                # Encode with add_special_tokens=True and let tokenizer handle truncation
                # This ensures special tokens ([CLS], [SEP]) are accounted for
                tokens = pipeline.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self._max_length,
                    truncation=True
                )
                # Decode back to text (tokenizer will have truncated appropriately)
                truncated_text = pipeline.tokenizer.decode(tokens, skip_special_tokens=True)
                return truncated_text
            except Exception:
                pass  # Fallback to character-based truncation
        
        # Fallback: character-based truncation (conservative estimate: 1 token ≈ 4 chars)
        # Account for special tokens by reducing max_chars slightly
        max_chars = (self._max_length - 2) * 3
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        try:
            # Truncate text to prevent tensor size mismatches
            text = self._truncate_text(text)
            
            pipeline = self._get_safety_pipeline()
            result = pipeline(text)[0]
            
            # Reward non-toxic content
            if result['label'].lower() in ['non-toxic', 'safe', 'clean']:
                return result['score']
            else:
                return 1.0 - result['score']
        except Exception as e:
            logger.warning(f"Safety analysis failed: {e}")
            return 0.5
    
    def _batch_compute_pipeline(self, texts: List[str], references: Optional[List[str]], **kwargs) -> List[float]:
        """Batch compute using pipeline for efficiency."""
        try:
            # Truncate all texts first
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            pipeline = self._get_safety_pipeline()
            # Use batch processing if pipeline supports it
            results = pipeline(truncated_texts)
            
            rewards = []
            for result in results:
                if isinstance(result, list):
                    result = result[0]
                # Reward non-toxic content
                if result['label'].lower() in ['non-toxic', 'safe', 'clean']:
                    rewards.append(result['score'] * self.config.weight)
                else:
                    rewards.append((1.0 - result['score']) * self.config.weight)
            return rewards
        except Exception as e:
            logger.warning(f"Batch safety analysis failed, falling back to sequential: {e}")
            return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


class BLEUReward(RewardFunction):
    """Reward based on BLEU score."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        try:
            from sacrebleu import BLEU
            self.bleu = BLEU()
        except ImportError:
            logger.warning("sacrebleu not available, using simple BLEU implementation")
            self.bleu = None
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        if self.bleu:
            try:
                score = self.bleu.sentence_score(text, [reference])
                return score.score / 100.0  # Normalize to 0-1
            except Exception as e:
                logger.warning(f"BLEU computation failed: {e}")
                return 0.0
        else:
            # Simple BLEU implementation
            return self._simple_bleu(text, reference)
    
    def _simple_bleu(self, text: str, reference: str) -> float:
        """Simple BLEU implementation."""
        text_tokens = text.lower().split()
        ref_tokens = reference.lower().split()
        
        if not text_tokens or not ref_tokens:
            return 0.0
        
        # 1-gram precision
        text_1grams = set(text_tokens)
        ref_1grams = set(ref_tokens)
        precision_1 = len(text_1grams & ref_1grams) / len(text_1grams)
        
        # 2-gram precision
        text_2grams = set(zip(text_tokens[:-1], text_tokens[1:]))
        ref_2grams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        precision_2 = len(text_2grams & ref_2grams) / len(text_2grams) if text_2grams else 0
        
        # Brevity penalty
        bp = min(1.0, len(text_tokens) / len(ref_tokens))
        
        # Combined score
        return bp * (precision_1 * precision_2) ** 0.5


class ROUGEReward(RewardFunction):
    """Reward based on ROUGE score."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            logger.warning("rouge_score not available, using simple ROUGE implementation")
            self.scorer = None
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        if self.scorer:
            try:
                scores = self.scorer.score(reference, text)
                # Use ROUGE-L F1 score
                return scores['rougeL'].fmeasure
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
                return 0.0
        else:
            # Simple ROUGE implementation
            return self._simple_rouge(text, reference)
    
    def _simple_rouge(self, text: str, reference: str) -> float:
        """Simple ROUGE-L implementation."""
        text_words = text.lower().split()
        ref_words = reference.lower().split()
        
        if not text_words or not ref_words:
            return 0.0
        
        # Longest Common Subsequence
        lcs_length = self._lcs_length(text_words, ref_words)
        
        precision = lcs_length / len(text_words)
        recall = lcs_length / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute LCS length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class MathCorrectnessReward(RewardFunction):
    """Reward for mathematical correctness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Extract mathematical expressions
        math_expressions = self._extract_math_expressions(text)
        if not math_expressions:
            return 0.0
        
        correct_count = 0
        for expr in math_expressions:
            if self._evaluate_math_expression(expr):
                correct_count += 1
        
        return correct_count / len(math_expressions)
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        # Pattern for simple arithmetic expressions
        pattern = r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*'
        expressions = re.findall(pattern, text)
        
        # Also look for equations
        equation_pattern = r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=\s*\d+\.?\d*'
        equations = re.findall(equation_pattern, text)
        
        return expressions + equations
    
    def _evaluate_math_expression(self, expr: str) -> bool:
        """Evaluate if a mathematical expression is correct."""
        try:
            # Handle equations
            if '=' in expr:
                left, right = expr.split('=')
                left_result = eval(left.strip())
                right_result = eval(right.strip())
                return abs(left_result - right_result) < 1e-6
            else:
                # Simple expression
                eval(expr)
                return True
        except:
            return False


class CodeSyntaxReward(RewardFunction):
    """Reward for code syntax correctness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        if not code_blocks:
            return 0.0
        
        correct_count = 0
        for code_block in code_blocks:
            code = code_block.strip('```').strip()
            if self._check_syntax(code):
                correct_count += 1
        
        return correct_count / len(code_blocks)
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code has valid syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False


class ToxicityReward(RewardFunction):
    """Reward for low toxicity."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._toxicity_pipeline = None
        self._max_length = 512  # Most toxicity models (toxic-bert) have max_length=512
    
    def _get_toxicity_pipeline(self):
        """Get or create toxicity detection pipeline."""
        if self._toxicity_pipeline is None:
            model_name = self.config.model_name or "unitary/toxic-bert"
            self._toxicity_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            # Use pipeline's tokenizer (exact tokenizer used by pipeline)
            try:
                if hasattr(self._toxicity_pipeline, 'tokenizer') and self._toxicity_pipeline.tokenizer is not None:
                    # Get max_length from tokenizer's model_max_length
                    self._max_length = getattr(self._toxicity_pipeline.tokenizer, 'model_max_length', 512)
                    # Ensure it doesn't exceed 512 (most reward models have this limit)
                    self._max_length = min(self._max_length, 512)
            except Exception:
                pass  # Fallback to default 512
        return self._toxicity_pipeline
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_length tokens to prevent tensor size mismatches.
        
        Uses pipeline's tokenizer with add_special_tokens=True to let tokenizer
        automatically handle truncation accounting for special tokens.
        """
        if not text or len(text) == 0:
            return text
        
        # Use pipeline's tokenizer for accurate truncation
        pipeline = self._get_toxicity_pipeline()
        if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
            try:
                # Encode with add_special_tokens=True and let tokenizer handle truncation
                # This ensures special tokens ([CLS], [SEP]) are accounted for
                tokens = pipeline.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self._max_length,
                    truncation=True
                )
                # Decode back to text (tokenizer will have truncated appropriately)
                truncated_text = pipeline.tokenizer.decode(tokens, skip_special_tokens=True)
                return truncated_text
            except Exception:
                pass  # Fallback to character-based truncation
        
        # Fallback: character-based truncation (conservative estimate: 1 token ≈ 4 chars)
        # Account for special tokens by reducing max_chars slightly
        max_chars = (self._max_length - 2) * 3
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        try:
            # Truncate text to prevent tensor size mismatches
            text = self._truncate_text(text)
            
            pipeline = self._get_toxicity_pipeline()
            result = pipeline(text)[0]
            
            # Reward non-toxic content (invert toxicity score)
            if result['label'].lower() in ['non-toxic', 'safe']:
                return result['score']
            else:
                return 1.0 - result['score']
        except Exception as e:
            logger.warning(f"Toxicity analysis failed: {e}")
            return 0.5
    
    def _batch_compute_pipeline(self, texts: List[str], references: Optional[List[str]], **kwargs) -> List[float]:
        """Batch compute using pipeline for efficiency."""
        try:
            # Truncate all texts first
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            pipeline = self._get_toxicity_pipeline()
            # Use batch processing if pipeline supports it
            results = pipeline(truncated_texts)
            
            rewards = []
            for result in results:
                if isinstance(result, list):
                    result = result[0]
                # Reward non-toxic content (invert toxicity score)
                if result['label'].lower() in ['non-toxic', 'safe']:
                    rewards.append(result['score'] * self.config.weight)
                else:
                    rewards.append((1.0 - result['score']) * self.config.weight)
            return rewards
        except Exception as e:
            logger.warning(f"Batch toxicity analysis failed, falling back to sequential: {e}")
            return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references or [None] * len(texts))]


class FactualityReward(RewardFunction):
    """Reward for factual accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Enhanced factuality check with more indicators and NER
        factual_indicators = [
            "according to", "research shows", "studies indicate", "data suggests",
            "evidence shows", "findings reveal", "analysis shows", "results indicate",
            "published in", "peer-reviewed", "scientific study", "research paper",
            "empirical evidence", "statistical analysis", "data analysis", "study found",
            "research indicates", "evidence suggests", "findings show", "analysis reveals"
        ]
        
        text_lower = text.lower()
        factual_count = sum(1 for indicator in factual_indicators if indicator in text_lower)
        
        # Check for named entities (basic NER pattern matching)
        # Common entity patterns: dates, locations, organizations
        entity_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}',  # Dates
            r'\b[A-Z][a-z]+\s+University\b',  # Universities
            r'\b[A-Z][a-z]+\s+Institute\b',  # Institutes
            r'\b(Dr\.|Professor|Dr)\s+[A-Z][a-z]+',  # Titles
        ]
        entity_count = sum(1 for pattern in entity_patterns if re.search(pattern, text))
        
        # Confidence scoring based on indicators and entities
        indicator_score = min(1.0, factual_count / 4.0)  # Normalize to 0-1
        entity_score = min(0.5, entity_count * 0.1)  # Entities add up to 0.5
        
        # Combine scores with weights
        total_score = (indicator_score * 0.7) + (entity_score * 0.3)
        
        return min(1.0, total_score)


class BiasReward(RewardFunction):
    """Reward for low bias."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Enhanced bias detection with more indicators and loaded language detection
        biased_terms = [
            "obviously", "clearly", "undoubtedly", "everyone knows",
            "it's common sense", "naturally", "of course", "without question",
            "anyone can see", "it's obvious", "plainly", "evidently",
            "self-evident", "goes without saying", "needless to say"
        ]
        
        # Additional bias indicators: generalizations, stereotypes
        generalization_patterns = [
            r'\ball\s+\w+\s+are\b',  # "all X are"
            r'\bevery\s+\w+\s+is\b',  # "every X is"
            r'\bno\s+\w+\s+ever\b',  # "no X ever"
            r'\bnever\s+\w+\s+do\b',  # "never X do"
        ]
        
        text_lower = text.lower()
        bias_count = sum(1 for term in biased_terms if term in text_lower)
        
        # Check for generalization patterns
        generalization_count = sum(1 for pattern in generalization_patterns if re.search(pattern, text_lower))
        
        # Demographic parity check (basic): look for balanced representation
        # Check for gender-neutral language
        gender_biased_terms = ["he/she", "his/her", "him/her"]
        gender_neutral_terms = ["they", "their", "them"]
        gender_biased_count = sum(1 for term in gender_biased_terms if term in text_lower)
        gender_neutral_count = sum(1 for term in gender_neutral_terms if term in text_lower)
        
        # Calculate bias score
        loaded_language_penalty = min(1.0, bias_count / 6.0)
        generalization_penalty = min(0.5, generalization_count * 0.25)
        gender_balance = 0.0 if gender_biased_count > 0 and gender_neutral_count == 0 else 0.2
        
        # Reward absence of biased language
        total_penalty = loaded_language_penalty + generalization_penalty
        bias_score = max(0.0, 1.0 - total_penalty) + gender_balance
        
        return min(1.0, bias_score)


class METEORReward(RewardFunction):
    """Reward based on METEOR score."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            # Tokenize
            text_tokens = nltk.word_tokenize(text.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            
            score = meteor_score([ref_tokens], text_tokens)
            return float(score)
            
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            return 0.0


class BERTScoreReward(RewardFunction):
    """Reward based on BERTScore."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._bertscorer = None
    
    def _get_bertscorer(self):
        """Get or create BERTScore scorer."""
        if self._bertscorer is None:
            try:
                from bert_score import score
                self._bertscorer = score
            except ImportError:
                logger.warning("bert_score not available")
                self._bertscorer = None
        return self._bertscorer
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        scorer = self._get_bertscorer()
        if scorer is None:
            return 0.0
        
        try:
            P, R, F1 = scorer([text], [reference], lang="en")
            return float(F1[0])
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return 0.0


class CodeExecutionReward(RewardFunction):
    """Reward for code execution success.

    Handles both:
    - MBPP-style assertion strings: ['assert func(x) == y', ...]
    - Dict-style test cases: [{'input': x, 'expected_output': y}, ...]
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.timeout = config.params.get('timeout', 5.0)
        self.test_cases = config.params.get('test_cases', [])

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Get test cases from kwargs or config
        test_cases = kwargs.get('test_cases', self.test_cases)

        # Debug: log when test_cases are received
        if test_cases:
            logger.debug(f"CodeExecutionReward: received {len(test_cases)} test cases")
        else:
            logger.debug(f"CodeExecutionReward: NO test cases (kwargs keys: {list(kwargs.keys())})")

        # Extract code from response
        code = self._extract_code(text)
        if not code:
            logger.debug("CodeExecutionReward: no code extracted")
            return 0.0

        # If test cases provided, validate against them
        if test_cases:
            score = self._validate_with_test_cases(code, test_cases)
            logger.debug(f"CodeExecutionReward: validation score = {score}")
            return score
        else:
            # Otherwise, just check if code executes without error
            return 1.0 if self._execute_code_safely(code) else 0.0

    def _extract_code(self, text: str) -> str:
        """Extract Python code from model response.

        Returns the FIRST code block containing a function definition,
        or combines all code blocks if needed.
        """
        # Try markdown code blocks first
        patterns = [
            r'```python\s*(.*?)```',
            r'```\s*(.*?)```',
        ]

        all_code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            all_code_blocks.extend(matches)

        if all_code_blocks:
            # Find the FIRST block that contains a function definition
            for block in all_code_blocks:
                if 'def ' in block:
                    return block.strip()
            # Fallback: return first block
            return all_code_blocks[0].strip()

        # Try to find function definition directly
        lines = text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            if in_function:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # Return full text as fallback
        return text.strip()

    def _execute_code_safely(self, code: str) -> bool:
        """Safely execute code and return success status with timeout."""
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out")

            # Set timeout (Unix only)
            old_handler = None
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

            try:
                # Use full builtins for proper execution
                namespace = {'__builtins__': __builtins__}
                exec(code, namespace)
                return True
            finally:
                # Cancel timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    if old_handler:
                        signal.signal(signal.SIGALRM, old_handler)

        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False

    def _validate_with_test_cases(self, code: str, test_cases: List[Any]) -> float:
        """Validate code against test cases.

        Handles both:
        - MBPP-style assertion strings: 'assert func(x) == y'
        - Dict-style: {'input': x, 'expected_output': y}

        Returns: float score (0.0 to 1.0) based on fraction of tests passed
        """
        if not test_cases:
            return 0.0

        passed = 0
        total = len(test_cases)

        for test_case in test_cases:
            try:
                if isinstance(test_case, str) and test_case.strip().startswith('assert'):
                    # MBPP-style assertion string
                    if self._run_assertion_test(code, test_case):
                        passed += 1
                elif isinstance(test_case, dict):
                    # Dict-style test case
                    if self._run_dict_test(code, test_case):
                        passed += 1
                else:
                    logger.debug(f"Unknown test case format: {type(test_case)}")
            except Exception as e:
                logger.debug(f"Test case validation failed: {e}")
                continue

        return passed / total if total > 0 else 0.0

    def _run_assertion_test(self, code: str, assertion: str) -> bool:
        """Run a single MBPP-style assertion test using subprocess for safety."""
        import subprocess
        import tempfile
        import os

        # Create a temporary file with the test code
        test_code = f'''{code}

{assertion}
print("PASS")
'''
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name

            # Run in subprocess with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Clean up
            os.unlink(temp_file)

            # Check if test passed
            return result.returncode == 0 and "PASS" in result.stdout

        except subprocess.TimeoutExpired:
            logger.debug("Assertion test timed out")
            try:
                os.unlink(temp_file)
            except:
                pass
            return False
        except Exception as e:
            logger.debug(f"Assertion test error: {e}")
            try:
                os.unlink(temp_file)
            except:
                pass
            return False

    def _run_dict_test(self, code: str, test_case: Dict[str, Any]) -> bool:
        """Run a single dict-style test case using subprocess for safety."""
        import subprocess
        import tempfile
        import os
        import json

        test_input = test_case.get('input')
        expected_output = test_case.get('expected_output')

        # Build test script
        test_code = f'''{code}

import json

# Get the function (first callable that's not builtin)
functions = {{k: v for k, v in dict(globals()).items()
            if callable(v) and not k.startswith('_') and k not in ('json',)}}
if not functions:
    exit(1)

main_func = list(functions.values())[0]
test_input = {repr(test_input)}
expected_output = {repr(expected_output)}

if isinstance(test_input, (list, tuple)) and len(test_input) > 1:
    actual_output = main_func(*test_input)
else:
    actual_output = main_func(test_input) if test_input is not None else main_func()

# Compare outputs
def compare(actual, expected):
    import math
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-9)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(compare(a, e) for a, e in zip(actual, expected))
    return actual == expected

if compare(actual_output, expected_output):
    print("PASS")
'''
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name

            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            os.unlink(temp_file)
            return result.returncode == 0 and "PASS" in result.stdout

        except subprocess.TimeoutExpired:
            logger.debug("Dict test timed out")
            try:
                os.unlink(temp_file)
            except:
                pass
            return False
        except Exception as e:
            logger.debug(f"Dict test error: {e}")
            try:
                os.unlink(temp_file)
            except:
                pass
            return False

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected outputs."""
        if isinstance(actual, float) and isinstance(expected, float):
            return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-9)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        return actual == expected


class CodeCompletenessReward(RewardFunction):
    """Reward for code completeness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        if not code_blocks:
            return 0.0
        
        completeness_scores = []
        for code_block in code_blocks:
            code = code_block.strip('```').strip()
            score = self._assess_completeness(code)
            completeness_scores.append(score)
        
        return sum(completeness_scores) / len(completeness_scores)
    
    def _assess_completeness(self, code: str) -> float:
        """Assess code completeness."""
        score = 0.0
        
        # Check for function definitions
        if re.search(r'def\s+\w+\s*\(', code):
            score += 0.3
        
        # Check for class definitions
        if re.search(r'class\s+\w+', code):
            score += 0.2
        
        # Check for imports
        if re.search(r'import\s+\w+', code):
            score += 0.1
        
        # Check for proper indentation
        lines = code.split('\n')
        if len(lines) > 1:
            indented_lines = sum(1 for line in lines[1:] if line.startswith('    ') or line.startswith('\t'))
            if indented_lines > 0:
                score += 0.2
        
        # Check for return statements
        if re.search(r'return\s+', code):
            score += 0.2
        
        return min(1.0, score)


class LogicalConsistencyReward(RewardFunction):
    """Reward for logical consistency."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Simple logical consistency checks
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        consistency_score = 1.0
        
        # Check for contradictory statements
        contradictions = [
            ("always", "never"),
            ("all", "none"),
            ("every", "no"),
            ("impossible", "possible"),
            ("true", "false")
        ]
        
        text_lower = text.lower()
        for pos, neg in contradictions:
            if pos in text_lower and neg in text_lower:
                consistency_score -= 0.2
        
        # Check for logical connectors
        logical_connectors = ["therefore", "thus", "hence", "because", "since", "as a result"]
        connector_count = sum(1 for connector in logical_connectors if connector in text_lower)
        consistency_score += min(0.3, connector_count * 0.1)
        
        return max(0.0, min(1.0, consistency_score))


class CommonsenseReward(RewardFunction):
    """Reward for commonsense reasoning."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Simple commonsense checks
        text_lower = text.lower()
        
        # Check for commonsense indicators
        commonsense_indicators = [
            "common sense", "obvious", "logical", "reasonable", "makes sense",
            "naturally", "typically", "usually", "generally", "normally"
        ]
        
        indicator_count = sum(1 for indicator in commonsense_indicators if indicator in text_lower)
        
        # Check for absurd statements (negative indicators)
        absurd_indicators = [
            "impossible", "absurd", "ridiculous", "nonsensical", "illogical"
        ]
        
        absurd_count = sum(1 for indicator in absurd_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(1.0, indicator_count * 0.2)
        score = max(0.0, score - absurd_count * 0.3)
        
        return score


class HallucinationReward(RewardFunction):
    """Reward for low hallucination."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Enhanced hallucination detection with self-consistency and contradiction checks
        text_lower = text.lower()
        context = kwargs.get('context', [])
        
        # Hallucination indicators (uncertainty)
        hallucination_indicators = [
            "i don't know", "i'm not sure", "i can't", "i'm unable to",
            "i don't have", "i don't recall", "i'm not certain",
            "i'm not familiar", "i cannot confirm", "i'm uncertain",
            "i have no information", "i lack knowledge", "i'm unsure"
        ]
        
        # Confidence indicators (positive)
        confidence_indicators = [
            "definitely", "certainly", "absolutely", "without doubt",
            "i'm confident", "i'm certain", "i know for sure",
            "verified", "confirmed", "established", "proven"
        ]
        
        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in text_lower)
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in text_lower)
        
        # Self-consistency check: look for contradictions within the text
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        contradiction_score = 0.0
        if len(sentences) > 1:
            # Check for contradictory statements
            contradiction_pairs = [
                ("always", "never"), ("all", "none"), ("every", "no"),
                ("impossible", "possible"), ("true", "false"), ("yes", "no"),
                ("correct", "incorrect"), ("right", "wrong")
            ]
            for pos, neg in contradiction_pairs:
                if pos in text_lower and neg in text_lower:
                    contradiction_score -= 0.3
                    break
        
        # Context consistency check (if context provided)
        context_consistency = 0.0
        if context and isinstance(context, list) and len(context) > 0:
            # Check if text contradicts previous context
            context_text = ' '.join(context).lower()
            # Simple check: if text mentions opposite of context
            if any(word in text_lower for word in ["not", "never", "no"]) and \
               any(word in context_text for word in ["yes", "always", "all"]):
                context_consistency -= 0.2
        
        # Calculate base score
        base_score = min(1.0, confidence_count * 0.25)
        base_score = max(0.0, base_score - hallucination_count * 0.15)
        
        # Apply consistency penalties
        final_score = base_score + contradiction_score + context_consistency
        
        return max(0.0, min(1.0, final_score))


class PolitenessReward(RewardFunction):
    """Reward for politeness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Simple politeness detection
        text_lower = text.lower()
        
        # Politeness indicators
        politeness_indicators = [
            "please", "thank you", "thanks", "you're welcome", "excuse me",
            "sorry", "apologize", "kindly", "grateful", "appreciate"
        ]
        
        # Impoliteness indicators
        impoliteness_indicators = [
            "shut up", "stupid", "idiot", "dumb", "annoying", "hate",
            "disgusting", "terrible", "awful", "horrible"
        ]
        
        politeness_count = sum(1 for indicator in politeness_indicators if indicator in text_lower)
        impoliteness_count = sum(1 for indicator in impoliteness_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(1.0, politeness_count * 0.2)
        score = max(0.0, score - impoliteness_count * 0.3)
        
        return score


class HelpfulnessReward(RewardFunction):
    """Reward for helpfulness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Simple helpfulness detection
        text_lower = text.lower()
        
        # Helpfulness indicators
        helpfulness_indicators = [
            "here's how", "let me help", "i can help", "here's what you can do",
            "suggest", "recommend", "advise", "guide", "assist", "support"
        ]
        
        # Unhelpfulness indicators
        unhelpfulness_indicators = [
            "i can't help", "i don't know", "i'm not sure", "i can't assist",
            "i'm unable to help", "i don't have that information"
        ]
        
        helpfulness_count = sum(1 for indicator in helpfulness_indicators if indicator in text_lower)
        unhelpfulness_count = sum(1 for indicator in unhelpfulness_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(1.0, helpfulness_count * 0.2)
        score = max(0.0, score - unhelpfulness_count * 0.3)
        
        return score


class HonestyReward(RewardFunction):
    """Reward for honesty."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Simple honesty detection
        text_lower = text.lower()
        
        # Honesty indicators
        honesty_indicators = [
            "honestly", "truthfully", "frankly", "to be honest", "i must admit",
            "i should mention", "i need to tell you", "i have to say"
        ]
        
        # Dishonesty indicators
        dishonesty_indicators = [
            "lie", "lying", "false", "fake", "deceive", "mislead", "trick"
        ]
        
        honesty_count = sum(1 for indicator in honesty_indicators if indicator in text_lower)
        dishonesty_count = sum(1 for indicator in dishonesty_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(1.0, honesty_count * 0.2)
        score = max(0.0, score - dishonesty_count * 0.4)
        
        return score
# Add these classes to rewards/core.py after the existing reward classes

class MathReasoningReward(RewardFunction):
    """Reward function for mathematical reasoning quality."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        total_score = 0.0
        components = 0
        
        params = self.config.params
        check_correctness = params.get('check_correctness', True)
        check_reasoning = params.get('check_reasoning', True)
        check_format = params.get('check_format', True)
        
        if check_correctness and reference:
            correctness_score = self._evaluate_correctness(text, reference)
            total_score += correctness_score
            components += 1
        
        if check_reasoning:
            reasoning_score = self._evaluate_reasoning(text)
            total_score += reasoning_score
            components += 1
        
        if check_format:
            format_score = self._evaluate_format(text)
            total_score += format_score
            components += 1
        
        final_score = (total_score / components) if components > 0 else 0.0
        return final_score * self.config.weight
    
    def _evaluate_correctness(self, response: str, reference: str) -> float:
        """Evaluate answer correctness using shared math_grading module.

        Uses robust three-tier grading for MATH-style problems:
        1. Official MATH dataset normalization
        2. PRM800K normalization
        3. SymPy symbolic equivalence

        Falls back to numeric matching for simple GSM8K-style answers.
        """
        params = self.config.params
        use_robust_grading = params.get('use_robust_grading', True)
        dataset_type = params.get('dataset_type', 'auto')

        if use_robust_grading:
            # Auto-detect dataset type based on reference format
            if dataset_type == 'auto':
                if '\\boxed' in reference or '\\frac' in reference:
                    dataset_type = 'math'
                elif '####' in reference:
                    dataset_type = 'gsm8k'
                else:
                    dataset_type = 'gsm8k'  # Default to simpler grading

            # Extract answers
            if dataset_type == 'math':
                pred_answer = extract_math_answer(response)
                gold_answer = extract_math_answer(reference) if '\\boxed' in reference else reference
                return 1.0 if grade_math_answer(pred_answer, gold_answer) else 0.0
            else:
                pred_answer = extract_gsm8k_answer(response)
                gold_answer = extract_gsm8k_answer(reference) if '####' in reference else reference
                return 1.0 if grade_gsm8k_answer(pred_answer, gold_answer) else 0.0

        # Legacy: simple numeric matching (for backwards compatibility)
        response_nums = self._extract_numbers(response)
        reference_nums = self._extract_numbers(reference)

        if not response_nums or not reference_nums:
            return 0.0

        for resp_num in response_nums:
            for ref_num in reference_nums:
                if self._numbers_match(resp_num, ref_num):
                    return 1.0
        return 0.0
    
    def _evaluate_reasoning(self, response: str) -> float:
        score = 0.0
        step_indicators = ['step 1', 'step 2', 'step 3', 'first', 'second', 'third', 
                          'then', 'next', 'after that', 'subsequently',
                          'therefore', 'thus', 'so', 'because', 'since', 'hence',
                          'we have', 'we get', 'we obtain', 'it follows']
        step_count = sum(1 for indicator in step_indicators if indicator in response.lower())
        if step_count >= 2:
            score += 0.4
        elif step_count >= 1:
            score += 0.2
        
        # Check for mathematical operations
        if bool(re.search(r'[+\-*/=]|\d+', response)):
            score += 0.3
        
        # Check for step-by-step structure
        sentences = response.split('.')
        if len(sentences) >= 3:
            score += 0.2
        
        # Check for clear answer statement
        answer_indicators = ['answer is', 'answer:', 'final answer', 'solution is', 
                            'result is', '=', 'equals', 'therefore the answer']
        if any(indicator in response.lower() for indicator in answer_indicators):
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_format(self, response: str) -> float:
        score = 0.0
        
        # Check for LaTeX math expressions (basic detection)
        latex_patterns = [
            r'\$[^$]+\$',  # Inline math: $...$
            r'\\\[.*?\\\]',  # Display math: \[...\]
            r'\\\(.*?\\\)',  # Inline math: \(...\)
            r'\\begin\{equation\}',  # Equation environment
        ]
        has_latex = any(re.search(pattern, response) for pattern in latex_patterns)
        if has_latex:
            score += 0.2
        
        # Check for standard math expressions
        if re.search(r'\d+\s*[+\-*/÷×]\s*\d+', response):
            score += 0.3
        if re.search(r'[=]', response):
            score += 0.2
        
        # Check for units (unit conversion awareness)
        units = ['meter', 'm', 'kilogram', 'kg', 'second', 's', 'hour', 'h',
                'dollar', '$', 'percent', '%', 'degree', '°', 'celsius', 'fahrenheit',
                'inch', 'foot', 'mile', 'pound', 'ounce', 'liter', 'l', 'gallon']
        unit_count = sum(1 for unit in units if unit in response.lower())
        if unit_count > 0:
            score += min(0.2, unit_count * 0.1)
        
        # Check for proper formatting
        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Enhanced number extraction with support for fractions and scientific notation."""
        # Standard decimal numbers
        pattern1 = r'-?\d+\.?\d*'
        matches1 = re.findall(pattern1, text)
        
        # Fractions (e.g., "1/2", "3/4")
        pattern2 = r'(\d+)\s*/\s*(\d+)'
        matches2 = re.findall(pattern2, text)
        
        # Scientific notation (e.g., "1.5e-3", "2E10")
        pattern3 = r'-?\d+\.?\d*[eE][+-]?\d+'
        matches3 = re.findall(pattern3, text)
        
        numbers = []
        
        # Process standard numbers
        for match in matches1:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        # Process fractions
        for num, den in matches2:
            try:
                numbers.append(float(num) / float(den))
            except (ValueError, ZeroDivisionError):
                continue
        
        # Process scientific notation
        for match in matches3:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _numbers_match(self, num1: float, num2: float, rel_tol: float = 1e-5) -> bool:
        return math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=1e-9)


class CodeQualityReward(RewardFunction):
    """Reward function for code quality assessment."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        params = self.config.params
        check_syntax = params.get('check_syntax', True)
        check_style = params.get('check_style', True)
        check_comments = params.get('check_comments', True)
        check_efficiency = params.get('check_efficiency', True)
        language = params.get('language', 'python')
        
        total_score = 0.0
        components = 0
        
        if check_syntax:
            total_score += self._evaluate_syntax(text, language)
            components += 1
        if check_style:
            total_score += self._evaluate_style(text, language)
            components += 1
        if check_comments:
            total_score += self._evaluate_comments(text)
            components += 1
        if check_efficiency:
            total_score += self._evaluate_efficiency(text, language)
            components += 1
        
        final_score = (total_score / components) if components > 0 else 0.0
        return final_score * self.config.weight
    
    # ... [Include all the helper methods from your grpo_trainer.py]
    def _evaluate_syntax(self, code: str, language: str) -> float:
        if language.lower() == "python":
            try:
                ast.parse(code)
                return 1.0
            except SyntaxError:
                lines = code.split('\n')
                valid_lines = sum(1 for line in lines if self._is_valid_line(line))
                return valid_lines / len(lines) if lines else 0.0
        else:
            return 0.8 if self._check_balanced_brackets(code) else 0.0

    def _is_valid_line(self, line: str) -> bool:
        try:
            ast.parse(line)
            return True
        except:
            return False

    def _evaluate_style(self, code: str, language: str) -> float:
        score = 0.0
        lines = [l for l in code.split('\n') if l.strip()]
        if self._check_consistent_indentation(lines):
            score += 0.25
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b'
        if len(re.findall(var_pattern, code)) > 0:
            score += 0.25
        if re.search(r'\w\s*[+\-*/=]\s*\w', code):
            score += 0.25
        long_lines = sum(1 for line in lines if len(line) > 120)
        if len(lines) > 0 and long_lines / len(lines) < 0.2:
            score += 0.25
        return score

    def _evaluate_comments(self, code: str) -> float:
        score = 0.0
        if '"""' in code or "'''" in code:
            score += 0.4
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if '#' in line or '//' in line)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            if 0.1 <= comment_ratio <= 0.3:
                score += 0.3
            elif comment_ratio > 0:
                score += 0.15
        if re.search(r'def\s+\w+.*:\s*"""', code, re.MULTILINE):
            score += 0.3
        return min(score, 1.0)

    def _evaluate_efficiency(self, code: str, language: str) -> float:
        score = 0.0
        nested_loops = len(re.findall(r'for.*:\s*for.*:\s*for', code, re.DOTALL))
        if nested_loops == 0:
            score += 0.3
        if language.lower() == "python" and '[' in code and 'for' in code and ']' in code:
            score += 0.2
        efficient_functions = ['map', 'filter', 'reduce', 'enumerate', 'zip', 'any', 'all']
        if any(func in code for func in efficient_functions):
            score += 0.2
        nested_for_count = len(re.findall(r'for\s+\w+\s+in', code))
        if nested_for_count <= 2:
            score += 0.3
        return min(score, 1.0)

    def _check_balanced_brackets(self, code: str) -> bool:
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        for char in code:
            if char in pairs.keys():
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs[stack.pop()] != char:
                    return False
        return len(stack) == 0

    def _check_consistent_indentation(self, lines: List[str]) -> bool:
        if not lines:
            return True
        indents = []
        for line in lines:
            if line.startswith(' ') or line.startswith('\t'):
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.append(indent)
        return len(set(indents)) <= 3 if indents else True


class CodeCorrectnessReward(RewardFunction):
    """Reward function for code correctness based on test cases."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.test_cases = config.params.get('test_cases', [])
        self.timeout = config.params.get('timeout', 5.0)
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        test_cases = kwargs.get('test_cases', self.test_cases)
        
        if not test_cases:
            try:
                compile(text, '<string>', 'exec')
                return 0.5 * self.config.weight
            except:
                return 0.0
        
        passed = sum(1 for test_case in test_cases if self._run_test_case(text, test_case))
        return (passed / len(test_cases)) * self.config.weight if test_cases else 0.0
    
    # ... [Include all the helper methods]
    def _run_test_case(self, code: str, test_case: Dict[str, Any]) -> bool:
        try:
            namespace = {'__builtins__': __builtins__}
            exec(code, namespace)
            functions = {k: v for k, v in namespace.items() 
                        if callable(v) and not k.startswith('_') and k != '__builtins__'}
            if not functions:
                return False
            main_func = list(functions.values())[0]
            test_input = test_case.get('input')
            expected_output = test_case.get('expected_output')
            if isinstance(test_input, (list, tuple)) and len(test_input) > 1:
                actual_output = main_func(*test_input)
            else:
                actual_output = main_func(test_input)
            return self._compare_outputs(actual_output, expected_output)
        except:
            return False

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        if isinstance(actual, float) and isinstance(expected, float):
            return math.isclose(actual, expected, rel_tol=1e-5)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        return actual == expected


class DiversityReward(RewardFunction):
    """Reward function for response diversity and vocabulary richness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        params = self.config.params
        check_vocabulary = params.get('check_vocabulary', True)
        check_sentence_variety = params.get('check_sentence_variety', True)
        
        score = 0.0
        
        if check_vocabulary:
            words = text.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                vocab_diversity = unique_words / len(words)
                score += min(vocab_diversity * 1.5, 0.5)
        
        if check_sentence_variety:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences]
                if lengths:
                    import statistics
                    try:
                        length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
                        variety_score = min(length_std / 10, 0.5)
                        score += variety_score
                    except:
                        score += 0.25
        
        return min(score, 1.0) * self.config.weight


class FluencyReward(RewardFunction):
    """Reward function for language fluency and grammatical quality."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        params = self.config.params
        check_grammar = params.get('check_grammar', True)
        check_coherence = params.get('check_coherence', True)
        
        score = 0.0
        
        if check_grammar:
            grammar_score = 0.0
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                capitalized = sum(1 for s in sentences if s and s[0].isupper())
                grammar_score += (capitalized / len(sentences)) * 0.3
            
            has_periods = text.count('.') > 0
            has_commas = text.count(',') > 0
            if has_periods:
                grammar_score += 0.2
            if has_commas:
                grammar_score += 0.1
            
            words = text.split()
            avg_sentence_length = len(words) / max(len(sentences), 1)
            if 5 <= avg_sentence_length <= 25:
                grammar_score += 0.4
            
            score += min(grammar_score, 0.6)
        
        if check_coherence:
            transitions = ['however', 'therefore', 'moreover', 'furthermore', 
                          'additionally', 'consequently', 'thus', 'hence', 
                          'meanwhile', 'similarly', 'in contrast', 'for example']
            has_transitions = any(trans in text.lower() for trans in transitions)
            if has_transitions:
                score += 0.2
            
            connectors = ['and', 'but', 'or', 'because', 'so', 'then', 'when', 'if']
            connector_count = sum(1 for conn in connectors if conn in text.lower())
            score += min(connector_count / 10, 0.2)
        
        return min(score, 1.0) * self.config.weight


class RelevanceReward(RewardFunction):
    """Reward function for topic relevance and focus."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        keywords = self.config.params.get('keywords', [])
        score = 0.0
        text_lower = text.lower()
        
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            score += min(keyword_matches / len(keywords), 0.5)
        
        answer_patterns = ['answer', 'solution', 'result', 'conclusion', 
                          'therefore', 'thus', 'in summary']
        has_answer_pattern = any(pat in text_lower for pat in answer_patterns)
        if has_answer_pattern:
            score += 0.3
        
        words = text.split()
        if 20 <= len(words) <= 300:
            score += 0.2
        elif len(words) < 20:
            score += 0.1
        
        return min(score, 1.0) * self.config.weight


class BrevityReward(RewardFunction):
    """Reward function for conciseness and brevity."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        ideal_length = self.config.params.get('ideal_length', 100)
        max_length = self.config.params.get('max_length', 300)
        
        words = text.split()
        word_count = len(words)
        
        if word_count <= ideal_length:
            return 1.0 * self.config.weight
        elif word_count <= max_length:
            penalty = (word_count - ideal_length) / (max_length - ideal_length)
            return (1.0 - penalty * 0.5) * self.config.weight
        else:
            excess = word_count - max_length
            penalty = min(excess / max_length, 1.0)
            return max(0.0, 0.5 - penalty) * self.config.weight


class ImageRelevanceReward(RewardFunction):
    """Placeholder reward for image relevance (multi-modal, future implementation)."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Placeholder implementation - returns neutral reward
        # TODO: Implement actual image relevance scoring when multi-modal support is added
        logger.warning("ImageRelevanceReward is a placeholder - multi-modal support not yet implemented")
        return 0.5 * self.config.weight


class AudioQualityReward(RewardFunction):
    """Placeholder reward for audio quality (multi-modal, future implementation)."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        # Placeholder implementation - returns neutral reward
        # TODO: Implement actual audio quality scoring when multi-modal support is added
        logger.warning("AudioQualityReward is a placeholder - multi-modal support not yet implemented")
        return 0.5 * self.config.weight


class InstructionFollowingReward(RewardFunction):
    """Reward for following instructions accurately."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        instruction = kwargs.get('instruction', None)
        if not instruction:
            # If no instruction provided, return neutral score
            return 0.5
        
        text_lower = text.lower()
        instruction_lower = instruction.lower()
        
        score = 0.0
        
        # Check instruction coverage - extract key terms from instruction
        instruction_words = set(instruction_lower.split())
        text_words = set(text_lower.split())
        
        # Check for instruction keywords in response
        instruction_keywords = ['write', 'explain', 'describe', 'list', 'compare', 
                               'analyze', 'summarize', 'create', 'generate', 'provide']
        keyword_matches = sum(1 for kw in instruction_keywords if kw in instruction_lower and kw in text_lower)
        if keyword_matches > 0:
            score += 0.3
        
        # Check for task completion indicators
        completion_indicators = ['done', 'complete', 'finished', 'answer', 'solution', 'result']
        if any(indicator in text_lower for indicator in completion_indicators):
            score += 0.2
        
        # Check format adherence (if format specified in instruction)
        format_keywords = ['format', 'structure', 'style', 'template']
        if any(kw in instruction_lower for kw in format_keywords):
            # Check if response follows some structure
            if any(char in text for char in ['\n', '\t', ':', '-', '*']):
                score += 0.2
        
        # Check for length requirements (if specified)
        if 'length' in instruction_lower or 'words' in instruction_lower:
            # Extract length requirement if possible
            length_match = re.search(r'(\d+)\s*(words?|sentences?)', instruction_lower)
            if length_match:
                required = int(length_match.group(1))
                actual_words = len(text.split())
                if abs(actual_words - required) <= required * 0.2:  # Within 20%
                    score += 0.3
        
        return min(1.0, score)


class HarmlessnessReward(RewardFunction):
    """Reward for avoiding harmful content (complement to safety)."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        
        # Check for harmful instructions or misuse potential
        harmful_patterns = [
            r'how\s+to\s+(harm|hurt|kill|destroy|hack|steal|cheat)',
            r'instructions?\s+to\s+(harm|hurt|kill|destroy|hack|steal)',
            r'ways?\s+to\s+(harm|hurt|kill|destroy|hack|steal)',
        ]
        
        harmful_count = sum(1 for pattern in harmful_patterns if re.search(pattern, text_lower))
        
        # Check for ethical compliance indicators
        ethical_indicators = [
            'ethical', 'responsible', 'legal', 'appropriate', 'safe',
            'respectful', 'considerate', 'harmless', 'benign'
        ]
        ethical_count = sum(1 for indicator in ethical_indicators if indicator in text_lower)
        
        # Check for disclaimers (positive indicator)
        disclaimer_indicators = [
            'disclaimer', 'warning', 'caution', 'note that', 'important',
            'should not', 'not recommended', 'not advised'
        ]
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        # Calculate score
        base_score = 1.0
        base_score -= harmful_count * 0.5  # Penalize harmful content
        base_score += min(0.3, ethical_count * 0.1)  # Reward ethical language
        base_score += min(0.2, disclaimer_count * 0.1)  # Reward disclaimers
        
        return max(0.0, min(1.0, base_score))


class ConcisenessReward(RewardFunction):
    """Reward for being concise without losing information."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        # Check for redundancy
        unique_words = len(set(word.lower() for word in words))
        redundancy_ratio = 1.0 - (unique_words / word_count) if word_count > 0 else 0.0
        
        # Check for filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 
                       'literally', 'really', 'very', 'quite', 'rather', 'somewhat']
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0.0
        
        # Check for repetitive phrases
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        repetitive_phrases = 0
        if len(sentences) > 1:
            sentence_words = [set(s.lower().split()) for s in sentences]
            for i in range(len(sentence_words) - 1):
                overlap = len(sentence_words[i] & sentence_words[i+1])
                if overlap > len(sentence_words[i]) * 0.5:  # More than 50% overlap
                    repetitive_phrases += 1
        
        # Calculate information density (simple heuristic)
        # More unique words relative to total = higher density
        information_density = unique_words / word_count if word_count > 0 else 0.0
        
        # Calculate score
        score = information_density * 0.5  # Base score from density
        score -= redundancy_ratio * 0.3  # Penalize redundancy
        score -= filler_ratio * 0.2  # Penalize fillers
        score -= min(0.3, repetitive_phrases * 0.1)  # Penalize repetition
        
        return max(0.0, min(1.0, score))


class ContextRelevanceReward(RewardFunction):
    """Reward for maintaining context relevance in conversations."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        context = kwargs.get('context', [])
        if not context or not isinstance(context, list):
            # Without context, return neutral score
            return 0.5
        
        if not text.strip():
            return 0.0
        
        text_lower = text.lower()
        context_text = ' '.join(context).lower()
        
        # Extract key terms from context
        context_words = set(context_text.split())
        text_words = set(text_lower.split())
        
        # Calculate topic overlap
        common_words = context_words & text_words
        if len(context_words) > 0:
            topic_overlap = len(common_words) / len(context_words)
        else:
            topic_overlap = 0.0
        
        # Check for topic drift indicators
        topic_shift_indicators = ['by the way', 'changing the subject', 'off topic',
                                 'unrelated', 'different topic', 'anyway']
        topic_shift_count = sum(1 for indicator in topic_shift_indicators if indicator in text_lower)
        
        # Check for context utilization
        context_utilization_indicators = ['as mentioned', 'as discussed', 'as you said',
                                          'referring to', 'in context', 'regarding']
        context_utilization = sum(1 for indicator in context_utilization_indicators 
                                 if indicator in text_lower)
        
        # Calculate score
        score = topic_overlap * 0.6  # Main component
        score -= topic_shift_count * 0.2  # Penalize topic shifts
        score += min(0.2, context_utilization * 0.1)  # Reward context utilization
        
        return max(0.0, min(1.0, score))


class TemporalConsistencyReward(RewardFunction):
    """Reward for temporal coherence across responses."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        previous_texts = kwargs.get('previous_texts', [])
        if not previous_texts or not isinstance(previous_texts, list):
            return 0.5  # Neutral without previous context
        
        if not text.strip():
            return 0.0
        
        text_lower = text.lower()
        previous_text = ' '.join(previous_texts).lower()
        
        score = 1.0
        
        # Check for temporal contradictions
        temporal_pairs = [
            ('yesterday', 'tomorrow'), ('past', 'future'), ('before', 'after'),
            ('earlier', 'later'), ('previous', 'next'), ('old', 'new'),
            ('ancient', 'modern'), ('recent', 'ancient')
        ]
        
        for past_term, future_term in temporal_pairs:
            if past_term in previous_text and future_term in text_lower:
                score -= 0.3
            if future_term in previous_text and past_term in text_lower:
                score -= 0.3
        
        # Check for timeline consistency
        time_indicators = ['now', 'currently', 'today', 'yesterday', 'tomorrow',
                          'recently', 'soon', 'later', 'earlier', 'previously']
        text_times = [ind for ind in time_indicators if ind in text_lower]
        prev_times = [ind for ind in time_indicators if ind in previous_text]
        
        # Check for anachronisms (basic)
        modern_terms = ['computer', 'internet', 'smartphone', 'email', 'website']
        historical_terms = ['ancient', 'medieval', 'renaissance', 'antiquity']
        
        if any(term in text_lower for term in historical_terms):
            if any(term in text_lower for term in modern_terms):
                score -= 0.2  # Anachronism detected
        
        # Check for consistent temporal references
        if text_times and prev_times:
            # If both have temporal references, check consistency
            if len(set(text_times) & set(prev_times)) > 0:
                score += 0.1  # Consistent temporal reference
        
        return max(0.0, min(1.0, score))


class SemanticSimilarityReward(RewardFunction):
    """Reward based on advanced semantic similarity."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self._sentence_model = None
    
    def _get_sentence_model(self):
        """Get or create sentence transformer model."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.model_name or "all-MiniLM-L6-v2"
                self._sentence_model = SentenceTransformer(model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, using simple similarity")
                self._sentence_model = None
        return self._sentence_model
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        model = self._get_sentence_model()
        if model is None:
            # Fallback to simple word overlap
            text_words = set(text.lower().split())
            ref_words = set(reference.lower().split())
            if len(text_words) == 0 or len(ref_words) == 0:
                return 0.0
            overlap = len(text_words & ref_words)
            union = len(text_words | ref_words)
            return overlap / union if union > 0 else 0.0
        
        try:
            # Use sentence transformers for semantic similarity
            embeddings = model.encode([text, reference])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0


class ReadabilityReward(RewardFunction):
    """Reward for text readability."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple Flesch-like readability score
        # Shorter sentences and words = higher readability
        sentence_score = 1.0 - min(1.0, (avg_sentence_length - 10) / 30)  # Optimal around 10-20 words
        word_score = 1.0 - min(1.0, (avg_word_length - 4) / 6)  # Optimal around 4-5 chars
        
        # Check for complex punctuation (reduces readability)
        complex_punct = text.count(';') + text.count(':') + text.count('—')
        punct_penalty = min(0.2, complex_punct * 0.05)
        
        # Combine scores
        readability_score = (sentence_score * 0.5 + word_score * 0.5) - punct_penalty
        
        return max(0.0, min(1.0, readability_score))


class EngagementReward(RewardFunction):
    """Reward for engaging and interesting content."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if not text.strip():
            return 0.0
        
        text_lower = text.lower()
        
        # Engagement indicators
        engagement_indicators = [
            'interesting', 'fascinating', 'compelling', 'engaging', 'captivating',
            'intriguing', 'thought-provoking', 'stimulating', 'exciting', 'remarkable'
        ]
        engagement_count = sum(1 for indicator in engagement_indicators if indicator in text_lower)
        
        # Check for questions (engages reader)
        question_count = text.count('?')
        
        # Check for exclamations (shows enthusiasm)
        exclamation_count = text.count('!')
        
        # Check for storytelling elements
        story_indicators = ['story', 'narrative', 'tale', 'journey', 'adventure',
                           'experience', 'example', 'scenario', 'situation']
        story_count = sum(1 for indicator in story_indicators if indicator in text_lower)
        
        # Check for vivid language (adjectives, adverbs)
        vivid_patterns = r'\b(very|extremely|incredibly|amazingly|remarkably|particularly)\s+\w+'
        vivid_count = len(re.findall(vivid_patterns, text_lower))
        
        # Calculate score
        score = min(0.4, engagement_count * 0.1)
        score += min(0.2, question_count * 0.1)
        score += min(0.1, exclamation_count * 0.05)
        score += min(0.2, story_count * 0.1)
        score += min(0.1, vivid_count * 0.05)
        
        return min(1.0, score)


class MedicalAccuracyReward(RewardFunction):
    """Reward for medical information accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        
        # Medical accuracy indicators
        accuracy_indicators = [
            'peer-reviewed', 'clinical study', 'medical research', 'scientific evidence',
            'published study', 'medical journal', 'clinical trial', 'evidence-based',
            'medically verified', 'healthcare professional', 'doctor', 'physician'
        ]
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        # Medical misinformation indicators
        misinformation_indicators = [
            'miracle cure', 'guaranteed to work', 'no side effects', 'secret remedy',
            'doctors hate this', 'one weird trick', 'instant results'
        ]
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        # Check for disclaimers (positive)
        disclaimer_indicators = [
            'consult a doctor', 'seek medical advice', 'not medical advice',
            'talk to your physician', 'medical disclaimer'
        ]
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        # Check for specific medical claims (should be cautious)
        medical_claim_patterns = [
            r'cures?\s+\w+', r'treats?\s+\w+', r'prevents?\s+\w+',
            r'causes?\s+\w+', r'leads?\s+to\s+\w+'
        ]
        claim_count = sum(1 for pattern in medical_claim_patterns if re.search(pattern, text_lower))
        
        # Calculate score
        score = min(0.5, accuracy_count * 0.1)
        score += min(0.3, disclaimer_count * 0.15)
        score -= misinformation_count * 0.4  # Heavy penalty for misinformation
        score -= min(0.2, claim_count * 0.1)  # Penalize unsupported claims
        
        return max(0.0, min(1.0, score))


class LegalComplianceReward(RewardFunction):
    """Reward for legal compliance and accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        jurisdiction = kwargs.get('jurisdiction', None)
        
        # Legal accuracy indicators
        accuracy_indicators = [
            'legal precedent', 'case law', 'statute', 'regulation', 'legal code',
            'court decision', 'legal opinion', 'attorney', 'lawyer', 'legal counsel',
            'jurisdiction', 'legal framework', 'compliance', 'legal requirement'
        ]
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        # Legal misinformation indicators
        misinformation_indicators = [
            'guaranteed legal', 'always legal', 'never illegal', 'legal loophole',
            'get away with', 'no consequences', 'legal trick'
        ]
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        # Check for disclaimers (positive)
        disclaimer_indicators = [
            'not legal advice', 'consult an attorney', 'seek legal counsel',
            'legal disclaimer', 'not a substitute for legal advice'
        ]
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        # Check for jurisdiction awareness
        jurisdiction_score = 0.0
        if jurisdiction:
            if jurisdiction.lower() in text_lower:
                jurisdiction_score = 0.2
        
        # Calculate score
        score = min(0.4, accuracy_count * 0.1)
        score += min(0.3, disclaimer_count * 0.15)
        score += jurisdiction_score
        score -= misinformation_count * 0.4  # Heavy penalty
        
        return max(0.0, min(1.0, score))


class FinancialAccuracyReward(RewardFunction):
    """Reward for financial information accuracy."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        
        # Financial accuracy indicators
        accuracy_indicators = [
            'financial analysis', 'market data', 'financial report', 'sec filing',
            'audited', 'certified', 'financial advisor', 'cpa', 'cfa',
            'financial regulation', 'compliance', 'financial disclosure'
        ]
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
        
        # Financial misinformation indicators
        misinformation_indicators = [
            'guaranteed returns', 'risk-free investment', 'get rich quick',
            'secret strategy', 'insider tip', 'sure thing', 'can\'t lose'
        ]
        misinformation_count = sum(1 for indicator in misinformation_indicators if indicator in text_lower)
        
        # Check for disclaimers
        disclaimer_indicators = [
            'not financial advice', 'consult a financial advisor', 'do your own research',
            'financial disclaimer', 'investment risk', 'past performance'
        ]
        disclaimer_count = sum(1 for indicator in disclaimer_indicators if indicator in text_lower)
        
        # Check for calculation accuracy (basic)
        calculation_patterns = [
            r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',  # Simple calculations
            r'\$\d+',  # Currency amounts
            r'\d+%',  # Percentages
        ]
        calculation_count = sum(1 for pattern in calculation_patterns if re.search(pattern, text))
        
        # Calculate score
        score = min(0.4, accuracy_count * 0.1)
        score += min(0.2, disclaimer_count * 0.1)
        score += min(0.2, calculation_count * 0.1)
        score -= misinformation_count * 0.4  # Heavy penalty
        
        return max(0.0, min(1.0, score))


class CausalReasoningReward(RewardFunction):
    """Reward for correct causal reasoning."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        
        # Causal relationship indicators
        causal_indicators = [
            'because', 'since', 'due to', 'as a result', 'therefore', 'thus',
            'causes', 'caused by', 'leads to', 'results in', 'brings about',
            'if...then', 'when...then', 'consequently', 'hence'
        ]
        causal_count = sum(1 for indicator in causal_indicators if indicator in text_lower)
        
        # Check for causal chains
        causal_chain_patterns = [
            r'\w+\s+causes?\s+\w+\s+which\s+causes?',  # A causes B which causes C
            r'\w+\s+leads?\s+to\s+\w+\s+which\s+leads?\s+to',  # A leads to B which leads to C
        ]
        chain_count = sum(1 for pattern in causal_chain_patterns if re.search(pattern, text_lower))
        
        # Check for logical causality (not just correlation)
        correlation_vs_causation = [
            'correlation', 'correlated', 'associated with', 'linked to'
        ]
        correlation_count = sum(1 for term in correlation_vs_causation if term in text_lower)
        
        # Check for proper causal structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        causal_structure = 0.0
        if len(sentences) >= 2:
            # Check if sentences are connected causally
            for i in range(len(sentences) - 1):
                if any(indicator in sentences[i+1].lower() for indicator in 
                      ['therefore', 'thus', 'hence', 'consequently', 'as a result']):
                    causal_structure += 0.1
        
        # Calculate score
        score = min(0.5, causal_count * 0.1)
        score += min(0.3, chain_count * 0.15)
        score += min(0.2, causal_structure)
        score -= min(0.2, correlation_count * 0.1)  # Slight penalty for correlation vs causation
        
        return max(0.0, min(1.0, score))


class CounterfactualReasoningReward(RewardFunction):
    """Reward for counterfactual reasoning quality."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        text_lower = text.lower()
        
        # Counterfactual indicators
        counterfactual_indicators = [
            'if', 'what if', 'suppose', 'imagine', 'hypothetically',
            'counterfactual', 'alternative scenario', 'different outcome',
            'would have', 'could have', 'might have', 'had it been'
        ]
        counterfactual_count = sum(1 for indicator in counterfactual_indicators if indicator in text_lower)
        
        # Check for logical consistency in counterfactuals
        consistency_indicators = [
            'then', 'therefore', 'thus', 'consequently', 'as a result',
            'logically', 'reasonably', 'plausibly'
        ]
        consistency_count = sum(1 for indicator in consistency_indicators if indicator in text_lower)
        
        # Check for plausibility markers
        plausibility_indicators = [
            'plausible', 'reasonable', 'likely', 'possible', 'feasible',
            'realistic', 'credible', 'believable'
        ]
        plausibility_count = sum(1 for indicator in plausibility_indicators if indicator in text_lower)
        
        # Check for contradiction (negative)
        contradiction_indicators = [
            'impossible', 'illogical', 'contradictory', 'inconsistent',
            'unrealistic', 'absurd'
        ]
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(0.4, counterfactual_count * 0.1)
        score += min(0.3, consistency_count * 0.1)
        score += min(0.2, plausibility_count * 0.1)
        score -= contradiction_count * 0.3  # Penalize contradictions
        
        return max(0.0, min(1.0, score))

    
    
    
## NEW reawrds    
class CounterfactualMathReward(RewardFunction):
    """Reward function with process-level partial credit for math reasoning."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.step = 0
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
        
        result = self._compute_process_reward(text, text, reference)
        return result['total_reward']
    
    def batch_compute(self, texts: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        if references is None or len(references) == 0:
            return [0.0] * len(texts)
        
        # Handle answer expansion for multiple completions per prompt
        factor = max(1, len(texts) // len(references))
        expanded = [a for a in references for _ in range(factor)]
        expanded = expanded[:len(texts)]
        
        rewards = []
        stats = {'correct': 0, 'incorrect': 0, 'process_scores': []}
        
        for text, gold in zip(texts, expanded):
            result = self._compute_process_reward(text, text, gold)
            rewards.append(result['total_reward'])
            
            if result['final_correct'] > 0.5:
                stats['correct'] += 1
            else:
                stats['incorrect'] += 1
            stats['process_scores'].append(result['process_score'])
        
        self.step += 1
        
        # Log statistics
        mean_reward = sum(rewards) / len(rewards)
        mean_process = sum(stats['process_scores']) / len(stats['process_scores'])
        accuracy = stats['correct'] / (stats['correct'] + stats['incorrect']) if (stats['correct'] + stats['incorrect']) > 0 else 0.0
        
        print(f"\n🎯 Counterfactual Math Reward (step {self.step})")
        print(f"   Accuracy: {accuracy:.2%} ({stats['correct']}/{len(texts)})")
        print(f"   Mean reward: {mean_reward:.4f}, Mean process: {mean_process:.4f}")
        import sys; sys.stdout.flush()
        
        return rewards
    
    def _compute_process_reward(self, completion: str, final_answer: str, gold_answer: str) -> Dict[str, float]:
        """Compute reward with partial credit for intermediate steps."""
        # Use robust MATH grading (handles tuples, LaTeX, symbolic answers)
        pred = extract_math_answer(completion)
        final_correct = 1.0 if grade_math_answer(pred, gold_answer) else 0.0
        
        steps = self._extract_calculation_steps(completion)
        if len(steps) == 0:
            process_score = 0.0
        else:
            correct_steps = sum(self._verify_calculation(expr, res) for expr, res in steps)
            process_score = correct_steps / len(steps)
        
        total_reward = 0.7 * final_correct + 0.3 * process_score
        
        return {
            'final_correct': final_correct,
            'process_score': process_score,
            'total_reward': total_reward,
            'num_steps': len(steps)
        }
    
    def _extract_calculation_steps(self, text: str) -> List[Tuple[str, str]]:
        """Extract intermediate calculations from reasoning text."""
        steps = []
        calc_pattern = r'(\d+(?:\.\d+)?)\s*([+\-×*/÷])\s*(\d+(?:\.\d+)?)\s*(?:=|is)\s*(\d+(?:\.\d+)?)'
        
        for match in re.finditer(calc_pattern, text.replace(',', '')):
            left, op, right, result = match.groups()
            expr = f"{left}{op}{right}"
            steps.append((expr, result))
        
        return steps
    
    def _verify_calculation(self, expr: str, claimed_result: str) -> bool:
        """Check if a calculation is correct."""
        try:
            parts = re.match(r'(\d+(?:\.\d+)?)([+\-×*/÷])(\d+(?:\.\d+)?)', expr)
            if not parts:
                return False
            
            left, op, right = parts.groups()
            left, right = float(left), float(right)
            claimed = float(claimed_result)
            
            op_map = {
                '+': left + right, 
                '-': left - right, 
                '×': left * right, 
                '*': left * right,
                '/': left / right if right != 0 else float('inf'),
                '÷': left / right if right != 0 else float('inf')
            }
            
            if op not in op_map:
                return False
            
            actual = op_map[op]
            return math.isclose(actual, claimed, rel_tol=1e-4, abs_tol=1e-8)
        except:
            return False
    
    def _parse_pred_number(self, text: str) -> str:
        """Parse predicted number from text."""
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        candidate = nums[-1] if nums else ""
        if re.fullmatch(r"-?\d+/\d+", candidate):
            num, den = candidate.split("/")
            try:
                return str(float(num) / float(den))
            except:
                return candidate
        return candidate
    
    def _numeric_equal(self, a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
        """Check if two numeric strings are equal."""
        try:
            fa = float(a)
            fb = float(b)
            return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
        except:
            return a.strip() == b.strip()


class MBPPReward:
    """MBPP-specific code correctness reward with batch processing and grouping."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.step = 0
        self.start_time = time.time()
        self.debug_mode = config.params.get('debug_mode', False)
        self.test_data_by_id = config.params.get('test_data_by_id', {})
        
        # Setup helper functions
        self._setup_helper_functions()
    
    def _setup_helper_functions(self):
        """Setup helper functions from your training script."""
        import unicodedata
        
        # ASCII translation table
        self._ASCII_TRANS = {
            ord("–"): "-",
            ord("—"): "-",
            ord("−"): "-",  # dashes/minus
            ord("×"): "*",
            ord("·"): "*",
            ord("“"): '"',
            ord("”"): '"',
            ord("„"): '"',
            ord("’"): "'",
            ord("‘"): "'",
            ord("²"): "**2",
            ord("³"): "**3",
            ord("\u00a0"): " ",  # non-breaking space
            ord("\u200b"): "",  # zero-width space
        }

    
    def batch_compute(self, completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions with proper grouping logic.
        This matches the original mbpp_reward_function behavior.
        """
        self.step += 1
        
        # Get prompts from kwargs (TRL passes them here)
        prompts = kwargs.get("prompts", [""] * len(completions))
        queries = kwargs.get("queries", [""] * len(completions))
        
        # L2-GRPO Fix: Deduplicate prompts to find actual batch size
        unique_prompts = []
        seen_prompts = set()
        for p in prompts:
            p_str = str(p)
            if p_str not in seen_prompts:
                unique_prompts.append(p)
                seen_prompts.add(p_str)
        
        # Calculate group size based on unique prompts
        if len(unique_prompts) > 0:
            group_size = len(completions) // len(unique_prompts)
        else:
            group_size = len(completions)
        group_size = max(1, group_size)
        
        if self.debug_mode:
            print(f"🔍 Grouping debug: {len(completions)} completions, {len(prompts)} prompts ({len(unique_prompts)} unique), group_size={group_size}")
        
        # Helper: map completion idx -> prompt group index
        def group_of(idx: int) -> int:
            return idx // group_size
        
        rewards = []
        completion_lengths = []
        brevity_bonuses_applied = 0
        
        # Step 1-6: Compute base rewards for each completion
        for i, completion in enumerate(completions):
            # Extract text content
            if isinstance(completion, dict) and "content" in completion:
                completion_code = completion["content"]
            else:
                completion_code = str(completion)
            
            completion_lengths.append(len(completion_code))
            
            # Extract sample_id using proper per-prompt grouping
            g = group_of(i)
            prompt_text = unique_prompts[g] if g < len(unique_prompts) else ""
            sample_id = g
            id_match = re.search(r"__SAMPLE_ID:(\d+)__", prompt_text or "")
            if id_match:
                sample_id = int(id_match.group(1))
            
            # Get tests for this sample ID
            tests_info = self.test_data_by_id.get(sample_id, {})
            setup_code = tests_info.get("setup", "")
            tests = tests_info.get("tests", [])
            
            if not tests:
                rewards.append(-0.10)
                continue
            
            # Extract expected function name and arity
            expected_func, expected_arity = self._extract_function_info_from_test(tests[0])
            if not expected_func:
                rewards.append(-0.10)
                continue
            
            # Sanitize and execute
            sanitized_code = self._prepare_for_exec(completion_code, expected_func)
            execution_success, passed, error_messages = self._run_code_in_sandbox(
                code=sanitized_code, setup_code=setup_code, tests=tests, timeout_seconds=5
            )
            
            # Calculate base reward
            base_reward = float(passed / len(tests)) if tests else 0.0
            
            if passed == 0:
                base_reward -= 0.10
            
            if not execution_success:
                timeout_crash = any("timed out" in msg for msg in error_messages)
                base_reward = -0.25 if timeout_crash else -0.15
            
            # Format penalties
            format_penalties = self._calculate_format_penalties(completion_code)
            
            # Brevity bonus (only for working solutions)
            brevity_bonus = 0.0
            if passed > 0 and execution_success:
                if len(completion_code) < 100:
                    brevity_bonus = 0.02
                    brevity_bonuses_applied += 1
                elif len(completion_code) < 150:
                    brevity_bonus = 0.01
                    brevity_bonuses_applied += 1
            
            total_reward = base_reward + format_penalties + brevity_bonus
            rewards.append(total_reward)
        
        # Step 7: Group completions by sample_id
        groups = {}
        for i in range(len(completions)):
            # Extract sample_id
            sample_id = i % len(self.test_data_by_id)
            if i < len(prompts) and prompts[i]:
                id_match = re.search(r"__SAMPLE_ID:(\d+)__", prompts[i])
                if id_match:
                    sample_id = int(id_match.group(1))
            
            completion_text = str(completions[i])
            if isinstance(completions[i], dict) and "content" in completions[i]:
                completion_text = completions[i]["content"]
            
            if sample_id not in groups:
                groups[sample_id] = []
            
            groups[sample_id].append((i, completion_text, rewards[i], completion_lengths[i]))
        
        # Step 8: Apply duplicate penalty within each group
        duplicate_penalties = [0.0] * len(rewards)
        
        for sample_id, group_items in groups.items():
            seen_texts = {}
            for idx, text, reward, length in group_items:
                if text in seen_texts:
                    duplicate_penalties[idx] = -0.02
                else:
                    seen_texts[text] = idx
        
        for i in range(len(rewards)):
            rewards[i] += duplicate_penalties[i]
        
        # Step 9: Per-group variance analysis and tie-breaking
        groups_with_zero_variance = 0
        
        for sample_id, group_items in groups.items():
            group_rewards = [item[2] for item in group_items]
            group_mean = sum(group_rewards) / len(group_rewards)
            group_variance = sum((r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)
            group_std = group_variance ** 0.5
            
            if group_std < 1e-6:  # Zero variance
                groups_with_zero_variance += 1
                # Apply tie-breaking by code length
                sorted_items = sorted(group_items, key=lambda x: x[3])
                for rank, (orig_idx, text, reward, length) in enumerate(sorted_items):
                    if len(sorted_items) > 1:
                        tie_breaker = 0.03 * (len(sorted_items) - rank - 1) / (len(sorted_items) - 1) - 0.015
                        rewards[orig_idx] += tie_breaker
        
        # Step 10: Soft clip to [-0.5, 1.0]
        rewards = [max(-0.5, min(1.0, r)) for r in rewards]
        
        # Sanity check
        import math
        for r in rewards:
            if not math.isfinite(r):
                raise RuntimeError(f"Non-finite reward: {r}")
        
        # Log metrics if wandb available
        
        return rewards
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute reward for a single completion."""
        # Extract test data from kwargs
        test_cases = kwargs.get('test_cases', None)
        sample_id = kwargs.get('sample_id', 0)
        prompt_text = kwargs.get('prompt', "")
        
        if test_cases is not None:
            tests = test_cases if isinstance(test_cases, list) else []
            setup_code = ""
        else:
            if prompt_text:
                id_match = re.search(r"__SAMPLE_ID:(\d+)__", prompt_text)
                if id_match:
                    sample_id = int(id_match.group(1))
            
            tests_info = self.test_data_by_id.get(sample_id, {})
            setup_code = tests_info.get("setup", "")
            tests = tests_info.get("tests", [])
        
        if not tests:
            return -0.10
        
        expected_func, expected_arity = self._extract_function_info_from_test(tests[0])
        if not expected_func:
            return -0.10
        
        sanitized_code = self._prepare_for_exec(text, expected_func)
        execution_success, passed, error_messages = self._run_code_in_sandbox(
            code=sanitized_code, setup_code=setup_code, tests=tests, timeout_seconds=5
        )
        
        base_reward = float(passed / len(tests)) if tests else 0.0
        
        if passed == 0:
            base_reward -= 0.10
        
        if not execution_success:
            timeout_crash = any("timed out" in msg for msg in error_messages)
            base_reward = -0.25 if timeout_crash else -0.15
        
        format_penalties = self._calculate_format_penalties(text)
        
        brevity_bonus = 0.0
        if passed > 0 and execution_success:
            if len(text) < 100:
                brevity_bonus = 0.02
            elif len(text) < 150:
                brevity_bonus = 0.01
        
        total_reward = base_reward + format_penalties + brevity_bonus
        return max(-0.5, min(1.0, total_reward))
    
    def _extract_function_info_from_test(self, test_str: str) -> Tuple[Optional[str], int]:
        """Extract function name and arity from test string."""
        try:
            match = re.search(r'(\w+)\([^)]*\)', test_str)
            if match:
                func_name = match.group(1)
                args_match = re.search(r'\(([^)]*)\)', test_str)
                if args_match:
                    args_str = args_match.group(1)
                    if not args_str.strip():
                        arity = 0
                    else:
                        arity = args_str.count(',') + 1
                        if args_str.startswith('(') and args_str.endswith(')'):
                            inner = args_str[1:-1].strip()
                            if inner and ',' in inner:
                                arity = 1
                else:
                    arity = 0
                return func_name, arity
        except:
            pass
        return None, 0
    
    def _prepare_for_exec(self, generated: str, expected_name: str) -> str:
        """Sanitize and prepare generated code for execution."""
        import ast
        import textwrap
        
        code = self._extract_python_code(generated)
        code = self._ascii_only(code)
        code = self._ensure_entrypoint(code, expected_name)
        
        try:
            ast.parse(code)
        except SyntaxError:
            code = f"def {expected_name}(*args, **kwargs):\n    raise SyntaxError('bad submission')\n"
        return textwrap.dedent(code)
    
    def _extract_python_code(self, text: str) -> str:
        """Extract clean Python code from text."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"/think|/no_think", "", text)
        
        python_blocks = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        code_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        return text.strip()
    
    def _ascii_only(self, s: str) -> str:
        """Convert string to ASCII-only."""
        import unicodedata
        s = s.translate(self._ASCII_TRANS)
        s = unicodedata.normalize("NFKC", s)
        return s.encode("ascii", "ignore").decode("ascii")
    
    def _ensure_entrypoint(self, code: str, expected_name: str) -> str:
        """Ensure the expected function name is defined."""
        if re.search(rf"\bdef\s+{re.escape(expected_name)}\s*\(", code):
            return code
        m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
        if m:
            actual = m.group(1)
            return code + f"\n\n# alias for grader\n{expected_name} = {actual}\n"
        return f"def {expected_name}(*args, **kwargs):\n    raise NotImplementedError\n\n" + code
    
    def _run_code_in_sandbox(self, code: str, setup_code: str, tests: List[str], timeout_seconds: int = 5) -> Tuple[bool, int, List[str]]:
        """Run code in a subprocess sandbox with timeout."""
        import subprocess
        import tempfile
        import sys
        
        error_messages = []
        input_blocker = "import builtins; builtins.input = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no input'))"

        test_script = f"""
import sys
import traceback

# Block input
{input_blocker}

# Setup code
{setup_code}

# Generated code
{code}

# Test execution
results = []
for i, test_code in enumerate({repr(tests)}):
    try:
        exec(test_code)
        results.append(f"PASS:{{i}}")
    except Exception as e:
        results.append(f"FAIL:{{i}}:{{type(e).__name__}}:{{str(e)}}")

# Output results
for result in results:
    print(result)
"""

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_script)
                temp_file = f.name

            result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=timeout_seconds)

            tests_passed = 0
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("PASS:"):
                        tests_passed += 1
                    elif line.startswith("FAIL:"):
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            error_messages.append(f"Test {parts[1]}: {parts[2]} - {parts[3]}")
                execution_success = True
            else:
                execution_success = False
                if result.stderr:
                    error_messages.append(f"Execution error: {result.stderr[:200]}")

            return execution_success, tests_passed, error_messages

        except subprocess.TimeoutExpired:
            error_messages.append("Code execution timed out (infinite loop)")
            return False, 0, error_messages
        except Exception as e:
            error_messages.append(f"Sandbox error: {str(e)}")
            return False, 0, error_messages
        finally:
            try:
                if "temp_file" in locals():
                    os.unlink(temp_file)
            except:
                pass
    
    def _calculate_format_penalties(self, text: str) -> float:
        """Calculate format penalties."""
        penalties = 0.0
        
        banned_tokens = ["```", "print(", "input(", "Example:", "Explanation:", "# Example", "# Test"]
        for token in banned_tokens:
            if token in text:
                penalties -= 0.02
        
        try:
            text.encode("ascii")
        except UnicodeEncodeError:
            penalties -= 0.03
        
        return penalties        
class RewardFunctionFactory:
    """Factory for creating reward functions."""
    
    _reward_classes = {
        RewardType.LENGTH: LengthReward,
        RewardType.COHERENCE: CoherenceReward,
        RewardType.FLUENCY: FluencyReward,
        RewardType.SENTIMENT: SentimentReward,
        RewardType.SAFETY: SafetyReward,
        RewardType.FACTUALITY: FactualityReward,
        RewardType.BIAS: BiasReward,
        RewardType.BLEU: BLEUReward,
        RewardType.ROUGE: ROUGEReward,
        RewardType.METEOR: METEORReward,
        RewardType.BERTSCORE: BERTScoreReward,
        RewardType.MATH_CORRECTNESS: MathCorrectnessReward,
        RewardType.CODE_SYNTAX: CodeSyntaxReward,
        RewardType.CODE_EXECUTION: CodeExecutionReward,
        RewardType.CODE_COMPLETENESS: CodeCompletenessReward,
        RewardType.LOGICAL_CONSISTENCY: LogicalConsistencyReward,
        RewardType.COMMONSENSE: CommonsenseReward,
        RewardType.HALLUCINATION: HallucinationReward,
        RewardType.TOXICITY: ToxicityReward,
        RewardType.POLITENESS: PolitenessReward,
        RewardType.HELPFULNESS: HelpfulnessReward,
        RewardType.HONESTY: HonestyReward,
        RewardType.MATH_REASONING: MathReasoningReward,
        RewardType.CODE_QUALITY: CodeQualityReward,
        RewardType.CODE_CORRECTNESS: CodeCorrectnessReward,
        RewardType.DIVERSITY: DiversityReward,
        RewardType.RELEVANCE: RelevanceReward,
        RewardType.BREVITY: BrevityReward,
        RewardType.INSTRUCTION_FOLLOWING: InstructionFollowingReward,
        RewardType.HARMLESSNESS: HarmlessnessReward,
        RewardType.CONCISENESS: ConcisenessReward,
        RewardType.CONTEXT_RELEVANCE: ContextRelevanceReward,
        RewardType.TEMPORAL_CONSISTENCY: TemporalConsistencyReward,
        RewardType.SEMANTIC_SIMILARITY: SemanticSimilarityReward,
        RewardType.READABILITY: ReadabilityReward,
        RewardType.ENGAGEMENT: EngagementReward,
        RewardType.MEDICAL_ACCURACY: MedicalAccuracyReward,
        RewardType.LEGAL_COMPLIANCE: LegalComplianceReward,
        RewardType.FINANCIAL_ACCURACY: FinancialAccuracyReward,
        RewardType.CAUSAL_REASONING: CausalReasoningReward,
        RewardType.COUNTERFACTUAL_REASONING: CounterfactualReasoningReward,
        RewardType.IMAGE_RELEVANCE: ImageRelevanceReward,
        RewardType.AUDIO_QUALITY: AudioQualityReward,
        RewardType.COUNTERFACTUAL_MATH: CounterfactualMathReward, # New 
        RewardType.MBPP_REWARD: MBPPReward,
    }
    
    @classmethod
    def create_reward(cls, config: RewardConfig) -> RewardFunction:
        """Create a reward function from config."""
        if config.reward_type not in cls._reward_classes:
            raise ValueError(f"Unknown reward type: {config.reward_type}")
        
        reward_class = cls._reward_classes[config.reward_type]
        return reward_class(config)
    
    @classmethod
    def create_rewards(cls, configs: List[RewardConfig]) -> List[RewardFunction]:
        """Create multiple reward functions."""
        return [cls.create_reward(config) for config in configs]
    
    @classmethod
    def register_reward(cls, reward_type: RewardType, reward_class: type):
        """Register a custom reward function."""
        cls._reward_classes[reward_type] = reward_class


class CompositeReward:
    """Composite reward function that combines multiple rewards."""
    
    def __init__(self, reward_functions: List[RewardFunction], weights: Optional[List[float]] = None):
        self.reward_functions = reward_functions
        self.weights = weights or [1.0] * len(reward_functions)
        
        if len(self.weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute weighted composite reward."""
        rewards = []
        for reward_func in self.reward_functions:
            try:
                reward = reward_func.compute(text, reference, **kwargs)
                rewards.append(reward)
            except Exception as e:
                logger.warning(f"Reward computation failed: {e}")
                rewards.append(0.0)
        
        # Weighted average
        weighted_sum = sum(w * r for w, r in zip(self.weights, rewards))
        total_weight = sum(self.weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def batch_compute(self, texts: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        """Compute composite rewards for a batch."""
        if references is None:
            references = [None] * len(texts)
        
        return [self.compute(text, ref, **kwargs) for text, ref in zip(texts, references)]
    
    def get_individual_rewards(self, text: str, reference: Optional[str] = None, **kwargs) -> Dict[str, float]:
        """Get individual reward scores."""
        individual_rewards = {}
        for i, reward_func in enumerate(self.reward_functions):
            try:
                reward = reward_func.compute(text, reference, **kwargs)
                individual_rewards[f"reward_{i}_{reward_func.config.reward_type.value}"] = reward
            except Exception as e:
                logger.warning(f"Individual reward computation failed: {e}")
                individual_rewards[f"reward_{i}_{reward_func.config.reward_type.value}"] = 0.0
        
        return individual_rewards


    
