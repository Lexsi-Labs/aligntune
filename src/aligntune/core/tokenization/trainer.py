"""
Tokenization Trainer - Main orchestrator for vocabulary extension and adaptation.

This trainer orchestrates the complete tokenization adaptation workflow:
1. Vocabulary extension (naive or continued BPE)
2. Embedding initialization
3. Vocabulary pruning (optional)
4. Evaluation metrics

Follows AlignTune's trainer pattern for consistency with SFT/RL trainers.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import UnifiedTokenizationConfig
from .vocab.continued_bpe import extend_tokenizer_continued_bpe
from .vocab.naive_extension import extend_tokenizer_naive
from .vocab.pruning_wrapper import VocabularyPruner

logger = logging.getLogger(__name__)


class TokenizationTrainer:
    """
    Main trainer for tokenization adaptation.

    Orchestrates vocabulary extension, embedding initialization, pruning,
    and evaluation following AlignTune's trainer pattern.

    Examples:
        >>> from aligntune.core.tokenization import create_tokenization_trainer
        >>> trainer = create_tokenization_trainer(
        ...     base_model="meta-llama/Llama-2-7b-hf",
        ...     target_languages=["hi", "zh"],
        ...     dataset_name="wikimedia/wikipedia",
        ...     config_name="20231101.hi",
        ... )
        >>> result = trainer.train()
    """

    def __init__(self, config: UnifiedTokenizationConfig):
        """
        Initialize tokenization trainer.

        Args:
            config: UnifiedTokenizationConfig instance
        """
        self.config = config
        self.base_tokenizer = None
        self.results = {}

        logger.info("TokenizationTrainer initialized")
        logger.info(f"Method: {config.vocab_extension.method.value}, Pruning: {config.pruning.enabled}")

    def train(self) -> Dict[str, Any]:
        """
        Run the complete tokenization training workflow.

        Returns:
            Dictionary with training results and statistics

        Workflow:
            1. Load base tokenizer
            2. Extend vocabulary (naive or continued BPE)
            3. Prune vocabulary (optional)
            4. Save adapted tokenizer

        Note:
            Model loading and embedding initialization is handled by model_loader.py
            when using the extended tokenizer for training.
        """
        logger.info("="*80)
        logger.info("Starting Tokenization Training")
        logger.info("="*80)

        # Step 1: Load base tokenizer
        self._load_tokenizer()

        # Step 2: Vocabulary extension
        self._run_vocab_extension()

        # Step 3: Vocabulary pruning (optional)
        if self.config.pruning.enabled:
            self._run_pruning()

        # Step 4: Save tokenizer
        self._save_tokenizer()

        logger.info("="*80)
        logger.info("Tokenization Training Complete!")
        logger.info("="*80)

        return self.results

    def _load_tokenizer(self):
        """Load base tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer from: {self.config.model.base_model}")

        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_tokenizer or self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        logger.info(f"✓ Loaded tokenizer (vocab size: {len(self.base_tokenizer)})")

        self.results['original_vocab_size'] = len(self.base_tokenizer)

    def _run_vocab_extension(self):
        """Run vocabulary extension step."""
        logger.info("\n" + "-"*80)
        logger.info("Step: Vocabulary Extension")
        logger.info("-"*80)

        method = self.config.vocab_extension.method.value
        logger.info(f"Method: {method}")

        # Load corpus
        corpus = self._load_corpus()

        if method == "continued_bpe":
            result = extend_tokenizer_continued_bpe(
                base_tokenizer=self.base_tokenizer,
                corpus=corpus,
                num_new_tokens=self.config.model.new_tokens_count,
                is_sentencepiece=self._is_sentencepiece(),
                show_progress=True,
            )
        elif method == "naive_extension":
            result = extend_tokenizer_naive(
                base_tokenizer=self.base_tokenizer,
                corpus=corpus,
                new_tokenizer_vocab_size=self.config.model.new_tokens_count,
                model_type="bpe",
            )
        else:
            raise ValueError(f"Unknown extension method: {method}")

        logger.info(f"✓ Vocabulary extended: {result['num_added_tokens']} tokens added")
        self.results['vocab_extension'] = result


    def _run_pruning(self):
        """Run vocabulary pruning."""
        logger.info("\n" + "-"*80)
        logger.info("Step: Vocabulary Pruning")
        logger.info("-"*80)

        # Load eval corpus for pruning
        corpus = self._load_corpus(for_pruning=True)

        # Calculate number of tokens to prune
        current_size = len(self.base_tokenizer)
        target_size = int(current_size * (1 - self.config.pruning.pruning_ratio))
        n_prune = current_size - target_size

        logger.info(f"Pruning {n_prune} tokens ({self.config.pruning.pruning_ratio*100:.1f}%)")

        pruner = VocabularyPruner(method=self.config.pruning.method.value)
        pruner.train(self.base_tokenizer, corpus)
        result = pruner.prune(self.base_tokenizer, n_prune)

        logger.info(f"✓ Pruned to {result['new_vocab_size']} tokens")
        self.results['pruning'] = result

    def _save_tokenizer(self):
        """Save adapted tokenizer."""
        logger.info("\n" + "-"*80)
        logger.info("Step: Saving Tokenizer")
        logger.info("-"*80)

        output_dir = Path(self.config.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tokenizer locally
        self.base_tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ Saved tokenizer to {output_dir}")

        # Save results
        import json
        results_path = output_dir / "tokenization_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Saved results to {results_path}")

        self.results['output_dir'] = str(output_dir)

        # Push to HuggingFace Hub if requested
        hub_model_id = self.config.logging.hub_model_id
        if hub_model_id:
            logger.info(f"\nPushing tokenizer to HuggingFace Hub: {hub_model_id}")
            try:
                self.base_tokenizer.push_to_hub(
                    hub_model_id,
                    commit_message=f"Extended tokenizer with {self.results.get('vocab_extension', {}).get('num_added_tokens', 0)} new tokens"
                )
                logger.info(f"✓ Pushed tokenizer to https://huggingface.co/{hub_model_id}")
                self.results['hub_model_id'] = hub_model_id
            except Exception as e:
                logger.warning(
                    f"Failed to push to HuggingFace Hub: {e}\n"
                    f"Make sure you're logged in with: huggingface-cli login"
                )
                self.results['hub_push_error'] = str(e)

    def _load_corpus(self, for_pruning: bool = False) -> List[str]:
        """Load training or evaluation corpus using AlignTune's LoaderResolver."""
        from aligntune.data import load_corpus

        if for_pruning and self.config.pruning.eval_corpus_dataset:
            logger.info(f"Loading pruning corpus: {self.config.pruning.eval_corpus_dataset}")
            return load_corpus(
                dataset_name=self.config.pruning.eval_corpus_dataset,
                text_column=self.config.dataset.text_column,
                split=self.config.pruning.eval_corpus_split,
                max_samples=self.config.pruning.eval_corpus_samples,
                config_name=self.config.dataset.config_name,
            )
        else:
            logger.info(f"Loading corpus: {self.config.dataset.name}")
            return load_corpus(
                dataset_name=self.config.dataset.name,
                text_column=self.config.dataset.text_column,
                split=self.config.dataset.split,
                max_samples=self.config.dataset.max_samples,
                streaming=self.config.dataset.streaming,
                config_name=self.config.dataset.config_name,
            )

    def evaluate(self, corpus=None, eval_samples: int = 500) -> Dict[str, Any]:
        """
        Evaluate tokenizer fertility before and after extension.

        Compares the extended tokenizer with original base tokenizer on same corpus.
        Shows how much fertility improved (if any).

        Args:
            corpus: Corpus to evaluate on (if None, loads from config)
            eval_samples: Number of samples to evaluate (default 500)

        Returns:
            Dictionary with comparison results:
                - old_fertility: Base tokenizer fertility
                - new_fertility: Extended tokenizer fertility
                - fertility_reduction_percentage: % improvement
                - fertility_improvement_absolute: Absolute reduction in tokens/word
                - recommendation: User-friendly interpretation
                - sample_count: Number of texts evaluated

        Examples:
            >>> trainer = create_tokenization_trainer(...)
            >>> result = trainer.train()
            >>> eval_result = trainer.evaluate(eval_samples=1000)
            >>> print(f"Improvement: {eval_result['fertility_reduction_percentage']:.1f}%")
        """
        logger.info("="*80)
        logger.info("Evaluating Tokenizer Fertility: Before vs After Extension")
        logger.info("="*80)

        from transformers import AutoTokenizer
        from itertools import islice

        # Load corpus once
        if corpus is None:
            logger.info("Loading corpus from config for evaluation...")
            corpus = self._load_corpus()

        # Load base (original) tokenizer
        logger.info("Loading base tokenizer for comparison...")
        base_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_tokenizer or self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Extended tokenizer is already in self.base_tokenizer
        extended_tokenizer = self.base_tokenizer

        # Sample corpus ONCE
        logger.info(f"Sampling {eval_samples} texts for evaluation...")
        sample_texts = list(islice(corpus, eval_samples))
        sample_count = len(sample_texts)

        # Evaluate BOTH tokenizers in ONE loop (efficient)
        logger.info(f"Evaluating both tokenizers on {sample_count} texts...")

        total_tokens_before = 0
        total_tokens_after = 0
        total_words = 0

        for text in sample_texts:
            if not text or not isinstance(text, str):
                continue

            # Tokenize with BOTH tokenizers
            tokens_before = base_tokenizer.encode(text, add_special_tokens=False)
            tokens_after = extended_tokenizer.encode(text, add_special_tokens=False)
            words = text.split()

            total_tokens_before += len(tokens_before)
            total_tokens_after += len(tokens_after)
            total_words += len(words)

        # Calculate metrics
        if total_words == 0:
            logger.warning("No valid texts in corpus sample")
            result = {
                "sample_count": 0,
                "error": "No valid texts",
            }
            self.results['fertility_evaluation'] = result
            return result

        old_fertility = total_tokens_before / total_words
        new_fertility = total_tokens_after / total_words

        metrics_before = {
            "fertility": old_fertility,
            "total_tokens": total_tokens_before,
            "total_words": total_words,
            "sample_count": sample_count,
            "avg_tokens_per_text": total_tokens_before / sample_count if sample_count > 0 else 0,
        }

        metrics_after = {
            "fertility": new_fertility,
            "total_tokens": total_tokens_after,
            "total_words": total_words,
            "sample_count": sample_count,
            "avg_tokens_per_text": total_tokens_after / sample_count if sample_count > 0 else 0,
        }

        # Calculate improvement
        result = {
            "sample_count": len(sample_texts),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }

        if "error" not in metrics_before and "error" not in metrics_after:
            old_fertility = metrics_before["fertility"]
            new_fertility = metrics_after["fertility"]

            result["old_fertility"] = old_fertility
            result["new_fertility"] = new_fertility

            if old_fertility > 0:
                fertility_reduction = ((old_fertility - new_fertility) / old_fertility) * 100
                fertility_improvement = old_fertility - new_fertility
            else:
                fertility_reduction = 0
                fertility_improvement = 0

            result["fertility_reduction_percentage"] = fertility_reduction
            result["fertility_improvement_absolute"] = fertility_improvement

            # Generate recommendation
            if fertility_reduction < 0:
                recommendation = f"WARNING: Fertility INCREASED from {old_fertility:.2f} to {new_fertility:.2f}"
            elif fertility_reduction < 5:
                recommendation = f"Minimal improvement: {fertility_reduction:.1f}% reduction"
            elif fertility_reduction < 15:
                recommendation = f"Moderate improvement: {fertility_reduction:.1f}% reduction"
            elif fertility_reduction < 30:
                recommendation = f"Strong improvement: {fertility_reduction:.1f}% reduction"
            else:
                recommendation = f"Excellent improvement: {fertility_reduction:.1f}% reduction"

            result["recommendation"] = recommendation

            # Log results
            logger.info("="*80)
            logger.info("FERTILITY COMPARISON")
            logger.info("="*80)
            logger.info(f"Before: {old_fertility:.2f} tokens/word")
            logger.info(f"After:  {new_fertility:.2f} tokens/word")
            logger.info(f"Improvement: {fertility_reduction:.1f}%")
            logger.info(recommendation)
            logger.info("="*80)

        self.results['fertility_evaluation'] = result
        return result

    def _is_sentencepiece(self) -> bool:
        """Detect if tokenizer is SentencePiece-based."""
        from .vocab.detector import detect_tokenizer_type, TokenizerType

        tokenizer_type = detect_tokenizer_type(self.base_tokenizer)
        return tokenizer_type == TokenizerType.SENTENCEPIECE_BPE
