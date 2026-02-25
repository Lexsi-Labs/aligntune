"""
Evaluation system for unified SFT training.

This module provides evaluation capabilities for SFT models including
accuracy, perplexity, and task-specific metrics.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# from ...utils.inference_utils import maybe_enable_unsloth_inference

from .config import SFTConfig, TaskType

logger = logging.getLogger(__name__)

try:
    # tqdm is an optional dependency; fall back gracefully if unavailable
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - defensive import
    tqdm = None  # type: ignore


class SFTEvaluator:
    """Unified evaluator for SFT models."""
    
    def __init__(self, config: SFTConfig):
        """Initialize evaluator with configuration."""
        self.config = config
        self.eval_config = getattr(config, 'evaluation', None)
        if self.eval_config is None:
            # Backward compatibility: create default evaluation config
            from .config import EvaluationConfig
            self.eval_config = EvaluationConfig()
        logger.info("Initialized SFT evaluator")
    
    def evaluate(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any,
        config: SFTConfig
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            dataset: Evaluation dataset
            config: SFT configuration
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Enable Unsloth fast inference when available, then set eval mode
        try:
            from ..._imports import maybe_enable_unsloth_inference
            model = maybe_enable_unsloth_inference(model)
        except Exception as e:
            logger.warning(f"Skipping Unsloth inference: {e}")
        model.eval()
        
        metrics = {}
        
        # For classification tasks, compute loss-based metrics
        # For generation tasks, skip loss computation (handled by trainer.evaluate())
        task_type = config.dataset.task_type
        if task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            try:
                # Create evaluation data loader
                eval_loader = self._create_eval_loader(dataset, tokenizer, config)
                
                # Initialize metrics
                total_loss = 0.0
                total_samples = 0
                correct_predictions = 0
                
                with torch.no_grad():
                    for batch in eval_loader:
                        # Move batch to device
                        device = next(model.parameters()).device
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch.get("labels", input_ids).to(device)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        # Accumulate loss
                        loss = outputs.loss
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        
                        # Calculate accuracy for classification tasks
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        if "labels" in batch:
                            correct_predictions += (predictions == labels).sum().item()
                
                # Calculate final metrics
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                metrics = {
                    "eval_loss": avg_loss,
                }
                
                # Compute perplexity if enabled
                if self.eval_config.compute_perplexity:
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    metrics["eval_perplexity"] = perplexity
                
                # Add accuracy for classification tasks
                accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                metrics["eval_accuracy"] = accuracy
                
            except Exception as e:
                logger.warning(f"Could not compute loss-based metrics: {e}")
        
        # Add task-specific quality metrics (for all task types)
        quality_metrics = self._compute_task_specific_metrics(
            model,
            tokenizer,
            dataset,
            task_type,
            max_samples=self.eval_config.max_samples_for_quality_metrics,
        )
        metrics.update(quality_metrics)
        
        logger.info(f"Evaluation completed: {len(metrics)} metrics computed")
        return metrics
    
    def _compute_task_specific_metrics(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any,
        task_type: TaskType,
        max_samples: int = 50
    ) -> Dict[str, float]:
        """Compute task-specific quality metrics."""
        metrics: Dict[str, float] = {}
        
        # Limit dataset size for faster evaluation
        eval_dataset = dataset.select(range(min(max_samples, len(dataset)))) if len(dataset) > max_samples else dataset
        
        try:
            # Pre-generate predictions for generation-style tasks so that
            # multiple metrics (ROUGE, BLEU, BERTScore, etc.) can reuse them.
            predictions: Optional[List[str]] = None
            if task_type in [
                TaskType.INSTRUCTION_FOLLOWING,
                TaskType.SUPERVISED_FINE_TUNING,
                TaskType.TEXT_GENERATION,
                TaskType.CHAT_COMPLETION,
            ]:
                predictions = self._generate_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=eval_dataset,
                    max_samples=max_samples,
                )
            
            if task_type == TaskType.INSTRUCTION_FOLLOWING:
                metrics.update(
                    self._compute_instruction_following_metrics(
                        model,
                        tokenizer,
                        eval_dataset,
                        predictions=predictions,
                    )
                )
            elif task_type == TaskType.TEXT_GENERATION:
                metrics.update(
                    self._compute_text_generation_metrics(
                        model,
                        tokenizer,
                        eval_dataset,
                        predictions=predictions,
                    )
                )
            elif task_type == TaskType.CHAT_COMPLETION:
                metrics.update(
                    self._compute_chat_metrics(
                        model,
                        tokenizer,
                        eval_dataset,
                        predictions=predictions,
                    )
                )
            elif task_type == TaskType.TEXT_CLASSIFICATION:
                # Already computed accuracy above, but can add more
                pass
            elif task_type == TaskType.TOKEN_CLASSIFICATION:
                # Already computed accuracy above, but can add more
                pass
        except Exception as e:
            logger.warning(f"Could not compute task-specific metrics for {task_type.value}: {e}")
        
        return metrics

    def _generate_predictions(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any,
        max_samples: int,
    ) -> List[Optional[str]]:
        """
        Generate model predictions for evaluation examples.
        
        This performs a single pass over the evaluation subset and caches
        predictions so that multiple metrics can reuse them without
        re-generating.
        """
        model.eval()
        predictions: List[Optional[str]] = []
        
        iterable = dataset
        if tqdm is not None:
            # leave=True so the bar remains visible even for small eval sets
            iterable = tqdm(dataset, desc="evaluation", leave=True)
        
        with torch.no_grad():
            processed = 0
            for example in iterable:
                if processed >= max_samples:
                    break
                instruction, _ = self._extract_instruction_reference(example)
                if not instruction:
                    predictions.append(None)
                    processed += 1
                    continue
                
                pred = self._generate_response(model, tokenizer, instruction)
                predictions.append(pred)
                processed += 1
        
        return predictions
    
    def _compute_instruction_following_metrics(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute metrics for instruction following tasks."""
        metrics = {}

        # Optional perplexity for instruction-following tasks
        if getattr(self.eval_config, "compute_perplexity", False):
            try:
                ppl = self._compute_perplexity_generation(model, tokenizer, dataset)
                metrics["eval_perplexity"] = float(ppl)
            except Exception as e:
                logger.warning(f"Could not compute perplexity for instruction following: {e}")
        
        # ROUGE (if enabled)
        if self.eval_config.compute_rouge:
            rouge_metrics = self._compute_rouge_metrics(model, tokenizer, dataset, predictions)
            metrics.update(rouge_metrics)
        
        # BLEU (if enabled)
        if self.eval_config.compute_bleu:
            bleu_metrics = self._compute_bleu_metrics(model, tokenizer, dataset, predictions)
            metrics.update(bleu_metrics)
        
        # METEOR (if enabled)
        if self.eval_config.compute_meteor:
            meteor_metrics = self._compute_meteor_metrics(model, tokenizer, dataset, predictions)
            metrics.update(meteor_metrics)
        
        # BERTScore (if enabled)
        if self.eval_config.compute_bertscore:
            bertscore_metrics = self._compute_bertscore_metrics(model, tokenizer, dataset, predictions)
            metrics.update(bertscore_metrics)
        
        # Semantic Similarity (if enabled)
        if self.eval_config.compute_semantic_similarity:
            semantic_metrics = self._compute_semantic_similarity_metrics(model, tokenizer, dataset, predictions)
            metrics.update(semantic_metrics)
        
        # Custom metrics (if provided)
        if getattr(self.eval_config, "custom_metrics", None):
            custom_metrics = self._compute_custom_metrics(model, tokenizer, dataset, predictions)
            metrics.update(custom_metrics)
        
        # CodeBLEU (if enabled)
        if self.eval_config.compute_codebleu:
            codebleu_metrics = self._compute_codebleu_metrics(model, tokenizer, dataset)
            metrics.update(codebleu_metrics)
        
        return metrics

    def _compute_perplexity_generation(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any,
    ) -> float:
        """
        Compute perplexity for generation-style tasks by scoring the
        reference response conditioned on the extracted instruction.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for example in dataset:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    continue

                prompt = instruction
                target = reference

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        target,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )

                input_ids = inputs["input_ids"]
                labels_ids = labels["input_ids"]

                # Concatenate prompt and target; loss only on target tokens
                full_ids = torch.cat([input_ids, labels_ids], dim=1)
                labels_full = full_ids.clone()
                labels_full[:, : input_ids.shape[1]] = -100

                device = next(model.parameters()).device
                full_ids = full_ids.to(device)
                labels_full = labels_full.to(device)

                outputs = model(input_ids=full_ids, labels=labels_full)
                loss = outputs.loss

                num_target_tokens = labels_ids.numel()
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return float(torch.exp(torch.tensor(avg_loss)).item())

    def _compute_custom_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute user-provided custom metrics.
        
        Each callable in eval_config.custom_metrics should accept:
            (prediction: str, reference: str, instruction: str) -> Dict[str, float]
        and return a dict of metric_name -> value (numeric).
        """
        metrics: Dict[str, List[float]] = {}
        custom_fns: List[Callable] = getattr(self.eval_config, "custom_metrics", []) or []
        if not custom_fns:
            return {}

        processed = 0
        max_samples = getattr(self.eval_config, "max_samples_for_quality_metrics", 50)

        for idx, example in enumerate(dataset):
            if processed >= max_samples:
                break
            instruction, reference = self._extract_instruction_reference(example)
            if not instruction or not reference:
                continue

            prediction = None
            if predictions is not None and idx < len(predictions):
                prediction = predictions[idx]
            if prediction is None:
                prediction = self._generate_response(model, tokenizer, instruction)
            if prediction is None:
                continue

            for fn in custom_fns:
                try:
                    result = fn(prediction, reference, instruction)
                    if isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                metrics.setdefault(k, []).append(v)
                except Exception as e:
                    logger.debug(f"Custom metric failed: {e}")
                    continue

            processed += 1

        # Aggregate means
        aggregated = {k: float(np.mean(v)) for k, v in metrics.items() if v}
        return aggregated
    
    def _compute_rouge_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        metrics = {}
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except ImportError:
            logger.warning("rouge-score not available. Install with: pip install rouge-score")
            return metrics
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        processed_count = 0
        skipped_count = 0
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    skipped_count += 1
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    skipped_count += 1
                    continue
                
                # Compute ROUGE
                scores = scorer.score(reference, generated)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
                processed_count += 1
            except Exception as e:
                logger.debug(f"Error computing ROUGE: {e}")
                skipped_count += 1
                continue
        
        if rouge_scores['rouge1']:
            metrics["eval_rouge1"] = float(np.mean(rouge_scores['rouge1']))
            metrics["eval_rouge2"] = float(np.mean(rouge_scores['rouge2']))
            metrics["eval_rougeL"] = float(np.mean(rouge_scores['rougeL']))
        elif skipped_count > 0:
            logger.warning(f"ROUGE: Could not extract instruction/reference from {skipped_count} examples. Dataset format may not be supported.")
        
        return metrics
    
    def _compute_bleu_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute BLEU score."""
        metrics = {}
        try:
            from nltk.translate.bleu_score import sentence_bleu
            import nltk
            nltk.download('punkt', quiet=True)
        except ImportError:
            logger.warning("nltk not available for BLEU. Install with: pip install nltk")
            return metrics
        
        bleu_scores = []
        processed_count = 0
        skipped_count = 0
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    skipped_count += 1
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    skipped_count += 1
                    continue
                
                reference_tokens = reference.split()
                generated_tokens = generated.split()
                if reference_tokens and generated_tokens:
                    bleu = sentence_bleu([reference_tokens], generated_tokens)
                    bleu_scores.append(bleu)
                    processed_count += 1
            except Exception as e:
                logger.debug(f"Error computing BLEU: {e}")
                skipped_count += 1
                continue
        
        if bleu_scores:
            metrics["eval_bleu"] = float(np.mean(bleu_scores))
        elif skipped_count > 0:
            logger.warning(f"BLEU: Could not extract instruction/reference from {skipped_count} examples. Dataset format may not be supported.")
        
        return metrics
    
    def _compute_meteor_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute METEOR score (better than BLEU, considers synonyms)."""
        metrics = {}
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except ImportError:
            logger.warning("nltk not available for METEOR. Install with: pip install nltk")
            return metrics
        
        meteor_scores = []
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    continue
                
                reference_tokens = reference.split()
                generated_tokens = generated.split()
                if reference_tokens and generated_tokens:
                    meteor = meteor_score([reference_tokens], generated_tokens)
                    meteor_scores.append(meteor)
            except Exception as e:
                logger.debug(f"Error computing METEOR: {e}")
                continue
        
        if meteor_scores:
            metrics["eval_meteor"] = float(np.mean(meteor_scores))
        
        return metrics
    
    def _compute_bertscore_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute BERTScore (semantic similarity using BERT embeddings)."""
        metrics = {}
        try:
            from bert_score import score as bert_score
        except ImportError:
            logger.warning("bert-score not available. Install with: pip install bert-score")
            return metrics
        
        references = []
        predictions = []
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    continue
                
                references.append(reference)
                predictions.append(generated)
            except Exception as e:
                logger.debug(f"Error preparing BERTScore: {e}")
                continue
        
        if references and predictions:
            try:
                P, R, F1 = bert_score(
                    predictions,
                    references,
                    lang='en',
                    model_type=self.eval_config.bertscore_model,
                    verbose=False
                )
                metrics["eval_bertscore_precision"] = float(P.mean().item())
                metrics["eval_bertscore_recall"] = float(R.mean().item())
                metrics["eval_bertscore_f1"] = float(F1.mean().item())
            except Exception as e:
                logger.warning(f"Could not compute BERTScore: {e}")
        
        return metrics
    
    def _compute_semantic_similarity_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute semantic similarity using sentence transformers."""
        metrics = {}
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
            return metrics
        
        try:
            st_model = SentenceTransformer(self.eval_config.semantic_similarity_model)
        except Exception as e:
            logger.warning(f"Could not load sentence transformer model: {e}")
            return metrics
        
        references = []
        predictions = []
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    continue
                
                references.append(reference)
                predictions.append(generated)
            except Exception as e:
                logger.debug(f"Error preparing semantic similarity: {e}")
                continue
        
        if references and predictions:
            try:
                # Compute embeddings
                ref_embeddings = st_model.encode(references, convert_to_tensor=True)
                pred_embeddings = st_model.encode(predictions, convert_to_tensor=True)
                
                # Compute cosine similarity
                from torch.nn.functional import cosine_similarity
                similarities = []
                for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings):
                    sim = cosine_similarity(ref_emb.unsqueeze(0), pred_emb.unsqueeze(0))
                    similarities.append(sim.item())
                
                metrics["eval_semantic_similarity"] = float(np.mean(similarities))
            except Exception as e:
                logger.warning(f"Could not compute semantic similarity: {e}")
        
        return metrics
    
    def _compute_codebleu_metrics(
        self,
        model,
        tokenizer,
        dataset,
        predictions: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, float]:
        """Compute CodeBLEU for code generation tasks."""
        metrics = {}
        try:
            from codebleu import calc_codebleu
        except ImportError:
            logger.warning("codebleu not available. Install with: pip install codebleu")
            return metrics
        
        references = []
        predictions = []
        
        for idx, example in enumerate(dataset):
            try:
                instruction, reference = self._extract_instruction_reference(example)
                if not instruction or not reference:
                    continue
                
                # Check if it's code-related
                if "code" not in instruction.lower() and "function" not in instruction.lower():
                    continue
                
                generated = None
                if predictions is not None and idx < len(predictions):
                    generated = predictions[idx]
                if not generated:
                    generated = self._generate_response(model, tokenizer, instruction)
                if not generated:
                    continue
                
                references.append([reference])  # CodeBLEU expects list of references
                predictions.append(generated)
            except Exception as e:
                logger.debug(f"Error preparing CodeBLEU: {e}")
                continue
        
        if references and predictions:
            try:
                codebleu_score = calc_codebleu(references, predictions, lang="python")
                metrics["eval_codebleu"] = float(codebleu_score)
            except Exception as e:
                logger.warning(f"Could not compute CodeBLEU: {e}")
        
        return metrics
    
    def _extract_instruction_reference(self, example: Dict[str, Any]) -> tuple:
        """Extract instruction and reference from example."""
        # Try direct instruction/response keys first
        if "instruction" in example and "response" in example:
            return example["instruction"], example["response"]
        
        # Try Q&A format (question/answers) - common in QA datasets
        if "question" in example:
            question = example["question"]
            answer = example.get("answers", example.get("answer", ""))
            # Handle list of answers (take first)
            if isinstance(answer, list):
                answer = answer[0] if len(answer) > 0 else ""
            elif not isinstance(answer, str):
                answer = str(answer) if answer else ""
            
            if question and answer:
                # Format as instruction/response
                instruction = f"Question: {question}\n\nAnswer:"
                return instruction, answer
        
        # Try query/response format
        if "query" in example:
            query = example["query"]
            response = example.get("response", example.get("answer", example.get("answers", "")))
            if isinstance(response, list):
                response = response[0] if len(response) > 0 else ""
            elif not isinstance(response, str):
                response = str(response) if response else ""
            
            if query and response:
                instruction = f"Question: {query}\n\nAnswer:"
                return instruction, response
        
        # Try text field with various formats
        if "text" in example:
            text = example["text"]
            
            # Format 1: <response> tags
            if "<response>" in text and "</response>" in text:
                parts = text.split("<response>")
                instruction = parts[0].replace("<instruction>", "").replace("</instruction>", "").strip()
                reference = parts[1].replace("</response>", "").strip()
                return instruction, reference
            
            # Format 2: Unsloth chat format (<|im_start|>user...<|im_end|>\n<|im_start|>assistant...<|im_end|>)
            if "<|im_start|>" in text and "<|im_end|>" in text:
                import re
                # Extract user message (instruction)
                user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
                assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
                
                if user_match and assistant_match:
                    instruction = user_match.group(1).strip()
                    # Remove system context if present
                    if "Context:" in instruction:
                        instruction = instruction.split("Context:")[-1].strip()
                    reference = assistant_match.group(1).strip()
                    return instruction, reference
            
            # Format 3: Simple text with newline separator (instruction\nresponse)
            if "\n" in text and len(text.split("\n")) >= 2:
                lines = text.split("\n")
                # Heuristic: if text has clear instruction/response structure
                if any(keyword in text.lower() for keyword in ["instruction", "question", "prompt"]):
                    # Try to find response part
                    for i, line in enumerate(lines):
                        if i > 0 and len(line.strip()) > 10:  # Response likely starts after first line
                            instruction = "\n".join(lines[:i]).strip()
                            reference = "\n".join(lines[i:]).strip()
                            if instruction and reference:
                                return instruction, reference
        
        return None, None
    
    def _generate_response(self, model, tokenizer, instruction: str) -> Optional[str]:
        """Generate response from instruction."""
        try:
            inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated
        except Exception as e:
            logger.debug(f"Error generating response: {e}")
            return None
    
    def _compute_text_generation_metrics(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any
    ) -> Dict[str, float]:
        """Compute metrics for text generation tasks."""
        metrics = {}
        
        model.eval()
        response_lengths = []
        coherence_scores = []
        
        with torch.no_grad():
            for example in dataset:
                try:
                    # Extract prompt
                    prompt = example.get("prompt", example.get("input", example.get("text", "")))
                    if not prompt:
                        continue
                    
                    # Generate
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                    
                    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    
                    # Response length
                    response_lengths.append(len(generated.split()))
                    
                    # Coherence (simple heuristic: sentence structure)
                    sentences = generated.split('.')
                    if len(sentences) > 1:
                        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
                        coherence = min(1.0, avg_length / 20.0)
                        coherence_scores.append(coherence)
                        
                except Exception as e:
                    logger.debug(f"Error computing metrics for example: {e}")
                    continue
        
        if response_lengths:
            metrics["eval_avg_response_length"] = float(np.mean(response_lengths))
        
        if coherence_scores:
            metrics["eval_coherence"] = float(np.mean(coherence_scores))
        
        return metrics
    
    def _compute_chat_metrics(
        self,
        model: Any,
        tokenizer: AutoTokenizer,
        dataset: Any
    ) -> Dict[str, float]:
        """Compute metrics for chat completion tasks."""
        metrics = {}
        
        model.eval()
        response_lengths = []
        
        with torch.no_grad():
            for example in dataset:
                try:
                    # Extract messages
                    if "messages" in example:
                        messages = example["messages"]
                        # Format as prompt
                        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])
                    else:
                        continue
                    
                    # Generate
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                    
                    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    response_lengths.append(len(generated.split()))
                    
                except Exception as e:
                    logger.debug(f"Error computing metrics for example: {e}")
                    continue
        
        if response_lengths:
            metrics["eval_avg_response_length"] = float(np.mean(response_lengths))
        
        return metrics
    
    def _create_eval_loader(
        self,
        dataset: Any,
        tokenizer: AutoTokenizer,
        config: SFTConfig
    ) -> DataLoader:
        """Create evaluation data loader."""
        def collate_fn(batch):
            # Extract text from batch
            texts = []
            labels = []
            
            for example in batch:
                if "text" in example:
                    texts.append(example["text"])
                elif "input" in example and "output" in example:
                    # Instruction format
                    texts.append(f"{example['input']} {example['output']}")
                elif "messages" in example:
                    # Chat format
                    messages = example["messages"]
                    text = ""
                    for msg in messages:
                        text += f"{msg['role']}: {msg['content']}\n"
                    texts.append(text.strip())
                else:
                    # Fallback
                    texts.append(str(example))
                
                # Extract labels for classification
                if "label" in example:
                    labels.append(example["label"])
                elif "labels" in example:
                    labels.append(example["labels"])
                else:
                    labels.append(None)
            
            # Tokenize
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }
            
            # Add labels if available
            if any(label is not None for label in labels):
                result["labels"] = torch.tensor([label for label in labels if label is not None])
            
            return result
        
        return DataLoader(
            dataset,
            batch_size=config.train.per_device_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )


# Alias for backward compatibility
EnhancedEvaluator = SFTEvaluator
