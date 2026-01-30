"""
Utility functions for Bitext insurance dataset task
"""
import re
from typing import Any, Dict, List

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Install with: pip install rouge-score")

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation by removing extra whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Process results for a single document.
    
    Args:
        doc: The document dictionary containing 'response' (reference)
        results: List containing the generated text(s)
    
    Returns:
        Dictionary with metric scores for this document
    """
    # Get the generated response
    if not results or len(results) == 0:
        generated_response = ""
    else:
        generated_response = results[0] if isinstance(results, list) else results
    
    # Get the reference response
    reference_response = doc.get("response", "")
    
    # Normalize both texts
    generated_response = normalize_text(generated_response)
    reference_response = normalize_text(reference_response)
    
    # Calculate ROUGE scores
    rouge_results = {}
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        scores = scorer.score(reference_response, generated_response)
        rouge_results = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    
    # Calculate BLEU score
    bleu_results = {}
    if BLEU_AVAILABLE:
        bleu_score = sacrebleu.corpus_bleu(
            [generated_response],
            [[reference_response]],
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize="intl",
            use_effective_order=False,
        ).score
        bleu_results = {"bleu": bleu_score}
    
    # Combine all results
    return {**rouge_results, **bleu_results}