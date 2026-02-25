"""
Bitext retail banking chatbot evaluation.
Dataset: bitext/Bitext-retail-banking-llm-chatbot-training-dataset
Columns: instruction, response (instruction-following format).
Metrics: BLEU, ROUGE-1/2/L, chrF++, BERTScore (per-doc, aggregated with nanmean).
"""

import numpy as np

try:
    import evaluate
    _bleu = evaluate.load("bleu")
    _rouge = evaluate.load("rouge")
    _bertscore = evaluate.load("bertscore")
    try:
        _chrf = evaluate.load("chrf")
        _CHRF_AVAILABLE = True
    except Exception:
        _chrf = None
        _CHRF_AVAILABLE = False
except (ModuleNotFoundError, ImportError) as e:
    raise ModuleNotFoundError(
        "Install metrics: pip install evaluate bert-score rouge_score>=0.1.2"
    ) from e


def list_fewshot_samples():
    """Fixed few-shot exemplars for reproducible 2-shot eval (same schema as dataset)."""
    return [
        {
            "instruction": "How do I activate my credit card?",
            "response": "I'd be happy to help you activate your credit card. You can activate it by calling the number on the sticker on your card, visiting our website and logging into your account, or using our mobile app. Do you have your card handy?",
            "is_fewshot": True,
        },
        {
            "instruction": "What is the interest rate on my savings account?",
            "response": "To find your current savings account interest rate, please log in to online banking or our mobile app, or call the number on the back of your debit card. Rates can vary by account type and balance tier.",
            "is_fewshot": True,
        },
    ]


def doc_to_text(doc) -> str:
    """Prompt: ChatML prefix + instruction (no trailing response)."""
    return f"<|im_start|>user\n{doc['instruction']}<|im_end|>\n<|im_start|>assistant\n"


def doc_to_target(doc) -> str:
    """Reference response."""
    return doc["response"]


def process_results(doc, results):
    """Per-doc metrics: BLEU, ROUGE-1/2/L, chrF++, BERTScore. Short outputs → nan."""
    pred = (results[0] or "").strip()
    ref = doc_to_target(doc)
    if len(ref) < 3 or len(pred) < 3:
        out = {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bert_score": np.nan,
        }
        if _CHRF_AVAILABLE:
            out["chrf"] = np.nan
        return out

    out = {}
    try:
        bleu_res = _bleu.compute(predictions=[pred], references=[[ref]])
        out["bleu"] = float(bleu_res.get("bleu", np.nan))
        if out["bleu"] == 0.0:
            out["bleu"] = 1e-5  # avoid stderr issues
    except Exception:
        out["bleu"] = np.nan

    try:
        rouge_res = _rouge.compute(predictions=[pred], references=[ref])
        out["rouge1"] = float(rouge_res.get("rouge1", np.nan))
        out["rouge2"] = float(rouge_res.get("rouge2", np.nan))
        out["rougeL"] = float(rouge_res.get("rougeL", np.nan))
    except Exception:
        out["rouge1"] = out["rouge2"] = out["rougeL"] = np.nan

    if _CHRF_AVAILABLE and _chrf is not None:
        try:
            chrf_res = _chrf.compute(predictions=[pred], references=[[ref]])
            out["chrf"] = float(chrf_res.get("score", np.nan))
        except Exception:
            out["chrf"] = np.nan
    else:
        out["chrf"] = np.nan

    try:
        bs_res = _bertscore.compute(
            predictions=[pred], references=[ref], lang="en"
        )
        out["bert_score"] = float(np.mean(bs_res.get("f1", [np.nan])))
    except Exception:
        out["bert_score"] = np.nan

    return out
