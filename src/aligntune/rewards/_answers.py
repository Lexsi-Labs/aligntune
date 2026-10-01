"""Shared final-answer extraction and comparison for math rewards."""

import re
from typing import Optional

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def _last_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos]
    return None


def clean_answer(answer: str) -> str:
    """Strip formatting that never changes the value of an answer."""
    answer = answer.strip()
    boxed = _last_boxed(answer)
    if boxed is not None:
        answer = boxed
    answer = answer.strip().strip("$").strip()
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = answer.replace("\\$", "").replace("\\%", "").replace("\\,", "")
    answer = answer.rstrip(".").strip()
    if re.fullmatch(r"-?\$?\d[\d,]*(?:\.\d+)?%?", answer):
        answer = answer.replace(",", "").replace("$", "").replace("%", "")
    return answer


def extract_reference_answer(reference: str) -> str:
    """Reduce a reference (bare answer or full GSM8K/MATH solution) to its final answer."""
    reference = str(reference).strip()
    if "####" in reference:
        return clean_answer(reference.split("####")[-1].split("\n")[0])
    boxed = _last_boxed(reference)
    if boxed is not None:
        return clean_answer(boxed)
    return clean_answer(reference)


def extract_completion_answer(text: str) -> Optional[str]:
    """Extract the final answer from a completion, or None if none can be found."""
    if "####" in text:
        answer = clean_answer(text.split("####")[-1].split("\n")[0])
        if answer:
            return answer
    boxed = _last_boxed(text)
    if boxed:
        return clean_answer(boxed)
    patterns = [
        r"(?:final\s+)?answer\s*(?:is|=|:)\s*:?\s*([^\n]+)",
        r"therefore,?\s+.*?=\s*([^\n=]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer = clean_answer(matches[-1])
            if answer:
                return answer
    return None


def _to_number(value: str) -> Optional[float]:
    value = value.strip().replace(",", "")
    try:
        if re.fullmatch(r"-?\d+/\d+", value):
            num, den = value.split("/")
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(answer: str, reference: str) -> bool:
    """Compare two cleaned answers numerically when possible, else as normalized strings."""
    a_val, r_val = _to_number(answer), _to_number(reference)
    if a_val is not None and r_val is not None:
        return abs(a_val - r_val) < 1e-6
    return " ".join(answer.split()).lower() == " ".join(reference.split()).lower()
