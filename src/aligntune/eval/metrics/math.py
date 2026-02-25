"""
Math evaluation metrics using regex extraction logic from legacy eval2.py.
"""

import re
from typing import List, Dict, Any
import numpy as np
from .base import Metric

class MathAccuracyMetric(Metric):
    """
    Computes accuracy for Math tasks (GSM8K, MATH) using robust extraction.
    """
    
    def __init__(self):
        super().__init__("math_accuracy")

    def _extract_answer(self, text: str) -> str:
        r"""
        Extract numerical answer from various model output formats.
        
        Supports:
        1. GSM8K format: #### number
        2. <SOLUTION> tags: <SOLUTION>number</SOLUTION>
        3. Natural language conclusions: "Therefore, X makes $Y"
        4. LaTeX notation: \$X or \(X\)
        5. Equation patterns: = $X
        6. Fallback: last number in text
        
        Args:
            text: Model output or reference text
            
        Returns:
            Extracted numerical answer as string
        """
        text = text.strip()
        
        # 1. GSM8K format (highest priority): #### number
        m = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
        if m:
            return m.group(1).replace(",", "")
        
        # 2. <SOLUTION> tags (Llama format)
        m = re.search(r"<SOLUTION>\s*(.+?)(?:</SOLUTION>|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            solution_text = m.group(1).strip()
            # Extract only numbers from solution text
            nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", solution_text)
            if nums:
                return nums[-1].replace(",", "")
        
        # 3. Boxed format (common in MATH dataset)
        if "\\boxed" in text:
            m = re.search(r"\\boxed\{([^}]+)\}", text)
            if m:
                boxed_content = m.group(1)
                nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", boxed_content)
                if nums:
                    return nums[-1].replace(",", "")
        
        # 4. Natural language conclusions (Qwen-style)
        conclusion_patterns = [
            # "Therefore, Janet makes $18"
            r"(?:Therefore|Thus|So|Hence),?\s+(?:[\w\s]+\s+)?(?:makes?|profit|total|is|equals?)\s+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            # "answer is 18" or "total: 18"
            r"(?:answer|result|profit|total)(?:\s+is)?[:\s]+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            # "makes $18 every day"
            r"(?:makes?|profit|earns?)\s+\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:every|per|each|on)",
            # "it takes 3 bolts"
            r"(?:takes?|needs?|requires?)\s+(?:a\s+total\s+of\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:bolts?|units?|items?)",
        ]
        
        for pattern in conclusion_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).replace(",", "")
        
        # 5. LaTeX-style answers
        latex_patterns = [
            r"\\\$\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # \$18
            r"=\s*\\\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\\?\)",  # = \$18\)
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].replace(",", "")
        
        # 6. Equation patterns
        equation_patterns = [
            r"=\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\n|$)",  # = $18
            r"Profit\s*=\s*\$?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # Profit = $18
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].replace(",", "")
        
        # 7. Last sentence fallback (for natural language models)
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            last_sentence = sentences[-1]
            nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", last_sentence)
            if nums:
                return nums[-1].replace(",", "")
        
        # 8. Final fallback: last number anywhere in text
        nums = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if nums:
            return nums[-1].replace(",", "")
        
        return ""
    def _normalize_number(self, num_str: str) -> float:
        try:
            return float(num_str.replace(",", ""))
        except:
            return float('nan')

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_ans = self._extract_answer(pred)
            ref_ans = self._extract_answer(str(ref)) # Ensure ref is string
            # print(pred_ans)
            # print(ref_ans)
            # Numeric comparison
            try:
                if abs(self._normalize_number(pred_ans) - self._normalize_number(ref_ans)) < 1e-5:
                    correct += 1
            except:
                # String comparison fallback
                if pred_ans.strip() == ref_ans.strip():
                    correct += 1
            total += 1

        return {"math_accuracy": correct / total if total > 0 else 0.0}