"""
Code evaluation metrics using the legacy SafeCodeExecutor.
"""
from typing import List, Dict, Any
import numpy as np
import logging
from .base import Metric
from ..safe_executor import SafeCodeExecutor

logger = logging.getLogger(__name__)

class PassAtKMetric(Metric):
    """
    Computes Pass@K, Avg@K, Maj@K, and execution statistics for code generation.
    Uses the robust SafeCodeExecutor from the legacy system.
    """
    
    def __init__(self, k_list: List[int] = None, timeout: int = 5):
        super().__init__("pass_at_k")
        self.k_list = k_list or [1]
        self.timeout = timeout
        self.executor = SafeCodeExecutor(timeout=timeout, language="python")
        logger.info(f"PassAtKMetric initialized with k_list={self.k_list}, timeout={timeout}")
    
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        """
        Args:
            predictions: List of generated code. 
                         If k=1, list of strings. 
                         If k>1, list of lists of strings (candidates per prompt).
            references: List of test cases corresponding to each prediction.
        """
        logger.info(f"PassAtK compute - Predictions: {len(predictions)}, References: {len(references)}")
        
        if not predictions:
            return {f"pass@{k}": 0.0 for k in self.k_list}
        
        # Detailed structure check
        sample_pred = predictions[0]
        sample_ref = references[0] if references else None
        
        logger.info(f"Prediction[0] type: {type(sample_pred)}")
        if isinstance(sample_pred, list):
            logger.info(f"Prediction[0] has {len(sample_pred)} candidates")
            logger.info(f"Candidate[0] preview: {sample_pred[0][:100]}...")
        else:
            logger.info(f"Prediction[0] preview: {str(sample_pred)[:100]}...")
        
        logger.info(f"Reference[0] type: {type(sample_ref)}")
        if isinstance(sample_ref, list):
            logger.info(f"Reference[0] has {len(sample_ref)} test cases")
            if sample_ref:
                logger.info(f"Test case[0]: {sample_ref[0]}")
        else:
            logger.info(f"Reference[0] preview: {str(sample_ref)[:200]}...")
        
        # Initialize Execution Stats
        execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "syntax_errors": 0,
            "runtime_errors": 0,
            "timeouts": 0,
            "total_execution_time": 0.0
        }

        # Calculate pass@k
        total_problems = len(predictions)
        correct_counts = [] # Number of correct candidates per problem
        ns = [] # Total candidates per problem
        
        for i, (preds_for_problem, test_cases) in enumerate(zip(predictions, references)):
            # Ensure preds_for_problem is a list
            if isinstance(preds_for_problem, str):
                preds_for_problem = [preds_for_problem]
            
            n = len(preds_for_problem)
            c = 0
            
            # Detailed debugging for first sample
            debug_sample = (i == 0)
            
            if debug_sample:
                logger.info(f"\n{'='*60}")
                logger.info(f"DEBUGGING SAMPLE 0")
                logger.info(f"{'='*60}")
                logger.info(f"Number of candidates: {n}")
                logger.info(f"Test cases type: {type(test_cases)}")
                if isinstance(test_cases, list):
                    logger.info(f"Number of test cases: {len(test_cases)}")
                    if test_cases:
                        logger.info(f"First test case: {test_cases[0]}")
            
            # Execute each candidate
            for j, code in enumerate(preds_for_problem):
                # Clean code
                clean_code = self.executor.extract_code(code)
                
                if debug_sample and j == 0:
                    logger.info(f"\n--- Candidate 0 ---")
                    logger.info(f"RAW CODE (first 300 chars):\n{code[:300]}")
                    logger.info(f"\nCLEAN CODE (first 300 chars):\n{clean_code[:300]}")
                    logger.info(f"\nTest cases being used: {test_cases[:2] if isinstance(test_cases, list) else str(test_cases)[:200]}")
                
                # Execute
                exec_result = self.executor.execute(clean_code, test_cases=test_cases)
                
                if exec_result.success and exec_result.test_passed:
                    c += 1
                    if debug_sample and j == 0:
                        logger.info(f"✓ PASSED all tests!")
                elif debug_sample and j == 0:
                    logger.info(f"\n✗ FAILED:")
                    logger.info(f"  Success: {exec_result.success}")
                    logger.info(f"  Test Passed: {exec_result.test_passed}")
                    logger.info(f"  Error: {exec_result.error}")
                    logger.info(f"  Output: {exec_result.output[:200] if exec_result.output else 'None'}")
                    
                    if exec_result.test_results:
                        logger.info(f"  Test Results ({len(exec_result.test_results)} tests):")
                        for idx, test in enumerate(exec_result.test_results[:3]):
                            logger.info(f"    Test {idx}: {test}")
                # --- Gather Stats ---
                execution_stats["total_executions"] += 1
                exec_time = getattr(exec_result, 'execution_time', 0.0)
                execution_stats["total_execution_time"] += exec_time

                if exec_result.success:
                    execution_stats["successful_executions"] += 1
                    if exec_result.test_passed:
                        c += 1
                else:
                    error_msg = getattr(exec_result, 'error', 'Unknown error')
                    if "Syntax error" in error_msg:
                        execution_stats["syntax_errors"] += 1
                    elif "timeout" in error_msg.lower() or "time out" in error_msg.lower():
                        execution_stats["timeouts"] += 1
                    else:
                        execution_stats["runtime_errors"] += 1            
            
            correct_counts.append(c)
            ns.append(n)
            
            if i < 3:
                logger.info(f"Sample {i} summary: {c}/{n} candidates passed")
        
        # Calculate metric scores
        results = {}
        
        # 1. Pass@K, Avg@K, Maj@K
        for k in self.k_list:
            pass_at_k_scores = []
            avg_at_k_scores = []
            maj_at_k_scores = []
            
            for c, n in zip(correct_counts, ns):
                if n >= k:
                    # Pass@k (Unbiased Estimator)
                    pass_score = self._estimate_pass_at_k(n, c, k)
                    pass_at_k_scores.append(pass_score)
                    
                    # Avg@k (Standard Average Pass Rate)
                    # This calculates the raw accuracy: correct / total_generated
                    # Note: We use n (total samples generated) as denominator, not k
                    avg_score = c / n if n > 0 else 0.0
                    avg_at_k_scores.append(avg_score)
                    
                    # Maj@k (Majority Voting)
                    # If > 50% of the k samples passed
                    # Standard Maj@k checks if >= ceil(k/2 + 0.5) passed
                    # Strictly > k/2
                    maj_score = 1.0 if min(c, k) > (k / 2) else 0.0
                    maj_at_k_scores.append(maj_score)
                else:
                    # Fewer samples than k, usually default to 0.0
                    pass_at_k_scores.append(0.0)
                    avg_at_k_scores.append(0.0)
                    maj_at_k_scores.append(0.0)
            
            # Aggregate
            results[f"pass@{k}"] = float(np.mean(pass_at_k_scores)) if pass_at_k_scores else 0.0
            results[f"avg@{k}"] = float(np.mean(avg_at_k_scores)) if avg_at_k_scores else 0.0
            
            # Only compute Maj@k if k > 1 (Maj@1 is just Pass@1)
            if k > 1:
                results[f"maj@{k}"] = float(np.mean(maj_at_k_scores)) if maj_at_k_scores else 0.0
            
            logger.info(f"pass@{k}: {results[f'pass@{k}']:.4f}")

        # 2. Add Execution Statistics to results
        if execution_stats["total_executions"] > 0:
            results["execution_success_rate"] = execution_stats["successful_executions"] / execution_stats["total_executions"]
            results["avg_execution_time"] = execution_stats["total_execution_time"] / execution_stats["total_executions"]
        else:
            results["execution_success_rate"] = 0.0
            results["avg_execution_time"] = 0.0
            
        results["syntax_errors"] = execution_stats["syntax_errors"]
        results["runtime_errors"] = execution_stats["runtime_errors"]
        results["timeouts"] = execution_stats["timeouts"]
        
        # Summary logging
        total_correct = sum(correct_counts)
        total_attempts = sum(ns)
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL SUMMARY:")
        logger.info(f"  Total problems: {len(correct_counts)}")
        logger.info(f"  Total attempts: {total_attempts}")
        logger.info(f"  Total correct: {total_correct}")
        logger.info(f"  Syntax Errors: {results['syntax_errors']}")
        logger.info(f"  Avg Exec Time: {results['avg_execution_time']:.4f}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _estimate_pass_at_k(self, n, c, k):
        """
        Estimates pass@k using the unbiased estimator.
        score = 1 - C(n-c, k) / C(n, k)
        """
        if n < k:
            return 0.0
        if c == n:
            return 1.0
        if c == 0:
            return 0.0
        
        def combination(n, k):
            """Calculate binomial coefficient C(n, k)"""
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            if k > n // 2:
                k = n - k
            
            result = 1
            for i in range(k):
                result = result * (n - i) // (i + 1)
            return result
        
        total_combos = combination(n, k)
        fail_combos = combination(n - c, k)
        
        if total_combos == 0:
            return 0.0
        
        score = 1.0 - (fail_combos / total_combos)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]