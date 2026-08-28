"""
Reasoning Benchmarks for Process Reward Model Evaluation.

This module provides loaders for mathematical and logical reasoning benchmarks:
- AIME (American Invitational Mathematics Examination)
- MATH (Diverse mathematical problem collection)
- GPQA (Graduate-level Professional and Academic Questions)
- LiveCodeBench (Code execution benchmarks)
- GSM8K-CoT (Grade School Math with Chain-of-Thought)

Each benchmark provides:
- Questions/problems
- Reference solutions
- Step-level labels (when available)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReasoningBenchmarkData:
    """
    Container for reasoning benchmark data.

    Attributes:
        name: Benchmark name (e.g., 'AIME', 'MATH')
        questions: List of problem/question strings
        solutions: List of reference solution strings
        steps: Optional list of step-level breakdowns per solution
        step_labels: Optional binary correctness labels per step
        metadata: Additional metadata (source, difficulty, etc.)
    """

    name: str
    questions: List[str]
    solutions: List[str]
    steps: Optional[List[List[str]]] = None
    step_labels: Optional[List[List[int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of problems in benchmark."""
        return len(self.questions)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert benchmark to dictionary format.

        Returns:
            Dictionary with benchmark data
        """
        return {
            "name": self.name,
            "questions": self.questions,
            "solutions": self.solutions,
            "steps": self.steps,
            "step_labels": self.step_labels,
            "metadata": self.metadata,
            "num_samples": len(self.questions),
        }


class ReasoningBenchmark:
    """
    Loader and manager for reasoning benchmarks.

    Provides unified interface for loading various mathematical and logical
    reasoning benchmarks used to evaluate process reward models.

    Example:
        >>> benchmark = ReasoningBenchmark()
        >>> aime_data = benchmark.load_benchmark("AIME")
        >>> print(f"Loaded {len(aime_data.questions)} AIME problems")
    """

    # Supported benchmarks
    SUPPORTED_BENCHMARKS = {
        "AIME",
        "MATH",
        "GPQA",
        "LiveCodeBench",
        "GSM8K-CoT",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize reasoning benchmark loader.

        Args:
            cache_dir: Optional directory to cache downloaded benchmarks
        """
        self.cache_dir = cache_dir
        self._loaded_benchmarks: Dict[str, ReasoningBenchmarkData] = {}

    def load_benchmark(
        self,
        name: str,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load a reasoning benchmark.

        Args:
            name: Benchmark name (case-insensitive)
                  Options: 'AIME', 'MATH', 'GPQA', 'LiveCodeBench', 'GSM8K-CoT'
            split: Dataset split ('train', 'test', 'val')
            max_samples: Optional limit on number of samples to load

        Returns:
            ReasoningBenchmarkData with questions, solutions, steps, and labels

        Raises:
            ValueError: If benchmark name not supported
            RuntimeError: If benchmark cannot be loaded

        Note:
            This is a stub implementation for testing/structure.
            Actual loading would fetch from HuggingFace or original sources.
        """
        # Case-insensitive lookup to canonical name
        canonical = {s.upper(): s for s in self.SUPPORTED_BENCHMARKS}
        name = canonical.get(name.upper(), name)

        if name not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark: {name}. "
                f"Supported: {sorted(self.SUPPORTED_BENCHMARKS)}"
            )

        # Check if already loaded
        cache_key = f"{name}_{split}"
        if cache_key in self._loaded_benchmarks:
            return self._loaded_benchmarks[cache_key]

        logger.info(f"Loading benchmark: {name} ({split} split)")

        # Load appropriate benchmark
        if name == "AIME":
            data = self._load_aime(split, max_samples)
        elif name == "MATH":
            data = self._load_math(split, max_samples)
        elif name == "GPQA":
            data = self._load_gpqa(split, max_samples)
        elif name == "LiveCodeBench":
            data = self._load_livecodebench(split, max_samples)
        elif name == "GSM8K-CoT":
            data = self._load_gsm8k_cot(split, max_samples)
        else:
            raise RuntimeError(f"Failed to load benchmark: {name}")

        # Cache loaded benchmark
        self._loaded_benchmarks[cache_key] = data

        logger.info(
            f"Loaded {len(data)} {name} samples with "
            f"{len(data.steps) if data.steps else 0} step breakdowns"
        )

        return data

    def _load_aime(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load AIME (American Invitational Mathematics Examination) benchmark.

        AIME is a 15-question math competition featuring challenging algebra,
        geometry, number theory, and combinatorics problems.

        Dataset structure:
            - Questions: Competition problems
            - Solutions: Step-by-step mathematical derivations
            - Steps: Problem-solving steps with intermediate results
            - Labels: Step correctness (binary 0/1 per step)

        Args:
            split: Dataset split to load
            max_samples: Optional sample limit

        Returns:
            ReasoningBenchmarkData with AIME problems and solutions
        """
        # Stub implementation - actual loading from dataset source
        questions = [
            "Find the number of ordered pairs (a, b) of real numbers such that "
            "(a + b/2)^2 + (b - 1)^2 = 0.",
        ]

        solutions = [
            "For the sum of squares to equal 0, each squared term must be 0. "
            "So a + b/2 = 0 and b - 1 = 0. "
            "From b - 1 = 0, we get b = 1. "
            "From a + b/2 = 0, we get a = -1/2. "
            "Therefore, there is 1 ordered pair: (-1/2, 1).",
        ]

        steps = [
            [
                "Recognize that sum of non-negative terms equals 0",
                "Each term must equal 0: a + b/2 = 0 and b - 1 = 0",
                "Solve b - 1 = 0 to get b = 1",
                "Substitute b = 1 into a + b/2 = 0",
                "Solve for a to get a = -1/2",
                "Verify solution satisfies both equations",
            ],
        ]

        step_labels = [
            [1, 1, 1, 1, 1, 1],  # All steps correct
        ]

        metadata = {
            "source": "AIME competition",
            "difficulty": "hard",
            "domain": "mathematics",
            "split": split,
        }

        return ReasoningBenchmarkData(
            name="AIME",
            questions=questions[:max_samples] if max_samples else questions,
            solutions=solutions[:max_samples] if max_samples else solutions,
            steps=[s[:max_samples] if max_samples else s for s in steps],
            step_labels=[l[:max_samples] if max_samples else l for l in step_labels],
            metadata=metadata,
        )

    def _load_math(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load MATH benchmark (diverse mathematical problems).

        MATH dataset contains diverse mathematical problems from high school
        and undergraduate competitions, including algebra, counting, geometry,
        number theory, and pre-calculus.

        Dataset structure:
            - Questions: Mathematical problems of varying difficulty
            - Solutions: Detailed step-by-step solutions
            - Steps: Solution decomposed into reasoning steps
            - Labels: Step correctness labels

        Args:
            split: Dataset split to load
            max_samples: Optional sample limit

        Returns:
            ReasoningBenchmarkData with MATH problems
        """
        # Stub implementation
        questions = [
            "Solve for x: 2x + 3 = 11",
        ]

        solutions = [
            "We have the equation 2x + 3 = 11. "
            "Subtract 3 from both sides: 2x = 8. "
            "Divide both sides by 2: x = 4. "
            "Verification: 2(4) + 3 = 8 + 3 = 11. ✓",
        ]

        steps = [
            [
                "Start with 2x + 3 = 11",
                "Subtract 3 from both sides to get 2x = 8",
                "Divide by 2 to get x = 4",
                "Verify by substitution",
            ],
        ]

        step_labels = [
            [1, 1, 1, 1],
        ]

        metadata = {
            "source": "MATH dataset",
            "difficulty": "mixed",
            "domain": "mathematics",
            "split": split,
        }

        return ReasoningBenchmarkData(
            name="MATH",
            questions=questions[:max_samples] if max_samples else questions,
            solutions=solutions[:max_samples] if max_samples else solutions,
            steps=[s[:max_samples] if max_samples else s for s in steps],
            step_labels=[l[:max_samples] if max_samples else l for l in step_labels],
            metadata=metadata,
        )

    def _load_gpqa(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load GPQA (Graduate-level Professional and Academic Questions).

        GPQA contains graduate-level science questions covering biology,
        chemistry, medicine, and physics, requiring advanced reasoning.

        Dataset structure:
            - Questions: Graduate-level science questions
            - Solutions: Detailed expert explanations
            - Steps: Reasoning decomposed into steps
            - Labels: Step validity

        Args:
            split: Dataset split to load
            max_samples: Optional sample limit

        Returns:
            ReasoningBenchmarkData with GPQA questions
        """
        # Stub implementation
        questions = [
            "In the citric acid cycle, which enzyme catalyzes the "
            "conversion of isocitrate to alpha-ketoglutarate?",
        ]

        solutions = [
            "The enzyme isocitrate dehydrogenase catalyzes this conversion. "
            "This is the third step of the citric acid cycle, where isocitrate "
            "is oxidatively decarboxylated to form alpha-ketoglutarate. "
            "The reaction requires NAD+ as a cofactor.",
        ]

        steps = [
            [
                "Identify the citric acid cycle step: isocitrate -> alpha-ketoglutarate",
                "Recall the enzyme catalyzing this step: isocitrate dehydrogenase",
                "Identify cofactor requirement: NAD+",
            ],
        ]

        step_labels = [
            [1, 1, 1],
        ]

        metadata = {
            "source": "GPQA dataset",
            "difficulty": "very_hard",
            "domain": "science",
            "split": split,
        }

        return ReasoningBenchmarkData(
            name="GPQA",
            questions=questions[:max_samples] if max_samples else questions,
            solutions=solutions[:max_samples] if max_samples else solutions,
            steps=[s[:max_samples] if max_samples else s for s in steps],
            step_labels=[l[:max_samples] if max_samples else l for l in step_labels],
            metadata=metadata,
        )

    def _load_livecodebench(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load LiveCodeBench (code execution benchmarks).

        LiveCodeBench contains programming problems requiring code generation
        and execution verification, useful for evaluating reasoning in
        algorithmic domains.

        Dataset structure:
            - Questions: Programming problem descriptions
            - Solutions: Reference code implementations
            - Steps: Algorithm design steps
            - Labels: Step correctness (does algorithm work?)

        Args:
            split: Dataset split to load
            max_samples: Optional sample limit

        Returns:
            ReasoningBenchmarkData with coding problems
        """
        # Stub implementation
        questions = [
            "Write a function to find the maximum sum of any contiguous subarray.",
        ]

        solutions = [
            "def max_subarray(arr):\n"
            "    max_sum = arr[0]\n"
            "    current_sum = arr[0]\n"
            "    for i in range(1, len(arr)):\n"
            "        current_sum = max(arr[i], current_sum + arr[i])\n"
            "        max_sum = max(max_sum, current_sum)\n"
            "    return max_sum",
        ]

        steps = [
            [
                "Initialize max_sum and current_sum with first element",
                "Iterate through remaining elements",
                "Update current_sum using max(element, current+element)",
                "Update max_sum if current_sum is larger",
                "Return max_sum",
            ],
        ]

        step_labels = [
            [1, 1, 1, 1, 1],
        ]

        metadata = {
            "source": "LiveCodeBench",
            "difficulty": "medium",
            "domain": "algorithms",
            "split": split,
        }

        return ReasoningBenchmarkData(
            name="LiveCodeBench",
            questions=questions[:max_samples] if max_samples else questions,
            solutions=solutions[:max_samples] if max_samples else solutions,
            steps=[s[:max_samples] if max_samples else s for s in steps],
            step_labels=[l[:max_samples] if max_samples else l for l in step_labels],
            metadata=metadata,
        )

    def _load_gsm8k_cot(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> ReasoningBenchmarkData:
        """
        Load GSM8K-CoT (Grade School Math with Chain-of-Thought).

        GSM8K-CoT contains grade school math word problems with explicit
        chain-of-thought annotations, making it suitable for step-level
        reward modeling.

        Dataset structure:
            - Questions: Word problems
            - Solutions: Chain-of-thought reasoning with steps
            - Steps: Individual reasoning steps from CoT
            - Labels: Whether each step is valid

        Args:
            split: Dataset split to load
            max_samples: Optional sample limit

        Returns:
            ReasoningBenchmarkData with math word problems
        """
        # Stub implementation
        questions = [
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot now?",
        ]

        solutions = [
            "Initially there are 3 cars in the parking lot. "
            "2 more cars arrive, so we need to add them. "
            "3 + 2 = 5. "
            "Therefore, there are now 5 cars in the parking lot.",
        ]

        steps = [
            [
                "Identify initial number of cars: 3",
                "Identify number of arriving cars: 2",
                "Add them together: 3 + 2 = 5",
                "State final answer: 5 cars",
            ],
        ]

        step_labels = [
            [1, 1, 1, 1],
        ]

        metadata = {
            "source": "GSM8K with CoT annotations",
            "difficulty": "easy",
            "domain": "arithmetic",
            "split": split,
        }

        return ReasoningBenchmarkData(
            name="GSM8K-CoT",
            questions=questions[:max_samples] if max_samples else questions,
            solutions=solutions[:max_samples] if max_samples else solutions,
            steps=[s[:max_samples] if max_samples else s for s in steps],
            step_labels=[l[:max_samples] if max_samples else l for l in step_labels],
            metadata=metadata,
        )

    def list_benchmarks(self) -> List[str]:
        """
        List all supported benchmarks.

        Returns:
            Sorted list of benchmark names
        """
        return sorted(self.SUPPORTED_BENCHMARKS)

    def clear_cache(self) -> None:
        """Clear all loaded benchmarks from memory."""
        self._loaded_benchmarks.clear()
        logger.info("Benchmark cache cleared")
