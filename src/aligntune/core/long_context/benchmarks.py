"""
Long-context evaluation benchmarks for AlignTune.

This module provides a self-contained benchmark harness for measuring how
well a fine-tuned model handles long-context tasks:

* **Needle-in-a-Haystack (NIAH)**: A synthetic retrieval task where a
  short "needle" fact is inserted at varying *depth percentages* inside
  long "haystack" contexts.  The model must retrieve the fact by answering
  a simple query.  Accuracy is reported per ``(context_length, depth_pct)``
  cell.

* **LongBench** stubs: Lightweight stand-ins for the five LongBench tasks
  most relevant to instruction-following SFT – ``2wikimqa``, ``hotpotqa``,
  ``gov_report``, ``qasper``, and ``triviaqa``.  The stubs return small
  in-memory samples so the evaluation loop can be smoke-tested without
  network access.  Replace the stub bodies with actual HuggingFace dataset
  loads once internet access is available.

Usage example::

    from aligntune.core.long_context.benchmarks import LongContextBenchmark

    bench = LongContextBenchmark(seed=42)

    # Needle-in-a-Haystack
    results = bench.needle_in_haystack(
        model_adapter=my_model_adapter,
        context_lengths=[4096, 8192, 32768],
        depth_pcts=[0.1, 0.5, 0.9],
    )
    # results is a dict: {(ctx_len, depth_pct): accuracy_float}

    # LongBench stub
    samples = bench.load_longbench_task("hotpotqa")
    # samples is a list of dicts with "context", "question", "answer" keys.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_NEEDLE_TEMPLATE = (
    "The secret code for the {topic} is: {code}."
)
_QUERY_TEMPLATE = "What is the secret code for the {topic}?"
_HAYSTACK_SENTENCE = (
    "The researchers found that scaling language models improves performance "
    "across a wide range of natural language processing tasks. "
    "Further experiments confirmed that instruction tuning further boosts "
    "alignment quality and generalisation. "
)
_LONGBENCH_TASK_NAMES = frozenset(
    {"2wikimqa", "hotpotqa", "gov_report", "qasper", "triviaqa"}
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class NIAHResult:
    """Structured result for a single Needle-in-a-Haystack evaluation cell.

    Attributes
    ----------
    context_length:
        Approximate number of tokens in the haystack context used.
    depth_pct:
        Fractional depth (0–1) at which the needle was inserted.
    num_trials:
        Number of independent trials run for this cell.
    num_correct:
        Number of trials where the model retrieved the needle correctly.
    accuracy:
        ``num_correct / num_trials`` as a float in ``[0, 1]``.
    """

    context_length: int
    depth_pct: float
    num_trials: int
    num_correct: int
    accuracy: float = field(init=False)

    def __post_init__(self) -> None:
        if self.num_trials <= 0:
            raise ValueError(f"num_trials must be positive, got {self.num_trials}")
        self.accuracy = self.num_correct / self.num_trials

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NIAHResult(ctx={self.context_length}, depth={self.depth_pct:.1%}, "
            f"acc={self.accuracy:.3f} [{self.num_correct}/{self.num_trials}])"
        )


# ---------------------------------------------------------------------------
# Primary class
# ---------------------------------------------------------------------------


class LongContextBenchmark:
    """Benchmark harness for needle-in-a-haystack and LongBench evaluations.

    Parameters
    ----------
    seed:
        Random seed used for reproducible needle/topic generation.
        Default: 0.
    num_trials:
        Number of independent trials per ``(context_length, depth_pct)``
        cell in :py:meth:`needle_in_haystack`.  More trials reduce variance
        but increase runtime.  Default: 3.
    words_per_token_ratio:
        Approximate number of words per token used when constructing
        haystack strings of a target token length.  English prose is roughly
        0.75 words/token (≈ 1.33 tokens/word).  Default: 0.75.
    """

    def __init__(
        self,
        seed: int = 0,
        num_trials: int = 3,
        words_per_token_ratio: float = 0.75,
    ) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.words_per_token_ratio = words_per_token_ratio
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Needle-in-a-Haystack
    # ------------------------------------------------------------------

    def needle_in_haystack(
        self,
        model_adapter: Any,
        context_lengths: Optional[List[int]] = None,
        depth_pcts: Optional[List[float]] = None,
        num_trials: Optional[int] = None,
    ) -> Dict[Tuple[int, float], float]:
        """Evaluate retrieval accuracy across context lengths and depths.

        For each ``(context_length, depth_pct)`` pair the method:

        1. Generates a haystack of approximately ``context_length`` words.
        2. Inserts a unique "needle" sentence at position
           ``int(depth_pct * len(haystack_words))``.
        3. Queries the model via ``model_adapter`` to retrieve the needle.
        4. Compares the model's answer to the ground-truth code.
        5. Repeats ``num_trials`` times and reports accuracy.

        The *model_adapter* must be callable with signature::

            answer: str = model_adapter(context: str, query: str)

        where ``context`` is the full haystack+needle string and ``query``
        is the retrieval question.

        Parameters
        ----------
        model_adapter:
            Callable ``(context: str, query: str) -> str``.
            Typically a thin wrapper around a ``transformers`` pipeline or
            any object with a ``generate`` method.
        context_lengths:
            List of approximate context lengths (in tokens) to evaluate.
            Default: ``[4096, 8192, 32768]``.
        depth_pcts:
            List of fractional depths ``(0, 1]`` at which to insert the
            needle.  Default: ``[0.1, 0.5, 0.9]``.
        num_trials:
            Override the instance-level ``num_trials``.  Defaults to
            ``self.num_trials``.

        Returns
        -------
        Dict[Tuple[int, float], float]
            Keys are ``(context_length, depth_pct)`` tuples; values are
            retrieval accuracy floats in ``[0, 1]``.

        Raises
        ------
        TypeError
            If ``model_adapter`` is not callable.
        ValueError
            If ``depth_pcts`` contains values outside ``(0, 1]``.

        Examples
        --------
        >>> def dummy_adapter(context, query):
        ...     # trivial stub – always fails; replace with real model
        ...     return "0000"
        >>> bench = LongContextBenchmark(seed=1)
        >>> results = bench.needle_in_haystack(dummy_adapter,
        ...     context_lengths=[512], depth_pcts=[0.5])
        >>> isinstance(results, dict)
        True
        """
        if not callable(model_adapter):
            raise TypeError(
                f"model_adapter must be callable, got {type(model_adapter).__name__}"
            )

        context_lengths = context_lengths or [4096, 8192, 32_768]
        depth_pcts = depth_pcts or [0.1, 0.5, 0.9]
        n_trials = num_trials if num_trials is not None else self.num_trials

        for d in depth_pcts:
            if not (0.0 < d <= 1.0):
                raise ValueError(
                    f"All depth_pcts must be in (0, 1]; got {d}"
                )

        results: Dict[Tuple[int, float], float] = {}

        for ctx_len in context_lengths:
            for depth in depth_pcts:
                niah_result = self._run_niah_cell(
                    model_adapter=model_adapter,
                    context_length=ctx_len,
                    depth_pct=depth,
                    num_trials=n_trials,
                )
                results[(ctx_len, depth)] = niah_result.accuracy
                logger.info(
                    "NIAH cell ctx=%d depth=%.1f%% -> accuracy=%.3f (%d/%d)",
                    ctx_len,
                    depth * 100,
                    niah_result.accuracy,
                    niah_result.num_correct,
                    n_trials,
                )

        return results

    def _run_niah_cell(
        self,
        model_adapter: Callable[[str, str], str],
        context_length: int,
        depth_pct: float,
        num_trials: int,
    ) -> NIAHResult:
        """Run *num_trials* NIAH trials for one ``(context_length, depth_pct)`` cell.

        Parameters
        ----------
        model_adapter:
            Callable accepting ``(context, query)`` and returning a string
            answer.
        context_length:
            Approximate haystack size in tokens.
        depth_pct:
            Fractional insertion depth.
        num_trials:
            Number of independent trials.

        Returns
        -------
        NIAHResult
            Aggregated result for the cell.
        """
        num_correct = 0
        for _ in range(num_trials):
            context, query, expected_code = self._build_niah_sample(
                context_length=context_length,
                depth_pct=depth_pct,
            )
            try:
                answer = model_adapter(context, query)
                if self._extract_code(answer) == expected_code:
                    num_correct += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("model_adapter raised an exception during NIAH: %s", exc)

        return NIAHResult(
            context_length=context_length,
            depth_pct=depth_pct,
            num_trials=num_trials,
            num_correct=num_correct,
        )

    def _build_niah_sample(
        self,
        context_length: int,
        depth_pct: float,
    ) -> Tuple[str, str, str]:
        """Build a single (context, query, answer) NIAH sample.

        Parameters
        ----------
        context_length:
            Target context size in approximate tokens.
        depth_pct:
            Fractional depth at which to insert the needle.

        Returns
        -------
        Tuple[str, str, str]
            ``(context_string, query_string, ground_truth_code)``
        """
        # Generate a unique topic and secret code for this trial
        topic = self._random_topic()
        code = self._random_code()

        needle = _NEEDLE_TEMPLATE.format(topic=topic, code=code)
        query = _QUERY_TEMPLATE.format(topic=topic)

        # Build haystack: approximate token count using word count proxy
        target_words = int(context_length / self.words_per_token_ratio)
        haystack_words = self._build_haystack_words(target_words)

        # Insert needle at the target depth
        insert_pos = max(0, int(depth_pct * len(haystack_words)))
        needle_words = needle.split()
        haystack_words = (
            haystack_words[:insert_pos] + needle_words + haystack_words[insert_pos:]
        )

        context = " ".join(haystack_words)
        return context, query, code

    # ------------------------------------------------------------------
    # LongBench stubs
    # ------------------------------------------------------------------

    def load_longbench_task(
        self, task_name: str
    ) -> List[Dict[str, str]]:
        """Return stub samples for a LongBench task.

        These are *in-memory stubs* intended for smoke-testing the
        evaluation pipeline without internet access.  Each returned dict
        contains the keys ``"context"``, ``"question"``, and ``"answer"``.

        Replace the stub bodies with actual HuggingFace ``load_dataset``
        calls once network access is available::

            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task_name, split="test")

        Parameters
        ----------
        task_name:
            One of ``"2wikimqa"``, ``"hotpotqa"``, ``"gov_report"``,
            ``"qasper"``, or ``"triviaqa"``.

        Returns
        -------
        List[Dict[str, str]]
            A list of sample dicts.  Each dict has keys:

            * ``"context"`` – the long context passage.
            * ``"question"`` – the question to answer given the context.
            * ``"answer"`` – the ground-truth answer string.

        Raises
        ------
        ValueError
            If ``task_name`` is not one of the supported task names.

        Examples
        --------
        >>> bench = LongContextBenchmark()
        >>> samples = bench.load_longbench_task("hotpotqa")
        >>> isinstance(samples, list)
        True
        >>> all("context" in s and "question" in s and "answer" in s for s in samples)
        True
        """
        if task_name not in _LONGBENCH_TASK_NAMES:
            raise ValueError(
                f"Unknown LongBench task '{task_name}'. "
                f"Supported tasks: {sorted(_LONGBENCH_TASK_NAMES)}"
            )

        builder = {
            "2wikimqa": self._stub_2wikimqa,
            "hotpotqa": self._stub_hotpotqa,
            "gov_report": self._stub_gov_report,
            "qasper": self._stub_qasper,
            "triviaqa": self._stub_triviaqa,
        }[task_name]

        samples = builder()
        logger.info(
            "load_longbench_task('%s') -> %d stub samples", task_name, len(samples)
        )
        return samples

    # ------------------------------------------------------------------
    # Stub implementations (no downloads)
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_2wikimqa() -> List[Dict[str, str]]:
        """Stub for the 2WikiMultiHopQA task.

        Returns two-hop multi-document QA samples requiring bridging
        between a primary entity article and a related article.
        """
        return [
            {
                "context": (
                    "Article 1: Marie Curie was a Polish-French physicist and chemist "
                    "who conducted pioneering research on radioactivity. She was born "
                    "on 7 November 1867 in Warsaw, Poland.\n"
                    "Article 2: Warsaw is the capital and largest city of Poland. "
                    "It has been Poland's capital since 1596."
                ),
                "question": "In what country was the capital city where Marie Curie was born?",
                "answer": "Poland",
            },
            {
                "context": (
                    "Article 1: Alan Turing was a British mathematician and computer "
                    "scientist. He was born on 23 June 1912 in Maida Vale, London.\n"
                    "Article 2: London is the capital and largest city of England and "
                    "the United Kingdom."
                ),
                "question": "What is the capital city of the country where Alan Turing was born?",
                "answer": "London",
            },
        ]

    @staticmethod
    def _stub_hotpotqa() -> List[Dict[str, str]]:
        """Stub for the HotpotQA multi-hop task.

        Returns multi-hop QA samples that require reasoning over two
        supporting paragraphs.
        """
        return [
            {
                "context": (
                    "Paragraph 1: The Eiffel Tower is a wrought-iron lattice tower "
                    "on the Champ de Mars in Paris, France. It is 330 metres tall.\n"
                    "Paragraph 2: The Empire State Building is a 102-story Art Deco "
                    "skyscraper in Midtown Manhattan, New York City. It stands 443 "
                    "metres to the top of its antenna."
                ),
                "question": (
                    "Which building is taller: the Eiffel Tower or the Empire State Building?"
                ),
                "answer": "Empire State Building",
            },
            {
                "context": (
                    "Paragraph 1: Python is a high-level, general-purpose programming "
                    "language. Its design philosophy emphasises code readability.\n"
                    "Paragraph 2: Guido van Rossum is a Dutch programmer best known "
                    "as the creator of the Python programming language."
                ),
                "question": "What nationality is the creator of Python?",
                "answer": "Dutch",
            },
        ]

    @staticmethod
    def _stub_gov_report() -> List[Dict[str, str]]:
        """Stub for the GovReport summarisation task.

        Returns a long government-report passage with a reference summary.
        """
        long_context = (
            "EXECUTIVE SUMMARY\n"
            "This report evaluates the current state of renewable energy adoption "
            "in the United States as of fiscal year 2023. Solar and wind capacity "
            "additions reached record highs, driven by the Inflation Reduction Act "
            "incentives.\n\n"
            "SECTION 1: SOLAR ENERGY\n"
            "Utility-scale solar installations increased by 34% year-over-year, "
            "adding 28 GW of new capacity. Residential solar continued its upward "
            "trend, supported by the extended 30% investment tax credit.\n\n"
            "SECTION 2: WIND ENERGY\n"
            "Onshore wind added 12 GW of capacity while offshore wind saw its first "
            "major commercial deployments on the Atlantic coast. Supply chain "
            "challenges moderated growth below initial projections.\n\n"
            "SECTION 3: GRID INTEGRATION\n"
            "Grid operators implemented new flexibility mechanisms including "
            "demand response programs and utility-scale battery storage mandates "
            "to accommodate higher shares of variable renewable generation.\n\n"
            "CONCLUSIONS\n"
            "The United States remains on track to meet its 2030 renewable energy "
            "targets, though sustained policy support and grid modernisation "
            "investments are required to maintain momentum."
        )
        return [
            {
                "context": long_context,
                "question": "Summarise the key findings of this government report.",
                "answer": (
                    "Record solar and wind capacity additions were driven by IRA "
                    "incentives; grid integration improvements are needed to sustain "
                    "progress toward 2030 targets."
                ),
            }
        ]

    @staticmethod
    def _stub_qasper() -> List[Dict[str, str]]:
        """Stub for the QASPER scientific paper QA task.

        Returns samples from a synthetic academic paper passage.
        """
        paper_context = (
            "Title: Scaling Laws for Neural Language Models\n"
            "Abstract: We study empirical scaling laws for language model performance "
            "on the cross-entropy loss. The loss scales as a power-law with model size, "
            "dataset size, and the amount of compute used for training.\n\n"
            "Introduction: The performance of language models depends on three primary "
            "factors: the number of parameters N, the size of the training dataset D, "
            "and the total compute budget C = 6ND (for a single epoch).\n\n"
            "Methods: We trained over 70 language model variants ranging from "
            "768 parameters to 1.5 billion parameters on datasets of 22 million to "
            "23 billion tokens.\n\n"
            "Results: We find that model performance follows a power law in each of "
            "N, D, and C individually. Optimal compute-efficient training involves "
            "simultaneously scaling model size and data."
        )
        return [
            {
                "context": paper_context,
                "question": "What three factors determine language model performance?",
                "answer": "Number of parameters, dataset size, and compute budget.",
            },
            {
                "context": paper_context,
                "question": "What is the formula for compute budget used in this paper?",
                "answer": "C = 6ND",
            },
        ]

    @staticmethod
    def _stub_triviaqa() -> List[Dict[str, str]]:
        """Stub for the TriviaQA closed-book (long-context) task.

        Returns evidence-passage + question samples from the TriviaQA
        unfiltered setting.
        """
        return [
            {
                "context": (
                    "The Olympic Games are the world's foremost sports competition with "
                    "more than 200 nations participating. The Games are currently held "
                    "every four years, with the Summer and Winter Games alternating "
                    "occurring every four years but two years apart. The first modern "
                    "Olympics were held in 1896 in Athens, Greece."
                ),
                "question": "In which city were the first modern Olympic Games held?",
                "answer": "Athens",
            },
            {
                "context": (
                    "The Great Wall of China is a series of fortifications made of stone, "
                    "brick, tamped earth, wood, and other materials. Built along an east-to-"
                    "west line across the historical northern borders of China, the wall was "
                    "built to protect against raids and invasions of various nomadic groups. "
                    "Its official length is 21,196 kilometres."
                ),
                "question": "What is the official length of the Great Wall of China in kilometres?",
                "answer": "21,196 kilometres",
            },
        ]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _random_topic(self) -> str:
        """Generate a random two-word topic string for needle generation.

        Returns
        -------
        str
            A randomly generated topic such as ``"amber enigma"``.
        """
        adjectives = [
            "amber", "cobalt", "crimson", "emerald", "golden", "ivory",
            "jade", "midnight", "obsidian", "sapphire", "scarlet", "silver",
            "violet", "azure", "bronze", "chartreuse", "coral", "lavender",
        ]
        nouns = [
            "anvil", "beacon", "cipher", "compass", "delta", "eagle",
            "falcon", "harbour", "lantern", "nexus", "omega", "oracle",
            "phantom", "prism", "quartz", "raven", "sphinx", "zenith",
        ]
        adj = self._rng.choice(adjectives)
        noun = self._rng.choice(nouns)
        return f"{adj} {noun}"

    def _random_code(self, length: int = 8) -> str:
        """Generate a random alphanumeric secret code.

        Parameters
        ----------
        length:
            Number of characters in the code.  Default: 8.

        Returns
        -------
        str
            Upper-case alphanumeric string of *length* characters.
        """
        alphabet = string.ascii_uppercase + string.digits
        return "".join(self._rng.choice(alphabet) for _ in range(length))

    def _build_haystack_words(self, target_words: int) -> List[str]:
        """Build a list of haystack words of approximately *target_words* length.

        Repeats a fixed prose sentence until the target word count is reached.

        Parameters
        ----------
        target_words:
            Desired number of words in the haystack.

        Returns
        -------
        List[str]
            List of individual words.
        """
        sentence_words = _HAYSTACK_SENTENCE.split()
        repetitions = max(1, (target_words // len(sentence_words)) + 1)
        words = sentence_words * repetitions
        return words[:target_words]

    @staticmethod
    def _extract_code(answer: str) -> str:
        """Extract the secret code from a model response.

        Strips whitespace, punctuation, and lowercases the answer before
        attempting to extract an 8-character alphanumeric token.  Falls
        back to returning the raw uppercased answer if no 8-char token is
        found.

        Parameters
        ----------
        answer:
            Raw text returned by the model adapter.

        Returns
        -------
        str
            Best-effort extracted code, upper-cased and stripped.
        """
        import re

        if not answer:
            return ""
        # Try to find an 8-character alphanumeric token anywhere in the answer
        match = re.search(r"\b([A-Z0-9]{8})\b", answer.upper())
        if match:
            return match.group(1)
        # Fallback: strip non-alphanumeric and uppercase
        cleaned = re.sub(r"[^A-Z0-9]", "", answer.upper())
        return cleaned[:8] if cleaned else ""

    def summary_table(
        self,
        results: Dict[Tuple[int, float], float],
    ) -> str:
        """Format NIAH results as a human-readable ASCII table.

        Parameters
        ----------
        results:
            Dict returned by :py:meth:`needle_in_haystack`.

        Returns
        -------
        str
            Multi-line string suitable for logging or printing.
        """
        if not results:
            return "(no results)"

        ctx_lengths = sorted({k[0] for k in results})
        depths = sorted({k[1] for k in results})

        header = "Context\\Depth  " + "  ".join(f"{d:.0%}" for d in depths)
        rows = [header, "-" * len(header)]

        for ctx in ctx_lengths:
            row_vals = []
            for d in depths:
                acc = results.get((ctx, d), float("nan"))
                row_vals.append(f"{acc:.2f}" if acc == acc else " n/a")
            rows.append(f"{ctx:>14d}  " + "  ".join(f"{v:>5}" for v in row_vals))

        return "\n".join(rows)
