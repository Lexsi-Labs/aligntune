"""
Common reward handler for RL trainers supporting multiple reward functions.

Supports:
- Single reward function
- Multiple reward functions (averaged)
- Synchronous reward computation
- Asynchronous reward computation (concurrent)

Used by: GRPO, GSPO, etc.
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass
from functools import wraps
from numbers import Real
from typing import List, Callable, Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)


def extract_completion_text(completion: Any) -> str:
    """Flatten a completion to plain text regardless of dataset format.

    Conversational datasets (e.g. a chat/Instruct model's data preprocessed
    into `[{"role": "user", "content": "..."}]` form) make TRL represent a
    completion the same way: `[{"role": "assistant", "content": "..."}]`,
    rather than a plain string. aligntune's reward functions all expect
    plain text (they call `text.split()`, regex over `text`, etc.), so this
    normalizes either shape to a flat string before it ever reaches one.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


# Column names datasets commonly use for the ground-truth answer. Mirrors the
# alias list Unsloth's GRPO backend already searches (grpo.py's
# `_sync_combined_reward`) so the two backends behave consistently.
REFERENCE_ALIAS_KEYS = ["reference", "answer", "solution", "ground_truth", "response", "target"]


@dataclass(frozen=True)
class PreparedTRLRewards:
    """TRL-ready reward functions together with their native aggregation weights."""

    functions: List[Callable]
    weights: List[float]
    names: List[str]


class RewardBridge:
    """Convert AlignTune registry rewards to TRL's batch reward contract.

    Registry rewards score one generated completion at a time through
    ``compute(text, reference=None, **kwargs) -> float``.  TRL calls reward
    functions with batches of completions and every preserved dataset column.
    This bridge performs only that schema conversion.  User-provided callables
    are treated as already TRL-native and are returned unchanged.
    """

    def __init__(self, reward_specs: Optional[Sequence[Any]] = None):
        self.reward_specs = list(reward_specs or [])

    def build_trl_rewards(self) -> PreparedTRLRewards:
        """Resolve configured rewards into functions accepted by ``GRPOTrainer``.

        A string reward type is resolved through ``RewardRegistry``. Both that
        and a callable supplied in ``type``, ``params.function``, or
        ``params.reward_function`` are single-sample scorers
        (``fn(text, **kwargs) -> float``) and get wrapped identically into
        TRL's batch interface.
        """
        functions: List[Callable] = []
        weights: List[float] = []
        names: List[str] = []

        for index, spec in enumerate(self.reward_specs):
            reward_type, params, weight = self._parse_spec(spec, index)
            custom_callable = self._get_custom_callable(reward_type, params)

            if custom_callable is not None:
                custom_name = getattr(custom_callable, "__name__", f"custom_{index}")
                functions.append(self.wrap_registry_reward(custom_callable, custom_name))
                names.append(custom_name)
            elif isinstance(reward_type, str):
                reward_object = self._load_registry_reward(reward_type, params, weight)
                functions.append(self.wrap_registry_reward(reward_object, reward_type))
                names.append(reward_type)
            else:
                raise TypeError(
                    f"Reward spec {index} must provide a registry name or callable; "
                    f"got {type(reward_type).__name__}."
                )

            weights.append(weight)

        return PreparedTRLRewards(functions=functions, weights=weights, names=names)

    @staticmethod
    def wrap_registry_reward(reward_object: Any, reward_name: str = "registry_reward") -> Callable:
        """Wrap a scalar registry reward as a TRL batch reward function.

        Every batch-aligned dataset column is reduced to the row that belongs to
        the generated completion.  ``None`` is preserved because TRL uses it to
        represent rewards that do not apply to a particular sample.
        """
        scorer = getattr(reward_object, "compute", reward_object)
        if not callable(scorer):
            raise TypeError(f"Registry reward '{reward_name}' has no callable compute method.")

        @wraps(scorer)
        def trl_reward(
            prompts=None,
            completions=None,
            completion_ids=None,
            **kwargs,
        ) -> List[Optional[float]]:
            if completions is None:
                return []

            batch_size = len(completions)
            rewards: List[Optional[float]] = []

            for index, completion in enumerate(completions):
                row_kwargs = slice_batch_kwargs_for_sample(kwargs, index, batch_size)
                prompt = RewardBridge._sample_value(prompts, index, batch_size)
                if prompt is not None and "prompt" not in row_kwargs:
                    row_kwargs["prompt"] = prompt

                try:
                    score = call_reward_safely(scorer, completion, **row_kwargs)
                except Exception as exc:
                    raise RuntimeError(
                        f"Registry reward '{reward_name}' failed for completion {index}: {exc}"
                    ) from exc

                if score is None:
                    rewards.append(None)
                elif isinstance(score, Real):
                    rewards.append(float(score))
                else:
                    raise TypeError(
                        f"Registry reward '{reward_name}' returned {type(score).__name__} "
                        f"for completion {index}; expected float or None."
                    )

            return rewards

        trl_reward.__name__ = f"registry_{reward_name}"
        return trl_reward

    @staticmethod
    def _sample_value(value: Any, index: int, batch_size: int) -> Any:
        """Return the value aligned with a completion when a value is batched."""
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return value[index]
        return value

    @staticmethod
    def _get_custom_callable(reward_type: Any, params: dict) -> Optional[Callable]:
        """Return a user TRL-native callable from a reward specification."""
        if callable(reward_type):
            return reward_type
        for key in ("function", "reward_function"):
            candidate = params.get(key)
            if candidate is not None:
                if not callable(candidate):
                    raise TypeError(f"Reward params.{key} must be callable.")
                return candidate
        return None

    @staticmethod
    def _parse_spec(spec: Any, index: int) -> tuple[Any, dict, float]:
        """Normalize dict and dataclass reward specs used by the factory."""
        if isinstance(spec, dict):
            reward_type = spec.get("type")
            params = spec.get("params", {}) or {}
            weight = spec.get("weight", 1.0)
        else:
            reward_type = getattr(spec, "type", None)
            params = getattr(spec, "params", {}) or {}
            weight = getattr(spec, "weight", 1.0)

        if not isinstance(params, dict):
            raise TypeError(f"Reward spec {index} params must be a dict.")
        if reward_type is None:
            raise ValueError(f"Reward spec {index} is missing 'type'.")
        if not isinstance(weight, Real):
            raise TypeError(f"Reward spec {index} weight must be numeric.")
        return reward_type, params, float(weight)

    @staticmethod
    def _load_registry_reward(reward_name: str, params: dict, weight: float) -> Any:
        """Instantiate a named registry reward with the caller's parameters."""
        from aligntune.rewards.registry import RewardRegistry

        return RewardRegistry.get_reward_function(
            reward_name,
            {"type": reward_name, "weight": weight, "params": params},
        )


def prepare_trl_rewards(rewards: Sequence[Any]) -> PreparedTRLRewards:
    """Resolve reward specs once for every TRL-backed online trainer.

    This is the shared entry point for GRPO and Online DPO. It keeps
    registry rewards and user-supplied TRL callables in the same order as their
    configured weights.
    """
    specs = []
    for reward in rewards:
        if isinstance(reward, str):
            specs.append({"type": reward, "params": {}})
        elif callable(reward):
            specs.append({"type": "custom", "params": {"function": reward}})
        elif isinstance(reward, dict) or hasattr(reward, "type"):
            specs.append(reward)
        else:
            raise TypeError(
                "Each reward must be a registry name, a TRL-native callable, or a dict spec; "
                f"got {type(reward).__name__}."
            )
    return RewardBridge(specs).build_trl_rewards()


def resolve_trl_reward_weights(
    prepared_rewards: PreparedTRLRewards,
    explicit_weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Return validated weights in the exact order of prepared functions."""
    weights = list(explicit_weights) if explicit_weights is not None else prepared_rewards.weights
    if len(weights) != len(prepared_rewards.functions):
        raise ValueError(
            "reward_weights must contain one value for each configured reward "
            f"({len(weights)} weights for {len(prepared_rewards.functions)} rewards)."
        )
    if not all(isinstance(weight, Real) for weight in weights):
        raise TypeError("reward_weights must contain only numeric values.")
    return [float(weight) for weight in weights]


def build_trl_reward_functions(
    rewards: Sequence[Union[str, Callable, dict]],
) -> List[Callable]:
    """Return TRL-ready reward functions from registry names and user callables.

    Args:
        rewards: A mixed list such as ``["math_verifiable", length_reward]``.
            Strings are looked up in AlignTune's reward registry and wrapped.
            Callables are assumed to already follow TRL's batch contract and are
            returned unchanged. Dicts remain available for advanced per-reward
            parameters, for example ``{"type": "length", "params": {...}}``.

    Returns:
        A list suitable for ``GRPOTrainer(reward_funcs=...)``.
    """
    return prepare_trl_rewards(rewards).functions


def resolve_reward_call_kwargs(reward_func: Callable, completion: Any, **kwargs) -> dict:
    """Build a kwargs dict that can safely call `reward_func` for one completion.

    RL trainers (TRL's GRPOTrainer, SDPOTrainer, etc.) forward every extra
    dataset column as a kwarg to reward functions. Four failure modes recur
    across aligntune's reward call sites:

    1. A dataset column happens to share a name with the reward function's
       own first positional parameter (aligntune's RewardRegistry rewards all
       take `text` first) -> "got multiple values for argument 'text'".
    2. The completion is forwarded under the wrong keyword entirely (e.g.
       `completions=[completion]` where the function expects a positional
       `text` arg) -> "missing 1 required positional argument".
    3. The completion itself is a conversational message list rather than a
       string (see `extract_completion_text`) -> e.g. "'list' object has no
       attribute 'split'".
    4. The reward function declares a `reference` parameter (every built-in
       aligntune reward does), but the dataset's ground-truth column is named
       something else -- `answer`, `solution`, `ground_truth`, etc. (e.g.
       GSM8K-style datasets). Without a fallback, `reference` silently stays
       `None` and every reward that depends on it scores 0, with no error.

    This introspects the target callable and binds the (flattened)
    completion text to whatever its actual first positional parameter is
    named, resolves a `reference` alias if the function wants one and it's
    missing, then filters the remaining kwargs to what it can actually accept
    (or passes everything through if it declares **kwargs).
    """
    completion = extract_completion_text(completion)
    try:
        sig = inspect.signature(reward_func)
    except (TypeError, ValueError):
        # Can't introspect (e.g. some builtins/C extensions) - best effort.
        return {"text": completion, **kwargs}

    params = sig.parameters
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    positional = [
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name != "self"
    ]
    text_param = positional[0] if positional else "text"

    wants_reference = "reference" in params or accepts_var_keyword
    if wants_reference and "reference" not in kwargs:
        for alias in REFERENCE_ALIAS_KEYS:
            if alias in kwargs and kwargs[alias] is not None:
                kwargs = {**kwargs, "reference": kwargs[alias]}
                break

    if accepts_var_keyword:
        safe_kwargs = {k: v for k, v in kwargs.items() if k != text_param}
    else:
        safe_kwargs = {k: v for k, v in kwargs.items() if k in params and k != text_param}

    return {text_param: completion, **safe_kwargs}


def call_reward_safely(reward_func: Callable, completion: str, **kwargs) -> Any:
    """Call a reward function against a completion without crashing on kwargs mismatch.

    See `resolve_reward_call_kwargs` for the argument-binding logic.
    """
    return reward_func(**resolve_reward_call_kwargs(reward_func, completion, **kwargs))


def slice_batch_kwargs_for_sample(kwargs: dict, idx: int, batch_size: int) -> dict:
    """Slice batch-shaped kwargs down to the single value for sample `idx`.

    TRL forwards every extra dataset column (and `prompts`/`completion_ids`)
    as a list aligned to the whole `completions` batch, e.g.
    `reference=["ref for sample 0", "ref for sample 1", ...]`. Without this,
    a per-sample reward call would receive the ENTIRE batch list for every
    single completion instead of the one value that actually corresponds to
    it -- e.g. a reward doing `reference.strip() in text` gets a `list` and
    silently crashes (caught upstream, scoring 0), even at batch size 1
    where the list just happens to have length 1.

    Only keys whose value is a list/tuple of exactly `batch_size` length are
    sliced; everything else (scalars, callables, mismatched-length values)
    is passed through unchanged, since those aren't batch-aligned columns.
    """
    sliced = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            sliced[key] = value[idx]
        else:
            sliced[key] = value
    return sliced


class CommonRewardHandler:
    """Mixin for handling reward functions in RL trainers."""

    def _combined_reward_function(
            self,
            completions: List[str],
            **kwargs) -> List[float]:
        """Combined reward function dispatcher with async support.

        Args:
            completions: List of generated completions
            **kwargs: Additional arguments from TRL (prompts, answer, test_list, etc.)

        Returns:
            List of combined reward scores (averaged across all functions)
        """
        if not completions or not self.reward_functions:
            return [0.0] * len(completions)

        # Check if any reward functions are async
        has_async = any(asyncio.iscoroutinefunction(rf) for rf in self.reward_functions)

        if has_async:
            # Use async path for concurrent execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._async_compute_rewards(completions, **kwargs)
                )
            finally:
                loop.close()
        else:
            # Use sync path
            return self._sync_compute_rewards(completions, **kwargs)

    def _sync_compute_rewards(
            self,
            completions: List[str],
            **kwargs) -> List[float]:
        """Synchronous reward computation for all completions.

        Args:
            completions: List of generated completions
            **kwargs: Additional arguments from TRL

        Returns:
            List of averaged reward scores
        """
        batch_rewards = []
        batch_size = len(completions)

        for idx, completion in enumerate(completions):
            sample_kwargs = slice_batch_kwargs_for_sample(kwargs, idx, batch_size)
            rewards = []
            for reward_func in self.reward_functions:
                name = getattr(reward_func, "__name__", repr(reward_func))
                try:
                    reward = call_reward_safely(reward_func, completion, **sample_kwargs)
                    # Handle both single value and list returns
                    if isinstance(reward, list) and reward:
                        rewards.append(reward[0])
                    elif isinstance(reward, (int, float)):
                        rewards.append(float(reward))
                    else:
                        logger.warning(
                            f"Reward function '{name}' returned non-numeric value "
                            f"{reward!r} for sample {idx}; treating as no contribution."
                        )
                except Exception as e:
                    logger.warning(
                        f"Reward function '{name}' raised {type(e).__name__} on sample "
                        f"{idx}: {e}. Contributing nothing for this sample -- if this "
                        f"keeps happening, that reward function is silently scoring 0."
                    )
                    continue

            # Average rewards from all functions
            batch_rewards.append(sum(rewards) / len(rewards) if rewards else 0.0)

        return batch_rewards

    async def _async_compute_rewards(
            self,
            completions: List[str],
            **kwargs) -> List[float]:
        """Asynchronous reward computation with concurrent execution.

        Args:
            completions: List of generated completions
            **kwargs: Additional arguments from TRL

        Returns:
            List of averaged reward scores
        """
        batch_rewards = []
        batch_size = len(completions)

        for idx, completion in enumerate(completions):
            sample_kwargs = slice_batch_kwargs_for_sample(kwargs, idx, batch_size)
            tasks = []
            for reward_func in self.reward_functions:
                call_kwargs = resolve_reward_call_kwargs(reward_func, completion, **sample_kwargs)
                if asyncio.iscoroutinefunction(reward_func):
                    # Async function - call directly
                    task = reward_func(**call_kwargs)
                else:
                    # Sync function - run in thread pool
                    task = asyncio.to_thread(
                        reward_func,
                        **call_kwargs
                    )
                tasks.append(task)

            # Execute all reward functions concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid rewards
            rewards = []
            for reward_func, result in zip(self.reward_functions, results):
                name = getattr(reward_func, "__name__", repr(reward_func))
                if isinstance(result, Exception):
                    logger.warning(
                        f"Reward function '{name}' raised {type(result).__name__} on "
                        f"sample {idx}: {result}. Contributing nothing for this sample."
                    )
                    continue

                # Handle both list and scalar returns
                if isinstance(result, list) and result:
                    rewards.append(result[0])
                elif isinstance(result, (int, float)):
                    rewards.append(float(result))
                else:
                    logger.warning(
                        f"Reward function '{name}' returned non-numeric value "
                        f"{result!r} for sample {idx}; treating as no contribution."
                    )

            # Average rewards from all functions
            batch_rewards.append(sum(rewards) / len(rewards) if rewards else 0.0)

        return batch_rewards
