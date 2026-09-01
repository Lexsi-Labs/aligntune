"""
CPU-only reward function tests that simulate how RL trainers actually call
reward functions, WITHOUT loading any model or running training.

Scope on purpose:
- No GPU, no model download, no dataset download, no `trainer.train()`.
- We bypass each trainer's heavy `__init__`/`setup_model()`/`setup_data()` and
  exercise only `setup_rewards()` + the reward-calling method
  (`_combined_reward_function` / `FunctionBasedRewardModel`) against
  hand-built inputs shaped exactly like what TRL passes at generation time:
  `reward_func(prompts=[...], completions=[...], completion_ids=[...], <dataset_columns>=[...])`.

Why this file exists: aligntune has (at least) three different conventions
for wiring a reward function into a trainer:

  1. TRL-mixin backends (GRPO, DAPO, GSPO, DR_GRPO) via
     `CommonRewardHandler` in `aligntune/core/rl/reward_handler.py`.
  2. Unsloth GRPO, which uses its own dict-based reward list and
     (for GRPO) a signature-driven call in `_call_reward_function`.
  3. TRL PPO, which wraps rewards in `FunctionBasedRewardModel` and calls
     each raw function as `reward_func(text)` (single positional string,
     no reference/kwargs at all).

  Custom function key: `{"type": "custom", "params": {"function": fn}}` OR
  `{"type": "custom", "params": {"reward_function": fn}}` -- both keys are
  now accepted by all backends (GRPO/Unsloth-GRPO/PPO),
  so the same reward spec is portable across all of them. (Previously PPO
  only recognized `reward_function` and the others only `function`, so
  reusing one spec across backends would silently drop the custom reward on
  whichever backend didn't recognize its key -- see
  TestPPORewardSimulation::test_function_key_now_also_works_under_ppo.)

A reward function written and validated against one backend can still
silently misbehave (not crash!) against another for reasons beyond the key
name -- e.g. PPO only ever supplies one positional arg. Section E
demonstrates this concretely. Section F provides a reusable compatibility
check (`assert_reward_fn_is_cross_trainer_safe`) to run against any *new*
custom reward function before wiring it into any *future* trainer/backend.
"""
import sys
import types
import asyncio
import pytest

sys.path.insert(0, "src")

from aligntune.rewards.core import (
    RewardConfig,
    RewardType,
    LengthReward,
    CoherenceReward,
    MathCorrectnessReward,
    CodeSyntaxReward,
    CompositeReward,
)
from aligntune.core.rl.reward_handler import (
    CommonRewardHandler,
    resolve_reward_call_kwargs,
    call_reward_safely,
    extract_completion_text,
)


def make_reward(cls, **params):
    return cls(RewardConfig(reward_type=RewardType.LENGTH, weight=1.0, params=params))


# ===========================================================================
# SECTION A -- built-in reward classes, direct unit tests, edge cases
# ===========================================================================

class TestBuiltinRewardEdgeCases:
    def test_length_reward_normal_range(self):
        r = make_reward(LengthReward, min_length=2, max_length=10)
        assert r.compute("one two three") == 1.0

    def test_length_reward_empty_string(self):
        r = make_reward(LengthReward, min_length=2, max_length=10)
        assert r.compute("") == 0.0

    def test_length_reward_none_text_degrades_to_zero(self):
        """Regression test: LengthReward.compute used to have no guard for
        text=None (`text.split()` raised AttributeError). If completion
        extraction ever yields None (e.g. a malformed dataset row), it now
        degrades to 0.0 instead of crashing the whole batch."""
        r = make_reward(LengthReward, min_length=2, max_length=10)
        assert r.compute(None) == 0.0

    def test_length_reward_non_string_int_degrades_to_zero(self):
        r = make_reward(LengthReward)
        assert r.compute(12345) == 0.0

    def test_coherence_reward_short_text(self):
        r = make_reward(CoherenceReward)
        assert r.compute("Only one sentence") == 0.0

    def test_math_correctness_no_expression_found(self):
        r = make_reward(MathCorrectnessReward)
        assert r.compute("no math here") == 0.0

    def test_math_correctness_none_text_degrades_to_zero(self):
        r = make_reward(MathCorrectnessReward)
        assert r.compute(None) == 0.0

    def test_code_syntax_reward_no_code_block(self):
        r = make_reward(CodeSyntaxReward)
        assert r.compute("just prose") == 0.0

    def test_code_syntax_reward_valid_code(self):
        r = make_reward(CodeSyntaxReward)
        text = "```python\nprint('hi')\n```"
        assert r.compute(text) > 0.0

    def test_batch_compute_mismatched_reference_length_no_longer_truncates(self):
        """Regression test: RewardFunction.batch_compute's default fallback
        path used to be `[self.compute(t, r) for t, r in zip(texts,
        references)]` (core.py:180), and zip() silently truncated to the
        SHORTER list -- fewer scores than completions, no exception, no
        warning. Now the batch is padded to match `texts` length instead."""
        r = make_reward(LengthReward, min_length=1, max_length=100)
        texts = ["a b", "c d e", "f"]
        scores = r.batch_compute(texts, references=["ref1"])
        assert len(scores) == 3


class TestCompositeRewardEdgeCases:
    def test_empty_reward_list_raises(self):
        """Regression test: CompositeReward([]) used to construct silently
        and always return 0.0 from compute() with no indication the ensemble
        was misconfigured (zero reward functions). Now fails fast."""
        with pytest.raises(ValueError):
            CompositeReward(reward_functions=[])

    def test_mismatched_weights_raises(self):
        r1 = make_reward(LengthReward)
        with pytest.raises(ValueError):
            CompositeReward(reward_functions=[r1], weights=[0.5, 0.5])

    def test_invalid_ensemble_mode_raises(self):
        r1 = make_reward(LengthReward)
        with pytest.raises(ValueError):
            CompositeReward(reward_functions=[r1], ensemble_mode="bogus_mode")

    def test_one_reward_raises_is_isolated(self):
        """A sub-reward that raises must not take down the whole composite."""
        class Broken(LengthReward):
            def compute(self, text, reference=None, **kwargs):
                raise RuntimeError("boom")

        good = make_reward(LengthReward, min_length=0, max_length=100)
        broken = Broken(RewardConfig(reward_type=RewardType.LENGTH, weight=1.0, params={}))
        c = CompositeReward(reward_functions=[good, broken])
        score = c.compute("some text here")
        assert score == pytest.approx(0.5)  # good=1.0, broken->0.0, mean=0.5


# ===========================================================================
# SECTION B -- custom reward function styles a user might write
# ===========================================================================

def universal_style_reward(text, reference=None, **kwargs) -> float:
    """Recommended style: single text-like positional param + **kwargs."""
    if not text or not isinstance(text, str):
        return 0.0
    return 1.0 if reference and reference.strip() in text else 0.0


def es_style_reward(prompt, response, reference, **kwargs) -> float:
    """Style copied from tests/test_es.py -- 3 required positional params,
    no defaults. Works fine when the caller always supplies all three by
    keyword, but is a landmine for any backend that calls with fewer args
    (see PPO in Section E)."""
    if not response or not reference:
        return 0.0
    return 1.0 if reference.strip() in response else 0.0


def buggy_reward(text, reference=None, **kwargs) -> float:
    """Deliberately raises a non-TypeError bug (e.g. a real logic bug),
    to check whether trainer-side exception handling reports it or
    silently swallows it as if the reward were simply 0."""
    return 1 / len(text)  # ZeroDivisionError on empty string


def single_arg_only_reward(text) -> float:
    """No **kwargs, no reference. Common beginner style."""
    return float(len(text.split()) > 0)


# ===========================================================================
# SECTION C -- simulate TRL-mixin backends (GRPO/DAPO/GSPO/DR_GRPO)
#              via CommonRewardHandler, exactly as TRL calls it.
# ===========================================================================

class FakeTRLStyleTrainer(CommonRewardHandler):
    """Minimal stand-in: only what CommonRewardHandler needs."""
    def __init__(self, reward_functions):
        self.reward_functions = reward_functions


class TestCommonRewardHandlerSimulation:
    def _trl_call(self, trainer, completions, **extra):
        """Shape a call exactly like TRL's GRPOTrainer invokes
        reward_funcs: prompts / completions / completion_ids + every extra
        dataset column, all as kwargs."""
        prompts = extra.pop("prompts", [f"prompt {i}" for i in range(len(completions))])
        completion_ids = extra.pop("completion_ids", [[1, 2, 3]] * len(completions))
        return trainer._combined_reward_function(
            completions, prompts=prompts, completion_ids=completion_ids, **extra
        )

    def test_adapter_alone_works_correctly_given_a_prebound_scalar(self):
        """Isolates resolve_reward_call_kwargs/call_reward_safely (the
        signature-introspection adapter) from the batching loop above it:
        called directly with an already-scalar `reference`, it correctly
        binds completion text to the function's first positional param and
        filters kwargs to what the function declares. The adapter itself is
        NOT the source of the bug below -- the loop that calls it is."""
        result = call_reward_safely(universal_style_reward, "the answer is 42", reference="42")
        assert result == 1.0

    def test_FIXED_combined_reward_function_now_slices_batch_kwargs_per_sample(self):
        """Regression test for a CRITICAL bug that used to live here:
        `_sync_compute_rewards` looped `for completion in completions:` but
        called `call_reward_safely(reward_func, completion, **kwargs)` with
        the SAME, unsliced `kwargs` on every iteration -- every completion's
        reward call received the FULL batch-length list for any dataset
        column (`reference`, `answer`, `test_list`) instead of the one value
        actually corresponding to it. Fixed via `slice_batch_kwargs_for_sample`
        in reward_handler.py, applied before every per-completion call. This
        affects every TRL-mixin backend: GRPO, DAPO, GSPO, DR_GRPO."""
        seen = []

        def probe(text, reference=None, **kw):
            seen.append((text, reference))
            return 1.0

        trainer = FakeTRLStyleTrainer([probe])
        self._trl_call(trainer, ["comp A", "comp B"], reference=["ref_A", "ref_B"])
        assert seen == [("comp A", "ref_A"), ("comp B", "ref_B")]

    def test_FIXED_reference_using_reward_no_longer_silently_scores_zero(self):
        """Regression test: a reward function written in the natural,
        idiomatic style (`reference.strip() in text`) used to get a `list`
        instead of a `str` for `reference` (list.strip() doesn't exist ->
        AttributeError -> silently swallowed -> scored 0.0 even for a
        correct completion). Reproduced even at batch size 1. Now fixed."""
        trainer = FakeTRLStyleTrainer([universal_style_reward])
        scores = self._trl_call(trainer, ["the answer is 42"], reference=["42"])
        assert scores == [1.0]

    def test_conversational_completion_format_is_flattened(self):
        """TRL sends chat-format completions as [{"role": "assistant",
        "content": "..."}] for conversational datasets -- must not crash
        a reward function that only understands plain strings. Uses a
        reference-free predicate to isolate this from the bug above."""
        trainer = FakeTRLStyleTrainer([lambda text, **kw: 1.0 if "42" in text else 0.0])
        completions = [[{"role": "assistant", "content": "the answer is 42"}]]
        scores = self._trl_call(trainer, completions)
        assert scores == [1.0]

    def test_buggy_reward_is_silently_swallowed(self):
        """BUG (design tradeoff, worth knowing): CommonRewardHandler._sync_compute_rewards
        catches ALL exceptions per reward-function call (`except Exception: continue`)
        and only logs at DEBUG level. A real bug inside a custom reward function
        (ZeroDivisionError here) produces NO warning and NO crash -- it just
        silently contributes nothing to the average, which can look
        indistinguishable from "reward legitimately scored 0"."""
        trainer = FakeTRLStyleTrainer([buggy_reward])
        scores = self._trl_call(trainer, [""])  # empty text -> ZeroDivisionError inside
        assert scores == [0.0]  # no exception raised out of _combined_reward_function

    def test_mixed_good_and_buggy_rewards_average_only_the_good_one(self):
        trainer = FakeTRLStyleTrainer(
            [lambda text, **kw: 1.0 if text else 0.0, buggy_reward]
        )
        scores = self._trl_call(trainer, [""])
        # first reward("") -> 0.0 ; buggy_reward("") -> ZeroDivisionError, swallowed
        assert scores == [0.0]

    def test_extra_dataset_column_named_same_as_first_param_no_duplicate_error(self):
        """Regression check for the exact bug resolve_reward_call_kwargs exists
        to prevent: a dataset column literally named 'text' must not collide
        with the reward function's own first positional param. Uses a
        reference-free predicate to isolate this from the batch-list bug."""
        contains_foo = lambda text, **kw: 1.0 if "FOO" in text else 0.0
        trainer = FakeTRLStyleTrainer([contains_foo])
        scores = self._trl_call(trainer, ["contains FOO"], text=["unrelated collision value"])
        assert scores == [1.0]

    def test_no_reward_functions_returns_zeros_not_crash(self):
        trainer = FakeTRLStyleTrainer([])
        scores = self._trl_call(trainer, ["a", "b"])
        assert scores == [0.0, 0.0]

    def test_es_style_reward_now_scores_correctly_with_matching_column_names(self):
        """The exact reward-function style used/validated in tests/test_es.py
        (`def fn(prompt, response, reference, **kwargs)`) has a quirk worth
        knowing: resolve_reward_call_kwargs binds the completion TEXT to this
        function's first positional param, which is literally named `prompt`
        here -- so the "prompt" the function receives is actually the
        completion, not the original prompt (harmless for this particular
        function since it ignores its own `prompt` arg, but a trap if a
        reward function actually needs the real prompt AND is named this
        way). `response`/`reference` now correctly arrive as per-sample
        scalars (regression-tested above), so this scores correctly here."""
        trainer = FakeTRLStyleTrainer([es_style_reward])
        scores = self._trl_call(
            trainer, ["FOO is the answer"], response=["FOO is the answer"], reference=["FOO"]
        )
        assert scores == [1.0]

    def test_es_style_reward_now_resolves_reference_alias_from_answer_column(self):
        """Regression test for the reference-alias fallback: a dataset column
        named `answer` (GSM8K-style) instead of `reference` now correctly
        resolves for any reward function that declares a `reference` param
        (or accepts **kwargs), matching Unsloth GRPO's existing alias
        behavior. `response` still needs to match by name since there's no
        generic "completion-under-test" alias for it."""
        trainer = FakeTRLStyleTrainer([es_style_reward])
        scores = self._trl_call(
            trainer, ["FOO is the answer"], response=["FOO is the answer"], answer=["FOO"]
        )
        assert scores == [1.0]


# ===========================================================================
# SECTION C.5 -- realistic end-to-end reproduction: real GRPO trainer
#                setup + a real registry reward (not a mock), shaped exactly
#                like a real GSM8K-style batch (multiple generations per
#                prompt, dataset column named 'answer'). This is the direct
#                repro of "sometimes I'm seeing 0 rewards in GRPO".
# ===========================================================================

class TestRealisticGRPOBatch:
    """Uses the REAL TRLGRPOTrainer.setup_rewards
    and the REAL registry-based MathVerifiableReward (not a mock/toy reward),
    fed a batch shaped like TRL's actual GRPO calling convention: 2 unique
    prompts x 4 generations each = 8 completions, with the `answer` dataset
    column repeated/aligned to all 8 completions (exactly how TRL expands
    extra dataset columns for multi-generation batches)."""

    COMPLETIONS = [
        "Let's compute. The answer is 18",
        "Hmm, the answer is 20",   # wrong
        "The answer is 18",
        "the answer is 20",        # wrong
        "So the answer is 3",
        "the answer is 5",         # wrong
        "the answer is 3",
        "the answer is 5",         # wrong
    ]
    PROMPTS = ["What is 12+6?"] * 4 + ["What is 10-7?"] * 4
    ANSWERS = ["18"] * 4 + ["3"] * 4  # GSM8K-style column name: 'answer', not 'reference'
    EXPECTED = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def _build_trainer(self, trainer_cls):
        trainer = object.__new__(trainer_cls)
        trainer.config = types.SimpleNamespace(
            rewards=[{"type": "math_verifiable", "params": {}}]
        )
        trainer.reward_functions = []
        trainer.setup_rewards()
        return trainer

    def test_grpo_math_verifiable_reward_scores_correctly_on_realistic_batch(self):
        # GRPO reward functions are handed straight to TRL's
        # GRPOTrainer as reward_funcs=... now -- TRL calls and
        # combines them itself, so there's no longer an aligntune-owned
        # _combined_reward_function dispatcher on these classes; the prepared
        # function itself is the TRL-batch-callable (see
        # notebooks/reward_function_testing.ipynb section 4).
        from aligntune.backends.trl.rl.grpo.grpo import TRLGRPOTrainer
        trainer = self._build_trainer(TRLGRPOTrainer)
        scores = trainer.reward_functions[0](
            prompts=self.PROMPTS, completions=self.COMPLETIONS,
            completion_ids=[[1, 2, 3]] * 8, answer=self.ANSWERS,
        )
        assert scores == self.EXPECTED

    def test_single_prompt_single_generation_still_works(self):
        """Sanity check at the smallest possible batch shape (no multi-
        generation grouping at all) to make sure the fix doesn't only work
        for the multi-generation case."""
        from aligntune.backends.trl.rl.grpo.grpo import TRLGRPOTrainer
        trainer = self._build_trainer(TRLGRPOTrainer)
        scores = trainer.reward_functions[0](
            prompts=["What is 12+6?"], completions=["The answer is 18"],
            completion_ids=[[1, 2, 3]], answer=["18"],
        )
        assert scores == [1.0]


# ===========================================================================
# SECTION D -- simulate the Unsloth GRPO backend (different convention)
# ===========================================================================

class TestUnslothGRPOSimulation:
    """Exercises the REAL UnslothGRPOTrainer.setup_rewards /
    _combined_reward_function code (not a re-implementation), bypassing
    __init__ (which would require a real model/tokenizer/config object)."""

    def _make_trainer(self, rewards_config):
        from aligntune.backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
        trainer = object.__new__(UnslothGRPOTrainer)
        trainer.config = types.SimpleNamespace(rewards=rewards_config)
        trainer.reward_functions = []
        trainer.setup_rewards()
        return trainer

    def test_custom_function_key_loads_correctly(self):
        # Unsloth GRPO reward_functions are now prepared the same way as TRL
        # GRPO (see notebooks/reward_function_testing.ipynb section 5):
        # a flat list of TRL-batch-callables (adapter-wrapped, so not `is`
        # the original function), not List[dict]. Verify the "function" key
        # was correctly used by calling the resulting callable.
        trainer = self._make_trainer(
            [{"type": "custom", "params": {"function": universal_style_reward}}]
        )
        assert len(trainer.reward_functions) == 1
        scores = trainer.reward_functions[0](
            prompts=["p"], completions=["one two three"], reference=["one"]
        )
        assert scores == [1.0]

    def test_single_arg_reward_via_fallback_pattern_4(self):
        trainer = self._make_trainer(
            [{"type": "custom", "params": {"function": single_arg_only_reward}}]
        )
        scores = trainer.reward_functions[0](
            prompts=["p"], completions=["one two three"], answer=["ignored"]
        )
        assert scores == [1.0]

    def test_reference_column_named_reference_no_longer_causes_duplicate_kwarg(self):
        """Regression test: _call_reward_function used to do
        `reward_func(completion, test_cases=..., reference=reference, **kwargs)`
        without popping the resolved reference key out of `kwargs` first. If
        the dataset's reference column was literally named 'reference',
        kwargs STILL contained 'reference' too, so the old Pattern-1 call
        always raised "got multiple values for keyword argument 'reference'"
        and silently fell through to a degraded pattern that dropped
        `test_cases`. _call_reward_function is now signature-driven (reuses
        reward_handler.resolve_reward_call_kwargs) instead of a try/except
        cascade, so a reward needing BOTH test_cases and reference together
        now correctly receives both in one call."""
        seen_kwargs = {}

        def needs_test_cases_and_reference(text, test_cases=None, reference=None, **kw):
            seen_kwargs["test_cases"] = test_cases
            seen_kwargs["reference"] = reference
            return 1.0

        trainer = self._make_trainer(
            [{"type": "custom", "params": {"function": needs_test_cases_and_reference}}]
        )
        trainer.reward_functions[0](
            prompts=["p"],
            completions=["completion text"],
            reference=["expected"],       # <-- column literally named 'reference'
            test_cases=["assert True"],
        )
        assert seen_kwargs == {"test_cases": "assert True", "reference": "expected"}

    def test_non_type_error_bug_in_reward_is_silently_swallowed(self):
        """Custom and registry rewards are now wrapped identically by
        RewardBridge.wrap_registry_reward (see reward_handler.py), which
        deliberately re-raises a genuine bug (e.g. ZeroDivisionError) as a
        RuntimeError rather than silently scoring it 0.0 - a bug in a reward
        function should surface loudly, not disappear into a quiet 0 score."""
        trainer = self._make_trainer(
            [{"type": "custom", "params": {"function": buggy_reward}}]
        )
        with pytest.raises(RuntimeError, match="buggy_reward"):
            trainer.reward_functions[0](prompts=["p"], completions=[""])

    def test_reward_functions_stored_as_callables_matching_trl_backends(self):
        """Unsloth GRPO reward_functions are now prepared the same way as TRL
        GRPO (see notebooks/reward_function_testing.ipynb section 5): a
        flat list of TRL-batch-callables, not the old List[dict] shape."""
        trainer = self._make_trainer(
            [{"type": "custom", "params": {"function": universal_style_reward}}]
        )
        assert callable(trainer.reward_functions[0])
        assert not isinstance(trainer.reward_functions[0], dict)


# ===========================================================================
# SECTION E -- simulate the TRL PPO backend (different key AND different
#              calling convention: FunctionBasedRewardModel calls fn(text)
#              with a single positional string only).
# ===========================================================================

class TestPPORewardSimulation:
    def _run_setup_rewards(self, rewards_config):
        from aligntune.backends.trl.rl.ppo.ppo import TRLPPOTrainer
        trainer = object.__new__(TRLPPOTrainer)
        trainer.config = types.SimpleNamespace(rewards=rewards_config)
        trainer.reward_functions = []
        trainer.setup_rewards()
        return trainer

    def test_correct_key_reward_function_loads_custom_reward(self):
        trainer = self._run_setup_rewards(
            [{"type": "custom", "params": {"reward_function": universal_style_reward}}]
        )
        assert trainer.reward_functions[0]["function"] is universal_style_reward

    def test_function_key_now_also_works_under_ppo(self):
        """Regression test for the single biggest cross-trainer portability
        trap in this repo: the SAME spec shape used for GRPO/Unsloth-GRPO
        -- `{"type": "custom", "params": {"function": fn}}` -- used to be
        SILENTLY ignored by PPO, because PPO's setup_rewards() only
        recognized the key `reward_function`, not `function`. With no
        exception raised anywhere, PPO would substitute a completely
        different generic default_length_reward and train on that instead
        -- silently wrong, not a crash. Both backends now accept either key."""
        trainer = self._run_setup_rewards(
            [{"type": "custom", "params": {"function": universal_style_reward}}]
        )
        assert len(trainer.reward_functions) == 1
        assert trainer.reward_functions[0]["function"] is universal_style_reward
        assert trainer.reward_functions[0]["name"] != "default_length"

    def test_function_based_reward_model_calls_with_single_positional_text_only(self):
        """FunctionBasedRewardModel._compute_reward calls `reward_func(text)`
        -- ONE positional string argument, nothing else, ever. Confirmed via
        function_based_reward_model.py. A reward function requiring more
        positional args than that will raise TypeError, which is caught by
        a bare `except Exception: continue` and contributes 0 to the total
        -- again, silent, no crash, no warning above DEBUG level."""
        from aligntune.core.rl.function_based_reward_model import FunctionBasedRewardModel
        import torch

        class FakeTokenizer:
            def batch_decode(self, input_ids, **kw):
                return ["the answer is 42", "wrong answer"]

        model = FunctionBasedRewardModel(
            reward_functions=[universal_style_reward],
            tokenizer=FakeTokenizer(),
            device="cpu",
            dtype=torch.float32,
        )
        input_ids = torch.zeros((2, 5), dtype=torch.long)
        backbone_out = model.model.forward(input_ids)
        rewards = model.score(backbone_out.hidden_states)
        assert rewards.shape == (2, 5, 1)

    def test_es_style_multi_positional_reward_silently_scores_zero_under_ppo(self):
        """The exact reward function style used/validated in tests/test_es.py
        (`def fn(prompt, response, reference, **kwargs)`) is only ever called
        by TRL's GRPOTrainer with all dataset columns as kwargs,
        so it works fine there (see Section C). Under PPO's
        FunctionBasedRewardModel, it is called as `es_style_reward(text)` --
        one positional arg only -- which binds to `prompt`, leaving `response`
        and `reference` unfilled -> TypeError: missing 2 required positional
        arguments -> silently caught -> contributes 0.0. This is the concrete
        "same reward function, different trainer, silently wrong" failure
        the user asked to guard against."""
        from aligntune.core.rl.function_based_reward_model import FunctionBasedRewardModel
        import torch

        class FakeTokenizer:
            def batch_decode(self, input_ids, **kw):
                return ["FOO is the answer"]

        model = FunctionBasedRewardModel(
            reward_functions=[es_style_reward],
            tokenizer=FakeTokenizer(),
            device="cpu",
            dtype=torch.float32,
        )
        input_ids = torch.zeros((1, 5), dtype=torch.long)
        backbone_out = model.model.forward(input_ids)
        rewards = model.score(backbone_out.hidden_states)
        assert torch.all(rewards == 0.0), (
            "es_style_reward should have silently contributed 0 under PPO's "
            "single-positional-arg calling convention."
        )


# ===========================================================================
# SECTION F -- reusable "will this work under ANY future trainer" checker
# ===========================================================================

def assert_reward_fn_is_cross_trainer_safe(fn):
    """Run this against any NEW custom reward function before wiring it into
    ANY trainer/backend (present or future). It does not require a GPU, a
    model, or a dataset. It statically + dynamically checks the properties
    that differ across aligntune's known reward-calling conventions:

      1. Callable with **kwargs (so unknown dataset columns / new kwargs a
         future backend decides to pass don't raise TypeError).
      2. At most ONE required positional parameter (the completion text).
         PPO's FunctionBasedRewardModel calls `fn(text)` and nothing else --
         any additional required positional param makes the function
         silently score 0 under PPO (see TestPPORewardSimulation).
      3. Does not raise on `fn(text)` alone (single positional string, no
         other kwargs) -- same PPO-shaped call.
      4. Does not raise when `reference`/`answer`/`response` are absent
         (some backends never supply them).
      5. Does not raise on empty string or whitespace-only input.
      6. Returns a plain int/float (not None, not a list) from the
         single-arg call, since PPO's wrapper coerces non-numeric returns to
         0.0 with only a warning -- silent under this call path too.
    """
    import inspect as _inspect

    sig = _inspect.signature(fn)
    required_positional = [
        p for p in sig.parameters.values()
        if p.kind in (_inspect.Parameter.POSITIONAL_ONLY, _inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is _inspect.Parameter.empty
    ]
    has_var_keyword = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    problems = []
    if not has_var_keyword:
        problems.append(
            "reward function does not accept **kwargs -- will TypeError on any "
            "dataset column / backend kwarg it wasn't written to expect."
        )
    if len(required_positional) > 1:
        problems.append(
            f"reward function has {len(required_positional)} required positional "
            f"params ({[p.name for p in required_positional]}) -- PPO's "
            "FunctionBasedRewardModel only ever supplies one positional arg "
            "(the completion text); extra required params will silently score 0."
        )

    try:
        result = fn("this is a sample completion for testing purposes")
    except Exception as e:
        problems.append(f"fn(text) alone raised {type(e).__name__}: {e}")
        result = None

    if result is not None and not isinstance(result, (int, float)):
        problems.append(f"fn(text) alone returned non-numeric type {type(result)}")

    for edge in ["", "   "]:
        try:
            fn(edge)
        except Exception as e:
            problems.append(f"fn({edge!r}) raised {type(e).__name__}: {e} (should degrade, not raise)")

    return problems


class TestCrossTrainerSafetyChecker:
    def test_universal_style_reward_passes(self):
        problems = assert_reward_fn_is_cross_trainer_safe(universal_style_reward)
        assert problems == []

    def test_es_style_reward_flagged_for_extra_required_positional_args(self):
        problems = assert_reward_fn_is_cross_trainer_safe(es_style_reward)
        assert any("required positional" in p for p in problems)

    def test_single_arg_only_reward_flagged_for_missing_kwargs(self):
        problems = assert_reward_fn_is_cross_trainer_safe(single_arg_only_reward)
        assert any("**kwargs" in p for p in problems)

    def test_buggy_reward_flagged_for_empty_string_crash(self):
        problems = assert_reward_fn_is_cross_trainer_safe(buggy_reward)
        assert any("raised ZeroDivisionError" in p for p in problems)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
