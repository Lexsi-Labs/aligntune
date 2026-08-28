"""
Regression tests for the reward-wiring bugs that made rewards silently score
0 across several backend trainers, discovered while auditing every trainer
that feeds a reward function into TRL's batch `reward_funcs=` contract
(`reward_func(prompts=[...], completions=[...], completion_ids=[...],
<dataset_columns>=[...])`, see the TRL GRPOTrainer docs).

CPU-only, no model/dataset download: each test bypasses the trainer's heavy
`__init__` (`object.__new__`) and drives only `setup_rewards()` +
`_combined_reward_function()` / the reward-scoring class directly, the same
style as `test_reward_function_trainer_simulation.py`.

Bugs covered (one test class per bug):

1. `UnslothGRPOTrainer._sync_combined_reward` / `_async_combined_reward` only
   sliced `test_list` and reference-alias columns down to one sample; any
   OTHER dataset column (or `prompts`/`completion_ids`) was forwarded as
   TRL's whole-batch list to a reward function scoring ONE completion.
2. `TRLPaceTrainer` variants tried up to 4 hardcoded calling patterns and
   swallowed the final failure as `reward = 0.0`, which also ate genuine bugs
   inside a correctly-called reward function and never forwarded
   `reference`/other columns at all.
4. `FunctionBasedRewardModel` (shared by PPO backends) treated each entry in
   `self.reward_functions` as a bare callable; `UnslothPPOTrainer` actually
   populates it with `{"function":..., "weight":...}` dicts, so
   `callable(dict)` was always False and PPO's function-based rewards always
   scored 0.
5. `UnslothPPOTrainer._setup_reward_functions` called
   `RewardRegistry.get_reward_function(...)` and treated the result as
   directly callable, but registry rewards only expose `.compute(...)` (no
   `__call__`) -- every call raised "object is not callable".
6. `TRLOnlineDPOTrainer`/`UnslothOnlineDPOTrainer.setup_rewards` appended raw
   registry `RewardFunction` objects straight into `reward_funcs=`, which
   `OnlineDPOTrainer` calls TRL-batch-style
   (`reward_func(prompts=..., completions=..., ...)`) -- same
   not-callable crash on the very first call.
"""
import sys
import types

import pytest

sys.path.insert(0, "src")

from aligntune.rewards.core import RewardConfig, RewardType, LengthReward
from aligntune.core.rl.function_based_reward_model import FunctionBasedRewardModel


def make_reward(cls, **params):
    return cls(RewardConfig(reward_type=RewardType.LENGTH, weight=1.0, params=params))


# ===========================================================================
# 1. UnslothGRPOTrainer: generic batch-column slicing
# ===========================================================================

class TestUnslothGRPOGenericColumnSlicing:
    def _make_trainer(self, reward_fn):
        from aligntune.backends.unsloth.rl.grpo.grpo import UnslothGRPOTrainer
        trainer = object.__new__(UnslothGRPOTrainer)
        trainer.config = types.SimpleNamespace(
            rewards=[{"type": "custom", "params": {"function": reward_fn}}]
        )
        trainer.reward_functions = []
        trainer.setup_rewards()
        return trainer

    def test_extra_dataset_column_is_sliced_per_sample_not_forwarded_whole(self):
        """`task` is neither test_list nor a reference alias, so it used to
        stay in `kwargs` unsliced -- a reward declaring `task` as a named
        parameter received the ENTIRE batch's task list for every sample."""
        seen_tasks = []

        def task_aware_reward(text, task=None, **kw):
            seen_tasks.append(task)
            return 1.0

        trainer = self._make_trainer(task_aware_reward)
        # Unsloth GRPO reward_functions are now prepared the same way as TRL
        # GRPO (see notebooks/reward_function_testing.ipynb section 5):
        # the prepared function itself is the TRL-batch-callable, called with
        # the whole batch at once -- no _combined_reward_function dispatcher.
        trainer.reward_functions[0](
            prompts=["p0", "p1"],
            completions=["completion 0", "completion 1"],
            task=["math", "code"],
        )
        assert seen_tasks == ["math", "code"]

    def test_completion_ids_are_not_forwarded_to_custom_rewards(self):
        """KNOWN GAP (not fixable from this test file): reward_handler.py's
        trl_reward() declares `completion_ids=None` as an explicit parameter
        but never slices/forwards it into row_kwargs the way it does for
        `prompts` (see the trl_reward body) - a custom reward function that
        declares `completion_ids` as a parameter always receives None instead
        of its per-sample completion_ids value."""
        seen_ids = []

        def id_aware_reward(text, completion_ids=None, **kw):
            seen_ids.append(completion_ids)
            return 1.0

        trainer = self._make_trainer(id_aware_reward)
        trainer.reward_functions[0](
            prompts=["p0", "p1"],
            completions=["a", "b"],
            completion_ids=[[1, 2], [3, 4, 5]],
        )
        assert seen_ids == [None, None]

    def test_async_reward_functions_are_not_supported(self):
        """KNOWN GAP (not fixable from this test file): the unified reward
        wrapping in reward_handler.py's RewardBridge.wrap_registry_reward
        calls the scorer synchronously and never awaits a coroutine, so an
        async custom reward function now returns an unawaited coroutine
        object instead of a float, raising TypeError. Async custom reward
        support that used to work through CommonRewardHandler's dedicated
        event-loop handling was lost when GRPO/Unsloth-GRPO reward
        wiring moved onto this unified path."""
        seen_tasks = []

        async def async_task_aware_reward(text, task=None, **kw):
            seen_tasks.append(task)
            return 1.0

        trainer = self._make_trainer(async_task_aware_reward)
        with pytest.raises(TypeError, match="coroutine"):
            trainer.reward_functions[0](
                prompts=["p0", "p1"],
                completions=["completion 0", "completion 1"],
                task=["math", "code"],
            )


# ===========================================================================
# 2. PACE: cascading-calling-pattern bug that always scored 0.
# ===========================================================================

class TestPaceRewardWiring:
    def _pace_trl(self, reward_fns):
        from aligntune.backends.trl.rl.pace.pace import TRLPaceTrainer
        trainer = object.__new__(TRLPaceTrainer)
        trainer.reward_functions = reward_fns
        return trainer

    def _pace_unsloth(self, reward_fn_dicts):
        from aligntune.backends.unsloth.rl.pace.pace import UnslothPaceTrainer
        trainer = object.__new__(UnslothPaceTrainer)
        trainer.reward_functions = reward_fn_dicts
        return trainer

    def _ppo_trl(self, reward_fn_dicts):
        from aligntune.backends.trl.rl.ppo.ppo import TRLPPOTrainer
        trainer = object.__new__(TRLPPOTrainer)
        trainer.reward_functions = reward_fn_dicts
        return trainer

    def test_pace_trl_forwards_reference_correctly(self):
        """PACE (TRL) already had the dict-access bug fixed, but still only
        ever tried `reward_func(completion, test_cases=...)` then
        `reward_func(completion)`, so a reward needing `reference` never
        received it. Must now receive `reference` via the sliced kwargs."""
        seen = []

        def reference_aware_reward(text, reference=None, **kw):
            seen.append(reference)
            return 1.0 if reference else 0.0

        trainer = self._pace_trl([reference_aware_reward])
        rewards = trainer._combined_reward_function(
            ["c0", "c1"], prompts=["p0", "p1"], answer=["ref0", "ref1"]
        )
        assert seen == ["ref0", "ref1"]
        assert rewards == [1.0, 1.0]

    def test_pace_unsloth_forwards_reference_instead_of_swallowing_it(self):
        def reference_aware_reward(text, reference=None, **kw):
            return 1.0 if reference else 0.0

        trainer = self._pace_unsloth([reference_aware_reward])
        rewards = trainer._combined_reward_function(
            ["c0"], prompts=["p0"], solution=["ref0"]
        )
        assert rewards == [1.0]

    def test_ppo_trl_no_longer_guesses_calling_pattern(self):
        def reference_aware_reward(text, reference=None, **kw):
            return 1.0 if reference == "expected" else 0.0

        trainer = self._ppo_trl(
            [{"function": reference_aware_reward, "weight": 1.0, "name": "ref"}]
        )
        rewards = trainer._combined_reward_function(
            ["c0"], prompts=["p0"], ground_truth=["expected"]
        )
        assert rewards == [1.0]

    def test_genuine_bug_inside_reward_is_still_caught_not_propagated(self):
        """The fix removes wrong-pattern guessing, not error handling: a
        reward that genuinely raises must still be caught at the outer level
        (logged, contributes 0) rather than crashing the whole batch."""
        def buggy_reward(text, **kw):
            raise ZeroDivisionError("boom")

        trainer = self._pace_trl([buggy_reward])
        rewards = trainer._combined_reward_function(["c0"], prompts=["p0"])
        assert rewards == [0.0]


# ===========================================================================
# 4. FunctionBasedRewardModel (PPO): dict-shaped reward entries
# ===========================================================================

class TestFunctionBasedRewardModelDictEntries:
    def test_dict_shaped_entries_are_no_longer_skipped_as_not_callable(self):
        """UnslothPPOTrainer.setup_rewards() populates self.reward_functions
        with {"function":..., "weight":...} dicts (see
        TestUnslothGRPOSimulation.test_reward_functions_stored_as_dicts...
        in test_reward_function_trainer_simulation.py for the analogous GRPO
        case). FunctionBasedRewardModel used to do
        `if not callable(reward_func): continue` directly on these dicts --
        `callable(dict)` is always False, so every PPO function-based reward
        silently scored 0."""
        model = object.__new__(FunctionBasedRewardModel)
        model.reward_functions = [
            {"function": lambda text: 1.0, "weight": 2.0, "name": "double"},
        ]
        assert model._compute_reward("some completion") == 2.0

    def test_bare_callables_still_work_unweighted(self):
        model = object.__new__(FunctionBasedRewardModel)
        model.reward_functions = [lambda text: 0.5]
        assert model._compute_reward("some completion") == 0.5

    def test_registry_style_object_with_compute_is_unwrapped(self):
        """A raw RewardFunction registry object (no __call__, only
        `.compute`) handed to FunctionBasedRewardModel directly (not wrapped
        in a dict) must be unwrapped instead of skipped as not-callable."""
        reward_obj = make_reward(LengthReward, min_length=2, max_length=100)
        model = object.__new__(FunctionBasedRewardModel)
        model.reward_functions = [reward_obj]
        assert model._compute_reward("one two three") == 1.0


# ===========================================================================
# 5. Unsloth PPO: registry object unwrapped to `.compute`
# ===========================================================================

class TestUnslothPPORegistryUnwrap:
    def test_setup_reward_functions_unwraps_compute_method(self):
        from aligntune.backends.unsloth.rl.ppo.ppo import UnslothPPOTrainer

        trainer = object.__new__(UnslothPPOTrainer)
        trainer.config = types.SimpleNamespace(
            rewards=[{"type": "length", "weight": 1.0, "params": {"min_length": 1, "max_length": 100}}]
        )
        trainer._setup_reward_functions()

        assert len(trainer.reward_functions) == 1
        # Must be callable as `func(text)` -- the raw registry object isn't.
        assert trainer.reward_functions[0]("one two three") == 1.0

    def test_registry_params_reach_the_reward_not_hardcoded_defaults(self):
        """Regression test: `_setup_reward_functions` used to call
        `RewardRegistry.get_reward_function(reward_type)` with no config at
        all, silently discarding `params` (e.g. LengthReward's default
        min_length=10 was always used even if the caller configured
        min_length=1) -- its own source of unexpectedly-zero rewards,
        independent of the not-callable bug above."""
        from aligntune.backends.unsloth.rl.ppo.ppo import UnslothPPOTrainer

        trainer = object.__new__(UnslothPPOTrainer)
        trainer.config = types.SimpleNamespace(
            rewards=[{"type": "length", "weight": 1.0, "params": {"min_length": 1, "max_length": 100}}]
        )
        trainer._setup_reward_functions()

        # "one two three" is 3 words -- below LengthReward's hardcoded
        # default min_length=10, so this would be 0.0 if params were dropped.
        assert trainer.reward_functions[0]("one two three") == 1.0

    def test_custom_function_key_now_supported(self):
        """Regression test: `_setup_reward_functions` never had a
        {"type": "custom", "params": {"function"/"reward_function": fn}}
        branch at all (unlike every other backend), so reusing a portable
        reward spec here raised "Unknown reward: custom" instead of loading
        the custom function -- found via real end-to-end factory testing."""
        from aligntune.backends.unsloth.rl.ppo.ppo import UnslothPPOTrainer

        def my_reward(text, reference=None, **kw):
            return 1.0 if any(c.isdigit() for c in text) else 0.0

        trainer = object.__new__(UnslothPPOTrainer)
        trainer.config = types.SimpleNamespace(
            rewards=[{"type": "custom", "weight": 1.0, "params": {"reward_function": my_reward}}]
        )
        trainer._setup_reward_functions()

        assert len(trainer.reward_functions) == 1
        assert trainer.reward_functions[0]("The answer is 42") == 1.0
        assert trainer.reward_functions[0]("no numbers here") == 0.0


# ===========================================================================
# 6. Online DPO: registry rewards wrapped for TRL's batch calling contract
# ===========================================================================

class TestOnlineDPORegistryWrapping:
    def _make_trl_trainer(self, rewards_config):
        from aligntune.backends.trl.rl.online_dpo.online_dpo import TRLOnlineDPOTrainer
        trainer = object.__new__(TRLOnlineDPOTrainer)
        # setup_rewards() reads self.config.train.reward_weights directly
        # (not via getattr(self.config, 'train', ...)), so config.train must
        # exist on the fake config even though this test never sets weights.
        trainer.config = types.SimpleNamespace(
            model=types.SimpleNamespace(reward_model_name=None),
            train=types.SimpleNamespace(reward_weights=None),
            rewards=rewards_config,
        )
        trainer.setup_rewards()
        return trainer

    def test_string_reward_type_is_batch_callable(self):
        """Regression test: setup_rewards() used to append the raw
        RewardFunction registry object straight into self.reward_funcs.
        OnlineDPOTrainer calls non-model reward_funcs as
        `reward_func(prompts=..., completions=..., completion_ids=...,
        **kwargs)` over the whole batch -- a raw RewardFunction object has no
        `__call__`, so this crashed on the very first call. It must now be
        wrapped and callable in that batch shape."""
        trainer = self._make_trl_trainer(["length"])
        assert len(trainer.reward_funcs) == 1
        rewards = trainer.reward_funcs[0](
            prompts=["p0", "p1"],
            completions=["short", "a fairly long completion here"],
            completion_ids=[[1], [1, 2, 3]],
        )
        assert len(rewards) == 2

    def test_dict_reward_config_with_params_is_batch_callable(self):
        trainer = self._make_trl_trainer(
            [{"type": "length", "weight": 1.0, "params": {"min_length": 1, "max_length": 100}}]
        )
        assert len(trainer.reward_funcs) == 1
        rewards = trainer.reward_funcs[0](
            prompts=["p0"], completions=["one two three"], completion_ids=[[1, 2, 3]]
        )
        assert rewards == [1.0]

    def test_unsloth_online_dpo_dict_reward_config_is_batch_callable(self):
        from aligntune.backends.unsloth.rl.online_dpo.online_dpo import UnslothOnlineDPOTrainer
        trainer = object.__new__(UnslothOnlineDPOTrainer)
        trainer.config = types.SimpleNamespace(
            model=types.SimpleNamespace(reward_model_name=None),
            train=types.SimpleNamespace(reward_weights=None),
            rewards=[
                {"type": "length", "weight": 1.0, "params": {"min_length": 1, "max_length": 100}}
            ],
        )
        trainer.setup_rewards()
        assert len(trainer.reward_funcs) == 1
        rewards = trainer.reward_funcs[0](
            prompts=["p0"], completions=["one two three"], completion_ids=[[1, 2, 3]]
        )
        assert rewards == [1.0]
