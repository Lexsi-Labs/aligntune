# Changelog

All notable changes to AlignTune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full changelog.

## [Unreleased]

### Added
- SGLang rollout backend for GRPO (`rollout_backend="sglang"`): generation goes to a running SGLang server over HTTP, and weights sync once per optimizer step, as a LoRA adapter on a `--enable-lora` server or as streamed full weights (#32, @chayan-bit).
- SGLang rollout on Unsloth GRPO-family and counterfactual GRPO trainers, with Gemma 3 weight names, vision adapter tensors, regex LoRA `target_modules` and full sync for 4-bit models (#41, @zeralyngkhoi19).

### Changed
- SFT runs that don't set `max_steps` now train for exactly `num_epochs`. Before, the epochs were turned into steps with a floor of 10, so small datasets ran more steps than asked (#36, fixed in #39, @zeralyngkhoi19).
- `create_rl_trainer` now passes `rollout_backend` to the trainer, so `rollout_backend="vllm"` turns vLLM rollout on. Before, the argument was ignored and the trainer used `hf` (#32, @chayan-bit).
- Code rewards: a function must return plain Python data (`None`, `bool`, `int`, `float`, `str`, `list`, `tuple`, `set`, `frozenset`, `dict`); NumPy scalars, `bytes`, custom classes and results of more than 100,000 values in all now fail the test. An `assert` shape that can't be split, or a non-`assert` test string, fails with a warning. A completion runs with `__name__ == "__main__"`, so one that calls `sys.exit` under that guard fails. All of a completion's tests run in one child, so a test that hangs fails them all. The child doesn't get the trainer's environment variables or `PYTHONPATH`.
- Reward weights are applied once. These rewards multiplied their score by `weight` and the trainer then applied it again, so a run with weight `w` scored `w` squared: `math_reasoning`, `diversity`, `fluency`, `relevance`, `brevity`, `image_relevance` and `audio_quality` now return the unweighted score (a run with `weight` 0.2 sees scores 5 times larger than before; weight 1.0 changes nothing). `sentiment`, `safety` and `toxicity` weighted only `batch_compute`, so it differed from `compute`; it is now unweighted too. Code that calls these rewards directly and relied on the weight being applied inside them must multiply by `weight` itself.

### Fixed
- Counterfactual GRPO (TRL): a batch whose distinct completions all have no answer region, and which repeats one of them, no longer crashes with `'NoneType' object is not subscriptable` in `compute_soft_weights`. The repeated completions now get the same uniform importance as the first.
- Counterfactual GRPO code tasks (TRL) grade by value like the code rewards. Before, a completion that printed `PASS:0`, `PASS:1` and so on, or exited with status 0, could score full marks, and the MBPP+ output comparison ran inside the same process as the candidate, so an object with an `__eq__` that always returns True matched everything. The candidate's results now come back as plain data and are compared in the trainer, against the reference solution's results from its own child process.
- Data: chat lists stored as text, as in a CSV (`"[{'content': ..., 'role': ...}]"`), are parsed and flattened like real lists, so KTO and other trainers no longer train on the repr string.
- Data: `DataManager` accepts the SFT task names `supervised_fine_tuning` (the config default), `text_generation` and `chat_completion`; they failed with `is not a valid TaskType`.
- Data: SFT rows whose `prompt` CuratorKIT left blank take it from `instruction` (e.g. `column_mapping={"input": "prompt"}`), so user messages are no longer empty and PPO no longer fails on an empty prompt.
- Data: with `keep_columns`, a blank CuratorKIT field no longer erases the dataset's own column of the same name (the `input` prompt of `argilla/distilabel-intel-orca-dpo-pairs`), and null cells in `metadata` stay null, so Arrow no longer fails with `cannot mix list and non-list`.
- Unsloth SFT: `dataset.map` fingerprinting no longer pickles the model into host RAM (#33, @chayan-bit).
- `create_sft_trainer` / `create_rl_trainer`: values from `config` are no longer replaced by the function's own defaults. Explicit argument, then config, then default (#34, @chayan-bit).
- `create_sft_trainer` / `create_rl_trainer`: an explicit `backend=` argument is no longer replaced by the config's `backend`, nor by the `auto` filled in when the config has none (which picked Unsloth on a GPU machine). Same order: explicit argument, then config.
- YAML `train:` keys such as `gradient_checkpointing`, `optimizer` and `dataset_num_proc` are no longer dropped (#35, @chayan-bit).
- Merged models stay readable by transformers 4.x loaders: top-level `rope_theta` / `rope_scaling` / `torch_dtype` and legacy tokenizer fields are written, also with `keep_base_config=True` (#38, fixed in #39, @zeralyngkhoi19).
- Unsloth `SFTConfig` pickling at checkpoint time (#39, @zeralyngkhoi19).
- Math grading: a sentence answer such as `The answer is: 42.` drops the colon, full stop and closing paragraph around it (#42, @ANDROIDHASSAN).
- Math grading: markdown bold around a sentence answer (`**The answer is: 8.**`, `the answer is: **230.**`) is dropped too.
- GSM8K grading compares by value, so `$18`, `18%` and `3/4` grade correctly. A sign is stripped only at the ends, so `1$8` stays wrong (#43, @ANDROIDHASSAN).
- Counterfactual GRPO gold answers: MATH expressions such as `\frac{14}{3}` and a numeric zero stay as written, and a boxed answer in `solution` is used (#45, @Yuvraj27-5; reported in #44 by @ANDROIDHASSAN). On top: TRL reads the answer column it chose, one number in plain text (`72 eggs`) is the gold, and `#### 1/3` keeps the fraction.
- TRL counterfactual GRPO: a dataset named in capitals, such as `HuggingFaceH4/MATH-500`, is set up as a math dataset, as on Unsloth. Before, it went to the generic loader and failed with `KeyError: 'train'`.
- SGLang adapter sync: every training process generates with the synced adapter, not only the main one, and only the main process unloads it. Unsloth PACE raises for `rollout_backend="sglang"` instead of generating with transformers. Adapter tensors are dropped only for whole vision or audio module names (on top of #32 and #41).
- SGLang adapter sync with Unsloth 4-bit models, such as counterfactual GRPO by default: the pre-quantized `unsloth/<model>-bnb-4bit` base now matches a server running the full-precision `<model>`, instead of failing the base-model check.
- Code rewards read the tests a dataset ships: MBPP's `test_list` and HumanEval's `test` plus `entry_point`, as well as `test_cases`. A HumanEval completion that only writes the function body is graded as prompt plus completion. Before, every reward took its no-test branch on these datasets.
- `code_correctness` strips a Markdown fence, and a bare answer keeps the imports above its first `def`.
- `code_execution` without tests works on a worker thread; it used `signal.alarm`, which fails there.
- `code_correctness` and `code_quality` return unweighted scores. The trainer applies `weight`, so it was applied twice.
- `mbpp_reward` no longer leaves a temporary file behind on every call, and honours `params["timeout"]`.
- The `deepseek_humaneval` and `gbmpo_l2_code` recipes name the real dataset, `openai/openai_humaneval` (`openai/humaneval` returns 401).
- `examples/grpo_code_gen_trl_mbpp.py` trains with `mbpp_reward`, as it says, not `code_quality`.

### Security
- Code rewards (`code_execution`, `code_correctness`, `mbpp_reward`, `code_verifiable`) no longer run completions in the training process, and grade by value. Each call runs the code in a fresh `sys.executable -I` child with a temporary working directory, a minimal environment, a process group killed on timeout, CPU and file-size limits, and a memory limit on Linux. The child returns plain values and the trainer compares them, so printed `PASS` lines, exit status 0, `sys.exit`/`os._exit`, an always-equal `__eq__` and patching `math.isclose` no longer earn a pass, end the trainer or change later grades, and `while True` no longer hangs it. `code_correctness` scores 0 and warns when no tests arrive, instead of 0.5 for anything that compiles. The child is not a sandbox: it can still read the user's files, use the network and signal the trainer (see the reward functions guide). Reported by Ebin Babu Thomas (@ebt55).

## [0.1.10] — 2026-08-05

Initial public release.

### Added
- **Distillation framework**: Standard, GOLD (cross-tokenizer), SDFT, and SDPO (self-distillation) methods under a unified config.
- **ES backend rollout abstraction**: a standard `transformers`-based rollout backend and a vLLM-backed one for hyperscale generation.
- **GBMPO**: four divergence-specific implementations consolidated into a single configurable trainer (`divergence_type` field).
