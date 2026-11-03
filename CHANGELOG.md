# Changelog

All notable changes to AlignTune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full changelog.

## [Unreleased]

### Security
- **Code rewards no longer run model-generated code in the trainer process.** `code_execution`,
  `code_correctness`, `mbpp_reward`, `code_verifiable` and the counterfactual-GRPO code reward now run
  each completion in a separate `sys.executable -I` process (temp working directory, minimal
  environment, CPU/memory/file-size limits, wall-clock timeout, process-group kill). Previously a
  completion could patch modules such as `math.isclose` for later grades, end a training run with
  status 0 via `sys.exit(0)`/`os._exit(0)`, or hang it with an infinite loop.
- **Tests are graded "values out".** The child process returns only the values the candidate's
  function produced, as plain data, and each assert / expected-output comparison runs in the
  trainer. Printing `PASS`/`PASS:i`, exiting with status 0, or returning an object whose `__eq__`
  is always true no longer passes a test.
- `sql_verifiable` opens `db_path` read-only with a time limit, compares result rows with the
  reference (a SQL query or literal rows), and returns 0 when a reference is given but
  `execute=False` (it used to pay any parseable query).
- Reported privately by Ebin Babu Thomas on 2 Nov 2026; the fix builds on the patch and regression
  tests he provided (MIT).

### Added
- SGLang rollout backend for GRPO (`rollout_backend="sglang"`): generation goes to a running SGLang server over HTTP, and weights sync once per optimizer step, as a LoRA adapter on a `--enable-lora` server or as streamed full weights (#32, @chayan-bit).
- SGLang rollout on Unsloth GRPO-family and counterfactual GRPO trainers, with Gemma 3 weight names, vision adapter tensors, regex LoRA `target_modules` and full sync for 4-bit models (#41, @zeralyngkhoi19).

### Changed
- SFT runs that don't set `max_steps` now train for exactly `num_epochs`. Before, the epochs were turned into steps with a floor of 10, so small datasets ran more steps than asked (#36, fixed in #39, @zeralyngkhoi19).
- `create_rl_trainer` now passes `rollout_backend` to the trainer, so `rollout_backend="vllm"` turns vLLM rollout on. Before, the argument was ignored and the trainer used `hf` (#32, @chayan-bit).
- `code_quality` and `code_correctness` no longer multiply by their weight inside `compute()`; the
  trainer applies the weight (it was applied twice).
- HumanEval recipe weights: `code_correctness` 0.7, `code_quality` 0.1, `code_execution` 0.2
  (were 0.5 / 0.3 / 0.2), so a stub cannot outscore a correct answer.
- `code_execution` with no tests still pays any code that runs and now warns once: pass tests.
- **The child confines itself on Linux** before running the candidate. A seccomp filter refuses
  `kill`/`tgkill`/`pidfd_send_signal`, `ptrace`, `socket`, `io_uring_setup`, `fork`/`clone` and
  `execve`, so a completion can no longer SIGKILL the trainer via `os.getppid()`. A Landlock ruleset
  limits writes to the child's temp directory and reads to the Python installation and system
  directories (not the user's home), and blocks TCP on Landlock ABI 4+. Set
  `ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1` to refuse to run candidate code where the kernel offers
  neither; otherwise a one-time warning is logged.
- `code_verifiable` now reads the row's tests (`test_cases`, MBPP `test_list` + `test_imports`,
  HumanEval `test` + `entry_point`). Through TRL it ignored them and paid 1.0 to any code that ran,
  a stub included.
- `SafeCodeExecutor` (pass@k evaluation) runs code in the same sandbox and grades values out; it
  used a forked `multiprocessing` child with the asserts executed beside the candidate.
- `code_verifiable`'s stdout check runs in the sandbox (it had no limits and left grandchildren
  running after a timeout); its `input`/`expected_output` pairs compare plain values, and `input`
  may be argument values (as in `code_correctness`) as well as a call expression.

### Fixed
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
- HumanEval recipes (`grpo/deepseek_humaneval.yaml`, `advanced/gbmpo_l2_code.yaml`) used the
  non-existent dataset `openai/humaneval`; they now use `openai/openai_humaneval`. Code rewards read
  HumanEval's `test` + `entry_point` and MBPP's `test_list` (+ `test_imports`/`test_setup_code`) as
  well as `test_cases`, and body-only HumanEval answers are graded with the prompt prepended.
- HumanEval's `check(candidate)` now runs in the trainer with `candidate` forwarded to the sandbox,
  so asserts inside loops, asserts using locals and multi-line asserts are graded. 9 of 164
  canonical solutions (HumanEval/1, 32, 38, 44, 50, 53, 87, 113, 151) scored below 1.0 before;
  all 164 now score 1.0.
- `examples/grpo_code_gen_trl_mbpp.py` said it used `mbpp_reward` but configured `code_quality`.
- `examples/dapo_trl__code.py` trained on MBPP with `code_quality` only (it never runs the code and
  scored comment-only answers above correct ones); it now uses `mbpp_reward`, and its printout says
  DAPO and lists the rewards actually configured. `code_syntax`, `code_completeness` and
  `code_quality` are documented as heuristics not to be used alone for code training.
- `code_correctness` now strips code fences (every fenced answer scored 0) and returns 0, with a
  warning, when no tests arrive (it gave 0.5 to anything that compiled, including the empty string).
- Bare-code extraction keeps imports above the first `def`.
- The interpreter is `sys.executable` (correct answers scored 0 when `python` was not on PATH).
- The no-test `code_execution` path no longer fails on worker threads (it used `signal.alarm`).
- `MBPPReward` no longer leaks a temp file on every call.

### Known limitations
- Kernel confinement is best effort: without seccomp/Landlock (non-Linux, old kernels, some
  containers) the child is a separate process with resource limits only. `RLIMIT_NPROC` has no
  effect when running as root (the seccomp filter still blocks `fork`/`clone` there).
- Threads cannot be started in candidate code (as before, via `RLIMIT_NPROC=0`). BLAS libraries are told to use one
  thread, so `numpy` imports and numpy return values (scalars, arrays) are graded as plain values.
- HumanEval helpers (`poly`, `encode_cyclic`, ...) are taken from the task prompt. When no prompt
  reaches the reward, tasks that need one score 0, rather than trusting the candidate's own copy.
  Body-only answers need the raw (untemplated) prompt to be completed.


## [0.1.10] — 2026-08-05

Initial public release.

### Added
- **Distillation framework**: Standard, GOLD (cross-tokenizer), SDFT, and SDPO (self-distillation) methods under a unified config.
- **ES backend rollout abstraction**: a standard `transformers`-based rollout backend and a vLLM-backed one for hyperscale generation.
- **GBMPO**: four divergence-specific implementations consolidated into a single configurable trainer (`divergence_type` field).
