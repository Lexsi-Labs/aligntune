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

### Fixed
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

### Changed
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
