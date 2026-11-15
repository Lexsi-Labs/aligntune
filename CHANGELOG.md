# Changelog

All notable changes to AlignTune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full changelog.

## [Unreleased]

### Added
- TRL SFT can resume an interrupted run: `create_sft_trainer(..., resume_from_checkpoint="output/checkpoint-500")`, or `True` for the latest checkpoint in `output_dir` (also `train.resume_from_checkpoint` in a config), is passed to `Trainer.train()`, which restores the weights, optimizer, scheduler and step count. A missing folder, or `True` with no checkpoint, stops with a `ValueError` before the model loads. The default `None` keeps the current behaviour. From the command line, `aligntune train --resume latest` (or a checkpoint folder) does the same, and `python -m aligntune.cli.finetune --resume` now reaches it too. The Unsloth SFT trainers and the classification tasks do not support it yet and raise a `ValueError`, from `create_sft_trainer` and from `BackendFactory.create_trainer`, instead of starting again from step 0 (#74, @W-OK-E).

### Fixed
- SFT runs from a config file on the TRL backend no longer import Unsloth. Config files default `chat_template` to `"auto"`, which the model loader treated as an Unsloth template name, so it imported `unsloth.chat_templates`, and that replaced TRL's `SFTTrainer` (on macOS with MLX installed, training then stopped with `NotImplementedError: Unsloth MLX: callbacks requires ...`). `"auto"` now keeps the tokenizer's own template on every backend; it was never an Unsloth template name, so nothing was applied before either. A named Unsloth template such as `"chatml"` is applied on the Unsloth backend only; on TRL it logs a warning and keeps the tokenizer's template. A `chat_template` set on a dataset in a config file is no longer replaced by the top-level default.
- `python -m aligntune.cli.finetune` always failed, for every `--method`: building its config stopped with `TypeError: DistributedConfig.__init__() got an unexpected keyword argument 'num_gpus'`, and that config was then passed to `create_sft_trainer` / `create_rl_trainer` as the model name. It now passes its options to the factories as keyword arguments (`--rewards length:1.0` becomes a reward entry; `--rl-loss-type` and `--rl-loss-weight` apply to DPO only, so GRPO and PPO keep their own loss). `--num-gpus` and `--distributed` are not used by the factories; their help now says so (launch with accelerate or torchrun for multi-GPU).
- **Code rewards: the child that runs model-written code is confined more tightly on Linux.** Landlock used to let it read
  all of `/proc`, `/dev`, `/sys`, `/etc`, `/opt` and every `sys.path` entry, so a completion could read other processes'
  `/proc` entries, the trainer's command line among them. Reads are now limited to the interpreter's prefixes and library
  paths, the system library directories, the child's own `/proc/<pid>` and a short list of files (`/dev/urandom`,
  `/proc/cpuinfo`, `/etc/ld.so.cache` and the like); the user's home, a repository checkout and other `sys.path` entries are
  refused too. The seccomp filter also refuses calls that act on other processes or on objects shared with them
  (`prlimit64` and the `sched_set*` calls on any process but the caller, `setpriority`, `ioprio_set`, `migrate_pages`,
  `move_pages`, `process_madvise`, `perf_event_open`, `kcmp`, SysV IPC and POSIX message queues, namespace, mount and
  file-handle calls, `bpf`, `userfaultfd`, keyrings, `F_SETOWN`/`FIOSETOWN`) and the file changes Landlock does not cover
  (`chown`, extended-attribute writes, `truncate` by path; `chmod` and timestamps are still allowed). The child drops every
  capability before the completion runs. Each child, in both places that start one (asserts and input/output tests, and
  HumanEval `check()` programs), reports the isolation it got over a pipe it closes before the completion runs, and the
  trainer decides on that report instead of on a probe or on a reply the completion could write. The trainer marks
  itself non-dumpable while a child is alive, and where seccomp is missing it also kills the child's leftover processes,
  found by a marker in their environment. Replies from a HumanEval child are read with `selectors`, so a trainer with
  more than 1024 open files no longer fails to grade.
- **Results change for code rewards on Linux without seccomp.** A child that reports no seccomp is refused: the completion
  is not run, scores 0, and one error names the reason. `ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP=1` restores the best-effort
  behaviour. `ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1` (or `true`, `yes`, `full`) now needs seccomp and Landlock both and the
  capabilities gone; `=any` needs one of the two, which is what `1` meant before. macOS and Windows have no seccomp and keep
  the process limits and a warning.
- Code rewards no longer score correct answers 0 or partly when the answer returns a `collections.Counter`, an
  `OrderedDict`, a `defaultdict` or a complex number (these now cross as plain data and are rebuilt in the trainer), or
  an object that a test only checks for truth (`assert f(x)` on an `re.Match`: only its truth value crosses, and it
  equals nothing), or when the task's function shares a builtin's name (MBPP's `sum(a, b)`). For the last, the completion's
  definition is used only as the function under test (the whole assert, or one whole side of a single comparison, named
  nowhere else in the assert, in an assert that names nothing else of the completion's); a builtin the completion redefines anywhere
  else, on the expected side or around its answer, still fails the test.
- `mbpp_reward`, and the code reward of the counterfactual GRPO trainer, find the function under test from the assert's
  syntax tree. They took the first name before a `(`, so `assert set(f(x)) == ...` named the builtin `set` (and
  `assert f (x) == ...` named nothing in `mbpp_reward`), and correct answers scored below 0.
- **Results change for `mbpp_reward`: scores are clipped to [0, 1]** (they were [-0.5, 1]), last, after the tie-breaker. A
  wrong answer, a crash, a timeout and a row with no tests all score 0, and the format penalties and the brevity bonus only move passing
  answers (a correct answer in a code fence still scores 0.98).
- Text and token classification keep the class names: with text labels, the trainers (including `ClassificationTrainer` in `Classification_trainer.py`, for TRL and Unsloth) save `model.config.id2label` and `label2id` (padded to the model's number of outputs with `LABEL_i`, so `num_labels` and the saved model are unchanged), and a saved model's predictions read back as names. LoRA runs save only the adapter, so there the names stay on the in-memory model. Numeric labels (text "2" and "10", or floats such as `10.0`) get ids in numeric order, not text order; if any label is not a number, all labels sort as text. Labels that need more classes than the model has outputs stop with a clear `ValueError` before training, in `ClassificationTrainer` too.
- Text and token classification train on transformers 5: the trainers passed `tokenizer=` to `Trainer`, which transformers 5 removed, so every run stopped with a `TypeError`. They pass `processing_class=`.
- Text classification with text labels ("positive", or "0"/"1") no longer raises a `TypeError`: `Classification_trainer.py` encodes labels with `ClassificationLabelEncoder` over the whole dataset. `sft_classification.py` no longer shifts already-encoded labels again for a batch that has no 0, which put those rows in the wrong class.
- Multi-turn `messages` SFT data trains on the conversation's last assistant turn: `prompt` and `completion` came from the first exchange, and `instruction_following` dropped `messages`, so it trained on the first reply. `prompt` is now the last user turn, `completion` the last assistant turn, and `messages` is kept for `instruction_following` as for `sft` (#64).
- Hugging Face Hub uploads work again: `HFHubExporter` created the repository with `private=` passed to `upload_file` and `upload_folder`, which never accepted it (a `TypeError`), so it now sets the visibility when it creates the repository. `setup_hf_auth` no longer passes `write_permission` to `login`, which does not take it (the error was swallowed and logged as a failed login). `aligntune recipes run` checks the login with `huggingface_hub.whoami()` instead of the `huggingface-cli` command, which huggingface_hub 2.x no longer installs, and messages say `hf auth login`.
- `HonestyReward` matches dishonesty terms at a word start, so "lie" no longer fires inside "believe" (#55, @ANDROIDHASSAN).
- `FluencyReward` matches transitions and connectors as whole words: "or" is not in "words" (#56, @ANDROIDHASSAN).
- `ConcisenessReward` counts a filler next to punctuation ("Basically,") and the two-word "you know" (#57, @ANDROIDHASSAN).
- `ReadabilityReward` caps the sentence and word scores at 1, so a very short sentence cannot hide long words or punctuation (#58, @ANDROIDHASSAN).
- `TemporalConsistencyReward` matches temporal pairs and time words as whole words: "old" is not in "told" (#59, @ANDROIDHASSAN).
- `MathReasoningReward` matches step words and units as whole words ("the" holds no unit "h"), and recognises minutes, feet, yards, km, cm, mm, ml and mg (#60, @ANDROIDHASSAN).
- `aligntune info` and the dependency checks report the TRL status from the live check; `rollout_backend="vllm"` without `ALIGNTUNE_ENABLE_VLLM=1` raises a clear error; a registry reward that fails with Unsloth's `apply_qkv` error logs a hint to use `backend="trl"` (#61, @N0VA06).

## [0.1.31] — 2026-11-05

### Changed
- **Lighter core install.** `vllm` (`trl[vllm]`), `lm-eval`, `wandb`, `PyICU`, `xformers` and `diffusers` are no longer installed by `pip install aligntune`; on macOS the install failed on vLLM, and on Kaggle or Colab vLLM could replace the preinstalled torch. They are extras: `aligntune[vllm]`, `aligntune[eval]`, `aligntune[wandb]`, `aligntune[tokenizer]`, `aligntune[unsloth]`, or `aligntune[all]` for the old set. The package imports without them. torch is capped below 2.14 so a plain install resolves a torch and torchvision pair that Unsloth accepts. If you used vLLM generation, LM-eval tasks or W&B logging, add the extra (#37).
- `code_execution` and `code_verifiable` with no tests now give 0.0 (they used to pay any code that runs,
  a stub included) and warn once: pass `test_cases`, `reference`, MBPP's `test_list`, or HumanEval's
  `test` + `entry_point`.

### Fixed
- `CurriculumSampler.update_pass_rate()` keeps a rate seeded with `set_pass_rate()`; a pass no longer lowers it (#48, @ANDROIDHASSAN).
- `math_correctness` counts each equation once; a wrong equation such as `2 + 2 = 5` now scores 0.0, not 0.5 (#49, @ANDROIDHASSAN).
- `LogicalConsistencyReward` and `HallucinationReward` match contradiction pairs as whole words, so "impossible" or "know" alone no longer lowers the score (#50, @ANDROIDHASSAN).
- ES trainer: chat-message (list) prompts are rendered with the chat template instead of failing in the tokenizer (`text input must be of type str ...`).
- `counterfact_grpo` (TRL and Unsloth) no longer crashes with `TypeError: unhashable type: 'list'` when the dataset's prompts are chat-message lists.
- Unsloth Online DPO: prompts are cut to `max_prompt_length` before training, so a prompt plus its completion no longer overruns `max_seq_length` (`The size of tensor a (256) must match the size of tensor b (257)`).
- PPO (TRL and Unsloth): `save_model` no longer fails with `'PPOTrainer' object has no attribute 'is_distributed_loading_by_transformers'` on transformers 5.22 (works on 5.17 and later alike).
- TRL SFT: `model.device_map: cpu` now also trains on the CPU (`use_cpu=True`); the Trainer used to move the
  model to a GPU when one was present.
- Unsloth ORPO: prompts are cut to `max_prompt_length` and rows whose chosen or rejected completion has no
  token left are dropped (long prompts made the log-probabilities, then the weights, NaN).

## [0.1.30] — 2026-11-03 (includes the unpublished 0.1.29)

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


### Fixed
- HumanEval `check()` with a loop, a local variable or a multi-line assert is graded instead of counted as failed: the trainer runs `check` and forwards the candidate's calls to a confined child, so a correct solution to those tasks no longer scores 0.
- `SafeCodeExecutor` in its default restricted mode can import a short list of harmless standard modules (`typing`, `math`, `re`, `collections`, ...). Before, any import made a correct solution fail, including the `from typing import List` at the top of most HumanEval prompts. `os`, `sys` and `subprocess` are still refused.
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
### Security
- Code rewards (`code_execution`, `code_correctness`, `mbpp_reward`, `code_verifiable`) no longer run completions in the training process, and grade by value. Each call runs the code in a fresh `sys.executable -I` child with a temporary working directory, a minimal environment, a process group killed on timeout, CPU and file-size limits, and a memory limit on Linux. The child returns plain values and the trainer compares them, so printed `PASS` lines, exit status 0, `sys.exit`/`os._exit`, an always-equal `__eq__` and patching `math.isclose` no longer earn a pass, end the trainer or change later grades, and `while True` no longer hangs it. `code_correctness` scores 0 and warns when no tests arrive, instead of 0.5 for anything that compiles. The child is not a sandbox: it can still read the user's files, use the network and signal the trainer (see the reward functions guide). Reported by Ebin Babu Thomas (@ebt55).

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
