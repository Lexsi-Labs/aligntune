# Vendored unsloth

Source: https://github.com/unslothai/unsloth, PyPI release `2026.7.2`
(`unsloth-2026.7.2.tar.gz` / `unsloth-2026.7.2-py3-none-any.whl`).

## Why vendored instead of installed from PyPI

Unsloth's published metadata caps `transformers<=5.5.0` and `trl<=0.24.0`,
while aligntune pins `transformers==5.14.1` and `trl==1.7.1`. Installing it
normally (or via a separate `pip install --no-deps unsloth==...` step, the
previous approach — see git history / CHANGELOG) works today, but requires
a second install command. Vendoring it here — the same pattern already used
for `mergekit` (see `third_party/mergekit/PATCH_NOTES.md`) — folds it into
aligntune's own wheel so `pip install aligntune` alone is enough.

Only the actual importable `unsloth` package is vendored (2.7MB). The
upstream sdist/wheel also ships a `studio/` web UI (~110MB, AGPL-3.0-only,
unrelated to training) and `tests/`/`images/`/dev `scripts/` — none of that
is included here.

## Licensing

Unsloth is dual-licensed: most of the package is Apache-2.0, but two
subtrees are explicitly **AGPL-3.0**, per their own file headers and a
`LICENSE` file in each directory:

- `unsloth/kernels/moe/` (incl. `unsloth/kernels/moe/grouped_gemm/`) — MoE
  grouped-GEMM kernels. Only referenced by `unsloth/models/glm4_moe.py`
  (GLM-4-MoE support). aligntune's own MoE backend
  (`aligntune.backends.moe`) is not yet wired into the Backend Factory or
  CLI, so nothing in aligntune currently exercises this path.
- `unsloth/utils/prefix_grouper.py` and `prefix_grouper_kernel.py` — a
  packed-GRPO attention optimization. Only pulled in via function-local
  (not top-level) imports inside `unsloth/utils/attention_dispatch.py`.
- One function inside `unsloth/models/rl_replacements.py`
  (`grpo_trainer__get_per_token_logps_and_entropies`'s inner replacement)
  carries an inline `# All Unsloth code here in this function is licensed
  under AGPL3` comment, in an otherwise Apache-2.0 file.

**This vendored copy includes those AGPL-3.0 files as-is (nothing
stripped).** This was flagged explicitly during vendoring and reviewed and
cleared by Legal. If that determination ever needs to be revisited, the
fix is either:

1. Remove `unsloth/kernels/moe/`, `unsloth/utils/prefix_grouper*.py`, and
   the flagged function in `rl_replacements.py` (confirmed removable
   without breaking imports — MoE/prefix-grouping become unavailable), and
   revert to the separate `pip install --no-deps unsloth==...` step for the
   parts that can't be cleanly vendored (see `unsloth_zoo/PATCH_NOTES.md` —
   its AGPL code is *not* cleanly separable the way unsloth core's is), or
2. Drop this vendoring entirely and go back to the two-command install
   (`pip install aligntune` + `pip install --no-deps unsloth==...` /
   `aligntune install-unsloth`).

## Patches applied

1. **`unsloth/_gpu_init.py`** — upstream checks
   `importlib.metadata.version("unsloth_zoo")` to guard against a stale
   `unsloth_zoo` install, and raises `ImportError: Please install
   unsloth_zoo via pip install unsloth_zoo` if that lookup fails. It assumes
   `unsloth_zoo` is always its own separately pip-installed distribution
   with its own dist-info; here it's vendored as part of aligntune's own
   distribution instead, so there's no distribution literally named
   `unsloth_zoo` for `importlib.metadata` to find, and the check always
   fails even though the code is present and importable. Replaced the
   version-checked `import unsloth_zoo` with a plain one — safe because both
   are vendored from the same upstream release (`2026.7.2`) in lockstep, so
   the version-skew this check guards against can't happen here. If
   `unsloth`/`unsloth_zoo` are ever bumped independently, re-add an
   equivalent check.

Verified end-to-end: `from unsloth import FastLanguageModel` succeeds from
this vendored tree (via `third_party/unsloth` + `third_party/unsloth_zoo` on
`sys.path`, same layout `setup.py` wires up), alongside this project's
pinned `transformers==5.14.1`, `trl==1.7.1`.

## Upgrading unsloth later

Re-download the target version's sdist from PyPI, diff its `unsloth/`
directory against this one, and re-check the AGPL file list above (upstream
may move code between Apache/AGPL subtrees between releases).
