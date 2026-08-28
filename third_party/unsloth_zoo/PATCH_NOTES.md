# Vendored unsloth_zoo

Source: https://github.com/unslothai/unsloth-zoo (PyPI project `unsloth_zoo`),
release `2026.7.2` (`unsloth_zoo-2026.7.2.tar.gz`).

## Why vendored instead of installed from PyPI

`unsloth` requires `unsloth_zoo>=2026.7.2`; it's vendored here for the same
reason as `unsloth` itself (see `third_party/unsloth/PATCH_NOTES.md`) — so
`pip install aligntune` alone is enough, no separate `--no-deps` step.

Only the importable `unsloth_zoo` package is vendored (4.0MB) — not
`tests/`, dev `scripts/`, or build artifacts.

## Licensing

Unlike `unsloth` core, the AGPL-3.0 code in `unsloth_zoo` is **not cleanly
separable**: 31 of 134 `.py` files carry AGPL-3.0 headers, and at least two
of them are imported unconditionally at the top of `unsloth_zoo/__init__.py`
(`.mlx.runtime.is_mlx_available` and `.model_lists.FORCE_FLOAT32`) — meaning
`import unsloth_zoo` fails outright if those files are simply deleted.
Confirmed by actually trying it (stripped copy raised
`ModuleNotFoundError: No module named 'unsloth_zoo.mlx'`, and after stubbing
that, an unrelated `ImportError: Please install Unsloth via
\`pip install unsloth\`!` from deeper init logic — `unsloth_zoo` also can't
be imported standalone without the real `unsloth` package present).

AGPL-3.0 files found (via `grep -rl "Affero General Public License"
unsloth_zoo --include="*.py"`):

- `fused_losses/` (whole subtree — fused loss kernels)
- `mlx/` (whole subtree — Apple MLX backend; imported unconditionally at
  `unsloth_zoo/__init__.py` top level)
- `stubs/` (bitsandbytes/triton fallback stubs)
- `temporary_patches/*moe*` (9 files — MoE model patches)
- `gated_delta_vjp.py`, `hf_cache_state.py`, `hf_xet_fallback.py`,
  `pad_token.py`
- `model_lists.py` — imported unconditionally at `unsloth_zoo/__init__.py`
  top level; provides `FORCE_FLOAT32`, a set of model architectures that
  overflow fp16 and must load in bf16/fp32. Removing this without a
  replacement is a real correctness regression (silent NaN/Inf at training
  time for those architectures), not just a feature loss.

**This vendored copy includes all of the above as-is (nothing stripped).**
Same MIT-vs-AGPL incompatibility described in
`third_party/unsloth/PATCH_NOTES.md` applies here, more severely, since the
AGPL code can't be cleanly carved out without either accepting the
correctness regression above or writing substantial replacement logic
(itself legally uncertain, since a rewrite this close to the original is
still arguably derivative).

**If this determination ever needs to be revisited, `unsloth_zoo` cannot be
partially fixed the way `unsloth` core can** — the realistic fallback is
dropping vendoring entirely and going back to installing `unsloth`/
`unsloth_zoo` from PyPI with `--no-deps` (either as a manual README step or
via an `aligntune install-unsloth` helper command).

## Upgrading unsloth_zoo later

Re-download the target version's sdist, diff against this copy, and re-run
the AGPL grep above — the file list may change between releases.
