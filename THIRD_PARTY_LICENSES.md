# Third-Party Licenses

AlignTune is distributed under the Lexsi Labs Source Available License (LSAL)
v1.1 (see [LICENSE.md](LICENSE.md)). It vendors three third-party projects directly
and depends on several others at install/runtime. Each remains under its own
license; the ones with non-permissive or otherwise notable terms are detailed
below.

---

## 1. mergekit (vendored, patched)

- **Upstream:** https://github.com/arcee-ai/mergekit
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0) — see
  `third_party/mergekit/LICENSE` for the full text.
- **Vendored at:** `third_party/mergekit/`, built into the same wheel as
  `aligntune` (see `setup.py`'s package discovery and the console-script
  entries in `pyproject.toml`).
- **Modifications:** two small compatibility patches (a `Task` pydantic-schema
  fix and an `Any`-typing workaround for `PretrainedConfig`) for pydantic
  2.11+/transformers 5.x, documented in `third_party/mergekit/PATCH_NOTES.md`.
  Re-verify these patches against upstream before bumping the vendored
  version.
- mergekit's own `CLA.md` and `CONTRIBUTING.md` (upstream's, not AlignTune's)
  ship alongside it under `third_party/mergekit/`.

Because mergekit is compiled into the AlignTune wheel rather than kept as a
separate installed distribution, the combined distribution includes LGPL-3.0
code alongside AlignTune's own LSAL-1.1 code. LGPL-3.0 permits this as long as
the LGPL-covered portion remains available under its own license (satisfied
here — the vendored copy keeps its own `LICENSE`) and users retain the ability
to obtain, inspect, and relink a modified version of the LGPL component.

---

## 2. unsloth (vendored, patched)

- **Upstream:** https://github.com/unslothai/unsloth
- **License:** Apache-2.0, except `unsloth/kernels/moe/`, `unsloth/utils/prefix_grouper.py` + `prefix_grouper_kernel.py`, and one function in `unsloth/models/rl_replacements.py`, which are AGPL-3.0.
- **Vendored at:** `third_party/unsloth/`.

---

## 3. unsloth_zoo (vendored, patched)

- **Upstream:** https://github.com/unslothai/unsloth-zoo
- **License:** Apache-2.0, except `fused_losses/`, `mlx/`, `stubs/`, the MoE-related files under `temporary_patches/`, `model_lists.py`, `gated_delta_vjp.py`, `hf_cache_state.py`, `hf_xet_fallback.py`, and `pad_token.py`, which are AGPL-3.0.
- **Vendored at:** `third_party/unsloth_zoo/`.

---

## 4. Runtime dependencies (not vendored)

These are installed via `pip`/`uv` as separate distributions, not copied into
this repository. Most (transformers, trl, peft, accelerate, torch, datasets,
bitsandbytes, torchao, etc.) are under standard permissive licenses
(Apache-2.0, BSD, or MIT) and are not called out individually — see
`pyproject.toml` / `requirements.txt` for the full list. The following carry
terms worth knowing about specifically:

- **CuratorKIT** — [Lexsi Labs Source Available License (LSAL) v1.1](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/LICENSE.md)
  — https://github.com/Lexsi-Labs/CuratorKIT
  Installed automatically as a hard dependency (`curatorkit[connectors]`, see
  `pyproject.toml`) for AlignTune's data-loading/curation pipeline. Same
  noncommercial-without-a-separate-license terms as AlignTune itself (Section
  2 of `LICENSE.md`) — using CuratorKIT through AlignTune does not add a
  *new* restriction on top of AlignTune's own license, but it is a distinct
  legal work with its own license file, so it's listed here explicitly rather
  than assumed.

- **tokenizer-extension** — Apache-2.0 —
  https://github.com/taidopurason/tokenizer-extension
  Installed automatically as a hard dependency (research code from Purason et
  al., 2025, used for Indic/multilingual tokenizer vocabulary extension).

See each project's repository for full license text. This file covers
license terms only and is not legal advice; contact **support@lexsi.ai** with
questions about combining AlignTune with any of the above in a commercial
product.
