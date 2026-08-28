# Vendored & patched mergekit

Source: https://github.com/arcee-ai/mergekit, tag `v0.1.4` (commit `7683563`).

Vendored (instead of installed from PyPI) because upstream 0.1.4 hard-pins
`pydantic~=2.10.6`, `safetensors~=0.5.2`, `accelerate~=1.6.0`, `click==8.2.1`,
`tqdm==4.67.1` — installing it normally downgrades those packages project-wide
and breaks `vllm`, `unsloth`, `mcp`, and `openai-harmony`, which need newer
versions. Installing this vendored copy with `--no-deps` (see
`scripts/install_mergekit.sh`) avoids touching any of that, but upstream's
actual code has two real incompatibilities with the newer `pydantic`/
`transformers` already in this project, fixed below.

## Patches applied

1. **`mergekit/graph.py`** — `Task` base class was missing
   `arbitrary_types_allowed=True`. Every merge-method task subclass
   (`MultislerpMergeTask`, `NearswapMergeTask`, etc.) has `torch.Tensor`
   fields; pydantic >=2.11 refuses to build a schema for those without this
   flag (raises `PydanticSchemaGenerationError`).

2. **`mergekit/architecture/base.py`** — `ConfiguredModuleArchitecture.config`
   and `ConfiguredModelArchitecture.config` were typed as
   `transformers.PretrainedConfig`. `transformers>=5.x` implements that as a
   dataclass with an unresolvable `torch` forward reference, which pydantic
   tries to auto-introspect even under `arbitrary_types_allowed=True` (raises
   `PydanticUndefinedAnnotation: name 'torch' is not defined`). Typed the
   field `Any` for pydantic's purposes only — the runtime value is still a
   real `PretrainedConfig` instance; only pydantic's schema builder ignores
   its internals now.

Verified end-to-end with a real SLERP merge via
`aligntune.core.backend_factory.merge_models(...)`, in-project, alongside
`transformers==5.14.1`, `torch==2.11.0`, `pydantic==2.13.4` (i.e. none of
mergekit's own pins).

## Upgrading mergekit later

If bumping to a newer upstream mergekit release, re-check whether these two
issues still reproduce (`pip install --no-deps <new-version>`, run a real
merge) before reapplying the patches — a newer release may have already
fixed them upstream, or introduced new incompatibilities with whatever
`pydantic`/`transformers` versions are current in this project at the time.
