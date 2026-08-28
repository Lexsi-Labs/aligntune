# Vendoring notes — tokenizer-extension

- **Upstream:** https://github.com/taidopurason/tokenizer-extension
- **License:** Apache-2.0 (see `LICENSE`)
- **Why vendored:** upstream is not published on PyPI, so depending on it via
  `git+https://…` made `aligntune` impossible to upload to PyPI (PyPI rejects
  direct-URL dependencies). Vendoring it in-tree lets a plain
  `pip install aligntune` work.

## Modifications

**None.** The `tokenizer_extension/` package is a verbatim copy of upstream
`main`. Only the parts not needed at runtime were left out of this copy:
`experiments/`, `scripts/`, and the example notebook.

## Wiring

`setup.py` adds `tokenizer_extension` (and its `pruning` subpackage) to the
`aligntune` wheel via `package_dir`. Its runtime dependencies
(`transformers`, `tokenizers`, `datasets`, `tqdm`, `requests`, `heapdict`) are
folded into `pyproject.toml`. `dask` is only used by `tokenizer_extension/data.py`,
which nothing in `aligntune` imports, so it stays optional (as upstream has it).
