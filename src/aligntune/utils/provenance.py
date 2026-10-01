"""``lexsi_provenance.json``: the lineage record shared across the Lexsi stack.

Every output directory AlignTune writes (adapter, full model, merged model,
checkpoint) gets one. When an input (dataset folder, base model folder,
adapter) carries its own ``lexsi_provenance.json``, that object is embedded
under ``inputs[].provenance`` so lineage chains CuratorKIT -> AlignTune.
AuditKIT does not read this file yet; the link to it is one-way.
"""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROVENANCE_FILE = "lexsi_provenance.json"
PROVENANCE_SCHEMA = "lexsi.provenance/1"


def read_provenance(ref: Any, levels: int = 2) -> Optional[Dict[str, Any]]:
    """Provenance of a local dir, or of the nearest of a file's ``levels`` parent dirs.

    A dataset file such as ``export/train/dpo.jsonl`` resolves to
    ``export/lexsi_provenance.json``. Hub ids, in-memory objects and folders
    without the file give ``None``.
    """
    if not isinstance(ref, (str, Path)) or not Path(ref).exists():
        return None
    path = Path(ref)
    for folder in [path] if path.is_dir() else list(path.parents)[:levels]:
        try:
            return json.loads((folder / PROVENANCE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _portable_ref(ref: Any) -> Optional[str]:
    """Absolute path for refs that exist locally; Hub ids and other refs stay as given.

    A relative local path such as ``../base`` is meaningless once the output
    directory moves, so it is resolved at write time.
    """
    if ref is None:
        return None
    text = str(ref)
    try:
        path = Path(text)
        if path.exists():
            return str(path.resolve())
    except OSError:
        pass
    return text


def provenance_input(ref: Any, kind: str, config: Optional[str] = None) -> Dict[str, Any]:
    """One ``inputs[]`` entry, embedding the input's own provenance when present."""
    return {
        "kind": kind,
        "ref": _portable_ref(ref) if isinstance(ref, (str, Path)) else type(ref).__name__,
        "config": config,
        "provenance": read_provenance(ref),
    }


def build_provenance(
    method: Optional[str],
    base_model: Optional[str] = None,
    inputs: Iterable[Dict[str, Any]] = (),
    params: Optional[Dict[str, Any]] = None,
    library: str = "aligntune",
) -> Dict[str, Any]:
    """The ``lexsi.provenance/1`` object."""
    try:
        lib_version = version(library)
    except PackageNotFoundError:
        lib_version = None
    return {
        "schema": PROVENANCE_SCHEMA,
        "library": library,
        "version": lib_version,
        "git_sha": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": _portable_ref(base_model),
        "method": method,
        "inputs": list(inputs),
        "params": params or {},
    }


def write_provenance(output_dir: Any, provenance: Dict[str, Any]) -> Path:
    """Write ``provenance`` to ``output_dir/lexsi_provenance.json``."""
    path = Path(output_dir) / PROVENANCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return path
