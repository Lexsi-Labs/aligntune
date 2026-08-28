"""
Tests for aligntune.data.decontamination.

Tries the real package import first; falls back to a file-path shim when
the heavy aligntune.__init__ chain (vllm/torch/trl) is unavailable.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import: prefer real package path, fall back to direct file load
# ---------------------------------------------------------------------------
_DIRECT_IMPORT = False
try:
    from aligntune.data.decontamination import (
        DeconReport,
        clean_dataset,
        decontaminate,
        exact_jaccard,
    )

    import aligntune.data.decontamination as decon
except Exception:
    # Heavy optional deps (vllm/torch) unavailable -> load module directly
    _DIRECT_IMPORT = True
    REPO = Path(__file__).resolve().parent.parent
    SRC = REPO / "src"

    def _stub_pkg(name: str, path: str) -> None:
        if name in sys.modules:
            return
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        sys.modules[name] = mod

    def _load_mod(fqn: str, filepath: str) -> types.ModuleType:
        if fqn in sys.modules:
            return sys.modules[fqn]
        spec = importlib.util.spec_from_file_location(fqn, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    _stub_pkg("aligntune", str(SRC / "aligntune"))
    _stub_pkg("aligntune.data", str(SRC / "aligntune" / "data"))

    decon = _load_mod(
        "aligntune.data.decontamination",
        str(SRC / "aligntune" / "data" / "decontamination.py"),
    )

    decontaminate = decon.decontaminate
    clean_dataset = decon.clean_dataset
    exact_jaccard = decon.exact_jaccard
    DeconReport = decon.DeconReport

# Check if Rust backend is available
try:
    import aligntune_fast  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Marker for tests that need Rust
requires_rust = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="aligntune_fast not installed"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VOCAB = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()


def _make_docs(n: int, min_w: int = 15, max_w: int = 30, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(VOCAB, k=rng.randint(min_w, max_w))) for _ in range(n)]


# Shared small params for fast tests
SMALL = dict(num_perm=64, shingle=3, bands=8, rows=8, seed=7)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_rust
class TestBackendEquivalence:
    """Rust and Python backends produce identical results."""

    def test_backend_equivalence(self):
        docs = _make_docs(200, seed=100)
        train, eval_ = docs[:100], docs[100:]

        r_rust = decontaminate(train, eval_, backend="rust", **SMALL)
        r_py = decontaminate(train, eval_, backend="python", **SMALL)

        rust_matches = {(ti, ei) for ti, ei, _ in r_rust.matches}
        py_matches = {(ti, ei) for ti, ei, _ in r_py.matches}
        assert rust_matches == py_matches, (
            f"Match sets differ: rust_only={rust_matches - py_matches}, "
            f"py_only={py_matches - rust_matches}"
        )

        # Spot-check a few signature values via the internal helpers
        coeff_a, coeff_b = decon._make_coeffs(SMALL["num_perm"], SMALL["seed"])
        rust_sigs = decon._compute_signatures(train[:5], coeff_a, coeff_b, SMALL["shingle"], "rust")
        py_sigs = decon._compute_signatures(train[:5], coeff_a, coeff_b, SMALL["shingle"], "python")
        for i in range(5):
            assert list(rust_sigs[i]) == list(py_sigs[i]), f"Signature mismatch at doc {i}"


class TestDetectsExactDuplicate:
    def test_detects_exact_duplicate(self):
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
        train = [shared, "totally different words here nothing in common at all really truly unique"]
        eval_ = [shared]

        report = decontaminate(train, eval_, threshold=0.8, backend="python", **SMALL)
        assert 0 in report.contaminated_train_indices
        assert any(ti == 0 and ei == 0 for ti, ei, _ in report.matches)


class TestNoMatchOnDisjoint:
    def test_no_match_on_disjoint(self):
        # Two corpora with completely non-overlapping vocabularies
        train = [f"aaa{i} bbb{i} ccc{i} ddd{i} eee{i} fff{i} ggg{i} hhh{i} iii{i} jjj{i} kkk{i} lll{i} mmm{i} nnn{i} ooo{i}" for i in range(50)]
        eval_ = [f"xxx{i} yyy{i} zzz{i} www{i} vvv{i} uuu{i} ttt{i} sss{i} rrr{i} qqq{i} ppp{i} ooo{i} nnn{i} mmm{i} lll{i}" for i in range(50, 100)]

        report = decontaminate(train, eval_, backend="python", **SMALL)
        assert len(report.matches) == 0
        assert len(report.contaminated_train_indices) == 0


class TestShortDocsNoCrash:
    def test_short_docs_no_crash(self):
        # Docs shorter than shingle=3 should get all-0xFFFFFFFF sigs
        train = ["a", "ab", ""]
        eval_ = ["x", "xy", ""]

        report = decontaminate(train, eval_, backend="python", **SMALL)
        # Should not crash, and no matches (all signatures are identical
        # u32::MAX vectors, but that's a degenerate case — they might match
        # each other via LSH, but exact Jaccard of empty shingle sets is 0)
        # Just verify no exception
        assert isinstance(report, DeconReport)
        assert report.n_train == 3
        assert report.n_eval == 3


class TestEmptyInputs:
    def test_empty_train(self):
        report = decontaminate([], ["some doc here with enough words for shingles"], backend="python", **SMALL)
        assert report.n_train == 0
        assert report.n_eval == 1
        assert len(report.matches) == 0

    def test_empty_eval(self):
        report = decontaminate(["some doc here with enough words for shingles"], [], backend="python", **SMALL)
        assert report.n_train == 1
        assert report.n_eval == 0
        assert len(report.matches) == 0

    def test_both_empty(self):
        report = decontaminate([], [], backend="python", **SMALL)
        assert report.n_train == 0
        assert report.n_eval == 0
        assert len(report.matches) == 0


class TestVerifyImprovesPrecision:
    def test_verify_improves_precision(self):
        rng = random.Random(55)
        eval_docs = _make_docs(100, seed=200)

        # Plant contaminated docs at various Jaccard levels
        train_docs: list[str] = []
        planted: list[dict] = []  # {train_idx, eval_idx, true_jac}

        # Add some clean docs first
        train_docs.extend(_make_docs(50, seed=300))

        # Plant pairs: 10 each at target Jaccard ~0.65, ~0.75, ~0.85, ~0.95
        for target in [0.65, 0.75, 0.85, 0.95]:
            for _ in range(10):
                eval_idx = rng.randint(0, len(eval_docs) - 1)
                source = eval_docs[eval_idx]
                words = source.split()
                n = len(words)
                # Replace fraction of words to hit target
                # Higher replacement -> lower Jaccard
                k = int(n * (1 - target) * 1.3)  # rough approximation
                k = max(0, min(k, n - 1))
                w = list(words)
                for pos in rng.sample(range(n), k):
                    w[pos] = rng.choice(VOCAB)
                contam = " ".join(w)
                train_idx = len(train_docs)
                train_docs.append(contam)
                tj = exact_jaccard(contam, source, shingle=SMALL["shingle"])
                planted.append({"train_idx": train_idx, "eval_idx": eval_idx, "true_jac": tj})

        # Without verify
        r_no = decontaminate(
            train_docs, eval_docs, threshold=0.8,
            verify=False, backend="python", **SMALL,
        )
        assert not r_no.verified

        # With verify
        r_yes = decontaminate(
            train_docs, eval_docs, threshold=0.8,
            verify=True, backend="python", **SMALL,
        )
        assert r_yes.verified

        # verify=True: every match must have exact Jaccard >= threshold
        for ti, ei, jac_val in r_yes.matches:
            ej = exact_jaccard(train_docs[ti], eval_docs[ei], shingle=SMALL["shingle"])
            assert ej >= 0.8, f"Verified match ({ti},{ei}) has exact Jaccard {ej:.3f} < 0.8"

        # Recall on planted >=0.8 pairs should not drop more than 1
        planted_08 = {(p["train_idx"], p["eval_idx"]) for p in planted if p["true_jac"] >= 0.8}
        if planted_08:
            caught_no = sum(1 for ti, ei, _ in r_no.matches if (ti, ei) in planted_08)
            caught_yes = sum(1 for ti, ei, _ in r_yes.matches if (ti, ei) in planted_08)
            # Verify should not lose more than 1 true positive vs unverified
            assert caught_yes >= caught_no - 1, (
                f"Verify lost too many: {caught_no} -> {caught_yes}"
            )


class TestCleanDataset:
    def test_clean_dataset_removes_contaminated(self):
        shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
        train = [
            "unique doc one with many words for shingle hashing test",
            shared,
            "another unique doc completely different from the rest here",
            shared,
        ]
        eval_ = [shared]

        kept, report = clean_dataset(train, eval_, threshold=0.8, backend="python", **SMALL)

        # Contaminated indices (1 and 3) should be removed
        assert 1 in report.contaminated_train_indices
        assert 3 in report.contaminated_train_indices

        # Kept docs preserve order of non-contaminated
        assert len(kept) == 2
        assert kept[0] == train[0]
        assert kept[1] == train[2]


class TestBandsRowsValidation:
    def test_bands_rows_mismatch(self):
        with pytest.raises(ValueError, match="bands.*rows.*num_perm"):
            decontaminate(
                ["doc"], ["doc"],
                num_perm=64, bands=10, rows=8,
                backend="python",
            )

    def test_bands_rows_valid(self):
        # Should not raise
        report = decontaminate(
            ["doc one two three four five six seven eight nine ten"],
            ["doc one two three four five six seven eight nine ten"],
            num_perm=64, shingle=3, bands=8, rows=8,
            backend="python",
        )
        assert isinstance(report, DeconReport)


class TestLazyExport:
    def test_lazy_export_resolves_without_heavy_init(self):
        """Load data/__init__.py in isolation and verify PEP 562 __getattr__ works."""
        REPO_ = Path(__file__).resolve().parent.parent
        SRC_ = REPO_ / "src"

        pkg = types.ModuleType("aligntune")
        pkg.__path__ = [str(SRC_ / "aligntune")]
        sys.modules.setdefault("aligntune", pkg)

        spec = importlib.util.spec_from_file_location(
            "aligntune.data", SRC_ / "aligntune" / "data" / "__init__.py",
            submodule_search_locations=[str(SRC_ / "aligntune" / "data")],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["aligntune.data"] = mod
        spec.loader.exec_module(mod)

        assert callable(mod.decontaminate)
        assert callable(mod.clean_dataset)
        assert mod.DeconReport is not None
