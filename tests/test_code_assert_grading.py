"""The trainer decides an assert's boolean structure, and builtins can't be changed by the candidate."""

import pytest

from aligntune.rewards import _code_grader


def passes(code, test):
    status, passed = _code_grader.grade(code, [test], timeout=10)
    assert status == "ok"
    return passed[0]


THREE = "def f(): return [1, 2, 3]\n"

CODE = (
    "def f(n): return n if n == 1 else 99 if n == 3 else 0\n"
    "def two(): return 2\n"
    "import types\n"
    "class M(types.ModuleType):\n    def __eq__(self, o): return True\n"
    "m = M('m')\n"
    "class Always:\n    def __eq__(self, o): return True\n"
    "def obj(): return Always()\n"
)


@pytest.mark.parametrize(
    "extra, test",
    [
        ("def len(x): return 5\n", "assert len(f()) == 5"),
        ("def abs(x): return 5\n", "assert abs(f()[0]) == 5"),
        ("def isinstance(a, b): return True\n", "assert isinstance(f(), tuple)"),
        ("def sorted(x, **k): return [1, 2, 3, 4, 5]\n", "assert sorted(f()) == [1, 2, 3, 4, 5]"),
        ("import builtins\nbuiltins.len = lambda x: 5\n", "assert len(f()) == 5"),
        ("import builtins\nbuiltins.isinstance = lambda a, b: True\n", "assert isinstance(f(), tuple)"),
    ],
)
def test_changed_builtins_do_not_change_the_test(extra, test):
    assert not passes(THREE + extra, test)


def test_correct_answer_still_passes_with_patched_builtins():
    assert passes(THREE + "import builtins\nbuiltins.len = lambda x: 5\n", "assert len(f()) == 3")


def test_name_clash_with_a_builtin_fails_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert not passes("def sorted(x): return [1]\n", "assert sorted([2, 1]) == [1, 2]")
    assert "builtin's name" in caplog.text


@pytest.mark.parametrize(
    "test, want",
    [
        ("assert f(1) == 1 and f(3) == 99", True),
        ("assert f(1) == 1 and f(2) == 99", False),
        ("assert f(1) == 1 or garbage()", True),
        ("assert garbage() or f(1) == 1", True),
        ("assert f(2) == 5 or garbage()", False),
        ("assert not (f(1) == 2)", True),
        ("assert not (f(1) == 1)", False),
        ("assert not garbage()", False),
        ("assert not (garbage() or f(2) == 5)", False),
        ("assert 1 < two() < 3", True),
        ("assert 1 < two() < 2", False),
        ("assert two() == 2 and (f(1) == 1 or f(2) == 1)", True),
        ("assert m == 5 and two() == 2", False),
        ("assert obj() == 1 and two() == 2", False),
        ("assert all(obj() == i for i in range(3))", False),
        ("assert (obj() == 1) is True", False),
        ("assert f(3) + 1 == 100", True),
        ("assert f(1) + 1 == 3", False),
        ("assert isinstance(two(), int) and any([two()]) and all([two()])", True),
        ("assert two()._x == 1", False),
        ("assert open('x')", False),
        ("assert (lambda: True)()", False),
    ],
)
def test_compound_asserts_are_decided_in_the_trainer(test, want):
    assert passes(CODE, test) is want


def test_assert_that_cant_be_decomposed_fails_with_a_warning(caplog):
    _code_grader.plan_assert.cache_clear()
    with caplog.at_level("WARNING"):
        assert not passes(CODE, "assert all(obj() == i for i in range(3))")
    assert "Can't grade" in caplog.text


TAMPER = {
    "frame": (
        "import sys\ndef f():\n    return [1]\n"
        "fr = sys._getframe(0)\nwhile fr.f_back is not None:\n    fr = fr.f_back\n"
        "for k, v in list(fr.f_locals.items()):\n    if isinstance(v, dict) and 'len' in v: v['len'] = lambda x: 5\n"
    ),
    "gc": (
        "import gc\ndef f():\n    return [1]\n"
        "for o in gc.get_objects():\n    if isinstance(o, dict) and o.get('len') is len: o['len'] = lambda x: 5\n"
    ),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_candidate_walking_frames_or_gc_cannot_change_a_test(name):
    assert not passes(TAMPER[name], "assert len(f()) == 5")
    assert passes(TAMPER[name], "assert len(f()) == 1")


def test_object_with_a_lying_class_or_eq_is_rejected():
    code = "class L(list):\n    __class__ = int\n    def __eq__(self, o): return True\ndef f(): return L([1])\n"
    for test in ("assert f() == [2]", "assert isinstance(f(), list)", "assert f() == f()"):
        assert not passes(code, test)


def test_patched_predicates_in_the_candidate_namespace_do_not_apply():
    code = "def isinstance(a, b): return True\ndef any(x): return True\ndef f(): return 5\n"
    assert not passes(code, "assert isinstance(f(), list)")
    assert not passes(code, "assert any(f())")
    code = "import builtins\nbuiltins.isinstance = lambda a, b: True\nbuiltins.any = lambda x: True\ndef f(): return []\n"
    assert not passes(code, "assert isinstance(f(), list) and False")
    assert not passes(code, "assert any(f())")
    assert passes(code, "assert isinstance(f(), list)")


BIG = "def f(): return 10**4000\ndef n(): return 5\ndef s(): return 'x' * 1000000\ndef sl(): return ['a'] * 9000\n"


@pytest.mark.parametrize(
    "test",
    [
        "assert s() == 'x' * 10**9",
        "assert len(s()) * 10**9 == 0",
        "assert '%99999999d' % n() == ''",
        "assert '%s' % n() == '5'",
        "assert n() ** 10**6 == 0",
        "assert 2 ** f() == 0",
        "assert sum(range(10**10)) == f()",
        "assert [0] * f() == []",
        "assert f() << 10**7 == 0",
        "assert ('b' * 20).join(sl()) == ''",
        "assert 'a'.ljust(10**9) == ''",
        "assert 'a'.replace('a', 'b' * 10**6) == ''",
        "assert '{0.__class__}'.format(1) == ''",
        "assert f'{n():99999999}' == ''",
        "assert sorted(s()) == []",
        "assert list(s()) == []",
    ],
)
def test_resource_hungry_asserts_fail_fast(test):
    import time
    import tracemalloc

    tracemalloc.start()
    start = time.time()
    try:
        assert not passes(BIG, test)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert time.time() - start < 5
    assert peak < 60 * 1024 * 1024  # the trainer never builds the huge value


def test_normal_arithmetic_and_sizes_still_pass():
    for test in ("assert len(s()) == 1000000", "assert n() * 2 == 10", "assert n() % 3 == 2", "assert n() ** 2 == 25",
                 "assert [0] * n() == [0] * 5", "assert ','.join(['a', 'b']) == 'a,b'"):
        assert passes(BIG, test), test


HASHES = (
    "M = 2**61 - 1\n"
    "def f(): return [M * i for i in range(1, 9000)]\n"
    "def g(): return [M * i for i in range(1, 9000)]\n"
    "def d(): return list(range(9000))\n"
)


@pytest.mark.parametrize(
    "test",
    [
        "assert set(f()) == {1}",
        "assert frozenset(f()) == frozenset()",
        "assert dict.fromkeys(f()) == {}",
        "assert set(f()) | set(g()) == set()",
        "assert set(f()).union(g()) == set()",
        "assert set([1]).issubset(f())",
        "assert len(set(f())) == 9000",
    ],
)
def test_colliding_hashes_cannot_make_the_trainer_build_a_slow_set(test):
    import time

    start = time.time()
    assert not passes(HASHES, test)
    assert time.time() - start < 5


def test_distinct_values_still_go_through_sets():
    for test in ("assert len(set(d())) == 9000", "assert set(d()) | {-1} != set()", "assert len(frozenset(d())) == 9000",
                 "assert d().count(5) == 1", "assert len(dict(enumerate(d()))) == 9000"):
        assert passes(HASHES, test), test


def test_colliding_hashes_are_refused_before_the_set_is_built():
    import time

    bad = [(2**61 - 1) * i for i in range(1, 9000)]
    for test in ("assert set(f()) == {1}", "assert frozenset(f()) == frozenset()", "assert len(set(f())) == 1"):
        plan = _code_grader.plan_assert(test)
        assert plan is not None
        check = plan[1]
        start = time.time()
        with pytest.raises(_code_grader._Fail):
            check([bad])
        assert time.time() - start < 0.2


# Dict views and iterators (zip, enumerate, reversed) go through the same collision check as the
# containers they come from. The values here are what the child would return.
_M = 2**61 - 1
_BAD = [_M * i for i in range(1, 9000)]
_BAD_KEYS = dict.fromkeys(_BAD, 0)  # keys that hash alike
_BAD_VALUES = {i: _M * i for i in range(1, 9000)}  # distinct keys, values that hash alike
_GOOD = list(range(9000))
_GOOD_KEYS = dict.fromkeys(_GOOD, 0)
_GOOD_VALUES = {i: i for i in _GOOD}


def _run(test, *values):
    plan = _code_grader.plan_assert(test)
    assert plan is not None
    return plan[1](list(values))


@pytest.mark.parametrize(
    "test, values",
    [
        ("assert set(f().values()) == set()", [_BAD_VALUES]),
        ("assert frozenset(f().items()) == frozenset()", [_BAD_KEYS]),
        ("assert set(f().keys()) == set()", [_BAD_KEYS]),
        ("assert set(zip(f(), f())) == set()", [_BAD, _BAD]),
        ("assert set(zip(f().values(), f().values())) == set()", [_BAD_VALUES, _BAD_VALUES]),
        ("assert dict(zip(f().values(), f().keys())) == {}", [_BAD_VALUES, _BAD_VALUES]),
        ("assert dict(zip(f(), g())) == {}", [_BAD, _GOOD]),
        ("assert dict(enumerate(f().values())) == {}", [_BAD_VALUES]),
        ("assert dict(zip(f().values(), g())) == {}", [_BAD_VALUES, _GOOD]),
        ("assert set(reversed(f().values())) == set()", [_BAD_VALUES]),
        ("assert set(reversed(f())) == set()", [_BAD]),
        ("assert f().keys() | g().keys() == set()", [_BAD_KEYS, _BAD_KEYS]),
        ("assert f().keys() & g().keys() == set()", [_BAD_KEYS, _BAD_KEYS]),
        ("assert f().keys() - g().keys() == set()", [_BAD_KEYS, _BAD_KEYS]),
        ("assert f().keys() ^ g().keys() == set()", [_BAD_KEYS, _BAD_KEYS]),
        ("assert set([1]).union(f().values()) == set()", [_BAD_VALUES]),
        ("assert set([1]).issubset(f().keys()) == set()", [_BAD_KEYS]),
    ],
)
def test_views_and_iterators_cannot_dodge_the_collision_check(test, values):
    import time

    start = time.time()
    with pytest.raises(_code_grader._Fail):
        _run(test, *values)
    assert time.time() - start < 1


@pytest.mark.parametrize(
    "test, values",
    [
        ("assert len(set(f().values())) == 9000", [_GOOD_VALUES]),
        ("assert len(frozenset(f().items())) == 9000", [_GOOD_KEYS]),
        ("assert len(set(zip(f(), f()))) == 9000", [_GOOD, _GOOD]),
        ("assert len(dict(zip(f(), g()))) == 9000", [_GOOD, _GOOD]),
        ("assert len(dict(enumerate(f().values()))) == 9000", [_GOOD_VALUES]),
        ("assert len(set(reversed(f()))) == 9000", [_GOOD]),
        ("assert len(f().keys() | g().keys()) == 9000", [_GOOD_KEYS, _GOOD_KEYS]),
        ("assert len(f().keys() & g().keys()) == 9000", [_GOOD_KEYS, _GOOD_KEYS]),
        ("assert len(set([-1]).union(f().values())) == 9001", [_GOOD_VALUES]),
        ("assert sorted(f().items())[0] == (0, 0)", [_GOOD_VALUES]),
        ("assert sum(f().values()) == 9000 * 8999 // 2", [_GOOD_VALUES]),
    ],
)
def test_the_same_shapes_with_distinct_values_still_work(test, values):
    assert _run(test, *values) is True


_SETS_BAD = [{_M * i for i in range(1, 4001)}, {_M * i for i in range(4001, 8001)}]
_SETS_GOOD = [set(range(4000)), set(range(4000, 8000))]
_PAIRS_BAD = [(k, 0) for k in _BAD]


@pytest.mark.parametrize(
    "test, values",
    [
        ("assert dict(f()) == {}", [_PAIRS_BAD + [(1, 2, 3)]]),
        ("assert dict(f()) == {}", [_PAIRS_BAD + [7]]),
        ("assert dict(f()) == {}", [_PAIRS_BAD]),
        ("assert dict(f()) == {}", [[frozenset({k, 0}) for k in _BAD]]),
        ("assert dict(f()) == {}", [[{k, 0} for k in _BAD]]),
        ("assert dict(f()) == {}", [["ab"]]),
        ("assert dict(f()) == {}", [[{1: 2, 3: 4}]]),
        ("assert f() | g() == set()", _SETS_BAD),
        ("assert f() & g() == set()", _SETS_BAD),
        ("assert f().union(g()) == set()", _SETS_BAD),
        ("assert f().intersection(g()) == set()", _SETS_BAD),
        ("assert f().difference(g()) == set()", _SETS_BAD),
        ("assert f().symmetric_difference(g()) == set()", _SETS_BAD),
    ],
)
def test_dict_pairs_and_set_operators_are_checked_per_element_and_on_results(test, values):
    import time

    start = time.time()
    with pytest.raises(_code_grader._Fail):
        _run(test, *values)
    assert time.time() - start < 1


@pytest.mark.parametrize(
    "test, values",
    [
        ("assert len(dict(f())) == 9000", [[(k, 0) for k in _GOOD]]),
        ("assert len(dict(f())) == 9000", [[[k, 0] for k in _GOOD]]),
        ("assert len(f() | g()) == 8000", _SETS_GOOD),
        ("assert len(f() & g()) == 0", _SETS_GOOD),
        ("assert len(f().union(g())) == 8000", _SETS_GOOD),
        ("assert len(f().difference(g())) == 4000", _SETS_GOOD),
    ],
)
def test_ordinary_pairs_and_sets_still_work(test, values):
    assert _run(test, *values) is True


@pytest.mark.parametrize(
    "code, test",
    [
        ("def f(): return [0]*100\n", "assert f() == [0]*100"),
        ("def f(): return [0]*100\n", "assert set(f()) == {0}"),
        ("def f(): return [7]*200\n", "assert f().count(7) == 200"),
        ("def f(): return [(1, 2)]*100\n", "assert dict(f()) == {1: 2}"),
        ("def f(): return [1]*100\ndef g(): return [2]*100\n", "assert set(f()).union(g()) == {1, 2}"),
        ("def f(): return [1]*100\ndef g(): return [2]*100\n", "assert set(f()) | set(g()) == {1, 2}"),
        ("def f(): return [2**61 - 1]*9000\n", "assert len(set(f())) == 1"),
        ("def f(): return {1: 0, 2: 0}\n", "assert dict(zip([3]*100, [4]*100)) == {3: 4}"),
        (HASHES, "assert sorted(f())[0] == 2**61 - 1"),
        (HASHES, "assert len(sorted(f())) == 8999 and f().count(2**61 - 1) == 1"),
        (HASHES, "assert max(f()) == 8999 * (2**61 - 1)"),
    ],
)
def test_equal_elements_and_unhashed_lists_are_not_collisions(code, test):
    assert passes(code, test)


def test_equal_elements_are_accepted_and_distinct_colliding_ones_refused_in_the_helper():
    assert not _code_grader._collides([_M] * 9000)
    assert not _code_grader._collides([1] * 100 + [1.0] * 30)
    assert _code_grader._collides(_BAD)
    assert not _code_grader._collides(_BAD[:64])


@pytest.mark.parametrize(
    "test",
    ["assert set(f()) == set()", "assert frozenset(f()) == frozenset()", "assert dict.fromkeys(f()) == {}",
     "assert len(dict(zip(f(), g()))) == 0", "assert set(f()).union(g()) == set()", "assert f()[0] in set(f())"],
)
def test_distinct_colliding_ints_are_still_refused(test):
    assert not passes(HASHES, test)
