"""Regression tests for the ratchet mechanism the fitness functions share.

Issue #629: ``tests/architecture/`` was 11-red on clean ``dev`` because the
baseline keyed on ``path (N loc)`` and ``ratchet()`` was a plain set difference.
Adding a line to an already-baselined oversized file changed its key, so it
re-reported as a BRAND-NEW violation — the gate could not stay green through
normal development. These tests pin the two properties that fix implies:
identity-keyed membership, and a measurement that is inert.
"""

from __future__ import annotations

import ast

import pytest

from ._fitness_lib import (
    CANONICAL_TRAIN_STEP,
    CANONICAL_VALIDATION_STEP,
    _as_baseline_line,
    accepts_canonical_call,
    baseline_identity,
    find_exception_dispatch,
    load_baseline,
    ratchet,
)

pytestmark = pytest.mark.architecture


# --------------------------------------------------------------------------- #
# baseline_identity: the measurement must not be part of the key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("src/mriforge/bootstrap.py (336 loc)", "src/mriforge/bootstrap.py"),
        ("src/mriforge/bootstrap.py  # 336 loc", "src/mriforge/bootstrap.py"),
        ("AdversarialRobustnessStrategy (depth 4)", "AdversarialRobustnessStrategy"),
        ("AdversarialRobustnessStrategy  # depth 4", "AdversarialRobustnessStrategy"),
        ("src/x.py::list_features (6 branches)", "src/x.py::list_features"),
    ],
)
def test_baseline_identity_strips_the_measurement(entry: str, expected: str) -> None:
    assert baseline_identity(entry) == expected


def test_baseline_identity_keeps_signature_param_tuples_whole() -> None:
    """Signature entries end in a param tuple, not a count — there the
    parenthesised part IS the identity and must survive."""
    entry = "src/x.py::validation_step('val_batch', 'batch_idx')"
    assert baseline_identity(entry) == entry


def test_growing_a_baselined_file_is_not_a_new_violation(tmp_path, monkeypatch) -> None:
    """The exact #629 failure: bootstrap.py 336 -> 371 LOC must stay silent."""
    monkeypatch.setattr("tests.architecture._fitness_lib.BASELINE_DIR", tmp_path)
    monkeypatch.delenv("MRIFORGE_UPDATE_ARCH_BASELINE", raising=False)
    (tmp_path / "b.txt").write_text(
        "# header\nsrc/mriforge/bootstrap.py  # 336 loc\n", encoding="utf-8"
    )

    assert ratchet("b.txt", {"src/mriforge/bootstrap.py (371 loc)"}) == set()


def test_a_previously_unseen_file_is_still_reported(tmp_path, monkeypatch) -> None:
    """The ratchet must not go blind — a new path is still a new violation."""
    monkeypatch.setattr("tests.architecture._fitness_lib.BASELINE_DIR", tmp_path)
    monkeypatch.delenv("MRIFORGE_UPDATE_ARCH_BASELINE", raising=False)
    (tmp_path / "b.txt").write_text("src/mriforge/bootstrap.py  # 336 loc\n", encoding="utf-8")

    assert ratchet("b.txt", {"src/mriforge/brand_new.py (900 loc)"}) == {
        "src/mriforge/brand_new.py (900 loc)"
    }


def test_written_baseline_round_trips_through_the_ratchet(tmp_path, monkeypatch) -> None:
    """Regenerate, then re-run: the second run must report nothing new."""
    monkeypatch.setattr("tests.architecture._fitness_lib.BASELINE_DIR", tmp_path)
    current = {"src/a.py (400 loc)", "src/b.py (500 loc)"}

    monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")
    assert ratchet("b.txt", current) == set()

    monkeypatch.delenv("MRIFORGE_UPDATE_ARCH_BASELINE")
    assert ratchet("b.txt", current) == set()
    # ...and still nothing new once the measurements move.
    assert ratchet("b.txt", {"src/a.py (401 loc)", "src/b.py (499 loc)"}) == set()


def test_write_baseline_demotes_the_measurement_to_a_comment(tmp_path, monkeypatch) -> None:
    """Storing ``# 400 loc`` is what makes the number visibly non-load-bearing."""
    monkeypatch.setattr("tests.architecture._fitness_lib.BASELINE_DIR", tmp_path)
    monkeypatch.setenv("MRIFORGE_UPDATE_ARCH_BASELINE", "1")

    ratchet("b.txt", {"src/a.py (400 loc)"}, header="h")

    assert "src/a.py  # 400 loc" in (tmp_path / "b.txt").read_text().splitlines()


def test_as_baseline_line_leaves_non_measurements_alone() -> None:
    entry = "src/x.py::validation_step('val_batch', 'batch_idx')"
    assert _as_baseline_line(entry) == entry


# --------------------------------------------------------------------------- #
# accepts_canonical_call: extending a step signature compatibly is not drift
# --------------------------------------------------------------------------- #
def test_defaulted_extra_params_are_not_drift() -> None:
    """The field/ULF cohort appends ``field_strength_target=None`` and forwards
    through ``super()``; ``validation_step(input_batch, target_batch)`` still
    works, so this must not be flagged (#629)."""
    params = ("input_batch", "target_batch", "field_strength_target", "contrast_id")
    required = ("input_batch", "target_batch")
    assert accepts_canonical_call(params, required, CANONICAL_VALIDATION_STEP)


def test_a_required_extra_param_is_drift() -> None:
    """Without a default, the canonical call raises TypeError — real drift."""
    params = ("input_batch", "target_batch", "field_strength_target")
    assert not accepts_canonical_call(params, params, CANONICAL_VALIDATION_STEP)


def test_renamed_canonical_params_are_drift() -> None:
    params = required = ("val_batch", "batch_idx")
    assert not accepts_canonical_call(params, required, CANONICAL_VALIDATION_STEP)


def test_defaulting_a_canonical_param_is_not_drift() -> None:
    """field_cocycle's train_step defaults input_batch/target_batch/iteration —
    strictly more permissive than canonical, so the canonical call still works."""
    params = ("batch", "epoch", "input_batch", "target_batch", "iteration")
    assert accepts_canonical_call(params, ("batch", "epoch"), CANONICAL_TRAIN_STEP)


# --------------------------------------------------------------------------- #
# The committed baselines must parse under the format the ratchet now writes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        "large_files.txt",
        "deep_inheritance.txt",
        "dispatch_hell.txt",
        "step_signature_drift.txt",
        "builder_signature_drift.txt",
        "exception_dispatch.txt",
    ],
)
def test_committed_baselines_have_unique_identities(name: str) -> None:
    """Two entries collapsing to one identity means a stale duplicate survived
    a regeneration — the ratchet would silently exempt whichever came second."""
    entries = load_baseline(name)
    identities = [baseline_identity(e) for e in entries]
    dupes = {i for i in identities if identities.count(i) > 1}
    assert not dupes, f"{name} has duplicate identities: {sorted(dupes)}"


# --------------------------------------------------------------------------- #
# find_exception_dispatch: flag the degraded retry, and ONLY that
# --------------------------------------------------------------------------- #
def _hits(source: str) -> set[tuple[str, str, int]]:
    return set(find_exception_dispatch(ast.parse(source)))


def test_flags_a_retry_with_fewer_arguments() -> None:
    hits = _hits(
        "def step(gen, x, t):\n"
        "    try:\n"
        "        return gen(x, t)\n"
        "    except TypeError:\n"
        "        return gen(x)\n"
    )
    assert hits == {("step", "gen", 1)}


def test_flags_a_dropped_keyword_the_same_way() -> None:
    hits = _hits(
        "def step(model, x, t):\n"
        "    try:\n"
        "        return model(x, timesteps=t)\n"
        "    except TypeError:\n"
        "        return model(x)\n"
    )
    assert hits == {("step", "model", 1)}


def test_attributes_the_site_to_the_innermost_function_once() -> None:
    # A `try` inside a nested closure belongs to the closure, not to both it and
    # the enclosing method -- otherwise every nested site double-counts.
    hits = _hits(
        "class S:\n"
        "    def losses(self, gen, x, t):\n"
        "        def _fwd(m, a, b):\n"
        "            try:\n"
        "                return m(a, b)\n"
        "            except TypeError:\n"
        "                return m(a)\n"
        "        return _fwd(gen, x, t)\n"
    )
    assert hits == {("S.losses._fwd", "m", 1)}


def test_does_not_flag_a_bounded_strip_and_continue_loop() -> None:
    # The `physics_driven_strategy` shape: the handler pops one kwarg and
    # `continue`s, re-raising anything it cannot account for. The retry is the
    # loop, the handler makes no call, and nothing is silently degraded.
    hits = _hits(
        "def step(gen, x, kw):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            return gen(x, **kw)\n"
        "        except TypeError as exc:\n"
        "            if 'unexpected' in str(exc):\n"
        "                kw.pop('a')\n"
        "                continue\n"
        "            raise\n"
    )
    assert hits == set()


def test_does_not_flag_a_retry_that_supplies_more_arguments() -> None:
    # The `meta_learning_strategy` shape: the retry ADDS an argument rather than
    # dropping one. Same unsound probe, but a different defect (it fabricates
    # the extra argument), so it is reported separately rather than folded in
    # here -- a gate that fires on one thing is worth more than one that fires
    # on everything.
    hits = _hits(
        "def fwd(call, model, p, x, mask):\n"
        "    try:\n"
        "        return call(model, p, (x,))\n"
        "    except TypeError:\n"
        "        return call(model, p, (x, mask))\n"
    )
    assert hits == set()


def test_does_not_flag_a_handler_catching_something_else() -> None:
    hits = _hits(
        "def step(gen, x, t):\n"
        "    try:\n"
        "        return gen(x, t)\n"
        "    except ValueError:\n"
        "        return gen(x)\n"
    )
    assert hits == set()


def test_flags_a_tuple_handler_that_includes_type_error() -> None:
    hits = _hits(
        "def step(gen, x, t):\n"
        "    try:\n"
        "        return gen(x, t)\n"
        "    except (TypeError, ValueError):\n"
        "        return gen(x)\n"
    )
    assert hits == {("step", "gen", 1)}


def test_counts_repeated_sites_in_one_function_as_one_entry() -> None:
    # `ddim_sampler.sample` retries `model` three times. One identity, count 3 --
    # so the measurement moves but the ratchet key does not.
    hits = _hits(
        "def sample(model, x, t):\n"
        "    try:\n"
        "        a = model(x, t)\n"
        "    except TypeError:\n"
        "        a = model(x)\n"
        "    try:\n"
        "        b = model(x, t)\n"
        "    except TypeError:\n"
        "        b = model(x)\n"
        "    return a, b\n"
    )
    assert hits == {("sample", "model", 2)}
