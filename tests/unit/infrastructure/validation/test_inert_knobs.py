"""Unit tests for the declared-but-unread model-knob detector.

Pairs with ``src/mriforge/infrastructure/validation/inert_knobs.py``.

The detector's whole value is its *precision*: a false positive sends an author
to delete a live knob, which is strictly worse than the defect it reports. Most
of what follows therefore pins the ways a parameter legitimately counts as READ.
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.validation.inert_knobs import (
    DELIBERATELY_UNREAD,
    InertKnob,
    find_inert_declared_knobs,
    unread_init_params,
)


class _Swallows:
    """Declares a parameter, documents it, never references it."""

    def __init__(self, used: int = 1, ignored: str = "complex"):
        """Args: used (int): kept. ignored (str): documented and dropped."""
        self.used = used


class _AssignsAttribute:
    def __init__(self, knob: str = "x"):
        self.knob = knob


class _ForwardsToSuper(_AssignsAttribute):
    def __init__(self, knob: str = "x"):
        super().__init__(knob)


class _ReadsInBranchOnly:
    def __init__(self, flag: bool = False):
        self.mode = "on" if flag else "off"


class _ReadsInFString:
    def __init__(self, label: str = "a"):
        self.name = f"model-{label}"


class _UsesLocals:
    """Undecidable by AST — the detector must decline rather than accuse."""

    def __init__(self, alpha: int = 1, beta: int = 2):
        self._cfg = dict(locals())


class _NoParams:
    def __init__(self):
        self.x = 1


def test_unreferenced_parameter_is_reported():
    assert unread_init_params(_Swallows) == frozenset({"ignored"})


@pytest.mark.parametrize(
    "cls",
    [_AssignsAttribute, _ForwardsToSuper, _ReadsInBranchOnly, _ReadsInFString],
    ids=["self-assign", "super-forward", "branch-condition", "f-string"],
)
def test_genuine_reads_are_not_reported(cls):
    """Every one of these consumes its parameter; none may be flagged."""
    assert unread_init_params(cls) == frozenset()


def test_reflective_escape_declines_to_answer():
    """``locals()`` can consume a parameter without naming it.

    The empty set here means "no answer", not "no problems" — asserting it
    pins that the detector stays silent rather than reporting ``alpha``/``beta``.
    """
    assert unread_init_params(_UsesLocals) == frozenset()


def test_parameterless_init_is_empty():
    assert unread_init_params(_NoParams) == frozenset()


def test_builtin_init_is_empty():
    """A class with no Python-level ``__init__`` yields no answer, not a crash."""

    class _Plain:
        pass

    assert unread_init_params(_Plain) == frozenset()


def test_only_declared_knobs_are_reported():
    """Arm-scoped: an unread parameter the arm never set is not this check's business."""
    hits = find_inert_declared_knobs("demo", {"used": 3}, _Swallows)
    assert hits == []


def test_declared_and_unread_is_reported_with_provenance():
    hits = find_inert_declared_knobs("demo", {"used": 3, "ignored": "relu"}, _Swallows)
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, InertKnob)
    assert hit.key == "ignored"
    assert hit.yaml_path == "model.model_kwargs.ignored"
    assert hit.declared_value == "relu"
    assert hit.model_type == "demo"
    assert hit.class_name == "_Swallows"


def test_unresolved_model_class_returns_no_answer():
    assert find_inert_declared_knobs("demo", {"ignored": "relu"}, None) == []


def test_empty_kwargs_returns_no_answer():
    assert find_inert_declared_knobs("demo", {}, _Swallows) == []
    assert find_inert_declared_knobs("demo", None, _Swallows) == []


def test_allowlist_suppresses_deliberate_parameter(monkeypatch):
    """An allowlisted entry records intent and must not be reported."""
    monkeypatch.setitem(DELIBERATELY_UNREAD, ("_Swallows", "ignored"), "on purpose")
    assert find_inert_declared_knobs("demo", {"ignored": "relu"}, _Swallows) == []


def test_results_are_sorted_by_key():
    class _TwoDead:
        def __init__(self, zebra: int = 1, alpha: int = 2, live: int = 3):
            self.live = live

    hits = find_inert_declared_knobs("demo", {"zebra": 1, "alpha": 2, "live": 3}, _TwoDead)
    assert [h.key for h in hits] == ["alpha", "zebra"]


def test_regression_kspace_cold_diffusion_swallows_four_knobs():
    """The defect that motivated the check, pinned against silent repair.

    ``KSpaceColdDiffusionGenerator.__init__`` declares and documents these four
    and references none: flipping ``activation``/``use_complex_conv`` leaves the
    module tree and forward output bit-identical. If a fix lands, this test
    fails loudly and should be updated — that is the point.
    """
    from mriforge.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    assert unread_init_params(KSpaceColdDiffusionGenerator) == frozenset(
        {"activation", "use_complex_conv", "time_embedding_type", "training_mode"}
    )
