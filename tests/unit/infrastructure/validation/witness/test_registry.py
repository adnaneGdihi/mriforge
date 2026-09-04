"""Registry semantics: unique names, idempotent re-import, enumerable coverage."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness.registry import (
    Applicability,
    Severity,
    Stage,
    Subject,
    Tier,
    Witness,
    WitnessRegistry,
    WitnessVerdict,
)


def _witness(name="w", fn=None, class_ids=(), tiers=(Tier.T1,)):
    return Witness(
        name=name,
        fn=fn or (lambda s: WitnessVerdict(name, True, "ok")),
        category="test",
        stage=Stage.PARSE,
        tiers=frozenset(tiers),
        subjects=frozenset({Subject.CONFIG}),
        class_ids=tuple(class_ids),
    )


def test_duplicate_name_with_a_different_function_raises():
    """A verdict must be traceable to exactly one detector."""
    reg = WitnessRegistry()
    reg.register(_witness("dup", fn=lambda s: None))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_witness("dup", fn=lambda s: None))


def test_reimporting_the_same_function_is_idempotent():
    """Module re-import must not explode; only a genuine collision is an error."""
    reg = WitnessRegistry()
    fn = lambda s: None  # noqa: E731
    reg.register(_witness("same", fn=fn))
    reg.register(_witness("same", fn=fn))
    assert len(reg) == 1


def test_for_tier_selects_only_matching_witnesses():
    reg = WitnessRegistry()
    reg.register(_witness("cheap", tiers=(Tier.T1,)))
    reg.register(_witness("expensive", tiers=(Tier.T3,)))
    assert [w.name for w in reg.for_tier(Tier.T1)] == ["cheap"]


def test_covered_class_ids_is_enumerable():
    """The registry must be able to say what it covers.

    This is what a coverage meta-witness needs and what the 126 hand-called
    check_* methods cannot answer.
    """
    reg = WitnessRegistry()
    reg.register(_witness("a", class_ids=("S4.1", "S4.2")))
    reg.register(_witness("b", class_ids=("S11.3",)))
    assert reg.covered_class_ids() == frozenset({"S4.1", "S4.2", "S11.3"})


class TestApplicability:
    def test_no_constraint_matches_everything(self):
        assert Applicability().matches({}) is True

    def test_model_type_constraint_filters(self):
        app = Applicability(model_types=frozenset({"unet"}))
        assert app.matches({"model": {"model_type": "unet"}}) is True
        assert app.matches({"model": {"model_type": "swinir"}}) is False

    def test_missing_block_does_not_crash(self):
        assert Applicability(model_types=frozenset({"unet"})).matches({}) is False

    def test_enum_valued_config_matches_on_its_value_not_its_repr(self):
        """The (str, Enum) trap: str(member) is 'Class.MEMBER', not the value.

        A resolved settings dump can carry enum members. Comparing their repr
        would match nothing and the witness would silently never apply.
        """
        from enum import Enum

        class Mode(str, Enum):  # noqa: UP042 - the (str, Enum) shape IS what this test exercises
            RECON = "reconstruction"

        app = Applicability(training_modes=frozenset({"reconstruction"}))
        assert app.matches({"training": {"training_mode": Mode.RECON}}) is True


def test_verdict_serialises_for_the_json_payload():
    v = WitnessVerdict("w", False, "boom", severity=Severity.ERROR, class_ids=("S1.1",))
    d = v.to_dict()
    assert d["severity"] == "error"
    assert d["class_ids"] == ["S1.1"]


# ---------------------------------------------------------------------------
# Sub-package failures during witness discovery (2026-08-28).
#
# ``walk_packages`` recurses by IMPORTING a sub-package, inside pkgutil, so the
# ``except ImportError`` around the loop's own ``import_module`` never sees that
# failure and pkgutil's default discards it with the whole sub-tree. For a
# witness package that means detectors silently ceasing to exist, and a clean
# report for a class nothing is watching -- the exact outcome the module
# docstring says must never happen. ``checks/`` is flat today, which makes the
# hole latent rather than absent.
#
# That the walk is HANDED an onerror is owned by
# tests/architecture/test_discovery_walks_report_errors.py, repo-wide -- these
# tests own what the callback then decides.
# ---------------------------------------------------------------------------


def test_witness_import_error_classification_reraises_in_repo() -> None:
    """A broken in-repo witness module is a detector that does not exist."""
    from spectramr.infrastructure.validation.witness import (
        _classify_witness_import_error,
    )

    for exc in (
        ImportError("boom", name="spectramr.infrastructure.validation.witness.checks.x"),
        ImportError("boom"),  # falsy name: unattributable, so also loud
        RuntimeError("not an import problem"),
    ):
        with pytest.raises(type(exc)):
            _classify_witness_import_error("some.module", exc)


def test_witness_import_error_classification_downgrades_third_party() -> None:
    """A genuinely absent optional dependency warns rather than raising, so one
    missing package cannot take down every detector in the package."""
    from spectramr.infrastructure.validation.witness import (
        _classify_witness_import_error,
    )

    exc = ModuleNotFoundError("No module named 'piq'", name="piq")
    _classify_witness_import_error("some.module", exc)  # must not raise
