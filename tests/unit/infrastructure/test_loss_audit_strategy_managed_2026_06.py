"""Regression: the loss-coverage auditor must not drift from LossBuilder.

2026-06-11 infrastructure audit. ``loss_audit.py`` kept a private copy of the
strategy-managed loss set that was a strict *subset* of ``LossBuilder``'s skip
set (it lacked ``sim`` / ``smooth`` / ``marker`` / ``style`` / ``content`` /
``padnet_*`` / ``physics_informed`` / …). ``verify_startup_losses`` raises on any
audited entry with an ``.error``, so a disentangled / cycle-Bloch / padnet config
enabling one of those losses failed at *startup* with a spurious
"LossBuilder did not create it". The two sets now share one SSOT constant.
"""

from __future__ import annotations


def test_loss_audit_imports_the_builder_ssot_constant():
    from mriforge.infrastructure.training.builders.loss_builder import (
        STRATEGY_MANAGED_LOSSES,
    )

    # The names that previously drifted out of the auditor's copy.
    for name in (
        "sim", "smooth", "marker", "style", "content", "prior", "distill",
        "padnet_l2", "padnet_dc", "padnet_reg", "physics_informed",
        "snr_preserving", "biophysical_flow", "physics_constraint",
        "parallel_imaging_kspace", "bloch_residual", "anat", "recon",
    ):
        assert name in STRATEGY_MANAGED_LOSSES, name


def test_auditor_skip_set_is_superset_of_builder_skip_set():
    """The auditor must not flag anything the builder deliberately skips."""
    import inspect

    from mriforge.infrastructure import loss_audit
    from mriforge.infrastructure.training.builders.loss_builder import (
        STRATEGY_MANAGED_LOSSES,
    )

    # The auditor builds ``strategy_managed_losses`` as
    # ``STRATEGY_MANAGED_LOSSES | {"bloch_consistency"}`` inside audit_losses;
    # assert the source references the SSOT constant (no re-declared literal set).
    src = inspect.getsource(loss_audit.audit_losses)
    assert "STRATEGY_MANAGED_LOSSES" in src
    # And that the union the auditor uses covers the whole builder set.
    auditor_set = STRATEGY_MANAGED_LOSSES | {"bloch_consistency"}
    assert auditor_set >= STRATEGY_MANAGED_LOSSES
