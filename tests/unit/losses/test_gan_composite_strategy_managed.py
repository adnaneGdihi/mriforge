"""Regression test for F6/E26 (smoke_audit_20260521) — GAN-composite
sub-losses must be treated as strategy-managed by LossBuilder.

Root cause: ``gradient_penalty`` and ``feature_matching`` are
GAN-composite regularizers built inside ``_build_composite_gan``,
not standalone declarative losses. They live inside the
``losses.gan`` sub-block (``schemas/training/gan.py::lambda_gp``,
``enable_gradient_penalty``), and ``LossSchema.get_enabled_losses``
synthesizes top-level loss-name strings (``gradient_penalty``,
``feature_matching``) for the LossBuilder's enabled-losses dict.

Before F6, the LossBuilder's list-based path treated these as
"unmigrated v6.0 keys" and refused to build, crashing 4 experiments
(dummy_gan + experiment_77_padnet_multicontrast +
exp_ulf_05_gan_uncertainty + exp_up_04_spectral_band_split) at
config-build time.

Fix: extend the LossBuilder's ``strategy_managed`` set to include
``gradient_penalty`` and ``feature_matching`` so they pass the
unmigrated check (the GAN composite handles them internally).

The two inline ``strategy_managed = {…}`` literals this test used to
pin have since been collapsed into one module-level SSOT frozenset,
``STRATEGY_MANAGED_LOSSES``, referenced by both the list-based and
legacy-flag-based paths (and imported by ``loss_audit.py`` so the two
cannot drift). The tests now assert membership in that frozenset and
that both consumer paths still read it, so re-inlining one path drops
the guarantee visibly.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOSS_BUILDER = (
    REPO
    / "src"
    / "mriforge"
    / "infrastructure"
    / "training"
    / "builders"
    / "loss_builder.py"
)


def _read() -> str:
    return LOSS_BUILDER.read_text()


def _both_paths_read_ssot() -> int:
    """Count the loss-builder paths that resolve the skip set from the SSOT."""
    return len(re.findall(r"strategy_managed\s*=\s*STRATEGY_MANAGED_LOSSES", _read()))


def test_gradient_penalty_is_strategy_managed() -> None:
    """``gradient_penalty`` must be in the strategy-managed skip set so the
    LossBuilder doesn't flag it as an unmigrated v6.0 loss key.

    Asserts the runtime SSOT (``STRATEGY_MANAGED_LOSSES``) rather than the old
    inline literals, and that BOTH consumer paths (list-based + legacy flag)
    still resolve from it — so re-inlining one path is caught.
    """
    from mriforge.infrastructure.training.builders.loss_builder import (
        STRATEGY_MANAGED_LOSSES,
    )

    assert "gradient_penalty" in STRATEGY_MANAGED_LOSSES, (
        "GAN configs with WGAN-GP discriminators refuse to build unless "
        "'gradient_penalty' is strategy-managed. See F41 / smoke_audit_20260521."
    )
    assert _both_paths_read_ssot() == 2, (
        "Both the list-based and legacy-flag paths must read "
        "STRATEGY_MANAGED_LOSSES; found "
        f"{_both_paths_read_ssot()} references. A path with its own inline set "
        "can silently drift from the SSOT."
    )


def test_feature_matching_is_strategy_managed() -> None:
    """Feature matching is the second GAN composite companion that needs the
    same exemption. Pinned alongside gradient_penalty so a half-revert (one but
    not the other) is caught.
    """
    from mriforge.infrastructure.training.builders.loss_builder import (
        STRATEGY_MANAGED_LOSSES,
    )

    assert "feature_matching" in STRATEGY_MANAGED_LOSSES


def test_skip_set_is_traceable_to_its_audit_reason() -> None:
    """The skip set must document WHY these entries are exempt (and that it is
    the SSOT shared with the loss auditor) so a future cleanup doesn't strip
    them as unused.
    """
    text = _read()
    assert "SSOT for the skip set" in text and "loss_audit" in text, (
        "STRATEGY_MANAGED_LOSSES must state it is the SSOT and name the "
        "loss auditor that imports it, keeping the exemption traceable."
    )
