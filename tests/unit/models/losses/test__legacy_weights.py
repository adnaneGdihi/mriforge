"""The legacy-family selector: pick the resolver an arm's loss path ACTUALLY reached.

Getting this wrong is not a rounding error. ``hfen`` declared as
``image_losses: [{name: hfen, weight: 0.2}]`` resolves to **0.2** under ``folding`` (the
inline strategies read the list entry) and to **0.0** under ``computer`` (which scans
``lambda_hfen``, finds only the schema default). The first migration asked ``computer`` for
folding arms, concluded the loss had never been computed, and disabled it on 37 arms.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.models.losses._legacy_weights import (
    FOLDING_STRATEGIES,
    RECON_FAMILY_MODES,
    legacy_effective_weight,
    legacy_family_for,
)


class _Root:
    def __init__(self, losses: LossConfigSchema) -> None:
        self.losses = losses
        self.objectives = None


def _cfg() -> LossConfigSchema:
    """A loss block whose only declaration is a LIST entry -- no ``lambda_hfen`` anywhere."""
    return LossConfigSchema(
        output_domain="image",
        reconstruction={"warmup_iterations": 0},
        image_losses=[{"name": "hfen", "weight": 0.2}],
    )


@pytest.mark.parametrize("strategy", sorted(FOLDING_STRATEGIES))
def test_folding_strategies_select_the_folding_family(strategy: str) -> None:
    assert (
        legacy_family_for(strategy_class=strategy, training_mode="reconstruction")
        == "folding"
    )


def test_strategy_class_beats_training_mode() -> None:
    """A folding strategy in a recon-family MODE is still a folding arm.

    ``scattering_besov`` arms carry ``training_mode: reconstruction``, so a selector that
    looks only at the mode routes them to ``recon`` -- which cannot see a list weight. This
    is the exact collision the 74 disabled loss terms came through.
    """
    assert "reconstruction" in RECON_FAMILY_MODES
    assert (
        legacy_family_for(
            strategy_class="scattering_besov", training_mode="reconstruction"
        )
        == "folding"
    )


def test_training_mode_selects_recon_vs_computer_when_not_folding() -> None:
    assert legacy_family_for(strategy_class="gan", training_mode="diffusion") == "recon"
    assert legacy_family_for(strategy_class="gan", training_mode="gan") == "computer"


def test_missing_strategy_class_falls_back_to_training_mode_as_the_key() -> None:
    """``strategy_class`` is optional; the factory uses ``training_mode`` as the key."""
    assert (
        legacy_family_for(strategy_class=None, training_mode="scattering_besov")
        == "folding"
    )
    assert legacy_family_for(strategy_class=None, training_mode=None) == "computer"


def test_the_two_families_disagree_on_a_list_declared_loss() -> None:
    """The regression itself: a live 0.2 reads as a never-computed 0.0 under the wrong family."""
    root = _Root(_cfg())
    assert legacy_effective_weight(root, "hfen", family="folding") == pytest.approx(0.2)
    assert legacy_effective_weight(root, "hfen", family="computer") == pytest.approx(
        0.0
    )


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown legacy family"):
        legacy_effective_weight(_Root(_cfg()), "hfen", family="nope")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy, arm, term, wrong",
    [
        # `wrong` is the weight the MISCLASSIFICATION produced, measured -- not
        # assumed to be 0.0. `computer` answers with whatever `lambda_<n>` defaults
        # to, which differs per term, so the damage differs per arm.
        ("acq_hypernetwork", "lcah_recon_multifield", "l1", 10.0),
        # Unaffected BY COINCIDENCE: lambda_multifield_data_consistency happens to
        # default to the same 1.0 the arm declares. Pinned at the value rather than
        # dropped, so a change to that default surfaces here instead of silently
        # starting to matter.
        (
            "dispersion_bloch_ae",
            "dlbae_brain_5field",
            "multifield_data_consistency",
            1.0,
        ),
        # No live arm yet, but the set describes the CODE, not the corpus: each of
        # these calls `_apply_builder_image_losses`, so the first arm to declare a
        # list loss under one would silently get its lambda default instead.
        ("mrs_quantification", None, None, None),
        ("perfusion_kinetic", None, None, None),
        ("phase_contrast_flow", None, None, None),
    ],
)
def test_the_five_late_adopters_select_folding(
    strategy: str, arm: str | None, term: str | None, wrong: float | None
) -> None:
    """Regression for the 2026-08-09 drift (cluster 8012333).

    These five adopted the folding seam without landing in ``FOLDING_STRATEGIES``,
    so ``legacy_family_for`` answered ``computer``, whose lambda scan cannot see a
    list weight and substitutes the ``lambda_<n>`` schema default.

    That is NOT always the 0.0 of the 37- and 52-arm incidents in the module
    docstring: ``lambda_l1`` defaults to **10.0**, so ``lcah_recon_multifield`` was
    training its only objective term at ten times the weight it declared. The
    assertion below pins the measured wrong value per term, because "it resolves to
    something plausible" is exactly how this class of bug survives.
    """
    assert legacy_family_for(strategy_class=strategy, training_mode=None) == "folding"
    if term is None:
        return
    root = _Root(
        LossConfigSchema(
            output_domain="image",
            reconstruction={"warmup_iterations": 0},
            image_losses=[{"name": term, "weight": 1.0}],
        )
    )
    assert legacy_effective_weight(root, term, family="folding") == pytest.approx(
        1.0
    ), f"{arm}: the folding family must read the list weight"
    assert legacy_effective_weight(root, term, family="computer") == pytest.approx(
        wrong
    ), f"{arm}: the misclassification resolved '{term}' to {wrong}, not the declared 1.0"
