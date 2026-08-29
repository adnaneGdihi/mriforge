"""One signal-domain vocabulary, and a gate that keeps it that way.

The vocabulary lived as bare strings in four places that had drifted apart:

* ``domain/workflows/profiles.py`` — knew ``spectrum``, not ``latent``
* ``config/schemas/loss.py`` — hardcoded the opposite pair
* ``data/datasets/axis_exposure.py`` — knew only three
* ``models/capabilities.Domain`` — knew all seven

The practical cost: ``mri_spectroscopy`` is a **LIVE** regime whose only signal
domain is ``spectrum``, and the loss layer had no way to name it. These tests
pin every consumer to :class:`SignalDomain` so that cannot recur.
"""

from __future__ import annotations

import pytest

from mriforge.config.schemas.enums import SignalDomain


def _values() -> set[str]:
    return {d.value for d in SignalDomain}


class TestEveryConsumerDrawsFromTheEnum:
    def test_workflow_profiles(self) -> None:
        from mriforge.domain.workflows.profiles import WORKFLOW_PROFILES

        used = {d for p in WORKFLOW_PROFILES.values() for d in p.signal_domains}
        unknown = sorted(used - _values())
        assert not unknown, (
            f"WorkflowProfile.signal_domains names {unknown}, which SignalDomain "
            "does not define. Add the member or fix the profile."
        )

    def test_dataset_signal_domains(self) -> None:
        from mriforge.data.datasets.axis_exposure import DATASET_TYPE_SIGNAL_DOMAINS

        used = {d for s in DATASET_TYPE_SIGNAL_DOMAINS.values() for d in s}
        unknown = sorted(used - _values())
        assert not unknown, f"axis_exposure names {unknown}, absent from SignalDomain"

    def test_model_capabilities_domain(self) -> None:
        """``capabilities.Domain`` is the widest of the four and is where these
        members came from, so it must match the enum exactly — not merely be a
        subset. A member here that the enum lacks means the SSOT was seeded from
        a stale copy."""
        import typing

        from mriforge.models.capabilities import Domain

        literal = set(typing.get_args(Domain))
        assert literal == _values(), (
            "models.capabilities.Domain and SignalDomain have diverged.\n"
            f"  only in capabilities: {sorted(literal - _values())}\n"
            f"  only in SignalDomain: {sorted(_values() - literal)}"
        )

    def test_loss_output_domain_accepts_exactly_the_enum(self) -> None:
        """The validator used a hardcoded 4-tuple; it now derives its legal set,
        so a spectroscopy arm can finally declare its own output domain."""
        from mriforge.config.schemas.loss import LossConfigSchema

        for value in sorted(_values()):
            LossConfigSchema(output_domain=value)  # must not raise

        with pytest.raises(ValueError, match="invalid"):
            LossConfigSchema(output_domain="not_a_domain")


class TestRepresentationsAreNotAcquiredSignals:
    """``latent`` / ``pde_grid`` / ``mesh`` are representations, so a model or
    loss may operate in them while no imaging regime lists them. The asymmetry is
    deliberate — this pins it so a future edit does not "fix" it by adding them
    to a profile."""

    def test_no_profile_claims_a_representation_domain(self) -> None:
        from mriforge.domain.workflows.profiles import WORKFLOW_PROFILES

        representations = {
            SignalDomain.LATENT.value,
            SignalDomain.PDE_GRID.value,
            SignalDomain.MESH.value,
        }
        for regime, profile in WORKFLOW_PROFILES.items():
            overlap = sorted(set(profile.signal_domains) & representations)
            assert not overlap, (
                f"{regime.value} lists {overlap} as a signal domain, but those are "
                "representations, not acquired signals. A regime declares what the "
                "scanner produces."
            )

    def test_spectroscopy_can_now_express_its_own_domain(self) -> None:
        """The regression that motivated the whole change."""
        from mriforge.config.schemas.enums import Regime
        from mriforge.config.schemas.loss import LossConfigSchema
        from mriforge.domain.workflows.profiles import WORKFLOW_PROFILES

        profile = WORKFLOW_PROFILES[Regime.SPECTROSCOPIC]
        assert profile.signal_domains == {SignalDomain.SPECTRUM.value}
        LossConfigSchema(output_domain=SignalDomain.SPECTRUM.value)
