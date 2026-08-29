"""Unit tests for the ``physics:`` config block.

Scope note: this file guards the *advertised set* of physics knobs against the
implementation behind them. The schema layer cannot import ``infrastructure``
(the layer rule points inward only), so a field description naming a vocabulary
is an unenforced claim — exactly the shape of pitfall #15. A test can import
both sides, so the claim is checked here.
"""

from __future__ import annotations

from mriforge.config.schemas.physics import DigitalTwinConfig


class TestProgressiveDegradationsAdvertisedSet:
    """``DigitalTwinConfig.progressive_degradations`` documents two banks of
    names. Both must exist, and the simulator must accept exactly what the
    description promises."""

    @staticmethod
    def _description() -> str:
        return DigitalTwinConfig.model_fields["progressive_degradations"].description

    def test_every_native_axis_named_in_the_description_exists(self) -> None:
        from mriforge.infrastructure.physics.digital_twin_simulator import (
            NATIVE_DEGRADATION_AXES,
        )

        text = self._description()
        missing = sorted(a for a in NATIVE_DEGRADATION_AXES if a not in text)
        assert not missing, (
            f"native axes absent from the advertised set: {missing}. The "
            "description enumerates all 14; adding an axis must update it."
        )

    def test_the_registry_size_claim_is_current(self) -> None:
        """The description says '31 keys'. If the registry grows, the number in
        the docs is wrong the moment it does."""
        from mriforge.infrastructure.physics.digital_twin_extensions import (
            DEGRADATION_REGISTRY,
        )

        assert f"{len(DEGRADATION_REGISTRY)} keys" in self._description(), (
            f"registry now has {len(DEGRADATION_REGISTRY)} entries; the "
            "progressive_degradations description still claims a different count"
        )

    def test_registry_examples_in_the_description_are_real_keys(self) -> None:
        from mriforge.infrastructure.physics.digital_twin_extensions import (
            DEGRADATION_REGISTRY,
        )

        text = self._description()
        cited = [
            "rigid_motion",
            "pulsatile_motion",
            "rician",
            "spike",
            "nyquist_ghost",
            "susceptibility",
            "t2star_blur",
            "partial_fourier",
        ]
        assert all(c in text for c in cited), "test's own citation list drifted"
        unknown = [c for c in cited if c not in DEGRADATION_REGISTRY]
        assert not unknown, (
            f"the description cites degradations the registry does not have: "
            f"{unknown} — an advertised knob with no implementation"
        )

    def test_default_is_reachable(self) -> None:
        """The shipped default must itself be constructible — it was, but only
        because all four names happened to be native."""
        from mriforge.infrastructure.physics.digital_twin_simulator import (
            known_degradation_axes,
        )

        known = known_degradation_axes()
        assert set(DigitalTwinConfig().progressive_degradations) <= known
        assert set(DigitalTwinConfig().degradation_ranges) <= known


class TestDataConsistencyNoiseVocabulary:
    """The schema must advertise the SHIPPED vocabulary, not the designed one (#1525).

    ``noise_type``'s description named ``'rician'`` for a long time; no DC layer
    has ever implemented it, and two of the three silently degraded such a value
    to Gaussian instead of raising.
    """

    def test_noise_type_description_does_not_advertise_an_unimplemented_model(self) -> None:
        from mriforge.config.schemas.physics import DataConsistencyConfig
        from mriforge.infrastructure.physics.dc_settings import SUPPORTED_NOISE_TYPES

        description = DataConsistencyConfig.model_fields["noise_type"].description or ""
        for shipped in SUPPORTED_NOISE_TYPES:
            assert shipped in description
        # 'rician' may be MENTIONED, but only as unimplemented.
        if "rician" in description:
            assert "never had an implementation" in description

    def test_default_noise_type_is_one_the_layers_implement(self) -> None:
        from mriforge.config.schemas.physics import DataConsistencyConfig
        from mriforge.infrastructure.physics.dc_settings import SUPPORTED_NOISE_TYPES

        assert DataConsistencyConfig().noise_type in SUPPORTED_NOISE_TYPES

    def test_noise_level_defaults_match_the_layer_defaults(self) -> None:
        """A divergence here is a silent default substitution once the knob is wired."""
        import inspect

        from mriforge.config.schemas.physics import DataConsistencyConfig
        from mriforge.infrastructure.physics.data_consistency import HardDataConsistency

        cfg = DataConsistencyConfig()
        params = inspect.signature(HardDataConsistency.__init__).parameters
        assert cfg.train_noise_level == params["train_noise_level"].default
        assert cfg.eval_noise_level == params["eval_noise_level"].default
