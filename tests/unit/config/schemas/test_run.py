"""The ``run:`` block, and the three things it is easy to get wrong.

Four scalars sat bare on :class:`TrainingSettings`. Two of them describe the run
(``seed``, ``device``) and moved here; the other two were a write-only alias and
a stray loss weight, and moved to where they are read. ``config_version`` is
declared here but stays a root-level key in the file — the reasoning is in
:mod:`mriforge.config.schemas.run`, and the tests below pin it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.base import CANONICAL_CONFIG_VERSION
from mriforge.config.schemas.run import RunConfigSchema
from mriforge.config.settings import TrainingSettings

#: The four blocks `TrainingSettings` requires. Kept minimal so a failure here
#: is about the `run:` move, not about an unrelated required field.
#: The only accepted version since 2026-08-08. These fixtures used the
#: literal "6.1" as a stand-in for "some valid version"; that spelling is
#: now refused, and a stand-in must not be a value under test.
_V = CANONICAL_CONFIG_VERSION

_MINIMAL = {
    "model": {"model_type": "unet"},
    "data": {"dataset_type": "image"},
    "optimization": {},
    "logging": {},
}


class TestDefaultsAreCarriedVerbatim:
    def test_device_defaults_to_cuda_not_auto(self) -> None:
        """The single highest-risk line in the move.

        ZERO corpus files declare ``device``, so every arm reaches
        ``resolve_compute_device`` on this default. ``cuda`` is an *explicit*
        request; ``auto`` is not, and the two take different branches and stamp
        different provenance. Normalising it while relocating the field would
        change device resolution corpus-wide inside a diff that reads as a move.
        """
        assert RunConfigSchema().device == "cuda"

    def test_seed_defaults_to_42(self) -> None:
        assert RunConfigSchema().seed == 42


class TestConfigVersionIsPopulatedNotAuthored:
    def test_the_root_key_populates_the_field(self) -> None:
        """It used to be validated and then ``del``'d, so it never existed on the
        object and survived only into provenance.

        Updated 2026-08-03: the field is populated with the **folded** version,
        not the authored one. Until 2026-08-08 that distinction was load-bearing:
        a legacy 6.x file bound ``1.0`` here, which is what made a version bump a
        provable no-op on the resolved document. With the legacy tier drained the
        two coincide -- the only accepted spelling is the canonical one -- so what
        remains to pin is that the version lands on ``run`` at all, and that a
        retired spelling is refused rather than bound.
        """
        from mriforge.config.schemas.base import CANONICAL_CONFIG_VERSION

        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.settings_from_dict(
                {"config_version": "6.0", **_MINIMAL}
            )

        canonical = TrainingSettings.settings_from_dict(
            {"config_version": CANONICAL_CONFIG_VERSION, **_MINIMAL}
        )
        assert canonical.run.config_version == CANONICAL_CONFIG_VERSION

    def test_authoring_it_under_run_raises(self) -> None:
        """Otherwise the block meant to remove a second spelling introduces one."""
        with pytest.raises((ValidationError, ValueError)) as exc:
            TrainingSettings.settings_from_dict(
                {
                    "config_version": _V,
                    "run": {"config_version": _V},
                    **_MINIMAL,
                }
            )
        assert "config_version" in str(exc.value)

    def test_the_message_says_where_the_version_goes(self) -> None:
        with pytest.raises((ValidationError, ValueError)) as exc:
            TrainingSettings.settings_from_dict(
                {"config_version": _V, "run": {"config_version": "6.0"}, **_MINIMAL}
            )
        msg = str(exc.value)
        assert "top level" in msg
        assert "pre-schema discriminator" in msg


class TestTheRetiredRootScalars:
    """All four raise, and each names where its value now lives.

    ``TrainingSettings`` is ``extra="forbid"``, so these would already fail —
    with "Extra inputs are not permitted", which does not tell an author that
    the value is still wanted under a different key.
    """

    @pytest.mark.parametrize(
        ("legacy", "value", "canonical"),
        [
            ("seed", 7, "run.seed"),
            ("device", "cpu", "run.device"),
            ("model_domain", "kspace", "model.model_domain"),
            ("deep_supervision_weight", 0.5, "losses.lambda_deep_supervision"),
        ],
    )
    def test_raises_naming_its_replacement(
        self, legacy: str, value: object, canonical: str
    ) -> None:
        with pytest.raises((ValidationError, ValueError)) as exc:
            TrainingSettings.settings_from_dict(
                {"config_version": _V, legacy: value, **_MINIMAL}
            )
        msg = str(exc.value)
        assert canonical in msg
        assert "migrate_config_keys.py" in msg
        assert "Extra inputs" not in msg, (
            "extra='forbid' answered first — the root shim must be a "
            "mode='before' validator so it sees the key at all."
        )


class TestTheCanonicalKeysWork:
    def test_run_seed_and_device_load(self) -> None:
        settings = TrainingSettings.settings_from_dict(
            {"config_version": _V, "run": {"seed": 7, "device": "cpu"}, **_MINIMAL}
        )
        assert settings.run.seed == 7
        assert settings.run.device == "cpu"

    def test_deep_supervision_moved_to_the_loss_weight_ssot(self) -> None:
        settings = TrainingSettings.settings_from_dict(
            {
                "config_version": _V,
                "losses": {"lambda_deep_supervision": 0.5},
                **_MINIMAL,
            }
        )
        assert settings.losses.lambda_deep_supervision == 0.5


class TestThereIsOnlyOneSeed:
    def test_training_seed_is_gone(self) -> None:
        """``training.seed`` WON over the root ``seed:`` at runtime, so a
        canonical home that lost to it would be a third spelling, not a fix."""
        with pytest.raises((ValidationError, ValueError)) as exc:
            TrainingSettings.settings_from_dict(
                {"config_version": _V, "training": {"seed": 7}, **_MINIMAL}
            )
        assert "run.seed" in str(exc.value)

    def test_the_pipeline_reads_run_seed(self) -> None:
        """Guards against the move landing a field nothing consumes. The reader
        had a ``config.training.seed`` fallback that took precedence; if it
        survived, ``run.seed`` would be inert on day one."""
        import inspect

        from mriforge.pipelines import train

        src = inspect.getsource(train.run_training_pipeline)
        assert "config.run.seed" in src
        assert "config.training.seed" not in src, (
            "the training.seed fallback is still in the seeding path, so it can "
            "still win over run.seed"
        )


def test_the_not_authored_here_message_names_the_canonical_version() -> None:
    """The error tells an author where to put the version -- so it must not tell
    them to put a LEGACY one there.

    A user-facing string is a gate too. This one is the only instruction most
    authors will read about where ``config_version`` lives, and it named '6.1'
    for as long as 6.1 was current; left alone it would keep manufacturing new
    legacy configs while the corpus is being drained of them.
    """
    from mriforge.config.schemas.base import CANONICAL_CONFIG_VERSION
    from mriforge.config.schemas.run import CONFIG_VERSION_NOT_AUTHORED_HERE

    assert f"config_version: '{CANONICAL_CONFIG_VERSION}'" in (
        CONFIG_VERSION_NOT_AUTHORED_HERE
    )
    # Literals rather than the (now empty) constant: the message must never
    # teach a retired spelling, and an empty set would make this vacuous.
    for retired in ("6.0", "6.1"):
        assert f"'{retired}'" not in CONFIG_VERSION_NOT_AUTHORED_HERE
