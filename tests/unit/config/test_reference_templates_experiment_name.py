"""The v1.0 reference template must load cleanly through ``from_yaml()``.

``src/spectramr/config/schemas/templates/v1.0_reference.yaml`` is the schema SSOT
and the live "copy-and-edit" surface a new contributor starts from.
``TrainingSettings`` is ``extra="forbid"`` — every key MUST be a real Pydantic
field — so any silent schema drift makes the template itself unloadable. That is
the point of loading it here: prose can drift, an executed file cannot.

This module asserts three contracts:

1. ``experiment_name:`` is NEVER declared at top level. The only schema-defined
   spelling lives on :class:`~spectramr.config.schemas.logging.LoggingConfigSchema`
   (at ``logging.identity.experiment`` since phase 10b), so a top-level key was
   the original 2026-05-28 regression that fooled ``extra="forbid"``.
2. The template round-trips through
   :meth:`spectramr.config.settings.TrainingSettings.from_yaml`, and the value
   arrives at its CANONICAL home.
3. The template declares ``CANONICAL_CONFIG_VERSION`` — a reference that teaches
   a legacy version is worse than no reference, since it is the file people copy.

**On the retired v6.0/v6.1 templates.** This module used to pin two files. They
were consolidated into ``v1.0_reference.yaml``, which is a rename of the v6.1
content, not a merge: of v6.0's 239 documented paths, 41 were unique to it and
all 41 were legacy spellings whose canonical target v6.1 already carried — zero
had a canonical target absent from it. So nothing it documented was lost, and
what it uniquely taught was the retired spelling. Its former drift (lowercase
``level: 'INFO'`` against the ``LogLevel`` enum, ``dataset_type: 'fastmri'``,
legacy ``src.infrastructure...`` strategy paths after the ``src→spectramr``
rename) is history rather than a live risk, but the round-trip below is what
would catch its recurrence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import spectramr.config.schemas as _schemas

# Anchored off the package, not the repo root, so the test runs from any cwd.
TEMPLATES = Path(str(_schemas.__file__)).parent / "templates"
V1_0 = TEMPLATES / "v1.0_reference.yaml"


@pytest.fixture(scope="module")
def v1_0_dict() -> dict:
    return yaml.safe_load(V1_0.read_text(encoding="utf-8"))


def test_the_reference_template_exists_under_its_canonical_name() -> None:
    """A rename that leaves the file unreachable is worse than no rename."""
    assert V1_0.is_file(), f"{V1_0} is missing"


def test_only_one_reference_template_ships() -> None:
    """Two references are two SSOTs, which is none.

    The retired 6.x pair is the reason this assertion exists: they diverged by
    404 paths while both claiming to be canonical.
    """
    shipped = sorted(p.name for p in TEMPLATES.glob("*_reference.yaml"))
    assert shipped == ["v1.0_reference.yaml"], shipped


def test_no_top_level_experiment_name(v1_0_dict: dict) -> None:
    assert "experiment_name" not in v1_0_dict, (
        "v1.0_reference.yaml declares a top-level ``experiment_name`` — "
        "TrainingSettings is extra=forbid and this key only exists under "
        "``logging:``."
    )


def test_the_template_declares_the_canonical_version(v1_0_dict: dict) -> None:
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

    assert str(v1_0_dict.get("config_version")) == CANONICAL_CONFIG_VERSION


def test_reference_template_round_trips_through_training_settings() -> None:
    """The whole point: schema drift that breaks the reference fails here."""
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.from_yaml(V1_0)
    assert settings is not None
    # Phase 10b moved the leaf: ``logging.experiment_name`` ->
    # ``logging.identity.experiment``. Asserting the CANONICAL home is what
    # makes this a round-trip check rather than a document check.
    assert settings.logging.identity.experiment == "v1_0_reference_config"


def test_the_reference_is_free_of_legacy_spellings() -> None:
    """A reference must not teach a key the corpus is draining.

    Resolved through ``RENAMES`` rather than a hand-written list, so a record
    added tomorrow is covered without editing this test.
    """
    from spectramr.config.schemas.renames import RENAMES

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                yield path
                yield from walk(value, path)

    doc = yaml.safe_load(V1_0.read_text(encoding="utf-8"))
    legacy = sorted(p for p in walk(doc) if p in RENAMES)
    assert (
        not legacy
    ), "the reference template declares retired spellings: " + ", ".join(legacy)
