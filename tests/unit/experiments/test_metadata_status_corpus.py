"""Every tracked inprogress arm declares its status in the closed vocabulary, one spelling.

Corpus driver for the ``ExperimentMetadataSchema.status`` contract (cohort
review 2026-09-02, T0.4). Raw YAML, so it needs no data and runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spectramr.config.schemas.base import EXPERIMENT_STATUSES
from tests.utils.corpus import tracked_yamls

_ARMS = tracked_yamls("experiments/inprogress")


def _arm_id(path: Path) -> str:
    parts = path.parts
    return "/".join(parts[parts.index("inprogress") + 1 :]) if "inprogress" in parts else path.name


@pytest.mark.skipif(not _ARMS, reason="experiments/inprogress not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_status_is_canonical_and_singly_spelled(arm: Path) -> None:
    doc = yaml.safe_load(arm.read_text()) or {}
    metadata = doc.get("metadata") or {}
    if not isinstance(metadata, dict):
        pytest.skip("no metadata block")
    tags = metadata.get("tags")
    assert not (isinstance(tags, dict) and "status" in tags), (
        f"{_arm_id(arm)}: metadata.tags.status is retired; use metadata.status"
    )
    status = metadata.get("status")
    if status is None:
        pytest.skip("arm declares no status")
    assert status in EXPERIMENT_STATUSES, f"{_arm_id(arm)}: status={status!r} is not canonical"
