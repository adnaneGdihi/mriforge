"""``execution: array`` campaign mode — schema surface.

The campaign gains a third execution mode, ``array``, that submits the whole
cohort as ONE SLURM job array (a single tracked job id + ``%N`` concurrency
throttle) instead of the ``parallel`` mode's N independent ``sbatch`` calls.
This file pins the schema contract only; the orchestrator submission path is
covered in
``tests.unit.infrastructure.orchestration.test_campaign_orchestrator``.

These tests import ONLY ``mriforge.config.schemas.campaign`` (no evaluator /
torch chain), so they collect on a torch-less interpreter.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mriforge.config.schemas.campaign import (
    CampaignConfigSchema,
    CampaignExperimentSchema,
)


def _arm(name: str, cfg: str = "a.yaml") -> CampaignExperimentSchema:
    return CampaignExperimentSchema(name=name, config=cfg)


def test_array_is_a_valid_execution_mode() -> None:
    # Like 'parallel', 'array' carries a flat experiments list.
    config = CampaignConfigSchema(
        name="arr",
        execution="array",
        experiments=[_arm("a"), _arm("b", "b.yaml")],
    )
    assert config.execution == "array"
    assert len(config.enabled_experiments) == 2


def test_array_concurrency_defaults_to_8() -> None:
    config = CampaignConfigSchema(
        name="arr", execution="array", experiments=[_arm("a")]
    )
    assert config.array_concurrency == 8


def test_array_concurrency_carries_through_parallel_mode_too() -> None:
    # The knob is defined on the top-level schema, so a parallel campaign can
    # still declare it (harmless there) — its default is the same.
    config = CampaignConfigSchema(name="p", experiments=[_arm("a")])
    assert config.array_concurrency == 8


def test_array_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CampaignConfigSchema(
            name="arr",
            execution="array",
            array_concurrency=0,
            experiments=[_arm("a")],
        )


def test_array_mode_requires_experiments() -> None:
    # Empty 'array' campaign is the same error class as empty 'parallel'.
    with pytest.raises(ValidationError):
        CampaignConfigSchema(name="arr", execution="array")


def test_array_mode_rejects_stage_groups() -> None:
    # stage_groups belong to 'sequential'; mixing them with 'array' is a config
    # error (array uses the flat experiments list).
    with pytest.raises(ValidationError):
        CampaignConfigSchema(
            name="arr",
            execution="array",
            stage_groups=[
                {
                    "name": "g",
                    "experiments": [{"name": "a", "config": "a.yaml"}],
                }
            ],
        )
