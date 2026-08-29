"""Smoke test: real sampler classes register themselves on import.

This test guards against regressions where a sampler stops being
discoverable through the canonical registry surface — e.g. someone
removes the ``@register_sampler`` decorator or moves the file to a
sub-package that is never imported.

Targets the ``@register_sampler`` decorations applied to
``DDIMSampler``, ``PhysicsGuidedReverseSampler``, ``PULASampler``,
``MCGSampler``, ``DPSMRISampler`` (CC-1 from
``TODO/backlog_paradigm_expansion_roadmap.md``).
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "module_path,canonical",
    [
        ("mriforge.models.diffusion.ddim_sampler", "ddim"),
        ("mriforge.models.diffusion.physics_guided_sampler", "physics_guided"),
        ("mriforge.models.diffusion.pula_sampler", "pula"),
    ],
)
def test_sampler_registers_on_import(module_path: str, canonical: str) -> None:
    """Importing the sampler module installs the canonical name in the registry."""
    import importlib

    importlib.import_module(module_path)
    from mriforge.models.diffusion.samplers import list_available

    assert canonical in list_available(), (
        f"Importing {module_path} should register sampler {canonical!r}"
    )


def test_pula_module_exposes_three_samplers() -> None:
    """``pula_sampler`` registers PULA, MCG, and DPS-MRI."""
    import importlib

    importlib.import_module("mriforge.models.diffusion.pula_sampler")
    from mriforge.models.diffusion.samplers import list_available

    available = set(list_available())
    assert {"pula", "mcg", "dps_mri"} <= available


def test_aliases_dispatch_to_canonical() -> None:
    """Common upper-case aliases route to the canonical samplers."""
    import importlib

    importlib.import_module("mriforge.models.diffusion.ddim_sampler")
    importlib.import_module("mriforge.models.diffusion.pula_sampler")
    from mriforge.models.diffusion.samplers import SamplerRegistry

    assert SamplerRegistry.is_registered("DDIM") is True
    assert SamplerRegistry.is_registered("PULA") is True
    assert SamplerRegistry.is_registered("DPS") is True
