"""Tests for the ``cold_mri`` sampler registration.

Audit finding B3: every cold-diffusion YAML under
``experiments/inprogress/{kspace_filling,diffusion}/`` declares
``training.diffusion.sampler: cold_mri`` but no ``@register_sampler``
decoration existed. These tests pin the registration so a future
refactor cannot silently remove it again.
"""

from __future__ import annotations

import inspect
from unittest.mock import Mock

import pytest
import torch

# Importing the module triggers ``@register_sampler`` registration.
from mriforge.models.diffusion import cold_mri_sampler  # noqa: F401
from mriforge.models.diffusion.cold_mri_sampler import ColdMRISampler
from mriforge.models.diffusion.samplers import SamplerRegistry, get_sampler


def test_cold_mri_is_registered() -> None:
    """The canonical name ``cold_mri`` resolves through the registry."""
    assert SamplerRegistry.is_registered("cold_mri")
    assert SamplerRegistry.get_class("cold_mri") is ColdMRISampler


def test_cold_mri_alias_is_registered() -> None:
    """The ``ColdMRI`` alias dispatches to the same class."""
    assert SamplerRegistry.is_registered("ColdMRI")
    assert SamplerRegistry.get_class("ColdMRI") is ColdMRISampler


def test_get_sampler_returns_object_with_sample_method() -> None:
    """``get_sampler('cold_mri', ...)`` instantiates a usable sampler."""
    sampler = get_sampler(
        "cold_mri",
        kspace_log_scaled=False,
        model=Mock(),
        num_timesteps=10,
    )
    assert isinstance(sampler, ColdMRISampler)
    assert hasattr(sampler, "sample")
    assert callable(sampler.sample)


def test_constructor_forwards_kwargs() -> None:
    """Constructor kwargs land on the wrapped diffusion object."""
    sampler = get_sampler(
        "cold_mri",
        kspace_log_scaled=False,
        model=Mock(),
        num_timesteps=7,
        max_acceleration=4.0,
        center_fraction=0.08,
        dc_method="soft",
        dc_weight=0.5,
        sampling_steps=3,
    )
    inner = sampler._diffusion
    assert inner.num_timesteps == 7
    assert inner.dc_method == "soft"
    assert inner.dc_weight == pytest.approx(0.5)
    assert inner.sampling_steps == 3


def test_reverse_mode_kwargs_forwarded() -> None:
    """``reverse_mode`` / ``reverse_clip_ratio`` reach the wrapped diffusion."""
    sampler = get_sampler(
        "cold_mri",
        kspace_log_scaled=False,
        model=Mock(),
        num_timesteps=7,
        reverse_mode="replace_freeze",
        reverse_clip_ratio=2.5,
    )
    inner = sampler._diffusion
    assert inner.reverse_mode == "replace_freeze"
    assert inner.reverse_clip_ratio == pytest.approx(2.5)


def test_reverse_mode_defaults_to_additive() -> None:
    """Omitting the knob preserves the legacy additive loop (no behavior change)."""
    sampler = get_sampler("cold_mri", model=Mock(), num_timesteps=7, kspace_log_scaled=False)
    assert sampler._diffusion.reverse_mode == "additive"


# ---------------------------------------------------------------------------
# ``start_timestep`` forwarding (#1422)
#
# ``KSpaceColdDiffusionGenerator.sample`` decides whether to forward the
# caller's trajectory head with
# ``"start_timestep" in inspect.signature(sampler.sample).parameters``. This
# wrapper omitted the parameter and declares no ``**kwargs``, so the argument
# was dropped and the #535/#1388 cascading-validation fix
# (``diffusion.py`` -> ``start_timestep=t_used``) never reached
# ``PhysicsInformedColdDiffusion.sample``, which has always supported it.
#
# Two INDEPENDENT failure shapes are pinned below, because fixing only the
# first makes the symptom WORSE than the bug: adding the parameter to the
# signature silences the generator's warning while the argument is still
# dropped, converting a loud defect into a silent one (pitfall #9).
#   shape A -- absent from the signature  -> generator's gate fails, warns
#   shape B -- present but not forwarded  -> gate passes, argument vanishes
# ---------------------------------------------------------------------------


def _make_sampler(num_timesteps: int = 10) -> ColdMRISampler:
    """A registry-resolved sampler whose wrapped diffusion can be spied on."""
    return get_sampler(
        "cold_mri",
        model=Mock(),
        num_timesteps=num_timesteps,
        kspace_log_scaled=False,
    )


def test_sample_signature_exposes_start_timestep() -> None:
    """Shape A: the generator's ``inspect.signature`` gate must pass.

    This is the exact predicate at
    ``KSpaceColdDiffusionGenerator.sample``; if it is False the generator logs
    "sampler 'cold_mri' does not accept 'start_timestep'" and drops the value.
    """
    params = inspect.signature(SamplerRegistry.get_class("cold_mri").sample).parameters
    assert "start_timestep" in params


def test_signature_gate_discriminates() -> None:
    """The shape-A predicate must be capable of returning False.

    A gate nobody has watched fail is not a gate (non-negotiable 15). This
    plants the pre-fix signature and asserts the predicate rejects it, so the
    assertion above cannot pass vacuously.
    """

    class _PreFixSampler:
        def sample(self, measurement, mask, return_trajectory=False):
            """The signature this wrapper shipped before #1422."""

    assert "start_timestep" not in inspect.signature(_PreFixSampler.sample).parameters


def test_start_timestep_reaches_the_wrapped_diffusion() -> None:
    """Shape B: the value must ARRIVE at ``PhysicsInformedColdDiffusion.sample``.

    Spy on the wrapped object, never on the wrapper -- reading the wrapper is
    precisely what kept this invisible: ``inspect.signature`` reports
    ``ColdMRISampler``, while the class that implements the parameter is the
    ``PhysicsInformedColdDiffusion`` it delegates to.
    """
    sampler = _make_sampler()
    spy = Mock(return_value=torch.zeros(1, 2, 4, 4))
    sampler._diffusion.sample = spy

    sampler.sample(
        measurement=torch.zeros(1, 2, 4, 4),
        mask=torch.ones(1, 1, 4, 4),
        start_timestep=3,
    )

    assert spy.call_args.kwargs["start_timestep"] == 3


def test_start_timestep_defaults_to_none() -> None:
    """Omitting it forwards ``None`` -- the fully-degraded head, unchanged."""
    sampler = _make_sampler()
    spy = Mock(return_value=torch.zeros(1, 2, 4, 4))
    sampler._diffusion.sample = spy

    sampler.sample(
        measurement=torch.zeros(1, 2, 4, 4),
        mask=torch.ones(1, 1, 4, 4),
    )

    assert spy.call_args.kwargs["start_timestep"] is None


def test_return_trajectory_still_forwarded() -> None:
    """The pre-existing kwarg must survive the signature change."""
    sampler = _make_sampler()
    spy = Mock(return_value=(torch.zeros(1, 2, 4, 4), []))
    sampler._diffusion.sample = spy

    sampler.sample(
        measurement=torch.zeros(1, 2, 4, 4),
        mask=torch.ones(1, 1, 4, 4),
        return_trajectory=True,
    )

    assert spy.call_args.kwargs["return_trajectory"] is True


def test_wrapper_still_declares_no_var_keyword() -> None:
    """The hard-filter property is deliberate and must not be widened.

    The docstring pins ``**kwargs`` as ABSENT on purpose: the parameter list is
    the filter that makes dropped knobs (``sampler_sigma`` / ``sampler_seed`` /
    ``selection_rule``, issue #1286) visible instead of silently swallowed.
    Fixing #1422 with ``**kwargs`` would close this issue by hiding #1286.
    """
    kinds = [p.kind for p in inspect.signature(ColdMRISampler.sample).parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds
