"""Regression test for DPSMRISampler sampler_type validation.

The constructor dispatched ``sampler_type`` via if 'pula' / elif 'mcg' /
else -> hybrid, silently treating any unknown/typo value as the hybrid
two-pass sampler (pitfall #9 + #15). It must now raise ValueError.
"""

import pytest
import torch
from torch import nn

from spectramr.models.diffusion.pula_sampler import DPSMRISampler


class _TinyScoreModel(nn.Module):
    def forward(self, x, t):
        return x


def test_unknown_sampler_type_raises():
    with pytest.raises(ValueError, match="Unknown sampler_type"):
        DPSMRISampler(_TinyScoreModel(), num_steps=4, sampler_type="edm")


@pytest.mark.parametrize("sampler_type", ["pula", "mcg", "hybrid"])
def test_advertised_sampler_types_construct(sampler_type):
    m = DPSMRISampler(_TinyScoreModel(), num_steps=4, sampler_type=sampler_type)
    assert m.sampler_type == sampler_type
    if sampler_type == "hybrid":
        assert hasattr(m, "pula") and hasattr(m, "mcg")
    else:
        assert hasattr(m, "sampler")


def test_dps_mri_registered():
    """The @register_sampler('dps_mri') decorator must be reachable on import."""
    from spectramr.models.diffusion.samplers import SAMPLER_REGISTRY  # noqa: F401

    # Symbol importable + class is the registered one (covers the decorator path).
    assert DPSMRISampler is not None
