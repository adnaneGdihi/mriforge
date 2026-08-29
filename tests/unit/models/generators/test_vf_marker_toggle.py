"""use_marker toggle on the 5 VF baseline-pair models.

Ships the paired ablation baselines (pitfall #17): each model already branches
on ``marker_prior is None`` (blind Wiener / unseeded PRELUDE / native-ACS GRAPPA
/ uniform complex-sum / no de-apodization). ``use_marker=False`` forces that path
so the baseline is a one-knob config change. These pin the contract: with
``use_marker=False`` the model is INVARIANT to whether a marker is supplied.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.vf_field_generators import GraphCutUnwrapGenerator
from mriforge.models.generators.vf_kspace_operators import (
    MarkerGRAPPAConvolution,
    MTFDeApodizationOperator,
)
from mriforge.models.generators.vf_reconstruction_generators import (
    NeuralComplexSumGenerator,
    WienerUNetGenerator,
)

MODELS = [
    WienerUNetGenerator,
    NeuralComplexSumGenerator,
    GraphCutUnwrapGenerator,
    MTFDeApodizationOperator,
    MarkerGRAPPAConvolution,
]


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_use_marker_false_ignores_marker(cls):
    torch.manual_seed(0)
    m = cls(in_channels=2, out_channels=2, use_marker=False).eval()
    x = torch.randn(2, 2, 32, 32)
    marker = torch.randn(2, 1, 32, 32, dtype=torch.complex64) + 1.0
    with torch.no_grad():
        out_with_marker = m(x.clone(), marker_prior=marker)
        out_no_marker = m(x.clone())
    # use_marker=False forces the no-marker branch → supplying a marker changes
    # nothing (the baseline is genuinely marker-free, not a facade).
    assert torch.allclose(out_with_marker, out_no_marker, atol=1e-5), (
        f"{cls.__name__}: use_marker=False must drop the marker"
    )


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_use_marker_default_true_and_stored(cls):
    assert cls(in_channels=2, out_channels=2).use_marker is True
    assert cls(in_channels=2, out_channels=2, use_marker=False).use_marker is False
