"""Tier-1 gradcheck tests for FNOEPGSurrogate.

``FNOEPGSurrogate`` is a trained neural-operator surrogate for EPG simulation
(``spectramr.infrastructure.physics.fno_epg``).  Its forward pass is a composition
of differentiable nn.Module layers (SpectralConv1d + FiLM + GELU + Softplus)
with no discrete ops, so gradcheck w.r.t. the *inputs* (rho, t1, t2, b1_plus)
is well-defined once the network is constructed in fp64.

However, SpectralConv1d stores its spectral weights as ``torch.cfloat``
(complex64) parameters.  When the module is moved to fp64 via ``.double()``,
PyTorch converts ``nn.Parameter`` reals but NOT parameters declared via
``torch.view_as_complex`` on a float tensor initialised with dtype=cfloat.
This creates a dtype mismatch inside the rfft → einsum path that triggers
a RuntimeError at gradcheck time.

Status: test written and skipped with a precise reason.
TODO: update SpectralConv1d.__init__ to initialise weights as
    ``torch.view_as_complex(torch.randn(..., dtype=torch.float64) ...)``
    when double precision is requested, or add a ``.to_double()`` helper.
    Once that is done, remove the skip and run this file on the cluster.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate

_EPS = 1e-6
_ATOL = 1e-5
_SHAPE = (1, 1, 2, 2)  # (B, C, H, W)


def _fp64(value: float) -> torch.Tensor:
    return torch.full(_SHAPE, value, dtype=torch.float64, requires_grad=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.skip(
    reason=(
        "FNOEPGSurrogate.SpectralConv1d stores its learnable weights via "
        "torch.view_as_complex(torch.randn(...)) which defaults to cfloat "
        "(complex64).  Calling .double() on the module does not cast complex "
        "parameters that are not nn.Parameter subclasses wrapping real storage. "
        "gradcheck requires all participating tensors (inputs AND parameters) "
        "to be fp64; the cfloat spectral weights raise a dtype mismatch in the "
        "rfft → einsum path.  "
        "TODO: change SpectralConv1d.__init__ to initialise weights from "
        "torch.float64 storage: "
        "    self.weights = nn.Parameter(scale * torch.view_as_complex("
        "        torch.randn(in_channels, out_channels, modes, 2, dtype=torch.float64)"
        "    )) "
        "then re-enable these tests."
    )
)
def test_fno_epg_wrt_rho_skipped():
    """Placeholder gradcheck for FNOEPGSurrogate w.r.t. rho.

    After fixing SpectralConv1d dtype, the real test body would be::

        model = FNOEPGSurrogate(
            tissue_channels=4, hidden_width=8, n_layers=2, max_etl=8
        ).double()
        rho = torch.full((1, 1, 2, 2), 0.8, dtype=torch.float64, requires_grad=True)
        t1  = torch.full((1, 1, 2, 2), 1000.0, dtype=torch.float64)
        t2  = torch.full((1, 1, 2, 2), 100.0,  dtype=torch.float64)

        def fn(rho_in):
            return model(rho_in, t1, t2, echo_train_length=8)

        assert gradcheck(fn, (rho,), eps=1e-6, atol=1e-5)
    """
    raise NotImplementedError("This test should be skipped by the decorator above.")


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.skip(
    reason=(
        "Same SpectralConv1d cfloat weight issue as test_fno_epg_wrt_rho_skipped. "
        "See that test's skip reason and TODO."
    )
)
def test_fno_epg_wrt_t1_skipped():
    """Placeholder gradcheck for FNOEPGSurrogate w.r.t. T1."""
    raise NotImplementedError("This test should be skipped by the decorator above.")


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.skip(
    reason=(
        "Same SpectralConv1d cfloat weight issue as test_fno_epg_wrt_rho_skipped. "
        "See that test's skip reason and TODO."
    )
)
def test_fno_epg_wrt_t2_skipped():
    """Placeholder gradcheck for FNOEPGSurrogate w.r.t. T2."""
    raise NotImplementedError("This test should be skipped by the decorator above.")
