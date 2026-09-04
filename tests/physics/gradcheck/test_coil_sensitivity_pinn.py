"""Tier-1 gradcheck tests for the SIREN / PINN coil sensitivity estimator.

``SirenSensNet`` (``spectramr.models.generators.siren_pinn``) is the coordinate
network used inside ``estimate_csm_pinn``.  Its forward pass maps (N, 2)
spatial coordinates to per-coil real and imaginary sensitivity values using a
chain of sinusoidal layers (SIREN) followed by a linear output layer.

Two gradcheck scenarios are tested:

1. **w.r.t. input coordinates** — checks that the Jacobian of the output
   w.r.t. the coordinate tensor is correctly computed by autograd.  This is
   the primary differentiability requirement for the PINN PDE loss, which
   computes second-order spatial derivatives via ``torch.autograd.grad``.

2. **w.r.t. network parameters (weight of first layer)** — verifies that the
   sinusoidal activations give correct parameter gradients. This is tested on
   the first-layer linear weights only (a small 2×H matrix) to keep the
   gradcheck tractable.

Notes
-----
* The PINN optimisation loop in ``estimate_csm_pinn`` is a full training loop
  (1500 Adam steps per batch item) — gradchecking the *loop* is infeasible.
  We gradcheck the *per-forward-call* differentiability of ``SirenSensNet``
  directly instead.
* ``SirenSensNet`` is constructed with a tiny configuration
  (hidden_features=8, hidden_layers=1) so gradcheck remains fast.
* The network is moved to fp64 via ``.double()`` — all ``nn.Linear`` weights
  are real-valued, so ``.double()`` correctly casts them.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from spectramr.models.generators.siren_pinn import SirenSensNet

_EPS = 1e-6
_ATOL = 1e-5

# Tiny network for gradcheck speed
_HIDDEN = 8
_LAYERS = 1
_NUM_COILS = 2          # out_features = num_coils * 2
_N_COORDS = 4           # number of spatial coordinates (H*W = 4 → 2×2 grid)


def _make_model() -> SirenSensNet:
    """Return a small double-precision SirenSensNet."""
    return SirenSensNet(
        in_channels=2,
        out_channels=_NUM_COILS * 2,
        hidden_features=_HIDDEN,
        hidden_layers=_LAYERS,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ).double()


def _make_coords() -> torch.Tensor:
    """Return fp64 coordinate tensor with requires_grad=True."""
    # 4 points on a 2×2 grid, normalised to [-1, 1]
    coords = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    return coords


# ---------------------------------------------------------------------------
# Gradcheck w.r.t. input coordinates
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_siren_wrt_input_coords_real_part():
    """Gradcheck of SirenSensNet forward w.r.t. coordinates — real part output."""
    torch.manual_seed(0)
    model = _make_model()
    coords = _make_coords()

    # Freeze all parameters so gradcheck only checks the input Jacobian
    for p in model.parameters():
        p.requires_grad_(False)

    def fn(coords_in):
        s_real, _s_imag = model(coords_in)
        return s_real  # (N, num_coils) real-valued

    assert gradcheck(fn, (coords,), eps=_EPS, atol=_ATOL, raise_exception=True)


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_siren_wrt_input_coords_imag_part():
    """Gradcheck of SirenSensNet forward w.r.t. coordinates — imaginary part output."""
    torch.manual_seed(0)
    model = _make_model()
    coords = _make_coords()

    for p in model.parameters():
        p.requires_grad_(False)

    def fn(coords_in):
        _s_real, s_imag = model(coords_in)
        return s_imag  # (N, num_coils) real-valued

    assert gradcheck(fn, (coords,), eps=_EPS, atol=_ATOL, raise_exception=True)


# ---------------------------------------------------------------------------
# Gradcheck w.r.t. first-layer parameters (weight + bias)
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
def test_siren_wrt_first_layer_weight():
    """Gradcheck w.r.t. the weight matrix of the first SIREN layer.

    The first layer applies  sin(ω₀ · (W x + b)).  We fix the input
    coordinates and check that autograd correctly differentiates through the
    sinusoidal activation w.r.t. W.
    """
    torch.manual_seed(0)
    model = _make_model()
    coords = _make_coords().detach()  # freeze coords; vary weights

    # Freeze everything except the first layer weight
    for name, p in model.named_parameters():
        p.requires_grad_(False)

    first_layer = model.net.Siren_0  # type: ignore[attr-defined]
    first_layer.linear.weight.requires_grad_(True)
    w = first_layer.linear.weight  # shape (hidden_features, 2) = (8, 2)

    def fn(weight):
        # Temporarily replace the weight, run forward, restore
        original = first_layer.linear.weight.data.clone()
        first_layer.linear.weight.data.copy_(weight)
        s_real, _ = model(coords)
        first_layer.linear.weight.data.copy_(original)
        return s_real

    assert gradcheck(fn, (w,), eps=_EPS, atol=_ATOL, raise_exception=True)


# ---------------------------------------------------------------------------
# estimate_csm_pinn — skipped: full optimisation loop
# ---------------------------------------------------------------------------


@pytest.mark.gradcheck
@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.skip(
    reason=(
        "estimate_csm_pinn (spectramr.infrastructure.physics.coil_sensitivity) runs "
        "a full zero-shot PINN optimisation loop (default 1500 Adam steps per "
        "batch item).  The function is not a simple differentiable callable — it "
        "contains an inner training loop with optimizer.step() calls and is not "
        "designed to be differentiated end-to-end.  A gradcheck on the outer "
        "function signature (kspace → smaps) would require differentiating through "
        "~1500 Adam steps and all the control-flow branches (ACS masking, GradNorm "
        "adaptive weighting), which is both infeasible and not the intended usage.  "
        "TODO: extract the per-forward-pass differentiable core "
        "(SirenSensNet + HelmholtzPDELoss) as a standalone callable and gradcheck "
        "that instead (already covered by test_siren_wrt_input_coords_* above)."
    )
)
def test_estimate_csm_pinn_skipped():
    """Placeholder: estimate_csm_pinn cannot be gradchecked as a whole."""
    raise NotImplementedError("This test should be skipped by the decorator above.")
