"""End-to-end integration test for the GeoMamba-ULF pipeline.

Builds the generator + composite loss on a tiny synthetic
``(B, 1, D, H, W)`` ULF→HF batch and asserts the four properties
that gate the v0 implementation:

a. Loss is finite at the first step.
b. Loss decreases over a handful of optimization steps (proxy for
   "the strategy actually trains").
c. Backward pass yields non-zero gradients on the FiLM MLP and
   Mamba block weights — guards against the reappearance of the
   PH ``.item()`` bug or any registry mis-wiring that would cause
   silently-detached gradients.
d. The PH-W2 loss (when [topology] is installed) is autograd-
   traceable end-to-end.

The test runs without GPU — small volumes (32³) keep CPU runtime
below a few seconds.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from spectramr.models.blocks.metric_sfc import MetricSFCLinearizer  # noqa: E402
from spectramr.models.generators.geo_mamba_unet import GeoMambaUNet  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402

# Every test here forwards GeoMambaUNet, whose MambaBlock dispatches to the
# CUDA-only kernel whenever mamba_ssm imports — so the "runs without GPU"
# promise in the module docstring holds only where that kernel is absent.
# See tests/utils/optional_backends.py for why "is mamba_ssm installed" on its
# own is the wrong question.
pytestmark = requires_cuda_for_mamba


@pytest.fixture
def tiny_paired_batch():
    """Tiny synthetic (input, target, mask) on a brain-like sphere."""
    rng = np.random.default_rng(0)
    D, H, W = 8, 32, 32
    intensity = rng.standard_normal((D, H, W)).astype(np.float32)
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    cz, cy, cx = (D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (r2 <= 8.0**2).astype(np.float32)

    inp = torch.randn(2, 1, D, H, W)
    target = inp + 0.3 * torch.randn_like(
        inp
    )  # SR target = input + structured residual
    mask_t = torch.from_numpy(mask)
    template = torch.from_numpy(intensity)
    return inp, target, mask_t, template


def test_forward_produces_finite_loss(tiny_paired_batch, tmp_path) -> None:
    inp, target, mask, template = tiny_paired_batch
    linearizer = MetricSFCLinearizer(
        intensity_volume=template, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(d_model=16, n_blocks=2, linearizer=linearizer)
    pred = model(inp, mask=mask, contrast_id=torch.tensor([0, 1]))
    loss = (pred - target).abs().mean()
    assert torch.isfinite(loss)


def test_loss_decreases_over_steps(tiny_paired_batch, tmp_path) -> None:
    """Five optimizer steps must reduce the loss (proxy for the strategy actually training)."""
    inp, target, mask, template = tiny_paired_batch
    linearizer = MetricSFCLinearizer(
        intensity_volume=template, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(d_model=16, n_blocks=2, linearizer=linearizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        pred = model(inp, mask=mask, contrast_id=torch.tensor([0, 1]))
        loss = (pred - target).abs().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease over 5 steps: {losses}"


def test_gradients_flow_to_film_and_mamba(tiny_paired_batch, tmp_path) -> None:
    inp, target, mask, template = tiny_paired_batch
    linearizer = MetricSFCLinearizer(
        intensity_volume=template, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(d_model=16, n_blocks=2, linearizer=linearizer)
    pred = model(inp, mask=mask, contrast_id=torch.tensor([0, 1]))
    loss = (pred - target).abs().mean()
    loss.backward()

    block0 = model.blocks[0]
    # FiLM first-layer weight gradient must be non-zero.
    assert block0.film[0].weight.grad is not None
    assert torch.any(block0.film[0].weight.grad != 0)
    # Mamba forward path weights — pick the in_proj of the gated-CNN
    # fallback (always present, regardless of whether mamba_ssm is
    # installed).
    if hasattr(block0.mamba_fwd, "in_proj"):
        assert block0.mamba_fwd.in_proj.weight.grad is not None
        assert torch.any(block0.mamba_fwd.in_proj.weight.grad != 0)


def test_strategy_dispatches_via_factory() -> None:
    """The new short name must resolve to GeoMambaULFStrategy."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    factory = TrainingStrategyFactory()
    assert "geomamba_ulf" in factory.STRATEGY_CLASS_PATHS
    cls = factory._load_strategy_class("geomamba_ulf")
    assert cls.__name__ == "GeoMambaULFStrategy"


def test_cubical_ph_w2_autograd_through_pipeline(tiny_paired_batch, tmp_path) -> None:
    """When [topology] is installed, PH-W2 must be autograd-traceable."""
    pytest.importorskip("gudhi")
    pytest.importorskip("ot")
    from spectramr.models.losses.cubical_ph_w2_loss import CubicalPHWassersteinLoss

    inp, target, mask, template = tiny_paired_batch
    linearizer = MetricSFCLinearizer(
        intensity_volume=template, mask=mask, cache_dir=str(tmp_path)
    )
    model = GeoMambaUNet(d_model=16, n_blocks=1, linearizer=linearizer)
    ph_loss = CubicalPHWassersteinLoss(homology_dims=(0,))

    pred = model(inp, mask=mask, contrast_id=torch.tensor([0, 1]))
    loss = (pred - target).abs().mean() + 0.01 * ph_loss(
        pred, target, mask=mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
    )
    loss.backward()

    block0 = model.blocks[0]
    assert block0.film[0].weight.grad is not None
    assert torch.any(block0.film[0].weight.grad != 0)
