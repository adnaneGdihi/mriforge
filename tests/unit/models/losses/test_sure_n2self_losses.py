"""Tests for the self-supervised denoising losses ``SURELoss`` / ``Noise2SelfLoss``.

Targets ``mriforge.models.losses.sure_n2self_losses``. SURE (Stein's Unbiased
Risk Estimator) needs both the denoiser and its noisy input via
``context["model"]`` / ``context["noisy_input"]``; its Hutchinson divergence
term re-evaluates the model on a perturbed input. When the model returns a
dict, the loss extracts ``"output"`` or ``"reconstruction"`` — and must FAIL
LOUD (KeyError) rather than crash later with an opaque ``TypeError`` if neither
key is present (CLAUDE.md pitfall #9: no silent degradation).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.sure_n2self_losses import GSUREKspaceLoss, SURELoss

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_sigma_raises() -> None:
    """σ ≤ 0 fails loud — the noise std must be positive."""
    with pytest.raises(ValueError, match="sigma"):
        SURELoss(sigma=0.0)


def test_invalid_n_samples_raises() -> None:
    """n_samples < 1 fails loud."""
    with pytest.raises(ValueError, match="n_samples"):
        SURELoss(sigma=0.1, n_samples=0)


# ---------------------------------------------------------------------------
# Model-output contract (the regression: KeyError, not TypeError)
# ---------------------------------------------------------------------------


def test_dict_output_missing_keys_raises_keyerror() -> None:
    """A model dict lacking 'output'/'reconstruction' must raise KeyError.

    Pre-fix, ``pert.get("output", pert.get("reconstruction"))`` returned None
    and the subsequent ``(pert - prediction.detach())`` raised an opaque
    ``TypeError``. Post-fix, the loss raises a clear ``KeyError`` naming the
    expected keys, far closer to the cause.
    """
    torch.manual_seed(0)
    loss_fn = SURELoss(sigma=0.1, n_samples=1)
    y = torch.randn(2, 1, 4, 4)
    prediction = torch.randn(2, 1, 4, 4)

    def bad_model(x: torch.Tensor) -> dict:
        # Dict without 'output' or 'reconstruction'.
        return {"foo": x}

    context = {"model": bad_model, "noisy_input": y}
    with pytest.raises(KeyError, match="output.*reconstruction|reconstruction"):
        loss_fn(prediction, prediction, context=context)


def test_dict_output_missing_keys_is_not_typeerror() -> None:
    """The fix must NOT surface as a TypeError (the old, misleading failure)."""
    torch.manual_seed(0)
    loss_fn = SURELoss(sigma=0.1, n_samples=1)
    y = torch.randn(2, 1, 4, 4)
    prediction = torch.randn(2, 1, 4, 4)

    def bad_model(x: torch.Tensor) -> dict:
        return {"foo": x}

    context = {"model": bad_model, "noisy_input": y}
    with pytest.raises(Exception) as exc_info:
        loss_fn(prediction, prediction, context=context)
    assert not isinstance(exc_info.value, TypeError)


# ---------------------------------------------------------------------------
# Happy paths: dict with a valid key, and a plain-tensor model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["output", "reconstruction"])
def test_dict_output_with_valid_key_runs(key: str) -> None:
    """A dict carrying 'output' or 'reconstruction' produces a finite scalar."""
    torch.manual_seed(0)
    loss_fn = SURELoss(sigma=0.1, n_samples=1)
    y = torch.randn(2, 1, 4, 4)
    prediction = torch.randn(2, 1, 4, 4)

    def good_model(x: torch.Tensor) -> dict:
        return {key: x}

    out = loss_fn(prediction, prediction, context={"model": good_model, "noisy_input": y})
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_plain_tensor_model_runs() -> None:
    """A model returning a bare tensor (no dict) is unaffected by the fix."""
    torch.manual_seed(0)
    loss_fn = SURELoss(sigma=0.1, n_samples=1)
    y = torch.randn(2, 1, 4, 4)
    prediction = torch.randn(2, 1, 4, 4)

    def good_model(x: torch.Tensor) -> torch.Tensor:
        return x

    out = loss_fn(prediction, prediction, context={"model": good_model, "noisy_input": y})
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Context validation (existing contract, kept green)
# ---------------------------------------------------------------------------


def test_missing_context_raises() -> None:
    """No context → ValueError naming the required keys."""
    loss_fn = SURELoss(sigma=0.1)
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(ValueError, match="context"):
        loss_fn(x, x, context=None)


def test_registered_under_canonical_name() -> None:
    """The ``@register_loss`` decorator registers 'sure'."""
    from mriforge.models.losses.registry import list_available

    available = list_available()
    assert "sure" in available, f"'sure' not in registry; available: {available[:10]}"


# ---------------------------------------------------------------------------
# Divergence term carries gradient (regression: it was fully detached)
# ---------------------------------------------------------------------------


def test_divergence_term_contributes_gradient() -> None:
    r"""The Hutchinson divergence must back-propagate into model parameters.

    Pre-fix, both the perturbed forward (``torch.no_grad``) and ``prediction``
    (``.detach()``) were detached, so ``2σ²·div`` was a constant and SURE
    collapsed to ``‖f(y)−y‖²−σ²`` — whose minimiser is the identity map.

    Take a linear denoiser ``f(y)=w·y`` evaluated at ``w=1``. There the data-fit
    gradient ``d/dw ‖w·y−y‖²`` vanishes, so the *only* gradient on ``w`` comes
    from the divergence term (with Rademacher probes, ``div = w`` exactly, so
    ``dL/dw = 2σ²``). A near-zero ``w.grad`` would mean the divergence is inert.
    """
    torch.manual_seed(0)
    sigma = 0.1
    w = torch.nn.Parameter(torch.tensor(1.0))

    def model(x: torch.Tensor) -> torch.Tensor:
        return w * x

    y = torch.randn(2, 1, 8, 8)
    prediction = model(y)
    loss_fn = SURELoss(sigma=sigma, n_samples=1, rademacher=True)
    out = loss_fn(prediction, prediction, context={"model": model, "noisy_input": y})
    out.backward()

    assert w.grad is not None
    # Divergence contributes ≈ 2σ² = 0.02; pre-fix this was ≈ 0.
    assert w.grad.abs().item() > 1e-3


# ---------------------------------------------------------------------------
# GSUREKspaceLoss — noise-model-aware k-space measurement fidelity (SOTA T1)
# ---------------------------------------------------------------------------


def test_gsure_unknown_noise_model_raises() -> None:
    """An unadvertised noise_model fails loud (pitfall #9: no silent fallback)."""
    with pytest.raises(ValueError, match="noise_model"):
        GSUREKspaceLoss(sigma=0.1, noise_model="rician_3d")


def test_gsure_invalid_n_coils_raises() -> None:
    """n_coils < 1 fails loud."""
    with pytest.raises(ValueError, match="n_coils"):
        GSUREKspaceLoss(sigma=0.1, noise_model="ncchi_magnitude", n_coils=0)


def test_gsure_gaussian_kspace_is_half_squared_residual() -> None:
    """Gaussian branch on complex k-space = 0.5·‖pred−meas‖²/σ² (mean)."""
    torch.manual_seed(0)
    sigma = 0.2
    pred = torch.complex(torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8))
    meas = torch.complex(torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8))
    loss_fn = GSUREKspaceLoss(sigma=sigma, noise_model="gaussian_kspace")
    out = loss_fn(pred, meas)
    diff = pred - meas
    expected = 0.5 * (diff.conj() * diff).real.mean() / (sigma * sigma)
    assert torch.allclose(out, expected, atol=1e-6)


def test_gsure_ncchi_reduces_to_gaussian_magnitude_at_high_snr() -> None:
    """Corner-case: as the noise floor 2Lσ² becomes negligible vs m² (high SNR),
    the nc-χ branch → the Gaussian magnitude fidelity 0.5(|pred|−m)²/σ².

    This is the load-bearing property that makes the repair a *generalization*
    of the published Gaussian method, not a different objective.
    """
    torch.manual_seed(1)
    sigma = 1e-3  # tiny noise → high SNR
    L = 4
    # Strong signal so m² >> 2Lσ².
    pred = 5.0 + torch.rand(2, 1, 8, 8)
    meas = 5.0 + torch.rand(2, 1, 8, 8)  # measured magnitude (real, positive)
    ncchi = GSUREKspaceLoss(sigma=sigma, noise_model="ncchi_magnitude", n_coils=L)(pred, meas)
    gaussian_mag = 0.5 * (pred - meas).pow(2).mean() / (sigma * sigma)
    assert torch.allclose(ncchi, gaussian_mag, rtol=1e-3)


def test_gsure_ncchi_debias_fires_at_low_snr() -> None:
    """Mechanism-fires: at low SNR the 2Lσ² debias materially changes the value,
    so the nc-χ branch is NOT an inert relabelling of the Gaussian fidelity."""
    torch.manual_seed(2)
    sigma = 0.7
    L = 4
    # Weak signal so the noise floor 2Lσ² ≈ 2·4·0.49 ≈ 3.9 is comparable to m².
    pred = 1.0 + 0.2 * torch.rand(2, 1, 8, 8)
    meas = 1.5 + 0.2 * torch.rand(2, 1, 8, 8)
    ncchi = GSUREKspaceLoss(sigma=sigma, noise_model="ncchi_magnitude", n_coils=L)(pred, meas)
    naive_gaussian_mag = 0.5 * (pred - meas).pow(2).mean() / (sigma * sigma)
    assert not torch.allclose(ncchi, naive_gaussian_mag, rtol=1e-2)


def test_gsure_registered_under_canonical_and_alias() -> None:
    """gsure_kspace registered; ncchi_gsure alias resolves to it."""
    from mriforge.models.losses.registry import LossRegistry, list_available

    assert "gsure_kspace" in list_available()
    assert LossRegistry.create("ncchi_gsure", sigma=0.1).__class__.__name__ == "GSUREKspaceLoss"
