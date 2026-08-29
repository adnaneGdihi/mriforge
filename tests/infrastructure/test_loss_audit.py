from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.loss_audit import (
    LossAuditResult,
    LossRegistration,
    _test_loss_forward,
    _test_metrics_computation,
    audit_losses,
    get_loss_metrics_capability,
    verify_startup_losses,
)


class MockGoodLoss(nn.Module):
    def forward(self, pred, target):
        return torch.tensor(0.5, requires_grad=True)

    def forward_with_metrics(self, pred, target):
        loss = self.forward(pred, target)
        return loss, {"mse": 0.5, "psnr": 30.0}


class MockBadLossNonTensor(nn.Module):
    def forward(self, pred, target):
        return 0.5  # Returns float instead of tensor


class MockBadLossNonScalar(nn.Module):
    def forward(self, pred, target):
        return torch.ones(2)  # Returns vector


class MockBadLossNaN(nn.Module):
    def forward(self, pred, target):
        return torch.tensor(float("nan"))


class MockLossTupleReturn(nn.Module):
    def forward(self, pred, target):
        return torch.tensor(0.1), {"metric1": 0.1}


def test_loss_registration():
    reg = LossRegistration(name="l1", enabled=True, weight=1.0, has_metrics=True)
    assert reg.is_healthy() is True

    reg_err = LossRegistration(
        name="l1", enabled=True, weight=1.0, has_metrics=False, error="Failed"
    )
    assert reg_err.is_healthy() is False


def test_top_level_build_errors_make_result_unhealthy():
    """A total LossBuilder build failure populates ``result.errors`` but leaves
    ``result.losses`` empty. Health must NOT report healthy — otherwise the
    startup gate (``verify_startup_losses``) silently passes a config whose
    losses did not construct at all (pitfall #10: a green that masks a red)."""
    result = LossAuditResult()
    result.errors.append("Failed to build losses with LossBuilder: boom")
    result._calculate_health()

    assert result.all_healthy is False
    assert result.get_status() == "🔴"


def test_audit_losses_reports_when_loss_build_fails(monkeypatch):
    """If the LossBuilder chain raises (unknown loss / bad kwargs), ``audit_losses``
    must report it unhealthy — not return an empty (and therefore 'healthy')
    result. Regression for the false-GREEN where the except branch swallowed the
    failure.

    NB: ``verify_startup_losses`` itself is now a *registry-only* pre-check that
    constructs nothing (it no longer runs ``LossBuilder``), so its raise-on-bad-
    name contract is covered separately by
    ``test_verify_startup_losses_raises_for_unregistered`` /
    ``test_verify_startup_losses_does_not_build_losses``."""
    import mriforge.infrastructure.loss_audit as la

    class _ExplodingBuilder:
        def __init__(self, *a, **k):
            raise RuntimeError("loss construction blew up")

    monkeypatch.setattr(la, "LossBuilder", _ExplodingBuilder)

    result = la.audit_losses(MagicMock(name="loss_config"))
    assert result.all_healthy is False
    assert any("blew up" in e for e in result.errors)


def test_loss_audit_result_computation():
    result = LossAuditResult()

    result.losses["l1"] = LossRegistration("l1", True, 1.0, True)
    result.losses["l2"] = LossRegistration("l2", True, 0.5, False)
    result.losses["bad"] = LossRegistration("bad", True, 1.0, False, error="Crit")
    result.losses["warn"] = LossRegistration("warn", False, 0.0, False, warning="Warn")

    result._calculate_health()

    assert result.total_losses == 4
    assert result.enabled_losses == 3
    assert result.losses_with_metrics == 1
    assert result.critical_errors == 1
    assert result.warnings_count == 1
    assert result.all_healthy is False
    assert result.get_status() == "🔴"

    result.losses["bad"].error = None
    result._calculate_health()
    assert result.all_healthy is True
    assert result.get_status() == "🟡"

    result.losses["warn"].warning = None
    result._calculate_health()
    assert result.get_status() == "🟢"


def test_test_loss_forward_good():
    loss = MockGoodLoss()
    error = _test_loss_forward(loss)
    assert error is None

    # Gradient flow audit constraint: the dummy forward pass should maintain gradients
    # Even though _test_loss_forward just checks output shape/type, we verify the module behaves
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    out = loss(pred, target)
    assert out.requires_grad is True


@pytest.mark.parametrize(
    "loss_module,expected_err",
    [
        (MockBadLossNonTensor(), "expected Tensor"),
        (MockBadLossNonScalar(), "expected scalar"),
        (MockBadLossNaN(), "not finite"),
    ],
)
def test_test_loss_forward_bad(loss_module, expected_err):
    assert expected_err in _test_loss_forward(loss_module)


class MockTrainingGuardedLoss(nn.Module):
    """Mimics SENSEAdjointL1Loss: raises when probed in TRAIN mode without a
    required runtime kwarg, returns a finite zero in EVAL (the wiring-only
    contract the synthetic loss audit must honour)."""

    def forward(self, pred, target, smaps=None):
        if smaps is None and self.training:
            raise ValueError("smaps=None during training")
        return (pred.float().sum() * 0.0).to(torch.float32)


def test_loss_forward_probes_in_eval_mode_and_restores():
    """Regression (2026-05-30 DC-blob): a training-only missing-input guard
    must NOT fail the synthetic audit probe. The probe runs in eval mode, so
    the guard returns its finite zero; the module's original mode is restored.
    """
    loss = MockTrainingGuardedLoss()
    loss.train()  # caller hands us a train-mode module
    assert _test_loss_forward(loss) is None
    assert loss.training is True  # restored

    _has_metrics, error = _test_metrics_computation(loss)
    assert error is None
    assert loss.training is True  # restored


def test_loss_forward_audits_real_sense_adjoint_l1():
    """The real SENSEAdjointL1Loss — whose train-time smaps guard raised and
    broke experiment_11 smoke — now audits cleanly (the guard's eval/no-coil
    zero is the legitimate wiring-check result)."""
    from mriforge.models.losses.complex_losses import SENSEAdjointL1Loss

    loss = SENSEAdjointL1Loss()
    loss.train()
    assert _test_loss_forward(loss) is None
    assert loss.training is True  # restored


def test_test_metrics_computation():
    has_metrics, error = _test_metrics_computation(MockGoodLoss())
    assert has_metrics is True
    assert error is None

    # Test module that naturally returns tuple
    has_metrics, error = _test_metrics_computation(MockLossTupleReturn())
    assert has_metrics is True
    assert error is None

    has_metrics, error = _test_metrics_computation(MockBadLossNonTensor())
    assert has_metrics is False
    assert error is None


@patch("mriforge.infrastructure.loss_audit.LossBuilder")
def test_audit_losses(mock_builder_class):
    mock_builder = MagicMock()
    mock_builder_class.return_value = mock_builder

    mock_builder.build_reconstruction_losses.return_value = mock_builder
    mock_builder.build_adversarial_losses.return_value = mock_builder
    mock_builder.build_physics_losses.return_value = mock_builder
    mock_builder.build_regularization_losses.return_value = mock_builder
    mock_builder.build_diffusion_losses.return_value = mock_builder
    mock_builder.build_latent_losses.return_value = mock_builder
    mock_builder.build_ssl_losses.return_value = mock_builder

    # Return mock loss modules
    mock_builder.build.return_value = {
        "good_loss": MockGoodLoss(),
        "bad_loss": MockBadLossNaN(),
    }
    mock_builder.get_enabled_losses.return_value = {
        "good_loss": 1.0,
        "bad_loss": 1.0,
        "missing_loss": 1.0,
    }

    result = audit_losses({})

    assert "good_loss" in result.losses
    assert result.losses["good_loss"].is_healthy() is True
    assert result.losses["good_loss"].has_metrics is True

    assert "bad_loss" in result.losses
    assert result.losses["bad_loss"].is_healthy() is False
    assert "not finite" in result.losses["bad_loss"].error

    assert "missing_loss" in result.losses
    assert result.losses["missing_loss"].is_healthy() is False
    assert "did not create it" in result.losses["missing_loss"].error


def _loss_config_stub(*, kspace=(), image=(), complex_=()):
    """Minimal stand-in for LossConfigSchema with list-based loss blocks."""
    from types import SimpleNamespace

    return SimpleNamespace(
        kspace_losses=[SimpleNamespace(name=n) for n in kspace],
        image_losses=[SimpleNamespace(name=n) for n in image],
        complex_losses=[SimpleNamespace(name=n) for n in complex_],
    )


def test_verify_startup_losses_passes_for_registered(monkeypatch):
    """Registry-only check: passes (and constructs nothing) for known names."""
    from mriforge.models.losses.registry import LossRegistry

    monkeypatch.setattr(LossRegistry, "is_registered", lambda name: True)
    cfg = _loss_config_stub(image=("l1", "perceptual"), kspace=("ssim",))
    assert verify_startup_losses(cfg) is True


def test_verify_startup_losses_raises_for_unregistered(monkeypatch):
    """A list-based loss name that is not registered fails fast."""
    from mriforge.models.losses.registry import LossRegistry

    monkeypatch.setattr(
        LossRegistry, "is_registered", lambda name: name != "not_a_loss"
    )
    monkeypatch.setattr(LossRegistry, "list_available", lambda: ["l1", "ssim"])
    cfg = _loss_config_stub(image=("l1", "not_a_loss"))
    with pytest.raises(RuntimeError, match="Loss configuration validation failed"):
        verify_startup_losses(cfg)


def test_verify_startup_losses_does_not_build_losses(monkeypatch):
    """Regression: the startup check must not invoke the full audit build."""
    import mriforge.infrastructure.loss_audit as la
    from mriforge.models.losses.registry import LossRegistry

    def _boom(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("verify_startup_losses must not build losses")

    monkeypatch.setattr(la, "audit_losses", _boom)
    monkeypatch.setattr(LossRegistry, "is_registered", lambda name: True)
    assert verify_startup_losses(_loss_config_stub(image=("perceptual",))) is True


@patch("mriforge.infrastructure.loss_audit.audit_losses")
def test_get_loss_metrics_capability(mock_audit):
    result = LossAuditResult()
    result.losses = {
        "has_m": LossRegistration("has_m", True, 1.0, True),
        "no_m": LossRegistration("no_m", True, 1.0, False),
    }
    mock_audit.return_value = result

    caps = get_loss_metrics_capability({})
    assert caps["has_m"] is True
    assert caps["no_m"] is False


# ----------------------------------------------------------------------
# F41a — gradient_penalty / feature_matching are composite-bundled
#         inside the 'adversarial' loss (gan_composite) via lambda_gp /
#         lambda_feat_match. They MUST NOT be flagged as "did not create
#         it" when LossBuilder rightly skips standalone instantiation.
#         See ``loss_builder.py::_build_composite_gan`` and the F6/E26
#         skip-list comment in ``loss_builder.py:144,379``.
#
# F41b — bloch_consistency IS built by LossBuilder but its forward()
#         requires a strategy-injected ``context={tissue_params,
#         acquisition_params}`` dict that the synthetic forward test
#         cannot supply. Treat it as strategy-managed for audit
#         purposes (the loss object exists; only the forward test
#         is the wrong tool).
# ----------------------------------------------------------------------


class _MockBlochConsistency(nn.Module):
    """Mirrors BlochConsistencyLoss.forward()'s context-required contract."""

    def forward(self, pred, target, context=None, **kwargs):
        if context is None:
            raise ValueError(
                "BlochConsistencyLoss requires a `context` dict containing "
                "`tissue_params` and `acquisition_params`."
            )
        return torch.tensor(0.5, requires_grad=True)


@pytest.mark.parametrize(
    "loss_name",
    [
        "gradient_penalty",  # F41a: bundled into 'adversarial' composite
        "feature_matching",  # F41a: bundled into 'adversarial' composite
    ],
)
@patch("mriforge.infrastructure.loss_audit.LossBuilder")
def test_audit_losses_skips_composite_bundled(mock_builder_class, loss_name):
    """Composite-bundled GAN regularizers (gradient_penalty, feature_matching)
    are absent from ``built_losses`` by design — the builder folds them into
    ``adversarial`` via ``gan_composite``'s ``lambda_gp`` / ``lambda_feat_match``
    parameters. The validator must NOT flag this as ``did not create it``.
    """
    mock_builder = MagicMock()
    mock_builder_class.return_value = mock_builder
    for chain in (
        "build_reconstruction_losses",
        "build_adversarial_losses",
        "build_physics_losses",
        "build_regularization_losses",
        "build_diffusion_losses",
        "build_latent_losses",
        "build_ssl_losses",
    ):
        getattr(mock_builder, chain).return_value = mock_builder

    # Builder skipped standalone instantiation (correct — bundled in 'adversarial').
    mock_builder.build.return_value = {"adversarial": MockGoodLoss()}
    mock_builder.get_enabled_losses.return_value = {
        "adversarial": 1.0,
        loss_name: 10.0,  # enabled with non-zero weight
    }

    result = audit_losses({})

    assert loss_name in result.losses
    reg = result.losses[loss_name]
    # The whole point: not a critical error anymore.
    assert reg.error is None, (
        f"{loss_name} flagged as critical error: {reg.error!r}. "
        "It should be treated as composite-bundled (warning only)."
    )
    assert reg.is_healthy(), f"{loss_name} should be healthy (composite-bundled)"


@patch("mriforge.infrastructure.loss_audit.LossBuilder")
def test_audit_losses_skips_context_requiring_bloch_consistency(mock_builder_class):
    """``bloch_consistency`` is created by LossBuilder, but its forward()
    requires ``context={tissue_params, acquisition_params}`` populated by the
    training strategy. The validator's synthetic forward test cannot supply
    this, so the loss must be treated as strategy-managed for audit purposes.
    """
    mock_builder = MagicMock()
    mock_builder_class.return_value = mock_builder
    for chain in (
        "build_reconstruction_losses",
        "build_adversarial_losses",
        "build_physics_losses",
        "build_regularization_losses",
        "build_diffusion_losses",
        "build_latent_losses",
        "build_ssl_losses",
    ):
        getattr(mock_builder, chain).return_value = mock_builder

    mock_builder.build.return_value = {"bloch_consistency": _MockBlochConsistency()}
    mock_builder.get_enabled_losses.return_value = {"bloch_consistency": 1.0}

    result = audit_losses({})

    assert "bloch_consistency" in result.losses
    reg = result.losses["bloch_consistency"]
    assert reg.error is None, (
        f"bloch_consistency flagged as critical error: {reg.error!r}. "
        "It should be treated as strategy-managed (context dict injected at train time)."
    )
    assert reg.is_healthy(), "bloch_consistency should be healthy (strategy-managed)"


def test_loss_forward_audits_bridged_losses_without_reshape_warning(caplog):
    """Regression (2026-06-08 experiment_11): iFFT-bridge losses
    (`complex_spatial_gradient`, `sense_adjoint_l1`) are wrapped so their
    runtime class name carries no 'complex'/'kspace'/'fft' token. The probe's
    ``is_complex`` heuristic therefore fed them the generic odd-channel real
    dummy (B,3,H,W); the bridge reshapes the channel axis into (coil, real/imag)
    PAIRS, and 3 is odd -> 'shape [B,1,2,H,W] is invalid for input of size ...'
    logged at startup on every run. The fix routes any 'bridge'-named loss to
    the complex probe. The losses were never broken in training (they compute
    real, non-zero values on the even multi-coil tensors there) — this guards
    the clean-wiring-check contract (pitfall #10: warnings are not OK).
    """
    import logging

    from mriforge.models.losses.physics_losses import DifferentiableFourierBridge
    from mriforge.models.losses.registry import create_loss

    for name in ("complex_spatial_gradient", "sense_adjoint_l1"):
        bridged = DifferentiableFourierBridge(
            spatial_loss_fn=create_loss(name), return_complex=False
        )
        # The wrapper name must be routed to the complex probe by the fix.
        assert "bridge" in bridged.__class__.__name__.lower()
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            error = _test_loss_forward(bridged)
        assert error is None, f"{name} bridged loss audited as error: {error!r}"
        assert not any(
            "invalid for input of size" in rec.getMessage() for rec in caplog.records
        ), f"{name} bridged loss still logs the spurious reshape warning"
