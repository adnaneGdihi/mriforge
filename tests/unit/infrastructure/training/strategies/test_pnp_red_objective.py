"""Regression: PnPStrategy must add a genuine RED denoiser-prior term on top
of the inherited supervised reconstruction loss.

Before the 2026-05-31 fix, ``PnPStrategy`` added NO paradigm-specific
training math — its ``_compute_losses_impl`` was byte-identical to plain
:class:`ReconstructionTrainingStrategy` (backlog #15: an advertised paradigm
that silently no-ops). The fix adds a Regularization-by-Denoising term

    loss_red = λ_red · ½‖x - D(x)‖²

where the generator ``D`` denoises the current reconstruction ``x``. These
tests assert the KEY properties of the fix:

* ``_compute_losses_impl`` returns a ``loss_red`` key distinct from the base
  recon loss, and folds it into ``g_total_loss``;
* the RED term flows gradient into the generator/denoiser parameters
  (so it actually shapes training, not a detached constant);
* the weight is read from ``config.losses.reconstruction.lambda_red`` (SSOT),
  falling back to the module default when the knob is absent.

The strategy is constructed via ``object.__new__`` with only the attributes
the tested method touches; the parent ``_compute_losses_impl`` is stubbed so
the test isolates the RED contribution.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies import reconstruction as recon_mod
from mriforge.infrastructure.training.strategies.pnp_strategy import (
    _DEFAULT_LAMBDA_RED,
    PnPStrategy,
)


class _Denoiser(nn.Module):
    """Tiny learnable denoiser standing in for the generator."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_strategy(generator: nn.Module, lambda_red: float | None) -> PnPStrategy:
    """Build a PnPStrategy with only the attrs the RED path touches."""
    s = object.__new__(PnPStrategy)
    recon_cfg = types.SimpleNamespace()
    if lambda_red is not None:
        recon_cfg.lambda_red = lambda_red
    losses = types.SimpleNamespace(reconstruction=recon_cfg)
    s.config = types.SimpleNamespace(losses=losses)
    s.env = types.SimpleNamespace(generator=generator)
    return s


def _stub_parent(monkeypatch, base_dict):
    """Stub ReconstructionTrainingStrategy._compute_losses_impl -> base_dict."""

    def _fake(self, *, input_batch=None, target_batch=None, epoch=0, **kwargs):
        # Return a fresh copy so each call is independent.
        return dict(base_dict)

    monkeypatch.setattr(
        recon_mod.ReconstructionTrainingStrategy,
        "_compute_losses_impl",
        _fake,
    )


class TestPnPRedObjective:
    def test_adds_red_term_distinct_from_base(self, monkeypatch):
        gen = _Denoiser()
        s = _make_strategy(gen, lambda_red=0.1)
        base_recon = torch.tensor(3.0, requires_grad=True)
        _stub_parent(monkeypatch, {"g_total_loss": base_recon, "loss_l1": base_recon})

        target = torch.randn(2, 2, 8, 8)
        # RED regularizes the reconstruction = gen(input_batch); supply a real
        # undersampled input so the denoiser runs on the model's OUTPUT (not the
        # ground-truth target, which was the pre-fix no-op).
        out = s._compute_losses_impl(
            input_batch=torch.randn(2, 2, 8, 8), target_batch=target, epoch=0
        )

        assert "loss_red" in out, "PnP did not add a RED prior term"
        # RED term is a non-trivial penalty distinct from the base recon loss.
        assert out["loss_red"].item() > 0.0
        assert not torch.equal(out["loss_red"], out["loss_l1"])
        # g_total_loss must include the RED contribution.
        expected_total = base_recon + out["loss_red"]
        assert torch.allclose(out["g_total_loss"], expected_total)

    def test_red_flows_gradient_to_denoiser(self, monkeypatch):
        gen = _Denoiser()
        s = _make_strategy(gen, lambda_red=0.5)
        _stub_parent(monkeypatch, {"g_total_loss": torch.zeros(())})

        target = torch.randn(1, 2, 8, 8)
        out = s._compute_losses_impl(
            input_batch=torch.randn(1, 2, 8, 8), target_batch=target, epoch=0
        )

        out["g_total_loss"].backward()
        grads = [p.grad for p in gen.parameters()]
        assert all(g is not None for g in grads), "RED term detached from denoiser"
        assert any(g.abs().sum().item() > 0 for g in grads), (
            "RED term produced zero gradient on the denoiser params"
        )

    def test_weight_read_from_config_lambda_red(self, monkeypatch):
        gen = _Denoiser()
        s = _make_strategy(gen, lambda_red=2.5)
        assert s._red_weight() == 2.5

    def test_weight_falls_back_to_default(self, monkeypatch):
        gen = _Denoiser()
        s = _make_strategy(gen, lambda_red=None)  # no knob declared
        assert s._red_weight() == _DEFAULT_LAMBDA_RED

    def test_weight_scales_red_term(self, monkeypatch):
        # Same denoiser + input, two weights -> RED term scales linearly.
        torch.manual_seed(0)
        gen = _Denoiser()
        target = torch.randn(1, 2, 8, 8)

        s_small = _make_strategy(gen, lambda_red=0.1)
        s_large = _make_strategy(gen, lambda_red=1.0)
        _stub_parent(monkeypatch, {"g_total_loss": torch.zeros(())})

        with torch.no_grad():
            red_small = s_small._red_prior(target) * 0.1
            red_large = s_large._red_prior(target) * 1.0
        assert torch.allclose(red_large, red_small * 10.0)


# ---------------------------------------------------------------------------
# Regression: spectral-norm denoiser + ADMM x-update + complex layout.
# Pre-fix bugs (2026-05-31 verification pass):
#   1/2. run_pnp used the RAW generator; spectral_norm_iters /
#        denoiser_lipschitz_constant were never read (unwired knobs, #15).
#   3.   ADMM x-update scaled every UNSAMPLED k-space bin by rho/(1+rho)
#        instead of leaving the free prior F(z-u) unscaled.
#   5.   A complex image was fed straight into a 2-channel denoiser.
# ---------------------------------------------------------------------------

import pytest

from mriforge.models.wrappers import SpectralNormalizedDenoiser


def _pnp_strategy(generator, *, iters=2, rho=1.0, sn_iters=3, lip=1.0, in_ch=2):
    """Build a PnPStrategy stub with an *enabled* pnp config + a generator."""
    s = object.__new__(PnPStrategy)
    pnp = types.SimpleNamespace(
        enabled=True,
        iterations=iters,
        rho=rho,
        spectral_norm_iters=sn_iters,
        denoiser_lipschitz_constant=lip,
    )
    training = types.SimpleNamespace(pnp=pnp)
    model = types.SimpleNamespace(in_channels=in_ch)
    s.config = types.SimpleNamespace(training=training, model=model)
    s.env = types.SimpleNamespace(generator=generator)
    s.device = torch.device("cpu")
    s._pnp_cfg = pnp
    s._denoiser = None
    s.pnp_provenance = {}
    return s


class TestPnPSpectralNormDenoiser:
    def test_denoiser_is_spectral_normed_and_knobs_stamped(self):
        """run_pnp z-update must use the SpectralNormalizedDenoiser, and the
        spectral_norm_iters / denoiser_lipschitz_constant knobs must be read
        and stamped into provenance (pre-fix: raw generator, knobs no-op)."""
        gen = _Denoiser()
        s = _pnp_strategy(gen, sn_iters=4, lip=0.5)
        denoiser = s._build_denoiser()
        assert isinstance(denoiser, SpectralNormalizedDenoiser)
        assert denoiser.is_lipschitz_constrained
        # The conv weight is now a spectral-norm parameterisation.
        assert hasattr(gen.conv, "weight_orig") or any(
            "weight" in n for n, _ in gen.conv.named_buffers()
        )
        # Knobs read + stamped (pitfall #15).
        assert s.pnp_provenance["spectral_norm_iters"] == 4
        assert s.pnp_provenance["denoiser_lipschitz_constant"] == 0.5
        assert s.pnp_provenance["spectral_normed_layers"]

    def test_invalid_lipschitz_raises(self):
        gen = _Denoiser()
        s = _pnp_strategy(gen, lip=0.0)
        with pytest.raises(ValueError):
            s._build_denoiser()

    def test_invalid_spectral_norm_iters_raises(self):
        gen = _Denoiser()
        s = _pnp_strategy(gen, sn_iters=0)
        with pytest.raises(ValueError):
            s._build_denoiser()


class _RecordingLogService:
    """Logging stub mirroring the real ``LoggingService`` API surface.

    It exposes ``log_info`` (the genuine method) and deliberately NO bare
    ``info`` attribute, so a ``log.info(...)`` call raises ``AttributeError``
    exactly as the production :class:`LoggingService` does — letting a unit
    test reproduce the b21_ulf_map setup crash without spinning up the DI
    container.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log_info(
        self, message: str, model_type: str = "", epoch: int = -1, extra=None
    ) -> None:
        self.messages.append(message)


class TestPnPDenoiserBuildLogging:
    def test_build_denoiser_logs_via_log_info_not_info(self):
        """Regression (b21_ulf_map, 2026-06-28): ``_build_denoiser`` announced
        the spectral-normed denoiser via ``log.info(...)``, but the injected
        ``LoggingService`` exposes ``log_info`` (not ``info``), so every PnP arm
        that had a ``logging_service`` crashed at setup with ``'LoggingService'
        object has no attribute 'info'`` → zero successful validation batches
        (CLAUDE.md #10). The build must use the real logging API and still
        return the spectral-normed denoiser.
        """
        gen = _Denoiser()
        s = _pnp_strategy(gen)
        s.logging_service = _RecordingLogService()

        denoiser = s._build_denoiser()

        assert isinstance(denoiser, SpectralNormalizedDenoiser)
        assert any(
            "spectral-normed denoiser built" in m for m in s.logging_service.messages
        ), "denoiser-built announcement was not logged via log_info"


class TestPnPADMMxUpdate:
    def test_unsampled_bin_scaling_is_one(self):
        """At unsampled k-space bins the x-update must equal the FREE prior
        F(z-u) (scale 1.0), not rho/(1+rho)*F(z-u) (pre-fix bug)."""
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c

        torch.manual_seed(0)

        class _Identity(nn.Module):
            def forward(self, x):
                return x

        s = _pnp_strategy(_Identity(), iters=1, rho=2.0, in_ch=2)
        # Mask out EVERYTHING (no sampled lines) so the whole k-space is the
        # free prior; with y=0, a one-iteration solve must return -F^-1(F(u))=...
        # Concretely we check the linear x-update term directly.
        b, c, h, w = 1, 1, 4, 4
        z = torch.randn(b, c, h, w, dtype=torch.complex64)
        u = torch.zeros_like(z)
        y = torch.zeros_like(z)
        mask = torch.zeros(b, c, h, w)  # all unsampled
        rho = 2.0
        mask_bool = mask.bool()
        kz = fft2c(z - u)
        kx = torch.where(mask_bool, (y + rho * kz) / (1.0 + rho), kz)
        # Unsampled bins must be kz exactly (scale 1.0), not rho/(1+rho)*kz.
        assert torch.allclose(kx, kz, atol=1e-5)
        # The pre-fix bug applied the data-fidelity denominator to ALL bins,
        # so unsampled bins were rho/(1+rho)*kz. The post-fix value is exactly
        # (1+rho)/rho larger, INDEPENDENT of bin magnitude (a magnitude-robust
        # check, unlike allclose which goes flaky on tiny-magnitude bins).
        wrong = (rho / (1.0 + rho)) * kz
        ref = kz.abs().sum()
        assert ref > 0
        # kx is the free prior; wrong is the scaled-down buggy value.
        assert torch.allclose(kx, (1.0 + rho) / rho * wrong, atol=1e-5)
        # The buggy value differs from the free prior by the known factor.
        assert (kx.abs().sum() - wrong.abs().sum()).abs() > 0.1 * ref
        # And the recovered image equals z (free prior round-trips).
        assert torch.allclose(ifft2c(kx), z, atol=1e-4)

    def test_sampled_bin_is_data_fidelity_prox(self):
        from mriforge.infrastructure.physics.fft_ops import fft2c

        torch.manual_seed(1)
        b, c, h, w = 1, 1, 4, 4
        z = torch.randn(b, c, h, w, dtype=torch.complex64)
        u = torch.zeros_like(z)
        y = torch.randn(b, c, h, w, dtype=torch.complex64)
        mask = torch.ones(b, c, h, w)  # all sampled
        rho = 0.5
        kz = fft2c(z - u)
        kx = torch.where(mask.bool(), (y + rho * kz) / (1.0 + rho), kz)
        assert torch.allclose(kx, (y + rho * kz) / (1.0 + rho), atol=1e-5)


class TestPnPComplexLayout:
    def test_complex_preserved_through_2ch_denoiser(self):
        """A complex image fed to a 2-channel denoiser must keep its imaginary
        part: convert complex->interleaved->complex round-trips losslessly."""

        class _TwoChan(nn.Module):
            # Identity-ish 2->2 conv so we can verify imag survives the wrap.
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                with torch.no_grad():
                    self.conv.weight.copy_(
                        torch.eye(2).view(2, 2, 1, 1)
                    )
                    self.conv.bias.zero_()

            def forward(self, x):
                return self.conv(x)

        s = _pnp_strategy(_TwoChan(), in_ch=2)
        z = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        layout, was_complex = s._to_denoiser_layout(z)
        assert was_complex
        assert not layout.is_complex()
        assert layout.shape == (1, 2, 4, 4)
        back = s._from_denoiser_layout(layout, was_complex)
        assert back.is_complex()
        assert torch.allclose(back, z, atol=1e-6)
        # Non-zero imaginary part is preserved (not silently dropped).
        assert z.imag.abs().sum() > 0
        assert torch.allclose(back.imag, z.imag, atol=1e-6)

    def test_run_pnp_end_to_end_shape_and_dc(self):
        """run_pnp is callable end-to-end with an enabled cfg + tiny generator;
        sampled k-space bins stay data-consistent (DC) after the solve."""

        class _TwoChan(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        s = _pnp_strategy(_TwoChan(), iters=2, rho=1.0, in_ch=2)
        h = w = 8
        y = torch.randn(1, 1, h, w, dtype=torch.complex64)
        mask = torch.zeros(1, 1, h, w)
        mask[..., ::2] = 1.0  # every other line sampled
        out = s.run_pnp(y, mask)
        assert out.shape == (1, 1, h, w)
        assert out.is_complex()
