"""Unit tests for ClinicalTrustAnalyzer."""

import pytest
import torch
import torch.nn as nn


class SimpleGenerator(nn.Module):
    """Minimal generator for testing: identity + small perturbation."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.ones_(self.conv.weight)

    def forward(self, x, **kwargs):
        return self.conv(x)


class TestEstimateNoiseFloor:
    def test_noise_floor_positive(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            estimate_noise_floor,
        )

        kspace = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        sigma = estimate_noise_floor(kspace)
        assert sigma.shape == (2, 1, 1, 1)
        assert (sigma > 0).all()

    def test_higher_noise_higher_estimate(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            estimate_noise_floor,
        )

        low = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 0.01
        high = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 10.0
        s_low = estimate_noise_floor(low).item()
        s_high = estimate_noise_floor(high).item()
        assert s_high > s_low


class TestNullSpaceResidual:
    def test_perfect_reconstruction_zero_residual(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        analyzer = ClinicalTrustAnalyzer()
        # If prediction exactly matches what produced the k-space,
        # residual should be near zero
        image = torch.randn(1, 1, 16, 16)
        from spectramr.infrastructure.physics.fft_ops import fft2c

        kspace = fft2c(image.to(torch.complex64))

        trust_map, residual = analyzer.compute_null_space_residual(image, kspace)
        assert trust_map.shape == (1, 1, 16, 16)
        assert residual.abs().max() < 1e-4, "Perfect recon should have zero residual"

    def test_hallucination_detected(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        analyzer = ClinicalTrustAnalyzer(noise_threshold_sigma=0.0)
        # Create kspace from one image, but predict a different image
        image_real = torch.randn(1, 1, 16, 16)
        from spectramr.infrastructure.physics.fft_ops import fft2c

        kspace = fft2c(image_real.to(torch.complex64))
        # Predict something totally different
        prediction = image_real + 5.0 * torch.randn_like(image_real)

        trust_map, _ = analyzer.compute_null_space_residual(prediction, kspace)
        assert trust_map.sum() > 0, "Hallucination should produce nonzero trust map"

    def test_masked_residual(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        analyzer = ClinicalTrustAnalyzer(noise_threshold_sigma=0.0)
        image = torch.randn(1, 1, 16, 16)
        from spectramr.infrastructure.physics.fft_ops import fft2c

        kspace = fft2c(image.to(torch.complex64))
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 6:10, :] = 1.0  # Sample center lines

        trust_map, _ = analyzer.compute_null_space_residual(image, kspace, mask=mask)
        assert trust_map.shape == (1, 1, 16, 16)


class TestJacobianSensitivity:
    def test_jacobian_computes(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        analyzer = ClinicalTrustAnalyzer()
        gen = SimpleGenerator()
        x = torch.randn(1, 1, 8, 8)
        t1 = torch.randn(1, 1, 8, 8, requires_grad=True)

        jacs = analyzer.compute_jacobian(gen, x, {"t1": t1})
        assert "t1" in jacs
        assert jacs["t1"].shape == (1, 1, 8, 8)

    def test_jacobian_nonzero_for_used_param(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        class ParamAwareGen(nn.Module):
            def forward(self, x, t1=None, **kw):
                return x * t1 if t1 is not None else x

        analyzer = ClinicalTrustAnalyzer()
        gen = ParamAwareGen()
        x = torch.randn(1, 1, 8, 8)
        t1 = torch.ones(1, 1, 8, 8, requires_grad=True)

        jacs = analyzer.compute_jacobian(gen, x, {"t1": t1})
        assert jacs["t1"].abs().sum() > 0, "Jacobian should be nonzero for used params"

    def test_inner_typeerror_propagates_not_swallowed(self):
        """A real TypeError raised INSIDE a kwarg-accepting generator must propagate.

        Regression for the silent-fallback bug: ``compute_jacobian`` used a bare
        ``except TypeError`` to route generators that do not accept physical-param
        kwargs to the input-only call. That also swallowed genuine TypeErrors
        raised *inside* a kwarg-accepting generator (bad reshape / None layer),
        silently re-running without physical params and masking the crash as a
        benign "parameter unused" Jacobian (CLAUDE.md pitfall #9). The fix uses
        signature introspection, so the inner crash now surfaces.
        """
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        class BuggyKwargGen(nn.Module):
            """Accepts the t1 kwarg but raises a real TypeError in forward()."""

            def forward(self, x, t1=None, **kw):
                return x + "oops"  # noqa: RUF100 — intentional TypeError

        analyzer = ClinicalTrustAnalyzer()
        x = torch.randn(1, 1, 8, 8)
        t1 = torch.randn(1, 1, 8, 8, requires_grad=True)
        with pytest.raises(TypeError):
            analyzer.compute_jacobian(BuggyKwargGen(), x, {"t1": t1})

    def test_named_param_without_varkwargs_uses_kwarg_path(self):
        """A generator that names ``t1`` (no ``**kwargs``) is routed via kwargs.

        Confirms ``accepts_all_named`` correctly detects parameter acceptance
        from the signature even without ``**kwargs`` — the Jacobian must be
        nonzero, proving the physical param actually reached the generator.
        """
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        class NamedGen(nn.Module):
            def forward(self, x, t1=None):
                return x * t1 if t1 is not None else x

        analyzer = ClinicalTrustAnalyzer()
        x = torch.randn(1, 1, 8, 8)
        t1 = torch.ones(1, 1, 8, 8, requires_grad=True)
        jacs = analyzer.compute_jacobian(NamedGen(), x, {"t1": t1})
        assert jacs["t1"].abs().sum() > 0

    def test_no_kwarg_generator_input_only_path(self):
        """A forward(self, x)-only generator takes the input-only path.

        The physical param is genuinely unused -> the Jacobian is None and is
        reported as such (no crash, no silent kwarg injection).
        """
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )

        class NoKwargGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(1, 1, 1, bias=False)
                nn.init.ones_(self.c.weight)

            def forward(self, x):  # no physical-param kwargs
                return self.c(x)

        analyzer = ClinicalTrustAnalyzer()
        x = torch.randn(1, 1, 8, 8)
        t1 = torch.randn(1, 1, 8, 8, requires_grad=True)
        jacs = analyzer.compute_jacobian(NoKwargGen(), x, {"t1": t1})
        # Param unused by an input-only generator -> grad is None -> reported as
        # an all-zero Jacobian (the documented "parameter unused" outcome).
        assert "t1" in jacs
        assert jacs["t1"].abs().sum() == 0


class TestFullPipeline:
    def test_full_analysis_runs(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )
        from spectramr.infrastructure.physics.fft_ops import fft2c

        analyzer = ClinicalTrustAnalyzer()
        gen = SimpleGenerator()
        x = torch.randn(1, 1, 16, 16)
        kspace = fft2c(x.to(torch.complex64))
        t1 = torch.randn(1, 1, 16, 16)
        b0 = torch.randn(1, 1, 16, 16)

        result = analyzer(gen, x, kspace, t1_map=t1, b0_map=b0)
        assert result.prediction.shape == (1, 1, 16, 16)
        assert result.trust_map.shape == (1, 1, 16, 16)
        assert result.jacobian_t1 is not None
        assert result.jacobian_b0 is not None

    def test_no_physical_params(self):
        from spectramr.infrastructure.physics.clinical_trust_analyzer import (
            ClinicalTrustAnalyzer,
        )
        from spectramr.infrastructure.physics.fft_ops import fft2c

        analyzer = ClinicalTrustAnalyzer()
        gen = SimpleGenerator()
        x = torch.randn(1, 1, 16, 16)
        kspace = fft2c(x.to(torch.complex64))

        result = analyzer(gen, x, kspace)
        assert result.jacobian_t1 is None
        assert result.jacobian_b0 is None
        assert result.trust_map is not None
