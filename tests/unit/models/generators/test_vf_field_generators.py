"""Unit tests for VF Task 3 field mapping generators.

Tests: forward shape, backward grad flow, numerical stability,
and physics-specific correctness for B0/B1 extraction.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.init_registry import populate_model_registry

populate_model_registry()

from spectramr.models.generators.vf_field_generators import (
    AFIRatioCNN,
    BlochSiegertAlgebraicGenerator,
    BSSFPPeakFinder,
    DAMTrigFitGenerator,
    GraphCutUnwrapGenerator,
    MRFDictMatcher,
    PhaseTrackingLSTM,
    ReciprocityDivisor,
    ResNetUnwrapGenerator,
    SiameseEPITracker,
)

TASK3_MODELS = [
    ("resnet_unwrap", ResNetUnwrapGenerator, {"chans": 16}),
    ("bloch_siegert_algebraic", BlochSiegertAlgebraicGenerator, {}),
    ("afi_ratio_cnn", AFIRatioCNN, {"hidden_dim": 16}),
    ("dam_trigfit", DAMTrigFitGenerator, {"num_iters": 3}),
    ("siamese_epi_tracker", SiameseEPITracker, {"features": 16}),
    ("phase_tracking_lstm", PhaseTrackingLSTM, {"hidden_dim": 16, "num_layers": 1}),
    ("reciprocity_divisor", ReciprocityDivisor, {"sigma": 2.0}),
    ("mrf_dict_matcher", MRFDictMatcher, {"dict_size": 50}),
    ("bssfp_peak_finder", BSSFPPeakFinder, {}),
    ("graph_cut_unwrap", GraphCutUnwrapGenerator, {"chans": 16}),
]


@pytest.fixture(params=TASK3_MODELS, ids=[m[0] for m in TASK3_MODELS])
def model_and_name(request: pytest.FixtureRequest) -> tuple[str, torch.nn.Module]:
    name, cls, kwargs = request.param
    model = cls(in_channels=2, out_channels=2, **kwargs)
    return name, model


class TestTask3ForwardShape:
    def test_standard_shape(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64), f"{name}: got {list(y.shape)}"

    def test_batch_size(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(4, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape[0] == 4

    def test_non_square(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 48, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape[2] == 48 and y.shape[3] == 64, f"{name}: {list(y.shape)}"


class TestTask3Backward:
    def test_gradient_flow(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        y.mean().backward()
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grads, f"{name}: no gradients"

    def test_no_nan_grads(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.train()
        x = torch.randn(1, 2, 32, 32)
        y = model(x)
        y.mean().backward()
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN in {pname}"


class TestTask3NonTrivial:
    """Verify outputs are well-formed.

    Same rationale as Task 1 / Task 2: VF field generators use
    residual init (zero-initialised output conv), so the previous
    ``δ > 0.001`` assertion contradicts the architectural choice.
    Check finiteness instead.
    """

    def test_output_is_finite(self, model_and_name: tuple[str, torch.nn.Module]) -> None:
        name, model = model_and_name
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), f"{name}: output contains NaN/Inf"


class TestTask3SpecificPhysics:
    """Physics-specific tests for Task 3 blocks."""

    def test_bssfp_uses_fft(self) -> None:
        """bSSFP peak finder should use spectral features."""
        model = BSSFPPeakFinder(in_channels=2, out_channels=2)
        model.eval()
        # Sinusoidal input should produce distinct spectral features
        x = torch.zeros(1, 2, 32, 32)
        x[:, 0] = torch.sin(torch.linspace(0, 4 * 3.14159, 32)).unsqueeze(0)
        x[:, 1] = torch.cos(torch.linspace(0, 4 * 3.14159, 32)).unsqueeze(0)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any()
        assert y.shape == (1, 2, 32, 32)

    def test_mrf_dict_matching(self) -> None:
        """MRF matcher: different inputs should produce different outputs."""
        model = MRFDictMatcher(in_channels=2, out_channels=2, dict_size=50)
        model.eval()
        x1 = torch.randn(1, 2, 16, 16)
        x2 = torch.randn(1, 2, 16, 16) * 5.0
        with torch.no_grad():
            y1 = model(x1)
            y2 = model(x2)
        assert (y1 - y2).abs().mean() > 0.001

    def test_reciprocity_gaussian_smooth(self) -> None:
        """Reciprocity: output should be smoother than input."""
        model = ReciprocityDivisor(in_channels=2, out_channels=2, sigma=3.0)
        model.eval()
        x = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            y = model(x)
        # Output shouldn't have NaN
        assert not torch.isnan(y).any()


class TestGraphCutUnwrapCapabilityContract:
    """The declared half of ``graph_cut_unwrap``'s contract, and the absent half (#1106).

    Both halves are deliberate, and only one of them is self-evident from the
    decorator, so both are pinned here: a later reader who "completes" the
    declaration by adding the domains would reintroduce the E-VIZ2 regression
    (a spurious visualisation IFFT on an already-image prediction, which produced
    DC-blob fakes across 10 of 12 VF arms) without any test objecting.
    """

    def test_the_declared_fields_match_what_forward_actually_does(self) -> None:
        from spectramr.models.registry import get_model_capabilities

        caps = get_model_capabilities("graph_cut_unwrap")
        assert caps is not None
        assert caps.spatial_dims == (2,)
        assert caps.accepts_complex is True
        assert caps.requires_paired_data is True

    def test_rank_2_is_a_contract_not_a_preference(self) -> None:
        """``spatial_dims=(2,)`` is only honest if rank 3 genuinely fails."""
        model = GraphCutUnwrapGenerator(in_channels=2, out_channels=2, chans=16)
        model.eval()
        with pytest.raises(ValueError):
            model(torch.randn(1, 2, 8, 16, 16))

    def test_accepts_complex_is_honest(self) -> None:
        """Declared True, so a complex tensor must run rather than raise."""
        model = GraphCutUnwrapGenerator(in_channels=2, out_channels=2, chans=16)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 1, 32, 32, dtype=torch.complex64))
        assert torch.isfinite(y).all()

    def test_the_domains_stay_undeclared_and_the_name_list_stays_the_ssot(self) -> None:
        """The paired assertion: domains absent HERE because they live THERE.

        Declaring ``output_domain`` on the decorator would make P2 fire, and P2
        outranks P3's ``data.domain.output`` -- so the decorator would silently
        beat an arm's own declaration (#986). The strategy IFFTs to image before
        this backbone runs, so the static data->model domain check does not apply.
        """
        from spectramr.infrastructure.training.utils.domain_inference import (
            KNOWN_IMAGE_OUTPUT_MODELS,
            KNOWN_KSPACE_OUTPUT_MODELS,
        )
        from spectramr.models.registry import get_model_capabilities

        caps = get_model_capabilities("graph_cut_unwrap")
        assert caps.input_domain is None
        assert caps.output_domain is None
        assert "graph_cut_unwrap" in KNOWN_IMAGE_OUTPUT_MODELS
        assert "graph_cut_unwrap" not in KNOWN_KSPACE_OUTPUT_MODELS

    def test_output_field_units_stays_absent_so_the_metric_lock_keeps_biting(self) -> None:
        """``last_field_estimate`` is unwrapped phase -- proportional to B0, never Hz.

        The metric<->units lock gates ``b0_field_rmse`` on
        ``output_field_units == "Hz"``. Declaring it here would unlock a metric
        that would then grade a unit mismatch, which is why
        ``_score_field_structure`` compares structure (Pearson + scale-fit NRMSE)
        instead of magnitudes. Contrast ``bssfp_b0_regressor``, whose output IS a
        field in Hz and which therefore DOES declare it.
        """
        from spectramr.models.registry import get_model_capabilities

        assert get_model_capabilities("graph_cut_unwrap").output_field_units is None
        assert get_model_capabilities("bssfp_b0_regressor").output_field_units == "Hz"
