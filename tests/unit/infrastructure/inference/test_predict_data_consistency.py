"""``physics.data_consistency.apply_at_predict``: the projection step itself.

Everything here drives :class:`PredictDataConsistency` directly. The hook that
calls it from ``BaseInferenceStrategy.infer`` has its own file
(``test_base_inference_strategy.py``); the pipeline's attachment of the mask
and the measurement is in ``tests/unit/pipelines/test_infer.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.inference import predict_data_consistency as pdc
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.capabilities import ModelCapabilities

KNOB = "physics.data_consistency.apply_at_predict"


def _register(monkeypatch, model_type: str, output_domain):
    """Pretend ``model_type`` is registered with the given ``output_domain``."""
    caps = None if output_domain is None else ModelCapabilities(output_domain=output_domain)
    monkeypatch.setitem(pdc.MODEL_REGISTRY, model_type, {"capabilities": caps})
    monkeypatch.setattr(
        pdc, "get_model_capabilities", lambda name: caps if name == model_type else None
    )


def _cfg(model_type="stub", **dc):
    dc = {"apply_at_predict": True, **dc}
    return {"physics": {"data_consistency": dc}, "model": {"model_type": model_type}}


@pytest.fixture
def kspace_step(monkeypatch):
    _register(monkeypatch, "stub", "kspace")
    step = pdc.PredictDataConsistency.from_config(_cfg())
    assert step is not None
    step.begin()
    return step


@pytest.fixture
def image_step(monkeypatch):
    _register(monkeypatch, "stub_img", "image")
    step = pdc.PredictDataConsistency.from_config(_cfg("stub_img"))
    assert step is not None
    step.begin()
    return step


def _checkerboard(h=8, w=8):
    mask = torch.zeros(h, w)
    mask[:, ::2] = 1.0
    return mask


class TestTheProjectionFires:
    def test_kspace_prediction_takes_the_measurement_on_the_mask_bit_for_bit(self, kspace_step):
        pred = torch.randn(2, 2, 8, 8)
        meas = torch.randn(2, 2, 8, 8)
        mask = _checkerboard()
        assert not torch.equal(pred[..., mask.bool()], meas[..., mask.bool()])

        out = kspace_step.finalize(pred.clone(), mask=mask, measured_kspace=meas, strategy="t")

        on = mask.bool()
        assert torch.equal(out[..., on], meas[..., on]), "sampled bins must BE the measurement"
        assert torch.equal(out[..., ~on], pred[..., ~on]), "unsampled bins must be untouched"
        assert out.shape == pred.shape and out.dtype == pred.dtype

    def test_complex_kspace_is_projected_too(self, kspace_step):
        pred = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        meas = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        mask = _checkerboard()
        out = kspace_step.project(pred, mask=mask, measured_kspace=meas)
        on = mask.bool()
        assert torch.equal(out[..., on], meas[..., on])
        assert torch.equal(out[..., ~on], pred[..., ~on])

    def test_image_prediction_is_projected_through_the_fft_pair(self, image_step):
        """Magnitude image in, magnitude image out; the extremes pin the path."""
        image = torch.rand(1, 1, 8, 8) + 0.5
        meas = fft2c(torch.rand(1, 1, 8, 8, dtype=torch.complex64))

        nothing = image_step.project(image, mask=torch.zeros(8, 8), measured_kspace=meas)
        assert torch.allclose(nothing, image, atol=1e-5), "an empty mask keeps the prediction"

        everything = image_step.project(image, mask=torch.ones(8, 8), measured_kspace=meas)
        assert torch.allclose(everything, ifft2c(meas).abs(), atol=1e-5)
        assert not everything.is_complex() and everything.shape == image.shape

    def test_a_real_two_channel_measurement_is_accepted_for_a_single_coil(self, image_step):
        image = torch.rand(1, 1, 8, 8)
        meas = torch.randn(1, 2, 8, 8)
        out = image_step.project(image, mask=torch.ones(8, 8), measured_kspace=meas)
        expected = ifft2c(torch.complex(meas[:, 0:1], meas[:, 1:2])).abs()
        assert torch.allclose(out, expected, atol=1e-5)

    def test_the_measurement_is_used_as_acquired(self, kspace_step):
        """Zero noise on both levels and an eval-mode layer; the stamp says so."""
        layer = kspace_step._layer
        assert layer.train_noise_level == 0.0 and layer.eval_noise_level == 0.0
        assert layer.training is False
        assert kspace_step.provenance()["measurement_noise_added"] is False


class TestTheKnob:
    def test_off_or_absent_resolves_to_none(self):
        assert pdc.PredictDataConsistency.from_config(None) is None
        assert pdc.PredictDataConsistency.from_config({}) is None
        assert pdc.PredictDataConsistency.from_config({"physics": {}}) is None
        assert pdc.PredictDataConsistency.from_config(_cfg(apply_at_predict=False)) is None

    def test_a_non_boolean_value_is_refused(self):
        with pytest.raises(ConfigurationError, match="must be a boolean"):
            pdc.PredictDataConsistency.from_config(_cfg(apply_at_predict="maybe"))

    def test_an_unsupported_noise_type_is_refused_at_construction(self, monkeypatch):
        _register(monkeypatch, "stub", "kspace")
        with pytest.raises(ConfigurationError, match="unsupported noise_type"):
            pdc.PredictDataConsistency.from_config(_cfg(noise_type="rician"))

    def test_provenance_reports_the_run(self, kspace_step):
        pred, meas, mask = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8), _checkerboard()
        kspace_step.finalize(pred, mask=mask, measured_kspace=meas, strategy="t")
        kspace_step.begin()
        kspace_step.note_applied("SomeSampler.run_inference")
        kspace_step.finalize(pred, mask=mask, measured_kspace=meas, strategy="t")
        assert kspace_step.provenance() == {
            "apply_at_predict": True,
            "domain": "kspace",
            "model_type": "stub",
            "calls": 2,
            "projections_by_predict_step": 1,
            "skipped_already_applied": 1,
            "applied_by": {"predict_step": 1, "SomeSampler.run_inference": 1},
            "measurement_noise_added": False,
        }


class TestMissingInputsRaiseNamingTheKnob:
    def test_no_mask(self, kspace_step):
        with pytest.raises(
            ConfigurationError, match=f"{KNOB} is true but this batch carries no mask"
        ):
            kspace_step.project(
                torch.randn(1, 2, 8, 8), mask=None, measured_kspace=torch.randn(1, 2, 8, 8)
            )

    def test_no_measurement(self, kspace_step):
        with pytest.raises(ConfigurationError, match="carries no measured_kspace"):
            kspace_step.project(torch.randn(1, 2, 8, 8), mask=_checkerboard(), measured_kspace=None)

    def test_neither_names_both(self, kspace_step):
        with pytest.raises(ConfigurationError, match="no mask and no measured_kspace"):
            kspace_step.project(torch.randn(1, 2, 8, 8), mask=None, measured_kspace=None)

    def test_the_finalize_path_raises_the_same_way(self, kspace_step):
        """The hook goes through ``finalize``; a missing mask must not become a skip."""
        with pytest.raises(ConfigurationError, match=KNOB.replace(".", r"\.")):
            kspace_step.finalize(
                torch.randn(1, 2, 8, 8), mask=None, measured_kspace=None, strategy="t"
            )


class TestNotTwice:
    def test_a_sampler_that_pinned_the_measurement_is_not_projected_again(
        self, kspace_step, monkeypatch
    ):
        calls: list[int] = []
        real = kspace_step.project
        monkeypatch.setattr(
            kspace_step, "project", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        pred = torch.randn(1, 2, 8, 8)
        kspace_step.note_applied("ColdDiffusionInferenceStrategy.run_inference")
        out = kspace_step.finalize(
            pred, mask=_checkerboard(), measured_kspace=torch.randn(1, 2, 8, 8), strategy="t"
        )
        assert calls == [] and out is pred
        assert kspace_step.skipped_already_applied == 1

    def test_begin_resets_the_ledger_per_call(self, kspace_step):
        kspace_step.note_applied("x")
        assert kspace_step.applied_this_call
        kspace_step.begin()
        assert not kspace_step.applied_this_call
        assert kspace_step.calls == 2


class TestDomainResolution:
    @pytest.mark.parametrize(
        ("domain", "expected"),
        [("kspace", "is not in MODEL_REGISTRY"), (None, "declares no output_domain")],
    )
    def test_unregistered_and_unannotated_models_refuse(self, monkeypatch, domain, expected):
        if domain is None:
            _register(monkeypatch, "stub", None)
        else:
            monkeypatch.setattr(pdc, "get_model_capabilities", lambda name: None)
            monkeypatch.delitem(pdc.MODEL_REGISTRY, "stub", raising=False)
        with pytest.raises(ConfigurationError, match=expected):
            pdc.PredictDataConsistency.from_config(_cfg())

    def test_complex_image_is_refused_by_name(self, monkeypatch):
        _register(monkeypatch, "stub", "complex_image")
        with pytest.raises(ConfigurationError, match="output_domain='complex_image'"):
            pdc.PredictDataConsistency.from_config(_cfg())

    def test_a_tuple_of_domains_is_refused(self, monkeypatch):
        _register(monkeypatch, "stub", ("kspace", "image"))
        with pytest.raises(ConfigurationError, match="declares output_domain="):
            pdc.PredictDataConsistency.from_config(_cfg())

    def test_the_two_supported_domains_pick_the_layer_path(self, kspace_step, image_step):
        assert kspace_step.is_kspace_domain is True
        assert image_step.is_kspace_domain is False


class TestMaskLayouts:
    @pytest.mark.parametrize("shape", [(8, 8), (1, 8, 8), (2, 8, 8), (2, 1, 8, 8), (2, 2, 8, 8)])
    def test_accepted_layouts_broadcast_to_the_batch(self, kspace_step, shape):
        mask = torch.zeros(*shape)
        mask[..., ::2] = 1.0
        pred, meas = torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
        out = kspace_step.project(pred, mask=mask, measured_kspace=meas)
        assert torch.equal(out[..., ::2], meas[..., ::2])
        assert torch.equal(out[..., 1::2], pred[..., 1::2])

    @pytest.mark.parametrize(
        ("mask", "reason"),
        [
            (torch.ones(8), "1-D line mask"),
            (torch.ones(1, 1, 1, 8, 8), "1-D line mask"),
            (torch.ones(8, 4), "does not match"),
            (torch.ones(3, 8, 8), "leading entries"),
            (torch.ones(2, 3, 8, 8), "channel"),
            (torch.full((8, 8), 0.5), "must be binary"),
            (torch.ones(8, 8, dtype=torch.complex64), "real-valued"),
        ],
    )
    def test_layouts_that_would_need_a_guess_are_refused(self, kspace_step, mask, reason):
        with pytest.raises(ConfigurationError, match=reason):
            kspace_step.project(
                torch.randn(2, 2, 8, 8), mask=mask, measured_kspace=torch.randn(2, 2, 8, 8)
            )

    def test_bool_and_int_masks_are_accepted(self, kspace_step):
        pred, meas = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
        for mask in (_checkerboard().bool(), _checkerboard().long()):
            out = kspace_step.project(pred, mask=mask, measured_kspace=meas)
            assert torch.equal(out[..., ::2], meas[..., ::2])


class TestMeasurementLayouts:
    """The layer truncates or recasts silently; the step refuses instead."""

    def test_kspace_shape_mismatch_is_refused(self, kspace_step):
        with pytest.raises(ConfigurationError, match="share shape and dtype family"):
            kspace_step.project(
                torch.randn(1, 2, 8, 8),
                mask=_checkerboard(),
                measured_kspace=torch.randn(1, 4, 8, 8),
            )

    def test_kspace_dtype_family_mismatch_is_refused(self, kspace_step):
        with pytest.raises(ConfigurationError, match="share shape and dtype family"):
            kspace_step.project(
                torch.randn(1, 1, 8, 8),
                mask=_checkerboard(),
                measured_kspace=torch.randn(1, 1, 8, 8, dtype=torch.complex64),
            )

    def test_image_with_an_even_real_channel_count_is_refused(self, image_step):
        """The layer reads such a tensor as k-space whatever the declared domain."""
        with pytest.raises(ConfigurationError, match="reads as k-space regardless"):
            image_step.project(
                torch.rand(1, 2, 8, 8),
                mask=_checkerboard(),
                measured_kspace=torch.randn(1, 1, 8, 8, dtype=torch.complex64),
            )

    def test_multi_coil_real_measurement_is_refused(self, image_step):
        with pytest.raises(ConfigurationError, match="interleaved"):
            image_step.project(
                torch.rand(1, 1, 8, 8),
                mask=_checkerboard(),
                measured_kspace=torch.randn(1, 4, 8, 8),
            )

    def test_channel_count_mismatch_is_refused_instead_of_truncated(self, image_step):
        with pytest.raises(ConfigurationError, match="would truncate silently"):
            image_step.project(
                torch.rand(1, 1, 8, 8),
                mask=_checkerboard(),
                measured_kspace=torch.randn(1, 3, 8, 8, dtype=torch.complex64),
            )

    def test_measurement_is_moved_to_the_prediction_device(self, kspace_step):
        pred = torch.randn(1, 2, 8, 8)
        out = kspace_step.project(
            pred, mask=_checkerboard(), measured_kspace=torch.randn(1, 2, 8, 8)
        )
        assert out.device == pred.device
