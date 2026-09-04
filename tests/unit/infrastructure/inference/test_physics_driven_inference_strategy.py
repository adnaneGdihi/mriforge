"""``PhysicsDrivenInferenceStrategy``: its own DC loop and the predict-time ledger."""

from __future__ import annotations

import torch
from torch import nn

from spectramr.infrastructure.inference import predict_data_consistency as pdc
from spectramr.infrastructure.inference.physics_driven_inference_strategy import (
    PhysicsDrivenInferenceStrategy,
)
from spectramr.models.capabilities import ModelCapabilities


class _Identity(nn.Module):
    def forward(self, x):
        return x


def _register(monkeypatch):
    caps = ModelCapabilities(output_domain="kspace")
    monkeypatch.setitem(pdc.MODEL_REGISTRY, "stub", {"capabilities": caps})
    monkeypatch.setattr(
        pdc, "get_model_capabilities", lambda name: caps if name == "stub" else None
    )


def _cfg(apply_at_predict: bool, dc_enabled: bool = True):
    return {
        "physics": {
            "data_consistency": {
                "enabled": dc_enabled,
                "method": "hard",
                "apply_at_predict": apply_at_predict,
            }
        },
        "model": {"model_type": "stub"},
    }


def _batch():
    image = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    kspace = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2] = 1.0
    return image, kspace, mask


class TestOwnLoopIsRecorded:
    def test_the_loop_notes_its_projection_so_the_hook_skips(self, monkeypatch):
        _register(monkeypatch)
        strategy = PhysicsDrivenInferenceStrategy(_Identity(), torch.device("cpu"), _cfg(True))
        assert strategy.predict_dc is not None and strategy.dc_enabled
        image, kspace, mask = _batch()

        strategy.predict_dc.begin()
        out, meta = strategy.run_inference(
            image, preprocess_metadata={"original_kspace": kspace, "mask": mask}
        )
        assert meta["dc_enabled"] is True
        assert strategy.predict_dc.applied_this_call
        assert strategy.predict_dc_provenance()["applied_by"] == {
            "PhysicsDrivenInferenceStrategy.run_inference": 1
        }

        projected = strategy._project_onto_measurement(out, mask=mask, measured_kspace=kspace)
        assert projected is out, "already pinned by the loop; the hook must not project again"
        assert strategy.predict_dc_provenance()["skipped_already_applied"] == 1

    def test_no_kspace_in_the_metadata_means_no_note(self, monkeypatch):
        """The loop applied nothing, so the hook is still owed a projection."""
        _register(monkeypatch)
        strategy = PhysicsDrivenInferenceStrategy(_Identity(), torch.device("cpu"), _cfg(True))
        image, _kspace, _mask = _batch()
        strategy.predict_dc.begin()
        strategy.run_inference(image, preprocess_metadata={})
        assert not strategy.predict_dc.applied_this_call

    def test_off_knob_keeps_the_loop_and_records_nothing(self):
        strategy = PhysicsDrivenInferenceStrategy(_Identity(), torch.device("cpu"), _cfg(False))
        assert strategy.predict_dc is None
        image, kspace, mask = _batch()
        out, meta = strategy.run_inference(
            image, preprocess_metadata={"original_kspace": kspace, "mask": mask}
        )
        assert meta["dc_enabled"] is True and out.shape == image.shape
        assert strategy.predict_dc_provenance() == {"apply_at_predict": False}
