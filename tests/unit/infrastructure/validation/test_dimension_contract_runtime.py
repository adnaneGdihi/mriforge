"""Tests for the Layer-2 runtime dimension-contract guard.

Covers the pure firewall (``assert_input_contract``, ``first_tensor``,
``resolve_contract_mode``) and an end-to-end demonstration that the same
``register_forward_pre_hook`` mechanism the strategy installs raises in
``enforce`` mode and only logs in ``observe`` mode.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.validation.dimension_contract import (
    DimensionContractError,
    assert_input_contract,
    first_tensor,
    resolve_contract_mode,
)
from spectramr.models.capabilities import ModelCapabilities

# --------------------------------------------------------------------------
# resolve_contract_mode (the validated knob)
# --------------------------------------------------------------------------


def test_mode_default_is_observe(monkeypatch):
    monkeypatch.delenv("SPECTRAMR_DIMENSION_CONTRACT", raising=False)
    assert resolve_contract_mode() == "observe"


@pytest.mark.parametrize("mode", ["off", "observe", "enforce", "ENFORCE", " observe "])
def test_mode_valid_values(monkeypatch, mode):
    monkeypatch.setenv("SPECTRAMR_DIMENSION_CONTRACT", mode)
    assert resolve_contract_mode() in ("off", "observe", "enforce")


def test_mode_unknown_raises(monkeypatch):
    monkeypatch.setenv("SPECTRAMR_DIMENSION_CONTRACT", "3d")
    with pytest.raises(ValueError):
        resolve_contract_mode()


# --------------------------------------------------------------------------
# first_tensor
# --------------------------------------------------------------------------


def test_first_tensor_skips_non_tensors():
    x = torch.randn(1, 2, 4, 4)
    assert first_tensor((7, "label", x)) is x
    assert first_tensor(x) is x
    assert first_tensor((1, 2, 3)) is None


# --------------------------------------------------------------------------
# assert_input_contract — the firewall
# --------------------------------------------------------------------------


def test_unannotated_is_noop():
    # caps None → never raises (legacy models unaffected).
    assert_input_contract(torch.randn(1, 1, 8, 8, 8), None, where="legacy")


def test_2d_model_accepts_4d():
    caps = ModelCapabilities(spatial_dims=(2,))
    assert_input_contract(torch.randn(2, 3, 16, 16), caps, where="unet")  # no raise


def test_2d_model_rejects_5d():
    caps = ModelCapabilities(spatial_dims=(2,))
    with pytest.raises(DimensionContractError) as ei:
        assert_input_contract(torch.randn(2, 3, 4, 16, 16), caps, where="unet")
    assert "spatial rank 3" in str(ei.value)


def test_3d_model_accepts_5d():
    caps = ModelCapabilities(spatial_dims=(3,))
    assert_input_contract(torch.randn(2, 3, 4, 16, 16), caps, where="vol")  # no raise


def test_never_asserts_channel_count():
    # A 2D model with ANY channel count on a 4D tensor must pass — the guard must
    # NOT assert channels == in_channels (conditioning-concat changes it).
    caps = ModelCapabilities(spatial_dims=(2,))
    for ch in (1, 2, 3, 7, 64):
        assert_input_contract(torch.randn(1, ch, 8, 8), caps, where="unet")


def test_interleaved_requires_even_channels():
    caps = ModelCapabilities(
        spatial_dims=(2,), expects_real_imag_interleaved=True
    )
    assert_input_contract(torch.randn(1, 4, 8, 8), caps, where="cx")  # even ok
    with pytest.raises(DimensionContractError):
        assert_input_contract(torch.randn(1, 3, 8, 8), caps, where="cx")  # odd


def test_reshape_internal_model_untouched():
    # The guard only sees the INPUT. A model that flattens coils internally still
    # receives a normal [B, C, H, W] input — no false positive.
    caps = ModelCapabilities(spatial_dims=(2,), expects_real_imag_interleaved=True)
    assert_input_contract(torch.randn(1, 8, 32, 32), caps, where="coil_flatten")


# --------------------------------------------------------------------------
# End-to-end: the register_forward_pre_hook mechanism the strategy installs
# --------------------------------------------------------------------------


class _Tiny(nn.Module):
    def forward(self, x):
        return x


def _install_guard(module, caps, *, enforce):
    def _pre(_m, inputs):
        x = first_tensor(inputs)
        if x is None:
            return
        try:
            assert_input_contract(x, caps, where="tiny")
        except DimensionContractError:
            if enforce:
                raise

    return module.register_forward_pre_hook(_pre)


def test_pre_hook_enforce_raises_on_bad_rank():
    m = _Tiny()
    _install_guard(m, ModelCapabilities(spatial_dims=(2,)), enforce=True)
    m(torch.randn(1, 2, 8, 8))  # 4D ok
    with pytest.raises(DimensionContractError):
        m(torch.randn(1, 2, 4, 8, 8))  # 5D into 2D model → raise at forward


def test_pre_hook_observe_does_not_raise():
    m = _Tiny()
    _install_guard(m, ModelCapabilities(spatial_dims=(2,)), enforce=False)
    # observe mode swallows the violation (logged, not raised)
    out = m(torch.randn(1, 2, 4, 8, 8))
    assert out.shape == (1, 2, 4, 8, 8)
