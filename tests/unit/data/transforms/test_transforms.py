"""Tests for deprecated coil-compression shims in ``transforms.py``.

``VirtualCoilCompression`` is a legacy duplicate of the canonical
``SVDCoilCompressionTransform``. It is retained as an importable, still-functional
shim (no live dispatch path references it) but now emits a ``DeprecationWarning``
on construction steering callers to the canonical transform / the unified
``physics.coil_processing`` config block.

NOTE: ``mriforge.*`` deprecation warnings are promoted to errors during tests
(``pyproject.toml`` ``filterwarnings``), so every construction here is wrapped in
``pytest.warns``.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from mriforge.data.transforms.transforms import VirtualCoilCompression


def test_virtualcoilcompression_construction_warns() -> None:
    """Constructing the legacy transform emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="SVDCoilCompressionTransform"):
        VirtualCoilCompression(target_coils=4)


def test_virtualcoilcompression_still_compresses() -> None:
    """The shim keeps its original behavior on R/I-stacked input.

    For even channel counts the legacy transform interprets the data as
    interleaved real/imag pairs (the codebase convention): ``2*C`` real
    channels deinterleave to ``C`` complex coils, compress to ``target_coils``,
    and re-stack to ``2*target_coils`` real channels.
    """
    torch.manual_seed(0)
    with pytest.warns(DeprecationWarning):
        transform = VirtualCoilCompression(target_coils=4)

    # 8 complex coils stacked as 16 real R/I channels: [2*C, H, W, D].
    stacked = torch.randn(16, 16, 16, 1)
    subject = tio.Subject(kspace=tio.ScalarImage(tensor=stacked))

    out = transform(subject)
    # 4 virtual complex coils → 8 real R/I channels.
    assert out["kspace"].data.shape[0] == 8


def test_virtualcoilcompression_pads_when_too_few_coils() -> None:
    """Fewer physical coils than target → zero-padded up to target_coils."""
    torch.manual_seed(0)
    with pytest.warns(DeprecationWarning):
        transform = VirtualCoilCompression(target_coils=8)

    # 4 complex coils stacked as 8 real channels; target 8 → pad to 8 complex.
    stacked = torch.randn(8, 8, 8, 1)
    subject = tio.Subject(kspace=tio.ScalarImage(tensor=stacked))

    out = transform(subject)
    # 8 complex coils → 16 real R/I channels.
    assert out["kspace"].data.shape[0] == 16
