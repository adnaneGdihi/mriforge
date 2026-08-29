"""Tests for SliceToVolumeRequest.validate (WS-3).

Regression: ``validate()`` used ``all(dim <= 0 ...)`` so a shape with a single
non-positive dimension (e.g. ``(-1, 64, 64)``) wrongly passed; it now uses
``any(...)``.
"""

from __future__ import annotations

from mriforge.domain.entities.entities_3d import SliceToVolumeRequest


def _req(conditioning_axis: int = 2, target_shape: tuple[int, ...] = (64, 64, 64)):
    return SliceToVolumeRequest(
        request_id="r",
        slice_data=None,
        conditioning_axis=conditioning_axis,
        target_shape=target_shape,
        generation_params={},
    )


def test_valid_request() -> None:
    assert _req().validate() is True


def test_rejects_bad_conditioning_axis() -> None:
    assert _req(conditioning_axis=5).validate() is False


def test_rejects_wrong_rank_shape() -> None:
    assert _req(target_shape=(64, 64)).validate() is False


def test_rejects_single_negative_dim() -> None:
    # The bug: (-1, 64, 64) passed under all(); must fail under any().
    assert _req(target_shape=(-1, 64, 64)).validate() is False


def test_rejects_zero_dim() -> None:
    assert _req(target_shape=(0, 64, 64)).validate() is False
