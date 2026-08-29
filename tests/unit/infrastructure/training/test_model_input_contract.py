"""The content half of the model-input snapshot contract (#1298).

The presence half lives in
``strategies/test_base_model_input_contract_2026_08.py``: it proves a snapshot
arrived under the declared tag. These cases cover the part that was missing --
whether what arrived describes the tensor the model was actually fed.

The split between raising and warning is the subject of several cases below and
is not stylistic. Naming failures are static: a class that never says which of
its tensors is the model input, or names one it does not emit, is wrong before a
batch is loaded. Width failures rest on a heuristic read of the backbone, so a
false positive there would abort a real run over a diagnostic.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mriforge.infrastructure.training.model_input_contract import (
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_UNRESOLVED,
    require_model_input_key,
    resolve_model_in_channels,
    verify_model_input,
)

# ── Naming: raises ─────────────────────────────────────────────────────────


def test_an_unnamed_model_input_raises() -> None:
    with pytest.raises(ValueError, match="without naming which of its tensors"):
        require_model_input_key(
            strategy_name="S", tag="t", tensors={"a": None, "b": None}, model_input_key=None
        )


def test_the_message_lists_the_keys_the_reader_would_have_to_guess_between() -> None:
    """The point of the raise is to be actionable at the emitter, not generic."""
    with pytest.raises(ValueError) as exc:
        require_model_input_key(
            strategy_name="VirtualFiducialStrategy",
            tag="vf_twin",
            tensors={"twin_corrupted_input": None, "twin_target_clean": None},
            model_input_key=None,
        )
    assert "VirtualFiducialStrategy" in str(exc.value)
    assert "vf_twin" in str(exc.value)
    assert "twin_corrupted_input" in str(exc.value)


def test_naming_an_absent_key_raises() -> None:
    """#1298: the label pointed at a tensor the snapshot did not contain."""
    with pytest.raises(ValueError, match="but that snapshot holds"):
        require_model_input_key(
            strategy_name="DiffusionTrainingStrategy",
            tag="diffusion_step",
            tensors={"noisy_kspace": None},
            model_input_key="model_input",
        )


def test_an_empty_string_is_not_a_name() -> None:
    """`extra.get(...)` on a dict that never set the key yields None; a caller
    passing "" is the same defect and must not slip through a truthiness gap."""
    with pytest.raises(ValueError, match="without naming"):
        require_model_input_key(
            strategy_name="S", tag="t", tensors={"a": None}, model_input_key=""
        )


def test_a_correct_name_is_returned_unchanged() -> None:
    assert (
        require_model_input_key(
            strategy_name="S", tag="t", tensors={"a": None}, model_input_key="a"
        )
        == "a"
    )


# ── Resolving the backbone's declared width ────────────────────────────────


def test_a_declared_in_channels_attribute_wins() -> None:
    """Most authoritative: what the architecture says it takes. It also survives
    a first layer this resolver would not otherwise recognise."""

    class _Net(nn.Module):
        in_channels = 16

        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(3, 4, 1)  # deliberately disagrees

    channels, source = resolve_model_in_channels(_Net())
    assert channels == 16
    assert source == "module.in_channels"


def test_the_first_conv_is_used_when_nothing_is_declared() -> None:
    net = nn.Sequential(nn.Conv2d(7, 4, 3), nn.ReLU(), nn.Conv2d(4, 4, 3))
    assert resolve_model_in_channels(net) == (7, "first Conv2d")


def test_conv3d_counts_too() -> None:
    net = nn.Sequential(nn.Conv3d(5, 2, 3))
    channels, source = resolve_model_in_channels(net)
    assert channels == 5
    assert "Conv3d" in source


def test_a_linear_first_layer_falls_back_to_in_features() -> None:
    net = nn.Sequential(nn.Linear(12, 4))
    assert resolve_model_in_channels(net) == (12, "first Linear.in_features")


def test_a_conv_is_preferred_over_a_linear_that_comes_first() -> None:
    """Ordering, not position: a channel width is what the caller compares
    against, and a projection head placed early would answer a different
    question."""
    net = nn.Sequential(nn.Linear(12, 4), nn.Conv2d(9, 4, 3))
    assert resolve_model_in_channels(net) == (9, "first Conv2d")


def test_an_unrecognisable_backbone_is_unresolved_not_zero() -> None:
    assert resolve_model_in_channels(nn.Sequential(nn.ReLU())) == (None, "unresolved")
    assert resolve_model_in_channels(None) == (None, "unresolved")


def test_a_nonsense_in_channels_does_not_win() -> None:
    """A Mock's auto-attribute is truthy and not an int; taking it would make
    every mock-fed strategy test report a mismatch."""

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = "sixteen"
            self.head = nn.Conv2d(6, 4, 1)

    assert resolve_model_in_channels(_Net()) == (6, "first Conv2d")


# ── The comparison ─────────────────────────────────────────────────────────


def _verdict(tensor, in_channels, *, key="model_input", source="test"):
    return verify_model_input(
        tensors={key: tensor},
        model_input_key=key,
        in_channels=in_channels,
        in_channels_source=source,
    )


def test_matching_widths_report_a_match() -> None:
    v = _verdict(torch.randn(2, 16, 8, 8), 16)
    assert v.status == STATUS_MATCH
    assert v.declared_channels == 16
    assert v.model_in_channels == 16


def test_the_exact_1298_defect_reports_a_mismatch() -> None:
    """8 recorded channels against the 16-plane first conv the arm really had."""
    v = _verdict(torch.randn(1, 8, 320, 320), 16, key="noisy_kspace")
    assert v.status == STATUS_MISMATCH
    assert v.declared_channels == 8
    assert v.model_in_channels == 16


def test_the_detail_says_which_of_the_two_readings_it_could_be() -> None:
    """A mismatch is ambiguous by nature -- a mislabelled key or real miswiring
    -- and a verdict that asserted one would send half its readers the wrong
    way."""
    detail = _verdict(torch.randn(1, 8, 8, 8), 16).detail
    assert "wrong tensor" in detail
    assert "did not declare" in detail


def test_a_complex_tensor_may_feed_twice_its_channel_count() -> None:
    """Half this repo's k-space backbones take a real view of a complex input.
    The snapshot records the tensor as the strategy holds it, before that view."""
    v = _verdict(torch.randn(1, 8, 8, 8, dtype=torch.complex64), 16)
    assert v.status == STATUS_MATCH
    assert v.is_complex is True
    assert v.accepted_channels == (8, 16)


def test_a_complex_tensor_also_matches_its_own_width() -> None:
    """A native complex conv takes C, not 2C. Both are legal, so both accept."""
    assert _verdict(torch.randn(1, 8, 8, 8, dtype=torch.complex64), 8).status == STATUS_MATCH


def test_a_real_tensor_does_not_get_the_doubling_allowance() -> None:
    """The allowance exists for the real-view transform; extending it to real
    inputs would wave through exactly the 8-vs-16 confusion of #1298."""
    v = _verdict(torch.randn(1, 8, 8, 8), 16)
    assert v.status == STATUS_MISMATCH
    assert v.accepted_channels == (8,)


def test_an_unresolvable_width_is_unresolved_and_still_records_the_tensor() -> None:
    v = _verdict(torch.randn(1, 16, 8, 8), None, source="unresolved")
    assert v.status == STATUS_UNRESOLVED
    assert v.declared_channels == 16, "the knowable half is still recorded"
    assert v.model_in_channels is None


def test_a_tensor_without_a_channel_axis_is_unresolved_not_a_mismatch() -> None:
    v = _verdict(torch.randn(8), 16)
    assert v.status == STATUS_UNRESOLVED


def test_a_non_tensor_under_the_named_key_is_unresolved() -> None:
    """`_declare_model_input` takes `dict[str, Any]`; a scalar or None there is
    a defect, but not one this comparison can adjudicate."""
    assert _verdict(None, 16).status == STATUS_UNRESOLVED


# ── The record that reaches the artifact ───────────────────────────────────


def test_the_record_carries_declared_and_applied_side_by_side() -> None:
    """Non-negotiable 14: the divergence must be readable as a subtraction, not
    re-derived by a reader who still happens to have the model."""
    record = _verdict(torch.randn(1, 8, 8, 8), 16, key="noisy_kspace").as_record()
    assert record["model_input_key"] == "noisy_kspace"
    assert record["declared"]["channels"] == 8
    assert record["declared"]["shape"] == [1, 8, 8, 8]
    assert record["applied"]["in_channels"] == 16
    assert record["applied"]["resolved_from"] == "test"
    assert record["status"] == STATUS_MISMATCH
    assert record["detail"]


def test_the_record_is_json_serialisable() -> None:
    """It is stamped as a top-level JSON key, bypassing `_coerce_extra` -- so
    nothing downstream will stringify a stray tuple for it."""
    import json

    json.dumps(_verdict(torch.randn(1, 16, 8, 8), 16).as_record())
