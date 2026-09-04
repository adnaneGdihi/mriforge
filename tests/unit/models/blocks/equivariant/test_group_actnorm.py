"""Unit tests for :class:`GroupActNorm2d`, the C_n-equivariant ActNorm.

This is the sibling of ``tests/unit/models/blocks/flow/test_actnorm.py`` and
pins the same two halves of the data-dependent-init contract -- the hot-path
device sync (non-negotiable 9) and the ``initialized`` buffer as the
serialization SSOT. The cases are written out rather than shared with the
Glow variant through a parametrised loop: the two classes have different
parameter shapes and different log-det algebra, and a loop that walks both
lets one of them fail while the other keeps the assertion honest.

The group-specific invariant on top of that: scale/bias are stored per *base*
channel and broadcast across the group axis, so the init must average over the
orbit. ``test_init_shares_parameters_across_the_group_orbit`` is what turns a
regression to per-channel statistics red.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.equivariant.group_actnorm import (  # noqa: E402
    GroupActNorm2d,
)

GROUP_ORDER = 4
BASE_CHANNELS = 3
CHANNELS = GROUP_ORDER * BASE_CHANNELS


@pytest.fixture
def item_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Count every ``Tensor.item()`` call, i.e. every host/device sync."""
    calls: list[str] = []
    original = torch.Tensor.item

    def counting(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        calls.append(str(tuple(self.shape)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counting)
    yield calls


def _block() -> GroupActNorm2d:
    return GroupActNorm2d(CHANNELS, group_order=GROUP_ORDER)


# ── data-dependent init ───────────────────────────────────────────


def test_init_shares_parameters_across_the_group_orbit() -> None:
    """Parameters are per base channel and the init standardises per orbit.

    Shape alone is not enough: it is invariant under "the init never ran", so
    the statistics are asserted too -- the post-transform activations must be
    zero-mean / unit-variance when reduced over batch, group axis and space.
    """
    block = _block()
    x = torch.randn(16, CHANNELS, 8, 8) * 2.0 + 1.0
    y, _ = block(x)
    assert block.logs.shape == (1, BASE_CHANNELS, 1, 1)
    assert block.bias.shape == (1, BASE_CHANNELS, 1, 1)

    b, _, h, w = y.shape
    orbits = y.view(b, BASE_CHANNELS, GROUP_ORDER, h, w)
    assert torch.allclose(orbits.mean(dim=(0, 2, 3, 4)), torch.zeros(BASE_CHANNELS), atol=1e-4)
    assert torch.allclose(orbits.std(dim=(0, 2, 3, 4)), torch.ones(BASE_CHANNELS), atol=1e-3)


def test_a_reverse_first_call_does_not_initialize() -> None:
    """``reverse=True`` must never trigger the data-dependent init."""
    block = _block()
    block(torch.randn(2, CHANNELS, 5, 5), reverse=True)
    assert not bool(block.initialized.item())
    assert torch.equal(block.logs.data, torch.zeros(1, BASE_CHANNELS, 1, 1))


# ── serialization SSOT ────────────────────────────────────────────


def test_state_dict_still_carries_the_init_flag() -> None:
    """``initialized`` is a buffer; dropping it breaks strict checkpoint loads."""
    assert "initialized" in _block().state_dict()


def test_resume_does_not_reinitialize() -> None:
    """A checkpoint saved post-init survives a forward on a different batch."""
    trained = _block()
    trained(torch.randn(8, CHANNELS, 6, 6) * 3.0 + 1.5)
    checkpoint = {k: v.clone() for k, v in trained.state_dict().items()}

    resumed = _block()
    resumed.load_state_dict(checkpoint)
    resumed(torch.randn(8, CHANNELS, 6, 6) * 0.1 - 7.0)  # different statistics

    assert torch.equal(resumed.logs.data, checkpoint["logs"])
    assert torch.equal(resumed.bias.data, checkpoint["bias"])


def test_a_checkpoint_saved_before_init_still_initializes() -> None:
    """The other direction: a False flag must still trigger the init."""
    fresh = _block()
    checkpoint = {k: v.clone() for k, v in fresh.state_dict().items()}
    assert not bool(checkpoint["initialized"].item())

    resumed = _block()
    resumed.load_state_dict(checkpoint)
    resumed(torch.randn(8, CHANNELS, 6, 6) * 3.0 + 1.5)

    assert bool(resumed.initialized.item())
    assert not torch.equal(resumed.logs.data, checkpoint["logs"])


# ── hot-path device syncs (non-negotiable 9) ──────────────────────


def test_steady_state_forward_makes_no_device_sync(item_calls: list[str]) -> None:
    """After the first forward, the flag is answered from Python, not the buffer."""
    block = _block()
    x = torch.randn(2, CHANNELS, 5, 5)
    block(x)
    item_calls.clear()
    block(x)
    block(x)
    assert item_calls == []


def test_the_first_forward_reads_the_buffer_exactly_once(item_calls: list[str]) -> None:
    """The memo must not skip the buffer entirely -- a resume depends on it."""
    block = _block()
    block(torch.randn(2, CHANNELS, 5, 5))
    assert item_calls == ["(1,)"]


# ── transform algebra ─────────────────────────────────────────────


def test_reverse_inverts_forward() -> None:
    block = _block()
    x = torch.randn(4, CHANNELS, 7, 7) * 2.0 - 1.0
    y, log_det = block(x)
    x_rt, log_det_rev = block(y, reverse=True)
    assert torch.allclose(x_rt, x, atol=1e-5)
    assert torch.allclose(log_det_rev, -log_det)


def test_log_det_counts_the_group_axis() -> None:
    """log_det = H * W * group_order * sum_base(logs) -- the orbit multiplicity."""
    block = _block()
    x = torch.randn(3, CHANNELS, 7, 5)
    _, log_det = block(x)
    expected = block.logs.sum() * float(7 * 5) * GROUP_ORDER
    assert log_det.shape == (3,)
    assert torch.allclose(log_det, expected.expand(3))
