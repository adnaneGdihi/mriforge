"""Unit tests for :class:`ActNorm2d`, the Glow activation-normalization block.

These pin the two halves of the data-dependent-init contract, which are easy
to break together because only one of them is visible at training time:

* the **hot-path cost** — reading the ``initialized`` buffer used to cost a
  ``.item()`` device sync on every forward (non-negotiable 9). A Python memo
  now answers after the first call.
* the **serialization SSOT** — ``initialized`` must stay a registered buffer.
  Demoting it to a plain Python attribute (the shape production-plan row D08#6
  originally prescribed) removes it from ``state_dict``, so a resumed model
  re-runs ``_initialize`` on its first batch and silently overwrites the
  trained ``logs``/``bias``. Measured drift on an 8x4x6x6 resume:
  ``max |Δlogs| = 3.465``. ``test_state_dict_still_carries_the_init_flag`` and
  ``test_resume_does_not_reinitialize`` exist to turn that change red.

The two directions of the flag are both tested on purpose: a memo that
unconditionally reports "initialized" passes the resume test and still breaks
every checkpoint saved before the first forward.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.flow.actnorm import ActNorm2d  # noqa: E402


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


# ── data-dependent init ───────────────────────────────────────────


def test_first_forward_standardises_the_activations() -> None:
    """The init makes the post-transform output zero-mean / unit-variance."""
    block = ActNorm2d(4)
    x = torch.randn(16, 4, 8, 8) * 3.0 + 1.5
    y, _ = block(x)
    assert torch.allclose(y.mean(dim=(0, 2, 3)), torch.zeros(4), atol=1e-4)
    assert torch.allclose(y.std(dim=(0, 2, 3)), torch.ones(4), atol=1e-3)


def test_a_reverse_first_call_does_not_initialize() -> None:
    """``reverse=True`` must never trigger the data-dependent init."""
    block = ActNorm2d(4)
    block(torch.randn(2, 4, 5, 5), reverse=True)
    assert not bool(block.initialized.item())
    assert torch.equal(block.logs.data, torch.zeros(1, 4, 1, 1))


# ── serialization SSOT ────────────────────────────────────────────


def test_state_dict_still_carries_the_init_flag() -> None:
    """``initialized`` is a buffer; dropping it breaks strict checkpoint loads."""
    assert "initialized" in ActNorm2d(4).state_dict()


def test_resume_does_not_reinitialize() -> None:
    """A checkpoint saved post-init survives a forward on a different batch."""
    trained = ActNorm2d(4)
    trained(torch.randn(8, 4, 6, 6) * 3.0 + 1.5)
    checkpoint = {k: v.clone() for k, v in trained.state_dict().items()}

    resumed = ActNorm2d(4)
    resumed.load_state_dict(checkpoint)
    resumed(torch.randn(8, 4, 6, 6) * 0.1 - 7.0)  # wildly different statistics

    assert torch.equal(resumed.logs.data, checkpoint["logs"])
    assert torch.equal(resumed.bias.data, checkpoint["bias"])


def test_a_checkpoint_saved_before_init_still_initializes() -> None:
    """The other direction: a False flag must still trigger the init."""
    fresh = ActNorm2d(4)
    checkpoint = {k: v.clone() for k, v in fresh.state_dict().items()}
    assert not bool(checkpoint["initialized"].item())

    resumed = ActNorm2d(4)
    resumed.load_state_dict(checkpoint)
    resumed(torch.randn(8, 4, 6, 6) * 3.0 + 1.5)

    assert bool(resumed.initialized.item())
    assert not torch.equal(resumed.logs.data, checkpoint["logs"])


# ── hot-path device syncs (non-negotiable 9) ──────────────────────


def test_steady_state_forward_makes_no_device_sync(item_calls: list[str]) -> None:
    """After the first forward, the flag is answered from Python, not the buffer."""
    block = ActNorm2d(4)
    x = torch.randn(2, 4, 5, 5)
    block(x)
    item_calls.clear()
    block(x)
    block(x)
    assert item_calls == []


def test_the_first_forward_reads_the_buffer_exactly_once(item_calls: list[str]) -> None:
    """The memo must not skip the buffer entirely -- a resume depends on it."""
    block = ActNorm2d(4)
    block(torch.randn(2, 4, 5, 5))
    assert item_calls == ["(1,)"]


# ── transform algebra ─────────────────────────────────────────────


def test_reverse_inverts_forward() -> None:
    block = ActNorm2d(4)
    x = torch.randn(4, 4, 7, 7) * 2.0 - 1.0
    y, log_det = block(x)
    x_rt, log_det_rev = block(y, reverse=True)
    assert torch.allclose(x_rt, x, atol=1e-5)
    assert torch.allclose(log_det_rev, -log_det)


def test_log_det_matches_the_closed_form() -> None:
    block = ActNorm2d(4)
    x = torch.randn(3, 4, 7, 5)
    _, log_det = block(x)
    expected = block.logs.sum() * float(7 * 5)
    assert log_det.shape == (3,)
    assert torch.allclose(log_det, expected.expand(3))
