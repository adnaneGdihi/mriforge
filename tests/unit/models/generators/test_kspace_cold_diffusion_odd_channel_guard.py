"""Regression test for the odd-channel guard in kspace_cold_diffusion_generator.

The 2026-05-10 cluster rerun saw four cold-diffusion arms
(exp_pillar_02, exp_ulf_03, exp_ulf_06, exp_up_01) crash with::

    RuntimeError: cuFFT error: CUFFT_INVALID_SIZE

at ``fft2c(output_complex)``. Root cause: the YAMLs declared
``in_channels=1, out_channels=1`` against a model that internally
splits the channel dim with ``[:, 0::2]`` (real) and ``[:, 1::2]``
(imag). For an odd channel count, ``[:, 1::2]`` returns a zero-channel
slice → ``torch.complex(...)`` produces ``[B, 0, H, W]`` → cuFFT chokes.

The fix is two-pronged: (1) the YAMLs were corrected to ``in_channels:
2`` and (2) the generator now raises a clear ValueError on odd-channel
inputs. This test pins (2).
"""
from __future__ import annotations

SRC = "src/spectramr/models/generators/kspace_cold_diffusion_generator.py"
ANCHOR = "[KSpaceColdDiffusionGenerator] interleaved Re/Im layout"


def _read_source() -> str:
    with open(SRC) as f:
        return f.read()


def test_guard_block_present():
    """The forward path must contain a ``shape[1] % 2 != 0`` guard
    that raises ``ValueError`` before reaching the FFT call."""
    text = _read_source()
    # Anchor on the guard's distinctive condition + the raise.
    assert "attentioned_output.shape[1] % 2 != 0" in text, (
        "odd-channel guard's parity check missing"
    )
    assert ANCHOR in text, (
        "guard's tagged ValueError message missing — the cuFFT_INVALID_SIZE "
        "regression class can recur silently."
    )


def test_guard_message_actionable():
    """A vague exception leaves the user grepping. The user-facing fix
    here is YAML-level, so the message must point at ``in_channels``
    and explain the parity requirement."""
    text = _read_source()
    idx = text.find(ANCHOR)
    assert idx >= 0
    # Take ~800 chars of context — enough to span the multi-line string.
    msg_window = text[idx : idx + 800]
    assert "in_channels" in msg_window, (
        "Guard error message must hint at the YAML field to fix"
    )
    assert "even" in msg_window, (
        "Guard error message must explain the parity requirement"
    )
