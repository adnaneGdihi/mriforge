"""Phantom-builder helper for the Tier-2 forward probe.

White-noise dummy tensors test that the model's forward / backward
graph is structurally correct, but they don't tell you anything about
whether the data path is *handling structure* correctly. A
Shepp-Logan phantom does — when it goes in, you can eyeball whether
it comes out (a) as a recognisable image, (b) as a sensible k-space
spectrum (low-frequency-dominated), or (c) as garbled noise.

The repo already ships
:func:`mriforge.infrastructure.physics.signal_models.tse_b0_model.shepp_logan_2d`.
This module wraps it into a single helper that produces a tensor of
any rank matching the dataset's promised output shape, with optional
phase modulation (so paired ULF↔HF / k-space pipelines see a
non-trivial complex signal rather than a real one tiled into the
imag channel).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["save_probe_images", "synthetic_phantom"]


def _phase_ramp(spatial: tuple[int, ...], device: str):
    """``exp(i·2π·Σ axes)`` over the phantom's spatial dims.

    Rank-3 ``(B, C, L)`` ramps along ``L``; rank-4/5 keep the historical
    ``(H, W)`` ramp, which broadcasts across ``D`` for 5-D input.
    """
    import torch

    dims = spatial if len(spatial) == 1 else spatial[-2:]
    axes = [torch.linspace(0.0, 2.0 * math.pi, int(n), device=device) for n in dims]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.exp(1j * sum(grid))


def _apply_dtype_and_phase(out, *, dtype: str, add_phase: bool, device: str):
    """Cast ``out`` to ``dtype`` and optionally modulate it with a phase ramp.

    Shared by every rank branch of :func:`synthetic_phantom` so a 1-D ``(B,C,L)``
    phantom honours ``add_phase`` exactly like the 2-D/3-D ones. The rank-3 branch
    used to return early and skip this tail, so ``add_phase=True`` silently yielded
    a zero imaginary part -- a real signal wearing a complex dtype, which is the
    one thing this module exists to avoid (see the module docstring).
    """
    import torch

    if dtype == "complex64":
        out = out.to(torch.float32)
        if add_phase:
            phase = _phase_ramp(tuple(out.shape[2:]), device)
            return (out.to(torch.complex64) * phase).to(torch.complex64)
        return out.to(torch.complex64)
    if dtype != "float32":
        # Anything else: best-effort cast. We don't try to be clever here.
        try:
            return out.to(getattr(torch, dtype))
        except (AttributeError, TypeError):
            logger.warning("synthetic_phantom: unrecognised dtype=%s, returning float32.", dtype)
            return out.to(torch.float32)
    return out.to(torch.float32)


def synthetic_phantom(
    shape: Sequence[int],
    device: str = "cpu",
    dtype: str = "float32",
    add_phase: bool = False,
    seed: int | None = None,
):
    """Build a Shepp-Logan phantom shaped to ``shape``.

    Args:
        shape: Output tensor shape. Accepted ranks:

            * 3-D (B, C, L) — 1D signal stack (HRF time courses,
              tangent-score nets, MRF fingerprints). See F-PHANTOM-RANK3.
            * 4-D (B, C, H, W) — 2D image stack.
            * 5-D (B, C, D, H, W) — 3D volume (each slice is the same
              phantom; structure is identical along D).
        device: Target device.
        dtype: ``"float32"`` (default) or ``"complex64"`` — when complex,
          and ``add_phase`` is True, the phantom is multiplied by a
          slowly-varying phase ramp so the imag channel is non-trivial.
        add_phase: Only meaningful for complex output. Adds a
          ``exp(i·2π·(x+y)/N)`` phase ramp — along ``L`` for 3-D input,
          over ``(H, W)`` otherwise.
        seed: Reserved (future: deterministic rotation / shift). Currently
          ignored — the phantom is fully deterministic.

    Returns:
        A torch tensor of shape ``shape`` and the requested dtype, on
        the requested device. For real output, values are in ``[0, 1]``.
        For complex output, the magnitude is in ``[0, 1]``.

    Raises:
        ValueError: if ``shape`` rank is not 3, 4 or 5.
        ImportError: if torch is unavailable (the probe would have
          already returned a no_probe ProbeResult before reaching here).
    """
    from mriforge.infrastructure.physics.signal_models.tse_b0_model import shepp_logan_2d

    del seed  # reserved
    shape = tuple(int(s) for s in shape)
    if len(shape) == 4:
        b, c, h, w = shape
        depth = None
    elif len(shape) == 5:
        b, c, depth, h, w = shape
    elif len(shape) == 3:
        # F-PHANTOM-RANK3 / 2026-05-20 — 1-D models (HRF time courses,
        # tangent-score networks, MRF fingerprints) consume rank-3
        # ``[B, C, L]`` tensors and previously fell through to a
        # ValueError → ``forward_probe`` then fell back to ``randn``,
        # losing the phantom's deterministic spectral content. Build a
        # 1-D Shepp-Logan signature by reading the central row of the
        # 2-D phantom so the probe still exercises a structured input.
        b, c, length = shape
        base_size = max(length, 32)
        phantom_2d = shepp_logan_2d(size=base_size, device=device).squeeze()
        center_row = phantom_2d[base_size // 2]
        l_off = (base_size - length) // 2
        signal_1d = center_row[l_off : l_off + length]
        out = signal_1d.expand(b, c, length).contiguous()
        # Shared tail, not a private cast: the branch used to return here and
        # silently drop `add_phase`, handing back a real signal in a complex
        # dtype (MRF fingerprints are complex — the probe needs the phase).
        return _apply_dtype_and_phase(out, dtype=dtype, add_phase=add_phase, device=device)
    else:
        raise ValueError(
            f"synthetic_phantom: shape must be 3-D (B,C,L), 4-D (B,C,H,W) "
            f"or 5-D (B,C,D,H,W), got rank-{len(shape)} shape {shape}."
        )

    # Build the base 2D phantom at max(H, W) and crop, so non-square
    # patches still see a recognisable phantom (vs. distorted aspect).
    base_size = max(h, w)
    phantom_2d = shepp_logan_2d(size=base_size, device=device).squeeze()  # (S, S)
    # Centre-crop to (H, W).
    h_off = (base_size - h) // 2
    w_off = (base_size - w) // 2
    phantom_2d = phantom_2d[h_off : h_off + h, w_off : w_off + w]

    # Tile across batch and channel.
    if depth is None:
        out = phantom_2d.expand(b, c, h, w).contiguous()
    else:
        out = phantom_2d.expand(b, c, depth, h, w).contiguous()

    return _apply_dtype_and_phase(out, dtype=dtype, add_phase=add_phase, device=device)


def save_probe_images(
    out_dir: str,
    arm_name: str,
    *,
    input_tensor=None,
    output_tensor=None,
    target_tensor=None,
) -> list[str]:
    """Save the probe's input / output / target tensors as PNG mosaics.

    Writes one PNG per role (``input``, ``output``, ``target``). For
    multi-slice volumes, the centre slice is saved (cheaper and the
    most diagnostic). Multi-channel tensors are tiled horizontally so
    you can see if a real/imag-as-channels layout looks right.

    The images go to ``<out_dir>/<arm_name>_<role>.png``. Returns the
    list of paths actually written.

    Failures (missing matplotlib, weird shapes, etc.) are silently
    swallowed — the probe must never fail because of an inspection
    artefact.
    """
    import os

    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written  # matplotlib missing — nothing to do.

    os.makedirs(out_dir, exist_ok=True)

    roles = {
        "input": input_tensor,
        "output": output_tensor,
        "target": target_tensor,
    }
    for role, tensor in roles.items():
        if tensor is None:
            continue
        try:
            arr = _tensor_to_2d_mosaic(tensor)
            if arr is None:
                continue
            path = os.path.join(out_dir, f"{arm_name}_{role}.png")
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(arr, cmap="gray", interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"{arm_name} — {role}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(path, dpi=80, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
        except Exception as exc:  # pragma: no cover — best-effort
            logger.debug("save_probe_images: skipping %s due to %s", role, exc)
    return written


def _tensor_to_2d_mosaic(tensor):
    """Reduce a (B, C, [D,] H, W) tensor to a 2-D ndarray for plotting.

    Channels are tiled horizontally; complex tensors are split into
    magnitude / phase rows. For 5-D volumes the centre slice is used.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    t = tensor.detach().to("cpu")

    if t.is_complex():
        mag = t.abs()
        phase = t.angle()
        return _tensor_to_2d_mosaic(
            torch.stack([mag, phase], dim=2).view(t.shape[0], 2 * t.shape[1], *t.shape[2:])
        )
    if t.ndim == 5:
        # (B, C, D, H, W) -> centre slice
        d_mid = t.shape[2] // 2
        t = t[:, :, d_mid, :, :]
    if t.ndim != 4:
        return None  # unsupported

    # Pick the first batch element only; tile channels horizontally.
    sample = t[0]  # (C, H, W)
    sample = sample - sample.min()
    sample = sample / (sample.max() + 1e-12)
    # Concatenate channels horizontally with 1-pixel separator.
    sep = sample.new_zeros((sample.shape[0], sample.shape[1], 1)) if sample.shape[0] > 0 else None
    panels = []
    for c in range(sample.shape[0]):
        if c > 0 and sep is not None:
            panels.append(sep[0])
        panels.append(sample[c])
    mosaic = (
        panels[0]
        if len(panels) == 1
        else torch.cat([p if p.ndim == 2 else p.squeeze(0) for p in panels], dim=-1)
    )
    return mosaic.numpy()
