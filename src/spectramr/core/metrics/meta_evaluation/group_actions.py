"""Group action library used by ESD.

Two action sets, both implemented as pure functions ``(x, hat_x, params) ->
(x', hat_x')`` so the same transformation is applied jointly to both
arguments of a metric.

* ``G_NUIS`` (clinically benign — invariance is desirable):
    - global complex phase rotation
    - smooth multiplicative bias field with bounded log-norm
    - sub-voxel rigid translation
    - mild gamma compression

* ``G_PERT`` (clinically meaningful — sensitivity is desirable):
    - additive lesion injection (controlled amplitude + spatial extent)
    - non-rigid B-spline-like deformation
    - magnitude inversion
    - contrast permutation across multi-contrast channels (only when input
      has >1 channel)

Each entry is registered with sampling bounds and a default amplitude so
the ``rank()`` method can sweep the group with Haar-like measure (uniform
over the parameter range).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

GroupAction = Callable[[torch.Tensor, torch.Tensor, dict], tuple[torch.Tensor, torch.Tensor]]


def _make_2d(t: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reshape to [B, C, H, W] for transformations; remember original shape."""
    orig = t.shape
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0), orig
    if t.ndim == 3:
        return t.unsqueeze(0), orig
    return t, orig


# ---------------------------------------------------------------------------
# G_NUIS members
# ---------------------------------------------------------------------------


def nuis_phase(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    phi = float(params.get("phi", 0.0))
    if not torch.is_complex(x) and not torch.is_complex(hat):
        # Phase has no effect on magnitude data — return unchanged so the
        # invariance test trivially passes for magnitude-only metrics.
        return x, hat
    z = torch.complex(
        torch.tensor(float(torch.cos(torch.tensor(phi)).item())),
        torch.tensor(float(torch.sin(torch.tensor(phi)).item())),
    )
    return x * z.to(x.dtype), hat * z.to(hat.dtype)


def nuis_bias(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    h, w = x.shape[-2], x.shape[-1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    field = (1.0 + 0.05 * (a * xx + b * yy)).to(x.dtype)
    while field.ndim < x.ndim:
        field = field.unsqueeze(0)
    return x * field, hat * field


def nuis_translation(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    return _affine_translate(x, dx, dy), _affine_translate(hat, dx, dy)


def _affine_translate(t: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    t2, orig = _make_2d(t)
    n, c, h, w = t2.shape
    # theta and grid must live on t2.device — without the device kwarg
    # PyTorch errors on mixed CPU/CUDA grid_sample inputs (or, worse,
    # implicitly transfers per call and looks like a hang inside the
    # ESD ranker's O(M·|G|·N·haar) loop).
    theta = (
        torch.tensor(
            [[1, 0, dx / max(w, 1)], [0, 1, dy / max(h, 1)]],
            dtype=torch.float32,
            device=t2.device,
        )
        .unsqueeze(0)
        .repeat(n, 1, 1)
    )
    grid = torch.nn.functional.affine_grid(theta, list(t2.shape), align_corners=False)
    is_complex = torch.is_complex(t2)
    if is_complex:
        real = torch.nn.functional.grid_sample(
            t2.real.float(), grid, mode="bilinear", align_corners=False
        )
        imag = torch.nn.functional.grid_sample(
            t2.imag.float(), grid, mode="bilinear", align_corners=False
        )
        out = torch.complex(real, imag).to(t2.dtype)
    else:
        out = torch.nn.functional.grid_sample(
            t2.float(), grid, mode="bilinear", align_corners=False
        ).to(t2.dtype)
    return out.view(orig)


def nuis_gamma(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    g = float(params.get("gamma", 1.0))
    eps = 1e-6
    return (
        torch.sign(x.float()) * torch.pow(x.float().abs().clamp(min=eps), g).to(x.dtype),
        torch.sign(hat.float()) * torch.pow(hat.float().abs().clamp(min=eps), g).to(hat.dtype),
    )


# ---------------------------------------------------------------------------
# G_PERT members
# ---------------------------------------------------------------------------


def pert_lesion(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject a Gaussian blob into the *prediction* only.

    The expectation is that a sensitive metric will *react* to this; the
    clean reference is left untouched.
    """
    amp = float(params.get("amplitude", 0.5))
    cx = float(params.get("cx", 0.5))
    cy = float(params.get("cy", 0.5))
    sigma = float(params.get("sigma", 0.05))
    h, w = hat.shape[-2], hat.shape[-1]
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=hat.device),
        torch.linspace(0, 1, w, device=hat.device),
        indexing="ij",
    )
    blob = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    while blob.ndim < hat.ndim:
        blob = blob.unsqueeze(0)
    return x, hat + blob.to(hat.dtype)


def pert_deformation(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    amp = float(params.get("amp", 4.0))  # in pixels
    return _bspline_warp(x, amp, params), _bspline_warp(hat, amp, params)


def _bspline_warp(t: torch.Tensor, amp: float, params: dict) -> torch.Tensor:
    seed = int(params.get("seed", 0))
    # Build the control field on CPU (deterministic seeded RNG) then
    # move once to t.device. All downstream tensors that feed
    # grid_sample MUST live on t.device — see _affine_translate note.
    gen = torch.Generator().manual_seed(seed)
    t2, orig = _make_2d(t)
    n, c, h, w = t2.shape
    grid_h, grid_w = 4, 4
    ctrl = ((torch.rand((1, 2, grid_h, grid_w), generator=gen) - 0.5) * 2 * amp).to(
        device=t2.device
    )
    ctrl_up = torch.nn.functional.interpolate(
        ctrl, size=(h, w), mode="bilinear", align_corners=False
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=t2.device),
        torch.linspace(-1, 1, w, device=t2.device),
        indexing="ij",
    )
    grid = (
        torch.stack(
            [
                xx + ctrl_up[0, 0] / max(w, 1),
                yy + ctrl_up[0, 1] / max(h, 1),
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .repeat(n, 1, 1, 1)
    )
    if torch.is_complex(t2):
        real = torch.nn.functional.grid_sample(
            t2.real.float(), grid, mode="bilinear", align_corners=False
        )
        imag = torch.nn.functional.grid_sample(
            t2.imag.float(), grid, mode="bilinear", align_corners=False
        )
        out = torch.complex(real, imag).to(t2.dtype)
    else:
        out = torch.nn.functional.grid_sample(
            t2.float(), grid, mode="bilinear", align_corners=False
        ).to(t2.dtype)
    return out.view(orig)


def pert_inversion(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    return x, -hat


def pert_contrast_permute(
    x: torch.Tensor, hat: torch.Tensor, params: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 3 or x.shape[-3] < 2:
        return x, hat
    perm = torch.randperm(x.shape[-3])
    if torch.equal(perm, torch.arange(x.shape[-3])):
        perm = torch.tensor([1, 0, *list(range(2, x.shape[-3]))])
    return x, hat.index_select(-3, perm)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class GroupActionEntry:
    name: str
    action: GroupAction
    sample_params: Callable[[torch.Generator], dict]


def _sample_phase(gen: torch.Generator) -> dict:
    return {"phi": float((torch.rand(1, generator=gen).item() - 0.5) * 2 * float(torch.pi))}


def _sample_translation(gen: torch.Generator) -> dict:
    return {
        "dx": float((torch.rand(1, generator=gen).item() - 0.5) * 0.5),
        "dy": float((torch.rand(1, generator=gen).item() - 0.5) * 0.5),
    }


def _sample_lesion(gen: torch.Generator) -> dict:
    return {
        "amplitude": float(0.05 + 0.4 * torch.rand(1, generator=gen).item()),
        "cx": float(0.3 + 0.4 * torch.rand(1, generator=gen).item()),
        "cy": float(0.3 + 0.4 * torch.rand(1, generator=gen).item()),
        "sigma": 0.05,
    }


def _sample_deformation(gen: torch.Generator) -> dict:
    return {
        "amp": float(2.0 + 4.0 * torch.rand(1, generator=gen).item()),
        "seed": int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()),
    }


def _sample_bias(gen: torch.Generator) -> dict:
    return {
        "a": float((torch.rand(1, generator=gen).item() - 0.5) * 2),
        "b": float((torch.rand(1, generator=gen).item() - 0.5) * 2),
    }


def _sample_gamma(gen: torch.Generator) -> dict:
    return {"gamma": float(0.95 + 0.10 * torch.rand(1, generator=gen).item())}


G_NUIS: list[GroupActionEntry] = [
    GroupActionEntry("phase", nuis_phase, _sample_phase),
    GroupActionEntry("bias", nuis_bias, _sample_bias),
    GroupActionEntry("translation", nuis_translation, _sample_translation),
    GroupActionEntry("gamma", nuis_gamma, _sample_gamma),
]


G_PERT: list[GroupActionEntry] = [
    GroupActionEntry("lesion", pert_lesion, _sample_lesion),
    GroupActionEntry("deformation", pert_deformation, _sample_deformation),
    GroupActionEntry("inversion", pert_inversion, lambda gen: {}),
    GroupActionEntry("contrast_permute", pert_contrast_permute, lambda gen: {}),
]
