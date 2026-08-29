"""Model-specific Tier-2 invariant probes for the Phase-3 models.

The generic ``synthetic_forward_probe`` checks shape / NaN / gradient flow.
These probes go further: they assert each model's *defining mathematical
invariant* on a synthetic input, so a config that builds a structurally
broken model (e.g. a Glow whose round-trip drifts, a divergence-free flow
that isn't divergence-free) fails at audit time rather than producing
silently-wrong science. See ``TODO/phase3_model_implementations_detailed.md``
§*.5 ("Tier-2 probe").

Each probe takes ``(model, x, device)`` and returns ``(ok, message)``.
``run_invariant_probe`` returns ``None`` for models without a registered
probe, so the caller treats "no probe" and "probe passed" distinctly.
Every probe is defensive: an exception becomes a ``(False, ...)`` result,
never a raw traceback that would crash the audit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_TOL_ROUNDTRIP = 1e-3
_TOL_DIV = 1e-4
_TOL_SPHERE = 1e-4
_TOL_EQUIVAR = 1e-2


def _glow_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    z, log_det = model(x)
    lp = model.log_prob(x)
    if not torch.isfinite(lp).all():
        return False, "glow: log_prob is non-finite"
    x_rec = model.inverse(z, x.shape)
    err = (x - x_rec).abs().max().item()
    if err >= _TOL_ROUNDTRIP:
        return False, f"glow: round-trip error {err:.2e} >= {_TOL_ROUNDTRIP}"
    return True, f"glow: round-trip {err:.2e}, log_prob finite"


def _equivariant_flow_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    # Rotation-invariance of the learned density is the observable
    # consequence of C_n equivariance: log p(x) == log p(R x).
    if not hasattr(model, "log_prob"):
        # Fall back to finiteness of the forward latent.
        z, _ = model(x)
        if not torch.isfinite(z).all():
            return False, "equivariant_flow: latent non-finite"
        return True, "equivariant_flow: latent finite (no log_prob to test invariance)"
    lp = model.log_prob(x)
    lp_rot = model.log_prob(torch.rot90(x, k=1, dims=(-2, -1)))
    if not (torch.isfinite(lp).all() and torch.isfinite(lp_rot).all()):
        return False, "equivariant_flow: log_prob non-finite"
    rel = (lp - lp_rot).abs().max().item() / (lp.abs().max().item() + 1e-6)
    if rel >= _TOL_EQUIVAR:
        return False, f"equivariant_flow: rotation-invariance rel-err {rel:.2e}"
    return True, f"equivariant_flow: rotation-invariant log_prob (rel {rel:.2e})"


def _divergence_free_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    t = torch.zeros(x.shape[0], device=x.device)
    try:
        v = model(x, t)
    except TypeError:
        v = model(x)
    if v.shape[1] < 2:
        return False, "divergence_free_flow: velocity has < 2 channels"
    # interior discrete divergence d v_x/dx + d v_y/dy
    vx, vy = v[:, 0:1], v[:, 1:2]
    dvx_dx = vx[:, :, :, 1:] - vx[:, :, :, :-1]
    dvy_dy = vy[:, :, 1:, :] - vy[:, :, :-1, :]
    div = dvx_dx[:, :, :-1, :] + dvy_dy[:, :, :, :-1]
    m = div.abs().max().item()
    if m >= _TOL_DIV:
        return False, f"divergence_free_flow: |div v|_inf {m:.2e} >= {_TOL_DIV}"
    return True, f"divergence_free_flow: |div v|_inf {m:.2e}"


def _hyperspherical_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    from mriforge.models.blocks.vae.von_mises_fisher import VonMisesFisher

    mu, kappa = model.encode(x)
    z = VonMisesFisher(mu, kappa, validate=False).rsample()
    norms = z.norm(dim=-1)
    err = (norms - torch.ones_like(norms)).abs().max().item()
    if err >= _TOL_SPHERE:
        return False, f"hyperspherical_vae: |‖z‖-1|_inf {err:.2e} >= {_TOL_SPHERE}"
    return True, f"hyperspherical_vae: samples on sphere (err {err:.2e})"


def _vqvae2_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    out = model(x)
    idx_top = out.get("idx_top")
    idx_bot = out.get("idx_bot")
    k_top = getattr(getattr(model, "vq_top", None), "num_embeddings", None)
    k_bot = getattr(getattr(model, "vq_bot", None), "num_embeddings", None)
    for name, idx, k in (("idx_top", idx_top, k_top), ("idx_bot", idx_bot, k_bot)):
        if idx is None:
            continue
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or (k is not None and hi >= int(k)):
            return False, f"hierarchical_vq_vae: {name} out of range [0,{k}) -> [{lo},{hi}]"
    return True, "hierarchical_vq_vae: codebook indices in range"


def _blurring_probe(model: Any, x: torch.Tensor, device: str) -> tuple[bool, str]:
    from mriforge.infrastructure.physics.dct_ops import dct2_neumann

    T = int(getattr(model, "num_timesteps", 1000))
    t = torch.full((x.shape[0],), T // 2, dtype=torch.long, device=x.device)
    x_t = model.q_sample(x, t)
    if not torch.isfinite(x_t).all():
        return False, "blurring_diffusion: q_sample produced non-finite values"

    # High-frequency DCT energy must not grow (the heat process dissipates it).
    def hf_energy(img: torch.Tensor) -> float:
        c = dct2_neumann(img).abs()
        h, w = c.shape[-2:]
        return c[..., h // 2 :, w // 2 :].pow(2).sum().item()

    if hf_energy(x_t) > hf_energy(x) + 1e-3:
        return False, "blurring_diffusion: high-frequency energy increased after blurring"
    return True, "blurring_diffusion: high-frequency energy dissipated"


_PROBES: dict[str, Callable[[Any, torch.Tensor, str], tuple[bool, str]]] = {
    "glow": _glow_probe,
    "equivariant_flow": _equivariant_flow_probe,
    "divergence_free_flow": _divergence_free_probe,
    "hyperspherical_vae": _hyperspherical_probe,
    "hierarchical_vq_vae": _vqvae2_probe,
    "blurring_diffusion": _blurring_probe,
}


def has_invariant_probe(model_type: str | None) -> bool:
    return model_type in _PROBES


def run_invariant_probe(
    model_type: str | None, model: Any, x: torch.Tensor, device: str = "cpu"
) -> tuple[bool, str] | None:
    """Run the registered invariant probe for ``model_type``.

    Returns ``None`` when no probe is registered, else ``(ok, message)``.
    Any exception inside a probe is converted to ``(False, ...)`` so the
    audit reports a clean failure rather than crashing.
    """
    probe = _PROBES.get(model_type or "")
    if probe is None:
        return None
    try:
        with torch.no_grad():
            return probe(model, x, device)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{model_type}: invariant probe raised {type(exc).__name__}: {exc}"
