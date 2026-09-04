r"""Voxel-wise projection onto the Bloch manifold :math:`\mathcal{M}_{\text{Bloch}}(\mathcal{P})`.

Idea 2 (B-2) of ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md``. This module
implements the *projection step* that the Bloch-manifold-constrained DPS
sampler interleaves with its likelihood gradient.

Mathematical statement
-----------------------
For a fixed pulse sequence :math:`\mathcal{P}` (here SPGR, reusing the
framework's :class:`~spectramr.infrastructure.physics.differentiable_bloch.DifferentiableBlochLayer`),
the *Bloch manifold* is the image of the parameter set under the forward map

.. math::

    \mathcal{M}_{\text{Bloch}}(\mathcal{P}) =
      \bigl\{\, \mathrm{Bloch}(\theta;\mathcal{P}) : \theta=(M_0,T_1,T_2)
      \in \Theta \,\bigr\}, \qquad
      \Theta = [M_0^-,M_0^+]\times[T_1^-,T_1^+]\times[T_2^-,T_2^+].

A single scalar signal per voxel makes the inverse problem ill-posed
(one equation, three unknowns), so the projector probes the sequence at
a small fixed bank of flip angles :math:`\{\alpha_k\}` (a variable-flip-angle
/ DESPOT1-style acquisition), producing a per-voxel signal *vector*
:math:`s(\theta)\in\mathbb{R}^{K}`. The projection of an observed
per-voxel vector :math:`x_i` is

.. math::

    \theta_i^\star = \arg\min_{\theta_i\in\Theta}
      \tfrac{1}{2}\,\lVert x_i - s(\theta_i)\rVert_2^2,
    \qquad
    P_{\mathcal{M}}(x)_i = s(\theta_i^\star).

We solve the per-voxel least-squares problem with a small number of
damped Gauss-Newton (Levenberg-Marquardt) iterations, vectorised across
all voxels and coils. The Jacobian :math:`J = \partial s/\partial\theta`
is obtained from the differentiable Bloch forward map by autograd, so any
future change to the forward physics propagates automatically.

T1 = T2 barrier
---------------
The relaxation manifold carries the hard physical constraint
:math:`T_2 \le T_1`. The boundary :math:`T_1 = T_2` is a non-differentiable
corner of :math:`\Theta` (cf. Risk (2) in the plan and
:meth:`BlochRelaxationManifold.project_tangent`). We keep iterates in the
interior with a **log-barrier penalty** on the gap :math:`g=T_1-T_2`,
adding :math:`-\mu\log g` to the objective and folding its (diagonal)
Hessian contribution into the Gauss-Newton normal equations. The penalty
weight :math:`\mu` is small so it only bites near the corner. After every
step iterates are clamped to :math:`\Theta` and reflected across
:math:`T_1=T_2` if violated (mirroring
:meth:`BlochRelaxationManifold._reflect_to_manifold`).

The projector has **no learnable parameters** — it is registered as a
``utility`` model so it can be composed into the DPS sampler and resolved
through the registry-dispatcher.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

# Default variable-flip-angle probe (radians). A spread of flip angles
# makes the (M0, T1, T2) -> signal map injective enough for a stable
# Gauss-Newton fit. 3 deg .. 25 deg is a standard DESPOT1-style set.
_DEFAULT_FLIP_ANGLES_DEG: Final[tuple[float, ...]] = (3.0, 8.0, 15.0, 25.0)
_DEFAULT_TR_MS: Final[float] = 8.0
_DEFAULT_TE_MS: Final[float] = 3.0
_EPS: Final[float] = 1e-6


@register_model(
    name="bloch_manifold_projector",
    training_mode="utility",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
)
class BlochManifoldProjector(nn.Module):
    r"""Voxel-wise NLLS / Gauss-Newton projection onto the Bloch manifold.

    Args:
        pulse_sequence: Pulse-sequence label. Only ``"spgr"`` / ``"gre"``
            (steady-state SPGR) is wired to the differentiable Bloch
            layer; any other value raises (no silent fallback,
            CLAUDE.md #9).
        param_bounds: Mapping with keys ``T1``, ``T2``, ``M0`` each a
            ``(lo, hi)`` tuple in milliseconds (T1/T2) / arbitrary (M0).
            Iterates are clamped to these box bounds.
        n_inner_iterations: Number of Gauss-Newton inner iterations.
            Kept small (default 10) — the cost is
            ``O(n_inner * K * n_voxels)``.
        gn_damping: Levenberg-Marquardt damping :math:`\lambda` added to
            the Gauss-Newton normal-equation diagonal.
        flip_angles_deg: Variable-flip-angle probe (degrees). The forward
            map is evaluated at each, producing a length-``K`` signal
            vector per voxel.
        tr_ms / te_ms: Sequence timing for the SPGR forward map.
        barrier_weight: Log-barrier weight :math:`\mu` on the gap
            :math:`T_1-T_2` that keeps iterates off the non-differentiable
            ``T1 == T2`` corner.

    Shapes:
        ``forward`` accepts ``x`` of shape ``[B, C, H, W]`` (optionally a
        trailing depth ``[B, C, H, W, D]``) in image domain. ``C`` may be
        complex (``torch.complex``) or real; only the **magnitude** drives
        the relaxation fit (SPGR magnitude is the manifold coordinate).
        The returned tensor has the same shape and dtype as the input —
        for complex input the original phase is re-attached to the
        projected magnitude.
    """

    # Pure-projection module: no parameters, never trained.
    synthetic_forward_probe_skip: bool = False

    def __init__(
        self,
        pulse_sequence: str = "spgr",
        param_bounds: dict[str, tuple[float, float]] | None = None,
        n_inner_iterations: int = 10,
        gn_damping: float = 1e-4,
        flip_angles_deg: tuple[float, ...] = _DEFAULT_FLIP_ANGLES_DEG,
        tr_ms: float = _DEFAULT_TR_MS,
        te_ms: float = _DEFAULT_TE_MS,
        barrier_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        seq = str(pulse_sequence).lower()
        if seq not in ("spgr", "gre"):
            raise ValueError(
                f"BlochManifoldProjector: pulse_sequence={pulse_sequence!r} is not "
                "supported. The differentiable Bloch forward map only implements "
                "steady-state SPGR ('spgr'/'gre'). Add the sequence to "
                "DifferentiableBlochLayer before requesting it here (no silent "
                "fallback, CLAUDE.md #9)."
            )
        self.pulse_sequence = seq
        self.n_inner = int(n_inner_iterations)
        if self.n_inner < 1:
            raise ValueError("n_inner_iterations must be >= 1.")
        self.damping = float(gn_damping)
        self.barrier_weight = float(barrier_weight)
        self.tr_ms = float(tr_ms)
        self.te_ms = float(te_ms)

        bounds = param_bounds or {
            "M0": (0.0, 2.0),
            "T1": (100.0, 5000.0),
            "T2": (10.0, 2000.0),
        }
        for key in ("M0", "T1", "T2"):
            if key not in bounds:
                raise ValueError(f"param_bounds missing required key {key!r}.")
        self.bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}

        # Fixed VFA probe stored as a buffer so .to(device) moves it.
        flips = torch.tensor([float(a) for a in flip_angles_deg], dtype=torch.float32) * (
            torch.pi / 180.0
        )
        if flips.numel() < 1:
            raise ValueError("flip_angles_deg must contain at least one angle.")
        self.register_buffer("flip_angles_rad", flips, persistent=False)

        self._bloch = DifferentiableBlochLayer(sequence_type="SPGR")

    # ------------------------------------------------------------------ #
    # Forward physics: theta -> signal vector
    # ------------------------------------------------------------------ #
    def _bloch_signal_bank(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the SPGR magnitude at every probe flip angle.

        Args:
            m0/t1/t2: per-voxel parameter tensors, shape ``[N, 1]``.

        Returns:
            Signal bank ``[N, K]`` where ``K = len(flip_angles_rad)``.
        """
        # Reshape to a degenerate image [N, 1, 1, 1] so the Bloch layer's
        # softplus/clamp pipeline runs unchanged, then evaluate per angle.
        n = m0.shape[0]
        pd = m0.reshape(n, 1, 1, 1)
        t1m = t1.reshape(n, 1, 1, 1)
        t2m = t2.reshape(n, 1, 1, 1)
        cols = []
        for k in range(self.flip_angles_rad.numel()):
            alpha = self.flip_angles_rad[k]
            sig = self._bloch(
                t1_map=t1m,
                t2_map=t2m,
                pd_map=pd,
                tr=self.tr_ms,
                te=self.te_ms,
                flip_angle_rad=alpha,
                b0_map=None,
                b1_map=None,
            )  # real magnitude [N, 1, 1, 1]
            cols.append(sig.reshape(n, 1))
        return torch.cat(cols, dim=1)  # [N, K]

    def _jacobian(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Signal bank and its Jacobian w.r.t. (M0, T1, T2).

        Returns ``(s, J)`` with ``s`` of shape ``[N, K]`` and ``J`` of
        shape ``[N, K, 3]``. Computed with autograd through the Bloch
        layer so the physics stays the single source of truth.
        """
        theta = (
            torch.stack([m0.squeeze(-1), t1.squeeze(-1), t2.squeeze(-1)], dim=-1)
            .detach()
            .requires_grad_(True)
        )  # [N, 3]
        with torch.enable_grad():
            s = self._bloch_signal_bank(theta[:, 0:1], theta[:, 1:2], theta[:, 2:3])  # [N, K]
            k = s.shape[1]
            jac_cols = []
            for j in range(k):
                (g,) = torch.autograd.grad(
                    s[:, j].sum(),
                    theta,
                    retain_graph=(j < k - 1),
                    create_graph=False,
                )
                jac_cols.append(g)  # [N, 3]
            jac = torch.stack(jac_cols, dim=1)  # [N, K, 3]
        return s.detach(), jac.detach()

    # ------------------------------------------------------------------ #
    # Box / barrier helpers
    # ------------------------------------------------------------------ #
    def _clamp_to_box(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m0 = m0.clamp(*self.bounds["M0"])
        t1 = t1.clamp(*self.bounds["T1"])
        t2 = t2.clamp(*self.bounds["T2"])
        # Enforce T2 <= T1 by reflecting violators across the corner, then
        # re-clamp T2 to its own box (BlochRelaxationManifold convention).
        violator = t2 > t1
        t2 = torch.where(violator, 2.0 * t1 - t2, t2)
        t2 = t2.clamp(*self.bounds["T2"])
        t2 = torch.minimum(t2, t1 - _EPS)
        return m0, t1, t2

    def _init_params(self, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Initialise (M0, T1, T2) at the box midpoint, M0 from signal scale."""
        n = target.shape[0]
        device, dtype = target.device, target.dtype
        t1_0 = torch.full((n, 1), 0.5 * sum(self.bounds["T1"]), device=device, dtype=dtype)
        t2_0 = torch.full((n, 1), 0.5 * sum(self.bounds["T2"]), device=device, dtype=dtype)
        t2_0 = torch.minimum(t2_0, t1_0 - _EPS)
        # Rough M0 from the max observed signal (sin-weighted), kept in box.
        m0_0 = target.abs().amax(dim=1, keepdim=True).clamp_min(_EPS)
        m0_0 = m0_0.clamp(*self.bounds["M0"])
        return m0_0, t1_0, t2_0

    # ------------------------------------------------------------------ #
    # Core least-squares solve
    # ------------------------------------------------------------------ #
    @property
    def _param_scales(self) -> tuple[float, float, float]:
        """Characteristic scales ``(sM0, sT1, sT2)`` from the box widths.

        Gauss-Newton on ``(M0, T1, T2)`` directly is hopeless: the SPGR
        signal's sensitivity to T1/T2 (units of ms) is ~6 orders of
        magnitude smaller than to M0, so ``JᵀJ`` is catastrophically
        ill-conditioned. We solve in normalised coordinates
        ``u = (M0/sM0, T1/sT1, T2/sT2)``; the chain rule multiplies each
        Jacobian column by its scale, equalising the columns.
        """
        sm0 = max(self.bounds["M0"][1], 1.0)
        st1 = max(self.bounds["T1"][1], 1.0)
        st2 = max(self.bounds["T2"][1], 1.0)
        return float(sm0), float(st1), float(st2)

    def fit_parameters(
        self, target_bank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Gauss-Newton fit of ``(M0, T1, T2)`` to a signal bank.

        Solved in scale-normalised coordinates (see :attr:`_param_scales`)
        with Levenberg-Marquardt damping and a log-barrier on the gap
        ``T1 - T2`` that keeps iterates off the non-differentiable
        ``T1 == T2`` corner. A backtracking guard accepts a step only if
        it does not increase the data residual, guaranteeing the manifold
        residual is **non-increasing across inner iterations**.

        Args:
            target_bank: ``[N, K]`` observed (magnitude) signal vectors.

        Returns:
            ``(m0, t1, t2)`` each ``[N, 1]`` — the projected parameters.
        """
        sm0, st1, st2 = self._param_scales
        scales = torch.tensor(
            [sm0, st1, st2], device=target_bank.device, dtype=target_bank.dtype
        )  # [3]
        m0, t1, t2 = self._init_params(target_bank)
        m0, t1, t2 = self._clamp_to_box(m0, t1, t2)
        eye = torch.eye(3, device=target_bank.device, dtype=target_bank.dtype)

        def _data_res(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            s = self._bloch_signal_bank(a, b, c)
            return ((s - target_bank) ** 2).sum(dim=1)  # [N]

        prev_res = _data_res(m0, t1, t2)  # [N]
        for _ in range(self.n_inner):
            s, jac = self._jacobian(m0, t1, t2)  # [N,K], [N,K,3]
            # Rescale Jacobian columns to normalised coordinates: J_u = J * diag(scales).
            jac_u = jac * scales.view(1, 1, 3)
            residual = (s - target_bank).unsqueeze(-1)  # [N, K, 1]
            jt = jac_u.transpose(1, 2)  # [N, 3, K]
            jtj = jt @ jac_u  # [N, 3, 3]
            jtr = (jt @ residual).squeeze(-1)  # [N, 3]

            # Log-barrier on g = T1 - T2 (in physical units), mapped to u-space.
            gap = (t1 - t2).clamp_min(_EPS).squeeze(-1)  # [N]
            mu = self.barrier_weight
            barrier_grad = torch.zeros_like(jtr)
            barrier_grad[:, 1] = (-mu / gap) * st1  # d/dT1 * sT1
            barrier_grad[:, 2] = (mu / gap) * st2  # d/dT2 * sT2
            hb1 = mu / (gap * gap) * (st1 * st1)
            hb2 = mu / (gap * gap) * (st2 * st2)
            hbx = mu / (gap * gap) * (st1 * st2)

            # LM damping relative to the JᵀJ diagonal magnitude for scale
            # invariance (Marquardt's recommendation).
            diag = torch.diagonal(jtj, dim1=1, dim2=2)  # [N, 3]
            lm = self.damping * diag.clamp_min(_EPS)  # [N, 3]
            h = jtj + lm.unsqueeze(-1) * eye  # [N, 3, 3]
            h[:, 1, 1] = h[:, 1, 1] + hb1
            h[:, 2, 2] = h[:, 2, 2] + hb2
            h[:, 1, 2] = h[:, 1, 2] - hbx
            h[:, 2, 1] = h[:, 2, 1] - hbx

            rhs = -(jtr + barrier_grad).unsqueeze(-1)  # [N, 3, 1]
            try:
                delta_u = torch.linalg.solve(h, rhs).squeeze(-1)  # [N, 3]
            except RuntimeError:
                delta_u = torch.linalg.lstsq(h, rhs).solution.squeeze(-1)
            delta_u = torch.nan_to_num(delta_u, nan=0.0, posinf=0.0, neginf=0.0)
            delta = delta_u * scales.view(1, 3)  # back to physical units

            # Backtracking line search per voxel: halve the step until the
            # residual does not increase (guarantees monotone descent).
            step = torch.ones(
                target_bank.shape[0], 1, device=target_bank.device, dtype=target_bank.dtype
            )
            accepted_res = prev_res.clone()
            new_m0, new_t1, new_t2 = m0, t1, t2
            for _ls in range(6):
                cand_m0 = m0 + step * delta[:, 0:1]
                cand_t1 = t1 + step * delta[:, 1:2]
                cand_t2 = t2 + step * delta[:, 2:3]
                cand_m0, cand_t1, cand_t2 = self._clamp_to_box(cand_m0, cand_t1, cand_t2)
                cand_res = _data_res(cand_m0, cand_t1, cand_t2)  # [N]
                improved = cand_res <= accepted_res + 1e-12
                sel = improved.view(-1, 1)
                new_m0 = torch.where(sel, cand_m0, new_m0)
                new_t1 = torch.where(sel, cand_t1, new_t1)
                new_t2 = torch.where(sel, cand_t2, new_t2)
                accepted_res = torch.where(improved, cand_res, accepted_res)
                if bool(improved.all()):
                    break
                step = torch.where(improved.view(-1, 1), step, step * 0.5)
            m0, t1, t2 = new_m0, new_t1, new_t2
            prev_res = accepted_res

        return m0, t1, t2

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Per-voxel L2 distance ``||x - P_M(x)||`` (magnitude domain).

        Measured in the *signal-bank* space the fit operates on, so an
        on-manifold input (a genuine SPGR VFA bank) returns ~0. Returns a
        tensor of shape ``[B, 1, H, W]`` (depth is squeezed if singleton).
        """
        xm = x.abs() if torch.is_complex(x) else x
        if xm.ndim == 5 and xm.shape[-1] == 1:
            xm = xm.squeeze(-1)
        bank, b, h, w = self._to_bank(xm.to(torch.float32))
        m0, t1, t2 = self.fit_parameters(bank)
        proj_bank = self._bloch_signal_bank(m0, t1, t2)  # [N, K]
        diff = ((bank - proj_bank) ** 2).sum(dim=1)  # [N]
        res = torch.sqrt(diff.clamp_min(0.0)).reshape(b, 1, h, w)
        return res

    def _to_bank(self, mag_f: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Reshape a magnitude image ``[B,C,H,W]`` to a ``[N,K]`` signal bank.

        Treats the channel axis as the VFA / coil probe axis, tiling or
        truncating to the probe size ``K`` so the per-voxel fit is always
        well-posed. Returns ``(bank, B, H, W)``.
        """
        b, c, h, w = mag_f.shape[0], mag_f.shape[1], mag_f.shape[2], mag_f.shape[3]
        k = self.flip_angles_rad.numel()
        per_pixel = mag_f.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]
        if c == k:
            bank = per_pixel
        elif c < k:
            reps = (k + c - 1) // c
            bank = per_pixel.repeat(1, reps)[:, :k]
        else:
            bank = per_pixel[:, :k]
        return bank.contiguous(), b, h, w

    # ------------------------------------------------------------------ #
    # nn.Module API
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Project ``x`` onto :math:`\mathcal{M}_{\text{Bloch}}`.

        The channel axis is treated as the signal-bank (VFA / coil) axis.
        When the channel count equals the probe size ``K`` the projected
        bank is returned channel-for-channel (so an on-manifold input maps
        to itself). When ``C != K`` the projected per-voxel signal is
        broadcast back to ``C`` channels (shape-preserving for any ``C``).
        Complex inputs keep their original per-channel phase.
        """
        if x.ndim not in (4, 5):
            raise ValueError(
                f"BlochManifoldProjector expects [B,C,H,W] or [B,C,H,W,D]; got {tuple(x.shape)}"
            )
        squeezed_depth = False
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            squeezed_depth = True

        is_complex = torch.is_complex(x)
        mag = x.abs() if is_complex else x
        mag_f = mag.to(torch.float32)
        c = mag_f.shape[1]
        k = self.flip_angles_rad.numel()

        bank, b, h, w = self._to_bank(mag_f)
        m0, t1, t2 = self.fit_parameters(bank)  # each [N, 1]
        proj_bank = self._bloch_signal_bank(m0, t1, t2)  # [N, K]

        if c == k:
            # Channel-for-channel projection: [N, K] -> [B, K, H, W].
            proj_img = proj_bank.reshape(b, h, w, k).permute(0, 3, 1, 2)
        else:
            # Collapse to per-voxel scalar magnitude, broadcast to C.
            proj_scalar = proj_bank.mean(dim=1, keepdim=True)  # [N, 1]
            proj_img = proj_scalar.reshape(b, h, w, 1).permute(0, 3, 1, 2)
            proj_img = proj_img.expand(b, c, h, w)
        proj_img = proj_img.to(mag.dtype)

        if is_complex:
            phase = torch.angle(x)
            out = (proj_img * torch.exp(1j * phase)).to(x.dtype)
        else:
            out = proj_img

        if squeezed_depth:
            out = out.unsqueeze(-1)
        return out

    def extra_repr(self) -> str:
        return (
            f"seq={self.pulse_sequence}, n_inner={self.n_inner}, "
            f"damping={self.damping}, K={self.flip_angles_rad.numel()}, "
            f"barrier={self.barrier_weight}"
        )


__all__ = ["BlochManifoldProjector"]
