"""q-Space Diffusion MRI Strategy (PR-17 / L4).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-17.

Diffusion-weighted MRI reconstruction with a real spherical-harmonic
angular regulariser instead of the previous angular-finite-difference
proxy.

Wired components:

- :func:`build_sh_basis_unit_sphere` — assembles a real-valued SH
  basis matrix ``Y ∈ R^{N×K}`` for the ``N`` diffusion directions in
  ``batch["b_vectors"]`` (the key ``LoadDWIMetadata`` writes; #350).
- :meth:`fit_sh_coefs` — least-squares projection of the per-voxel
  signal onto the SH basis.
- Loss = parent recon + λ_smooth · ‖Δ_SO3 Y c‖²
       + λ_b0 · ‖S(b=0) − M0‖₁     (b=0 anchor)
       + λ_sparse · ‖high-order SH coefs‖₁  (angular sparsity)
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.models.capabilities import StrategyCapabilities

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


def build_sh_basis_unit_sphere(
    bvecs: torch.Tensor, max_order: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real spherical-harmonic basis evaluated at unit-vector directions.

    bvecs: (N, 3).  Returns (Y (N, K), order_per_col (K,)).
    Uses Descoteaux 2007 real-SH basis.
    """
    bvecs = bvecs / bvecs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    theta = torch.acos(z.clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)

    cols: list[torch.Tensor] = []
    orders: list[int] = []
    for L in range(0, max_order + 1, 2):
        for m in range(-L, L + 1):
            P_lm = _legendre_assoc(L, abs(m), torch.cos(theta))
            norm = math.sqrt(
                (2 * L + 1)
                / (4 * math.pi)
                * math.factorial(L - abs(m))
                / math.factorial(L + abs(m))
            )
            if m < 0:
                col = math.sqrt(2.0) * norm * P_lm * torch.sin(abs(m) * phi)
            elif m == 0:
                col = norm * P_lm
            else:
                col = math.sqrt(2.0) * norm * P_lm * torch.cos(m * phi)
            cols.append(col)
            orders.append(L)
    Y = torch.stack(cols, dim=1)
    return Y, torch.tensor(orders, device=bvecs.device, dtype=torch.float32)


def _legendre_assoc(L: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Associated Legendre polynomial P_L^m(x), m ≥ 0.  Recurrence-stable."""
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if m == L:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if m + 1 == L:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, L + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _shared_directions(b_vectors: Any) -> torch.Tensor:
    """Reduce collated gradient directions to the shared ``(N, 3)`` SH basis needs.

    ``LoadDWIMetadata`` attaches ``(N, 3)`` per subject; the default collate stacks
    those to ``(B, N, 3)``. All samples in a batch must share the SAME directions to
    share one spherical-harmonic basis, so a heterogeneous batch (mixed protocols)
    RAISES rather than silently picking or averaging one sample's directions.
    Homogeneous batching (one acquisition protocol) is the realistic single-dataset
    case; ragged per-sample direction sets are a documented follow-up (not silently
    reduced here).
    """
    t = torch.as_tensor(b_vectors, dtype=torch.float32)
    if t.dim() == 2:  # (N, 3) — already a single direction set
        dirs = t
    elif t.dim() == 3:  # (B, N, 3) — collated; require homogeneity across the batch
        if not torch.allclose(t, t[:1].expand_as(t), atol=1e-5):
            raise ValueError(
                "QSpaceDiffusionStrategy: heterogeneous 'b_vectors' across the batch "
                "— samples have different diffusion directions and cannot share one "
                "SH basis. Batch by acquisition protocol (per-sample bases are a "
                "documented follow-up)."
            )
        dirs = t[0]
    else:
        raise ValueError(f"'b_vectors' must be (N, 3) or (B, N, 3); got shape {tuple(t.shape)}.")
    if dirs.shape[-1] != 3:
        raise ValueError(f"'b_vectors' last dim must be 3 (x, y, z); got {tuple(dirs.shape)}.")
    return dirs


class QSpaceDiffusionStrategy(ReconstructionTrainingStrategy):
    """SH-regularised diffusion MRI reconstruction."""

    #: Diffusion-weighted MRI: a real Descoteaux-2007 spherical-harmonic basis is
    #: built from batch['b_vectors'] (the diffusion-encoding directions --
    #: Axis.DIFFUSION_ENCODING, the profile's required axis) and regularised by a
    #: Laplace-Beltrami angular penalty. Reads the exact key LoadDWIMetadata writes
    #: and raises if it is absent (#350 — was 'bvecs', which nothing produced).
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.DIFFUSION_WEIGHTED}),
        tasks=frozenset({Task.RECONSTRUCTION}),
    )

    sh_max_order: int = 4
    lambda_smooth: float = 1e-3
    lambda_b0: float = 0.1
    lambda_sparse_high: float = 1e-4

    def _setup_strategy_specific_components(self) -> None:
        """Resolve SH-regularisation knobs from the typed
        ``training.qspace_diffusion`` block (pitfall #15: read + validate +
        stamp); fall back to the class-attribute defaults when absent."""
        super()._setup_strategy_specific_components()
        cfg = getattr(self.config.training, "qspace_diffusion", None)
        if cfg is not None:
            self.sh_max_order = int(cfg.sh_max_order)
            self.lambda_smooth = float(cfg.lambda_smooth)
            self.lambda_b0 = float(cfg.lambda_b0)
            self.lambda_sparse_high = float(cfg.lambda_sparse_high)
        logger.info(
            "QSpaceDiffusionStrategy knobs (source=%s): sh_max_order=%d, "
            "lambda_smooth=%.4g, lambda_b0=%.4g, lambda_sparse_high=%.4g",
            "config" if cfg is not None else "defaults",
            self.sh_max_order,
            self.lambda_smooth,
            self.lambda_b0,
            self.lambda_sparse_high,
        )

    def fit_sh_coefs(self, signal: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Least-squares projection of (B, N_dir, H, W) → (B, K, H, W)."""
        B, N, H, W = signal.shape
        flat = signal.permute(0, 2, 3, 1).reshape(-1, N)  # (B*H*W, N)
        coefs, *_ = torch.linalg.lstsq(
            Y.unsqueeze(0).expand(flat.shape[0], -1, -1), flat.unsqueeze(-1)
        )
        coefs = coefs.squeeze(-1)
        return coefs.view(B, H, W, -1).permute(0, 3, 1, 2)

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base
        signal = pick_present(batch.get("dwi_signal"), batch.get("prediction"))
        if signal is None or signal.dim() != 4:
            # No (B, N_dir, H, W) diffusion signal to regularise in this batch —
            # the q-space term is N/A here. This is the term not applying, not a
            # silent fallback of the mechanism (contrast the b_vectors branch).
            return base
        # Canonical directions key is 'b_vectors' — exactly what LoadDWIMetadata
        # writes. Pre-#350 this read 'bvecs', which nothing produced, so get()
        # always returned None and the strategy silently degraded to vanilla
        # reconstruction (pitfall #16). A batch that carries a diffusion signal but
        # no directions is a broken DWI pipeline: raise, don't quietly drop the
        # regulariser (pitfall #9).
        b_vectors = batch.get("b_vectors")
        if b_vectors is None:
            raise ValueError(
                "QSpaceDiffusionStrategy: the batch has a (B, N_dir, H, W) diffusion "
                "signal but no 'b_vectors' (gradient directions), so the SH / "
                "Laplace-Beltrami regulariser cannot run. Ensure LoadDWIMetadata is "
                "in the transform pipeline (it attaches 'b_vectors')."
            )
        bvecs = _shared_directions(b_vectors)

        # No silent fallback (pitfall #3/#9): a malformed bvecs shape, an
        # lstsq non-convergence, or a device mismatch must fail loudly so the
        # arm does not quietly train as vanilla reconstruction without the
        # q-space regulariser. Co-locate the SH basis with ``signal`` before
        # the lstsq so the (otherwise hidden) device mismatch cannot occur.
        Y, orders = build_sh_basis_unit_sphere(bvecs, max_order=self.sh_max_order)
        Y = Y.to(signal.device)
        orders = orders.to(signal.device)
        coefs = self.fit_sh_coefs(signal, Y)

        # Laplace-Beltrami penalty: ∑ L(L+1) c_L²
        K = coefs.shape[1]
        if orders.shape[0] >= K:
            w = (orders[:K] * (orders[:K] + 1.0)).to(coefs.device).view(1, K, 1, 1)
        else:
            w = torch.ones((1, K, 1, 1), device=coefs.device)
        smooth_l = self.lambda_smooth * (w * coefs.pow(2)).mean()
        base["loss_sh_laplace_beltrami"] = smooth_l

        # B=0 anchor (first SH coefficient ↔ isotropic component).
        # ``base`` is the parent's loss ``dict`` and never has ``new_zeros``;
        # the SH coefficients live on ``signal.device`` after the move above.
        b0_anchor = torch.zeros((), device=signal.device)
        if "M0_map" in batch and K >= 1:
            b0_anchor = self.lambda_b0 * (coefs[:, 0:1] - batch["M0_map"]).abs().mean()
            base["loss_b0_anchor"] = b0_anchor

        # Angular sparsity on high-order coefficients.
        if K > 6:
            high = coefs[:, 6:].abs().mean()
            sparse_high = self.lambda_sparse_high * high
            base["loss_sh_sparse_high"] = sparse_high
        else:
            sparse_high = torch.zeros((), device=signal.device)

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + smooth_l + b0_anchor + sparse_high
                break
        return base
