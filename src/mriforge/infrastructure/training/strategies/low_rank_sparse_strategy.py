"""Low-Rank + Sparse Decomposition for Dynamic MRI (PR-12 / M4).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-12.

Robust-PCA splitting

    X = L + S    subject to    rank(L) ≤ r,    ‖S‖₁ small

solved with one inner ADMM step per training iteration:

- **Singular-value soft-thresholding (SVT)** on the temporal stack of
  predictions ⇒ low-rank component ``L``.
- **Element-wise soft-thresholding** on the residual ⇒ sparse
  component ``S``.

Both ``L`` and ``S`` are differentiable w.r.t. the network output (we
keep the SVT projection in the autograd graph through the closed-form
SVD-times-thresholded-singular-values reassembly).

Loss = parent recon fidelity
     + ``λ_n · ‖L‖_*``    (nuclear surrogate via the soft-thresholded SVs)
     + ``λ_s · ‖S‖_1``    (sparse residual)
     + ``λ_c · ‖X − (L+S)‖_F²``  (consistency)
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.models.capabilities import StrategyCapabilities

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy


class LowRankSparseStrategy(ReconstructionTrainingStrategy):
    """Differentiable RPCA-style L+S decomposition."""

    #: Dynamic MRI: robust-PCA X = L + S (differentiable SVT on the temporal stack
    #: + sparse residual) is the canonical dynamic-MRI decomposition; the temporal
    #: stack is Axis.TEMPORAL, the profile's required axis.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.DYNAMIC}),
        tasks=frozenset({Task.RECONSTRUCTION}),
    )

    # Documented defaults (used only when ``training.low_rank_sparse`` is
    # absent from YAML). The authoritative values are read from the frozen
    # config in ``__init__`` and assigned to the instance — never mutate the
    # frozen TrainingSettings. See LowRankSparseTrainingConfigSchema.
    tau_low_rank: float = 0.05  # SVT threshold
    tau_sparse: float = 0.05  # soft-threshold on residual
    lambda_nuclear: float = 1e-3
    lambda_sparse: float = 1e-2
    lambda_consistency: float = 1e-3

    #: Knob names read from ``config.training.low_rank_sparse`` and stamped
    #: into provenance. Stable order keeps the logged knob-table reproducible.
    _RPCA_KNOBS: tuple[str, ...] = (
        "tau_low_rank",
        "tau_sparse",
        "lambda_nuclear",
        "lambda_sparse",
        "lambda_consistency",
    )

    def __init__(self, env: Any = None, device: Any = None, **kwargs: Any) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # Read + validate + stamp the RPCA knobs (pitfall #15). The schema
        # (LowRankSparseTrainingConfigSchema) already enforces ``ge=0`` bounds
        # at load time, but we defend against a raw/duck-typed config block and
        # raise on a negative weight rather than degrade to the default
        # (pitfall #9 — no silent fallback).
        block = getattr(self.config.training, "low_rank_sparse", None)
        resolved: dict[str, float] = {}
        for name in self._RPCA_KNOBS:
            value = getattr(block, name, None) if block is not None else None
            if value is None:
                value = float(getattr(type(self), name))  # documented default
            else:
                value = float(value)
            if value < 0.0:
                raise ValueError(
                    f"LowRankSparseStrategy.{name} must be >= 0; got {value}. "
                    "Set a non-negative value in training.low_rank_sparse."
                )
            setattr(self, name, value)
            resolved[name] = value

        # Stamp the resolved knobs into provenance so an audit can confirm
        # which weights actually ran (the instance attrs above are the SSOT
        # the loss code reads; ``_resolved_rpca_knobs`` is the auditable record).
        self._resolved_rpca_knobs = resolved
        logging_service = getattr(self, "logging_service", None)
        if logging_service is not None:
            logging_service.log_info(
                "[LowRankSparseStrategy] resolved RPCA knobs: "
                + ", ".join(f"{k}={v:g}" for k, v in resolved.items())
            )

    def _svt(self, X: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Singular-value soft-threshold.  Returns (L, sum_svs_thresholded)."""
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            return X, X.new_tensor(0.0)
        S_thresh = torch.relu(S - tau)
        L = U @ torch.diag_embed(S_thresh) @ Vh
        return L, S_thresh.sum()

    @staticmethod
    def _soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
        return x.sign() * torch.relu(x.abs() - tau)

    def _decompose(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """frames: (B, T, *) → returns dict with L, S, residual norms."""
        if frames.dim() < 4:
            return {
                "L": frames,
                "S": torch.zeros_like(frames),
                "nuclear": frames.new_tensor(0.0),
                "sparse": frames.new_tensor(0.0),
                "consistency": frames.new_tensor(0.0),
            }
        B = frames.shape[0]
        T = frames.shape[1]
        X = frames.flatten(start_dim=2)  # (B, T, D)
        L, sv_sum = self._svt(X, self.tau_low_rank)
        residual = X - L
        S = self._soft_threshold(residual, self.tau_sparse)
        consistency = (X - L - S).pow(2).mean()
        return {
            "L": L.view_as(frames),
            "S": S.view_as(frames),
            "nuclear": sv_sum / (B * T),
            "sparse": S.abs().mean(),
            "consistency": consistency,
        }

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

        # RPCA decomposition runs on the *generator output* — the temporal
        # stack X in X = L + S. The previous implementation keyed off
        # ``batch.get("prediction")``, a key the parent never writes, so the
        # decomposition (the entire point of this paradigm) silently never
        # ran. Re-run the generator on the resolved input to obtain X.
        gen = getattr(self.env, "generator", None)
        if gen is None:
            # The L+S split is the paradigm contract; a missing generator is a
            # misconfiguration, not an optional knob — fail loudly (no silent
            # no-op, pitfall #9).
            raise RuntimeError(
                "LowRankSparseStrategy requires a generator (self.env.generator) "
                "to produce the prediction stack for RPCA decomposition; none found."
            )

        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(batch, dict):
            gen_input = pick_present(batch.get("input"), batch.get("image"), batch.get("kspace"))
        else:
            gen_input = batch if batch is not None else input_batch
        if gen_input is None:
            return base

        pred = gen(gen_input)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred is None or pred.dim() < 4:
            return base

        comps = self._decompose(pred)
        nuclear_l = self.lambda_nuclear * comps["nuclear"]
        sparse_l = self.lambda_sparse * comps["sparse"]
        consist_l = self.lambda_consistency * comps["consistency"]

        base["loss_nuclear"] = nuclear_l
        base["loss_sparse"] = sparse_l
        base["loss_ls_consistency"] = consist_l

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + nuclear_l + sparse_l + consist_l
                break
        return base
