r"""Sparsifying-frame coherence-optimisation training (A-6.5 *SCO-Frame*).

Trains a :class:`~spectramr.models.generators.tight_frame_learner.TightFrameLearner`
(a learnable convolutional sparsifying frame, sparse-coding autoencoder) with a
reconstruction loss PLUS two structural penalties on the learned atoms:

* a **Parseval-tightness** term (always-on, ``lambda_parseval``) so the
  analyse/synthesise pair stays a tight frame (energy-preserving round-trip);
* a **mutual-coherence** term (the A-6.5 **one-knob**, ``lambda_coherence``) that
  drives the atom Gram off-diagonal toward 0, making the atoms incoherent — the
  property that makes sparse recovery well-posed.

The one-knob ablation (``training.sparse_frame.lambda_coherence`` 0 vs >0) isolates
coherence optimisation from tightness and reconstruction: the oracle is that the
treatment's learned mutual coherence (``val_mutual_coherence``) is lower than the
control's at matched reconstruction. Knobs are read from config (#15); the
resolved values + initial coherence are stamped into provenance.

The coherence term is the registered ``frame_coherence`` loss (a parameter
penalty, wired here via ``frame=`` so it is never inert, #16); tightness is the
model's own ``parseval_penalty()``. The penalty is added to the canonical
grad-carrying ``g_total_loss`` key (the key the loop reads for backward).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.losses.registry import create_loss

from .reconstruction import ReconstructionTrainingStrategy


class SparseFrameStrategy(ReconstructionTrainingStrategy):
    """Train a learnable sparsifying frame with coherence optimisation (A-6.5)."""

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        cfg = getattr(self.config.training, "sparse_frame", None)
        self._lambda_coherence = float(getattr(cfg, "lambda_coherence", 0.0)) if cfg else 0.0
        self._lambda_parseval = float(getattr(cfg, "lambda_parseval", 0.0)) if cfg else 0.0
        self._coherence_loss = create_loss("frame_coherence")

        if getattr(self, "logging_service", None):
            gen = getattr(self.env, "generator", None)
            mu = gen.mutual_coherence() if hasattr(gen, "mutual_coherence") else float("nan")
            self.logging_service.log_info(
                f"[A-6.5 SCO-Frame] lambda_coherence={self._lambda_coherence}, "
                f"lambda_parseval={self._lambda_parseval}; initial mutual_coherence={mu:.4g}"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        losses = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        gen = getattr(self.env, "generator", None)
        total = losses.get("g_total_loss") if isinstance(losses, dict) else None
        if gen is None or not (isinstance(total, torch.Tensor) and total.requires_grad):
            return losses

        extra: torch.Tensor | None = None
        lam_c = getattr(self, "_lambda_coherence", 0.0)
        lam_p = getattr(self, "_lambda_parseval", 0.0)
        if lam_c > 0.0 and hasattr(gen, "coherence_penalty"):
            coh = self._coherence_loss(frame=gen)
            losses["frame_coherence"] = coh.detach()
            extra = lam_c * coh if extra is None else extra + lam_c * coh
        if lam_p > 0.0 and hasattr(gen, "parseval_penalty"):
            par = gen.parseval_penalty()
            losses["frame_parseval"] = par.detach()
            extra = lam_p * par if extra is None else extra + lam_p * par
        if extra is not None:
            losses["g_total_loss"] = total + extra
        return losses

    def validation_step(self, input_batch: Any, target_batch: Any, **kwargs: Any) -> Any:
        """Emit the mutual-coherence witness alongside the usual metrics."""
        metrics = super().validation_step(input_batch, target_batch, **kwargs)
        if isinstance(metrics, dict):
            gen = getattr(self.env, "generator", None)
            if hasattr(gen, "mutual_coherence"):
                metrics["val_mutual_coherence"] = float(gen.mutual_coherence())
        return metrics
