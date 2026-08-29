"""Cross-contrast / cross-patient no-reference (NR) MRI quality metrics.

This module implements the ``nr_cross`` group of the NR metric battery
(spec §3 ``nr_cross``). All four metrics answer a *cross* question — "is the
reconstruction self-consistent with its sibling contrasts, with the
acquisition physics that links them, and with the population it came from?".
None of them needs a reference ``target``; they read the measurement /
sibling / prior bundle from a :class:`~mriforge.core.metrics.context.MetricContext`.

Metrics
-------
* :class:`CrossContrastStructuralAgreement` (``ccsa``, ↓) — penalises
  structural disagreement between the reconstructed contrast and its siblings,
  combining a MIND-descriptor L1 term with a normalised-mutual-information
  reward.
* :class:`BlochReSynthesisConsistency` (``brc``, ↓) — re-synthesises every
  measured contrast from the predicted quantitative maps through the
  differentiable Bloch forward model and reports the relative residual; it is
  ``0`` exactly when the predicted maps equal the maps used to acquire the
  measurements.
* :class:`NormativeCrossPatientDeviation` (``ncpd``, ↓, **gated**) — per-region
  Mahalanobis distance of learned embeddings to a fitted normative
  :math:`(\\mu_R, \\Sigma_R)`. Needs a fitted encoder + normative asset; returns
  ``nan`` when neither is supplied.
* :class:`ConditionalCrossContrastPredictiveResidual` (``ccpr``, ↓, **gated**) —
  relative residual of a trained sibling-to-contrast predictor :math:`g_\\psi`.
  Returns ``nan`` when no predictor is supplied.

All return a single Python ``float`` reduced over the batch. On missing /
degenerate context they return ``float('nan')`` (never raise ``ValueError`` —
``core/metrics/computer.py`` re-raises that and aborts the whole validation).
"""

from __future__ import annotations

from typing import Any

import torch

from mriforge.core.metrics.context import MetricContext, resolve_context
from mriforge.core.metrics.hallucination_metrics import _to_magnitude
from mriforge.core.metrics.registry import register_metric

# --------------------------------------------------------------------------- #
# shared layout helpers                                                        #
# --------------------------------------------------------------------------- #


def _as_bchw_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Coerce an arbitrary metric input to a real magnitude image ``[B, 1, H, W]``.

    Accepts ``[B, C, H, W]`` (C even ⇒ interleaved real/imag), ``[B, 1, H, W]``,
    ``[H, W]``, ``[C, H, W]`` or complex tensors. Uses the physics-side
    :func:`_to_magnitude` (handles complex / 2-channel) and normalises rank.
    """
    if x.dim() == 2:  # [H, W]
        x = x[None, None]
    elif x.dim() == 3:  # [C, H, W]
        x = x[None]
    mag = _to_magnitude(x)
    if mag.dim() == 4 and mag.shape[1] != 1:
        # multi-channel real (e.g. 4 coils): collapse to a single magnitude map
        mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True) + 1e-12)
    return mag.float()


def _normalised_mutual_information(
    a: torch.Tensor, b: torch.Tensor, *, bins: int = 64, eps: float = 1e-12
) -> float:
    r"""Symmetric normalised mutual information :math:`\mathrm{NMI}=2 I(a;b)/(H(a)+H(b))`.

    Computed from a joint 2-D histogram so the marginal entropies come from the
    *same* binning as the joint — keeping ``NMI`` in ``[0, 1]`` (1 for an
    invertible relationship, 0 for independence). Inputs are min-max normalised
    to ``[0, 1]`` for stable binning (matches ``MutualInformationMetric``).
    """
    av = a.reshape(-1).float()
    bv = b.reshape(-1).float()
    av = (av - av.min()) / (av.max() - av.min() + eps)
    bv = (bv - bv.min()) / (bv.max() - bv.min() + eps)
    # 2-D histogram via integer bin indices + bincount (torch, deterministic).
    ia = torch.clamp((av * bins).long(), 0, bins - 1)
    ib = torch.clamp((bv * bins).long(), 0, bins - 1)
    flat = ia * bins + ib
    joint = torch.bincount(flat, minlength=bins * bins).float().reshape(bins, bins)
    total = joint.sum()
    if total <= 0:
        return 0.0
    p = joint / total
    px = p.sum(dim=1)
    py = p.sum(dim=0)

    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        nz = prob > 0
        return -(prob[nz] * torch.log(prob[nz])).sum()

    h_a = _entropy(px)
    h_b = _entropy(py)
    nz = p > 0
    ratio = p[nz] / (px.unsqueeze(1) * py.unsqueeze(0))[nz].clamp(min=eps)
    mi = (p[nz] * torch.log(ratio.clamp(min=eps))).sum()
    denom = h_a + h_b
    if float(denom) <= eps:
        # both images constant ⇒ perfect (trivial) agreement
        return 1.0
    return float((2.0 * mi / denom).clamp(0.0, 1.0))


# --------------------------------------------------------------------------- #
# ccsa — Cross-Contrast Structural Agreement                                   #
# --------------------------------------------------------------------------- #


@register_metric(
    "ccsa",
    aliases=["cross_contrast_structural_agreement"],
    requires_reference=False,
    needs=("sibling_contrasts",),
)
class CrossContrastStructuralAgreement:
    r"""Cross-Contrast Structural Agreement (``ccsa``, lower is better).

    .. math::

        \mathrm{CCSA} = \frac{1}{|\mathcal{C}|}\sum_{c'\neq c}\Big[
            \lambda\,\lVert\mathrm{MIND}(\hat{\mathbf{x}}_c)
                       -\mathrm{MIND}(\mathbf{x}_{c'})\rVert_1
            - (1-\lambda)\,\mathrm{NMI}(\hat{\mathbf{x}}_c, \mathbf{x}_{c'})
        \Big]

    The MIND descriptor (modality-independent neighbourhood descriptor)
    captures *local self-similarity structure* that should be shared across
    co-registered contrasts even though their intensities differ; the NMI term
    rewards a consistent (possibly nonlinear) intensity relationship. CCSA
    therefore *rises* when a structure present in a sibling contrast is absent
    or distorted in the reconstruction.

    :math:`\lambda` is read from ``ctx.nr_params['ccsa_lambda']`` (default 0.5).
    Returns ``nan`` when no sibling contrasts are supplied.
    """

    def __init__(self, bins: int = 64) -> None:
        self.bins = int(bins)
        # MINDSSCLoss is a torch.nn.Module under models/ — import lazily so a
        # heavy/optional import never kills registration at module-discovery.
        from mriforge.models.losses.mind_ssc_loss import MINDSSCLoss

        self._mind = MINDSSCLoss()

    @property
    def name(self) -> str:
        return "ccsa"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("sibling_contrasts"):
            return float("nan")
        siblings = ctx.sibling_contrasts
        if not isinstance(siblings, dict) or len(siblings) == 0:
            return float("nan")

        lam = 0.5
        if ctx.nr_params is not None:
            lam = float(ctx.nr_params.get("ccsa_lambda", 0.5))

        pred = _as_bchw_magnitude(prediction)
        mind_pred = self._mind._compute_mind_descriptor(pred)

        terms: list[float] = []
        for sib in siblings.values():
            sib_t = _as_bchw_magnitude(torch.as_tensor(sib, dtype=pred.dtype, device=pred.device))
            if sib_t.shape[-2:] != pred.shape[-2:]:
                # shape-incompatible sibling — skip rather than crash
                continue
            mind_sib = self._mind._compute_mind_descriptor(sib_t)
            mind_l1 = (mind_pred - mind_sib).abs().mean()
            nmi = _normalised_mutual_information(pred, sib_t, bins=self.bins)
            terms.append(float(lam * mind_l1 - (1.0 - lam) * nmi))

        if not terms:
            return float("nan")
        return float(sum(terms) / len(terms))


# --------------------------------------------------------------------------- #
# brc — Bloch Re-Synthesis Consistency                                         #
# --------------------------------------------------------------------------- #


@register_metric(
    "brc",
    aliases=["bloch_resynthesis_consistency"],
    requires_reference=False,
    needs=("acq_params",),
)
class BlochReSynthesisConsistency:
    r"""Bloch Re-Synthesis Consistency (``brc``, lower is better; ``=0`` at the true maps).

    .. math::

        \mathrm{BRC} = \frac{1}{|\mathcal{C}|}\sum_c
            \frac{\lVert\mathcal{B}_c(\hat{\boldsymbol\theta};\boldsymbol\alpha_c)
                       -\mathbf{m}_c\rVert_2}
                 {\lVert\mathbf{m}_c\rVert_2}

    :math:`\mathcal{B}` is the differentiable SPGR Bloch forward model
    (:class:`~mriforge.infrastructure.physics.differentiable_bloch.DifferentiableBlochLayer`),
    :math:`\hat{\boldsymbol\theta}=(T_1, T_2, \mathrm{PD})` the predicted
    quantitative maps, :math:`\boldsymbol\alpha_c=(\mathrm{TR}, \mathrm{TE},
    \mathrm{FA})` the per-contrast acquisition parameters, and
    :math:`\mathbf{m}_c` the measured contrast. BRC :math:`\to 0` noiselessly at
    the correct maps; interpret it relative to a model-matched floor.

    **qmaps source** — the predicted maps :math:`\hat{\boldsymbol\theta}` are
    taken (in priority order):
      1. ``ctx.nr_params['brc_qmaps']`` — a dict/tensor of ``t1/t2/pd`` maps;
      2. the channels of ``prediction`` when it carries ≥3 channels
         (``[B, 3, H, W]`` ⇒ T1, T2, PD), with PD the magnitude otherwise.
    **measured contrasts** — from ``ctx.sibling_contrasts`` (a dict keyed by
    contrast name); **acq params** — from ``ctx.acq_params`` (a dict mapping the
    same contrast names to ``{'tr','te','fa'}`` in ms / radians). Returns
    ``nan`` when acq-params or measured contrasts are absent.
    """

    def __init__(self, sequence_type: str = "SPGR") -> None:
        from mriforge.infrastructure.physics.differentiable_bloch import (
            DifferentiableBlochLayer,
        )

        self._bloch = DifferentiableBlochLayer(sequence_type)

    @property
    def name(self) -> str:
        return "brc"

    @property
    def higher_is_better(self) -> bool:
        return False

    # --- qmap resolution ---------------------------------------------------
    def _resolve_qmaps(
        self, prediction: torch.Tensor, ctx: MetricContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return ``(t1, t2, pd)`` each ``[B, 1, H, W]`` or ``None`` if unavailable."""
        params = ctx.nr_params or {}
        qmaps = params.get("brc_qmaps")
        if isinstance(qmaps, dict) and {"t1", "t2", "pd"} <= set(qmaps):
            t1 = torch.as_tensor(qmaps["t1"]).float()
            t2 = torch.as_tensor(qmaps["t2"]).float()
            pd = torch.as_tensor(qmaps["pd"]).float()
            return tuple(self._to_bchw(m) for m in (t1, t2, pd))  # type: ignore[return-value]

        pred = prediction.float() if not prediction.is_complex() else prediction.abs().float()
        if pred.dim() == 2:
            pred = pred[None, None]
        elif pred.dim() == 3:
            pred = pred[None]
        if pred.dim() == 4 and pred.shape[1] >= 3:
            t1 = pred[:, 0:1]
            t2 = pred[:, 1:2]
            pd = pred[:, 2:3]
            return t1, t2, pd
        return None

    @staticmethod
    def _to_bchw(m: torch.Tensor) -> torch.Tensor:
        if m.dim() == 2:
            return m[None, None]
        if m.dim() == 3:
            return m[None]
        return m

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("acq_params"):
            return float("nan")
        acq = ctx.acq_params
        siblings = ctx.sibling_contrasts
        if not isinstance(acq, dict) or not acq:
            return float("nan")
        if not isinstance(siblings, dict) or not siblings:
            return float("nan")

        qmaps = self._resolve_qmaps(prediction, ctx)
        if qmaps is None:
            return float("nan")
        t1, t2, pd = qmaps

        residuals: list[float] = []
        for cname, alpha in acq.items():
            if cname not in siblings:
                continue
            try:
                tr = float(alpha["tr"])
                te = float(alpha["te"])
                fa = float(alpha["fa"])
            except (KeyError, TypeError, ValueError):
                continue
            sim = self._bloch.forward(t1, t2, pd, tr=tr, te=te, flip_angle_rad=fa)
            sim_mag = sim.abs() if sim.is_complex() else sim
            meas = _as_bchw_magnitude(
                torch.as_tensor(siblings[cname], dtype=sim_mag.dtype, device=sim_mag.device)
            )
            if meas.shape[-2:] != sim_mag.shape[-2:]:
                continue
            num = torch.linalg.vector_norm(sim_mag - meas)
            den = torch.linalg.vector_norm(meas)
            if float(den) <= 1e-12:
                continue
            residuals.append(float(num / den))

        if not residuals:
            return float("nan")
        return float(sum(residuals) / len(residuals))


# --------------------------------------------------------------------------- #
# ncpd — Normative Cross-Patient Deviation (gated)                             #
# --------------------------------------------------------------------------- #


@register_metric(
    "ncpd",
    aliases=["normative_cross_patient_deviation"],
    requires_reference=False,
    needs=("encoder",),
)
class NormativeCrossPatientDeviation:
    r"""Normative Cross-Patient Deviation (``ncpd``, lower is better; **gated**).

    .. math::

        \mathrm{NCPD} = \frac{1}{|\mathcal{R}|}\sum_R
            (\phi_R - \boldsymbol\mu_R)^{\top}
            \boldsymbol\Sigma_R^{-1}
            (\phi_R - \boldsymbol\mu_R)

    Per-region embeddings :math:`\phi_R` are produced by a fitted encoder
    (``ctx.encoder``) and scored by the Mahalanobis distance to a normative
    Gaussian :math:`(\boldsymbol\mu_R, \boldsymbol\Sigma_R)` estimated on a
    pristine corpus. A *focal* pathology lights up a sparse set of regions
    (few large terms); a *diffuse* degradation lights up many regions (dense
    support) — the separation rule documented in the spec.

    This metric is **asset-gated**: it requires both a fitted ``encoder`` and a
    normative asset :math:`(\boldsymbol\mu_R, \boldsymbol\Sigma_R)` supplied via
    ``ctx.nr_params['ncpd_normative']`` (a dict ``{'mu': [R, D], 'cov': [R, D, D]}``)
    or read from ``ctx.nr_params['ncpd_template']``. When either is absent it
    returns ``nan`` so the sim2rank sweep skips it cleanly rather than aborting.
    The encoder is expected to map an image ``[B, 1, H, W]`` → per-region
    embeddings ``[B, R, D]`` (or ``[R, D]``).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "ncpd"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        # Gate 1: no fitted encoder ⇒ asset-blocked, return nan.
        if not ctx.has("encoder"):
            return float("nan")
        encoder = ctx.encoder

        # Gate 2: normative (mu, Sigma) asset.
        params = ctx.nr_params or {}
        normative = params.get("ncpd_normative")
        if normative is None:
            # ncpd_template would be loaded by the eval harness into
            # 'ncpd_normative'; without it the metric is asset-blocked.
            return float("nan")
        try:
            mu = torch.as_tensor(normative["mu"]).float()  # [R, D]
            cov = torch.as_tensor(normative["cov"]).float()  # [R, D, D]
        except (KeyError, TypeError):
            return float("nan")

        img = _as_bchw_magnitude(prediction)
        try:
            phi = encoder(img)
        except Exception:
            return float("nan")
        phi = torch.as_tensor(phi).float()
        if phi.dim() == 3:  # [B, R, D] -> mean over batch
            phi = phi.mean(dim=0)
        if phi.dim() != 2:  # need [R, D]
            return float("nan")
        if phi.shape[0] != mu.shape[0] or phi.shape[1] != mu.shape[1]:
            return float("nan")

        # Per-region Mahalanobis: d_R = (phi-mu)^T Sigma^-1 (phi-mu).
        diff = (phi - mu).unsqueeze(-1)  # [R, D, 1]
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        cov_reg = cov + self.eps * eye
        try:
            inv = torch.linalg.inv(cov_reg)  # [R, D, D]
        except torch.linalg.LinAlgError:
            return float("nan")
        maha = (diff.transpose(-1, -2) @ inv @ diff).reshape(-1)  # [R]
        return float(maha.clamp(min=0.0).mean())


# --------------------------------------------------------------------------- #
# ccpr — Conditional Cross-Contrast Predictive Residual (gated)               #
# --------------------------------------------------------------------------- #


@register_metric(
    "ccpr",
    aliases=["conditional_cross_contrast_predictive_residual"],
    requires_reference=False,
    # Declares the trained predictor g_psi it truly needs (docstring below), not
    # just the siblings. prior_model is un-providable by the sim2rank engine, so
    # this makes the metric classify as not_applicable (expected) rather than
    # never_computed (a defect) on a sweep that cannot supply the predictor.
    needs=("sibling_contrasts", "prior_model"),
)
class ConditionalCrossContrastPredictiveResidual:
    r"""Conditional Cross-Contrast Predictive Residual (``ccpr``, lower is better; **gated**).

    .. math::

        \mathrm{CCPR} = \frac{\lVert g_\psi(\{\mathbf{x}_{c'}\}_{c'\neq c})
                                  -\hat{\mathbf{x}}_c\rVert_2}
                              {\lVert\hat{\mathbf{x}}_c\rVert_2}

    :math:`g_\psi` is a predictor of the held-out contrast from its siblings,
    trained on a pristine corpus. CCPR rises with cross-contrast inconsistency:
    if the reconstruction invents structure the siblings do not support, the
    predictor (which only ever saw self-consistent data) disagrees with it.

    This metric is **asset-gated**: it needs a trained predictor :math:`g_\psi`,
    supplied as a callable via ``ctx.nr_params['ccpr_predictor']`` (or
    ``ctx.prior_model`` as a fallback). Without it — and without sibling
    contrasts — it returns ``nan`` so the sweep skips it cleanly. The predictor
    is called as ``g(stacked_siblings)`` where ``stacked_siblings`` is
    ``[B, n_siblings, H, W]`` and must return an image broadcastable to the
    reconstructed contrast ``[B, 1, H, W]``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    @property
    def name(self) -> str:
        return "ccpr"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        siblings = ctx.sibling_contrasts
        if not isinstance(siblings, dict) or len(siblings) == 0:
            return float("nan")

        params = ctx.nr_params or {}
        predictor = params.get("ccpr_predictor")
        if predictor is None:
            predictor = ctx.prior_model
        if predictor is None or not callable(predictor):
            # asset-blocked: no trained g_psi.
            return float("nan")

        pred = _as_bchw_magnitude(prediction)
        sib_mags = []
        for sib in siblings.values():
            sib_t = _as_bchw_magnitude(torch.as_tensor(sib, dtype=pred.dtype, device=pred.device))
            if sib_t.shape[-2:] == pred.shape[-2:]:
                sib_mags.append(sib_t)
        if not sib_mags:
            return float("nan")
        stacked = torch.cat(sib_mags, dim=1)  # [B, n_siblings, H, W]

        try:
            predicted = predictor(stacked)
        except Exception:
            return float("nan")
        predicted = _as_bchw_magnitude(
            torch.as_tensor(predicted, dtype=pred.dtype, device=pred.device)
        )
        if predicted.shape[-2:] != pred.shape[-2:]:
            return float("nan")

        num = torch.linalg.vector_norm(predicted - pred)
        den = torch.linalg.vector_norm(pred)
        if float(den) <= self.eps:
            return float("nan")
        return float(num / den)


__all__ = [
    "BlochReSynthesisConsistency",
    "ConditionalCrossContrastPredictiveResidual",
    "CrossContrastStructuralAgreement",
    "NormativeCrossPatientDeviation",
]
