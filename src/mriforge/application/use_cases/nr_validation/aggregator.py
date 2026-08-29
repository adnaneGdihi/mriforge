"""§8.7 — the ``nr_quality_index`` meta-metric over the NR battery.

Fits a (ridge-regularised) linear model that maps the no-reference (NR) metric
battery onto the *simulator severity* :math:`\\theta` — never a human Likert
label (spec §8.7 is explicit: the supervision signal is the synthetic
degradation coordinate, which is known exactly, not a noisy subjective rating).

The fit reports calibration (ECE) and generalisation (held-out :math:`R^2`,
Spearman, AUROC, optional cross-degradation transfer), and the fitted predictor
is registered as a metric ``"nr_quality_index"`` so it can be referenced like
any other metric in the sim2rank sweep.

Design constraints (research-mode offline harness):

* **CPU-canonical & deterministic** — every split uses a seeded
  :class:`torch.Generator`; statistics run in ``float64``.
* **Never raises** — each statistical step is wrapped; on failure the metric is
  recorded as ``NaN`` (mirrors ``precompute_metric_values``' crash→NaN
  contract). On insufficient data the fit returns a model with ``NaN`` stats and
  a ``"reason"``.
* **scikit-learn is optional** — lazy-imported and gated inside the fit. When
  unavailable the fit falls back to a dependency-free closed-form ridge via the
  normal equations and records ``backend="ridge_closed_form"``.
* **Group split by ``content_id``** — the same subject never appears in both the
  train and test split, so the reported generalisation is honest.

The fitted model is *not* auto-registered at import (that would pollute the
global registry on every run); call :func:`register_nr_quality_index`
explicitly. Re-registration is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mriforge.application.use_cases.nr_validation.cards import spearman
from mriforge.core.metrics.meta_evaluation.types import MetricEvaluationDataset

NAN = float("nan")

# Default donor families for the cross-degradation transfer probe. Only the
# families actually present in the dataset are used; if fewer than two distinct
# families exist the probe is skipped (reported as ``None``).
_DEFAULT_TRAIN_FAMILIES: tuple[str, ...] = ("rician", "gaussian_noise", "motion")


@dataclass
class NRQualityIndexModel:
    """A fitted linear severity predictor over the NR battery.

    The model standardises features with train-split statistics, applies a
    linear weight vector plus intercept, and is consumed both as a callable
    metric (after :func:`register_nr_quality_index`) and as a JSON-able artifact
    (:meth:`to_dict` / :meth:`from_dict`).

    Attributes:
        weights: ``[n_features]`` linear coefficients (on standardised features).
        intercept: scalar bias term.
        feature_names: NR metric names in column order of ``weights``.
        feature_mean: ``[n_features]`` train-split feature means (standardiser).
        feature_std: ``[n_features]`` train-split feature stds (standardiser).
        backend: ``"sklearn_ridge"`` or ``"ridge_closed_form"``.
    """

    weights: torch.Tensor
    intercept: float
    feature_names: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    backend: str

    def predict(self, feats: torch.Tensor) -> torch.Tensor:
        """Predict severity for a batch of raw (un-standardised) feature rows.

        Args:
            feats: ``[N, n_features]`` (or ``[n_features]`` for a single row) of
                raw NR metric values in :attr:`feature_names` order. Non-finite
                entries are imputed with the corresponding train-split mean
                before standardisation (a degraded metric column should not
                NaN-poison the whole prediction).

        Returns:
            ``[N]`` predicted severity, soft-clamped to roughly ``[0, 1]`` (the
            severity coordinate's natural range). The clamp is a presentation
            convenience; the raw linear value is what the calibration stats are
            measured on at fit time.
        """
        x = torch.as_tensor(feats, dtype=torch.float64)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mean = self.feature_mean.to(torch.float64)
        std = self.feature_std.to(torch.float64)
        # Impute non-finite raw features with the train mean (==> standardises to 0).
        x = torch.where(torch.isfinite(x), x, mean)
        std_safe = torch.where(std > 1e-12, std, torch.ones_like(std))
        xs = (x - mean) / std_safe
        w = self.weights.to(torch.float64)
        pred = xs @ w + float(self.intercept)
        return pred.clamp(min=0.0, max=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-able representation (weights as lists)."""
        return {
            "weights": [float(v) for v in self.weights.tolist()],
            "intercept": float(self.intercept),
            "feature_names": list(self.feature_names),
            "feature_mean": [float(v) for v in self.feature_mean.tolist()],
            "feature_std": [float(v) for v in self.feature_std.tolist()],
            "backend": str(self.backend),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NRQualityIndexModel:
        """Reconstruct a model from :meth:`to_dict` output."""
        return cls(
            weights=torch.as_tensor(payload["weights"], dtype=torch.float64),
            intercept=float(payload["intercept"]),
            feature_names=list(payload["feature_names"]),
            feature_mean=torch.as_tensor(payload["feature_mean"], dtype=torch.float64),
            feature_std=torch.as_tensor(payload["feature_std"], dtype=torch.float64),
            backend=str(payload["backend"]),
        )


# ---------------------------------------------------------------------------
# pure-torch statistics (NaN-safe, deterministic)
# ---------------------------------------------------------------------------
def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination ``R^2`` over the finite-aligned entries."""
    try:
        mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
        if int(mask.sum().item()) < 2:
            return NAN
        yt = y_true[mask].to(torch.float64)
        yp = y_pred[mask].to(torch.float64)
        ss_res = float(((yt - yp) ** 2).sum().item())
        ss_tot = float(((yt - yt.mean()) ** 2).sum().item())
        if ss_tot < 1e-12:
            return NAN
        return 1.0 - ss_res / ss_tot
    except Exception:
        return NAN


def _expected_calibration_error(
    y_true: torch.Tensor, y_pred: torch.Tensor, *, n_bins: int = 10
) -> float:
    """Pure-torch ECE on min-max-normalised predictions.

    Predictions are min-max normalised to ``[0, 1]`` then bucketed into
    ``n_bins`` equal-width bins; the per-bin gap is ``|mean(pred) - mean(true)|``
    (with the target also min-max normalised so the two live on the same scale),
    weighted by bin occupancy. Returns a value in ``[0, 1]``.
    """
    try:
        mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
        if int(mask.sum().item()) < 2:
            return NAN
        yt = y_true[mask].to(torch.float64)
        yp = y_pred[mask].to(torch.float64)

        def _minmax(v: torch.Tensor) -> torch.Tensor:
            lo = float(v.min().item())
            hi = float(v.max().item())
            if hi - lo < 1e-12:
                return torch.zeros_like(v)
            return (v - lo) / (hi - lo)

        pn = _minmax(yp)
        tn = _minmax(yt)
        n_obs = pn.shape[0]
        edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
        ece = 0.0
        for b in range(n_bins):
            lo = edges[b]
            hi = edges[b + 1]
            in_bin = (pn >= lo) & (pn <= hi) if b == n_bins - 1 else (pn >= lo) & (pn < hi)
            cnt = int(in_bin.sum().item())
            if cnt == 0:
                continue
            gap = abs(float(pn[in_bin].mean().item()) - float(tn[in_bin].mean().item()))
            ece += (cnt / n_obs) * gap
        return float(ece)
    except Exception:
        return NAN


def _auroc(y_bin: torch.Tensor, scores: torch.Tensor) -> float:
    """Rank-based AUROC (Mann-Whitney U / (n_pos * n_neg)); pure torch.

    Args:
        y_bin: ``[N]`` binary labels (1 = positive / high-severity, 0 = negative).
        scores: ``[N]`` predictor scores (higher ⇒ more positive).

    Returns:
        AUROC in ``[0, 1]``; ``NaN`` if either class is empty. Ties contribute
        0.5 via average-rank (the standard tie correction).
    """
    try:
        mask = torch.isfinite(scores)
        yb = y_bin[mask].to(torch.float64)
        sc = scores[mask].to(torch.float64)
        n_pos = float((yb == 1).sum().item())
        n_neg = float((yb == 0).sum().item())
        if n_pos == 0 or n_neg == 0:
            return NAN
        # Average ranks (1-based) with tie handling.
        order = torch.argsort(sc)
        ranks = torch.empty_like(sc)
        ranks[order] = torch.arange(1, sc.shape[0] + 1, dtype=torch.float64)
        # Resolve ties to average rank.
        sorted_sc = sc[order]
        i = 0
        sorted_ranks = ranks[order].clone()
        m = sorted_sc.shape[0]
        while i < m:
            j = i
            while j + 1 < m and sorted_sc[j + 1] == sorted_sc[i]:
                j += 1
            if j > i:
                avg = float(sorted_ranks[i : j + 1].mean().item())
                sorted_ranks[i : j + 1] = avg
            i = j + 1
        ranks[order] = sorted_ranks
        sum_pos_ranks = float(ranks[yb == 1].sum().item())
        u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))
    except Exception:
        return NAN


# ---------------------------------------------------------------------------
# ridge backends
# ---------------------------------------------------------------------------
def _ridge_closed_form(
    x: torch.Tensor, y: torch.Tensor, *, alpha: float = 1.0
) -> tuple[torch.Tensor, float]:
    """Closed-form ridge via the normal equations (intercept un-penalised).

    Solves ``(X^T X + alpha I) w = X^T y`` on the centred design (so the
    intercept is the target mean), in ``float64``. Falls back to a least-norm
    solve if the system is singular.

    Args:
        x: ``[N, F]`` standardised design matrix (no intercept column).
        y: ``[N]`` targets.
        alpha: ridge penalty.

    Returns:
        ``(weights[F], intercept)``.
    """
    xd = x.to(torch.float64)
    yd = y.to(torch.float64)
    f = xd.shape[1]
    y_mean = float(yd.mean().item())
    yc = yd - y_mean
    # Features are already standardised (mean ~0) but centre defensively.
    x_mean = xd.mean(dim=0, keepdim=True)
    xc = xd - x_mean
    gram = xc.t() @ xc + alpha * torch.eye(f, dtype=torch.float64)
    rhs = xc.t() @ yc
    try:
        w = torch.linalg.solve(gram, rhs)
    except Exception:
        w = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)
    # Re-derive intercept on the original (centred-feature) basis.
    intercept = y_mean - float((x_mean @ w).item())
    return w, intercept


def _fit_ridge(
    x_train: torch.Tensor, y_train: torch.Tensor, *, alpha: float, force_closed_form: bool
) -> tuple[torch.Tensor, float, str]:
    """Fit ridge with sklearn if available, else the closed-form fallback.

    scikit-learn is lazy-imported INSIDE the body and gated behind
    ``try/except ImportError`` — a missing optional dep must never break import
    or the fit.

    Args:
        x_train: ``[N, F]`` standardised design.
        y_train: ``[N]`` targets.
        alpha: ridge penalty.
        force_closed_form: skip sklearn entirely (used by tests to exercise the
            dependency-free path).

    Returns:
        ``(weights[F], intercept, backend)``.
    """
    if not force_closed_form:
        try:
            from sklearn.linear_model import Ridge  # lazy + gated

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(
                x_train.to(torch.float64).numpy(),
                y_train.to(torch.float64).numpy(),
            )
            w = torch.as_tensor(model.coef_, dtype=torch.float64)
            intercept = float(model.intercept_)
            return w, intercept, "sklearn_ridge"
        except ImportError:
            pass
        except Exception:
            # Any sklearn-internal failure (e.g. degenerate data) also falls
            # back to the deterministic closed form rather than raising.
            pass
    w, intercept = _ridge_closed_form(x_train, y_train, alpha=alpha)
    return w, intercept, "ridge_closed_form"


# ---------------------------------------------------------------------------
# design-matrix assembly + group split
# ---------------------------------------------------------------------------
def _build_design(
    dataset: MetricEvaluationDataset, nr_names: list[str], target: str
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], torch.Tensor]:
    """Assemble ``(X, y, content_ids, families, valid_mask)``.

    Rows whose target is non-finite are dropped. Feature columns are NOT dropped
    here (per-feature non-finite entries are median-imputed) so a single
    degraded metric column does not discard otherwise-usable rows.

    Returns:
        X: ``[N, F]`` design matrix (median-imputed features).
        y: ``[N]`` severity targets.
        content_ids: per-row subject ids (for the group split).
        families: per-row degradation family.
        valid_mask: ``[N_total]`` bool selecting kept rows from the dataset.
    """
    samples = dataset.samples
    n_total = len(samples)
    cols: list[torch.Tensor] = []
    for name in nr_names:
        vals = dataset.metric_values.get(name)
        if vals is None:
            cols.append(torch.full((n_total,), NAN, dtype=torch.float64))
        else:
            cols.append(torch.as_tensor(vals, dtype=torch.float64))
    x_full = torch.stack(cols, dim=1) if cols else torch.zeros((n_total, 0), dtype=torch.float64)
    y_full = torch.tensor(
        [float(s.severity.get(target, NAN)) for s in samples], dtype=torch.float64
    )
    valid = torch.isfinite(y_full)
    x = x_full[valid]
    y = y_full[valid]
    kept = [i for i in range(n_total) if bool(valid[i].item())]
    content_ids = [samples[i].content_id for i in kept]
    families = [samples[i].degradation_family for i in kept]
    # Median-impute remaining non-finite feature entries (column medians on kept).
    for c in range(x.shape[1]):
        col = x[:, c]
        finite = col[torch.isfinite(col)]
        med = float(finite.median().item()) if finite.numel() > 0 else 0.0
        x[:, c] = torch.where(torch.isfinite(col), col, torch.tensor(med, dtype=torch.float64))
    return x, y, content_ids, families, valid


def _group_split(
    content_ids: list[str], *, test_size: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split row indices into train/test by ``content_id`` (no subject overlap).

    Groups (unique content_ids) are shuffled with a seeded generator; the first
    ``ceil(test_size * n_groups)`` go to the test set. Guarantees at least one
    group on each side when there are >= 2 groups.

    Returns:
        ``(train_idx, test_idx)`` long tensors of row positions.
    """
    groups = sorted(set(content_ids))
    n_groups = len(groups)
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_groups, generator=gen).tolist()
    shuffled = [groups[i] for i in perm]
    n_test = max(1, round(test_size * n_groups)) if n_groups >= 2 else 0
    n_test = min(n_test, n_groups - 1) if n_groups >= 2 else 0
    test_groups = set(shuffled[:n_test])
    train_idx = [i for i, c in enumerate(content_ids) if c not in test_groups]
    test_idx = [i for i, c in enumerate(content_ids) if c in test_groups]
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def _standardise(
    x_train: torch.Tensor, x_eval: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardise ``x_eval`` with ``x_train`` statistics (train-only fit)."""
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)
    std_safe = torch.where(std > 1e-12, std, torch.ones_like(std))
    xt = (x_train - mean) / std_safe
    xe = (x_eval - mean) / std_safe
    return xt, xe, mean, std


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
def fit_nr_quality_index(
    dataset: MetricEvaluationDataset,
    nr_names: list[str],
    *,
    target: str = "theta",
    test_size: float = 0.3,
    seed: int = 0,
    alpha: float = 1.0,
    force_closed_form: bool = False,
) -> tuple[NRQualityIndexModel, dict[str, Any]]:
    """Fit the ``nr_quality_index`` meta-metric onto simulator severity.

    Builds a design matrix from the NR battery columns of ``dataset``, targets
    the severity coordinate ``target`` (default ``"theta"``), standardises on a
    group split by ``content_id`` (train-only stats), and fits a ridge model.
    Reports calibration (ECE) and generalisation (R², Spearman, AUROC, optional
    cross-degradation transfer).

    Never raises: on insufficient data it returns a model with ``NaN`` stats and
    a ``"reason"`` in the metrics dict.

    Args:
        dataset: the shared sweep dataset (NR + FR columns aligned to samples).
        nr_names: NR metric column names to use as features (column order).
        target: severity-dict key to predict (spec: the simulator theta).
        test_size: fraction of *groups* (subjects) held out for the test split.
        seed: deterministic seed for the group shuffle.
        alpha: ridge penalty.
        force_closed_form: bypass sklearn and use the closed-form ridge (tests).

    Returns:
        ``(model, metrics)`` where ``metrics`` carries ``r2_test``,
        ``spearman_test``, ``ece``, ``auroc``, ``cross_degradation``,
        ``backend``, ``n_train``, ``n_test``, ``feature_names``.
    """
    feature_names = list(nr_names)
    n_feat = len(feature_names)

    def _nan_model(reason: str) -> tuple[NRQualityIndexModel, dict[str, Any]]:
        model = NRQualityIndexModel(
            weights=torch.zeros(n_feat, dtype=torch.float64),
            intercept=float("nan"),
            feature_names=feature_names,
            feature_mean=torch.zeros(n_feat, dtype=torch.float64),
            feature_std=torch.ones(n_feat, dtype=torch.float64),
            backend="ridge_closed_form",
        )
        metrics: dict[str, Any] = {
            "r2_test": NAN,
            "spearman_test": NAN,
            "ece": NAN,
            "auroc": NAN,
            "cross_degradation": None,
            "backend": model.backend,
            "n_train": 0,
            "n_test": 0,
            "feature_names": feature_names,
            "reason": reason,
        }
        return model, metrics

    if n_feat == 0:
        return _nan_model("no NR feature columns supplied")

    try:
        x, y, content_ids, _families, _valid = _build_design(dataset, feature_names, target)
    except Exception as exc:  # pragma: no cover - defensive
        return _nan_model(f"design assembly failed: {exc!r}")

    if x.shape[0] < 4:
        return _nan_model("insufficient finite samples (need >= 4)")
    if len(set(content_ids)) < 2:
        return _nan_model("need >= 2 distinct content_ids for a group split")

    train_idx, test_idx = _group_split(content_ids, test_size=test_size, seed=seed)
    if int(train_idx.numel()) < 2 or int(test_idx.numel()) < 1:
        return _nan_model("group split produced an empty train/test side")

    x_tr_raw, x_te_raw = x[train_idx], x[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    x_tr, _x_te, mean, std = _standardise(x_tr_raw, x_te_raw)

    try:
        weights, intercept, backend = _fit_ridge(
            x_tr, y_tr, alpha=alpha, force_closed_form=force_closed_form
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _nan_model(f"ridge fit failed: {exc!r}")

    model = NRQualityIndexModel(
        weights=weights,
        intercept=intercept,
        feature_names=feature_names,
        feature_mean=mean,
        feature_std=std,
        backend=backend,
    )

    # --- test-split statistics (predict on RAW features via model.predict) ---
    try:
        y_pred = model.predict(x_te_raw)
    except Exception:
        y_pred = torch.full_like(y_te, NAN)

    r2_test = _r2_score(y_te, y_pred)
    spearman_test = spearman(y_pred.tolist(), y_te.tolist())
    ece = _expected_calibration_error(y_te, y_pred)

    # Binarise theta at its median over the TEST split → high/low severity.
    try:
        med = float(y_te.median().item())
        y_bin = (y_te > med).to(torch.float64)
        auroc = _auroc(y_bin, y_pred)
    except Exception:
        auroc = NAN

    cross_degradation = _cross_degradation(
        dataset,
        feature_names,
        target,
        alpha=alpha,
        force_closed_form=force_closed_form,
    )

    metrics: dict[str, Any] = {
        "r2_test": r2_test,
        "spearman_test": spearman_test,
        "ece": ece,
        "auroc": auroc,
        "cross_degradation": cross_degradation,
        "backend": backend,
        "n_train": int(train_idx.numel()),
        "n_test": int(test_idx.numel()),
        "feature_names": feature_names,
    }
    return model, metrics


def _cross_degradation(
    dataset: MetricEvaluationDataset,
    feature_names: list[str],
    target: str,
    *,
    alpha: float,
    force_closed_form: bool,
) -> dict[str, Any] | None:
    """Train on donor families, test on the held-out families (transfer probe).

    Trains on the families present from :data:`_DEFAULT_TRAIN_FAMILIES` and
    evaluates on every other family in the data. Returns ``None`` when fewer
    than two distinct families exist (single-family data has no held-out
    families — spec: set to null and note it).
    """
    try:
        x, y, _cids, families, _valid = _build_design(dataset, feature_names, target)
        present = sorted(set(families))
        if len(present) < 2:
            return None
        train_fams = [f for f in present if f in set(_DEFAULT_TRAIN_FAMILIES)]
        if not train_fams or set(train_fams) == set(present):
            # No usable donor/held-out partition with the default donors;
            # fall back to "first family as donor, rest held out" so the probe
            # is still meaningful, and record the partition used.
            train_fams = [present[0]]
        held_out = [f for f in present if f not in set(train_fams)]
        if not held_out:
            return None
        fam_t = families
        tr_mask = torch.tensor([f in set(train_fams) for f in fam_t], dtype=torch.bool)
        te_mask = torch.tensor([f in set(held_out) for f in fam_t], dtype=torch.bool)
        if int(tr_mask.sum().item()) < 2 or int(te_mask.sum().item()) < 2:
            return None
        x_tr_raw, x_te_raw = x[tr_mask], x[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        x_tr, _x_te, mean, std = _standardise(x_tr_raw, x_te_raw)
        weights, intercept, backend = _fit_ridge(
            x_tr, y_tr, alpha=alpha, force_closed_form=force_closed_form
        )
        model = NRQualityIndexModel(
            weights=weights,
            intercept=intercept,
            feature_names=feature_names,
            feature_mean=mean,
            feature_std=std,
            backend=backend,
        )
        y_pred = model.predict(x_te_raw)
        return {
            "train_families": list(train_fams),
            "held_out_families": list(held_out),
            "r2": _r2_score(y_te, y_pred),
            "spearman": spearman(y_pred.tolist(), y_te.tolist()),
            "n_train": int(tr_mask.sum().item()),
            "n_test": int(te_mask.sum().item()),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# registration as a metric
# ---------------------------------------------------------------------------
class _NRQualityIndexMetric:
    """Callable metric wrapper around a fitted :class:`NRQualityIndexModel`.

    Research-mode meta-metric: it does NOT operate on ``(prediction, target)``
    pixels. Instead it expects a precomputed NR feature vector via
    ``kwargs["nr_features"]`` (a ``dict[name -> float]``). When the feature
    vector is absent it returns ``NaN`` rather than raising — the harness records
    NaN for an uncomputable meta-metric.

    ``higher_is_better`` is ``False``: a higher predicted severity means a worse
    reconstruction.
    """

    higher_is_better: bool = False

    def __init__(self, model: NRQualityIndexModel | None = None) -> None:
        # The registry instantiates metrics with no args; the live model is
        # injected as a class attribute by ``register_nr_quality_index``.
        self._model = model if model is not None else getattr(type(self), "_bound_model", None)

    @property
    def name(self) -> str:
        return "nr_quality_index"

    def __call__(
        self,
        prediction: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> float:
        """Predict severity from a precomputed NR feature dict.

        Args:
            prediction: ignored (meta-metric; kept for the ``IMetric`` contract).
            target: ignored.
            **kwargs: must carry ``nr_features: dict[str, float]`` mapping NR
                metric names to scalar values. Missing names are imputed with
                the model's train mean (via :meth:`NRQualityIndexModel.predict`).

        Returns:
            Predicted severity (float in ``[0, 1]``), or ``NaN`` when no feature
            dict is supplied or the model is unbound / prediction fails.
        """
        model = self._model
        if model is None:
            return NAN
        feats = kwargs.get("nr_features")
        if not isinstance(feats, dict) or not feats:
            return NAN
        try:
            row = torch.tensor(
                [float(feats.get(n, float("nan"))) for n in model.feature_names],
                dtype=torch.float64,
            )
            return float(model.predict(row)[0].item())
        except Exception:
            return NAN


def register_nr_quality_index(model: NRQualityIndexModel) -> None:
    """Register the fitted ``model`` as the metric ``"nr_quality_index"``.

    Idempotent: any prior ``nr_quality_index`` registration (and its capability
    metadata) is cleared first so re-registration in tests / re-fits cannot trip
    the registry's "duplicate with a different class" guard.

    The registered metric is a no-reference, no-measurement-context meta-metric:
    it reads its inputs from ``kwargs["nr_features"]`` at call time, so it
    declares no ``MetricContext`` ``needs`` and ``requires_reference=False``.

    Args:
        model: the fitted severity predictor to bind into the metric callable.
    """
    from mriforge.core.metrics.registry import MetricsRegistry

    name = "nr_quality_index"
    # Idempotent overwrite: drop any prior registration + capability metadata.
    MetricsRegistry._metrics.pop(name, None)
    MetricsRegistry._needs.pop(name, None)
    MetricsRegistry._requires_context.pop(name, None)
    MetricsRegistry._requires_reference.pop(name, None)
    # Drop any aliases that pointed at this canonical name (defensive; we set none).
    for alias in [a for a, t in list(MetricsRegistry._aliases.items()) if t == name]:
        MetricsRegistry._aliases.pop(alias, None)

    # Bind the live model onto the class so the no-arg registry constructor
    # (``cls._metrics[name]()``) still resolves it.
    bound_cls = type(
        "_NRQualityIndexMetricBound",
        (_NRQualityIndexMetric,),
        {"_bound_model": model},
    )
    MetricsRegistry.register(
        name,
        requires_reference=False,
        requires_measurement_context=False,
        needs=(),
    )(bound_cls)


__all__ = [
    "NRQualityIndexModel",
    "fit_nr_quality_index",
    "register_nr_quality_index",
]
