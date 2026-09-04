"""Gen-2 Task ranker — task-driven SROCC defensibility against a REAL ΔP.

The Task axis asks the most defensible meta-evaluation question there is: *does
this metric track the thing a downstream clinical task actually cares about?*
For each task ``T`` with a performance-drop curve :math:`\\Delta P_T(t)` measured
over the severity sweep, it computes :math:`\\mathrm{SROCC}(M_k, \\Delta P_T)` per
metric and aggregates the absolute correlations over tasks.

That question is only meaningful when :math:`\\Delta P_T` is **measured**, by
running a frozen downstream task network on the degraded images. It is therefore
capability-gated (``requires_task_net=True``) and :meth:`TaskScoreRanker.rank`
**raises** when no measured curve is supplied.

Why the synthetic curve is not a default (issue #253)
-----------------------------------------------------
The former production default synthesised :math:`\\Delta P_T` from the severity
grid alone (:func:`synthetic_task_perf_drop_fixture`). Every such curve is a
deterministic, *monotone* function of severity ``s``: seg/reg/cls have Spearman
+0.977 / +0.979 / +0.953 against ``s`` and +0.94 against each other. Correlating a
metric's trajectory with them therefore measures **only whether the metric is
monotone in severity** — which is what ADR already measures — while *claiming* to
measure downstream clinical task tracking. Worse, the only thing separating the
three "tasks" is a ``0.05·N(0,1)`` noise draw, so the ranking is seed-dependent:
over 8 seeds on an 8-metric pool the axis produced 8 distinct orderings and 3
distinct rank-1 winners. Registered with ``requires_task_net=False``, it ran and
voted on **every** production sweep.

That is pitfall #16 exactly — an advertised mechanism ("task-driven clinical
defensibility") that is inert, collapsing to a monotonicity detector whose output
is noise. The synthetic curves survive as a **unit-test fixture** and nothing
else: the name says so, and no production path can reach them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..ranking_guards import assert_not_degenerate
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from ._spearman_legacy import legacy_srocc
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import _build_trajectory_table, _per_family_average


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


# ---------------------------------------------------------------------------
# TEST FIXTURE ONLY — never a production default (issue #253).
# ---------------------------------------------------------------------------


def synthetic_task_perf_drop_fixture(
    severity_grid: np.ndarray, seed: int = 0
) -> dict[str, np.ndarray]:
    """**TEST FIXTURE.** Plausible-looking ``ΔP_T`` curves synthesised from θ alone.

    .. danger::

       This is **not** a stand-in for a measured task-performance drop and must
       never be wired into a production ranking path. Each curve is a
       deterministic monotone function of the severity grid plus a
       ``0.05·N(0, 1)`` noise draw. Its per-metric SROCC against a metric
       trajectory is a *monotonicity* statistic wearing a clinical label, and the
       ranking it induces is a function of ``seed``. It exists so unit tests have
       a cheap ΔP-shaped array with a known analytic form.

    Each task is given a different non-linear response to severity:

    - segmentation: piecewise-linear with low-severity tolerance
    - registration: convex (TRE grows quadratically)
    - classification: sigmoidal collapse near ``s=0.6``

    Args:
        severity_grid: ``(T,)`` ordered severity coordinates.
        seed: RNG seed for the additive ``0.05 * N(0, 1)`` noise (drawn once,
            per-curve, in the order seg → reg → cls).

    Returns:
        ``{task_name: (T,) drop array}`` with keys ``segmentation_dice_drop``,
        ``registration_tre_increase``, ``classification_auroc_drop``.
    """
    rng = np.random.default_rng(seed)
    s = severity_grid
    seg = np.where(s < 0.2, 0.0, (s - 0.2) / 0.8) + 0.05 * rng.standard_normal(len(s))
    reg = (s**2) + 0.05 * rng.standard_normal(len(s))
    cls = 1.0 / (1.0 + np.exp(-12.0 * (s - 0.6))) + 0.05 * rng.standard_normal(len(s))
    return {
        "segmentation_dice_drop": np.clip(seg, 0.0, None),
        "registration_tre_increase": np.clip(reg, 0.0, None),
        "classification_auroc_drop": np.clip(cls, 0.0, None),
    }


@dataclass
class TaskScoreConfig:
    """Configuration for :class:`TaskScoreRanker`.

    Attributes:
        task_perf_drop: ``{task_name: (T,) array}`` of **measured** downstream
            task-performance drops, one value per severity bucket, produced by
            running a frozen task network over the sweep. There is **no default**:
            ``None`` makes :meth:`TaskScoreRanker.rank` raise. Each array must
            have length ``T`` (the number of severity buckets).
        task_weights: Optional ``{task_name: weight}`` mapping. When ``None``,
            uniform ``1/T`` weights are used (renormalised to sum to 1).
    """

    task_perf_drop: Mapping[str, Sequence[float]] | None = None
    task_weights: Mapping[str, float] | None = None


@register_ranker(
    "task",
    generation=2,
    requires_task_net=True,
    data_kwargs=("task_perf_drop",),
    description=(
        "Task-driven SROCC defensibility against a MEASURED frozen-task "
        "performance-drop curve. Capability-gated: raises without one."
    ),
)
class TaskScoreRanker(BaseRanker):
    """Task-driven SROCC defensibility ranker (needs a measured ΔP curve).

    Per metric, the absolute Spearman rank correlation between its mean severity
    trajectory and each task's measured ΔP curve, averaged over tasks with
    (uniform-by-default) weights.
    """

    method_name = "Task"

    def __init__(
        self,
        config: TaskScoreConfig | None = None,
        *,
        task_perf_drop: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        # ``task_perf_drop`` is also accepted as a bare kwarg so the registry's
        # ``data_kwargs`` injection path (spec.instantiate(**ranker_kwargs)) can
        # supply it without the caller building a config object.
        self.config = config or TaskScoreConfig()
        if task_perf_drop is not None:
            self.config = TaskScoreConfig(
                task_perf_drop=task_perf_drop,
                task_weights=self.config.task_weights,
            )

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)

        if self.config.task_perf_drop is None:
            raise ValueError(
                "TaskScoreRanker requires a MEASURED task performance-drop curve "
                "(TaskScoreConfig.task_perf_drop), produced by running a frozen "
                "downstream task network over the severity sweep. It has no "
                "default: the former synthetic curve was a deterministic monotone "
                "function of severity, so the axis silently degenerated into a "
                "seed-dependent monotonicity detector while claiming to measure "
                "clinical task tracking (issue #253, pitfall #16). Inject it via "
                "MetaEvaluationPipeline.from_registry(ranker_kwargs={'task': "
                "{'task_perf_drop': ...}}), or leave have_task_net=False so this "
                "ranker is omitted from the pool."
            )

        names = metric_set.names()

        # Per-metric mean trajectory: mean over families of the per-family
        # averages.
        severity_grid: list[float] = []
        mean_traj: dict[str, np.ndarray] = {}
        for name in names:
            traj, severities, contents, families = _build_trajectory_table(dataset, name)
            if not severity_grid:
                severity_grid = severities
            family_avgs = _per_family_average(traj, families, contents)
            if family_avgs:
                stack = torch.stack(list(family_avgs.values()), dim=0)  # (F, T)
                mean_k = stack.mean(dim=0)  # (T,)
            else:
                mean_k = torch.full((len(severities),), float("nan"), dtype=torch.float64)
            mean_traj[name] = mean_k.cpu().numpy().astype(np.float64)

        sev = np.asarray(severity_grid, dtype=np.float64)

        curves: dict[str, np.ndarray] = {}
        for task, arr in self.config.task_perf_drop.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[0] != sev.shape[0]:
                raise ValueError(
                    f"task_perf_drop[{task!r}] has length {a.shape[0]} "
                    f"but the severity grid has {sev.shape[0]} buckets."
                )
            curves[task] = a
        if not curves:
            raise ValueError("task_perf_drop is empty — no task curve to score against.")

        tasks = list(curves)
        if self.config.task_weights is None:
            task_weights = {t: 1.0 / len(tasks) for t in tasks}
        else:
            task_weights = dict(self.config.task_weights)
        wsum = sum(task_weights.values())
        weights = {k: v / wsum for k, v in task_weights.items()}

        # Per-task SROCC (absolute) per metric, then weighted aggregate.
        per_task_srocc: dict[str, dict[str, float]] = {}
        aggregate: dict[str, float] = dict.fromkeys(names, 0.0)
        for task in tasks:
            delta = np.nan_to_num(np.asarray(curves[task], dtype=np.float64))
            delta_std = float(delta.std())  # numpy ddof=0
            srocc_task: dict[str, float] = {}
            for name in names:
                v = np.nan_to_num(mean_traj[name])
                # |spearman(v, delta)| via the legacy-faithful helper (scipy
                # mid-ranks + std(v)<1e-12 guard); the delta-std guard reproduces
                # the ``or std(delta_T)<1e-12 -> 0`` branch. The in-core ordinal
                # ``_spearman_rho`` diverged by ~0.29 on tied trajectories
                # (adversarial verification 2026-05-30).
                if delta_std < 1e-12:
                    srocc_task[name] = 0.0
                else:
                    srocc_task[name] = legacy_srocc(v, delta)
                aggregate[name] += weights[task] * srocc_task[name]
            per_task_srocc[task] = srocc_task

        scores = {name: float(aggregate[name]) for name in names}
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=self._ranks_from_scores(scores),
            diagnostics={
                "per_task_srocc": per_task_srocc,
                "task_weights": weights,
                # Provenance: a measured curve is the ONLY thing that gets here.
                "task_perf_drop_source": "measured",
                "tasks": tasks,
            },
        )


__all__ = [
    "TaskScoreConfig",
    "TaskScoreRanker",
    "synthetic_task_perf_drop_fixture",
]
