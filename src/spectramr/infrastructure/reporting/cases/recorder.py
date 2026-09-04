r"""Record best/median/worst validation cases for the qualitative figures.

Fed from the validation image-logging seam (CPU tensors already detached),
so it adds no GPU sync. At end of validation it writes a small npz bundle +
``cases_index.json`` that ``load_report_cases`` reconstructs at report time.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ReportCaseRecorder:
    """Keep a bounded set of representative validation cases.

    Args:
        n_cases: total cases to retain (0 disables recording).
        selection: ``best_median_worst`` | ``random`` | ``first``.
        primary_metric: metric used to rank cases.
        higher_is_better: ranking direction for ``primary_metric``.
    """

    def __init__(
        self,
        *,
        n_cases: int,
        selection: str,
        primary_metric: str,
        higher_is_better: bool,
        max_pool: int = 256,
        record_volumes: bool = False,
    ) -> None:
        self.n_cases = int(n_cases)
        self.selection = selection
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        # When True the feeder additionally stores ``*_volume`` (3-D) arrays for
        # the interactive volumetric viewer; no-op on single-slice data.
        self.record_volumes = bool(record_volumes)
        # Bound in-memory accumulation: the seam feeds one case per validation
        # call, so over a long run the pool would grow unboundedly. Keep at most
        # ``max_pool`` candidates, evicting the median-metric one (preserves the
        # extremes that best/median/worst selection needs).
        self.max_pool = max(int(max_pool), max(self.n_cases, 1))
        self._cases: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self.n_cases > 0

    def _metric_value(self, metrics: dict) -> float | None:
        """Resolve ``primary_metric`` against a case's stored metric keys.

        ``validation.primary_metric`` is a bare name (``psnr``), but the feed
        seam stores the validation dict verbatim, whose keys carry a monitor
        prefix (``val_psnr``). An exact-match lookup silently misses, collapsing
        best/median/worst to insertion order. Try the bare name, then the name
        under each monitor prefix, then (if ``primary_metric`` is itself
        prefixed) the stripped name — reusing the ``metric_directions`` SSOT
        prefix list so this stays aligned with monitor-key resolution.
        """
        # infrastructure -> core is a rightward (allowed) import; kept local to
        # avoid pulling the metrics package onto the module import path.
        from spectramr.core.metrics.metric_directions import _MONITOR_PREFIXES

        name = self.primary_metric
        candidates = [name, *(f"{p}{name}" for p in _MONITOR_PREFIXES)]
        for prefix in _MONITOR_PREFIXES:
            if name.startswith(prefix):
                candidates.append(name[len(prefix) :])
        for key in candidates:
            if key in metrics:
                return float(metrics[key])
        return None

    def observe(
        self,
        *,
        case_id: str,
        arrays: dict[str, np.ndarray],
        metrics: dict[str, float],
        domain: dict,
    ) -> None:
        """Record one case. Arrays are coerced to float32 numpy on CPU."""
        if not self.enabled:
            return
        coerced = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items() if v is not None}
        self._cases.append(
            {
                "case_id": case_id,
                "arrays": coerced,
                "metrics": dict(metrics),
                "domain": dict(domain),
            }
        )
        if len(self._cases) > self.max_pool:
            self._evict_median()

    def _evict_median(self) -> None:
        """Drop the case nearest the median primary-metric value (keep extremes)."""
        scored = [
            (v, c) for c in self._cases if (v := self._metric_value(c["metrics"])) is not None
        ]
        if len(scored) < 3:
            # not enough metric-bearing cases to rank — drop the oldest
            self._cases.pop(0)
            return
        ordered = sorted(scored, key=lambda vc: vc[0])
        victim = ordered[len(ordered) // 2][1]
        # Remove by IDENTITY, never ``list.remove``. ``remove``
        # short-circuits on identity but otherwise falls back to ``==``, and a
        # case dict holds ``{str: np.ndarray}``. Dict equality compares
        # ``case_id`` first, so it normally short-circuits to False — but
        # cascade validation emits one case per acceleration rung at the SAME
        # training iteration, and the feed seam labels them all
        # ``f"val_step{step}"``. On that tie the comparison advances to the
        # arrays: ``ndarray == ndarray`` returns an element-wise array whose
        # ``bool()`` raises "The truth value of an array with more than one
        # element is ambiguous". That surfaced only after the pool first
        # exceeded ``max_pool`` — hours into a run — and then killed the
        # caller's whole image-logging block on every subsequent validation.
        for index, case in enumerate(self._cases):
            if case is victim:
                del self._cases[index]
                return

    def _select(self) -> list[tuple[str, dict]]:
        if not self._cases:
            return []
        scored = [c for c in self._cases if self._metric_value(c["metrics"]) is not None]
        if not scored:
            scored = self._cases
        if self.selection == "first":
            return [("case", c) for c in scored[: self.n_cases]]
        if self.selection == "random":
            stride = max(1, len(scored) // self.n_cases)
            return [("case", c) for c in scored[::stride][: self.n_cases]]
        # best_median_worst
        ordered = sorted(
            scored,
            key=lambda c: self._metric_value(c["metrics"]) or 0.0,
            reverse=self.higher_is_better,
        )
        picks: list[tuple[str, dict]] = []
        if ordered:
            picks.append(("best", ordered[0]))
        if len(ordered) > 2:
            picks.append(("median", ordered[len(ordered) // 2]))
        if len(ordered) > 1:
            picks.append(("worst", ordered[-1]))
        extra = [c for c in ordered if all(c is not p for _, p in picks)]
        i = 0
        while len(picks) < self.n_cases and i < len(extra):
            picks.append(("case", extra[i]))
            i += 1
        return picks[: self.n_cases]

    def write(self, run_dir: str | Path, subdir: str = "report_cases") -> Path:
        """Write the selected cases to ``<run_dir>/<subdir>/`` and return it."""
        out = Path(run_dir) / subdir
        if not self.enabled or not self._cases:
            return out
        out.mkdir(parents=True, exist_ok=True)
        index: list[dict] = []
        for k, (rank, case) in enumerate(self._select()):
            npz_name = f"case_{k}.npz"
            # Write-then-rename (#1685). ``np.savez_compressed`` streams a zip
            # straight to its destination, so a reader that opens the path
            # mid-write -- or a second writer racing on it -- sees a truncated
            # archive and fails with ``Bad CRC-32``, which is what three of four
            # ranks produced. ``os.replace`` is atomic within a filesystem, so
            # the final name only ever names a complete archive.
            #
            # The tmp suffix goes BEFORE the extension: savez appends ``.npz``
            # to any filename lacking it, so ``case_0.npz.tmp`` would land on
            # disk as ``case_0.npz.tmp.npz`` and the rename would miss it.
            tmp_npz = out / f"case_{k}.tmp.npz"
            np.savez_compressed(tmp_npz, **case["arrays"])
            os.replace(tmp_npz, out / npz_name)
            index.append(
                {
                    "case_id": case["case_id"],
                    "npz": npz_name,
                    "rank": rank,
                    "metrics": case["metrics"],
                    "domain": case["domain"],
                }
            )
        # The index is the manifest readers trust to enumerate the archives, so
        # it lands atomically and LAST -- its presence then implies every
        # ``case_*.npz`` it names is already complete.
        tmp_index = out / "cases_index.tmp.json"
        tmp_index.write_text(json.dumps(index, indent=2, sort_keys=True))
        os.replace(tmp_index, out / "cases_index.json")
        logger.info("report cases: wrote %d cases to %s", len(index), out)
        return out
