"""Report artifacts for the inference verbs (``infer`` / ``predict``).

``report`` is entirely artifact-driven: :func:`generate_report` consumes on-disk
files through ``aggregator.py`` and never touches a live model or trainer. So the
only thing standing between a prediction run and a report was that the inference
path wrote **no metrics at all** -- ``infer`` saved output tensors and returned,
while its own docstring promised "output paths, metrics, and processing summary".

This module fills the socket the architecture already cut. ``aggregator.py``
documents ``final_eval.json`` as *"optional, written by inference pipelines"* and
had **four readers and zero writers** anywhere in the repo; that file is the
designated seam, and this is its writer. (``final_metrics.json`` is *not* the
seam -- it belongs to ``training_loop`` and carries ``best``/``final`` splits that
a prediction run has no basis to claim.)

Two constraints shaped the split across the files written here.

**Skip records cannot live in ``final_eval.json``.** ``_flatten_eval_json``
applies ``float(scalar)`` to every value it reads, so a string reason parked in
that file raises ``ValueError`` *inside the aggregator* -- turning a metric that
merely could not be computed into a report that cannot be built at all. The
reasons therefore go in a sibling manifest, and ``final_eval.json`` stays a pure
``{metric: {subject: float}}`` map. It is still written when nothing was
computable: an empty object distinguishes "evaluation ran and found nothing it
could measure" from "evaluation never ran", and those are different findings.

**Nothing is substituted for an unavailable metric.** The declared set comes from
the arm, resolved by the same selector training uses, and every declared name
ends up in the manifest as computed, skipped or failed. In particular the
no-reference battery is *not* swept in to fill the gap when a full-reference
metric has no target: ``metrics.nr`` is deliberately excluded from selection
during training ("research-mode / validation-pending", runs only through
``spectramr meta-evaluate --nr-battery``), and quietly promoting those numbers into
a report under an inference run's byline would launder a validation-pending
measurement into a headline figure. A metric with no reference is reported as
having no reference (pitfall #16).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.core.metrics.registry import MetricsRegistry

logger = logging.getLogger(__name__)

# Statuses carried by every declared metric in the manifest.
STATUS_COMPUTED = "computed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

# Skip reasons. These are stable identifiers -- downstream tooling and the
# report manifest match on them, so they are worded as causes, not apologies.
REASON_UNREGISTERED = "unregistered"
REASON_NO_REFERENCE = "no_reference_available_on_this_path"
REASON_NEEDS_CONTEXT = "measurement_context_not_assembled"

FINAL_EVAL_JSON = "final_eval.json"
FINAL_EVAL_MANIFEST_JSON = "final_eval_manifest.json"
METRICS_CSV = "inference_metrics.csv"


@dataclass(frozen=True)
class MetricOutcome:
    """What became of one declared metric on this run.

    ``values`` is per-subject and empty unless ``status`` is ``computed``.
    """

    name: str
    status: str
    reason: str | None = None
    values: dict[str, float] | None = None


def classify_metric(name: str, *, has_reference: bool) -> str | None:
    """Return the reason ``name`` cannot be computed here, or ``None`` if it can.

    The order is load-bearing -- each test is only meaningful once the previous
    one has passed, and reporting the *first* applicable cause is what makes the
    manifest actionable. A name that is not registered has no metadata to
    interrogate at all, so it must be checked before anything reads its flags.
    """
    if not MetricsRegistry.is_registered(name):
        return REASON_UNREGISTERED
    if MetricsRegistry.requires_reference(name) and not has_reference:
        return REASON_NO_REFERENCE
    # A MetricContext-consuming metric needs k-space, masks or coil maps that
    # only the training data pipeline assembles. The inference path builds none
    # of it, so calling the metric would either raise or -- worse -- silently
    # score against a default-constructed context (pitfall #18).
    if MetricsRegistry.needs_context(name) or MetricsRegistry.needs(name):
        return REASON_NEEDS_CONTEXT
    return None


class InferenceEvaluator:
    """Accumulate per-case metric values, then write the report artifacts.

    Mirrors ``ReportCaseRecorder``'s ``observe`` / ``write`` shape on purpose:
    the two are used side by side on the inference path and a caller should not
    have to remember which one is which.
    """

    def __init__(self, declared: list[str], *, device: str = "cpu") -> None:
        self._declared = list(dict.fromkeys(declared))  # de-dupe, keep order
        self._device = device
        self._values: dict[str, dict[str, float]] = {}
        self._failures: dict[str, str] = {}
        self._skips: dict[str, str] = {}
        self._saw_reference = False
        self._cases = 0

    @property
    def declared(self) -> list[str]:
        return list(self._declared)

    def observe(
        self,
        *,
        case_id: str,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> None:
        """Score one case, recording per-metric failures rather than raising.

        A metric that blows up on one subject must not abort a batch verb, but it
        must not vanish either -- the exception text is kept and surfaces in the
        manifest, because "this metric raised" and "this metric was not requested"
        are different states that an absent CSV column cannot tell apart.
        """
        self._cases += 1
        # The caller hands us whatever device its pipeline left the tensors on
        # (``infer`` moves outputs to CPU for saving). Aligning here rather than
        # at each call site keeps a device mismatch from surfacing as a per-metric
        # "failed" record that looks like a metric defect.
        prediction = prediction.to(self._device)
        if target is not None:
            target = target.to(self._device)
        has_ref = target is not None
        self._saw_reference = self._saw_reference or has_ref
        for name in self._declared:
            reason = classify_metric(name, has_reference=has_ref)
            if reason is not None:
                self._skips.setdefault(name, reason)
                continue
            try:
                metric = MetricsRegistry.get(name, device=self._device)
                raw = metric(prediction) if target is None else metric(prediction, target)
                # Terminal artifact write, not the training loop -- the
                # device sync non-negotiable 9 forbids is scoped to the hot
                # loop, and there is nothing left to overlap with here.
                value = float(raw.detach().cpu()) if torch.is_tensor(raw) else float(raw)
            except Exception as exc:  # recorded below, never swallowed
                self._failures.setdefault(name, f"{type(exc).__name__}: {exc}")
                continue
            self._values.setdefault(name, {})[case_id] = value

    def outcomes(self) -> list[MetricOutcome]:
        """Every declared metric, in declaration order, with what became of it."""
        out: list[MetricOutcome] = []
        for name in self._declared:
            if name in self._values:
                out.append(MetricOutcome(name, STATUS_COMPUTED, values=dict(self._values[name])))
            elif name in self._failures:
                out.append(MetricOutcome(name, STATUS_FAILED, reason=self._failures[name]))
            elif name in self._skips:
                out.append(MetricOutcome(name, STATUS_SKIPPED, reason=self._skips[name]))
            else:
                # Reachable only when no case was ever observed: the metric was
                # never classified because nothing was scored. Saying "no cases"
                # beats implying the metric itself was at fault.
                out.append(MetricOutcome(name, STATUS_SKIPPED, reason="no_cases_observed"))
        return out

    def write(self, run_dir: str | Path) -> dict[str, Path]:
        """Write ``final_eval.json``, its manifest and the long-form CSV."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        outcomes = self.outcomes()

        # final_eval.json -- floats only, per the aggregator's float() coercion.
        # Written even when empty: see the module docstring.
        eval_payload = {o.name: o.values for o in outcomes if o.status == STATUS_COMPUTED}
        eval_path = run_dir / FINAL_EVAL_JSON
        eval_path.write_text(json.dumps(eval_payload, indent=2, sort_keys=True))

        manifest_path = run_dir / FINAL_EVAL_MANIFEST_JSON
        manifest_path.write_text(
            json.dumps(
                {
                    "cases_observed": self._cases,
                    "reference_available": self._saw_reference,
                    "declared": self._declared,
                    "metrics": [
                        {"name": o.name, "status": o.status, "reason": o.reason} for o in outcomes
                    ],
                },
                indent=2,
            )
        )

        csv_path = run_dir / METRICS_CSV
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "subject_id", "value", "status", "reason"])
            for o in outcomes:
                if o.status == STATUS_COMPUTED:
                    for subject_id, value in sorted((o.values or {}).items()):
                        writer.writerow([o.name, subject_id, value, o.status, ""])
                else:
                    writer.writerow([o.name, "", "", o.status, o.reason or ""])

        n_ok = sum(1 for o in outcomes if o.status == STATUS_COMPUTED)
        logger.info(
            "[Report] Inference evaluation: %d/%d declared metric(s) computed over "
            "%d case(s); the rest are recorded with a reason in %s.",
            n_ok,
            len(outcomes),
            self._cases,
            FINAL_EVAL_MANIFEST_JSON,
        )
        return {
            "final_eval": eval_path,
            "manifest": manifest_path,
            "metrics_csv": csv_path,
        }


def write_inference_run_summary(
    run_dir: str | Path,
    *,
    model: torch.nn.Module | None = None,
    duration_sec: float | None = None,
    effective_batch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the ``run_summary.json`` facts a prediction run can honestly claim.

    ``aggregator._RUN_SUMMARY_FACTS`` folds exactly four keys into the tidy frame
    (``model_params``, ``iterations_per_sec``, ``duration_sec``,
    ``effective_batch``), which is what feeds ``fig_1_16_run_summary_card`` and
    ``fig_1_15_computational_profile``.

    ``iterations_per_sec`` is deliberately **not** written. On a prediction run
    the available rate is files-per-second, which is a different quantity from
    the training iteration rate the card labels it as; emitting it would put a
    number that is not wrong-looking under a label that makes it wrong. Its
    absence is handled -- the flattener skips non-numeric keys -- so the card
    simply omits that row rather than showing a fabricated one.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = dict(extra or {})
    if model is not None:
        summary["model_params"] = sum(p.numel() for p in model.parameters())
    if duration_sec is not None:
        summary["duration_sec"] = float(duration_sec)
    if effective_batch is not None:
        summary["effective_batch"] = int(effective_batch)
    path = run_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return path
