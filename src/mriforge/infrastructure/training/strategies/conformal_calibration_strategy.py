"""Conformal Calibration Strategy (PR-3 / H1 + ULF-PR-20..24).

This strategy does NOT train a model. It runs the relevant
calibration/certification primitive (``ConformalCalibrator``,
``CalibratedHallucinationDetector``, ``PathologyRecallCertificate``,
``PACBayesCertificate``, ``DiceRiskCertificate``) on the held-out
calibration split, and writes the resulting JSON artefact to
``certification.<name>.output_artefact``.

The model to certify is the generator wired onto ``self.env``. For the
``dice_risk`` certificate this MUST be a trained network: the branch loads
``certification.dice_risk.checkpoint_path`` into the generator and RAISES if
it is unset — certifying a randomly-initialised model is scientifically
meaningless (pitfall #20). The other certificate branches assume the operator
restored weights via the trainer's ``--resume`` path.

The strategy is dispatched by the ``training.strategy_class`` field
(directly importable path) rather than by ``training_mode``; it
therefore plays nicely with the existing factory.

When multiple ``certification.*`` blocks have ``enabled=True``, every
matching certificate is computed in one pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from mriforge.infrastructure.training.step_io import accepts_step_io

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


def _split_calibration_test(
    batches: list[Any],
) -> tuple[list[Any], list[Any] | None]:
    """Split a materialised calibration cohort into disjoint calibrate/test halves.

    Empirical coverage is only valid on data **disjoint** from the calibration
    fit, so the conformal runner needs a held-out test split. Fewer than two
    batches cannot be split — returns ``(all, None)`` and coverage stays ``None``.
    """
    if len(batches) < 2:
        return batches, None
    mid = len(batches) // 2
    return batches[:mid], batches[mid:]


def _calibration_input_target(batch: Any) -> tuple[Any, Any]:
    """Unpack a calibration batch to ``(model_input, target)`` for the runner.

    Handles dict batches (``input`` / ``target``) and 2-tuples. The runner runs
    the model on ``model_input`` itself, so this returns the INPUT, not a
    prediction (distinct from ``_unpack_calibration_pair``, which returns a
    prediction for the dice/CHD paths). The runner's default unpacker only
    accepts 2-tuples, so dict batches would otherwise raise.
    """
    if isinstance(batch, dict):
        return batch.get("input"), batch.get("target")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(
        "conformal calibration batch must be a dict with input/target or a "
        f"(input, target) tuple; got {type(batch).__name__}"
    )


class ConformalCalibrationStrategy(ReconstructionTrainingStrategy):
    """Calibration-only strategy. Does not perform gradient updates.

    Subclassing ``ReconstructionTrainingStrategy`` lets us reuse the loss
    plumbing, FFT transformer, and DC layer without re-implementing
    them, even though we never call ``optimizer.step()``.
    """

    #: Canonical opt-in certificate keys, in evaluation order. Single source of
    #: truth for both ``__init__`` and ``_collect_enabled_certs`` so a test can
    #: drive the SAME collection logic the strategy uses (no copied loop).
    _CERT_KEYS: tuple[str, ...] = (
        "conformal",
        "chd",
        "pathology_recall",
        "pac_bayes",
        "dice_risk",
        "badge",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # All certificates are opt-in — collect those flagged enabled.
        cert = getattr(self.config, "certification", None)
        self._enabled_certs: list[str] = self._collect_enabled_certs(cert)
        logger.info(
            "ConformalCalibrationStrategy initialised. Enabled certificates: %s",
            self._enabled_certs or "<none — strategy is a no-op>",
        )

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample target field + contrast into ``batch_context``.

        The pipeline's gated validation seam forwards a batch key ONLY when its name
        literally appears in this signature (``_VALIDATION_FORWARD_FIELDS`` in
        ``pipelines/train.py``). Without this override the seam dropped both keys, so
        ``_prepare_generator_inputs`` resolved ``field_strength = None`` and called the
        renderer bare — and ``AnatomyFieldRenderer.forward`` takes ``field_strength`` as
        a REQUIRED keyword-only argument. Every validation batch raised TypeError, the
        loop hit "Validation produced zero successful batches", and the run died before
        ``_maybe_run_calibration`` ever ran: the b17 Dice-risk RCPS certificate, which is
        this arm's entire deliverable, was never written. Mirrors the override its
        CrossFieldTranslationStrategy sibling already carries.
        """
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    @classmethod
    def _collect_enabled_certs(cls, cert: Any) -> list[str]:
        """Return the enabled certificate keys from a ``certification`` config.

        Extracted so the unit test exercises the real collection logic instead
        of a hand-copied duplicate (the copied-loop test could not detect a
        certificate being dropped from ``_CERT_KEYS``)."""
        if cert is None:
            return []
        return [
            letter
            for letter in cls._CERT_KEYS
            if getattr(cert, letter, None) is not None
            and getattr(getattr(cert, letter), "enabled", False)
        ]

    # Override training step — calibration takes no gradient steps.
    @accepts_step_io
    def train_step(
        self, batch: Any, epoch: int = 0, step: int = 0, **kwargs: Any
    ) -> list:  # pragma: no cover - exercised in integration tests
        """No-op train step (no optimizer steps); certificates run in
        ``run_calibration``. Returns an EMPTY step-config list — the old dict
        return hit KeyError('optimizer') in Trainer.execute_step."""
        return []

    def run_calibration(self, dataloader: Any) -> dict[str, dict[str, Any]]:
        """Iterate the calibration loader once and emit the requested certificates.

        Returns a dict mapping certificate-letter to the report dict.
        Side-effect: writes each report to its declared ``output_artefact``.
        """
        cert_cfg = getattr(self.config, "certification", None)
        if cert_cfg is None:
            return {}

        # Lazy-import primitives so the strategy class is still importable when
        # downstream models / packages are missing.
        from mriforge.infrastructure.calibration import (
            CalibratedHallucinationDetector,
            ConformalCalibrationRunner,
            ConformalCalibrator,
            PACBayesCertificate,
            PathologyRecallCertificate,
            resolve_calibration_score,
        )
        from mriforge.infrastructure.reporting.validation_badge import ValidationBadge

        reports: dict[str, dict[str, Any]] = {}
        model = getattr(self.env, "generator", None)
        if model is None:
            logger.warning("No generator on context — cannot run calibration.")
            return reports

        # Materialise the calibration cohort once: (a) it is re-iterable across
        # every certificate below (a one-shot generator would be exhausted by
        # the first cert), and (b) the conformal certificate splits it into a
        # disjoint calibrate/test half so empirical coverage can actually be
        # measured. The calibration split is held-out and small by design.
        batches: list[Any] = list(dataloader)

        # Conformal / R1 — dispatch the score function through the
        # calibration-score registry. score_kwargs (e.g. marker_basis_path
        # for the vf_residual score) are pulled from the conformal block.
        if "conformal" in self._enabled_certs:
            raw_score = resolve_calibration_score(cert_cfg.conformal.score_fn)
            score_kwargs: dict[str, Any] = {}
            if getattr(cert_cfg.conformal, "marker_basis_path", None) is not None:
                score_kwargs["marker_basis_path"] = cert_cfg.conformal.marker_basis_path

            def _bound_score(pred, target, _fn=raw_score, _kw=score_kwargs):
                return _fn(pred, target, **_kw)

            calibrator = ConformalCalibrator(score_fn=_bound_score, alpha=cert_cfg.conformal.alpha)
            runner = ConformalCalibrationRunner(
                model=model,
                calibrator=calibrator,
                device=self.device,
                unpack_batch=_calibration_input_target,
            )
            # Hold out a disjoint test split so empirical_coverage is actually
            # MEASURED, not left None. A coverage-certifying arm that never
            # computes coverage is pitfall #18 (metric <-> claim mismatch): the
            # arm grades on empirical_coverage, so it must be produced.
            calib_batches, test_batches = _split_calibration_test(batches)
            if test_batches is None:
                logger.warning(
                    "conformal calibration: only %d batch(es) — too few to hold "
                    "out a test split; empirical_coverage will be None.",
                    len(batches),
                )
            report = runner.run(calib_batches, test_batches)
            payload = (
                report.to_dict()
                if hasattr(report, "to_dict")
                else {
                    "quantile": report.quantile,
                    "n_calibration": report.n_calibration,
                    "alpha": report.alpha,
                    "empirical_coverage": report.empirical_coverage,
                    "avg_set_size": report.avg_set_size,
                    "score_fn": cert_cfg.conformal.score_fn,
                }
            )
            reports["r1_cdfc"] = payload
            self._write_artefact(cert_cfg.conformal.output_artefact, payload)

        # CHD / R2
        if "chd" in self._enabled_certs:
            detector = CalibratedHallucinationDetector(
                beta=cert_cfg.chd.beta, delta=cert_cfg.chd.delta
            )
            report = detector.calibrate(model, batches, device=self.device)
            reports["r2_chd"] = report.to_dict() if hasattr(report, "to_dict") else report
            self._write_artefact(cert_cfg.chd.output_artefact, reports["r2_chd"])

        # Pathology recall / R4
        if "pathology_recall" in self._enabled_certs:
            cert = PathologyRecallCertificate(
                classes=cert_cfg.pathology_recall.classes,
                delta=cert_cfg.pathology_recall.delta,
                target_recall=cert_cfg.pathology_recall.target_recall,
            )
            report = cert.evaluate(model, batches, device=self.device)
            reports["r4_prc"] = report.to_dict() if hasattr(report, "to_dict") else report
            self._write_artefact(cert_cfg.pathology_recall.output_artefact, reports["r4_prc"])

        # PAC-Bayes / R5
        if "pac_bayes" in self._enabled_certs:
            cert = PACBayesCertificate(
                delta=cert_cfg.pac_bayes.delta,
                tv_tolerance_tau=cert_cfg.pac_bayes.tv_tolerance_tau,
                loss_bound_M=cert_cfg.pac_bayes.loss_bound_M,
            )
            report = cert.evaluate(model, batches, device=self.device)
            reports["r5_pac_bayes"] = report.to_dict() if hasattr(report, "to_dict") else report
            self._write_artefact(cert_cfg.pac_bayes.output_artefact, reports["r5_pac_bayes"])

        # Dice-risk RCPS / B-1.7 — segment pred+target over the calibration cohort,
        # certify a Hoeffding upper bound on the expected Dice-risk (1 - Dice).
        if "dice_risk" in self._enabled_certs:
            import torch

            from mriforge.core.metrics.quantitative.segmentation import (
                dice_score,
                get_segmenter,
            )
            from mriforge.infrastructure.calibration import DiceRiskCertificate

            dr = cert_cfg.dice_risk
            # ANTI-#20: a Dice-risk certificate over an untrained model is
            # meaningless. Require an explicit checkpoint and load it into the
            # generator before certifying — never silently certify random weights.
            ckpt_path = getattr(dr, "checkpoint_path", None)
            if not ckpt_path:
                raise ValueError(
                    "certification.dice_risk.checkpoint_path is required: certifying "
                    "the Dice-risk of an uninitialised/untrained synthesiser is "
                    "scientifically meaningless (pitfall #20). Point it at the trained "
                    "synthesis arm's <output_dir>/checkpoints/best.pt."
                )
            self._load_generator_checkpoint(model, ckpt_path)
            segmenter = get_segmenter(dr.segmenter_backend, n_classes=dr.n_classes)
            risks: list[float] = []
            was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                for batch in batches:
                    pred, target = self._unpack_calibration_pair(batch, model)
                    if pred is None or target is None:
                        continue
                    seg_p = segmenter.segment(pred)
                    seg_t = segmenter.segment(target)
                    d = dice_score(seg_p, seg_t, segmenter.n_classes)
                    risks.extend((1.0 - d).reshape(-1).tolist())
            if was_training and hasattr(model, "train"):
                model.train()
            n = len(risks)
            mean_risk = (sum(risks) / n) if n else 1.0
            report = DiceRiskCertificate(alpha=dr.alpha, delta=dr.delta).certify(
                mean_risk, max(n, 1)
            )
            payload = report.to_dict()
            # Provenance (#15): record WHICH checkpoint was certified so the verdict
            # is attributable and reproducible.
            payload["checkpoint_path"] = str(ckpt_path)
            payload["segmenter_backend"] = dr.segmenter_backend
            reports["r6_dice_risk"] = payload
            self._write_artefact(dr.output_artefact, payload)

        # V.3 — bundles the upstream JSONs into a single ValidationBadge artefact.
        if "badge" in self._enabled_certs:
            badge = ValidationBadge.from_certificate_paths(
                cert_cfg.badge.upstream_certificate_artefacts,
                required_gates=cert_cfg.badge.gates,
            )
            reports["v3_badge"] = badge.to_dict()
            self._write_artefact(cert_cfg.badge.output_path, reports["v3_badge"])

        return reports

    @staticmethod
    def _unpack_calibration_pair(batch: Any, model: Any) -> tuple[Any, Any]:
        """Return ``(prediction, target)`` from a calibration batch (defensive).

        Accepts a dict batch (``input``/``target``) or a 2-tuple ``(input, target)``;
        runs ``model(input)`` and unwraps tuple/dict outputs. Returns ``(None, None)``
        when the batch shape is unrecognised (skipped, not silently mis-scored).
        """
        if isinstance(batch, dict):
            x = batch.get("input")
            target = batch.get("target")
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, target = batch[0], batch[1]
        else:
            return None, None
        if x is None or target is None:
            return None, None
        # Field-conditioned generators (AnatomyFieldRenderer) REQUIRE a field_strength
        # kwarg; detect it by signature (no except-TypeError retry, which would mask a
        # real error) and pass the batch's target field (or source as fallback).
        import inspect

        fwd = getattr(model, "forward", model)
        fwd_kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(fwd).parameters
        except (TypeError, ValueError):
            params = {}
        if "field_strength" in params:
            fs = None
            if isinstance(batch, dict):
                fs = batch.get("field_strength_target", batch.get("field_strength"))
            if fs is not None:
                fwd_kwargs["field_strength"] = fs
        pred = model(x, **fwd_kwargs)
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, dict):
            pred = pred.get("reconstruction", next(iter(pred.values())))
        return pred, target

    @staticmethod
    def _load_generator_checkpoint(model: Any, path: str) -> None:
        """Load a trained generator state-dict into ``model`` in place.

        Handles the checkpoint key formats the trainer/EMA writers emit
        (``generator_state_dict`` / ``model_state_dict`` / ``ema_state_dict`` /
        ``state_dict``) and a raw tensor state-dict. RAISES on a missing file,
        an unrecognised format, or a model with no ``load_state_dict`` — never
        silently leaves the model at its random init (pitfall #9/#20).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"dice_risk checkpoint not found: {path}. Train the synthesis arm "
                "first (its <output_dir>/checkpoints/best.pt) before certifying."
            )
        if not hasattr(model, "load_state_dict"):
            raise TypeError(
                f"Cannot load a checkpoint into {type(model).__name__}: no "
                "load_state_dict. The generator must be an nn.Module to be certified."
            )
        ckpt = torch.load(p, map_location="cpu")
        state = None
        if isinstance(ckpt, dict):
            for key in (
                "generator_state_dict",
                "model_state_dict",
                "ema_state_dict",
                "state_dict",
            ):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            if state is None and all(torch.is_tensor(v) for v in ckpt.values()):
                state = ckpt  # raw {param_name: tensor}
        if state is None:
            raise ValueError(
                f"Unrecognised checkpoint format at {path}: expected a dict with "
                "generator_state_dict/model_state_dict/ema_state_dict/state_dict or a "
                "raw tensor state-dict."
            )
        model.load_state_dict(state)
        logger.info("dice_risk: loaded synthesiser checkpoint %s", path)

    @staticmethod
    def _write_artefact(path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote calibration artefact to %s", out)


class ValidationBadgeStrategy(ConformalCalibrationStrategy):
    """Alias subclass — exists so YAMLs that point at a separate
    ``validation_badge_strategy.ValidationBadgeStrategy`` class still
    resolve. Behaviour is identical to ``ConformalCalibrationStrategy``;
    the badge is rolled in via the same ``run_calibration`` call.
    """
