"""Regression test for F5a (smoke_audit_20260521) — phase preservation
on multi-coil complex tensors in the diffusion validation path.

Root cause: ``diffusion.py::_log_validation_images_to_tensorboard``
previously did an unconditional ``.abs()`` on complex targets when
``is_image_domain=True``. For diffusion YAMLs with
``data.coil_processing_mode='none'``, the target reached this branch
as **complex multi-coil k-space** (the data builder skipped coil
combine), and the phase-strip produced Hermitian-symmetric magnitude
→ centro-symmetric ("doubled-brain") validation images in 80 / 103
experiments.

This test pins:

1. **Multi-coil complex tensors are NOT phase-stripped.** The fix
   guards ``.abs()`` with ``shape[1] <= 2`` so single-coil complex
   (genuinely image-domain) still gets magnitude-converted, but
   multi-coil complex (probably k-space mis-routed through here)
   stays complex for ``kspace_to_image`` to handle correctly.

2. **The single-coil image case still works** — backward-compatible.

If the next contributor reverts the guard to the unconditional
``targets.abs()``, this test fails BEFORE the cluster smoke produces
80 doubled-brain experiments.

See ``TODO/audit/smoke_audit_20260521.md §F5a`` for the full
diagnosis + the cohort list.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIFFUSION = REPO / "src" / "spectramr" / "infrastructure" / "training" / "strategies" / "diffusion.py"


def _read() -> str:
    return DIFFUSION.read_text()


def test_phase_strip_is_shape_guarded() -> None:
    """The ``targets.abs()`` call site must be guarded by a multi-coil
    check. Without this guard, ``coil_processing_mode='none'`` configs
    produce centro-symmetric (doubled-brain) validation images.

    We match the exact post-fix call site: complex check AND a
    rank/channel guard. The previous bug-prone form was an
    unconditional ``targets = targets.abs()``.
    """
    text = _read()
    # The fix introduces a shape-guarded ``targets.abs()``. The minimum
    # required test: at the location where targets were previously
    # abs'd unconditionally, there must now be a guard that excludes
    # multi-coil cases.
    pattern = re.compile(
        r"if\s+torch\.is_complex\(targets\)\s*:\s*\n"
        r"\s*if\s+targets\.dim\(\)\s*<\s*4\s+or\s+targets\.shape\[1\]\s*<=\s*2\s*:\s*\n"
        r"\s*targets\s*=\s*targets\.abs\(\)",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "diffusion.py must guard ``targets.abs()`` with "
        "``targets.dim() < 4 or targets.shape[1] <= 2`` so multi-coil "
        "complex tensors (e.g. from coil_processing_mode='none' "
        "configs) don't get phase-stripped into Hermitian-symmetric "
        "magnitude. See TODO/audit/smoke_audit_20260521.md §F5a."
    )


def test_predictions_also_phase_guarded() -> None:
    """Predictions get the same guard as targets — the fix must be
    symmetric. A revert of one side would silently re-introduce the
    bug for FAKE images (E24a cohort, 26 / 80 experiments)."""
    text = _read()
    pattern = re.compile(
        r"if\s+torch\.is_complex\(predictions\)\s*:\s*\n"
        r"\s*if\s+predictions\.dim\(\)\s*<\s*4\s+or\s+predictions\.shape\[1\]\s*<=\s*2\s*:\s*\n"
        r"\s*predictions\s*=\s*predictions\.abs\(\)",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "diffusion.py must guard ``predictions.abs()`` with "
        "``predictions.dim() < 4 or predictions.shape[1] <= 2`` for "
        "the same reason ``targets.abs()`` is guarded (E24a — "
        "model output inheriting the corruption)."
    )


def test_metric_target_abs_is_shape_guarded_f5b() -> None:
    """F5b (smoke_audit_20260524): the *upstream* ``target_for_metrics``
    abs() — the one feeding ``_apply_metric_transforms`` /
    ``ifft_magnitude`` and then the validation mosaic when
    ``compute_image_metrics`` is on — must carry the SAME multi-coil
    guard as the F5a logger site.

    Without it, a multi-coil complex k-space target (e.g.
    ``coil_processing_mode='none'``) is phase-stripped to ``|kspace|``
    before ``ifft2c``; the IFFT of a real Hermitian signal is
    centro-symmetric — the "doubled brain" real-image regression the
    user flagged in the 2026-05-24 mosaic review. A revert to the
    unconditional ``target_for_metrics = torch.abs(...)`` re-opens that
    path, so this test fails first.
    """
    text = _read()
    pattern = re.compile(
        r"if\s+torch\.is_complex\(target_for_metrics\)\s*:\s*\n"
        r"\s*if\s+target_for_metrics\.dim\(\)\s*<\s*4\s+or\s+"
        r"target_for_metrics\.shape\[1\]\s*<=\s*2\s*:\s*\n"
        r"\s*target_for_metrics\s*=\s*torch\.abs\(target_for_metrics\)",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        "diffusion.py must guard the upstream ``target_for_metrics."
        "abs()`` with ``target_for_metrics.dim() < 4 or "
        "target_for_metrics.shape[1] <= 2`` (F5b) so multi-coil complex "
        "k-space targets aren't phase-stripped before ifft_magnitude. "
        "See TODO/audit/smoke_audit_20260524.md §F5b."
    )


def test_f5b_audit_anchor_present() -> None:
    """The F5b guard carries its own dated anchor so a future cleanup
    can trace it to the 2026-05-24 audit doc rather than reading it as a
    redundant copy of F5a and deleting it."""
    text = _read()
    assert "F5b (2026-05-24" in text, (
        "F5b fix must carry a dated comment anchor (``F5b (2026-05-24 ``)."
    )
    assert "smoke_audit_20260524.md" in text, (
        "F5b fix must reference TODO/audit/smoke_audit_20260524.md."
    )


def test_audit_anchor_present() -> None:
    """The F5a fix carries a comment anchor (``F5a (2026-05-21``) and
    references the smoke-audit doc by name. This is the breadcrumb a
    future contributor follows to understand why the guard exists —
    without it, the guard reads as a redundant defense and is at risk
    of being removed in a cleanup pass.
    """
    text = _read()
    assert "F5a (2026-05-21" in text, (
        "F5a fix must carry a dated comment anchor (``F5a (2026-05-21 ``) "
        "so future cleanups can trace it back to the audit doc."
    )
    assert "smoke_audit_20260521.md" in text, (
        "F5a fix must reference TODO/audit/smoke_audit_20260521.md "
        "for the full diagnosis. Otherwise the guard reads as "
        "redundant and gets removed in a cleanup pass."
    )
