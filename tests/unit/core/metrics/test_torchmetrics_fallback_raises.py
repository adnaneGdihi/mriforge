"""Regression: torchmetrics-backed metrics must fail loud, not fabricate 0.0.

Review finding (2026-07-01): when ``torchmetrics`` cannot be imported
(undeclared dependency, or a ``huggingface-hub>=1.0`` version conflict that
breaks its import), LPIPS / FID / KID / MS-SSIM / UQI silently returned
``torch.tensor(0.0)`` — a value indistinguishable from a real perfect score.
Several arms set ``primary_metric: lpips``; a constant 0.0 makes the plateau
monitor register one spurious "improvement" then never again, so early stopping
fires at ``patience`` and checkpoint selection is arbitrary (pitfalls #9/#10/#18).

These metrics now raise ``RuntimeError`` (recorded as NaN + a warning by the
``ValidationMetricsComputer`` loop; a hard failure on a direct call) instead of
returning a fabricated 0.0.
"""

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics import evaluation_metrics as em  # noqa: E402


def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment where torchmetrics failed to import."""
    monkeypatch.setattr(em, "TORCHMETRICS_AVAILABLE", False)
    monkeypatch.setattr(
        em,
        "_TORCHMETRICS_IMPORT_ERROR",
        ImportError("forced unavailable for test"),
        raising=False,
    )


def test_lpips_falls_back_to_lpips_pkg_when_torchmetrics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LPIPS (unlike KID/MS-SSIM/UQI) has a real second backend: the `lpips` package
    # (AlexNet, the reference impl). When torchmetrics is missing but `lpips` is present,
    # the metric computes a GENUINE finite value + logs the backend swap once — a stale
    # cluster env gets real numbers, not a run-long NaN.
    pytest.importorskip("lpips")
    _force_unavailable(monkeypatch)
    em.LPIPS._fallback_warned = False  # reset the one-time-warning guard for this test
    metric = em.LPIPS(device="cpu")
    assert metric._backend == "lpips_pkg"
    preds = torch.rand(1, 1, 64, 64)
    target = torch.rand(1, 1, 64, 64)
    out = metric.compute_metric(preds, target)
    assert torch.isfinite(out).all()  # a real number, not NaN and not a raise


def test_lpips_raises_when_both_backends_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construction stays clean (plumbing); the raise is at the compute fabrication point,
    # so callers that inject a backbone still work. With BOTH torchmetrics AND the lpips
    # package absent there is no honest value to report -> raise (recorded as NaN upstream).
    import sys

    _force_unavailable(monkeypatch)
    monkeypatch.setitem(sys.modules, "lpips", None)  # make `import lpips` raise ImportError
    metric = em.LPIPS(device="cpu")
    assert metric._backend is None
    preds = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    with pytest.raises(RuntimeError, match="torchmetrics"):
        metric.compute_metric(preds, target)


def test_kid_finalize_raises_when_torchmetrics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # KID is a summary metric: per-batch accumulation is a legitimate no-op that
    # returns 0.0; the fabrication (and thus the raise) is at finalize/compute().
    _force_unavailable(monkeypatch)
    metric = em.KID(device="cpu")
    assert metric.subset_size == 8  # construction plumbing intact
    with pytest.raises(RuntimeError, match="torchmetrics"):
        metric.compute()


def test_ms_ssim_compute_raises_when_torchmetrics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_unavailable(monkeypatch)
    metric = em.MSSSIM(device="cpu")
    preds = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    with pytest.raises(RuntimeError, match="ms_ssim"):
        metric(preds, target)


def test_uqi_compute_raises_when_torchmetrics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_unavailable(monkeypatch)
    metric = em.UQI(device="cpu")
    preds = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    with pytest.raises(RuntimeError, match="uqi"):
        metric(preds, target)


def test_error_message_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raise must name the fix, not just the symptom."""
    _force_unavailable(monkeypatch)
    metric = em.MSSSIM(device="cpu")
    preds = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    with pytest.raises(RuntimeError) as excinfo:
        metric(preds, target)
    msg = str(excinfo.value)
    assert "fabricated 0.0" in msg
    assert "huggingface-hub" in msg
