"""Tests for the shared validation-metric key resolver.

Targets ``spectramr.infrastructure.services.metric_keys`` -- the single SSOT for
``val_*``-vs-bare / loss-alias / cascade-suffix resolution used by BOTH early
stopping (``pipelines/train.py``) and the loss-schedule controller. The full
alias matrix is exercised by ``tests/unit/pipelines/test_train_light_seams.py``
via the re-export; here we pin the new ``resolve_metric_key`` seam and the
layer-correct import location.
"""

from __future__ import annotations

from spectramr.infrastructure.services.metric_keys import (
    early_stop_monitor_candidates,
    resolve_metric_key,
)


def test_exact_match_is_first_candidate() -> None:
    cands = early_stop_monitor_candidates("val_psnr")
    assert cands[0] == "val_psnr"
    assert "psnr" in cands  # val_ prefix stripped


def test_resolve_prefers_exact_then_alias() -> None:
    # exact present -> exact wins
    assert resolve_metric_key("val_psnr", {"val_psnr", "psnr"}) == "val_psnr"
    # only the bare key present -> alias resolves (the F-EARLYSTOP-FUZZY case)
    assert resolve_metric_key("val_ssim", {"ssim", "psnr"}) == "ssim"


def test_resolve_returns_none_when_absent() -> None:
    # A typo'd / never-emitted metric resolves to None (caller must warn, not
    # silently never-fire).
    assert resolve_metric_key("val_made_up", {"ssim", "psnr"}) is None
    assert resolve_metric_key("val_psnr", None) is None
    assert resolve_metric_key("val_psnr", set()) is None


def test_train_reexport_is_same_object() -> None:
    # Back-compat: pipelines.train must re-export the canonical helper.
    from spectramr.pipelines.train import (
        early_stop_monitor_candidates as reexported,
    )

    assert reexported is early_stop_monitor_candidates
