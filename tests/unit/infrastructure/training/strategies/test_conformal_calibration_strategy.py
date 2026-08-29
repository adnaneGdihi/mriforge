"""Tests for the Dice-risk RCPS branch of ConformalCalibrationStrategy (B-1.7)."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from mriforge.infrastructure.training.strategies.conformal_calibration_strategy import (
    ConformalCalibrationStrategy,
)


class _Identity(nn.Module):
    """Trained-model stand-in: pred = input, and load_state_dict({}) succeeds (no
    params), so a checkpoint round-trips. Lets the math tests exercise the REAL
    checkpoint-gated branch instead of bypassing it."""

    def forward(self, x, **_):
        return x


def _write_ckpt(tmp_path, model: nn.Module) -> str:
    p = tmp_path / "best.pt"
    torch.save({"generator_state_dict": model.state_dict()}, p)
    return str(p)


def _bare_strategy(dice_cfg, model: nn.Module | None = None) -> ConformalCalibrationStrategy:
    strat = object.__new__(ConformalCalibrationStrategy)
    strat.device = torch.device("cpu")
    strat._enabled_certs = ["dice_risk"]
    strat.config = types.SimpleNamespace(
        certification=types.SimpleNamespace(dice_risk=dice_cfg)
    )
    strat.env = types.SimpleNamespace(generator=model if model is not None else _Identity())
    return strat


def _cfg(**over):
    base = dict(
        enabled=True,
        alpha=0.3,
        delta=0.05,
        segmenter_backend="label_dice",
        n_classes=5,
        checkpoint_path=None,
        output_artefact=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_dice_risk_branch_certifies_perfect_reconstruction(tmp_path) -> None:
    # input == target with an identity model -> Dice ~1 -> risk ~0. A realistic
    # cohort (n=32) makes the Hoeffding gap small enough to certify at alpha=0.3.
    model = _Identity()
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path=_write_ckpt(tmp_path, model)), model)
    loader = [
        {"input": (x := torch.rand(4, 1, 16, 16)), "target": x} for _ in range(8)
    ]
    reports = strat.run_calibration(loader)
    assert "r6_dice_risk" in reports
    rep = reports["r6_dice_risk"]
    assert rep["empirical_risk"] < 0.05
    assert rep["n"] == 32
    assert rep["is_certified"] is True
    # Provenance (#15): the certified checkpoint is stamped into the report.
    assert rep["checkpoint_path"].endswith("best.pt")


def test_dice_risk_branch_rejects_when_budget_too_tight(tmp_path) -> None:
    # Mismatched input/target -> nonzero Dice-risk; a tiny alpha cannot be certified.
    model = _Identity()
    strat = _bare_strategy(_cfg(alpha=0.01, checkpoint_path=_write_ckpt(tmp_path, model)), model)
    loader = [
        {"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}
        for _ in range(3)
    ]
    reports = strat.run_calibration(loader)
    assert reports["r6_dice_risk"]["is_certified"] is False


def test_dice_risk_raises_without_checkpoint() -> None:
    # ANTI-#20: certifying an untrained model is meaningless -> run_calibration must
    # raise when no checkpoint_path is declared (NOT silently certify random weights).
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path=None))
    loader = [{"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}]
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        strat.run_calibration(loader)


def test_dice_risk_raises_on_missing_checkpoint_file() -> None:
    # A declared-but-absent checkpoint must fail loudly, not fall back to random init.
    strat = _bare_strategy(_cfg(alpha=0.3, checkpoint_path="/nonexistent/best.pt"))
    loader = [{"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}]
    with pytest.raises(FileNotFoundError):
        strat.run_calibration(loader)


def test_dice_risk_enabled_collection() -> None:
    # Drive the REAL collection logic the strategy uses (not a hand-copied loop):
    # if 'dice_risk' were dropped from _CERT_KEYS this test would now fail.
    cert = types.SimpleNamespace(
        dice_risk=_cfg(enabled=True),
        conformal=types.SimpleNamespace(enabled=False),
        chd=types.SimpleNamespace(enabled=False),
        pathology_recall=types.SimpleNamespace(enabled=False),
        pac_bayes=types.SimpleNamespace(enabled=False),
        badge=types.SimpleNamespace(enabled=False),
    )
    enabled = ConformalCalibrationStrategy._collect_enabled_certs(cert)
    assert enabled == ["dice_risk"]
    # The canonical key set still advertises dice_risk.
    assert "dice_risk" in ConformalCalibrationStrategy._CERT_KEYS


# ---------------------------------------------------------------------------
# Empirical-coverage fix (#18): the conformal certificate must MEASURE coverage
# on a held-out test split, not leave it None.
# ---------------------------------------------------------------------------


def test_split_calibration_test_disjoint_halves() -> None:
    from mriforge.infrastructure.training.strategies.conformal_calibration_strategy import (
        _split_calibration_test,
    )

    calib, test = _split_calibration_test([1, 2, 3, 4])
    assert calib == [1, 2]
    assert test == [3, 4]
    assert set(calib).isdisjoint(set(test))


def test_split_calibration_test_too_few_to_hold_out() -> None:
    from mriforge.infrastructure.training.strategies.conformal_calibration_strategy import (
        _split_calibration_test,
    )

    calib, test = _split_calibration_test([1])
    assert calib == [1]
    assert test is None


def test_calibration_input_target_unpacks_dict_and_tuple() -> None:
    from mriforge.infrastructure.training.strategies.conformal_calibration_strategy import (
        _calibration_input_target,
    )

    assert _calibration_input_target({"input": "X", "target": "Y"}) == ("X", "Y")
    assert _calibration_input_target(("X", "Y")) == ("X", "Y")
    with pytest.raises(ValueError, match="dict with input/target"):
        _calibration_input_target(object())


def test_runner_measures_coverage_only_with_test_split() -> None:
    """End-to-end: empirical_coverage is None without a test loader (the old
    bug) and a real number once a disjoint test split is supplied."""
    from mriforge.infrastructure.calibration import (
        ConformalCalibrationRunner,
        ConformalCalibrator,
    )
    from mriforge.infrastructure.training.strategies.conformal_calibration_strategy import (
        _calibration_input_target,
        _split_calibration_test,
    )

    torch.manual_seed(0)

    def score(pred, target):
        return (pred - target).abs().flatten(1).amax(1)

    batches = [
        {"input": torch.randn(2, 1, 8, 8), "target": torch.randn(2, 1, 8, 8)}
        for _ in range(6)
    ]
    calib, test = _split_calibration_test(batches)
    runner = ConformalCalibrationRunner(
        model=nn.Identity(),
        calibrator=ConformalCalibrator(score_fn=score, alpha=0.1),
        device="cpu",
        unpack_batch=_calibration_input_target,
    )
    assert runner.run(calib).empirical_coverage is None
    cov = runner.run(calib, test).empirical_coverage
    assert cov is not None and 0.0 <= cov <= 1.0


def test_run_calibration_passes_held_out_test_split() -> None:
    """Source-pin: the conformal certificate runs the calibrator on a held-out
    split so empirical_coverage is produced (guards against a regression back to
    the single-arg ``runner.run(dataloader)`` that left coverage None)."""
    import pathlib

    src = pathlib.Path(
        "src/mriforge/infrastructure/training/strategies/conformal_calibration_strategy.py"
    ).read_text(encoding="utf-8")
    assert "runner.run(calib_batches, test_batches)" in src
    assert "_split_calibration_test(batches)" in src
