"""Regression: ablation must read validation metrics from where the training
pipeline actually writes them.

``run_training_pipeline`` writes ``validation_metrics.csv`` to
``<training.output_dir>/logs/`` (``run_dir = config.training.output_dir``).
``train_and_score`` read ``config.logging.log_dir/validation_metrics.csv``
instead — a path that defaults to ``./logs`` independently of
``training.output_dir``. Unless a YAML happened to align the two, the ablation
trained the full variant and then read an empty/absent CSV → empty impact
analysis after burning GPU.
"""

from __future__ import annotations

import types

from tests.utils.config_block_stub import block_stub

from mriforge.pipelines.ablation import _resolve_validation_metrics_csv


def test_resolves_from_training_output_dir(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "validation_metrics.csv").write_text("iteration,psnr\n1,30\n")

    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(output_dir=str(tmp_path)),
        logging=block_stub("logging", log_dir=str(tmp_path / "unrelated")),
    )

    resolved = _resolve_validation_metrics_csv(cfg)
    assert resolved == logs / "validation_metrics.csv"
    assert resolved.exists()


def test_falls_back_to_logging_log_dir(tmp_path):
    # When the canonical location is absent but the legacy logging.log_dir has
    # the file, fall back to it (backward compat).
    legacy = tmp_path / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "validation_metrics.csv").write_text("iteration,psnr\n1,30\n")

    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(output_dir=str(tmp_path / "nope")),
        logging=block_stub("logging", log_dir=str(legacy)),
    )

    resolved = _resolve_validation_metrics_csv(cfg)
    assert resolved == legacy / "validation_metrics.csv"


# ---------------------------------------------------------------------------
# Regression: per-variant output-dir isolation (metrics collision fix).
#
# Before the fix, ``create_ablation_variant`` never repointed
# ``training.output_dir``, so the baseline and every variant trained to and read
# back the SAME ``<output_dir>/logs/validation_metrics.csv``. Each arm clobbered
# the previous arm's CSV and the impact deltas were a meaningless mixture.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from mriforge.config.settings import TrainingSettings  # noqa: E402
from mriforge.pipelines.ablation import (  # noqa: E402
    create_ablation_variant,
    run_ablation_study,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "_fixtures"
    / "minimal_settings"
    / "gan.yaml"
)


def test_variant_repoints_output_dir(tmp_path):
    base = TrainingSettings.from_yaml(str(_FIXTURE))
    variant = create_ablation_variant(
        base, {"overrides": {}}, output_dir=tmp_path / "arm_a"
    )
    assert Path(variant.training.output_dir) == tmp_path / "arm_a"
    # The base config must be untouched (frozen SSOT): building a variant must
    # not mutate the shared baseline's output_dir.
    assert Path(base.training.output_dir) != tmp_path / "arm_a"


def test_study_isolates_each_arm_output_dir(tmp_path):
    """Baseline + every variant must resolve a DISTINCT training.output_dir."""
    seen: list[str] = []

    def fake_eval(config, device):  # noqa: ARG001 - signature parity
        seen.append(str(Path(config.training.output_dir)))
        return {"psnr": 30.0}

    study_dir = tmp_path / "study"
    run_ablation_study(
        config_path=_FIXTURE,
        ablations={
            "lr_low": {"overrides": {"optimization.optimizer.learning_rate": 1e-4}},
            "lr_high": {"overrides": {"optimization.optimizer.learning_rate": 1e-2}},
        },
        output_dir=study_dir,
        evaluation_fn=fake_eval,
        device="cpu",
    )

    # baseline + 2 variants = 3 evaluations, all distinct, all under study_dir.
    assert len(seen) == 3
    assert len(set(seen)) == 3
    for d in seen:
        assert str(study_dir) in d
