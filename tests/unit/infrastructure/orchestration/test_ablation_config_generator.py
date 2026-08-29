"""Unit tests for :mod:`mriforge.infrastructure.orchestration.ablation_config_generator`.

Exercises the pure config-generation logic without touching SLURM or any
real experiment YAML. A minimal nested dict written to a tmp YAML is
used as the base config.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest
import yaml

from mriforge.config.schemas.campaign import AblationAxisSchema
from mriforge.infrastructure.orchestration.ablation_config_generator import (
    AblationConfigGenerator,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _write_base_config(tmp_path: Path) -> Path:
    """Write a minimal nested config YAML and return its path."""
    base_cfg = {
        "metadata": {
            "name": "base_experiment",
            "description": "base description",
            "tags": ["base"],
        },
        # Nested, matching the axis `config_path`s below. The generator walks
        # the RAW dict by dotted path before any validation runs, so the
        # loader's fold never gets a chance to rescue a flat spelling here --
        # it fails with "key 'optimizer' missing at level 1".
        "optimization": {"optimizer": {"learning_rate": 1e-3}},
        "data": {"batch_size": 4},
        "training": {"output_dir": "results/base"},
        "model": {"model_kwargs": {"features": 32}},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False))
    return base_path


def _load_generated(paths: list[str]) -> list[dict]:
    return [yaml.safe_load(Path(p).read_text()) for p in paths]


# ── Combinatorial sweep ─────────────────────────────────────────────


def test_grid_mode_produces_n_times_m_arms(tmp_path: Path) -> None:
    """{lr: [1e-4, 1e-3], batch_size: [2, 4, 8]} → 2 * 3 = 6 arms."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
        AblationAxisSchema(config_path="data.batch_size", values=[2, 4, 8]),
    ]
    out_dir = tmp_path / "out"

    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(out_dir),
    )

    assert len(paths) == 6
    # All generated paths exist on disk.
    for p in paths:
        assert Path(p).exists()
    # No duplicate names (i.e. unique suffixes).
    assert len(set(paths)) == 6


def test_generated_arm_has_swept_fields_overridden(tmp_path: Path) -> None:
    """Each generated config is a valid dict with the swept fields overridden."""
    base = _write_base_config(tmp_path)
    lrs = [1e-4, 1e-3]
    bss = [2, 8]
    axes = [
        AblationAxisSchema(config_path="optimization.optimizer.learning_rate", values=lrs),
        AblationAxisSchema(config_path="data.batch_size", values=bss),
    ]
    out_dir = tmp_path / "out"

    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(out_dir),
    )

    arms = _load_generated(paths)
    # All overridden combos appear exactly once.
    seen = {(a["optimization"]["optimizer"]["learning_rate"], a["data"]["batch_size"]) for a in arms}
    expected = {(lr, bs) for lr in lrs for bs in bss}
    assert seen == expected
    # Untouched fields preserved.
    for arm in arms:
        assert arm["model"]["model_kwargs"]["features"] == 32
        assert "metadata" in arm
        # Metadata name updated, includes 'ablation'.
        assert "ablation" in arm["metadata"]["name"]
        assert "ablation" in arm["metadata"]["description"].lower()


def test_metadata_tags_list_appended(tmp_path: Path) -> None:
    """If metadata.tags is a list, 'ablation' is appended (not replaced)."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
    ]
    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(tmp_path / "out"),
    )
    arms = _load_generated(paths)
    for arm in arms:
        tags = arm["metadata"]["tags"]
        assert isinstance(tags, list)
        assert "ablation" in tags
        assert "base" in tags  # original tag preserved


def test_metadata_tags_dict_extended(tmp_path: Path) -> None:
    """If metadata.tags is a dict, ablation keys are inserted."""
    base_cfg = {
        "metadata": {
            "name": "base_experiment",
            "description": "base",
            "tags": {"arch": "unet"},
        },
        # Nested, matching the axis `config_path`s below. The generator walks
        # the RAW dict by dotted path before any validation runs, so the
        # loader's fold never gets a chance to rescue a flat spelling here --
        # it fails with "key 'optimizer' missing at level 1".
        "optimization": {"optimizer": {"learning_rate": 1e-3}},
    }
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_cfg))
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
    ]
    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(tmp_path / "out"),
    )
    arms = _load_generated(paths)
    for arm in arms:
        tags = arm["metadata"]["tags"]
        assert isinstance(tags, dict)
        assert tags["arch"] == "unet"
        assert tags["ablation"] == "true"
        assert "ablation_index" in tags


def test_training_output_dir_rewritten(tmp_path: Path) -> None:
    """training.output_dir is updated per-arm to the new results location."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
    ]
    out_dir = tmp_path / "out"
    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(out_dir),
    )
    arms = _load_generated(paths)
    for arm in arms:
        out = arm["training"]["output_dir"]
        assert str(out_dir / "results") in out
        # Per-arm leaf is unique.
        assert "ablation" in out


# ── Error / edge cases ──────────────────────────────────────────────


def test_empty_axis_raises_validation_error() -> None:
    """An axis with no values must be rejected with an actionable error."""
    with pytest.raises(pydantic.ValidationError) as ei:
        AblationAxisSchema(config_path="optimization.optimizer.learning_rate", values=[])
    msg = str(ei.value)
    assert "values" in msg
    # Message points at the offending field; not a silent skip.
    assert "at least 2" in msg or "too_short" in msg


def test_single_value_axis_rejected_at_schema_level() -> None:
    """Single-value axis is rejected by AblationAxisSchema (min_length=2).

    The "1 arm per other-axis combination" semantic is unreachable
    via the public schema — defending the actionable-error contract.
    """
    with pytest.raises(pydantic.ValidationError):
        AblationAxisSchema(config_path="data.batch_size", values=[4])


def test_two_value_axis_combined_with_three_value_axis(tmp_path: Path) -> None:
    """Smallest legal cardinality (2) combines correctly with a larger axis."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(config_path="data.batch_size", values=[2, 4]),
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-5, 1e-4, 1e-3]
        ),
    ]
    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(tmp_path / "out"),
    )
    assert len(paths) == 2 * 3


def test_missing_base_config_raises_file_not_found(tmp_path: Path) -> None:
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
    ]
    with pytest.raises(FileNotFoundError):
        AblationConfigGenerator.generate(
            base_config_path=str(tmp_path / "does_not_exist.yaml"),
            axes=axes,
            output_dir=str(tmp_path / "out"),
        )


def test_invalid_axis_path_raises_key_error(tmp_path: Path) -> None:
    """An axis pointing at a missing config path raises KeyError."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="nonexistent.deeply.nested.key", values=[1, 2]
        ),
    ]
    with pytest.raises(KeyError) as ei:
        AblationConfigGenerator.generate(
            base_config_path=str(base),
            axes=axes,
            output_dir=str(tmp_path / "out"),
        )
    # Message names the failing path and surfaces the missing segment.
    msg = str(ei.value)
    assert "nonexistent" in msg


def test_unknown_mode_raises_value_error(tmp_path: Path) -> None:
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
    ]
    with pytest.raises(ValueError, match="Unknown mode"):
        AblationConfigGenerator.generate(
            base_config_path=str(base),
            axes=axes,
            output_dir=str(tmp_path / "out"),
            mode="something_weird",  # type: ignore[arg-type]
        )


# ── Random mode ─────────────────────────────────────────────────────


def test_random_mode_caps_samples_at_grid_size(tmp_path: Path) -> None:
    """random mode never emits more arms than the grid has."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-5, 1e-4]
        ),
        AblationAxisSchema(config_path="data.batch_size", values=[2, 4]),
    ]
    # Grid size is 4; ask for 100.
    paths = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(tmp_path / "out"),
        mode="random",
        n_random_samples=100,
        seed=0,
    )
    assert len(paths) == 4


def test_random_mode_is_seeded(tmp_path: Path) -> None:
    """Same seed → same sampled arm combinations."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-5, 1e-4, 1e-3]
        ),
        AblationAxisSchema(config_path="data.batch_size", values=[2, 4, 8]),
    ]
    out_a = tmp_path / "outA"
    out_b = tmp_path / "outB"
    paths_a = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(out_a),
        mode="random",
        n_random_samples=4,
        seed=123,
    )
    paths_b = AblationConfigGenerator.generate(
        base_config_path=str(base),
        axes=axes,
        output_dir=str(out_b),
        mode="random",
        n_random_samples=4,
        seed=123,
    )
    arms_a = _load_generated(paths_a)
    arms_b = _load_generated(paths_b)
    keys_a = [(c["optimization"]["optimizer"]["learning_rate"], c["data"]["batch_size"]) for c in arms_a]
    keys_b = [(c["optimization"]["optimizer"]["learning_rate"], c["data"]["batch_size"]) for c in arms_b]
    assert keys_a == keys_b


# ── Reproducibility ─────────────────────────────────────────────────


def test_grid_mode_arm_ordering_is_deterministic(tmp_path: Path) -> None:
    """Same input → same arm ordering across runs (reruns hash-match)."""
    base = _write_base_config(tmp_path)
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate", values=[1e-4, 1e-3]
        ),
        AblationAxisSchema(config_path="data.batch_size", values=[2, 4, 8]),
    ]
    out_a = tmp_path / "outA"
    out_b = tmp_path / "outB"
    paths_a = AblationConfigGenerator.generate(
        base_config_path=str(base), axes=axes, output_dir=str(out_a)
    )
    paths_b = AblationConfigGenerator.generate(
        base_config_path=str(base), axes=axes, output_dir=str(out_b)
    )
    # Filenames (after stripping the output directory) must match in order.
    names_a = [Path(p).name for p in paths_a]
    names_b = [Path(p).name for p in paths_b]
    assert names_a == names_b
    # And the YAML contents must be identical per index, modulo the
    # output_dir which embeds the (different) tmp output path.
    for pa, pb in zip(paths_a, paths_b, strict=True):
        ca = yaml.safe_load(Path(pa).read_text())
        cb = yaml.safe_load(Path(pb).read_text())
        ca.get("training", {}).pop("output_dir", None)
        cb.get("training", {}).pop("output_dir", None)
        assert ca == cb


# ── Summary ─────────────────────────────────────────────────────────


def test_summarise_sweep_lists_axes_and_total() -> None:
    axes = [
        AblationAxisSchema(
            config_path="optimization.optimizer.learning_rate",
            values=[1e-4, 1e-3],
            label="lr",
        ),
        AblationAxisSchema(
            config_path="data.batch_size", values=[2, 4, 8], label="bs"
        ),
    ]
    summary = AblationConfigGenerator.summarise_sweep(axes)
    assert "lr" in summary
    assert "bs" in summary
    assert "6" in summary  # 2 * 3
    assert "Ablation Sweep Summary" in summary
