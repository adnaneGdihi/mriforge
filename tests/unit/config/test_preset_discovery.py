"""Unit tests for :mod:`spectramr.config.presets.discovery`.

The discovery module scans ``experiments/training/`` for
``experiment_*.yaml`` files and registers them as presets. The category
is inferred from the filename, and ``_infer_tags`` adds workflow tags
(``baseline``, ``ablation``, ``hpo``, …).

Tests use a temporary experiments directory so we don't depend on the
repo's own ``experiments/`` tree shifting under our feet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.presets import PresetRegistry
from spectramr.config.presets.discovery import (
    _infer_tags,
    discover_and_register_presets,
    get_baseline_presets,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    reg = PresetRegistry()
    reg._presets.clear()
    reg._categories.clear()
    reg._tags.clear()


def _write_yaml(path: Path) -> None:
    path.write_text(
        "config_version: '6.0'\n"
        "experiment:\n"
        "  name: stub\n"
    )


# ── _infer_tags ─────────────────────────────────────────────────────


class TestInferTags:
    def test_category_always_included(self) -> None:
        tags = _infer_tags("experiment_99_x.yaml", "ssl")
        assert "ssl" in tags

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("experiment_01_baseline_gan.yaml", "baseline"),
            ("experiment_03_ablation_loss.yaml", "ablation"),
            ("experiment_04_robustness.yaml", "robustness"),
            ("experiment_05_hyperparameter.yaml", "hpo"),
            ("experiment_06_hpo_sweep.yaml", "hpo"),
            ("experiment_07_ensemble.yaml", "ensemble"),
            ("experiment_08_federated.yaml", "federated"),
            ("experiment_09_transformer.yaml", "transformer"),
            ("experiment_10_vit.yaml", "transformer"),
            ("experiment_11_mamba.yaml", "mamba"),
            ("experiment_12_foundation_llm.yaml", "foundation"),
        ],
    )
    def test_workflow_tag_inference(self, filename: str, expected: str) -> None:
        tags = _infer_tags(filename, "gan")
        assert expected in tags, f"expected '{expected}' in {tags} for {filename}"

    def test_no_workflow_tag_for_plain_filename(self) -> None:
        tags = _infer_tags("experiment_99_plain.yaml", "diffusion")
        # Category is always present.
        assert "diffusion" in tags
        # Without any signal in the filename, no workflow tag is appended.
        for unwanted in (
            "baseline",
            "ablation",
            "robustness",
            "hpo",
            "ensemble",
            "federated",
            "transformer",
            "mamba",
            "foundation",
        ):
            assert unwanted not in tags

    def test_01_pattern_treated_as_baseline(self) -> None:
        # Files numbered "01_..." are conventionally baselines.
        tags = _infer_tags("experiment_01_my_run.yaml", "gan")
        assert "baseline" in tags


# ── discover_and_register_presets ───────────────────────────────────


class TestDiscoverAndRegister:
    def test_missing_directory_returns_zero_and_warns(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir"
        n = discover_and_register_presets(experiments_dir=missing)
        assert n == 0
        # Registry untouched.
        assert PresetRegistry().list_all() == []

    def test_scans_only_experiment_glob(self, tmp_path: Path) -> None:
        # Should pick up experiment_*.yaml, ignore others.
        _write_yaml(tmp_path / "experiment_01_baseline_gan.yaml")
        _write_yaml(tmp_path / "experiment_02_diffusion.yaml")
        (tmp_path / "not_an_experiment.yaml").write_text("config_version: '6.0'\n")
        (tmp_path / "README.md").write_text("# notes\n")

        n = discover_and_register_presets(experiments_dir=tmp_path)
        assert n == 2
        names = {p.name for p in PresetRegistry().list_all()}
        assert names == {
            "experiment_01_baseline_gan",
            "experiment_02_diffusion",
        }

    def test_categories_inferred_from_filename(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path / "experiment_01_baseline_gan.yaml")
        _write_yaml(tmp_path / "experiment_02_diffusion.yaml")
        _write_yaml(tmp_path / "experiment_03_reconstruction.yaml")
        _write_yaml(tmp_path / "experiment_04_vae.yaml")
        _write_yaml(tmp_path / "experiment_05_ssl.yaml")
        _write_yaml(tmp_path / "experiment_99_unknown.yaml")

        discover_and_register_presets(experiments_dir=tmp_path)

        # Map preset name → category.
        cats = {p.name: p.category for p in PresetRegistry().list_all()}
        assert cats["experiment_01_baseline_gan"] == "gan"
        assert cats["experiment_02_diffusion"] == "diffusion"
        assert cats["experiment_03_reconstruction"] == "reconstruction"
        assert cats["experiment_04_vae"] == "vae"
        assert cats["experiment_05_ssl"] == "ssl"
        # No keyword match → "other".
        assert cats["experiment_99_unknown"] == "other"

    def test_failed_register_does_not_abort_scan(self, tmp_path: Path) -> None:
        """A duplicate registration is swallowed by the try/except in
        ``discover_and_register_presets`` — subsequent files still
        register.
        """
        _write_yaml(tmp_path / "experiment_01_baseline_gan.yaml")
        # Pre-seed the registry so the first add will collide.
        from spectramr.config.presets import ConfigPreset

        PresetRegistry.register(
            ConfigPreset(
                name="experiment_01_baseline_gan",
                category="gan",
                description="pre-existing",
                yaml_path=str(tmp_path / "experiment_01_baseline_gan.yaml"),
            )
        )
        _write_yaml(tmp_path / "experiment_02_diffusion.yaml")

        n = discover_and_register_presets(experiments_dir=tmp_path)
        # One file (the diffusion one) registered successfully; the
        # duplicate name is skipped without aborting.
        assert n == 1
        names = {p.name for p in PresetRegistry().list_all()}
        assert "experiment_02_diffusion" in names


# ── get_baseline_presets ────────────────────────────────────────────


class TestGetBaselinePresets:
    def test_returns_one_per_category_with_baseline_tag(self, tmp_path: Path) -> None:
        _write_yaml(tmp_path / "experiment_01_baseline_gan.yaml")
        _write_yaml(tmp_path / "experiment_01_baseline_diffusion.yaml")
        # Non-baseline arm in the same category.
        _write_yaml(tmp_path / "experiment_07_ablation_gan.yaml")

        discover_and_register_presets(experiments_dir=tmp_path)
        baselines = get_baseline_presets()

        # gan + diffusion baselines present; ablation gan not picked.
        names = {p.name for p in baselines}
        assert "experiment_01_baseline_gan" in names
        assert "experiment_01_baseline_diffusion" in names
        # Each preset returned must have "baseline" tag.
        for p in baselines:
            assert "baseline" in p.tags

    def test_falls_back_to_first_in_category_if_no_baseline(
        self, tmp_path: Path
    ) -> None:
        _write_yaml(tmp_path / "experiment_42_ablation_gan.yaml")
        discover_and_register_presets(experiments_dir=tmp_path)
        # No "baseline" tag → fallback picks the only arm in the
        # category.
        baselines = get_baseline_presets()
        names = {p.name for p in baselines}
        assert "experiment_42_ablation_gan" in names

    def test_empty_when_registry_empty(self) -> None:
        # No discovery happened. Should return empty list.
        assert get_baseline_presets() == []
