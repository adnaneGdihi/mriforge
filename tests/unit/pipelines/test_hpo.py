"""Unit tests for the random-search HPO pipeline (``run_hpo_random``).

CPU-only, deterministic, no real training. ``TrainingSettings.from_yaml`` is
mocked so the heavy schema/loader is never exercised, and ``training_fn`` is
left as ``None`` so no trainer is ever invoked. These tests pin the
no-silent-fallback contract (CLAUDE.md pitfall #9): an unknown distribution
type in ``param_dist`` must raise ``ValueError`` during param sampling, never
silently drop the hyperparameter from the sweep.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spectramr.pipelines.hpo import _build_trial_yaml, run_hpo_random

pytestmark = pytest.mark.unit


def _write_dummy_config(tmp_path) -> str:
    """Create an on-disk YAML so ``Path(config_path).exists()`` is True.

    The contents are irrelevant: ``TrainingSettings.from_yaml`` is mocked, so
    the file is never actually parsed.
    """
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text("metadata:\n  name: dummy\n")
    return str(cfg_path)


class TestRunHpoRandomUnknownDistribution:
    """The no-silent-fallback regression: unknown dist_type must raise."""

    def test_unknown_dist_type_raises_value_error(self, tmp_path):
        cfg_path = _write_dummy_config(tmp_path)

        # Bogus first element => unknown distribution type. The tuple still has
        # len >= 2 so it reaches the dist_type dispatch (and the final raise),
        # rather than being skipped by the structural guard.
        param_dist = {"optimization.optimizer.learning_rate": ("bogus_dist", 1e-5, 1e-2)}

        with (
            patch(
                "spectramr.pipelines.hpo.TrainingSettings.from_yaml",
                return_value=MagicMock(name="base_config"),
            ),
            patch("spectramr.pipelines.hpo.override_config_param") as mock_override,
        ):
            with pytest.raises(ValueError) as excinfo:
                run_hpo_random(
                    config_path=cfg_path,
                    param_dist=param_dist,
                    num_trials=1,
                    output_dir=tmp_path / "out",
                    training_fn=None,  # never train
                    device="cpu",
                    seed=0,
                )

        # The raise must name the offending dist_type and the param it belongs
        # to, and it must fire BEFORE override_config_param (i.e. before the
        # param is committed to the config) — proving the param is not silently
        # dropped from the sweep.
        msg = str(excinfo.value)
        assert "bogus_dist" in msg
        assert "optimization.optimizer.learning_rate" in msg
        mock_override.assert_not_called()


def test_build_trial_yaml_applies_a_dotted_override(tmp_path):
    """Regression: the override path must RUN, not merely be importable.

    ``_build_trial_yaml`` is what every HPO trial calls, and it resolves
    ``apply_dotted_override`` through a *function-local* import. Commit
    ``366e3e1a6`` ("formating") deleted the re-export that import pointed at, so
    every trial that applied an override raised ``ImportError`` -- i.e. every HPO
    run. Nothing caught it:

    * the import is function-local, and ``check_layering.sh``'s greps are all
      anchored at ``^`` (non-negotiable 15's stated blind spot);
    * ruff's F401 removed the re-export precisely BECAUSE it could not see the
      downstream consumers, which is non-negotiable 19's "a mechanical rewrite is
      locally well-formed everywhere it is wrong";
    * the test that would have caught it could not even be COLLECTED -- the same
      commit broke ``test_hpo_search_spaces.py``'s imports, and a collection error
      makes pytest discard the whole session, so ``pytest tests/unit/`` ran nothing.

    Asserting the symbol is importable would not reproduce that: the defect was on
    a path only reached by calling the function. So this calls it.
    """
    out = _build_trial_yaml(
        base_config={"optimization": {"optimizer": {"learning_rate": 1e-3}}},
        trial_number=7,
        overrides={"optimization.optimizer.learning_rate": 0.5},
        output_root=tmp_path,
        max_iterations=1,
    )

    import yaml

    written = yaml.safe_load(out.read_text())
    assert written["optimization"]["optimizer"]["learning_rate"] == 0.5
