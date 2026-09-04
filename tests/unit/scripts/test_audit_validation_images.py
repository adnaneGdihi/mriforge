"""``scripts/ci/audit_validation_images.py`` must read the RUN, not the declaration.

The script decides which smoke-passed arms emitted no validation images. It used
to answer that from raw YAML (``cfg["validation"]["eval_interval"]``). 12 of its
15 keys are ``fold`` records, so a *drained* arm declares the canonical spelling
and every legacy ``.get()`` returned ``None`` -- rendering an arm that emits
images as one that has them switched off.

That went live when ``kspace_filling`` was drained (58 arms). These tests pin the
resolved read, the fail-loud marker, and the label->path map against the schema.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from tests.utils.corpus import tracked_yamls

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "audit_validation_images.py"
DRAINED_COHORT = REPO_ROOT / "experiments" / "inprogress" / "kspace_filling"


def _module() -> Any:
    """Load the script by path -- ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("audit_validation_images", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestValidationFlags:
    def test_a_drained_arm_reports_its_real_image_settings(self) -> None:
        """The regression: raw-YAML reads returned None on every drained arm.

        Uses a real cohort arm rather than a hand-built dict on purpose -- a
        hand-built dict is a second resolver, and believing one is exactly how
        the original defect survived.
        """
        arms = tracked_yamls(DRAINED_COHORT)
        if not arms:
            pytest.skip("kspace_filling cohort absent (published branch)")
        mod = _module()
        flags = mod._validation_flags(arms[0])
        assert "_unresolved" not in flags, flags
        # These four are ALL fold records; under the pre-fix raw read every one
        # of them was None on a drained arm.
        for label in (
            "validation.eval_interval",
            "logging.save_validation_images",
            "logging.log_validation_images",
            "validation.compute_image_metrics",
        ):
            assert flags[label] is not None, (
                f"{label} resolved to None on {arms[0].name} -- the read has "
                "reverted to the pre-decomposition spelling"
            )

    def test_an_unloadable_config_is_unresolved_not_silently_disabled(
        self, tmp_path: Path
    ) -> None:
        """ "could not resolve" and "images are off" must not render the same."""
        bad = tmp_path / "not_a_config.yaml"
        bad.write_text("config_version: '1.0'\nthis_key: does_not_exist\n")
        flags = _module()._validation_flags(bad)
        assert "_unresolved" in flags, flags
        assert not any(k.startswith(("validation.", "logging.")) for k in flags)

    def test_every_label_resolves_to_a_declared_schema_path(self) -> None:
        """Ratchet: a future rename must break this, not go quietly None."""
        from spectramr.config.settings import TrainingSettings

        mod = _module()
        arms = tracked_yamls(DRAINED_COHORT)
        if not arms:
            pytest.skip("kspace_filling cohort absent (published branch)")
        settings = TrainingSettings.from_yaml(str(arms[0]))
        assert mod._FLAG_PATHS, "the label->path map is empty -- it moved"
        broken = []
        for label, dotted in mod._FLAG_PATHS.items():
            obj: Any = settings
            for part in dotted.split("."):
                if not hasattr(obj, part):
                    broken.append(f"{label} -> {dotted} (no `{part}`)")
                    break
                obj = getattr(obj, part)
        assert not broken, "labels naming a path the schema no longer has: " + str(
            broken
        )
