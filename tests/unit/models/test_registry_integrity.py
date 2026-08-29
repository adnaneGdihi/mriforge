#!/usr/bin/env python3
"""Integrity tests for model registry.

Scans source code to verify:
- No duplicate @register_model decorators
- All registrations have valid names and training_modes
- Registry matches source code
"""

import re
from collections import defaultdict
from pathlib import Path

import pytest


class RegistrationScanner:
    """Scans source code for model registrations."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.registrations: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def scan_file(self, file_path: Path):
        """Scan a Python file for @register_model decorators."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return

        # Find all @register_model decorators
        pattern = r'@register_model\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'

        for match in re.finditer(pattern, content):
            model_name = match.group(1)
            line_number = content[: match.start()].count("\n") + 1

            relative_path = str(file_path.relative_to(self.base_path))
            self.registrations[model_name].append((relative_path, line_number))

    def scan_directory(self, directory: Path):
        """Recursively scan directory for registrations."""
        for py_file in directory.rglob("*.py"):
            if "test" not in str(py_file):  # Skip test files
                self.scan_file(py_file)

    def get_duplicates(self) -> dict[str, list[tuple[str, int]]]:
        """Return models registered multiple times."""
        return {
            name: locations
            for name, locations in self.registrations.items()
            if len(locations) > 1
        }

    def get_all_registrations(self) -> dict[str, list[tuple[str, int]]]:
        """Return all registrations."""
        return dict(self.registrations)


class TestRegistryIntegrity:
    """Test registry integrity by scanning source code."""

    @pytest.fixture
    def scanner(self):
        """Create scanner for src/models directory."""
        # __file__ = tests/unit/models/test_registry_integrity.py
        # parents[3] = repo root
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "mriforge" / "models"

        scanner = RegistrationScanner(base_path)
        scanner.scan_directory(models_path)
        return scanner

    # Names that are intentionally registered BOTH in the per-file
    # baseline implementation AND the v6.1 metadata manifest
    # (src/models/_v6_1_registrations.py). The manifest enrichment is a
    # known design point — the last-write-wins behaviour of the
    # registry promotes the enriched (input_domain, output_domain,
    # accepts_complex) record over the bare per-file one.
    KNOWN_DUAL_REGISTERED = {
        "cdiffmr_baseline",
        "shen2024_baseline",
        "fdb_baseline",
    }

    def test_no_duplicate_registrations(self, scanner):
        """Verify no model name is registered twice (except known dual sites)."""
        duplicates = scanner.get_duplicates()
        unexpected = {
            name: locs
            for name, locs in duplicates.items()
            if name not in self.KNOWN_DUAL_REGISTERED
        }

        if unexpected:
            error_msg = "Found duplicate model registrations:\n"
            for name, locations in unexpected.items():
                error_msg += f"\n'{name}' registered {len(locations)} times:\n"
                for file_path, line_num in locations:
                    error_msg += f"  - {file_path}:{line_num}\n"

            pytest.fail(error_msg)

    def test_minimum_registrations(self, scanner):
        """Verify we have minimum expected registrations."""
        all_registrations = scanner.get_all_registrations()
        total = len(all_registrations)

        assert total >= 100, f"Expected at least 100 registrations, found {total}"

    def test_newly_registered_models_present(self, scanner):
        """Verify our newly registered models are in source code.

        Names migrated to the ``vae_*`` / ``latent_*`` prefixes during
        the schema-6.0 refactor; pinning the *current* names so a
        future rename surfaces here.
        """
        expected_models = [
            "vae_fourier_kan_vae",
            "vae_wavelet_kan_vae",
            "sparse_vae",
            "vqvae_generator",
            "vqgan",
            "configurable_vae",
            "patch_latent_discriminator",
        ]

        all_registrations = scanner.get_all_registrations()

        for model_name in expected_models:
            assert (
                model_name in all_registrations
            ), f"Expected model '{model_name}' not found in source code"

    def test_no_commented_duplicates(self, scanner):
        """Verify we don't have both registered and commented registrations."""
        # This would need more sophisticated parsing
        # For now, we verify unique registrations
        all_registrations = scanner.get_all_registrations()

        # Each model should appear exactly once
        for name, locations in all_registrations.items():
            if len(locations) > 1:
                # Check if any are commented
                base_path = Path(__file__).resolve().parents[3]

                for file_path, line_num in locations:
                    full_path = base_path / file_path
                    with open(full_path) as f:
                        lines = f.readlines()
                        line = lines[line_num - 1]

                        # If line starts with #, it's commented
                        if line.strip().startswith("#"):
                            pytest.fail(
                                f"Model '{name}' has both active and commented "
                                f"registrations at {file_path}:{line_num}"
                            )


class TestRegistrationFormat:
    """Test that registrations follow correct format."""

    def test_registration_includes_training_mode(self):
        """Verify all registrations specify training_mode.

        Accepts both keyword form ``@register_model(name="x", training_mode="y")``
        and positional form ``@register_model("x", "y", ...)`` — the latter is
        idiomatic in older generators like ``pma_varnet``.
        """
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "mriforge" / "models"

        # Capture the decorator call's argument span (DOTALL because the
        # decorator commonly spans multiple lines with kwargs). Anchored to
        # line-start (MULTILINE ``^\s*``) so a ``@register_model`` mentioned
        # inside a trailing ``# noqa`` comment is not mis-parsed as a call.
        full_pattern = re.compile(
            r"^\s*@register_model\s*\(\s*(?P<args>.*?)\s*\)",
            re.DOTALL | re.MULTILINE,
        )
        # Positional form: two string literals as the first two args.
        positional_form = re.compile(
            r"""^\s*["'][^"']+["']\s*,\s*["'][^"']+["']""",
            re.VERBOSE,
        )

        issues = []

        for py_file in models_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
            except Exception:
                continue

            for match in full_pattern.finditer(content):
                args = match.group("args")
                has_kw = "training_mode" in args
                has_pos = bool(positional_form.match(args))
                if not (has_kw or has_pos):
                    line_num = content[: match.start()].count("\n") + 1
                    relative_path = str(py_file.relative_to(base_path))
                    issues.append(f"{relative_path}:{line_num}")

        if issues:
            pytest.fail(
                f"Found {len(issues)} registrations without training_mode:\n"
                + "\n".join(f"  - {issue}" for issue in issues[:10])
            )

    def test_valid_training_modes(self):
        """Verify all training_modes are valid.

        Scans both keyword and positional forms of ``@register_model``.
        """
        base_path = Path(__file__).resolve().parents[3]
        models_path = base_path / "src" / "mriforge" / "models"

        # Pinned universe of decorator-form registration modes (the paradigm
        # dispatch bucket, distinct from config VALID_TRAINING_MODES). This is
        # a source-scan ratchet: adding a new mode string is a deliberate act
        # that must be reflected here, so a typo'd mode fails the gate. Refreshed
        # 2026-07-02 after the src->mriforge rename left this test scanning a
        # non-existent path (silently vacuous) — the set had drifted to 24 while
        # the tree carried 52.
        # 2026-07-02 (sota_registry retirement): "acquisition" and "synthesis"
        # added — the retired module registered rl_mri / gen_3dgs / causal_gan
        # under these modes via CALL-form register_model(...) which this
        # decorator-scan never saw; the migration to @register_model decorators
        # makes them visible here. The census pins the exact (name, mode) pairs.
        valid_modes = {
            "acq_hypernetwork",
            "acquisition",
            "acquisition_learning",
            "active_acquisition",
            "adaptation",
            "bloch_field",
            # Pre-existing red on dev, repaired here: these three are registered
            # on live models and all dispatch, but this list (a fourth
            # hand-maintained copy of the mode names) had drifted behind them.
            "bloch_synth",
            "field_cocycle",
            "stargan_v2",
            "brenier_synthesis",
            "cold_diffusion",
            "compression",
            "corruption_calibration",
            "cross_field_translation",
            "diffusion",
            # 2026-08-07: DL-BAE (contrast/field-agnostic bundle M4) registers
            # `disp_bloch_ae` under this mode. `acq_hypernetwork` above was
            # already pinned by lcah_encoder, whose STRATEGY_CLASS_PATHS key
            # landed in the same change.
            "dispersion_bloch_ae",
            "domain_adaptation",
            "doob_bridge",
            "encoder",
            "experimental",
            "federated",
            "field_conditioned_inr",
            "field_flow",
            "field_fno",
            "field_guided_diffusion",
            "field_wiener",
            "fisher_rao_geodesic",
            "flow_matching",
            "gan",
            "generative",
            "geomamba_ulf",
            "ib_vf",
            "koopman_field",
            "learnable_acquisition",
            "lora_modulation",
            "mccann_field_path",
            "meta_learning",
            "monotone_field",
            "multi_param_mapping",
            "operator_id",
            "paired_synthesis",
            # siren_sens_net retagged here 2026-07-19 from the removed
            # "sensitivity_estimation" orphan (it named no strategy).
            "pinn",
            "reconstruction",
            "recoverability_vib",
            "scattering_besov",
            "sparse_frame",
            "ssl",
            "steerable_synthesis",
            "supervised",
            "synthesis",
            "tissue_diffusion_pretrain",
            "trajectory_recon",
            "transport",
            "utility",
            "vae",
            "virtual_fiducial",
            "vqvae",
            "wrapper",
        }

        kw_pattern = re.compile(r'training_mode\s*=\s*["\']([^"\']+)["\']')
        pos_pattern = re.compile(
            r"""^\s*@register_model\s*\(\s*["'][^"']+["']\s*,\s*["']([^"']+)["']""",
            re.DOTALL | re.MULTILINE,
        )

        invalid_modes = []

        for py_file in models_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    content = f.read()
            except Exception:
                continue

            for pattern in (kw_pattern, pos_pattern):
                for match in pattern.finditer(content):
                    mode = match.group(1)

                    if mode not in valid_modes:
                        line_num = content[: match.start()].count("\n") + 1
                        relative_path = str(py_file.relative_to(base_path))
                        invalid_modes.append(f"{relative_path}:{line_num} - '{mode}'")

        if invalid_modes:
            pytest.fail(
                f"Found invalid training_modes (valid: {valid_modes}):\n"
                + "\n".join(f"  - {issue}" for issue in invalid_modes[:10])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
