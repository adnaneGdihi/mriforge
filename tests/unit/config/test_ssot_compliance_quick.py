"""
Quick SSOT Compliance Test - Fast version for CI/Pre-commit

This is a simplified version of test_ssot_compliance.py that focuses on
the most critical checks without heavy AST parsing.

CRITICAL CHECKS:
1. Fixed files (Phase 2.3) have ZERO getattr(config.*) violations
2. No new violations introduced in known violation files
"""

import ast
import re
from pathlib import Path

import pytest


class TestSSOTComplianceQuick:
    """Quick SSOT compliance tests for fast feedback."""

    # Files fixed in Phase 2.3 - NO violations allowed
    FIXED_FILES = [
        "src/spectramr/infrastructure/training/strategies/disentangled_strategy.py",
        "src/spectramr/infrastructure/training/strategies/mixins/metrics_mixin.py",
        "src/spectramr/infrastructure/training/strategies/diffusion.py",
    ]

    # Pattern: getattr(*.config.FIELD, ...) but NOT getattr(model, ...) or getattr(self, ...)
    # Must have "config." followed by at least one more field
    CONFIG_GETATTR_PATTERN = re.compile(
        r"getattr\s*\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,",
        re.IGNORECASE,
    )

    @pytest.mark.parametrize("file_path", FIXED_FILES)
    def test_fixed_file_has_no_violations(self, file_path: str):
        """
        CRITICAL: Each fixed file must have ZERO getattr(config.*) violations.

        This prevents regression of Phase 2.3 fixes.
        """
        full_path = Path(file_path)
        if not full_path.exists():
            pytest.fail(f"Fixed file not found: {file_path}")

        violations = []
        with open(full_path, encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines, start=1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            if self.CONFIG_GETATTR_PATTERN.search(line):
                violations.append((i, line.strip()))

        if violations:
            error_msg = f"REGRESSION: {file_path} has getattr(config.*) violations:\n\n"
            for line_num, code in violations:
                error_msg += f"  Line {line_num}: {code}\n"

            error_msg += (
                "\nThis file was fixed in Phase 2.3 and must have ZERO violations.\n"
                "DO NOT reintroduce getattr(config.*) patterns!\n"
                "Use direct access: config.nested.field\n"
            )

            pytest.fail(error_msg)

    def test_no_nested_getattr_chains(self):
        """Prevent nested getattr chains on the frozen config SSOT.

        Scoped (AST) to nested getattr whose INNER base is ``config`` /
        ``self.config`` -- the documented anti-pattern
        ``getattr(getattr(self.config, "losses", None), "gan", None)`` (use guarded
        direct access instead). Nested getattr on a different object (numpy dtype,
        a model's own ``gen.config``, runtime state, a variable attr name) is
        legitimately dynamic and NOT a config-SSOT violation. See #368 and the
        fuller test in test_ssot_compliance.py.
        """

        def _is_config_base(node: ast.expr) -> bool:
            text = ast.unparse(node)
            return text in {"config", "self.config"} or text.startswith(("config.", "self.config."))

        violations = []
        for py_file in Path("src/spectramr/infrastructure/training/strategies/").rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and node.args
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id == "getattr"
                    and node.args[0].args
                    and _is_config_base(node.args[0].args[0])
                ):
                    violations.append((str(py_file), node.lineno, ast.unparse(node)[:90]))

        if violations:
            error_msg = "Found nested getattr() chains on the config SSOT (anti-pattern):\n\n"
            for file_path, line_num, code in violations:
                error_msg += f"{file_path}:{line_num}\n  {code}\n\n"

            error_msg += (
                "Nested getattr on the frozen config is FORBIDDEN.\n"
                "Use guarded direct access: config.losses.gan\n"
                "NOT: getattr(getattr(config, 'losses', None), 'gan', None)\n"
            )

            pytest.fail(error_msg)

    def test_schema_sections_exist(self):
        """
        Verify TrainingSettings has all major config sections.

        This is a quick schema completeness check.
        """
        from spectramr.config.settings import TrainingSettings

        # Get all field names from TrainingSettings
        fields = TrainingSettings.model_fields

        # Verify major config sections exist
        required_sections = [
            "data",
            "model",
            "optimization",
            "training",
            "logging",
            "validation",
            "checkpoint",
        ]

        missing = [s for s in required_sections if s not in fields]

        if missing:
            pytest.fail(
                f"Missing config sections in TrainingSettings: {', '.join(missing)}\n"
                f"Schema may be incomplete."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
