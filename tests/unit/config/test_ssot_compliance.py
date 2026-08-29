"""
SSOT (Single Source of Truth) Compliance Test Suite

PURPOSE:
Prevent config.getattr() fallback patterns that violate SSOT principle and
ensure all configuration access trusts Pydantic schema defaults.

CRITICAL INVARIANTS:
1. NO getattr(config.*) fallback chains in production code
2. ALL config access uses direct attribute access (config.nested.field)
3. ALL accessed fields exist in Pydantic schemas with defaults or are required
4. ALL defaults are in schemas, NOT in business logic

CONTEXT:
Phase 2.3 fixed 48/115 SSOT violations in top 3 critical strategy files:
- disentangled_strategy.py: 24 violations → 0 ✅
- metrics_mixin.py: 12 violations → 0 ✅
- diffusion.py: 12 violations → 0 ✅

This test suite ensures these fixes don't regress and remaining violations
are eliminated.

RATIONALE:
Config fallback patterns cause:
- Silent bugs (wrong hyperparameters used)
- Non-deterministic behavior (fallbacks differ from schema defaults)
- Maintenance burden (defaults scattered across codebase)
- Testing complexity (can't validate config without running training)

DESIGN DECISION:
We use EXPLICIT Pydantic defaults (Field(default=...)) instead of getattr()
fallbacks because:
1. Single source of truth (schema defines all defaults)
2. Type safety (Pydantic validates types at load time)
3. Immutability (frozen=True prevents runtime mutation)
4. Fail-fast (ValidationError at load, not cryptic runtime errors)

BUGS FOUND DURING REMEDIATION:
1. max_grad_norm field doesn't exist (should be gradient_clip_value)
2. lambda_kl inconsistency (fallback 0.01, schema 0.0)
3. enable_gradient_clipping inconsistency (fallback True, schema False)
"""

import ast
import re
from pathlib import Path

import pytest


class TestSSOTCompliance:
    """Test that production code follows SSOT principle for config access."""

    # Directories to scan for violations
    PRODUCTION_DIRS = [
        "src/mriforge/infrastructure/training/strategies/",
        "src/mriforge/pipelines/",
        "src/mriforge/models/",
        "src/mriforge/data/",
    ]

    # Files known to be FIXED (no violations allowed)
    FIXED_FILES = [
        "src/mriforge/infrastructure/training/strategies/disentangled_strategy.py",
        "src/mriforge/infrastructure/training/strategies/mixins/metrics_mixin.py",
        "src/mriforge/infrastructure/training/strategies/diffusion.py",
    ]

    # Known remaining violations (to be fixed in Phase 2.5-2.7)
    # This list should shrink to zero as we fix files
    KNOWN_REMAINING_VIOLATIONS = {
        "src/mriforge/infrastructure/training/strategies/vae.py": 10,
        "src/mriforge/infrastructure/training/strategies/base.py": 9,
        "src/mriforge/infrastructure/training/strategies/reconstruction.py": 6,
        "src/mriforge/infrastructure/training/strategies/gan.py": 5,
        "src/mriforge/infrastructure/training/strategies/vision_mamba_strategy.py": 4,
        "src/mriforge/infrastructure/training/strategies/mixins/adversarial_mixin.py": 4,
        "src/mriforge/infrastructure/training/strategies/domain_adaptation.py": 4,
        "src/mriforge/infrastructure/training/strategies/vqvae.py": 3,
        "src/mriforge/infrastructure/training/strategies/pretraining.py": 2,
        "src/mriforge/infrastructure/training/strategies/physics_driven_strategy.py": 2,
        "src/mriforge/infrastructure/training/strategies/mixins/eda_mixin.py": 2,
        "src/mriforge/infrastructure/training/strategies/mixins/visualization_mixin.py": 1,
    }

    @staticmethod
    def _find_getattr_violations(file_path: Path) -> list[tuple[int, str]]:
        """
        Find all getattr(config.*) patterns in a file.

        Returns:
            List of (line_number, code_snippet) tuples
        """
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Pattern: getattr(config.*, ...) or getattr(*.config.*, ...)
            # Matches: getattr(config.training, "field", default)
            # Matches: getattr(self.state.config.model, "in_channels", 2)
            # DOES NOT match: getattr(model, "use_vae", False)  - not config access
            # DOES NOT match: getattr(self, "validation_step_count", 0)  - not config access
            pattern = re.compile(
                r"getattr\s*\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,",
                re.IGNORECASE,
            )

            for i, line in enumerate(lines, start=1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                if pattern.search(line):
                    violations.append((i, line.strip()))

        except Exception as e:
            pytest.fail(f"Error reading {file_path}: {e}")

        return violations

    def test_fixed_files_have_no_violations(self):
        """
        CRITICAL: Files fixed in Phase 2.3 must have ZERO violations.

        This prevents regression of the 48 violations we just fixed.
        """
        all_violations = []

        for file_path in self.FIXED_FILES:
            full_path = Path(file_path)
            if not full_path.exists():
                pytest.fail(f"Fixed file not found: {file_path}")

            violations = self._find_getattr_violations(full_path)

            if violations:
                all_violations.append((file_path, violations))

        if all_violations:
            error_msg = "REGRESSION: Fixed files have getattr(config.*) violations:\n\n"
            for file_path, violations in all_violations:
                error_msg += f"{file_path}:\n"
                for line_num, code in violations:
                    error_msg += f"  Line {line_num}: {code}\n"
                error_msg += "\n"

            error_msg += (
                "These files were fixed in Phase 2.3 and should have ZERO violations.\n"
                "DO NOT reintroduce getattr(config.*) patterns!\n"
                "Use direct access: config.nested.field\n"
            )

            pytest.fail(error_msg)

    def test_track_remaining_violations(self):
        """
        Track known remaining violations and ensure they don't increase.

        This test ALLOWS known violations but ensures:
        1. We don't add NEW violations
        2. We track progress as we fix files
        3. We ultimately reach zero violations

        NOTE: This test will PASS with known violations but FAIL if count increases.
        """
        current_violations = {}

        for file_path, expected_count in self.KNOWN_REMAINING_VIOLATIONS.items():
            full_path = Path(file_path)
            if not full_path.exists():
                # File might be renamed/deleted - skip
                continue

            violations = self._find_getattr_violations(full_path)
            current_count = len(violations)

            if current_count > 0:
                current_violations[file_path] = current_count

            # FAIL if violations INCREASED
            if current_count > expected_count:
                pytest.fail(
                    f"VIOLATION COUNT INCREASED in {file_path}:\n"
                    f"  Expected: {expected_count}\n"
                    f"  Found: {current_count}\n"
                    f"  New violations added: {current_count - expected_count}\n\n"
                    f"Violations:\n"
                    + "\n".join(f"  Line {num}: {code}" for num, code in violations)
                    + "\n\nDO NOT add new getattr(config.*) patterns!"
                )

            # CELEBRATE if violations DECREASED (file partially/fully fixed)
            if current_count < expected_count:
                print(
                    f"\n✅ PROGRESS: {file_path}\n"
                    f"   {expected_count} → {current_count} violations "
                    f"({expected_count - current_count} fixed!)"
                )

        # Report overall progress
        total_expected = sum(self.KNOWN_REMAINING_VIOLATIONS.values())
        total_current = sum(current_violations.values())

        print(
            f"\n📊 SSOT Compliance Progress:\n"
            f"   Known violations: {total_current}/{total_expected}\n"
            f"   Fixed: {total_expected - total_current}\n"
            f"   Remaining: {total_current}\n"
            f"   Progress: {100 * (total_expected - total_current) / total_expected:.1f}%"
        )

    def test_no_nested_getattr_chains(self):
        """Prevent nested getattr chains on the frozen config SSOT, e.g.
        ``getattr(getattr(self.config, "losses", None), "gan", None)`` -- use
        guarded direct access ``self.config.losses.gan`` instead (the schema
        guarantees every declared section/field; ``| None`` sections take an
        explicit ``is not None`` guard).

        SCOPE (AST, not a blind regex): this flags a nested getattr only when the
        INNER getattr's base is the SSOT config object itself -- ``config`` /
        ``self.config`` and their attribute chains -- exactly the docstring
        example. A nested getattr whose inner base is a DIFFERENT object is
        legitimately dynamic and is NOT a config-SSOT violation: a numpy array
        (``getattr(getattr(arr, "dtype", None), "names", None)``), a model's own
        ``gen.config``, runtime state (``obj.loop_state``), ``self.env``, or a
        getattr with a VARIABLE attribute name (``getattr(cert, letter)``). The
        previous crude ``getattr\\(getattr\\(`` regex reported all of those as a
        "CRITICAL anti-pattern", which was a false positive and left the test
        permanently red on clean dev (issue #368).
        """

        def _is_config_base(node: ast.expr) -> bool:
            text = ast.unparse(node)
            return text in {"config", "self.config"} or text.startswith(("config.", "self.config."))

        all_nested_violations = []
        for dir_path in self.PRODUCTION_DIRS:
            dir_full_path = Path(dir_path)
            if not dir_full_path.exists():
                continue
            for py_file in dir_full_path.rglob("*.py"):
                if "__pycache__" in str(py_file) or "test_" in py_file.name:
                    continue
                try:
                    tree = ast.parse(py_file.read_text(encoding="utf-8"))
                except (OSError, SyntaxError):
                    continue
                for node in ast.walk(tree):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and node.args
                        and isinstance(node.args[0], ast.Call)
                        and isinstance(node.args[0].func, ast.Name)
                        and node.args[0].func.id == "getattr"
                        and node.args[0].args
                    ):
                        continue
                    if _is_config_base(node.args[0].args[0]):
                        all_nested_violations.append(
                            (str(py_file), node.lineno, ast.unparse(node)[:90])
                        )

        if all_nested_violations:
            error_msg = "Found nested getattr() chains on the config SSOT (anti-pattern):\n\n"
            for file_path, line_num, code in all_nested_violations:
                error_msg += f"{file_path}:{line_num}\n  {code}\n\n"

            error_msg += (
                "Nested getattr on the frozen config is FORBIDDEN.\n"
                "Use guarded direct access: config.losses.gan\n"
                "NOT: getattr(getattr(config, 'losses', None), 'gan', None)\n"
            )

            pytest.fail(error_msg)

    def test_nested_getattr_scoping_flags_config_not_dynamic(self):
        """Pin the scope of test_no_nested_getattr_chains (issue #368).

        A nested getattr is a config-SSOT violation ONLY when the inner base is the
        config object; nested getattr on any other object is legitimately dynamic.
        Guards against a regression to the crude 'any nested getattr' rule that
        false-flagged numpy / model / runtime access as CRITICAL.
        """

        def _flag(src: str) -> bool:
            tree = ast.parse(src)
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
                ):
                    text = ast.unparse(node.args[0].args[0])
                    if text in {"config", "self.config"} or text.startswith(
                        ("config.", "self.config.")
                    ):
                        return True
            return False

        # Config SSOT nested getattr -> a violation.
        assert _flag('getattr(getattr(config, "losses", None), "gan", None)')
        assert _flag('getattr(getattr(self.config, "losses", None), "gan", None)')
        assert _flag('getattr(getattr(self.config.training, "dtn2s", None), "w", 8)')
        # Legitimately dynamic (different inner base) -> exempt.
        assert not _flag('getattr(getattr(arr, "dtype", None), "names", None)')
        assert not _flag('getattr(getattr(gen, "config", None), "cond", False)')
        assert not _flag('getattr(getattr(obj, "loop_state", None), "x", None)')
        assert not _flag('getattr(getattr(self, "env", None), "model_type", "")')
        assert not _flag('getattr(getattr(cert, letter), "enabled", False)')
        assert not _flag('getattr(getattr(strategy, "config", None), "validation", None)')

    def test_config_access_uses_schemas(self):
        """
        Ensure all config field access corresponds to actual Pydantic schema fields.

        This is a meta-test that verifies schema completeness.
        """
        from mriforge.config.settings import TrainingSettings

        # Get all field names from TrainingSettings
        fields = TrainingSettings.model_fields

        # Verify schema has definitions for all major config sections
        # Note: 'experiment' was removed in v6.0, use actual field names
        required_sections = [
            "data",
            "model",
            "optimization",
            "training",
            "logging",
            "validation",
            "checkpoint",
        ]

        for section in required_sections:
            # Check if section exists in TrainingSettings fields
            if section not in fields:
                pytest.fail(
                    f"Config section '{section}' not found in TrainingSettings fields. "
                    f"Schema may be incomplete."
                )


class TestConfigDefaultsInSchemas:
    """Test that all defaults are in Pydantic schemas, not in business logic."""

    def test_no_hardcoded_defaults_in_strategies(self):
        """
        Verify strategy files don't hardcode default values for CONFIG access.

        Good: config.optimization.optimizer.learning_rate  # Uses schema default
        Bad:  getattr(config.optimization, "learning_rate", 1e-4)  # Hardcoded default

        Note: Only checks config access, not model or self attributes.
        """
        # This is a pattern test - we look for common default value patterns
        # Only match when accessing config (not model or self)
        problematic_patterns = [
            (
                r'getattr\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,\s*["\'][\w_]+["\']\s*,\s*\d',
                "numeric defaults",
            ),
            (
                r'getattr\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,\s*["\'][\w_]+["\']\s*,\s*True',
                "boolean defaults",
            ),
            (
                r'getattr\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,\s*["\'][\w_]+["\']\s*,\s*False',
                "boolean defaults",
            ),
            (
                r'getattr\(\s*[a-zA-Z_\.]*config\.[a-zA-Z_\.]+\s*,\s*["\'][\w_]+["\']\s*,\s*"',
                "string defaults",
            ),
        ]

        violations = []

        for strategy_file in Path("src/mriforge/infrastructure/training/strategies/").rglob("*.py"):
            if "__pycache__" in str(strategy_file):
                continue

            try:
                with open(strategy_file, encoding="utf-8") as f:
                    content = f.read()

                for pattern, description in problematic_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Find line number
                        line_num = content[: match.start()].count("\n") + 1
                        violations.append(
                            (
                                str(strategy_file),
                                line_num,
                                description,
                                match.group(0)[:80],
                            )
                        )

            except Exception:
                pass

        if violations:
            # Filter out known remaining violations (will be fixed in Phase 2.5-2.7)
            # Only fail on NEW violations in fixed files
            new_violations = [
                v
                for v in violations
                if any(fixed in v[0] for fixed in TestSSOTCompliance.FIXED_FILES)
            ]

            if new_violations:
                error_msg = "Found hardcoded defaults in FIXED strategy files:\n\n"
                for file_path, line_num, desc, code in new_violations:
                    error_msg += f"{file_path}:{line_num} - {desc}\n  {code}\n\n"

                error_msg += (
                    "Hardcoded defaults violate SSOT principle.\n"
                    "Move defaults to Pydantic schema Field(default=value).\n"
                )

                pytest.fail(error_msg)


class TestPreventCommonAntiPatterns:
    """Prevent common anti-patterns found during Phase 2.3 remediation."""

    def test_no_get_weight_wrapper_functions(self):
        """
        Prevent helper functions that wrap getattr.

        BAD PATTERN (found in disentangled_strategy.py):
        def get_weight(config, field, default):
            return getattr(config.losses, field, default)

        These wrappers hide getattr fallbacks and violate SSOT.
        """
        violations = []

        for strategy_file in Path("src/mriforge/infrastructure/training/strategies/").rglob("*.py"):
            if "__pycache__" in str(strategy_file):
                continue

            try:
                with open(strategy_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for functions that wrap getattr
                # Pattern: def some_name(...): ... getattr(...)
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function body contains getattr
                            for child in ast.walk(node):
                                if isinstance(child, ast.Call):
                                    if (
                                        isinstance(child.func, ast.Name)
                                        and child.func.id == "getattr"
                                    ):
                                        # Check if first arg involves 'config'
                                        if child.args and "config" in ast.unparse(child.args[0]):
                                            violations.append(
                                                (
                                                    str(strategy_file),
                                                    node.lineno,
                                                    node.name,
                                                )
                                            )
                                            break

                except SyntaxError:
                    # Skip files with syntax errors
                    pass

            except Exception:
                pass

        if violations:
            # Filter to fixed files only
            new_violations = [
                v
                for v in violations
                if any(fixed in v[0] for fixed in TestSSOTCompliance.FIXED_FILES)
            ]

            if new_violations:
                error_msg = "Found wrapper functions that hide getattr patterns:\n\n"
                for file_path, line_num, func_name in new_violations:
                    error_msg += f"{file_path}:{line_num} - function '{func_name}()'\n"

                error_msg += (
                    "\nWrapper functions that use getattr(config.*) are FORBIDDEN.\n"
                    "Use direct config access instead.\n"
                )

                pytest.fail(error_msg)

    def test_field_names_match_schema(self):
        """
        Verify field names in code match actual Pydantic schema field names.

        BUG FOUND: max_grad_norm used in code, but schema has gradient_clip_value
        This test prevents similar bugs.
        """
        from mriforge.config.schemas.optimization import OptimizationConfigSchema

        # Known schema field names
        optimization_fields = set(OptimizationConfigSchema.model_fields.keys())

        # Common mistakes (wrong field names)
        wrong_field_names = {
            "max_grad_norm": "gradient_clip_value",  # Bug found in Phase 2.3
            "lr": "learning_rate",
            "grad_clip": "gradient_clip_value",
            "clip_grad": "gradient_clip_value",
        }

        violations = []

        for strategy_file in Path("src/mriforge/infrastructure/training/strategies/").rglob("*.py"):
            if "__pycache__" in str(strategy_file):
                continue

            try:
                with open(strategy_file, encoding="utf-8") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, start=1):
                    for wrong_name, correct_name in wrong_field_names.items():
                        # Look for patterns like config.optimization.max_grad_norm
                        if f".optimization.{wrong_name}" in line:
                            violations.append(
                                (
                                    str(strategy_file),
                                    i,
                                    wrong_name,
                                    correct_name,
                                    line.strip(),
                                )
                            )

            except Exception:
                pass

        if violations:
            error_msg = "Found incorrect field names (don't match Pydantic schema):\n\n"
            for file_path, line_num, wrong, correct, code in violations:
                error_msg += (
                    f"{file_path}:{line_num}\n"
                    f"  Wrong: {wrong}\n"
                    f"  Correct: {correct}\n"
                    f"  Code: {code}\n\n"
                )

            error_msg += "Field names must match Pydantic schema definitions exactly.\n"

            pytest.fail(error_msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
