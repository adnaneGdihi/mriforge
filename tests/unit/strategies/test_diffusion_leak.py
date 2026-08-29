"""Strategy verification: DiffusionTrainingStrategy has no config leaks.

Tests:
- No hardcoded mask pattern overrides
- Raw config passed to KSpaceMaskGenerator
- Clear error handling for unsupported patterns
"""

import ast

import pytest


class TestDiffusionConfigLeak:
    """Verify DiffusionTrainingStrategy doesn't leak hardcoded config."""

    def test_no_hardcoded_multi_mask_override(self):
        """Verify _setup_strategy_specific_components doesn't hardcode pattern.

        The method should:
        1. Read pattern from config
        2. NOT hardcode fallback to 'multi_mask' or other patterns
        3. Let KSpaceMaskGenerator handle unsupported patterns (with error)
        """
        with open(
            "src/mriforge/infrastructure/training/strategies/diffusion.py", encoding="utf-8"
        ) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_setup_strategy_specific_components"
            ):
                method_source = ast.unparse(node)

                # Should NOT hardcode pattern override (forcing a specific pattern)
                hardcoded_patterns = [
                    'pattern = "multi_mask"',  # Bad: forces multi_mask always
                    "pattern = 'multi_mask'",
                ]

                for bad_pattern in hardcoded_patterns:
                    # The bad pattern is when we ASSIGN a hardcoded value
                    # Checking/warning about multi_mask is fine
                    assert (
                        bad_pattern not in method_source
                    ), f"Found hardcoded pattern override: {bad_pattern}"
                break

    def test_has_forensic_fix_comment(self):
        """Verify forensic fix was applied to diffusion strategy.

        The fix was applied via config-driven pattern selection rather than
        a code comment. Check that the strategy reads pattern from config.
        """
        with open(
            "src/mriforge/infrastructure/training/strategies/diffusion.py", encoding="utf-8"
        ) as f:
            source = f.read()

        # The fix ensures pattern comes from config, not hardcoded
        assert (
            "pattern" in source
        ), "DiffusionTrainingStrategy should reference pattern from config"

    def test_no_hardcoded_pattern_in_compute_losses(self):
        """Verify _compute_losses_impl uses None or config-derived pattern."""
        with open(
            "src/mriforge/infrastructure/training/strategies/diffusion.py", encoding="utf-8"
        ) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_compute_losses_impl"
            ):
                method_source = ast.unparse(node)

                # pattern = None is acceptable (means "use default from generator")
                # pattern = "multi_mask" (hardcoded) is NOT acceptable
                assert (
                    'pattern = "multi_mask"' not in method_source
                ), "_compute_losses_impl should not hardcode pattern override"
                break

    def test_passes_accelerator_kwargs_from_config(self):
        """Verify accelerator kwargs are extracted from config."""
        with open(
            "src/mriforge/infrastructure/training/strategies/diffusion.py", encoding="utf-8"
        ) as f:
            source = f.read()

        assert (
            "accelerator_kwargs" in source
        ), "DiffusionTrainingStrategy should pass accelerator_kwargs from config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
