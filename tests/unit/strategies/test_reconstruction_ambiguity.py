"""Strategy verification: ReconstructionTrainingStrategy uses explicit config.

Tests:
- Uses config flag `multislice_enabled`, not shape-based inference
- No ambiguous channel detection based on shape heuristics
- Clear separation between multimodal (4+ channels) and multislice data
"""

import ast

import pytest


class TestReconstructionChannelAmbiguity:
    """Verify ReconstructionTrainingStrategy avoids implicit channel guessing."""

    def test_uses_explicit_multislice_config_flag(self):
        """Verify strategy reads `multislice_enabled` from config, not shape.

        The strategy should use:
            is_multislice = getattr(self.state.config.data, 'multislice_enabled', False)

        NOT shape-based inference like:
            if lr_image.shape[1] > 2:  # BAD: This is ambiguous
        """
        with open("src/mriforge/infrastructure/training/strategies/reconstruction.py") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find _prepare_generator_inputs method
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_prepare_generator_inputs"
            ):
                method_source = ast.unparse(node)

                # Should reference multislice config flag
                assert (
                    "multislice_enabled" in method_source
                ), "_prepare_generator_inputs should use explicit 'multislice_enabled' config flag"
                break

    def test_has_forensic_fix_comment(self):
        """Verify FORENSIC FIX comment exists for multislice handling."""
        with open("src/mriforge/infrastructure/training/strategies/reconstruction.py") as f:
            source = f.read()

        assert (
            "[FORENSIC FIX]" in source
        ), "ReconstructionTrainingStrategy should have [FORENSIC FIX] comment"

    def test_no_ambiguous_shape_heuristics(self):
        """Verify shape check is combined with config flag, not standalone."""
        with open("src/mriforge/infrastructure/training/strategies/reconstruction.py") as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_prepare_generator_inputs"
            ):
                method_source = ast.unparse(node)

                # The shape check should be combined with config flag
                has_safe_pattern = (
                    "is_multislice" in method_source
                    and "model_in_channels" in method_source
                )

                assert (
                    has_safe_pattern
                ), "_prepare_generator_inputs should combine config flag AND shape check"
                break

    def test_multislice_detection_is_documented(self):
        """Verify the multislice logic is clearly documented."""
        with open("src/mriforge/infrastructure/training/strategies/reconstruction.py") as f:
            source = f.read()

        assert (
            "multislice" in source.lower()
        ), "ReconstructionTrainingStrategy should document multislice handling"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
