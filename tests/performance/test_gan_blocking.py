"""Performance test: Verify GAN strategy has no sync-blocking calls.

Tests:
- No .cpu() calls in _report_metrics hot path (aside from comments)
- Uses detach() only for tensor isolation
- CPU transfer deferred to async worker
"""

import ast

import pytest


class TestGANSyncBlocking:
    """Verify GANTrainingStrategy doesn't block on CPU sync in metrics."""

    def test_no_cpu_calls_in_report_metrics(self):
        """Verify _report_metrics uses detach() only, no .cpu() in code.

        The method should:
        1. Call .detach() on tensors (breaks gradient graph)
        2. NOT call .cpu() in actual code (causes GPU/CPU sync)
        3. Let AsyncMetricsReporter handle CPU transfer in background
        """
        with open("src/spectramr/infrastructure/training/strategies/gan.py") as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_report_metrics":
                # Get only the body (excluding docstring)
                # First statement might be the docstring
                body_start_idx = 0
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    body_start_idx = 1  # Skip docstring

                # Unparse only the body statements
                body_source = "\n".join(
                    ast.unparse(stmt) for stmt in node.body[body_start_idx:]
                )

                # Should have detach() call in code body
                assert (
                    ".detach()" in body_source
                ), "_report_metrics should call .detach() to isolate tensors"

                # Should NOT have .cpu() call in actual code body
                cpu_call_count = body_source.count(".cpu()")
                assert cpu_call_count == 0, (
                    f"_report_metrics has {cpu_call_count} .cpu() calls in code body. "
                    f"CPU transfer should be deferred to AsyncMetricsReporter."
                )
                break

    def test_has_forensic_fix_comment(self):
        """Verify FORENSIC FIX comment exists in GAN strategy."""
        with open("src/spectramr/infrastructure/training/strategies/gan.py") as f:
            source = f.read()

        assert (
            "[FORENSIC FIX]" in source
        ), "GANTrainingStrategy should have [FORENSIC FIX] comment"

    def test_uses_async_metrics_reporter(self):
        """Verify GANTrainingStrategy uses AsyncMetricsReporter."""
        with open("src/spectramr/infrastructure/training/strategies/gan.py") as f:
            source = f.read()

        assert (
            "AsyncMetricsReporter" in source
        ), "GANTrainingStrategy should use AsyncMetricsReporter for deferred CPU sync"

    def test_no_item_calls_in_report_metrics(self):
        """Verify _report_metrics doesn't call .item() which triggers sync."""
        with open("src/spectramr/infrastructure/training/strategies/gan.py") as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_report_metrics":
                method_source = ast.unparse(node)

                # Should NOT have .item() call (sync point)
                item_count = method_source.count(".item()")
                assert (
                    item_count == 0
                ), f"_report_metrics has {item_count} .item() calls which trigger sync."
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
