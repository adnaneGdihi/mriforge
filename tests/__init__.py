"""Test package initialization helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:
    import pytest
except ImportError:
    pass


def _format_subtest_context(params: dict[str, Any]) -> str:
    if not params:
        return ""
    keyed = ", ".join(f"{key}={value!r}" for key, value in params.items())
    return f" ({keyed})"


if "pytest" in locals() and not hasattr(pytest, "subtest"):

    @contextmanager
    def _pytest_subtest(**params: Any) -> Any:
        try:
            yield
        except AssertionError as exc:  # pragma: no cover - shim guard
            context = _format_subtest_context(params)
            raise AssertionError(f"SubTest failed{context}: {exc}") from exc
        except Exception:
            raise

    pytest.subtest = _pytest_subtest  # type: ignore
