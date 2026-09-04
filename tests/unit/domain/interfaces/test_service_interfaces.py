"""Tests for domain service interface contracts (WS-3).

Regression: ``I3DGenerator.generate_from_text`` declared no ``prompt`` parameter
— the primary text input was silently absorbed by ``**kwargs``.
"""

from __future__ import annotations

import inspect

from spectramr.domain.interfaces.service_interfaces import I3DGenerator


def test_generate_from_text_declares_prompt_first() -> None:
    params = list(inspect.signature(I3DGenerator.generate_from_text).parameters)
    assert "prompt" in params
    assert params.index("prompt") < params.index("output_format")
