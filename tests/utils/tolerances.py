"""Numerical tolerance constants for dtype-aware comparisons.

Use :func:`tol_for` to retrieve the appropriate tolerance for a given
``torch.dtype``.  Tests that compare tensors with ``torch.allclose`` or
``pytest.approx`` should use these constants so that the tolerance
thresholds are consistent across the test suite and easy to update in
one place.

Example::

    from tests.utils.tolerances import tol_for
    import torch

    pred = model(x)
    assert torch.allclose(pred, target, atol=tol_for(pred.dtype))
"""

from __future__ import annotations

import torch

# Canonical tolerance map keyed on *string* dtype names (as returned by
# ``str(dtype).replace("torch.", "")``) so that the dict is JSON-serialisable
# and easy to inspect in debugging output.
TOLERANCES: dict[str, float] = {
    "fp16": 5e-3,
    "bf16": 5e-3,
    "fp32": 1e-5,
    "fp64": 1e-10,
    "complex64": 1e-5,
    "complex128": 1e-10,
}

# Internal mapping from torch.dtype to the TOLERANCES key so the lookup
# works without string manipulation at call time.
_DTYPE_TO_KEY: dict[torch.dtype, str] = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}


def tol_for(dtype: torch.dtype) -> float:
    """Return the numerical tolerance for *dtype*.

    Falls back to the ``fp32`` tolerance when the dtype is not in the
    canonical map so that tests never silently use zero tolerance.

    Args:
        dtype: A :class:`torch.dtype` (e.g. ``torch.float32``).

    Returns:
        float: The absolute tolerance to use with ``torch.allclose`` or
        similar comparison functions.

    Example::

        >>> import torch
        >>> from tests.utils.tolerances import tol_for
        >>> tol_for(torch.complex64)
        1e-05
        >>> tol_for(torch.float16)
        0.005
    """
    key = _DTYPE_TO_KEY.get(dtype)
    if key is None:
        return TOLERANCES["fp32"]
    return TOLERANCES[key]
