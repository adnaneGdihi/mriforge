"""Fitness function: no NEW signature dispatch via ``except TypeError`` (SAQ-001).

Flags ``try: f(x, t) / except TypeError: f(x)`` -- using the exception as a
signature probe. It is unsound because ``TypeError`` cannot distinguish "this
callable has no such parameter", raised at the call boundary, from "the body
raised a ``TypeError`` three frames down". The second is a real bug, and the
retry answers it by silently running a *degraded* call: an un-time-conditioned
diffusion model, an unconditioned generator. Those still train and still report
a falling loss, so the failure never announces itself. Introspect the signature
instead -- ``_callable_accepts_kwarg`` for keyword dispatch,
``DiffusionTrainingStrategy._generator_accepts_time`` for positional.

Ratcheted, like every other detector here: the sites that predate the gate are
baselined and only NEW ones fail. #1189 fixed six (three named in the issue plus
three the sweep found); the remainder are the baseline, and the reason this gate
exists at all is that the previous sweep stopped at one file.
"""

from __future__ import annotations

import pytest

from ._fitness_lib import ratchet, scan_exception_dispatch

pytestmark = pytest.mark.architecture


def test_no_new_exception_based_signature_dispatch() -> None:
    current = scan_exception_dispatch()
    new = ratchet(
        "exception_dispatch.txt",
        current,
        header="Signature dispatch via `except TypeError` retry (SAQ-001, #1189)",
    )
    assert not new, (
        "New `except TypeError` signature probe detected — dispatch on the "
        "signature instead (`_callable_accepts_kwarg` for a keyword, "
        "`_generator_accepts_time` for positional time), so a TypeError raised "
        "inside the forward propagates rather than degrading the call:\n  "
        + "\n  ".join(sorted(new))
    )
