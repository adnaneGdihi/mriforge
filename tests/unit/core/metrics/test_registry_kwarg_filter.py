"""Registry robustness: get() filters kwargs to each metric constructor (B-2.9 fix).

The metrics computer constructs every metric as ``get(name, device=...)``. Metrics
whose ``__init__`` does not declare ``device`` (e.g. _QCRCBase) must not crash (the
error was previously caught and stored as a silent NaN). The fix filters kwargs to
the constructor signature; this regression guards it. Also guards that the
quantitative subpackage registers via a plain ``import mriforge.core.metrics``.
"""

from __future__ import annotations

import torch

import mriforge.core.metrics  # noqa: F401 - triggers registration (incl. quantitative)
from mriforge.core.metrics.registry import get_metric, register_metric


def test_quantitative_metrics_register_via_package_import() -> None:
    # The walk skips subpackages; core/metrics/__init__ must import quantitative
    # explicitly so these register without loading a VF strategy.
    for name in ("gaussian_nll", "synthseg_dice", "conformal_risk_control"):
        assert get_metric(name, device="cpu") is not None


def test_get_drops_unaccepted_kwargs() -> None:
    # conformal_risk_control's _QCRCBase.__init__(alpha, loss_bound) has no device;
    # get(device=...) must filter it out, not crash.
    m = get_metric("conformal_risk_control", device=torch.device("cpu"))
    assert m is not None


def test_get_passes_accepted_kwargs() -> None:
    @register_metric("kwarg_filter_probe_metric")
    class _Probe:
        def __init__(self, alpha: float = 0.1, device: object = None) -> None:
            self.alpha = alpha

        def __call__(self, p, t, **_):  # noqa: ANN001
            return 0.0

    m = get_metric("kwarg_filter_probe_metric", alpha=0.42, device="cpu", bogus=1)
    assert m.alpha == 0.42  # accepted kwarg passed; 'bogus' dropped, no crash
