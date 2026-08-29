import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import get
from mriforge.infrastructure.reporting.plotters import calibration_coverage  # noqa: F401


def test_calibration_coverage_renders(tmp_path):
    rng = np.random.default_rng(0)
    # nominal vs empirical coverage pairs
    nominal = np.linspace(0.1, 0.9, 9)
    empirical = np.clip(nominal + rng.normal(0, 0.02, nominal.shape), 0, 1)
    cov = pd.DataFrame({"nominal": nominal, "empirical": empirical, "method": "ours"})
    out = tmp_path / "calibration_coverage.pdf"
    res = get("calibration_coverage")(cov, out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
