import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters.mri_specific import (  # noqa: F401
    kspace_error_spectrum,
)


def test_kspace_error_spectrum_renders(tmp_path):
    cases = [{
        "prediction": np.random.rand(64, 64).astype(np.float32),
        "target": np.random.rand(64, 64).astype(np.float32),
        "rank": "median", "metrics": {}, "case_id": "s0",
    }]
    out = tmp_path / "kspace_error_spectrum.pdf"
    res = get("kspace_error_spectrum")(pd.DataFrame(), out, cases=cases,
                                       formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
