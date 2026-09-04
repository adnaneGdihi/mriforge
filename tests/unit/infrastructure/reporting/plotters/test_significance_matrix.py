import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters import significance_matrix  # noqa: F401


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for method, mu in (("a", 30.0), ("b", 31.0), ("c", 33.0)):
        for s in range(30):
            rows.append({"method": method, "metric": "psnr", "subject_id": f"s{s}",
                         "value": float(rng.normal(mu, 1.0)), "split": "test"})
    return pd.DataFrame(rows)


def test_significance_matrix_renders(tmp_path):
    out = tmp_path / "significance_matrix.pdf"
    res = get("significance_matrix")(_frame(), out, metric="psnr",
                                     formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
