import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import get
from mriforge.infrastructure.reporting.plotters import bland_altman  # noqa: F401


def test_bland_altman_renders(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    rows = []
    for s in range(n):
        ref = rng.uniform(0.5, 1.5)
        rows.append({"subject_id": f"s{s}", "method": "ours", "metric": "t2",
                     "value": ref + rng.normal(0, 0.05), "reference": ref, "split": "test"})
    out = tmp_path / "bland_altman.pdf"
    res = get("bland_altman")(pd.DataFrame(rows), out, metric="t2",
                              formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
