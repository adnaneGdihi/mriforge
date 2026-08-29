import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import get
from mriforge.infrastructure.reporting.plotters import forest_plot  # noqa: F401


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for method, mu in (("baseline", 30.0), ("ours", 33.0), ("v2", 32.0)):
        for s in range(30):
            rows.append({"method": method, "metric": "psnr", "subject_id": f"s{s}",
                         "value": float(rng.normal(mu, 1.0)), "split": "test"})
    return pd.DataFrame(rows)


def test_forest_plot_renders(tmp_path):
    out = tmp_path / "forest_plot.pdf"
    res = get("forest_plot")(_frame(), out, metric="psnr", baseline="baseline",
                             formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
