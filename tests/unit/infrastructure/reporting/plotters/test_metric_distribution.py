import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import get
from mriforge.infrastructure.reporting.plotters import metric_distribution  # noqa: F401


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for method, mu in (("baseline", 30.0), ("ours", 33.0)):
        for s in range(40):
            rows.append({"method": method, "metric": "psnr",
                         "value": float(rng.normal(mu, 1.0)),
                         "subject_id": f"s{s}", "split": "test"})
    return pd.DataFrame(rows)


def test_metric_distribution_renders(tmp_path):
    out = tmp_path / "metric_distribution.pdf"
    res = get("metric_distribution")(_frame(), out, metrics=["psnr"],
                                     formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_metric_distribution_does_not_touch_global_rng(tmp_path):
    import numpy as np

    np.random.seed(123)
    before = np.random.get_state()[1][0]
    get("metric_distribution")(_frame(), tmp_path / "d.pdf", metrics=["psnr"],
                               formats=("pdf",), dpi=80)
    after = np.random.get_state()[1][0]
    # the plotter must use a local seeded RNG, leaving the global RNG untouched
    assert before == after


def test_metric_distribution_multi_metric_panels_render(tmp_path):
    """2026-07-01 clarity pass: pretty y-labels + panel letters on multi-panel
    layouts must not break rendering."""
    import numpy as np
    import pandas as pd

    from mriforge.infrastructure.reporting.plotters import metric_distribution

    rng = np.random.default_rng(0)
    rows = []
    for metric in ("psnr", "ssim"):
        for method in ("ours", "baseline"):
            for i in range(12):
                rows.append({"run_id": method, "method": method,
                             "metric": metric, "subject_id": f"s{i}",
                             "value": rng.uniform(0.5, 1.0), "split": "test",
                             "step": -1})
    out = metric_distribution.make(pd.DataFrame(rows), tmp_path / "dist.pdf",
                                   metrics=["psnr", "ssim"],
                                   formats=("pdf",), dpi=150)
    assert out is not None and out.exists()
