import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters.mri_specific import (  # noqa: F401
    acceleration_sweep,
)


def test_acceleration_sweep_renders(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for R in (2, 4, 8):
        for seed in range(3):
            rows.append({"method": "ours", "metric": "psnr",
                         "acceleration": R, "value": 35 - R + rng.normal(0, 0.3),
                         "split": "test"})
    out = tmp_path / "acceleration_sweep.pdf"
    res = get("acceleration_sweep")(pd.DataFrame(rows), out, metric="psnr",
                                    formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_acceleration_sweep_two_seed_band_renders(tmp_path):
    # ddof=1 cross-seed band: with exactly 2 seeds the sample std is defined
    # (ddof=0 would understate it by √(1/2)≈29%); the figure must still render.
    rng = np.random.default_rng(3)
    rows = []
    for R in (2, 4, 8):
        for seed in range(2):
            rows.append({"method": "ours", "metric": "psnr",
                         "acceleration": R, "value": 35 - R + rng.normal(0, 0.5),
                         "split": "test"})
    out = tmp_path / "acceleration_sweep_2seed.pdf"
    res = get("acceleration_sweep")(pd.DataFrame(rows), out, metric="psnr",
                                    formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
