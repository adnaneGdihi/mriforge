import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import get
from mriforge.infrastructure.reporting.plotters.mri_specific import recon_panel  # noqa: F401


def _cases():
    out = []
    for rank, q in (("best", 0.95), ("median", 0.8), ("worst", 0.6)):
        out.append({
            "input": np.random.rand(32, 32).astype(np.float32) * 0.5,
            "prediction": np.random.rand(32, 32).astype(np.float32),
            "target": np.random.rand(32, 32).astype(np.float32),
            "metrics": {"ssim": q, "psnr": 20 + 10 * q},
            "rank": rank, "case_id": f"s_{rank}",
        })
    return out


def test_recon_panel_renders(tmp_path):
    out = tmp_path / "mri_recon_panel.pdf"
    res = get("mri_recon_panel")(pd.DataFrame(), out, cases=_cases(),
                                 formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_recon_panel_no_cases_returns_none(tmp_path):
    res = get("mri_recon_panel")(pd.DataFrame(), tmp_path / "x.pdf", cases=None)
    assert res is None


def test_recon_panel_renders_without_layout_warning(tmp_path):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get("mri_recon_panel")(pd.DataFrame(), tmp_path / "p.pdf", cases=_cases(),
                               formats=("pdf",), dpi=120)
    msgs = [str(w.message) for w in caught]
    assert not any("tight_layout" in m for m in msgs), msgs
