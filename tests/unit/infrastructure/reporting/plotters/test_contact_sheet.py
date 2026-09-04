import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters import contact_sheet  # noqa: F401


def test_contact_sheet_montages_pngs(tmp_path):
    # create two dummy png figures to montage
    for name in ("fig_a", "fig_b"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(tmp_path / f"{name}.png")
        plt.close(fig)
    out = tmp_path / "contact_sheet.pdf"
    res = get("contact_sheet")(pd.DataFrame(), out, figure_dir=tmp_path,
                               formats=("pdf",), dpi=150)
    assert res is not None and res.exists()
