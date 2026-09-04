import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters import ablation_heatmap  # noqa: F401


def test_ablation_heatmap_renders(tmp_path):
    rows = []
    for a in ("lr1", "lr2", "lr3"):
        for b in ("d1", "d2"):
            rows.append({"axis_x": a, "axis_y": b, "value": hash((a, b)) % 10})
    out = tmp_path / "ablation_heatmap.pdf"
    res = get("ablation_heatmap")(pd.DataFrame(rows), out,
                                  x="axis_x", y="axis_y", value="value",
                                  formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_ablation_heatmap_materializes_array_once(monkeypatch, tmp_path):
    # Compute-waste guard: the cell-label loop must reuse a single
    # pivot.to_numpy() materialisation, not call it once per cell (was O(n²)).
    rows = [
        {"axis_x": a, "axis_y": b, "value": 1.0}
        for a in ("c1", "c2", "c3")
        for b in ("r1", "r2")
    ]
    out = tmp_path / "heatmap_count.pdf"

    import pandas as _pd

    calls = {"n": 0}
    real_to_numpy = _pd.DataFrame.to_numpy

    def counting_to_numpy(self, *a, **k):
        calls["n"] += 1
        return real_to_numpy(self, *a, **k)

    monkeypatch.setattr(_pd.DataFrame, "to_numpy", counting_to_numpy)
    get("ablation_heatmap")(_pd.DataFrame(rows), out, x="axis_x", y="axis_y",
                            value="value", formats=("pdf",), dpi=150)
    # 3x2 grid = 6 cells. The old code called to_numpy() 1 (imshow) + 6 (cells)
    # = 7 times; the fix calls it once for the pivot. Allow a small constant for
    # any incidental DataFrame→array conversions, but it must NOT scale with the
    # 6 cells.
    assert calls["n"] <= 3
