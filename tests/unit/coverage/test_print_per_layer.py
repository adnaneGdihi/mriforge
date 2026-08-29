"""Tests for the per-layer coverage table.

The behaviour worth pinning is the layer LABEL. This tool feeds two consumers
that both present its output as "coverage of the codebase" -- ``make
test-coverage`` locally and the SLURM array's aggregated report -- so a
mislabelled row is a wrong answer that looks exactly like a right one.

The regression these tests exist for: ``_layer_of`` used to strip ``parts[0]``
unconditionally on the assumption that coverage emits ``mriforge/`` -prefixed
filenames. With ``[tool.coverage.run] source = ["src/mriforge"]`` it does not,
so the strip ate a real layer -- ``infrastructure/training/strategies/x.py``
was reported as ``training/strategies``, and ``core/metrics/*`` merged with
``models/metrics/*`` into one row describing neither.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "coverage" / "print_per_layer.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


per_layer = _load("_print_per_layer", _SCRIPT)


def _xml(classes: list[tuple[str, list[int]]]) -> str:
    """Build a minimal Cobertura document. Each entry is (filename, hit counts)."""
    body = []
    for filename, hits in classes:
        lines = "".join(f'<line number="{i + 1}" hits="{h}"/>' for i, h in enumerate(hits))
        body.append(f'<class filename="{filename}"><lines>{lines}</lines></class>')
    return (
        '<?xml version="1.0"?><coverage><packages><package><classes>'
        + "".join(body)
        + "</classes></package></packages></coverage>"
    )


def _write(tmp_path: Path, classes: list[tuple[str, list[int]]]) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(_xml(classes), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _layer_of -- the label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Package-relative (what `source = ["src/mriforge"]` actually emits).
        ("infrastructure/training/strategies/gan.py", "infrastructure/training"),
        ("infrastructure/physics/fft_ops.py", "infrastructure/physics"),
        ("models/generators/unet.py", "models/generators"),
        ("core/metrics/registry.py", "core/metrics"),
        ("cli/app.py", "cli/app.py"),
        ("main.py", "main.py"),
        # Prefixed (what `source = ["src"]` would emit) must land identically.
        ("mriforge/infrastructure/physics/fft_ops.py", "infrastructure/physics"),
        ("mriforge/models/generators/unet.py", "models/generators"),
        ("mriforge/main.py", "main.py"),
    ],
)
def test_layer_label(filename: str, expected: str) -> None:
    assert per_layer._layer_of(filename) == expected


def test_prefixed_and_unprefixed_agree() -> None:
    """The two source= spellings must not produce two different taxonomies."""
    for tail in ("infrastructure/physics/x.py", "models/losses/y.py", "config/schemas/z.py"):
        assert per_layer._layer_of(tail) == per_layer._layer_of(f"mriforge/{tail}")


def test_distinct_layers_sharing_a_leaf_name_do_not_merge(tmp_path: Path) -> None:
    """core/metrics and models/metrics are different layers, not one.

    Under the old unconditional strip both collapsed to ``metrics/...`` and the
    merged percentage described neither of them.
    """
    xml = _write(
        tmp_path,
        [("core/metrics/a.py", [1, 1, 1, 1]), ("models/metrics/b.py", [0, 0, 0, 0])],
    )
    labels = {r.layer for r in per_layer.layer_rows(xml)}
    assert labels == {"core/metrics", "models/metrics"}


# ---------------------------------------------------------------------------
# layer_rows -- the arithmetic
# ---------------------------------------------------------------------------


def test_rows_sum_hits_per_layer(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        [
            ("models/generators/a.py", [1, 1, 0, 0]),  # 2/4
            ("models/generators/b.py", [1, 0]),  # 1/2
        ],
    )
    (row,) = per_layer.layer_rows(xml)
    assert (row.layer, row.valid, row.covered, row.miss, row.n_files) == (
        "models/generators",
        6,
        3,
        3,
        2,
    )
    assert row.pct == pytest.approx(50.0)


def test_rows_sorted_by_missing_lines_descending(tmp_path: Path) -> None:
    xml = _write(
        tmp_path,
        [("a/x.py", [0]), ("b/y.py", [0, 0, 0]), ("c/z.py", [0, 0])],
    )
    assert [r.miss for r in per_layer.layer_rows(xml)] == [3, 2, 1]


def test_min_valid_filters_small_layers(tmp_path: Path) -> None:
    xml = _write(tmp_path, [("big/x.py", [0] * 10), ("tiny/y.py", [0])])
    assert [r.layer for r in per_layer.layer_rows(xml, min_valid=5)] == ["big/x.py"]


def test_missing_xml_raises_rather_than_reporting_zero(tmp_path: Path) -> None:
    """Absent data must never be presentable as 0% -- that reads as measured."""
    with pytest.raises(FileNotFoundError):
        per_layer.layer_rows(tmp_path / "nope.xml")


def test_empty_xml_raises(tmp_path: Path) -> None:
    xml = _write(tmp_path, [])
    with pytest.raises(ValueError, match="no coverage data"):
        per_layer.layer_rows(xml)


# ---------------------------------------------------------------------------
# totals / format_table
# ---------------------------------------------------------------------------


def test_total_is_consistent_with_the_rows_shown(tmp_path: Path) -> None:
    """A filtered table's TOTAL must sum the visible rows, not the whole repo."""
    xml = _write(tmp_path, [("big/x.py", [1] * 6 + [0] * 4), ("tiny/y.py", [0])])
    rows = per_layer.layer_rows(xml, min_valid=5)
    total = per_layer.totals(rows)
    assert total.valid == sum(r.valid for r in rows) == 10
    assert total.pct == pytest.approx(60.0)


def test_format_table_renders_header_rows_and_total(tmp_path: Path) -> None:
    xml = _write(tmp_path, [("models/generators/a.py", [1, 0])])
    table = per_layer.format_table(per_layer.layer_rows(xml))
    assert "cover%" in table and "layer" in table
    assert "models/generators" in table
    assert "TOTAL" in table


def test_format_table_can_omit_the_total(tmp_path: Path) -> None:
    xml = _write(tmp_path, [("models/generators/a.py", [1, 0])])
    table = per_layer.format_table(per_layer.layer_rows(xml), include_total=False)
    assert "TOTAL" not in table


def test_main_prints_table_and_exits_zero(tmp_path: Path, capsys) -> None:
    xml = _write(tmp_path, [("infrastructure/physics/a.py", [1, 0])])
    assert per_layer.main([str(xml)]) == 0
    assert "infrastructure/physics" in capsys.readouterr().out


def test_main_separates_layers_and_totals_them(tmp_path: Path, capsys) -> None:
    """Two files in one layer collapse; a third layer stays separate."""
    xml = _write(
        tmp_path,
        [
            ("infrastructure/physics/epg.py", [1, 0, 1, 0]),
            ("infrastructure/physics/sense.py", [1, 1]),
            ("core/metrics/dice.py", [1, 1, 1, 0]),
        ],
    )
    assert per_layer.main([str(xml)]) == 0
    out = capsys.readouterr().out
    assert "infrastructure/physics" in out
    assert "core/metrics" in out
    assert "TOTAL" in out
    assert "10" in out, "4 + 2 + 4 measurable lines"


def test_main_no_summary_suppresses_the_total(tmp_path: Path, capsys) -> None:
    xml = _write(tmp_path, [("infrastructure/physics/a.py", [1, 0])])
    assert per_layer.main([str(xml), "--no-summary"]) == 0
    assert "TOTAL" not in capsys.readouterr().out


def test_main_min_valid_hides_small_layers(tmp_path: Path, capsys) -> None:
    xml = _write(tmp_path, [("big/x.py", [0] * 10), ("tiny/y.py", [0])])
    assert per_layer.main([str(xml), "--min-valid", "5"]) == 0
    out = capsys.readouterr().out
    assert "big/x.py" in out
    assert "tiny/y.py" not in out


def test_main_reports_missing_file_without_traceback(tmp_path: Path, capsys) -> None:
    assert per_layer.main([str(tmp_path / "absent.xml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_returns_one_on_empty_coverage(tmp_path: Path) -> None:
    """No data must be an exit code, not a crash and not a 0% table."""
    assert per_layer.main([str(_write(tmp_path, []))]) == 1
