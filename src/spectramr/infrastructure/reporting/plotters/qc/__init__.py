r"""QC plotters.

A themed plotter family for QC-oriented reporting:

- ``group_strip``    — group IQM distribution, one horizontal box+strip per
  metric with Tukey-outlier flagging (the group report).
- ``subject_mosaic`` — per-case QC mosaic: prediction (with foreground-mask
  contour), target, error map, and a background-noise reportlet.
- ``carpet_spike``   — cases-by-column error carpet + a per-case spike/ghost strip.

Each module exposes ``make(df, out_path, **kwargs) -> Path | None`` and is
registered in :mod:`spectramr.infrastructure.reporting.plotters`.
"""

from __future__ import annotations
