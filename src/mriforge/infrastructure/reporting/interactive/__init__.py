r"""Interactive (plotly) reporting layer.

Produces self-contained inline ``<div>`` fragments that the QC HTML assembler
drops beside the static PNGs. Every builder returns ``None`` when plotly is
absent or the required data is missing, so the report degrades cleanly to the
embedded static figures — plotly is a soft dependency (``pip install
mriforge[viz]``).
"""

from __future__ import annotations

from .figures import (
    comparison_div,
    group_iqm_div,
    learning_curve_div,
    metric_distribution_div,
)
from .plotly_util import has_plotly, plotlyjs_script
from .viewer_2d import subject_viewer_2d_div
from .viewer_3d import volume_viewer_3d_div

__all__ = [
    "comparison_div",
    "group_iqm_div",
    "has_plotly",
    "learning_curve_div",
    "metric_distribution_div",
    "plotlyjs_script",
    "subject_viewer_2d_div",
    "volume_viewer_3d_div",
]
