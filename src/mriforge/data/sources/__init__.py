"""Volume sources for the sim2rank sweep.

A narrow Protocol in the data layer, not a ``DataPipelineDirector`` route: the
director needs a frozen ``TrainingSettings``, and sim2rank is an argparse analysis
harness with no training config -- fabricating a fake one to satisfy a rule that
was never about this would itself be a facade. See
:mod:`mriforge.data.sources.sim2rank_source`.
"""

from __future__ import annotations

from mriforge.data.sources.fastmri_brain_source import (
    ALLOWED_ACQUISITIONS,
    ContrastMismatchError,
    FastMRIBrainSource,
    QCNotApprovedError,
    acquisition_from_filename,
)
from mriforge.data.sources.m4raw_source import M4RawSource
from mriforge.data.sources.sim2rank_source import Sim2RankSlice, Sim2RankVolumeSource

__all__ = [
    "ALLOWED_ACQUISITIONS",
    "ContrastMismatchError",
    "FastMRIBrainSource",
    "M4RawSource",
    "QCNotApprovedError",
    "Sim2RankSlice",
    "Sim2RankVolumeSource",
    "acquisition_from_filename",
]
