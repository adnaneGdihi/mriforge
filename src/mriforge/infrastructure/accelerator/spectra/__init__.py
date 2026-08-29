"""SPECTRA hardware inference backend (Workstream C).

Wires the SPECTRA accelerator (WCSR systolic array + NMAE physics-remap, driven
over the RoCC contract) into the framework as a first-class *execution
provider*. Inference-on-SPECTRA is not a training paradigm, so it is exposed
through an executor the inference layer can select (``get_spectra_executor``)
rather than through the training ``STRATEGY_CLASS_PATHS`` dispatcher.

Honesty contract: the graph-to-RoCC lowering is **contract-level** until the
TVM-BYOC compiler (``compiler/spectra_byoc`` in the SPECTRA repo) and the
co-sim runtime land. The adapter/executor here capture the torch graph and
define the dispatch seam; they do not yet execute on real RTL. See the module
docstrings and the ``NotImplementedError`` paths for exactly where the stub is.
"""

from __future__ import annotations

from mriforge.infrastructure.accelerator.spectra.spectra_backend import (
    SpectraBackendAdapter,
    SpectraLoweringError,
)
from mriforge.infrastructure.accelerator.spectra.spectra_executor import (
    SpectraExecutor,
    get_spectra_executor,
)

__all__ = [
    "SpectraBackendAdapter",
    "SpectraExecutor",
    "SpectraLoweringError",
    "get_spectra_executor",
]
