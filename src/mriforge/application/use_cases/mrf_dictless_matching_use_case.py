"""Re-export shim: ``MRFDictlessMatcher`` moved to ``domain/services``.

The dictionary-free MRF matcher is a pure numpy/torch primitive with no
``application`` dependencies. It was imported *leftward* by the MRF training
strategies (infrastructure → application), a layer-direction violation
(pitfall #13). The implementation now lives at
:mod:`mriforge.domain.services.mrf_dictless_matching`; this module re-exports it
so existing imports (and the ``test_dead_orchestration_deleted_2026_06``
live-module pin) keep working. No ``DeprecationWarning`` — the repo promotes
``mriforge.*`` deprecation warnings to errors under test.
"""

from __future__ import annotations

from mriforge.domain.services.mrf_dictless_matching import (
    MRFDictionary,
    MRFDictlessMatcher,
)

__all__ = ["MRFDictionary", "MRFDictlessMatcher"]
