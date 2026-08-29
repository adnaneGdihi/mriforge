"""WS-7 TE-004: the deprecated ``strategy_collaborators`` re-export shim must not
emit a module-level ``DeprecationWarning`` at import time.

The project promotes ``DeprecationWarning`` from ``mriforge.*`` to an error under
strict ``filterwarnings`` (see ``.claude/rules/testing.md``); a warn-on-import on
a still-first-class shim would break any importer / star-import of it. The
deprecation remains documented in the module docstring, and the re-exports stay
functional.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def test_strategy_collaborators_import_emits_no_deprecation_warning() -> None:
    mod = "mriforge.infrastructure.training.strategy_collaborators"
    sys.modules.pop(mod, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sc = importlib.import_module(mod)
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deps, f"unexpected DeprecationWarning(s): {[str(w.message) for w in deps]}"
    # The shim must still re-export its utilities.
    assert hasattr(sc, "StandardOptimizerStepper")
    assert hasattr(sc, "AMPPolicy")
    assert hasattr(sc, "MetricsReporter")
