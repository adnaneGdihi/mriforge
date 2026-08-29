"""Public-API surface of the ``mriforge.data`` package (WS7).

``ConsolidatedDatasetFactory`` was re-exported here for back-compat and is now
DELETED (6a-iii) -- a 435-LOC parallel implementation of ``DataPipelineDirector``
with no production caller left. The eager import is gone with it, which also
removes the reason importing ``mriforge.data`` had to be checked for a
deprecation warning: there is no deprecated symbol left to construct.

The warning-free-import test stays anyway. It guards the property, not the one
symbol that used to threaten it.
"""

import warnings


def test_import_does_not_emit_deprecation_warning():
    import importlib

    import mriforge.data as data_pkg

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(data_pkg)  # re-import must stay warning-free


def test_public_exports_are_back_compat():
    import mriforge.data as data_pkg

    assert set(data_pkg.__all__) == {"IOStrategyFactory"}
    assert hasattr(data_pkg, "IOStrategyFactory")
    # The delete must be real: a lingering attribute would let a caller keep
    # importing it from here long after the module is gone.
    assert not hasattr(data_pkg, "ConsolidatedDatasetFactory")


def test_docstring_points_at_director():
    import mriforge.data as data_pkg

    assert "DataPipelineDirector" in (data_pkg.__doc__ or "")
