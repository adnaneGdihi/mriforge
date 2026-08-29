"""Package marker — load-bearing, do not delete.

``tests/smoke/test_strategies.py`` has the same basename as this directory's
``test_strategies.py``. Without ``__init__.py`` here, rootdir-relative
insertion gives both files the module name ``test_strategies`` and pytest
aborts the WHOLE session with "import file mismatch" — a full collect reports
``Interrupted: 1 error during collection`` and runs zero tests.

Same failure, same basename, different leaf as
``tests/unit/infrastructure/distributed/__init__.py``, whose marker was added
first. ``tests/unit/data/`` and every parent above it already carry a marker;
this leaf was the gap. ``tests/unit/test_module_basenames_are_unique.py``
now fails fast if a fourth one appears.
"""
