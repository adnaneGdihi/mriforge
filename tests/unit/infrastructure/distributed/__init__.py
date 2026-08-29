"""Package marker — load-bearing, do not delete.

`tests/smoke/test_strategies.py` has the same basename as this directory's
`test_strategies.py`. Without `__init__.py` here, rootdir-relative insertion
gives both files the module name `test_strategies`, and pytest aborts the
WHOLE session with "import file mismatch" — `pytest tests/` collected 48,528
tests and ran zero. Every parent already carries a marker; this leaf was the
only gap.
"""
