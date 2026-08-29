"""Fuzz / property-based test package.

Sub-packages
------------
schema_fuzz         Hypothesis-driven schema round-trips and audit-vs-train
                    disagreement detection.
shape_dtype_fuzz    Hypothesis-driven model shape/dtype envelope tests.
loss_composition_fuzz  Random loss composition finite-gradient defect detection.
audit_ladder_fuzz   Atheris-driven mutation fuzzing for config_health_checker.

All pytest files in this package require optional libraries:
- ``hypothesis`` and ``hypothesis-jsonschema`` for the property-based suites.
- ``atheris`` for the coverage-guided fuzzer runner.

Each pytest file guards its own optional import with
``pytest.importorskip(...)`` so collection succeeds (and tests are skipped
cleanly) when the libraries are absent.  The ``atheris_runner.py`` script uses
its own guard and exits 0 when atheris is absent.

Fuzz tests are marked ``@pytest.mark.fuzz`` which is declared in
``pyproject.toml`` ``[tool.pytest.ini_options].markers``.  They are
not executed by the default ``make test`` target (which filters to
``tests/unit/``); they are collected explicitly on the cluster with::

    pytest tests/fuzz/ -m fuzz -v

or via the ``make fuzz-audit-ladder`` target for the standalone atheris
runner.
"""
