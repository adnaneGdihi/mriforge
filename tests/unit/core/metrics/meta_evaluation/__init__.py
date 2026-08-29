"""Meta-evaluation unit tests.

Present so this package has a unique dotted module path. Without it,
``test_figures.py`` here and ``tests/unit/infrastructure/reporting/interactive/
test_figures.py`` both import as bare ``test_figures`` and pytest aborts
collection with an import-file-mismatch error.
"""
