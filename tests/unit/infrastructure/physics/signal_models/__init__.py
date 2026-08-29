"""Package marker — load-bearing, do not delete.

This directory's ``test_registry.py`` collides with
``tests/unit/models/diffusion/samplers/test_registry.py``. Without markers on
both chains, both import as the bare module ``test_registry`` and pytest
aborts the whole session with "import file mismatch".

See ``tests/unit/test_module_basenames_are_unique.py``.
"""
