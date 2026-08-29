"""Package marker — load-bearing, do not delete.

This directory's ``test_swin_mamba_kan.py`` has the same basename as
``tests/smoke/exotic/test_swin_mamba_kan.py``. Without ``__init__.py`` here
both import as the bare module ``test_swin_mamba_kan`` and pytest aborts the
whole session with "import file mismatch".

See ``tests/unit/test_module_basenames_are_unique.py``.
"""
