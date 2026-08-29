"""Package marker — load-bearing, do not delete.

``samplers/test_registry.py`` below this directory collides with
``tests/unit/infrastructure/physics/signal_models/test_registry.py``.

``samplers/`` already carried its own ``__init__.py``, but that alone does
nothing: pytest's ``prepend`` import mode walks UP from the test file only
while every parent has a marker, so one gap anywhere in the chain drops the
module back to its bare basename. This directory was that gap, which is why a
marker two levels down did not prevent the collision.

See ``tests/unit/test_module_basenames_are_unique.py``.
"""
