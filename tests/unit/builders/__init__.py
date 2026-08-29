"""Legacy flat builder tests.

Load-bearing: without this file pytest names the modules here after their bare
basename, so ``test_loss_builder.py`` / ``test_optimization_builder.py`` collide
with their namesakes under ``tests/unit/infrastructure/training/builders/`` and
the run aborts with "import file mismatch". Do not delete.
"""
