"""Differential tests — cross-implementation correctness harnesses.

Each module in this package compares a framework operator to an independent
reference implementation (SigPy, BART, torchkbnufft direct call, …).

Markers used
------------
- ``differential`` : cross-implementation comparison test
- ``requires_sigpy`` : SigPy must be importable
- ``requires_bart``  : BART CLI must be on ``$PATH``

Runtime guards
--------------
All reference-library imports are wrapped with ``pytest.importorskip`` or
``shutil.which`` checks so that the entire module can be *collected* without
errors even when the reference library is absent — the tests will skip at
runtime rather than failing to collect.
"""
