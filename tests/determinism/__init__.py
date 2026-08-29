"""Determinism test suite.

Validates that the framework produces bit-exact reproducible results across
independent runs under identical seeds — a prerequisite for reliable ablations
and cluster-distributed training.

Sub-modules
-----------
``test_determinism`` : single-step parameter-delta hash, DataLoader worker
    seed isolation.

Reference hashes
----------------
Calibrated parameter-delta hashes are stored in
``tests/determinism/_reference_hashes.json``.  On a fresh cluster node (new
PyTorch version, CUDA driver, …) re-generate with::

    pytest tests/determinism/ -m determinism -k "calibrate" --recalibrate

or simply delete the key from the JSON and run the suite once — the test
will skip and print a calibration command.
"""
