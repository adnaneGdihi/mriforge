"""Registry contract test suites.

Parametric tests that cover *every* entry in each component registry:
models, losses, metrics, strategies.  The cheap "is-registered + metadata
well-formed" assertions run in the default lane (no extra -m flag required).
The expensive "instantiate + forward + gradient" assertions are guarded by
``@pytest.mark.slow`` so they only execute on the cluster.
"""
