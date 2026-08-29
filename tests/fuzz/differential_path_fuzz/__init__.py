"""Differential-path fuzz sub-package: dispatch-equivalence and AMP-stability.

Tests two equivalence properties (testing_implementation_plan.md §5.21):

1. **Strategy-dispatch equivalence**: a config written with the legacy
   ``training.training_mode`` string and the same config written with an
   explicit ``training.strategy_class`` must resolve to the same strategy
   class via ``TrainingStrategyFactory.get_strategy_class``.

2. **AMP stability**: a tiny forward op executed under AMP (fp16 autocasting)
   vs plain fp32 must agree within a loose absolute tolerance (τ ≈ 3e-2).
   GPU-only paths are guarded by ``@pytest.mark.gpu``.

Both tests require ``hypothesis`` (``pytest.importorskip`` guard keeps
collection green when the library is absent).  The fuzz marker
(``@pytest.mark.fuzz``) means they are excluded from ``make test`` but
collected explicitly on the cluster with::

    pytest tests/fuzz/ -m fuzz -v
"""
