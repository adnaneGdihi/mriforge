#!/usr/bin/env python
"""Application Use Cases Package.

The live use cases are imported directly (not re-exported here) by their
consumers: ``HPOUseCase`` (CLI ``hpo``) and ``NRMetricValidationUseCase`` (CLI
``meta-evaluate``). ``MRFDictlessMatcher`` moved to
``spectramr.domain.services.mrf_dictless_matching`` (the MRF training strategies
now import it rightward, infrastructure → domain, instead of leftward into this
layer); ``mrf_dictless_matching_use_case`` is a re-export shim.
The train/infer paths run through ``pipelines/train.py`` / ``pipelines/infer.py``
and use no use-case object.

NOTE (2026-06-11): the legacy dead use-case layer (``use_cases.py``,
``training_use_case.py``, ``transfer_learning_use_case.py``, ``factory.py``, the
``*_use_case`` stubs, and the orphaned ``infrastructure/services/training_orchestration``
cluster they wrapped) was deleted — it was unreachable from every entry point
and several files imported a removed ``domain.repositories`` package.
"""

__all__: list[str] = []
