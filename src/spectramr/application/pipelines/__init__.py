"""Application-layer pipelines.

Concrete pipeline modules (e.g. ``pggs_pipeline``) are imported directly by
their consumers — the model factory registers ``PGGSReconstructionPipeline``
lazily (``models/factories/model_factory.py``), so this package does not need to
eager-import them. The deprecated ``inference_pipeline`` module was removed
2026-07-02 after ``cli.app.predict`` was rewired onto the SSOT
``run_inference_pipeline`` (``spectramr.pipelines.infer``).
"""

__all__: list[str] = []
