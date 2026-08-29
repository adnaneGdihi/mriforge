"""Directors: the two orchestrators that are actually reached.

``TrainingPipelineDirector``, ``InferencePipelineDirector`` and
``ExperimentDirector`` were removed — a parallel pipeline stack with no caller
anywhere in ``src/``. The live training stack is
``infrastructure/training/builders/`` (``TrainingEnvironmentDirector``, entered
from ``pipelines/train.py``); inference goes through
``pipelines/infer.py::run_inference_pipeline``.

- ``CheckpointDirector`` — checkpoint save/resume, used by
  ``pipelines/training_loop.py`` and ``pipelines/train.py``.
- ``DataPipelineDirector`` — the data SSOT every higher layer builds datasets
  through (non-negotiable #7). Imported from
  ``mriforge.infrastructure.builders.directors.data_pipeline_director``.
"""

from mriforge.infrastructure.builders.directors.checkpoint_director import (
    CheckpointDirector,
    CheckpointState,
)

__all__ = [
    "CheckpointDirector",
    "CheckpointState",
]
