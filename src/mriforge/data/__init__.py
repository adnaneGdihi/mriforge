"""Data Module - Unified MRI Data Loading.

This module handles all data loading, preprocessing, and augmentation tasks.
The canonical file→Dataset/DataLoader entry point is
:class:`~mriforge.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`
(the Phase-7 SSOT). ``ConsolidatedDatasetFactory`` is **deprecated** — it emits
a ``DeprecationWarning`` on use and is re-exported here only for back-compat.

.. mermaid::

    graph TD
        A[Config] -->|Settings| B(DataPipelineDirector)
        B -->|Builds| C[Builders]
        C -->|Uses| D[IO Strategies]
        C -->|Uses| E[Transforms]
        C -->|Constructs| F[TorchIO Queue]
        F -->|Yields| G[DataLoader]
        G -->|Batches| H[Training Loop]

        subgraph IO Layers
            D1[FastMRI H5]
            D2[NIfTI]
            D3[DICOM]
        end
        D --> D1 & D2 & D3
"""

from mriforge.data.io_strategies import IOStrategyFactory

# ``ConsolidatedDatasetFactory`` was re-exported here for the historical
# import path. It is deleted (6a-iii): a 435-LOC parallel implementation of
# ``DataPipelineDirector``'s path, deprecated since the 2026-05-28 audit and
# with no production caller left. The eager import was also the reason the
# module could not simply rot away unnoticed -- importing ``mriforge.data``
# constructed it.
__all__ = ["IOStrategyFactory"]
