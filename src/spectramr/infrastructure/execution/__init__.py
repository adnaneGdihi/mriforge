"""Execution backends — the WHERE axis of the execution-mode model.

Pipeline-agnostic backends that run a ``spectramr <verb>`` invocation in a given
place (Docker / Apptainer / SLURM). The in-process ``LocalBackend`` lives in the
``cli`` layer (it must call into the pipeline). See
:mod:`spectramr.infrastructure.execution.backends`.
"""

from spectramr.infrastructure.execution.backends import (
    ApptainerBackend,
    DockerBackend,
    ExecutionBackend,
    SpectraMRInvocation,
    ResourceSpec,
    RunHandle,
    SlurmBackend,
    export_launch_env,
    resolve_launch_provenance,
)

__all__ = [
    "ApptainerBackend",
    "DockerBackend",
    "ExecutionBackend",
    "SpectraMRInvocation",
    "ResourceSpec",
    "RunHandle",
    "SlurmBackend",
    "export_launch_env",
    "resolve_launch_provenance",
]
