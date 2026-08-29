"""HPO infrastructure utilities.

Houses helpers that the HPO coordinator (``infrastructure/coordination/``)
and the HPO pipeline (``pipelines/``) both need. Lives in the
infrastructure layer so the coordinator's import is inward-pointing
(was a layering violation when these helpers lived in ``pipelines/``).
See ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 2.
"""

from mriforge.infrastructure.hpo.dotted_override import apply_dotted_override

__all__ = ["apply_dotted_override"]
