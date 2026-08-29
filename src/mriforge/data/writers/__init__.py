"""Output writers (Phase 4d).

On-disk write helpers for inference outputs. The main public entry point
is :class:`mriforge.data.writers.output_writer.OutputWriter`.
"""

from mriforge.data.writers.output_writer import OutputWriter, write_output

__all__ = ["OutputWriter", "write_output"]
