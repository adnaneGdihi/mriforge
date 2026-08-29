"""Synthetic data generators for opt-in domain simulation.

The simulators here are not wired into ``ConsolidatedDatasetFactory``;
they are reusable physics utilities consumed by tests and
forward-looking ULF→HF experiments. Re-exporting them gives downstream
callers a public API surface to import from, instead of reaching into
``mriforge.data.synthesis.<file>`` directly.

See ``TODO/audit/16_data_layer.md`` F6 for the orphan-module finding
this re-export resolves.
"""

from mriforge.data.synthesis.vlf_simulator import VLFSimulator

__all__ = ["VLFSimulator"]
