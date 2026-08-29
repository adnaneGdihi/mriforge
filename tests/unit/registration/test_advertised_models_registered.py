"""Audit-surface-lie gate: every advertised model_type must resolve.

The framework dispatches models dynamically by name from YAML config through
the schema -> builder -> registry chain. A name listed in
``config.validation_constants.VALID_MODEL_TYPES`` therefore makes a promise:
``model_type: <name>`` passes the Tier-0 schema audit, so it MUST resolve in
``MODEL_REGISTRY`` after ``populate_model_registry()`` — otherwise training
dies with a ``KeyError`` at model-build time (an "audit-surface lie": green
audit, runtime crash).

Gate semantics (mirrors the architecture-fitness allowlist pattern):

* An advertised name that is registered -> pass.
* An advertised name that is NOT registered but IS in
  ``tests/contracts/_known_unregistered_advertised.json`` -> xfail (known
  backlog; the allowlist must shrink toward empty as impls are wired in).
* An advertised name that is neither registered nor allow-listed -> HARD FAIL
  (a NEW audit-surface lie / regression).

Wave 1 (2026-05-22): 10 names wired into population (sugar_sdf, gs_mri,
low_field_augmented, latent_aligner, compression_efficient_unet,
latent_input_wrapper, domain_discriminator, cross_scanner_adaptation,
multi_domain_extractor, conditioning_encoder).

Wave 2 (2026-05-22): the remaining 71 backlog names were all PURGED from
VALID_MODEL_TYPES — each was either a name-variant duplicate of an
already-registered model (e.g. vq_vae->vqvae, stylegan->stylegan2) or had no
implementing class anywhere. None had a unique unregistered implementation to
wire. The backlog JSON is now empty; the allowlist must remain empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mriforge.config.validation_constants import VALID_MODEL_TYPES
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY

_ALLOWLIST_PATH = Path(__file__).parent.parent.parent / "contracts" / "_known_unregistered_advertised.json"

_NEWLY_WIRED = (
    # Wave 1 (2026-05-22) — wired into init_registry.py walk list
    "sugar_sdf",
    "gs_mri",
    "low_field_augmented",
    "latent_aligner",
    "compression_efficient_unet",
    "latent_input_wrapper",
    "domain_discriminator",
    "cross_scanner_adaptation",
    "multi_domain_extractor",
    "conditioning_encoder",
)


def _load_backlog() -> frozenset[str]:
    try:
        data = json.loads(_ALLOWLIST_PATH.read_text())
        return frozenset(data.get("unregistered_advertised", []))
    except Exception:
        return frozenset()


_BACKLOG = _load_backlog()


@pytest.fixture(scope="module", autouse=True)
def _populated() -> None:
    populate_model_registry()


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", sorted(set(VALID_MODEL_TYPES)), ids=lambda n: n)
def test_advertised_model_type_resolves(name: str) -> None:
    """Every VALID_MODEL_TYPES name resolves, else is a tracked backlog item."""
    if name in MODEL_REGISTRY:
        return
    if name in _BACKLOG:
        pytest.xfail(f"{name} advertised but not registered (known backlog)")
    pytest.fail(
        f"NEW audit-surface lie: '{name}' is in VALID_MODEL_TYPES but not in "
        f"MODEL_REGISTRY after population, and not in the backlog allowlist "
        f"({_ALLOWLIST_PATH.name}). Either wire its impl into population, "
        f"register an alias, or remove it from VALID_MODEL_TYPES."
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("name", _NEWLY_WIRED, ids=lambda n: n)
def test_newly_wired_model_is_registered(name: str) -> None:
    """Regression: the 2026-05-22 wired-in models must stay registered."""
    assert name in MODEL_REGISTRY, (
        f"'{name}' regressed — it was wired into populate_model_registry() "
        f"but no longer resolves."
    )
    assert name not in _BACKLOG, (
        f"'{name}' was wired in; remove it from the backlog allowlist."
    )
