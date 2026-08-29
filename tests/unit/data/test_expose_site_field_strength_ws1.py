"""WS1-schemas-data-02/03: wire ``data.expose_field_strength`` /
``data.expose_site_id`` end-to-end on the data side. The strategy
(``ConditioningMixin._domain_conditioning_context``) and site-aware models
(``siren_universal``) already read ``batch['site_id']`` / ``batch['field_strength']``
— these tests cover the previously-missing data layer that populates them.
"""

from __future__ import annotations

import torch

from mriforge.data.builders.sfc_conformal_fmri_keys_wrapper import (
    SFCConformalFMRIKeysWrapper,
)
from mriforge.data.transforms.sfc_conformal_fmri_keys import (
    attach_field_strength,
    attach_site_id,
)


def _batch(b: int = 2) -> dict:
    return {"image": torch.zeros(b, 1, 4, 4)}


def test_attach_site_id_default_when_no_label() -> None:
    out = attach_site_id(_batch(2))
    assert out["site_id"].tolist() == [0, 0]


def test_attach_site_id_hashes_per_sample_labels() -> None:
    b = _batch(2)
    b["site"] = ["siteA", "siteB"]
    out = attach_site_id(b)
    assert out["site_id"].shape == (2,)
    assert out["site_id"].dtype == torch.long


def test_attach_field_strength_fills_default() -> None:
    out = attach_field_strength(_batch(2), default=0.3)
    assert torch.allclose(out["field_strength"], torch.tensor([0.3, 0.3]))


def test_attach_field_strength_uses_per_sample_values() -> None:
    b = _batch(2)
    b["field_strength"] = [0.3, 3.0]
    out = attach_field_strength(b)
    assert torch.allclose(out["field_strength"], torch.tensor([0.3, 3.0]))


def test_attach_field_strength_noop_without_source_or_default() -> None:
    out = attach_field_strength(_batch(2))
    assert "field_strength" not in out


class _StubDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return _batch(2)


def test_wrapper_populates_both_keys_when_flags_set() -> None:
    w = SFCConformalFMRIKeysWrapper(
        _StubDataset(),
        expose_site_id=True,
        expose_field_strength=True,
        field_strength_default=0.55,
    )
    sample = w[0]
    assert "site_id" in sample
    assert torch.allclose(sample["field_strength"], torch.tensor([0.55, 0.55]))


def test_wrapper_noop_when_flags_unset() -> None:
    w = SFCConformalFMRIKeysWrapper(_StubDataset())
    sample = w[0]
    assert "site_id" not in sample
    assert "field_strength" not in sample
