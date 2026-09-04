"""An acquisition tuple has exactly ONE definition. Issue #828.

``config/schemas/training/pmps.py`` re-declared ``AcquisitionParam`` with the
same eight fields, types and defaults as ``AcquisitionParamsSchema``
(``config/schemas/data.py``) — **except** ``contrast_type``:

===============================  ============================================
data.py                          ``spin_echo | inversion_recovery | gradient_echo | diffusion_weighted | ssfp | mprage``
training/pmps.py                 ``spin_echo | gradient_echo | inversion_recovery | ssfp | mprage``
===============================  ============================================

So the identical acquisition dict validated as a ``data:`` entry and was
**rejected** as a PMPS protocol. Two spellings of one concept that agree until
they don't — the shape pitfall #13b describes for loss-weight resolvers, in
schema form.

The data-layer schema is elected: it is the wider of the two (narrowing would
invalidate the four configs using ``diffusion_weighted``) and the PMPS module's
own dispatch test already imported it rather than the local copy.

These tests pin the *election*, not the field list. A future re-declaration —
however faithful at the time — reintroduces the drift these assertions exist to
prevent, so identity is asserted rather than field-by-field equality.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import AcquisitionParamsSchema
from spectramr.config.schemas.training.pmps import (
    AcquisitionParam,
    ProtocolSamplingConfig,
)


def test_acquisition_param_is_the_data_schema_not_a_copy() -> None:
    """Identity, not equivalence.

    A re-declared class with identical fields would satisfy any field-by-field
    comparison on the day it was written and drift the day either side changed —
    which is exactly what happened. Only `is` catches that.
    """
    assert AcquisitionParam is AcquisitionParamsSchema


def test_diffusion_weighted_validates_as_a_pmps_protocol() -> None:
    """The dict that used to split the two schemas.

    Rejected by the old PMPS copy, accepted by the data schema. It is a real
    corpus value: four configs declare `diffusion_weighted`.
    """
    p = AcquisitionParam(
        name="dwi", TE=80.0, TR=5000.0, contrast_type="diffusion_weighted"
    )
    assert p.contrast_type == "diffusion_weighted"


def test_protocol_sampling_still_accepts_the_alias() -> None:
    """`fixed_protocols` is annotated with the alias; the seam must still build."""
    p = AcquisitionParam(name="t1w", TE=10.0, TR=500.0)
    cfg = ProtocolSamplingConfig(mode="fixed", fixed_protocols=[p])
    assert len(cfg.fixed_protocols) == 1
    assert cfg.fixed_protocols[0].name == "t1w"


@pytest.mark.parametrize(
    "contrast",
    ["spin_echo", "inversion_recovery", "gradient_echo", "ssfp", "mprage"],
)
def test_the_previously_shared_vocabulary_still_validates(contrast: str) -> None:
    """Widening must not have dropped anything.

    Electing the wider schema can only add accepted values — but "can only" is
    the kind of claim worth checking, so every member the narrow Literal already
    allowed is asserted explicitly.
    """
    assert AcquisitionParam(name="x", TE=10.0, TR=500.0, contrast_type=contrast)


def test_an_unknown_contrast_is_still_rejected() -> None:
    """The election widened the set; it did not open it.

    Without this, deleting the Literal entirely would satisfy every assertion
    above.
    """
    with pytest.raises(ValidationError):
        AcquisitionParam(name="x", TE=10.0, TR=500.0, contrast_type="not_a_contrast")
