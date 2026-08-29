"""Pin the capability-domain contracts determined in the WS-X contract pass.

Each was determined by READING the model (+ its feeding strategy), not guessing
from the name — the names are deliberately misleading:
- hermitian_fno / hermitian_kspace_unet: spectral operators on REAL image-space
  feature maps (FNO transforms internally); output is real image space.
- cross_attention_oracle_unet: recon backbone fed an IMAGE tensor (strategy IFFTs
  k-space first; arms rss-combine to magnitude) — the VF DC-blob KNOWN_IMAGE fix.
- image_cold_diffusion: magnitude image in/out.
"""

from __future__ import annotations

import pytest

from mriforge.models.registry import get_model_capabilities


def test_image_cold_diffusion_accepts_image_and_complex_image() -> None:
    # Configurable backbone: a real backbone takes a 1-ch magnitude image
    # (image); a complex_unet backbone takes 2-ch real/imag (complex_image).
    # Declaring both lets a coil-compressed complex arm (svd → ifft →
    # complex_image) audit clean while a magnitude arm still matches.
    caps = get_model_capabilities("image_cold_diffusion")
    assert caps is not None
    assert set(caps.input_domain) == {"image", "complex_image"}
    assert set(caps.output_domain) == {"image", "complex_image"}
    assert caps.spatial_dims == (2,)


@pytest.mark.parametrize(
    "model_type",
    ["hermitian_fno", "hermitian_kspace_unet", "cross_attention_oracle_unet"],
)
def test_image_space_models_accept_image_and_complex_image(model_type: str) -> None:
    # These image-space backbones lift whatever real channels they're given:
    # 1-ch magnitude (image) OR 2-ch real/imag (complex_image). Declaring both
    # lets a k-space arm bridge via ifft_kspace_to_image (-> complex_image) and
    # still satisfy data_model_compatibility (probe-verified on 2ch).
    caps = get_model_capabilities(model_type)
    assert caps is not None, f"{model_type} not registered"
    assert set(caps.input_domain) == {"image", "complex_image"}
    assert set(caps.output_domain) == {"image", "complex_image"}
    assert caps.spatial_dims == (2,)


def test_fno_contract() -> None:
    # k-space input; IFFTs at output (output_domain image by default, kspace
    # configurable) — declared permissively so both configs audit clean.
    caps = get_model_capabilities("fno")
    assert caps is not None
    assert caps.input_domain == "kspace"
    assert set(caps.output_domain) == {"image", "kspace"}


def test_me_mno_operator_contract() -> None:
    # Wraps cs_mno_operator (real-valued PDE-grid / MRI-image); d4/oh are 2-D/3-D.
    caps = get_model_capabilities("me_mno_operator")
    assert caps is not None
    assert set(caps.input_domain) == {"pde_grid", "image"}
    assert set(caps.output_domain) == {"pde_grid", "image"}
    assert caps.spatial_dims == (2, 3)
