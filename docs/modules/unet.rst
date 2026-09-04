.. _module_unet:

Universal U-Net Generator
=========================

**File Path:** ``src/spectramr/models/reconstruction/unet.py``

High-Level Logic
----------------
This module implements a **Configurable U-Net** architecture, designed to act as the Generator :math:`G(x)` in reconstruction and GAN frameworks. It adheres to the classic encoder-decoder topology with skip connections but introduces significant architectural flexibility:

1.  **Polymorphic Blocks:** Supports `Standard`, `Bottleneck`, and `Dense` residual blocks.
2.  **Attention Mechanisms:** Integrated `Channel` and `Spatial` attention gates at multiple scales.
3.  **Dimensionality Agnostic:** Capable of handling 2D inputs or 2D+Time (via channel stacking).
4.  **Deep Supervision:** Optional auxiliary heads at decoder levels to improve gradient flow during early training.

Mathematical Core
-----------------
The U-Net approximates the inverse mapping :math:`f: y \to x` where :math:`y` is the aliased input.

**Skip Connections:**
To preserve high-frequency spatial details lost during downsampling, feature maps :math:`F_enc` from the encoder are concatenated with upsampled features :math:`F_dec`:

.. math::
   F_{out}^l = \text{Conv}(\text{Concat}(F_{dec}^l, F_{enc}^{L-l}))

**Attention Gate (Spatial):**
If enabled, the skip connection is modulated by an attention map :math:`\alpha`:

.. math::
   \alpha = \sigma(W_g \cdot g + W_x \cdot x + b)
   \hat{x} = \alpha \odot x

Where :math:`g` is the gating signal (from the decoder) and :math:`x` is the feature map (from the encoder).

Class Breakdown
---------------

.. autoclass:: spectramr.models.reconstruction.unet.UNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spectramr.models.reconstruction.unet.ConfigurableResidualBlock
   :members:
