"""Streamfunction / vector-potential to divergence-free velocity fields.

In 2D the constraint ``div v = d v_x/dx + d v_y/dy = 0`` is solved
exactly by the streamfunction parameterisation
``v_x = d psi/dy``, ``v_y = -d psi/dx`` for any scalar field ``psi``: the
divergence then vanishes by equality of mixed partials. In 3D the
analogue is the curl of a vector potential, ``v = curl(psi)``, which is
divergence-free because ``div(curl(.)) = 0``.

Both blocks use central differences on a staggered (MAC) grid, which keeps
the discrete divergence at machine precision (collocated differences leave
an ``O(h^2)`` residual).

Reference: standard incompressible-flow construction; see e.g. the MAC
grid of Harlow & Welch (1965).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Replicate-pad one element at the high side of spatial ``dim``.

    ``F.pad(mode="replicate")`` only accepts a pad spec for the spatial
    dimensions (last 2 of a 4D tensor, last 3 of a 5D tensor), NOT for the
    batch/channel axes — passing a full ``2*ndim`` spec raises
    ``NotImplementedError``. We therefore build a pad list covering only the
    ``ndim - 2`` spatial dims. ``dim`` indexes the full tensor (2/3 for 4D,
    2/3/4 for 5D); F.pad orders pairs from the last dim backwards, so the
    high-side slot for ``dim`` is ``2*(ndim-1-dim) + 1``.
    """
    ndim = x.dim()
    num_spatial = ndim - 2  # exclude batch + channel
    pad_full = [0] * (2 * num_spatial)
    idx = 2 * (ndim - 1 - dim)
    pad_full[idx + 1] = 1  # high side of spatial `dim`
    return F.pad(x, pad_full, mode="replicate")


class StreamfunctionTo2DVelocity(nn.Module):
    """Map a scalar streamfunction ``psi`` to a divergence-free 2D field.

    Input ``psi`` has shape ``[B, 1, H, W]``; output velocity has shape
    ``[B, 2, H, W]`` with channel 0 = ``v_x`` and channel 1 = ``v_y``.
    """

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        if psi.shape[1] != 1:
            raise ValueError(f"StreamfunctionTo2DVelocity expects 1 channel, got {psi.shape[1]}.")
        # H is dim=2, W is dim=3.  v_x = d psi/dy (along H), v_y = -d psi/dx (along W).
        dpsi_dy = psi[:, :, 1:, :] - psi[:, :, :-1, :]
        dpsi_dx = psi[:, :, :, 1:] - psi[:, :, :, :-1]
        v_x = _pad_last(dpsi_dy, dim=2)
        v_y = -_pad_last(dpsi_dx, dim=3)
        return torch.cat([v_x, v_y], dim=1)


class CurlOf3DVectorPotential(nn.Module):
    """Map a 3-vector potential ``psi`` to a divergence-free 3D field.

    Input ``psi`` has shape ``[B, 3, D, H, W]``; output velocity has the
    same shape and equals ``curl(psi)`` computed with forward differences
    on a staggered grid:
        v_x = d psi_z/dy - d psi_y/dz
        v_y = d psi_x/dz - d psi_z/dx
        v_z = d psi_y/dx - d psi_x/dy
    Spatial axes are (D=dim2, H=dim3, W=dim4) -> identify z=D, y=H, x=W.
    """

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        if psi.shape[1] != 3:
            raise ValueError(f"CurlOf3DVectorPotential expects 3 channels, got {psi.shape[1]}.")
        px, py, pz = psi[:, 0:1], psi[:, 1:2], psi[:, 2:3]

        def d(field: torch.Tensor, axis: int) -> torch.Tensor:
            # axis: 2=z(D), 3=y(H), 4=x(W)
            sl_hi = [slice(None)] * field.dim()
            sl_lo = [slice(None)] * field.dim()
            sl_hi[axis] = slice(1, None)
            sl_lo[axis] = slice(None, -1)
            diff = field[tuple(sl_hi)] - field[tuple(sl_lo)]
            return _pad_last(diff, dim=axis)

        v_x = d(pz, 3) - d(py, 2)  # d psi_z/dy - d psi_y/dz
        v_y = d(px, 2) - d(pz, 4)  # d psi_x/dz - d psi_z/dx
        v_z = d(py, 4) - d(px, 3)  # d psi_y/dx - d psi_x/dy
        return torch.cat([v_x, v_y, v_z], dim=1)
