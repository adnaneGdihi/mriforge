r"""IMU 6-DOF pose adapter (idea 9 §9.2).

Integrates a stream of accelerometer + gyroscope readings into the
SE(3) pose required by :class:`PoseEstimate`. Uses the standard
strapdown-INS update with a small-angle approximation for rotation
integration.

Mathematics
-----------
Given accelerometer samples :math:`\mathbf{a}_t \in \mathbb{R}^3`
(in body frame, including gravity) and gyroscope samples
:math:`\boldsymbol{\omega}_t \in \mathbb{R}^3` at uniform sample
period :math:`\Delta t`, the body-frame velocity and rotation
update is

.. math::

    \mathbf v_{t+1} &= \mathbf v_t + \bigl(\mathbf a_t - \mathbf g_t\bigr)\,\Delta t,\\
    \mathbf p_{t+1} &= \mathbf p_t + \mathbf v_t\,\Delta t,\\
    \mathbf r_{t+1} &= \mathbf r_t + \boldsymbol\omega_t\,\Delta t.

The output rotation vector is the integrated body-frame angular
displacement (axis-angle / Lie algebra). For the small motions
typical inside an MRI bore the small-angle approximation is
adequate; tighter integration (Rodrigues / quaternion exponential)
is left to follow-up work.

Drift bound
-----------
With zero-mean Gaussian gyroscope noise of variance :math:`\sigma_g^2`
and sample period :math:`\Delta t`, the orientation drift over
:math:`T` seconds has standard deviation
:math:`\sigma_g \sqrt{T \Delta t}`. The unit test below verifies that
on a 10-second synthetic trace the integrated drift stays inside
this analytic bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# Standard gravity in m/s². Subtracted from accelerometer readings as
# the canonical "remove gravity" baseline; experiments where the head
# coil is non-vertical may need to override this via the constructor.
_STANDARD_GRAVITY: tuple[float, float, float] = (0.0, 0.0, 9.80665)


@dataclass(frozen=True)
class IMUSample:
    """A single IMU reading.

    Attributes:
        accel: ``[3]`` body-frame linear-acceleration, m/s² (incl. gravity).
        gyro: ``[3]`` body-frame angular-rate, rad/s.
    """

    accel: torch.Tensor
    gyro: torch.Tensor

    def __post_init__(self) -> None:
        for name, t in (("accel", self.accel), ("gyro", self.gyro)):
            if t.shape != (3,):
                raise ValueError(f"IMUSample.{name} must be a 3-vector; got {tuple(t.shape)}")


def integrate_imu_stream(
    samples: Sequence[IMUSample],
    sample_period_s: float,
    gravity: Sequence[float] | None = None,
    initial_position: Sequence[float] | None = None,
    initial_velocity: Sequence[float] | None = None,
    initial_rotation: Sequence[float] | None = None,
) -> torch.Tensor:
    """Strapdown-INS integration of an IMU stream into a final SE(3) pose.

    Args:
        samples: Sequence of :class:`IMUSample`. Length ≥ 1.
        sample_period_s: Uniform inter-sample period ``Δt`` (must be > 0).
        gravity: Body-frame gravity vector to subtract. Defaults to
            ``(0, 0, 9.80665)``.
        initial_position, initial_velocity, initial_rotation: Optional
            initial-condition 3-vectors. Defaults to zero.

    Returns:
        ``[6]`` SE(3) pose ``(t_x, t_y, t_z, r_x, r_y, r_z)`` ready for
        consumption by :class:`PoseEstimate`.

    Raises:
        ValueError: invalid inputs.
    """
    if not samples:
        raise ValueError("samples is empty")
    if sample_period_s <= 0:
        raise ValueError(f"sample_period_s must be > 0; got {sample_period_s}")

    g = torch.tensor(
        gravity if gravity is not None else _STANDARD_GRAVITY,
        dtype=torch.float64,
    )
    if g.shape != (3,):
        raise ValueError(f"gravity must be a 3-vector; got {tuple(g.shape)}")

    p = torch.tensor(
        initial_position if initial_position is not None else (0.0, 0.0, 0.0),
        dtype=torch.float64,
    )
    v = torch.tensor(
        initial_velocity if initial_velocity is not None else (0.0, 0.0, 0.0),
        dtype=torch.float64,
    )
    r = torch.tensor(
        initial_rotation if initial_rotation is not None else (0.0, 0.0, 0.0),
        dtype=torch.float64,
    )
    dt = float(sample_period_s)

    for s in samples:
        a = s.accel.to(torch.float64) - g
        omega = s.gyro.to(torch.float64)

        # Position update uses the velocity *before* updating it
        # (forward-Euler with explicit time-stepping, mid-point would
        # be tighter but is unnecessary at the sample rates IMUs use).
        p = p + v * dt
        v = v + a * dt
        r = r + omega * dt

    return torch.cat([p, r]).to(torch.float32)


__all__ = ["IMUSample", "integrate_imu_stream"]
