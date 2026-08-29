"""Per-modality sensor adapters for multi-sensor motion fusion (idea 9 §9.2).

Each adapter consumes a raw sensor stream and emits an SE(3) pose
6-vector (translation + axis-angle rotation) suitable for the
:func:`cramer_rao_optimal_fusion` routine in
:mod:`mriforge.infrastructure.sensors.fusion`.

The submodules are intentionally narrow:

- :mod:`imu` — strapdown-INS integration for accelerometer + gyro
- :mod:`respiratory` — phase extraction from a respiratory-belt signal

RGB-D and ECG adapters are TBD in follow-up work; the protocol is
already exercised by the IMU + respiratory adapters and the fusion
routine accepts arbitrary :class:`PoseEstimate` instances.
"""

from .imu import IMUSample, integrate_imu_stream
from .respiratory import RespiratoryGate, extract_respiratory_phase

__all__ = [
    "IMUSample",
    "RespiratoryGate",
    "extract_respiratory_phase",
    "integrate_imu_stream",
]
