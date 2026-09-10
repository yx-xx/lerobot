"""Pure math helpers with no ROS or frankx dependency."""

from __future__ import annotations

import math
from typing import Sequence


def finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def pose_to_matrix(
    position: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, ...]:
    """Convert XYZ and XYZW quaternion to a column-major homogeneous matrix."""
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("position and quaternion must have lengths 3 and 4")
    p = tuple(float(value) for value in position)
    q = tuple(float(value) for value in quaternion)
    if not finite(p + q):
        raise ValueError("pose values must be finite")
    qx, qy, qz, qw = q
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("quaternion must be normalized")
    qx, qy, qz, qw = (value / norm for value in q)

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    x, y, z = p
    return (
        r00,
        r10,
        r20,
        0.0,
        r01,
        r11,
        r21,
        0.0,
        r02,
        r12,
        r22,
        0.0,
        x,
        y,
        z,
        1.0,
    )


def matrix_to_pose(
    matrix: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a column-major homogeneous matrix to XYZ and XYZW quaternion."""
    if len(matrix) != 16:
        raise ValueError("matrix must contain exactly 16 values")
    m = tuple(float(value) for value in matrix)
    if not finite(m):
        raise ValueError("matrix must contain only finite values")

    r00, r10, r20 = m[0], m[1], m[2]
    r01, r11, r21 = m[4], m[5], m[6]
    r02, r12, r22 = m[8], m[9], m[10]
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    quaternion = (qx / norm, qy / norm, qz / norm, qw / norm)
    return (m[12], m[13], m[14]), quaternion
