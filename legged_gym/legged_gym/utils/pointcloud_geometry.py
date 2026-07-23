# legged_gym/legged_gym/utils/pointcloud_geometry.py
"""Pure geometric functions for LiDAR point cloud manipulation.

All quaternions use Isaac Gym convention: [x, y, z, w] (scalar-last).
Shared by CmdSafeHistoryWrapper and symmetry augmentation.
No state, no classes, only torch dependency.
"""

from __future__ import annotations
import math
import torch


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate a quaternion [x, y, z, w] -> [-x, -y, -z, w]."""
    sign = torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)
    return q * sign


def quaternion_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions.  q: [..., 4], v: [..., 3] -> [..., 3]."""
    q_vec = q[..., :3]
    q_scalar = q[..., 3:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec, t, dim=-1)


def cartesian_to_spherical(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cartesian [..., 3] -> (r, azimuth, phi).
    azimuth = atan2(y, x) in [-pi, pi]; phi = asin(z / r) in [-pi/2, pi/2].
    Guards: nan->0, inf->1e5."""
    points = torch.nan_to_num(points, nan=0.0, posinf=1e5, neginf=-1e5)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    r = torch.norm(points, dim=-1)
    azimuth = torch.atan2(y, x)
    phi = torch.asin(torch.clamp(z / (r + 1e-9), -1.0, 1.0))
    return r, azimuth, phi


def to_sensor_frame(
    points_base: torch.Tensor,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
) -> torch.Tensor:
    """Base frame [B,N,3] -> sensor frame [B,N,3].
    sensor_quat: [1,4] or [B,4]; sensor_trans: [1,3] or [B,3]."""
    t = sensor_trans.to(points_base.device)
    pts = points_base - t.unsqueeze(1)
    q = sensor_quat.to(points_base.device)
    q_conj = quaternion_conjugate(q)
    B, N = pts.shape[:2]
    pts = quaternion_apply(q_conj.expand(B * N, 4), pts.reshape(-1, 3))
    return pts.reshape(B, N, 3)


def sort_points_by_angular_key(
    points_base: torch.Tensor,
    sensor_quat: torch.Tensor,
    sensor_trans: torch.Tensor,
) -> torch.Tensor:
    """Sort points by angular key (phi * 2*pi + azimuth).
    SINGLE sorting entry point. Input/output both in base frame.
    points_base: [B,N,3] -> [B,N,3] sorted."""
    pts_sensor = to_sensor_frame(points_base, sensor_quat, sensor_trans)
    _, azimuth, phi = cartesian_to_spherical(pts_sensor)
    key = phi * (2.0 * math.pi) + azimuth
    order = torch.argsort(key, dim=1)
    return torch.gather(points_base, 1, order.unsqueeze(-1).expand_as(points_base))
