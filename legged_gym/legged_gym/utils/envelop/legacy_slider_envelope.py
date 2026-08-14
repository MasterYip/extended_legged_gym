"""ZhangHT slider border sampled as the LiDAR point source."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor

from lidar_free_envelope import SyntheticLidarCloud


LEGACY_PARAMETER_RANGES = {
    "front_width": (0.3, 0.6),
    "middle_width": (0.3, 0.7),
    "back_width": (0.3, 0.6),
    "forward_limit": (0.6, 0.9),
    "backward_limit": (-0.9, -0.6),
}
LEGACY_PARAMETER_ORDER = tuple(LEGACY_PARAMETER_RANGES)
LEGACY_MIDPOINT = (0.45, 0.50, 0.45, 0.75, -0.75)
LEGACY_MAXIMUM = (0.60, 0.70, 0.60, 0.90, -0.90)


def parameter_tensor(
    values: Mapping[str, float] | Sequence[float], *,
    dtype=torch.float32, device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(values, Mapping):
        ordered = [values[name] for name in LEGACY_PARAMETER_ORDER]
    else:
        ordered = list(values)
    if len(ordered) != len(LEGACY_PARAMETER_ORDER):
        raise ValueError("legacy envelope requires exactly five parameters")
    tensor = torch.tensor(ordered, dtype=dtype, device=device)
    for index, name in enumerate(LEGACY_PARAMETER_ORDER):
        lower, upper = LEGACY_PARAMETER_RANGES[name]
        if not lower - 1e-6 <= float(tensor[index]) <= upper + 1e-6:
            raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return tensor


def legacy_border_vertices(parameters: Tensor) -> Tensor:
    """Return ZhangHT's exact six-point symmetric border in body XY."""
    if parameters.shape != (5,):
        raise ValueError("parameters must have shape [5]")
    front, middle, back, forward, backward = parameters.unbind()
    zero = torch.zeros((), dtype=parameters.dtype, device=parameters.device)
    return torch.stack((
        torch.stack((forward, front)),
        torch.stack((zero, middle)),
        torch.stack((backward, back)),
        torch.stack((backward, -back)),
        torch.stack((zero, -middle)),
        torch.stack((forward, -front)),
    ))


def _cross_2d(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def sample_border_points(parameters: Tensor, directions: Tensor) -> Tensor:
    """Intersect every registered LiDAR ray with the ZhangHT border."""
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    vertices = legacy_border_vertices(parameters)
    starts = vertices
    edges = torch.roll(vertices, shifts=-1, dims=0) - vertices
    ray = directions[:, None, :]
    edge = edges[None, :, :]
    start = starts[None, :, :]
    denominator = _cross_2d(ray, edge)
    safe = denominator.abs() > 1e-9
    t = _cross_2d(start, edge) / torch.where(
        safe, denominator, torch.ones_like(denominator),
    )
    u = _cross_2d(start, ray) / torch.where(
        safe, denominator, torch.ones_like(denominator),
    )
    valid = safe & (t >= 0.0) & (u >= -1e-6) & (u <= 1.0 + 1e-6)
    distance = t.masked_fill(~valid, torch.inf).amin(dim=1)
    if not bool(torch.isfinite(distance).all()):
        raise RuntimeError("a LiDAR ray did not intersect the ZhangHT border")
    return distance[:, None] * directions


def legacy_border_lidar_cloud(
    parameters: Tensor,
    directions: Tensor,
    baseline_support: Tensor,
    reference_support: Tensor,
    *,
    seed: int,
    reference_containment_margin: float = 0.005,
) -> SyntheticLidarCloud:
    """Build a one-return-per-sector cloud without changing LiDAR math."""
    parameters = parameters.to(directions)
    border_points = sample_border_points(parameters, directions)
    border_radii = border_points.norm(dim=-1)
    projections = directions @ directions.T
    eroded_reference = reference_support - reference_containment_margin
    radial_caps = torch.where(
        projections > 1e-7,
        eroded_reference.unsqueeze(0) / projections.clamp_min(1e-7),
        torch.full_like(projections, torch.inf),
    ).amin(dim=-1)
    radii = torch.minimum(border_radii, radial_caps)
    points = radii[:, None] * directions
    sectors = directions.shape[0]
    indices = torch.arange(sectors, device=directions.device)
    zeros = directions.new_empty((0,))
    return SyntheticLidarCloud(
        points_xy=points,
        angles_rad=torch.atan2(points[:, 1], points[:, 0]),
        radii_m=radii,
        sector_indices=indices,
        seed=int(seed),
        min_radius_m=float(radii.min()),
        max_radius_m=float(radii.max()),
        robot_clearance_m=0.0,
        lateral_robot_clearance_m=0.0,
        required_clearance_m=torch.zeros_like(radii),
        reference_containment_margin_m=float(reference_containment_margin),
        ray_inner_radius_m=baseline_support.clone(),
        ray_outer_radius_m=reference_support.clone(),
        near_cluster_centers_rad=zeros,
        far_gap_centers_rad=zeros,
        sector_counts=torch.ones(sectors, dtype=torch.long, device=directions.device),
        lateral_anchor_sectors=torch.empty(0, dtype=torch.long, device=directions.device),
        near_band_fraction=0.0,
    )
