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
    """Return the four-corner rectangle border in body XY.

    ENV-RECT-CONTROL-012: MasterYip asked that the legacy slider border behave
    like its rectangle instead of the old six-point barrel (fusiform). The five
    parameters describe a rectangle whose x extent is
    ``[backward_limit, forward_limit]`` and whose width is the *maximum* of the
    three width samples (the bounding box of the old hexagon), so the envelope
    inflates as much as possible toward that rectangle. Vertices are returned
    counter-clockwise starting at the front-right corner.
    """
    if parameters.shape != (5,):
        raise ValueError("parameters must have shape [5]")
    front, middle, back, forward, backward = parameters.unbind()
    width = torch.maximum(torch.maximum(front, middle), back)
    return torch.stack((
        torch.stack((forward, width)),
        torch.stack((backward, width)),
        torch.stack((backward, -width)),
        torch.stack((forward, -width)),
    ))


def legacy_border_support(parameters: Tensor, directions: Tensor) -> Tensor:
    """Support function of the rectangle border along every registered normal.

    ``support[j]`` is the maximum border projection onto ``directions[j]``, i.e.
    the radius of the *maximal convex support polygon* that still contains the
    whole border. The envelope built from this support reconstructs the border's
    true rectangle instead of the corner-rounded inscribed polygon that ray
    intersections produce.
    """
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    vertices = legacy_border_vertices(parameters)
    return (vertices @ directions.T).amax(dim=0)


def sample_border_points(parameters: Tensor, directions: Tensor) -> Tensor:
    """Return the ZhangHT border support point for every registered LiDAR ray.

    For each ray the return is the border vertex that maximizes the projection
    onto that ray's direction (the extreme point of the rectangle). Placing the
    LiDAR return at the extreme point makes the support-polygon envelope equal
    to the border's own support polygon, which is what lets the rectangle border
    produce a rectangle envelope.
    """
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    vertices = legacy_border_vertices(parameters)
    projections = vertices @ directions.T
    indices = projections.argmax(dim=0)
    return vertices[indices]


def legacy_border_lidar_cloud(
    parameters: Tensor,
    directions: Tensor,
    baseline_support: Tensor,
    reference_support: Tensor,
    *,
    seed: int,
    reference_containment_margin: float = 0.005,
) -> SyntheticLidarCloud:
    """Build a one-return-per-sector cloud without changing LiDAR math.

    Each return is the border's *support point* (the rectangle vertex farthest
    along the sector's normal), so the capped support-polygon envelope
    reconstructs the rectangle instead of a fusiform. ENV-RECT-CONTROL-012:
    ``reference_support`` is the cap the returns must lie inside; the caller may
    pass an inflated cap (``max(reference_support, border_support + margin)``)
    so the reference never shrinks the free envelope below the border. When a
    cap tighter than the border is passed, the return for the affected sector
    falls back to the ray-capped point so the cloud containment invariant holds
    for every caller.
    """
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
    ray_points = radii[:, None] * directions
    inside = (
        border_points @ directions.T <= eroded_reference.unsqueeze(0) + 1e-6
    ).all(dim=-1)
    points = torch.where(inside[:, None], border_points, ray_points)
    radii = points.norm(dim=-1)
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
