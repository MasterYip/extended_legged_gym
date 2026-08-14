"""Deterministic LiDAR free-space geometry for the EL4090 viewer example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from kinematic_envelope import BatchedUrdfKinematics, CapsuleProxy, capsule_support


@dataclass(frozen=True)
class SyntheticLidarCloud:
    points_xy: Tensor
    angles_rad: Tensor
    radii_m: Tensor
    sector_indices: Tensor
    seed: int
    min_radius_m: float
    max_radius_m: float
    robot_clearance_m: float


@dataclass(frozen=True)
class PointFreeEnvelope:
    support_m: Tensor
    limiting_point_indices: Tensor
    limiting_points_xy: Tensor
    clearance_feet_xy: Tensor
    point_clearance_m: float
    optimality_scope: str


@dataclass(frozen=True)
class BacktrackedPose:
    joint_positions: Tensor
    accepted_scale: Tensor
    envelope_excess_m: Tensor
    joint_excess_rad: Tensor


def _wrapped_angle_delta(angles: Tensor, centers: Tensor) -> Tensor:
    delta = angles[:, None] - centers[None, :]
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def generate_synthetic_lidar_cloud(
    directions: Tensor,
    baseline_support: Tensor,
    *,
    count: int,
    seed: int,
    min_radius: float,
    max_radius: float,
    robot_clearance: float,
) -> SyntheticLidarCloud:
    """Generate full-coverage returns with near clusters and far gaps.

    Every normal sector receives at least one return. Three angular clusters
    pull returns inward while two gap directions push returns outward. The
    assigned-normal separation from the baseline support polygon is at least
    ``robot_clearance``.
    """
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    sectors = directions.shape[0]
    if baseline_support.shape != (sectors,):
        raise ValueError("baseline_support must match the direction count")
    if count < sectors:
        raise ValueError("point count must cover every angular sector")
    if min_radius <= 0.0 or max_radius <= min_radius:
        raise ValueError("radius bounds must satisfy 0 < min < max")
    if robot_clearance <= 0.0:
        raise ValueError("robot_clearance must be positive")

    generator = torch.Generator(device=directions.device).manual_seed(int(seed))
    sector_indices = torch.arange(count, device=directions.device) % sectors
    sector_angles = torch.atan2(directions[:, 1], directions[:, 0])
    sector_width = 2.0 * torch.pi / sectors
    jitter = (torch.rand(count, generator=generator, dtype=directions.dtype, device=directions.device) - 0.5) * (0.70 * sector_width)
    angles = sector_angles[sector_indices] + jitter
    unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    assigned_projection_scale = (unit * directions[sector_indices]).sum(dim=-1)

    cluster_centers = torch.tensor((0.25, 2.35, 4.45), dtype=directions.dtype, device=directions.device)
    gap_centers = torch.tensor((1.35, 5.35), dtype=directions.dtype, device=directions.device)
    cluster_strength = torch.exp(-0.5 * (_wrapped_angle_delta(angles, cluster_centers) / 0.34) ** 2).amax(dim=1)
    gap_strength = torch.exp(-0.5 * (_wrapped_angle_delta(angles, gap_centers) / 0.28) ** 2).amax(dim=1)
    noise = torch.rand(count, generator=generator, dtype=directions.dtype, device=directions.device)
    radial_fraction = torch.clamp(0.58 - 0.34 * cluster_strength + 0.28 * gap_strength + 0.10 * (noise - 0.5), 0.08, 0.95)

    required_radius = (baseline_support[sector_indices] + robot_clearance) / assigned_projection_scale
    inner_radius = torch.maximum(required_radius, torch.full_like(required_radius, min_radius))
    if bool((inner_radius >= max_radius).any()):
        raise ValueError("max_radius cannot provide the requested baseline clearance")
    radii = inner_radius + radial_fraction * (max_radius - inner_radius)
    points = radii[:, None] * unit
    return SyntheticLidarCloud(
        points_xy=points,
        angles_rad=angles,
        radii_m=radii,
        sector_indices=sector_indices,
        seed=int(seed),
        min_radius_m=float(min_radius),
        max_radius_m=float(max_radius),
        robot_clearance_m=float(robot_clearance),
    )


def maximum_sector_point_free_envelope(
    cloud: SyntheticLidarCloud,
    directions: Tensor,
    *,
    point_clearance: float,
) -> PointFreeEnvelope:
    """Return the coordinatewise-maximal fixed-normal sector envelope.

    A return constrains only its assigned outward normal. Therefore each safe
    support is independently maximized by the smallest assigned projection
    minus ``point_clearance``. Raising any support violates its active return.
    """
    if point_clearance <= 0.0:
        raise ValueError("point_clearance must be positive")
    sectors = directions.shape[0]
    if cloud.sector_indices.shape != (cloud.points_xy.shape[0],):
        raise ValueError("cloud sector indices must match its points")
    projections = (cloud.points_xy * directions[cloud.sector_indices]).sum(dim=-1)
    mask = cloud.sector_indices[:, None] == torch.arange(sectors, device=directions.device)[None, :]
    assigned = projections[:, None].expand(-1, sectors).masked_fill(~mask, torch.inf)
    minimum_projection, limiting_indices = assigned.min(dim=0)
    if not bool(torch.isfinite(minimum_projection).all()):
        raise ValueError("every normal sector must contain at least one return")
    support = minimum_projection - point_clearance
    if bool((support <= 0.0).any()):
        raise ValueError("point clearance leaves a non-positive support")
    limiting_points = cloud.points_xy[limiting_indices]
    face_distance = (limiting_points * directions).sum(dim=-1) - support
    clearance_feet = limiting_points - face_distance[:, None] * directions
    return PointFreeEnvelope(
        support_m=support,
        limiting_point_indices=limiting_indices,
        limiting_points_xy=limiting_points,
        clearance_feet_xy=clearance_feet,
        point_clearance_m=float(point_clearance),
        optimality_scope=(
            "coordinatewise maximum in the declared fixed-normal polygon family "
            "under nearest-angular-sector point assignment"
        ),
    )


def assigned_point_clearances(
    cloud: SyntheticLidarCloud,
    directions: Tensor,
    support: Tensor,
) -> Tensor:
    """Signed separation of every return from its assigned support face."""
    return (
        cloud.points_xy * directions[cloud.sector_indices]
    ).sum(dim=-1) - support[cloud.sector_indices]


def envelope_excess(
    kinematics: BatchedUrdfKinematics,
    joint_positions: Tensor,
    capsules: Sequence[CapsuleProxy],
    directions: Tensor,
    allowed_support: Tensor,
) -> Tensor:
    """Maximum occupied-over-allowed support excess for a pose batch."""
    support = capsule_support(kinematics, joint_positions, capsules, directions)
    return (support - allowed_support).amax(dim=-1)


def backtrack_to_feasible_anchor(
    kinematics: BatchedUrdfKinematics,
    proposed_q: Tensor,
    anchor_q: Tensor,
    lower: Tensor,
    upper: Tensor,
    capsules: Sequence[CapsuleProxy],
    directions: Tensor,
    allowed_support: Tensor,
    *,
    tolerance: float = 1e-6,
    steps: int = 24,
) -> BacktrackedPose:
    """Backtrack a batched proposal toward a known feasible anchor."""
    if proposed_q.ndim != 2 or proposed_q.shape[1:] != anchor_q.shape:
        raise ValueError("expected proposed_q [B,D] and anchor_q [D]")
    if lower.shape != anchor_q.shape or upper.shape != anchor_q.shape:
        raise ValueError("joint bounds must match the anchor")
    if steps < 1:
        raise ValueError("steps must be positive")
    anchor_excess = envelope_excess(
        kinematics, anchor_q.unsqueeze(0), capsules, directions, allowed_support,
    )[0]
    if float(anchor_excess) > tolerance:
        raise ValueError("anchor pose is not envelope-feasible")

    proposed_q = torch.maximum(lower, torch.minimum(upper, proposed_q))
    anchors = anchor_q.unsqueeze(0).expand_as(proposed_q)
    scale = torch.ones(proposed_q.shape[0], dtype=proposed_q.dtype, device=proposed_q.device)
    accepted = proposed_q
    excess = envelope_excess(kinematics, accepted, capsules, directions, allowed_support)
    for _ in range(steps):
        infeasible = excess > tolerance
        if not bool(infeasible.any()):
            break
        scale = torch.where(infeasible, scale * 0.5, scale)
        accepted = anchors + scale[:, None] * (proposed_q - anchors)
        excess = envelope_excess(kinematics, accepted, capsules, directions, allowed_support)

    joint_excess = torch.maximum(lower - accepted, accepted - upper).clamp_min(0.0).amax(dim=-1)
    if bool((excess > tolerance).any()) or bool((joint_excess > tolerance).any()):
        raise RuntimeError("backtracking failed to produce a feasible pose")
    return BacktrackedPose(accepted, scale, excess, joint_excess)
