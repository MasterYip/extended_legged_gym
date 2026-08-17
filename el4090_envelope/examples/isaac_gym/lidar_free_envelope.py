"""Deterministic LiDAR free-space geometry for the EL4090 viewer example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from el4090_envelope import BatchedUrdfKinematics, CapsuleProxy, capsule_support


MIN_STRUCTURE_CENTER_SEPARATION_RAD = 0.60


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
    lateral_robot_clearance_m: float
    required_clearance_m: Tensor
    reference_containment_margin_m: float
    ray_inner_radius_m: Tensor
    ray_outer_radius_m: Tensor
    near_cluster_centers_rad: Tensor
    far_gap_centers_rad: Tensor
    sector_counts: Tensor
    lateral_anchor_sectors: Tensor
    near_band_fraction: float


@dataclass(frozen=True)
class PointFreeEnvelope:
    support_m: Tensor
    limiting_point_indices: Tensor
    limiting_points_xy: Tensor
    clearance_feet_xy: Tensor
    constrained_face_indices: Tensor
    unconstrained_face_indices: Tensor
    cap_support_m: Tensor
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


def _sample_separated_angles(
    count: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
    minimum_separation_rad: float,
) -> Tensor:
    """Sample circular angles with deterministic bounded rejection."""
    selected = []
    for _ in range(512):
        candidate = 2.0 * torch.pi * torch.rand(
            (), generator=generator, dtype=dtype, device=device,
        )
        if not selected or bool((
            _wrapped_angle_delta(candidate.reshape(1), torch.stack(selected))
            >= minimum_separation_rad
        ).all()):
            selected.append(candidate)
            if len(selected) == count:
                return torch.stack(selected)
    raise RuntimeError("failed to sample separated LiDAR structure centers")


def generate_synthetic_lidar_cloud(
    directions: Tensor,
    baseline_support: Tensor,
    reference_reachable_support: Tensor,
    *,
    count: int,
    seed: int,
    min_radius: float,
    max_radius: float,
    robot_clearance: float,
    reference_containment_margin: float,
    lateral_robot_clearance: float = 0.025,
    lateral_anchors_per_side: int = 5,
    near_band_fraction: float = 0.05,
) -> SyntheticLidarCloud:
    """Generate sparse returns inside a pre-obstacle reachable polygon.

    For counts up to the direction count, returns occupy unique random sectors;
    remaining faces stay unconstrained by points. Three angular clusters pull
    returns inward while two gap directions push returns outward. The
    assigned-normal separation from the baseline support polygon is at least
    ``lateral_robot_clearance`` for lateral anchors and ``robot_clearance``
    elsewhere. Each ray's upper radius is resolved from the pre-obstacle
    reachable polygon, independently of the constrained export.
    """
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    sectors = directions.shape[0]
    if baseline_support.shape != (sectors,):
        raise ValueError("baseline_support must match the direction count")
    if reference_reachable_support.shape != (sectors,):
        raise ValueError("reference_reachable_support must match the direction count")
    if count < 1:
        raise ValueError("point count must be positive")
    if min_radius < 0.0 or max_radius <= min_radius:
        raise ValueError("radius bounds must satisfy 0 <= min < max")
    if robot_clearance <= 0.0:
        raise ValueError("robot_clearance must be positive")
    if not 0.0 < lateral_robot_clearance <= robot_clearance:
        raise ValueError(
            "lateral_robot_clearance must be in (0, robot_clearance]"
        )
    if lateral_anchors_per_side < 1:
        raise ValueError("lateral_anchors_per_side must be positive")
    if reference_containment_margin <= 0.0:
        raise ValueError("reference_containment_margin must be positive")
    if not 0.0 < near_band_fraction <= 1.0:
        raise ValueError("near_band_fraction must be in (0,1]")
    eroded_reference = reference_reachable_support - reference_containment_margin
    if bool((eroded_reference <= 0.0).any()):
        raise ValueError("reference containment margin erodes through the origin")

    generator = torch.Generator(device=directions.device).manual_seed(int(seed))
    primary_count = min(count, sectors)
    side_count = min(lateral_anchors_per_side, primary_count // 2)
    if side_count:
        left = torch.topk(directions[:, 1], side_count).indices
        right = torch.topk(-directions[:, 1], side_count).indices
        lateral_anchor_sectors = torch.cat((left, right))
    else:
        lateral_anchor_sectors = torch.empty(
            0, dtype=torch.long, device=directions.device,
        )
    available = torch.ones(sectors, dtype=torch.bool, device=directions.device)
    available[lateral_anchor_sectors] = False
    random_pool = torch.arange(sectors, device=directions.device)[available]
    random_primary = random_pool[torch.randperm(
        random_pool.numel(), generator=generator, device=directions.device,
    )[:primary_count - lateral_anchor_sectors.numel()]]
    primary_sectors = torch.cat((lateral_anchor_sectors, random_primary))
    extra_sectors = torch.randint(
        sectors, (count - primary_count,), generator=generator,
        device=directions.device,
    )
    sector_indices = torch.cat((primary_sectors, extra_sectors))
    primary = torch.cat((
        torch.ones(primary_count, dtype=torch.bool, device=directions.device),
        torch.zeros(count - primary_count, dtype=torch.bool, device=directions.device),
    ))
    permutation = torch.randperm(count, generator=generator, device=directions.device)
    sector_indices = sector_indices[permutation]
    primary = primary[permutation]
    lateral_returns = (
        sector_indices[:, None] == lateral_anchor_sectors[None, :]
    ).any(dim=1)
    required_clearance = torch.where(
        lateral_returns,
        torch.full(
            (count,), lateral_robot_clearance,
            dtype=directions.dtype, device=directions.device,
        ),
        torch.full(
            (count,), robot_clearance,
            dtype=directions.dtype, device=directions.device,
        ),
    )
    sector_angles = torch.atan2(directions[:, 1], directions[:, 0])
    sector_width = 2.0 * torch.pi / sectors
    raw_jitter = torch.rand(
        count, generator=generator, dtype=directions.dtype,
        device=directions.device,
    ) - 0.5
    jitter_span = torch.where(
        primary,
        torch.full((count,), 0.36, dtype=directions.dtype, device=directions.device),
        torch.full((count,), 0.90, dtype=directions.dtype, device=directions.device),
    )
    jitter = raw_jitter * jitter_span * sector_width

    structure_centers = _sample_separated_angles(
        5,
        generator=generator,
        dtype=directions.dtype,
        device=directions.device,
        minimum_separation_rad=MIN_STRUCTURE_CENTER_SEPARATION_RAD,
    )
    cluster_centers = torch.sort(structure_centers[:3]).values
    gap_centers = torch.sort(structure_centers[3:]).values
    noise = torch.rand(
        count, generator=generator, dtype=directions.dtype,
        device=directions.device,
    )

    # Wide jitter makes the cloud visibly irregular. In geometrically narrow
    # sectors, contract only the affected rays until their annulus is valid.
    for _ in range(8):
        angles = sector_angles[sector_indices] + jitter
        unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        assigned_projection_scale = (unit * directions[sector_indices]).sum(dim=-1)
        required_radius = (
            baseline_support[sector_indices] + required_clearance
        ) / assigned_projection_scale
        inner_radius = torch.maximum(
            required_radius, torch.full_like(required_radius, min_radius),
        )
        all_projection_scale = unit @ directions.T
        radial_caps = torch.where(
            all_projection_scale > 1e-7,
            eroded_reference.unsqueeze(0) / all_projection_scale.clamp_min(1e-7),
            torch.full_like(all_projection_scale, torch.inf),
        )
        polygon_outer_radius = radial_caps.amin(dim=-1)
        outer_radius = torch.minimum(
            polygon_outer_radius, torch.full_like(polygon_outer_radius, max_radius),
        )
        infeasible = inner_radius >= outer_radius
        if not bool(infeasible.any()):
            break
        jitter = torch.where(infeasible, jitter * 0.5, jitter)

    cluster_strength = torch.exp(-0.5 * (_wrapped_angle_delta(angles, cluster_centers) / 0.34) ** 2).amax(dim=1)
    gap_strength = torch.exp(-0.5 * (_wrapped_angle_delta(angles, gap_centers) / 0.28) ** 2).amax(dim=1)
    limiting_fraction = torch.clamp(
        0.01 + near_band_fraction * (
            0.12 + 0.68 * noise - 0.08 * cluster_strength
            + 0.12 * gap_strength
        ),
        0.005,
        near_band_fraction,
    )
    lateral_fraction = torch.clamp(
        0.005 + 0.25 * near_band_fraction * noise,
        0.005,
        0.35 * near_band_fraction,
    )
    scattered_fraction = torch.clamp(
        0.06 + 0.88 * noise - 0.12 * cluster_strength + 0.10 * gap_strength,
        0.02,
        0.96,
    )
    radial_fraction = torch.where(
        primary, limiting_fraction, scattered_fraction,
    )
    radial_fraction = torch.where(lateral_returns, lateral_fraction, radial_fraction)
    infeasible = inner_radius >= outer_radius
    if bool(infeasible.any()):
        failed = torch.unique(sector_indices[infeasible]).detach().cpu().tolist()
        raise ValueError(
            "no feasible LiDAR annulus between baseline clearance and the "
            f"eroded reference reachable polygon for sectors {failed}"
        )
    radii = inner_radius + radial_fraction * (outer_radius - inner_radius)
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
        lateral_robot_clearance_m=float(lateral_robot_clearance),
        required_clearance_m=required_clearance,
        reference_containment_margin_m=float(reference_containment_margin),
        ray_inner_radius_m=inner_radius,
        ray_outer_radius_m=outer_radius,
        near_cluster_centers_rad=cluster_centers,
        far_gap_centers_rad=gap_centers,
        sector_counts=torch.bincount(sector_indices, minlength=sectors),
        lateral_anchor_sectors=torch.sort(lateral_anchor_sectors).values,
        near_band_fraction=float(near_band_fraction),
    )


def polygon_support_excess(points_xy: Tensor, directions: Tensor, support: Tensor) -> Tensor:
    """Maximum half-space excess for every point; non-positive means inside."""
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape [N,2]")
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape [K,2]")
    if support.shape != (directions.shape[0],):
        raise ValueError("support must match the direction count")
    return (points_xy @ directions.T - support.unsqueeze(0)).amax(dim=-1)


def maximum_sector_point_free_envelope(
    cloud: SyntheticLidarCloud,
    directions: Tensor,
    *,
    point_clearance: float,
    cap_support: Tensor,
) -> PointFreeEnvelope:
    """Return the coordinatewise-maximal capped fixed-normal envelope.

    A return constrains only its assigned outward normal. Point-supported faces
    use the minimum assigned projection minus ``point_clearance``; faces
    without returns retain ``cap_support``. Raising any face either violates
    its active return or the declared pre-obstacle cap.
    """
    if point_clearance <= 0.0:
        raise ValueError("point_clearance must be positive")
    sectors = directions.shape[0]
    if cap_support.shape != (sectors,):
        raise ValueError("cap_support must match the direction count")
    if cloud.sector_indices.shape != (cloud.points_xy.shape[0],):
        raise ValueError("cloud sector indices must match its points")
    projections = (cloud.points_xy * directions[cloud.sector_indices]).sum(dim=-1)
    mask = cloud.sector_indices[:, None] == torch.arange(sectors, device=directions.device)[None, :]
    assigned = projections[:, None].expand(-1, sectors).masked_fill(~mask, torch.inf)
    minimum_projection, limiting_indices = assigned.min(dim=0)
    constrained = torch.isfinite(minimum_projection)
    point_support = minimum_projection - point_clearance
    support = torch.where(
        constrained, torch.minimum(point_support, cap_support), cap_support,
    )
    if bool((support <= 0.0).any()):
        raise ValueError("point clearance leaves a non-positive support")
    constrained_faces = torch.nonzero(constrained, as_tuple=False).squeeze(-1)
    unconstrained_faces = torch.nonzero(~constrained, as_tuple=False).squeeze(-1)
    limiting_indices = limiting_indices[constrained]
    limiting_points = cloud.points_xy[limiting_indices]
    face_directions = directions[constrained_faces]
    face_distance = (
        limiting_points * face_directions
    ).sum(dim=-1) - support[constrained_faces]
    clearance_feet = limiting_points - face_distance[:, None] * face_directions
    return PointFreeEnvelope(
        support_m=support,
        limiting_point_indices=limiting_indices,
        limiting_points_xy=limiting_points,
        clearance_feet_xy=clearance_feet,
        constrained_face_indices=constrained_faces,
        unconstrained_face_indices=unconstrained_faces,
        cap_support_m=cap_support,
        point_clearance_m=float(point_clearance),
        optimality_scope=(
            "coordinatewise maximum in the declared fixed-normal capped polygon "
            "family under nearest-angular-sector point assignment"
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
