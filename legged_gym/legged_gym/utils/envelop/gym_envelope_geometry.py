"""Isaac-independent geometry helpers for the EL4090 envelope viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from kinematic_envelope import (
    EL4090_JOINT_NAMES,
    EL4090_LEG_NAMES,
    BatchedUrdfKinematics,
    CapsuleProxy,
    EnvelopeJointRangeExport,
    capsule_support,
    default_el4090_capsules,
    deterministic_joint_samples,
    export_envelope_joint_ranges,
    foot_positions,
    haa_ranges_from_joint_export,
    reachable_foot_support,
)


@dataclass(frozen=True)
class DemoPreset:
    name: str
    current_q: Tensor
    candidate_q: Tensor
    occupied_support: Tensor
    allowed_support: Tensor
    reachable_support: Tensor
    haa_ranges: Tensor
    range_export: EnvelopeJointRangeExport
    support_margin: float
    accent_rgb: Tuple[float, float, float]


def support_polygon(directions: Tensor | np.ndarray, support: Tensor | np.ndarray) -> np.ndarray:
    """Intersect planar half spaces and return counter-clockwise polygon vertices."""
    normals = np.asarray(directions.detach().cpu() if isinstance(directions, Tensor) else directions, dtype=np.float64)
    bounds = np.asarray(support.detach().cpu() if isinstance(support, Tensor) else support, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 2 or bounds.shape != (normals.shape[0],):
        raise ValueError("Expected directions [K,2] and support [K]")
    extent = max(1.0, 4.0 * float(np.max(np.abs(bounds))))
    polygon = np.array(((-extent, -extent), (extent, -extent), (extent, extent), (-extent, extent)))
    for normal, bound in zip(normals, bounds):
        clipped = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            start_in = float(normal @ start) <= bound + 1e-10
            end_in = float(normal @ end) <= bound + 1e-10
            if start_in:
                clipped.append(start)
            if start_in != end_in:
                delta = end - start
                denominator = float(normal @ delta)
                if abs(denominator) > 1e-12:
                    clipped.append(start + delta * ((bound - float(normal @ start)) / denominator))
        polygon = np.asarray(clipped, dtype=np.float64)
        if polygon.size == 0:
            raise ValueError("Half-space intersection is empty")
    return polygon


def polyline_segments(points: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Convert points ``[N,3]`` to Isaac Gym line vertices ``[2M,3]``."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("points must have shape [N,3] with N >= 2")
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    starts = points if closed else points[:-1]
    return np.stack((starts, ends), axis=1).reshape(-1, 3)


def interpolate_joint_ranges(
    lower: Tensor,
    upper: Tensor,
    phase: float | Tensor,
    *,
    phase_offsets: Tensor | None = None,
) -> Tensor:
    """Return a smooth deterministic pose inside exported joint ranges.

    ``phase`` is measured in cycles. Per-joint offsets allow coordinated motion
    without changing the exported bounds that define the interpolation.
    """
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower and upper must have matching one-dimensional shapes")
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("joint range bounds must be finite")
    if bool((lower > upper).any()):
        raise ValueError("joint range lower bounds must not exceed upper bounds")
    if phase_offsets is None:
        phase_offsets = torch.zeros_like(lower)
    elif phase_offsets.shape != lower.shape:
        raise ValueError("phase_offsets must match the joint range shape")
    phase_tensor = torch.as_tensor(phase, dtype=lower.dtype, device=lower.device)
    blend = 0.5 - 0.5 * torch.cos(2.0 * torch.pi * (phase_tensor + phase_offsets))
    pose = torch.lerp(lower, upper, blend)
    return torch.maximum(lower, torch.minimum(upper, pose))


def joint_range_violations(
    current_q: Tensor, lower: Tensor, upper: Tensor, *, tolerance: float = 1e-6,
) -> Tuple[int, float]:
    """Return violating joint count and maximum interval excess in radians."""
    if current_q.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError("current_q, lower, and upper must have matching shapes")
    excess = torch.maximum(lower - current_q, current_q - upper)
    excess = torch.clamp_min(excess, 0.0)
    return int((excess > tolerance).sum().item()), float(excess.max().item())


def haa_arc_geometry(
    kinematics: BatchedUrdfKinematics,
    current_q: Tensor,
    haa_ranges: Tensor,
    *,
    radius: float = 0.22,
    samples: int = 33,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return physical hip origins, interval arcs, and current HAA markers.

    Results have shapes ``[6,3]``, ``[6,A,3]``, and ``[6,3]`` in the body
    frame. The arc orientation comes from the actual URDF hip transforms.
    """
    if current_q.shape != (kinematics.num_dof,) or haa_ranges.shape != (6, 2):
        raise ValueError("Expected current_q [18] and haa_ranges [6,2]")
    if samples < 3 or radius <= 0.0:
        raise ValueError("samples must be >= 3 and radius must be positive")
    haa_indices = [EL4090_JOINT_NAMES.index(f"{leg}_HAA") for leg in EL4090_LEG_NAMES]
    alpha = torch.linspace(0.0, 1.0, samples, dtype=current_q.dtype, device=current_q.device)
    arc_q = current_q.repeat(samples, 1)
    arc_q[:, haa_indices] = torch.lerp(haa_ranges[:, 0], haa_ranges[:, 1], alpha[:, None])
    links = [f"{leg}_HIP" for leg in EL4090_LEG_NAMES]
    joints_by_name = {joint.name: joint for joint in kinematics.joints}
    hfe_offsets = current_q.new_tensor([
        joints_by_name[f"{leg}_HFE"].origin_xyz for leg in EL4090_LEG_NAMES
    ])
    hfe_offset_norms = hfe_offsets.norm(dim=-1, keepdim=True)
    if bool((hfe_offset_norms <= torch.finfo(current_q.dtype).eps).any()):
        raise ValueError("HFE joint origins must define nonzero outward leg directions")
    origins_local = torch.zeros((6, 1, 3), dtype=current_q.dtype, device=current_q.device)
    marker_local = radius * hfe_offsets[:, None, :] / hfe_offset_norms[:, None, :]
    origins = kinematics.points(current_q.unsqueeze(0), links, origins_local)[0, :, 0]
    arcs = kinematics.points(arc_q, links, marker_local)[:, :, 0].transpose(0, 1)
    markers = kinematics.points(current_q.unsqueeze(0), links, marker_local)[0, :, 0]
    return origins, arcs, markers


def build_demo_preset(
    name: str,
    kinematics: BatchedUrdfKinematics,
    directions: Tensor,
    current_q: Tensor,
    candidate_half_width: Tensor,
    effective_lower: Tensor,
    effective_upper: Tensor,
    *,
    support_margin: float,
    accent_rgb: Sequence[float],
    seed: int,
    candidate_count: int = 257,
    validation_count: int = 129,
    box_validation_samples: int = 128,
    capsules: Sequence[CapsuleProxy] | None = None,
) -> DemoPreset:
    """Build one deterministic, envelope-conditioned comparison preset."""
    proxies = default_el4090_capsules() if capsules is None else tuple(capsules)
    candidate_lower = torch.maximum(current_q - candidate_half_width, effective_lower)
    candidate_upper = torch.minimum(current_q + candidate_half_width, effective_upper)
    candidate_q = torch.cat((
        current_q.unsqueeze(0),
        deterministic_joint_samples(candidate_lower, candidate_upper, candidate_count - 1, seed=seed),
    ))
    validation_q = deterministic_joint_samples(
        candidate_lower, candidate_upper, validation_count, seed=seed + 1000,
    )
    occupied = capsule_support(kinematics, current_q.unsqueeze(0), proxies, directions)[0]
    allowed = occupied + support_margin
    export = export_envelope_joint_ranges(
        kinematics,
        candidate_q,
        validation_q,
        directions,
        allowed,
        effective_lower,
        effective_upper,
        capsules=proxies,
        box_validation_samples=box_validation_samples,
        box_validation_seed=seed + 2000,
    )
    if not bool(export.valid):
        raise RuntimeError(f"Preset {name!r} has no feasible candidate")
    candidate_feasible = (
        capsule_support(kinematics, candidate_q, proxies, directions) <= allowed.unsqueeze(0)
    ).all(dim=-1)
    reachable = reachable_foot_support(
        kinematics, candidate_q[candidate_feasible].unsqueeze(0), directions,
    )[0]
    return DemoPreset(
        name=name,
        current_q=current_q,
        candidate_q=candidate_q,
        occupied_support=occupied,
        allowed_support=allowed,
        reachable_support=reachable,
        haa_ranges=haa_ranges_from_joint_export(export),
        range_export=export,
        support_margin=float(support_margin),
        accent_rgb=tuple(float(value) for value in accent_rgb),
    )
