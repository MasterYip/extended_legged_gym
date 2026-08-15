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
    frame. Every direction comes from the current URDF hip-to-foot FK vector,
    projected into body XY and normalized in the physical hip-height plane.
    """
    if current_q.shape != (kinematics.num_dof,) or haa_ranges.shape != (6, 2):
        raise ValueError("Expected current_q [18] and haa_ranges [6,2]")
    if samples < 3 or radius <= 0.0:
        raise ValueError("samples must be >= 3 and radius must be positive")
    haa_indices = [EL4090_JOINT_NAMES.index(f"{leg}_HAA") for leg in EL4090_LEG_NAMES]
    alpha = torch.linspace(0.0, 1.0, samples, dtype=current_q.dtype, device=current_q.device)
    arc_q = current_q.repeat(samples, 1)
    arc_q[:, haa_indices] = torch.lerp(haa_ranges[:, 0], haa_ranges[:, 1], alpha[:, None])
    hip_links = [f"{leg}_HIP" for leg in EL4090_LEG_NAMES]
    foot_links = [f"{leg}_FOOT" for leg in EL4090_LEG_NAMES]
    local_origins = torch.zeros(
        (6, 1, 3), dtype=current_q.dtype, device=current_q.device,
    )

    def planar_hip_to_foot(q: Tensor) -> Tuple[Tensor, Tensor]:
        hips = kinematics.points(q, hip_links, local_origins)[..., 0, :]
        feet = kinematics.points(q, foot_links, local_origins)[..., 0, :]
        direction = feet - hips
        direction[..., 2] = 0.0
        norms = direction.norm(dim=-1, keepdim=True)
        if bool((norms <= torch.finfo(current_q.dtype).eps).any()):
            raise ValueError("hip-to-foot XY directions must be nonzero")
        return hips, direction / norms

    arc_origins, arc_directions = planar_hip_to_foot(arc_q)
    arcs = (arc_origins + radius * arc_directions).transpose(0, 1)
    current_origins, current_directions = planar_hip_to_foot(
        current_q.unsqueeze(0),
    )
    origins = current_origins[0]
    markers = origins + radius * current_directions[0]
    return origins, arcs, markers


def haa_arc_geometry_interval(
    kinematics: BatchedUrdfKinematics,
    current_q: Tensor,
    haa_ranges: Tensor,
    leg_index: int,
    interval_lo: float,
    interval_hi: float,
    *,
    radius: float = 0.22,
    samples: int = 17,
) -> Tensor:
    """Return one leg's body-frame arc polyline ``[samples,3]`` over an HAA sub-interval.

    Only the named leg's HAA is swept over ``[interval_lo, interval_hi]``; every
    other joint (including the other legs' HAA) is pinned at ``current_q``.
    Because each leg is an independent serial chain from the base, this matches
    the same leg's arc from ``haa_arc_geometry`` exactly when the interval is
    the full exported range. Used to overdraw rejected sub-intervals in a
    distinct color without changing ``haa_arc_geometry``.
    """
    if current_q.shape != (kinematics.num_dof,) or haa_ranges.shape != (6, 2):
        raise ValueError("Expected current_q [18] and haa_ranges [6,2]")
    if not 0 <= leg_index < 6:
        raise ValueError("leg_index must be in [0, 6)")
    if interval_hi < interval_lo:
        raise ValueError("interval_lo must not exceed interval_hi")
    if samples < 3 or radius <= 0.0:
        raise ValueError("samples must be >= 3 and radius must be positive")
    haa_indices = [EL4090_JOINT_NAMES.index(f"{leg}_HAA") for leg in EL4090_LEG_NAMES]
    leg_ranges = haa_ranges.clone()
    for index in range(6):
        if index != leg_index:
            value = current_q[haa_indices[index]]
            leg_ranges[index] = torch.stack((value, value))
    leg_ranges[leg_index] = current_q.new_tensor((interval_lo, interval_hi))
    _, arcs, _ = haa_arc_geometry(
        kinematics, current_q, leg_ranges, radius=radius, samples=samples,
    )
    return arcs[leg_index]


def joint_arc_geometry_interval(
    kinematics: BatchedUrdfKinematics,
    current_q: Tensor,
    leg_index: int,
    joint_kind: str,
    interval_lo: float,
    interval_hi: float,
    *,
    radius: float = 0.22,
    samples: int = 17,
) -> Tensor:
    """One leg's body-frame stylized foot-direction arc over a joint sub-interval.

    ``joint_kind`` is one of ``"HAA"``, ``"HFE"``, ``"KFE"``. The named joint of
    the given leg is swept over ``[interval_lo, interval_hi]`` with every other
    joint (including the other legs) pinned at ``current_q``. Arc points are the
    hip origin plus ``radius * unit(hip-to-foot XY)``, the same stylization as
    ``haa_arc_geometry``, so the HAA case reproduces the HAA arc exactly. Used to
    draw rejected HFE/KFE bands (forward/back motion) in the same style as the
    HAA rejection bands.
    """
    if current_q.shape != (kinematics.num_dof,):
        raise ValueError("current_q must be [num_dof]")
    if not 0 <= leg_index < len(EL4090_LEG_NAMES):
        raise ValueError("leg_index must be in [0, num_legs)")
    if joint_kind not in ("HAA", "HFE", "KFE"):
        raise ValueError("joint_kind must be one of HAA, HFE, KFE")
    if interval_hi < interval_lo:
        raise ValueError("interval_lo must not exceed interval_hi")
    if samples < 3 or radius <= 0.0:
        raise ValueError("samples must be >= 3 and radius must be positive")
    joint_index = EL4090_JOINT_NAMES.index(f"{EL4090_LEG_NAMES[leg_index]}_{joint_kind}")
    alpha = torch.linspace(0.0, 1.0, samples, dtype=current_q.dtype, device=current_q.device)
    sweep_q = current_q.repeat(samples, 1)
    sweep_q[:, joint_index] = interval_lo + alpha * (interval_hi - interval_lo)
    hip_link = f"{EL4090_LEG_NAMES[leg_index]}_HIP"
    foot_link = f"{EL4090_LEG_NAMES[leg_index]}_FOOT"
    local = torch.zeros((1, 1, 3), dtype=current_q.dtype, device=current_q.device)
    hips = kinematics.points(sweep_q, (hip_link,), local)[..., 0, 0, :]
    feet = kinematics.points(sweep_q, (foot_link,), local)[..., 0, 0, :]
    direction = feet - hips
    direction[..., 2] = 0.0
    norms = direction.norm(dim=-1, keepdim=True)
    if bool((norms <= torch.finfo(current_q.dtype).eps).any()):
        raise ValueError("hip-to-foot XY directions must be nonzero")
    return hips + radius * direction / norms


def accessible_interval_complement(
    lo: float,
    hi: float,
    rejected: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    """Complement of the rejected sub-intervals within ``[lo, hi]``.

    The rejected intervals are expected sorted and non-overlapping (as produced
    by ``joint_rejection_ranges``). The result tiles ``[lo, hi]`` exactly with
    the rejected intervals: union of the returned accessible gaps equals
    ``[lo, hi]`` minus the rejected union, and no accessible gap overlaps a
    rejected interval. Used to draw the amber "accessible" HAA arc only over
    the non-rejected sub-ranges so it never overlaps the magenta rejected bands.
    """
    accessible: list[Tuple[float, float]] = []
    cursor = lo
    for r_lo, r_hi in rejected:
        if r_hi <= lo or r_lo >= hi:
            continue  # entirely outside [lo, hi]; must not consume the range
        low = max(cursor, r_lo)
        high = min(hi, r_hi)
        if cursor < low:
            accessible.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < hi:
        accessible.append((cursor, hi))
    return tuple(accessible)


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
