"""Exact ZhangHT legacy foot-workspace controls and sampled range export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import torch
from torch import Tensor

from kinematic_envelope import (
    BatchedUrdfKinematics,
    JointRangeExport,
    deterministic_joint_samples,
    export_sample_bounding_ranges,
    foot_positions,
)


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
LEGACY_VISUAL_SEMANTICS = {
    "white": "ZhangHT legacy hard outer foot-workspace border",
    "light_cyan": "sampled maximal admissible front/rear foot workspace",
    "dark_teal": "current rear/front six-foot hulls",
    "amber": "exported HAA ranges and current markers",
    "red": "legacy foot-workspace violation only",
}


@dataclass(frozen=True)
class LegacyEnvelopeResult:
    parameters: Tuple[float, ...]
    border_vertices_xy: Tensor
    maximal_rear_vertices_xy: Tensor
    maximal_front_vertices_xy: Tensor
    feasible_mask: Tensor
    validation_feasible_mask: Tensor
    current_q: Tensor
    current_feet_xy: Tensor
    range_export: JointRangeExport
    box_validation_samples: int
    box_foot_violation_count: int
    max_box_foot_violation_m: float


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
    """Return ZhangHT's exact six-point symmetric footprint in body XY."""
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


def legacy_half_width(x: Tensor, parameters: Tensor) -> Tensor:
    front, middle, back, forward, backward = parameters.unbind()
    rear_alpha = ((x - backward) / (-backward).clamp_min(1e-9)).clamp(0.0, 1.0)
    front_alpha = (x / forward.clamp_min(1e-9)).clamp(0.0, 1.0)
    rear_width = torch.lerp(back, middle, rear_alpha)
    front_width = torch.lerp(middle, front, front_alpha)
    return torch.where(x <= 0.0, rear_width, front_width)


def legacy_foot_excess(feet_xy: Tensor, parameters: Tensor) -> Tensor:
    """Per-foot signed violation of ZhangHT's exact piecewise workspace."""
    if feet_xy.shape[-1] != 2:
        raise ValueError("feet_xy must end in XY coordinates")
    forward = parameters[3]
    backward = parameters[4]
    x = feet_xy[..., 0]
    y = feet_xy[..., 1]
    width = legacy_half_width(x, parameters)
    return torch.stack((x - forward, backward - x, y.abs() - width), dim=-1).amax(-1)


def legacy_feasible(feet_xy: Tensor, parameters: Tensor, tolerance: float = 1e-6) -> Tensor:
    return legacy_foot_excess(feet_xy, parameters).amax(dim=-1) <= tolerance


def convex_hull_2d(points: Tensor) -> Tensor:
    """Return a deterministic monotone-chain hull for a small 2D point set."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    ordered = sorted(set((float(x), float(y)) for x, y in points.detach().cpu()))
    if len(ordered) <= 2:
        return points.new_tensor(ordered)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return points.new_tensor(lower[:-1] + upper[:-1])


def compute_legacy_admissible_envelope(
    kinematics: BatchedUrdfKinematics,
    parameters: Tensor,
    candidate_q: Tensor,
    validation_q: Tensor,
    effective_lower: Tensor,
    effective_upper: Tensor,
    *,
    tolerance: float = 1e-6,
    box_validation_samples: int = 128,
    box_validation_seed: int = 4090,
) -> LegacyEnvelopeResult:
    """Compute sampled maximal foot workspaces and an approximate joint box.

    Maximality means the convex hull of registered feasible foot samples within
    each legacy convex half (rear/front). Keeping two hulls preserves the exact
    potentially concave legacy border. The axis-aligned joint range is audited,
    not claimed conservative, because it can combine incompatible joint modes.
    """
    parameters = parameters.to(candidate_q)
    candidate_feet = foot_positions(kinematics, candidate_q)[..., :2]
    validation_feet = foot_positions(kinematics, validation_q)[..., :2]
    feasible = legacy_feasible(candidate_feet, parameters, tolerance)
    validation_feasible = legacy_feasible(validation_feet, parameters, tolerance)
    if not bool(feasible.any()):
        raise ValueError("legacy border contains no sampled EL4090 foot pose")

    feasible_q = candidate_q[feasible]
    registered_validation = validation_q[validation_feasible]
    export = export_sample_bounding_ranges(
        feasible_q,
        registered_validation,
        effective_lower,
        effective_upper,
        feasibility_tolerance=tolerance,
    )
    closest = ((feasible_q - export.center.unsqueeze(0)) ** 2).sum(dim=-1).argmin()
    current_q = feasible_q[closest]
    current_feet = foot_positions(kinematics, current_q.unsqueeze(0))[0, :, :2]

    feasible_points = candidate_feet[feasible].reshape(-1, 2)
    rear_points = feasible_points[feasible_points[:, 0] <= 0.0]
    front_points = feasible_points[feasible_points[:, 0] >= 0.0]
    if rear_points.shape[0] < 3 or front_points.shape[0] < 3:
        raise ValueError("insufficient feasible foot samples for both workspace halves")

    if box_validation_samples < 1:
        raise ValueError("box_validation_samples must be positive")
    box_q = deterministic_joint_samples(
        export.lower, export.upper, box_validation_samples, seed=box_validation_seed,
    )
    box_excess = legacy_foot_excess(
        foot_positions(kinematics, box_q)[..., :2], parameters,
    ).amax(dim=-1)
    return LegacyEnvelopeResult(
        parameters=tuple(float(value) for value in parameters),
        border_vertices_xy=legacy_border_vertices(parameters),
        maximal_rear_vertices_xy=convex_hull_2d(rear_points),
        maximal_front_vertices_xy=convex_hull_2d(front_points),
        feasible_mask=feasible,
        validation_feasible_mask=validation_feasible,
        current_q=current_q,
        current_feet_xy=current_feet,
        range_export=export,
        box_validation_samples=box_validation_samples,
        box_foot_violation_count=int((box_excess > tolerance).sum()),
        max_box_foot_violation_m=float(box_excess.clamp_min(0.0).max()),
    )
