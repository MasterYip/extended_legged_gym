"""Batched Torch kinematics and capsule support envelopes for EL4090.

The module deliberately keeps two models separate:

* occupied-body support uses conservative capsule proxies for robot links;
* reachable-foot support uses sampled forward-kinematic foot positions.

Capsules are an explicit approximation of link meshes.  A zero violation rate
therefore means conservative only with respect to the registered validation
samples and capsule model, not collision completeness for the physical robot.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


EL4090_JOINT_NAMES: Tuple[str, ...] = (
    "LB_HAA", "LB_HFE", "LB_KFE", "LF_HAA", "LF_HFE", "LF_KFE",
    "LM_HAA", "LM_HFE", "LM_KFE", "RB_HAA", "RB_HFE", "RB_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE", "RM_HAA", "RM_HFE", "RM_KFE",
)
EL4090_LEG_NAMES: Tuple[str, ...] = ("LB", "LF", "LM", "RB", "RF", "RM")
LEGACY_HAA_ORDER: Tuple[str, ...] = ("RF", "RM", "RB", "LF", "LM", "LB")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class CapsuleProxy:
    """Capsule between two link-frame points, transformed by ``link``."""

    name: str
    link: str
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class JointRangeDiagnostics:
    validation_samples: int
    violation_count: int
    violation_rate: float
    max_violation: float
    conservative_on_validation: bool
    label: str


@dataclass(frozen=True)
class JointRangeExport:
    lower: Tensor
    upper: Tensor
    center: Tensor
    half_range: Tensor
    diagnostics: JointRangeDiagnostics


@dataclass(frozen=True)
class EnvelopeRangeDiagnostics:
    candidate_samples: int
    candidate_feasible_count: Tensor
    validation_samples: int
    validation_feasible_count: Tensor
    false_exclusion_count: Tensor
    box_validation_samples: int
    box_envelope_violation_count: Tensor
    box_envelope_violation_rate: Tensor
    max_box_envelope_violation: Tensor
    label: str


@dataclass(frozen=True)
class EnvelopeJointRangeExport:
    lower: Tensor
    upper: Tensor
    center: Tensor
    half_range: Tensor
    valid: Tensor
    diagnostics: EnvelopeRangeDiagnostics


def _vec(text: Optional[str], default: Sequence[float]) -> Tuple[float, float, float]:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got {values}")
    return tuple(values)  # type: ignore[return-value]


def load_urdf_joints(path: Path | str) -> Tuple[UrdfJoint, ...]:
    """Parse the kinematic fields required by Torch FK."""
    root = ET.parse(path).getroot()
    joints = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        joints.append(
            UrdfJoint(
                name=node.attrib["name"],
                joint_type=node.attrib["type"],
                parent=node.find("parent").attrib["link"],
                child=node.find("child").attrib["link"],
                origin_xyz=_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                origin_rpy=_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                lower=float(limit.attrib.get("lower", "-inf")) if limit is not None else -math.inf,
                upper=float(limit.attrib.get("upper", "inf")) if limit is not None else math.inf,
            )
        )
    return tuple(joints)


def _rpy_matrix(rpy: Tensor) -> Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    axis = axis / axis.norm().clamp_min(torch.finfo(angle.dtype).eps)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(3, 3)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    outer = axis[:, None] * axis[None, :]
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    return c * eye + (1.0 - c) * outer + s * skew


def _transform(rotation: Tensor, translation: Tensor) -> Tensor:
    shape = rotation.shape[:-2]
    result = torch.zeros((*shape, 4, 4), dtype=rotation.dtype, device=rotation.device)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


class BatchedUrdfKinematics:
    """Differentiable URDF forward kinematics with no loop over batch items."""

    def __init__(
        self,
        joints: Sequence[UrdfJoint],
        actuated_joint_names: Sequence[str] = EL4090_JOINT_NAMES,
    ) -> None:
        self.joints = tuple(joints)
        self.actuated_joint_names = tuple(actuated_joint_names)
        self._q_index = {name: index for index, name in enumerate(self.actuated_joint_names)}
        children = {joint.child for joint in self.joints}
        parents = {joint.parent for joint in self.joints}
        roots = sorted(parents - children)
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root, got {roots}")
        self.root_link = roots[0]
        ordered = []
        available = {self.root_link}
        remaining = list(self.joints)
        while remaining:
            ready = [joint for joint in remaining if joint.parent in available]
            if not ready:
                raise ValueError("URDF joint graph is disconnected or cyclic")
            for joint in ready:
                ordered.append(joint)
                available.add(joint.child)
                remaining.remove(joint)
        self.ordered_joints = tuple(ordered)

    @property
    def num_dof(self) -> int:
        return len(self.actuated_joint_names)

    def joint_limits(
        self,
        *,
        soft_fraction: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        by_name = {joint.name: joint for joint in self.joints}
        lower = torch.tensor([by_name[name].lower for name in self.actuated_joint_names], dtype=dtype, device=device)
        upper = torch.tensor([by_name[name].upper for name in self.actuated_joint_names], dtype=dtype, device=device)
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) * float(soft_fraction)
        return center - half, center + half

    def forward(self, q: Tensor, root_transform: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if q.ndim < 2 or q.shape[-1] != self.num_dof:
            raise ValueError(f"Expected q shape [...,{self.num_dof}], got {tuple(q.shape)}")
        batch_shape = q.shape[:-1]
        if root_transform is None:
            root_transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(*batch_shape, 4, 4).clone()
        elif root_transform.shape != (*batch_shape, 4, 4):
            raise ValueError(f"Expected root transform {(*batch_shape,4,4)}, got {tuple(root_transform.shape)}")
        transforms: Dict[str, Tensor] = {self.root_link: root_transform}
        for joint in self.ordered_joints:
            xyz = q.new_tensor(joint.origin_xyz).expand(*batch_shape, 3)
            rpy = q.new_tensor(joint.origin_rpy).expand(*batch_shape, 3)
            local = _transform(_rpy_matrix(rpy), xyz)
            if joint.joint_type in {"revolute", "continuous"}:
                if joint.name not in self._q_index:
                    raise ValueError(f"Actuated joint {joint.name!r} is missing from q order")
                axis = q.new_tensor(joint.axis)
                angle_tf = _transform(
                    _axis_angle_matrix(axis, q[..., self._q_index[joint.name]]),
                    torch.zeros((*batch_shape, 3), dtype=q.dtype, device=q.device),
                )
                local = local @ angle_tf
            transforms[joint.child] = transforms[joint.parent] @ local
        return transforms

    def points(self, q: Tensor, links: Sequence[str], local_points: Tensor) -> Tensor:
        """Transform ``[L,P,3]`` local points to ``[...,L,P,3]``."""
        if local_points.ndim != 3 or local_points.shape[0] != len(links) or local_points.shape[-1] != 3:
            raise ValueError("local_points must have shape [len(links),P,3]")
        transforms = self.forward(q)
        matrices = torch.stack([transforms[link] for link in links], dim=-3)
        rotation = matrices[..., :3, :3]
        translation = matrices[..., :3, 3]
        points = local_points.to(dtype=q.dtype, device=q.device)
        return torch.einsum("...lij,lpj->...lpi", rotation, points) + translation.unsqueeze(-2)


def default_el4090_capsules() -> Tuple[CapsuleProxy, ...]:
    """Explicit low-cost link proxies calibrated from EL4090 URDF joint spans."""
    proxies = [CapsuleProxy("base_x", "BASE", (-0.45, 0.0, 0.0), (0.45, 0.0, 0.0), 0.12)]
    for leg in EL4090_LEG_NAMES:
        proxies.extend(
            (
                CapsuleProxy(f"{leg}_hip", f"{leg}_HIP", (0.0, 0.0, 0.0), (-0.09, 0.0795, 0.0), 0.07),
                CapsuleProxy(f"{leg}_thigh", f"{leg}_THIGH", (0.0, 0.0, 0.0), (-0.1782, 0.090798, 0.0), 0.065),
                CapsuleProxy(f"{leg}_shank", f"{leg}_SHANK", (0.0, 0.0, 0.0), (0.0, -0.2992, 0.0), 0.055),
            )
        )
    return tuple(proxies)


def support_directions(
    count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    if count < 3:
        raise ValueError("At least three support directions are required")
    theta = torch.arange(count, dtype=dtype, device=device) * (2.0 * math.pi / count)
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)


def capsule_support(
    kinematics: BatchedUrdfKinematics,
    q: Tensor,
    capsules: Sequence[CapsuleProxy],
    directions: Tensor,
) -> Tensor:
    """Return occupied capsule support values with shape ``[...,K]``."""
    links = [capsule.link for capsule in capsules]
    local_points = q.new_tensor([[capsule.start, capsule.end] for capsule in capsules])
    endpoints = kinematics.points(q, links, local_points)[..., :2]
    directions = directions.to(dtype=q.dtype, device=q.device)
    endpoint_projection = torch.einsum("...lpi,ki->...lpk", endpoints, directions)
    radii = q.new_tensor([capsule.radius for capsule in capsules])
    return (endpoint_projection.max(dim=-2).values + radii[:, None]).max(dim=-2).values


def foot_positions(
    kinematics: BatchedUrdfKinematics,
    q: Tensor,
    leg_names: Sequence[str] = EL4090_LEG_NAMES,
) -> Tensor:
    links = [f"{leg}_FOOT" for leg in leg_names]
    local = torch.zeros((len(links), 1, 3), dtype=q.dtype, device=q.device)
    return kinematics.points(q, links, local).squeeze(-2)


def point_violations(points: Tensor, directions: Tensor, support: Tensor) -> Tensor:
    """Maximum half-space violation for body-frame points ``[...,N,2|3]``."""
    directions = directions.to(dtype=points.dtype, device=points.device)
    projections = torch.einsum("...ni,ki->...nk", points[..., :2], directions)
    return (projections - support.unsqueeze(-2)).max(dim=-1).values


def contains_points(points: Tensor, directions: Tensor, support: Tensor, tolerance: float = 0.0) -> Tensor:
    return point_violations(points, directions, support) <= tolerance


def add_support_margin(support: Tensor, margin: float | Tensor) -> Tensor:
    """Exact metric offset for unit-normal half spaces."""
    return support + torch.as_tensor(margin, dtype=support.dtype, device=support.device)


def reachable_foot_support(
    kinematics: BatchedUrdfKinematics,
    q_samples: Tensor,
    directions: Tensor,
) -> Tensor:
    """Support of FK foot samples; ``q_samples`` is ``[...,S,18]``."""
    feet = foot_positions(kinematics, q_samples)
    directions = directions.to(dtype=q_samples.dtype, device=q_samples.device)
    projected = torch.einsum("...sli,ki->...slk", feet[..., :2], directions)
    return projected.amax(dim=(-3, -2))


def export_sample_bounding_ranges(
    feasible_samples: Tensor,
    validation_samples: Tensor,
    effective_lower: Tensor,
    effective_upper: Tensor,
    *,
    feasibility_tolerance: float = 0.0,
) -> JointRangeExport:
    """Low-level box around samples already classified as feasible by a caller.

    ``feasible_samples`` defines the box. ``validation_samples`` independently
    diagnoses whether registered feasible samples fall outside that box.
    """
    if feasible_samples.ndim < 2 or feasible_samples.shape[-1] != effective_lower.numel():
        raise ValueError("feasible_samples must be [...,S,J]")
    if validation_samples.ndim != feasible_samples.ndim or validation_samples.shape[:-2] != feasible_samples.shape[:-2] or validation_samples.shape[-1] != effective_lower.numel():
        raise ValueError("validation_samples must be [...,V,J] with the same batch prefix")
    lower = torch.maximum(feasible_samples.amin(dim=-2), effective_lower)
    upper = torch.minimum(feasible_samples.amax(dim=-2), effective_upper)
    if torch.any(lower > upper):
        raise ValueError("Sample-derived ranges are empty after clipping")
    low_violation = torch.relu(lower.unsqueeze(-2) - validation_samples)
    high_violation = torch.relu(validation_samples - upper.unsqueeze(-2))
    per_sample = torch.maximum(low_violation, high_violation).amax(dim=-1)
    violation_mask = per_sample > feasibility_tolerance
    count = int(violation_mask.sum().item())
    total = int(validation_samples.numel() // validation_samples.shape[-1])
    max_violation = float(per_sample.max().item()) if total else 0.0
    zero = count == 0
    diagnostics = JointRangeDiagnostics(
        validation_samples=total,
        violation_count=count,
        violation_rate=count / max(total, 1),
        max_violation=max_violation,
        conservative_on_validation=zero,
        label="conservative on registered validation set" if zero else "approximate",
    )
    return JointRangeExport(
        lower=lower,
        upper=upper,
        center=0.5 * (lower + upper),
        half_range=0.5 * (upper - lower),
        diagnostics=diagnostics,
    )


def export_envelope_joint_ranges(
    kinematics: BatchedUrdfKinematics,
    candidate_q: Tensor,
    validation_q: Tensor,
    directions: Tensor,
    allowed_support: Tensor,
    effective_lower: Tensor,
    effective_upper: Tensor,
    *,
    capsules: Optional[Sequence[CapsuleProxy]] = None,
    tolerance: float = 0.0,
    box_validation_samples: int = 256,
    box_validation_seed: int = 4090,
) -> EnvelopeJointRangeExport:
    """Derive and audit joint boxes under an occupied-support constraint.

    Inputs use shapes ``[...,S,J]``, ``[...,V,J]``, and broadcastable
    ``[...,K]``.  Returned ranges are NaN where no candidate is feasible.
    Cartesian-box samples independently diagnose whether combinations inside
    each exported axis-aligned box still satisfy the requested envelope.
    """
    if candidate_q.ndim < 2 or candidate_q.shape[-1] != kinematics.num_dof:
        raise ValueError("candidate_q must be [...,S,J]")
    if candidate_q.shape[-2] == 0:
        raise ValueError("candidate_q must contain at least one sample")
    if validation_q.ndim != candidate_q.ndim or validation_q.shape[:-2] != candidate_q.shape[:-2] or validation_q.shape[-1] != kinematics.num_dof:
        raise ValueError("validation_q must be [...,V,J] with the same batch prefix")
    if box_validation_samples < 1:
        raise ValueError("box_validation_samples must be positive")
    prefix = candidate_q.shape[:-2]
    count = candidate_q.shape[-2]
    validation_count = validation_q.shape[-2]
    directions = directions.to(dtype=candidate_q.dtype, device=candidate_q.device)
    allowed = torch.broadcast_to(
        allowed_support.to(dtype=candidate_q.dtype, device=candidate_q.device),
        (*prefix, directions.shape[0]),
    )
    proxies = default_el4090_capsules() if capsules is None else tuple(capsules)
    effective_lower = torch.broadcast_to(effective_lower.to(candidate_q), (*prefix, kinematics.num_dof))
    effective_upper = torch.broadcast_to(effective_upper.to(candidate_q), (*prefix, kinematics.num_dof))
    if torch.any(effective_lower > effective_upper):
        raise ValueError("effective_lower must not exceed effective_upper")
    candidate_violation = capsule_support(kinematics, candidate_q, proxies, directions) - allowed.unsqueeze(-2)
    candidate_in_limits = ((candidate_q >= effective_lower.unsqueeze(-2)) & (candidate_q <= effective_upper.unsqueeze(-2))).all(dim=-1)
    candidate_feasible = (candidate_violation.amax(dim=-1) <= tolerance) & candidate_in_limits
    feasible_count = candidate_feasible.sum(dim=-1)
    valid = feasible_count > 0

    inf = torch.tensor(torch.inf, dtype=candidate_q.dtype, device=candidate_q.device)
    sample_lower = candidate_q.masked_fill(~candidate_feasible.unsqueeze(-1), inf).amin(dim=-2)
    sample_upper = candidate_q.masked_fill(~candidate_feasible.unsqueeze(-1), -inf).amax(dim=-2)
    lower = torch.maximum(sample_lower, effective_lower)
    upper = torch.minimum(sample_upper, effective_upper)
    nan = torch.tensor(torch.nan, dtype=candidate_q.dtype, device=candidate_q.device)
    lower = torch.where(valid.unsqueeze(-1), lower, nan)
    upper = torch.where(valid.unsqueeze(-1), upper, nan)

    validation_violation = capsule_support(kinematics, validation_q, proxies, directions) - allowed.unsqueeze(-2)
    validation_in_limits = ((validation_q >= effective_lower.unsqueeze(-2)) & (validation_q <= effective_upper.unsqueeze(-2))).all(dim=-1)
    validation_feasible = (validation_violation.amax(dim=-1) <= tolerance) & validation_in_limits
    inside_box = ((validation_q >= lower.unsqueeze(-2)) & (validation_q <= upper.unsqueeze(-2))).all(dim=-1)
    false_exclusion = (validation_feasible & ~inside_box).sum(dim=-1)

    engine = torch.quasirandom.SobolEngine(kinematics.num_dof, scramble=True, seed=box_validation_seed)
    unit = engine.draw(box_validation_samples).to(dtype=candidate_q.dtype, device=candidate_q.device)
    box_q = lower.unsqueeze(-2) + unit.reshape(*(1,) * len(prefix), box_validation_samples, -1) * (upper - lower).unsqueeze(-2)
    box_violation = capsule_support(kinematics, box_q, proxies, directions) - allowed.unsqueeze(-2)
    per_box_sample = box_violation.amax(dim=-1)
    box_bad = (per_box_sample > tolerance) & valid.unsqueeze(-1)
    box_bad_count = box_bad.sum(dim=-1)
    max_box_violation = torch.where(
        valid,
        per_box_sample.amax(dim=-1).clamp_min(0.0),
        nan,
    )
    conservative = bool(valid.all().item()) and int(box_bad_count.sum().item()) == 0
    if not bool(valid.all().item()):
        label = "empty: no feasible candidate samples in one or more batches"
    else:
        label = "conservative on registered box-validation samples" if conservative else "approximate"
    return EnvelopeJointRangeExport(
        lower=lower,
        upper=upper,
        center=0.5 * (lower + upper),
        half_range=0.5 * (upper - lower),
        valid=valid,
        diagnostics=EnvelopeRangeDiagnostics(
            candidate_samples=count,
            candidate_feasible_count=feasible_count,
            validation_samples=validation_count,
            validation_feasible_count=validation_feasible.sum(dim=-1),
            false_exclusion_count=false_exclusion,
            box_validation_samples=box_validation_samples,
            box_envelope_violation_count=box_bad_count,
            box_envelope_violation_rate=box_bad_count.to(candidate_q.dtype) / box_validation_samples,
            max_box_envelope_violation=max_box_violation,
            label=label,
        ),
    )


@dataclass(frozen=True)
class JointRejectionRanges:
    """Per-joint rejected sub-intervals of an exported joint box.

    ``feasible_reference`` is False when no validated feasible reference pose
    could be found (an empty NaN box, or an envelope so tight that neither
    ``reference_q`` nor the exported box center is envelope-feasible). In that
    case ``rejected_intervals`` is a tuple of empty tuples and the summary
    fields are zeroed, so callers can render a clean "not applicable".
    """

    feasible_reference: bool
    reference_source: str
    reference_q: Optional[Tensor]
    rejected_intervals: Tuple[Tuple[Tuple[float, float], ...], ...]
    rejected_joint_count: int
    max_rejected_span_rad: float
    max_rejected_joint_index: int

    def to_evidence_dict(self, joint_names: Sequence[str] = EL4090_JOINT_NAMES) -> dict:
        """Return a JSON-serializable evidence record for the viewer examples."""
        return {
            "feasible_reference": self.feasible_reference,
            "reference_source": self.reference_source,
            "reference_q_rad": (
                self.reference_q.detach().cpu().tolist()
                if self.reference_q is not None else None
            ),
            "per_joint_intervals_rad": [
                [list(interval) for interval in joint]
                for joint in self.rejected_intervals
            ],
            "rejected_joint_count": self.rejected_joint_count,
            "max_rejected_span_rad": self.max_rejected_span_rad,
            "max_rejected_joint_index": self.max_rejected_joint_index,
            "max_rejected_joint_name": (
                joint_names[self.max_rejected_joint_index]
                if self.max_rejected_joint_index >= 0 else None
            ),
        }


def _contiguous_true_runs(mask: Tensor) -> Tuple[Tuple[int, int], ...]:
    """Return inclusive index runs of True values in a one-dimensional mask."""
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if not bool(mask.any().item()):
        return ()
    diff = torch.diff(
        mask.to(torch.int64),
        prepend=torch.zeros(1, dtype=torch.int64, device=mask.device),
        append=torch.zeros(1, dtype=torch.int64, device=mask.device),
    )
    starts = torch.nonzero(diff == 1).flatten()
    ends = torch.nonzero(diff == -1).flatten()
    return tuple((int(start), int(end) - 1) for start, end in zip(starts, ends))


def joint_rejection_ranges(
    kinematics: BatchedUrdfKinematics,
    capsules: Sequence[CapsuleProxy],
    directions: Tensor,
    allowed_support: Tensor,
    lower: Tensor,
    upper: Tensor,
    reference_q: Tensor,
    *,
    tolerance: float = 1e-6,
    steps: int = 101,
) -> JointRejectionRanges:
    """Per-joint rejected sub-intervals of an exported admissible joint box.

    For every joint ``j`` the other joints are pinned at a validated feasible
    reference and ``q[j]`` is swept linearly over ``[lower[j], upper[j]]``.
    A swept pose is rejected when ``capsule_support(...) > allowed_support +
    tolerance`` for at least one support direction, the same feasibility test
    ``export_envelope_joint_ranges`` applies to its candidate samples. The
    returned intervals are the contiguous rejected runs in radians (endpoints
    are half a sweep step interpolated and clamped to the exported range).

    Reference design: ``reference_q`` is preferred (the natural visual anchor;
    for these examples it is the resting pose, inside the exported box whenever
    the export is valid and envelope-feasible whenever the resting pose fits
    the envelope). The exported box center ``0.5 * (lower + upper)`` is the
    fallback. Both are validated for box membership and envelope feasibility:
    the exported box is an axis-aligned over-approximation and its center is
    not guaranteed feasible. When the export is empty (all-NaN box) or neither
    candidate is feasible, ``feasible_reference`` is False and no intervals are
    reported rather than raising.
    """
    if steps < 2:
        raise ValueError("steps must be at least two")
    num_joints = kinematics.num_dof
    lower = lower.reshape(-1).to(dtype=reference_q.dtype, device=reference_q.device)
    upper = upper.reshape(-1).to(dtype=reference_q.dtype, device=reference_q.device)
    allowed = allowed_support.to(dtype=reference_q.dtype, device=reference_q.device)
    if lower.numel() != num_joints or upper.numel() != num_joints or allowed.shape != (directions.shape[0],):
        raise ValueError("lower/upper must have num_dof entries and allowed_support must match the directions")
    empty = tuple(() for _ in range(num_joints))
    if not bool(torch.isfinite(lower).all().item()) or not bool(torch.isfinite(upper).all().item()):
        return JointRejectionRanges(
            feasible_reference=False,
            reference_source="none: exported box is empty (NaN)",
            reference_q=None,
            rejected_intervals=empty,
            rejected_joint_count=0,
            max_rejected_span_rad=0.0,
            max_rejected_joint_index=-1,
        )
    base = reference_q.reshape(-1).to(dtype=reference_q.dtype, device=reference_q.device)
    center = 0.5 * (lower + upper)

    chosen: Optional[Tensor] = None
    source = ""
    for name, candidate in (("reference_q", base), ("box_center", center)):
        inside = bool((candidate >= lower - 1e-5).all().item()) and bool(
            (candidate <= upper + 1e-5).all().item()
        )
        if not inside:
            continue
        support = capsule_support(
            kinematics, candidate.unsqueeze(0), capsules, directions,
        )[0]
        if bool((support <= allowed + tolerance).all().item()):
            chosen = candidate
            source = name
            break
    if chosen is None:
        return JointRejectionRanges(
            feasible_reference=False,
            reference_source="no feasible reference among reference_q and box center",
            reference_q=None,
            rejected_intervals=empty,
            rejected_joint_count=0,
            max_rejected_span_rad=0.0,
            max_rejected_joint_index=-1,
        )

    alpha = torch.linspace(0.0, 1.0, steps, dtype=lower.dtype, device=lower.device)
    sweep_values = lower.unsqueeze(1) + alpha.unsqueeze(0) * (upper - lower).unsqueeze(1)
    sweep_q = chosen.unsqueeze(0).unsqueeze(0).expand(num_joints, steps, num_joints).clone()
    diagonal = torch.arange(num_joints, device=lower.device)
    sweep_q[diagonal, :, diagonal] = sweep_values
    support = capsule_support(kinematics, sweep_q, capsules, directions)
    feasible = (support <= allowed.unsqueeze(0).unsqueeze(0) + tolerance).all(dim=-1)
    rejected = ~feasible

    intervals: list[Tuple[Tuple[float, float], ...]] = []
    for joint in range(num_joints):
        joint_intervals: list[Tuple[float, float]] = []
        step_size = (float(upper[joint]) - float(lower[joint])) / (steps - 1)
        for start, end in _contiguous_true_runs(rejected[joint]):
            low = float(sweep_values[joint, start]) - 0.5 * step_size
            high = float(sweep_values[joint, end]) + 0.5 * step_size
            low = max(float(lower[joint]), low)
            high = min(float(upper[joint]), high)
            joint_intervals.append((low, high))
        intervals.append(tuple(joint_intervals))
    all_intervals = tuple(intervals)

    rejected_joint_count = sum(1 for joint in all_intervals if joint)
    spans = [
        (high - low, joint)
        for joint, joint_intervals in enumerate(all_intervals)
        for low, high in joint_intervals
    ]
    if spans:
        max_span, max_joint = max(spans)
    else:
        max_span, max_joint = 0.0, -1
    return JointRejectionRanges(
        feasible_reference=True,
        reference_source=source,
        reference_q=chosen,
        rejected_intervals=all_intervals,
        rejected_joint_count=rejected_joint_count,
        max_rejected_span_rad=max_span,
        max_rejected_joint_index=max_joint,
    )


def deterministic_joint_samples(
    lower: Tensor,
    upper: Tensor,
    count: int,
    *,
    seed: int = 4090,
) -> Tensor:
    """Deterministic scrambled Sobol samples, including both box corners."""
    if count < 2:
        raise ValueError("count must be at least two")
    engine = torch.quasirandom.SobolEngine(lower.numel(), scramble=True, seed=seed)
    interior = engine.draw(count - 2).to(dtype=lower.dtype, device=lower.device)
    return torch.cat((lower.unsqueeze(0), upper.unsqueeze(0), lower + interior * (upper - lower)), dim=0)


def legacy_condition_from_support(
    occupied_support: Tensor,
    directions: Tensor,
    *,
    morphology_priors: Optional[Tensor] = None,
    clamp_to_training_ranges: bool = True,
) -> Tensor:
    """Project support to the unchanged symmetric EL4090 8-D condition.

    Values are read at cardinal directions and at x=0 via the lateral support.
    This projection is intentionally approximate; it preserves policy I/O only.
    """
    directions = directions.to(dtype=occupied_support.dtype, device=occupied_support.device)
    targets = occupied_support.new_tensor(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    nearest = torch.argmax(targets @ directions.T, dim=-1)
    x_front = occupied_support[..., nearest[0]]
    y_left = occupied_support[..., nearest[1]]
    x_back = -occupied_support[..., nearest[2]]
    y_right = occupied_support[..., nearest[3]]
    width = torch.maximum(y_left, y_right)
    priors = (
        torch.zeros((*occupied_support.shape[:-1], 3), dtype=occupied_support.dtype, device=occupied_support.device)
        if morphology_priors is None
        else morphology_priors
    )
    if priors.shape != (*occupied_support.shape[:-1], 3):
        raise ValueError("morphology_priors must have shape [...,3]")
    condition = torch.cat((
        width.unsqueeze(-1), width.unsqueeze(-1), width.unsqueeze(-1),
        x_front.unsqueeze(-1), x_back.unsqueeze(-1), priors,
    ), dim=-1)
    if clamp_to_training_ranges:
        lower = condition.new_tensor((0.15, 0.15, 0.15, 0.25, -0.60))
        upper = condition.new_tensor((0.60, 0.60, 0.60, 0.60, -0.25))
        condition = torch.cat((condition[..., :5].clamp(lower, upper), condition[..., 5:]), dim=-1)
    return condition


def haa_ranges_from_joint_export(
    export: JointRangeExport | EnvelopeJointRangeExport,
    joint_names: Sequence[str] = EL4090_JOINT_NAMES,
    output_leg_order: Sequence[str] = EL4090_LEG_NAMES,
) -> Tensor:
    indices = [joint_names.index(f"{leg}_HAA") for leg in output_leg_order]
    return torch.stack((export.lower[..., indices], export.upper[..., indices]), dim=-1)


def append_legacy_envelop2_observation(
    proprioception: Tensor,
    condition: Tensor,
    haa_ranges: Tensor,
    gait_phase: Tensor,
) -> Tensor:
    """Assemble the unchanged ``[B,83]`` envelope-v2 observation tail."""
    if proprioception.shape[-1] != 66 or condition.shape[-1] != 8 or haa_ranges.shape[-2:] != (6, 2):
        raise ValueError("Expected proprioception [...,66], condition [...,8], HAA [...,6,2]")
    center = haa_ranges.mean(dim=-1)
    half = 0.5 * (haa_ranges[..., 1] - haa_ranges[..., 0])
    phase = torch.stack((torch.sin(gait_phase), torch.cos(gait_phase)), dim=-1)
    return torch.cat((proprioception, condition[..., 5:8], center, half, phase), dim=-1)
