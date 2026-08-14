#!/usr/bin/env python3
"""Visualize a LiDAR-prescribed point-free envelope on the real EL4090."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from isaacgym import gymapi  # Must precede Torch imports for Isaac Gym Preview 4.
import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ENVELOPE_DIR = PACKAGE_ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))

from gym_envelope_geometry import (  # noqa: E402
    haa_arc_geometry,
    haa_arc_geometry_interval,
    interpolate_joint_ranges,
    polyline_segments,
    support_polygon,
)
from kinematic_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES,
    EL4090_LEG_NAMES,
    BatchedUrdfKinematics,
    capsule_support,
    default_el4090_capsules,
    default_el4090_torso_capsules,
    deterministic_joint_samples,
    export_envelope_joint_ranges,
    haa_ranges_from_joint_export,
    joint_rejection_ranges,
    load_urdf_joints,
    reachable_foot_support,
    support_directions,
)
from lidar_free_envelope import (  # noqa: E402
    assigned_point_clearances,
    backtrack_to_feasible_anchor,
    envelope_excess,
    generate_synthetic_lidar_cloud,
    maximum_sector_point_free_envelope,
    polygon_support_excess,
)


GRAPHITE = (0.10, 0.12, 0.14)
GROUND = (0.34, 0.36, 0.38)
WHITE = (1.00, 1.00, 1.00)
LIGHT_CYAN = (0.08, 0.94, 1.00)
DARK_TEAL = (0.00, 0.72, 0.60)
AMBER = (1.00, 0.64, 0.06)
REACHABLE_BLUE = (0.16, 0.42, 1.00)
RED = (1.00, 0.10, 0.06)
REJECTION_RED = (1.00, 0.25, 0.65)
BASE_HEIGHT = 0.58
TOLERANCE = 1e-6


@dataclass(frozen=True)
class LidarProblem:
    seed: int
    baseline_q: torch.Tensor
    baseline_support: torch.Tensor
    cloud: object
    free_envelope: object
    range_export: object
    haa_ranges: torch.Tensor
    reference_reachable_support: torch.Tensor
    candidate_lower: torch.Tensor
    candidate_upper: torch.Tensor
    joint_shrinkage: torch.Tensor
    candidate_reduction_fraction: float
    required_candidate_reduction_fraction: float
    required_joint_shrink_rad: float
    rejection_ranges: object = None


@dataclass(frozen=True)
class MotionTrajectory:
    proposed_q: torch.Tensor
    joint_positions: torch.Tensor
    naive_excess_m: torch.Tensor
    envelope_excess_m: torch.Tensor
    joint_excess_rad: torch.Tensor
    accepted_scale: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--point_count", type=int, default=20)
    parser.add_argument("--directions", type=int, default=48)
    parser.add_argument("--min_radius", type=float, default=0.0)
    parser.add_argument("--max_radius", type=float, default=2.10)
    parser.add_argument("--robot_clearance", type=float, default=0.05)
    parser.add_argument("--lateral_robot_clearance", type=float, default=0.025)
    parser.add_argument("--lateral_anchors_per_side", type=int, default=5)
    parser.add_argument("--point_clearance", type=float, default=0.02)
    parser.add_argument("--reference_containment_margin", type=float, default=0.005)
    parser.add_argument(
        "--near_band_fraction", type=float, default=0.05,
        help="maximum feasible-annulus fraction for sparse primary returns",
    )
    parser.add_argument("--min_candidate_reduction_fraction", type=float, default=0.05)
    parser.add_argument("--min_joint_shrink_rad", type=float, default=0.03)
    parser.add_argument("--motion_period_steps", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps the viewer interactive")
    parser.add_argument(
        "--show_rejection", action="store_true",
        help="show rejected sub-intervals of the exported joint ranges",
    )
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--no_motion", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot_step", type=int, default=5)
    return parser.parse_args()


def baseline_joint_pose() -> torch.Tensor:
    return torch.tensor([0.0, 0.60, -0.60] * 6, dtype=torch.float32)


def build_problem(args, kinematics, directions, seed: int) -> LidarProblem:
    capsules = default_el4090_capsules()
    baseline_q = baseline_joint_pose()
    baseline_support = capsule_support(
        kinematics, baseline_q.unsqueeze(0), capsules, directions,
    )[0]
    # Candidate reach spans the full URDF mechanical range (±3 rad for every
    # joint) per MasterYip's request that the exported ranges genuinely reach
    # the URDF limits instead of being capped by the reference reach.
    effective_lower, effective_upper = kinematics.joint_limits(soft_fraction=1.0)
    half_width = (effective_upper - effective_lower) / 2
    candidate_lower = torch.maximum(baseline_q - half_width, effective_lower)
    candidate_upper = torch.minimum(baseline_q + half_width, effective_upper)
    candidate_q = torch.cat((
        baseline_q.unsqueeze(0),
        deterministic_joint_samples(candidate_lower, candidate_upper, 768, seed=seed + 100),
    ))
    validation_q = deterministic_joint_samples(
        candidate_lower, candidate_upper, 257, seed=seed + 200,
    )
    reference_reachable_support = reachable_foot_support(
        kinematics, candidate_q.unsqueeze(0), directions,
    )[0]
    cloud = generate_synthetic_lidar_cloud(
        directions,
        baseline_support,
        reference_reachable_support,
        count=args.point_count,
        seed=seed,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        robot_clearance=args.robot_clearance,
        lateral_robot_clearance=args.lateral_robot_clearance,
        lateral_anchors_per_side=args.lateral_anchors_per_side,
        reference_containment_margin=args.reference_containment_margin,
        near_band_fraction=args.near_band_fraction,
    )
    reference_excess = polygon_support_excess(
        cloud.points_xy, directions, reference_reachable_support,
    )
    if float(reference_excess.max()) > -args.reference_containment_margin + 5e-6:
        raise RuntimeError("LiDAR return escaped the eroded reference reachable envelope")
    free_envelope = maximum_sector_point_free_envelope(
        cloud,
        directions,
        point_clearance=args.point_clearance,
        cap_support=reference_reachable_support,
    )
    if float((baseline_support - free_envelope.support_m).max()) > TOLERANCE:
        raise RuntimeError("LiDAR envelope does not contain the feasible anchor")
    if float((free_envelope.support_m - reference_reachable_support).max()) > TOLERANCE:
        raise RuntimeError("prescribed envelope exceeds pre-obstacle reachable reference")

    export = export_envelope_joint_ranges(
        kinematics,
        candidate_q,
        validation_q,
        directions,
        free_envelope.support_m,
        effective_lower,
        effective_upper,
        capsules=capsules,
        box_validation_samples=256,
        box_validation_seed=seed + 300,
    )
    if not bool(export.valid):
        raise RuntimeError("LiDAR envelope produced no feasible exported joint range")
    feasible = (
        capsule_support(kinematics, candidate_q, capsules, directions)
        <= free_envelope.support_m.unsqueeze(0) + TOLERANCE
    ).all(dim=-1)
    feasible_count = int(feasible.sum())
    candidate_reduction_fraction = 1.0 - feasible_count / candidate_q.shape[0]
    joint_shrinkage = (candidate_upper - candidate_lower) - (
        export.upper - export.lower
    )
    if candidate_reduction_fraction < args.min_candidate_reduction_fraction:
        raise RuntimeError(
            "LiDAR constraints are not material: candidate reduction "
            f"{candidate_reduction_fraction:.6f} is below "
            f"{args.min_candidate_reduction_fraction:.6f}"
        )
    if float(joint_shrinkage.max()) < args.min_joint_shrink_rad:
        raise RuntimeError(
            "LiDAR constraints are not material: maximum joint shrinkage "
            f"{float(joint_shrinkage.max()):.6f} rad is below "
            f"{args.min_joint_shrink_rad:.6f} rad"
        )
    return LidarProblem(
        seed=seed,
        baseline_q=baseline_q,
        baseline_support=baseline_support,
        cloud=cloud,
        free_envelope=free_envelope,
        range_export=export,
        haa_ranges=haa_ranges_from_joint_export(export),
        reference_reachable_support=reference_reachable_support,
        candidate_lower=candidate_lower,
        candidate_upper=candidate_upper,
        joint_shrinkage=joint_shrinkage,
        candidate_reduction_fraction=candidate_reduction_fraction,
        required_candidate_reduction_fraction=args.min_candidate_reduction_fraction,
        required_joint_shrink_rad=args.min_joint_shrink_rad,
        rejection_ranges=joint_rejection_ranges(
            kinematics, capsules, directions, free_envelope.support_m,
            export.lower, export.upper, baseline_q, tolerance=TOLERANCE,
        ),
    )


def motion_offsets(reference: torch.Tensor) -> torch.Tensor:
    leg_offsets = (0.00, 0.50, 0.25, 0.75, 0.50, 0.00)
    joint_offsets = (0.00, 0.18, 0.68)
    return torch.tensor(
        [leg + joint for leg in leg_offsets for joint in joint_offsets],
        dtype=reference.dtype,
        device=reference.device,
    )


def build_motion_trajectory(problem, kinematics, directions, period_steps):
    """Build one smooth closed motion using a trajectory-wide feasible scale."""
    if period_steps < 2:
        raise ValueError("period_steps must be at least 2")
    proposed = torch.stack([
        interpolate_joint_ranges(
            problem.range_export.lower,
            problem.range_export.upper,
            step / period_steps,
            phase_offsets=motion_offsets(problem.baseline_q),
        )
        for step in range(period_steps)
    ])
    naive_excess = envelope_excess(
        kinematics,
        proposed,
        default_el4090_capsules(),
        directions,
        problem.free_envelope.support_m,
    )
    individually_accepted = backtrack_to_feasible_anchor(
        kinematics,
        proposed,
        problem.baseline_q,
        problem.range_export.lower,
        problem.range_export.upper,
        default_el4090_capsules(),
        directions,
        problem.free_envelope.support_m,
        tolerance=TOLERANCE,
    )
    global_scale = individually_accepted.accepted_scale.min()
    anchors = problem.baseline_q.unsqueeze(0)
    for _ in range(24):
        accepted_q = anchors + global_scale * (proposed - anchors)
        accepted_excess = envelope_excess(
            kinematics,
            accepted_q,
            default_el4090_capsules(),
            directions,
            problem.free_envelope.support_m,
        )
        joint_excess = torch.maximum(
            problem.range_export.lower.unsqueeze(0) - accepted_q,
            accepted_q - problem.range_export.upper.unsqueeze(0),
        ).clamp_min(0.0).amax(dim=-1)
        if bool((accepted_excess <= TOLERANCE).all()) and bool(
            (joint_excess <= TOLERANCE).all()
        ):
            break
        global_scale = global_scale * 0.5
    else:
        raise RuntimeError("failed to construct a continuous feasible trajectory")
    scale = global_scale.expand(period_steps).clone()
    return MotionTrajectory(
        proposed_q=proposed,
        joint_positions=accepted_q,
        naive_excess_m=naive_excess,
        envelope_excess_m=accepted_excess,
        joint_excess_rad=joint_excess,
        accepted_scale=scale,
    )


def accepted_motion_pose(trajectory, motion_step):
    index = motion_step % trajectory.joint_positions.shape[0]
    frame = MotionTrajectory(
        proposed_q=trajectory.proposed_q[index:index + 1],
        joint_positions=trajectory.joint_positions[index:index + 1],
        naive_excess_m=trajectory.naive_excess_m[index:index + 1],
        envelope_excess_m=trajectory.envelope_excess_m[index:index + 1],
        joint_excess_rad=trajectory.joint_excess_rad[index:index + 1],
        accepted_scale=trajectory.accepted_scale[index:index + 1],
    )
    return (
        frame.proposed_q[0], frame.joint_positions[0],
        frame.naive_excess_m[0], frame,
    )


def new_stats(trajectory=None) -> dict:
    stats = {
        "frame_count": 0,
        "joint_sample_count": 0,
        "joint_range_violation_count": 0,
        "max_joint_bound_excess_rad": 0.0,
        "envelope_violation_count": 0,
        "max_occupied_support_excess_m": 0.0,
        "naive_envelope_violation_count": 0,
        "max_naive_support_excess_m": 0.0,
        "backtracked_frame_count": 0,
        "minimum_accepted_scale": 1.0,
        "observed_min_rad": np.full(18, np.inf, dtype=np.float64),
        "observed_max_rad": np.full(18, -np.inf, dtype=np.float64),
    }
    if trajectory is not None:
        cyclic_q = torch.cat((
            trajectory.joint_positions, trajectory.joint_positions[:1],
        ))
        stats["trajectory_fixed_scale"] = float(trajectory.accepted_scale[0])
        stats["maximum_cyclic_joint_step_rad"] = float(
            torch.diff(cyclic_q, dim=0).abs().max()
        )
    return stats


def update_stats(stats, problem, pose, naive_excess, accepted) -> None:
    lower = problem.range_export.lower
    upper = problem.range_export.upper
    joint_excess = torch.maximum(lower - pose, pose - upper).clamp_min(0.0)
    occupied_excess = float(accepted.envelope_excess_m[0])
    joint_max = float(joint_excess.max())
    stats["frame_count"] += 1
    stats["joint_sample_count"] += pose.numel()
    stats["joint_range_violation_count"] += int((joint_excess > TOLERANCE).sum())
    stats["max_joint_bound_excess_rad"] = max(stats["max_joint_bound_excess_rad"], joint_max)
    stats["envelope_violation_count"] += int(occupied_excess > TOLERANCE)
    stats["max_occupied_support_excess_m"] = max(
        stats["max_occupied_support_excess_m"], max(0.0, occupied_excess),
    )
    stats["naive_envelope_violation_count"] += int(float(naive_excess) > TOLERANCE)
    stats["max_naive_support_excess_m"] = max(
        stats["max_naive_support_excess_m"], max(0.0, float(naive_excess)),
    )
    scale = float(accepted.accepted_scale[0])
    stats["backtracked_frame_count"] += int(scale < 1.0)
    stats["minimum_accepted_scale"] = min(stats["minimum_accepted_scale"], scale)
    values = pose.detach().cpu().numpy().astype(np.float64)
    stats["observed_min_rad"] = np.minimum(stats["observed_min_rad"], values)
    stats["observed_max_rad"] = np.maximum(stats["observed_max_rad"], values)
    if joint_max > TOLERANCE or occupied_excess > TOLERANCE:
        raise RuntimeError("accepted animation pose violated its joint or envelope constraint")


def compact_stats(stats) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats.items()
    }


def print_problem(problem, directions) -> None:
    clearance = assigned_point_clearances(
        problem.cloud, directions, problem.free_envelope.support_m,
    )
    baseline_clearance = (
        problem.cloud.points_xy * directions[problem.cloud.sector_indices]
    ).sum(dim=-1) - problem.baseline_support[problem.cloud.sector_indices]
    diagnostics = problem.range_export.diagnostics
    reference_excess = polygon_support_excess(
        problem.cloud.points_xy, directions, problem.reference_reachable_support,
    )
    print("\nLiDAR free-envelope definition")
    print(f"  seed: {problem.seed}")
    print(
        f"  returns: {problem.cloud.points_xy.shape[0]}; "
        f"{problem.free_envelope.constrained_face_indices.numel()} constrained "
        f"and {problem.free_envelope.unconstrained_face_indices.numel()} "
        f"reference-capped faces"
    )
    cluster_text = ", ".join(
        f"{float(value):.2f}" for value in problem.cloud.near_cluster_centers_rad
    )
    gap_text = ", ".join(
        f"{float(value):.2f}" for value in problem.cloud.far_gap_centers_rad
    )
    print(
        "  structure: sparse random face assignment; "
        f"near clusters at {cluster_text} rad; far gaps at {gap_text} rad"
    )
    print(
        "  proximity: returns use at most "
        f"{100.0 * problem.cloud.near_band_fraction:.1f}% of the feasible radial "
        f"annulus; lateral anchors {problem.cloud.lateral_anchor_sectors.tolist()}"
    )
    clearance_surplus = baseline_clearance - problem.cloud.required_clearance_m
    print(
        "  baseline capsule-envelope clearance: "
        f"{float(baseline_clearance.min()):.6f} m minimum; requirements "
        f"{problem.cloud.lateral_robot_clearance_m:.3f} m lateral / "
        f"{problem.cloud.robot_clearance_m:.3f} m other; "
        f"{float(clearance_surplus.min()):.6f} m minimum surplus"
    )
    print(f"  prescribed point clearance: {float(clearance.min()):.6f} m minimum")
    print(
        "  reference reachable containment: "
        f"{-float(reference_excess.max()):.6f} m minimum inward margin"
    )
    print(f"  optimality: {problem.free_envelope.optimality_scope}")
    print(f"  exported candidates: {int(diagnostics.candidate_feasible_count)}/{diagnostics.candidate_samples} feasible")
    print(f"  candidate reduction: {100.0 * problem.candidate_reduction_fraction:.2f}%")
    print(f"  maximum joint-interval shrinkage: {float(problem.joint_shrinkage.max()):.6f} rad")
    print("  colors: white returns inside blue pre-obstacle reachable reference; light cyan prescribed; dark teal occupied; amber HAA; magenta rejected; red = torso occupied envelope exceeds the declared envelope")
    if problem.rejection_ranges is not None:
        print_rejection(problem.rejection_ranges)


def print_rejection(rejection) -> None:
    if rejection is None:
        return
    print(f"\nPer-joint rejected sub-intervals (reference: {rejection.reference_source})")
    if not rejection.feasible_reference:
        print(f"  {rejection.reference_source}")
        return
    print(
        f"  rejected joints: {rejection.rejected_joint_count}; "
        f"max span {rejection.max_rejected_span_rad:.4f} rad at "
        f"{EL4090_JOINT_NAMES[rejection.max_rejected_joint_index]}"
    )
    for joint, intervals in enumerate(rejection.rejected_intervals):
        if intervals:
            rendered = ", ".join(f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in intervals)
            print(f"  {EL4090_JOINT_NAMES[joint]:8s} {rendered}")


def print_exported_box(problem) -> None:
    lower = problem.range_export.lower.detach().cpu().numpy()
    upper = problem.range_export.upper.detach().cpu().numpy()
    print("\nExported joint box [rad]")
    for joint, name in enumerate(EL4090_JOINT_NAMES):
        print(f"  {name:8s} [{lower[joint]:+.4f}, {upper[joint]:+.4f}]")


def print_controls() -> None:
    rows = (
        ("G", "regenerate with seed + 1"),
        ("M", "pause or resume feasible motion"),
        ("X", "reset deterministic motion phase"),
        ("L", "toggle white LiDAR returns and clearance spokes"),
        ("P", "toggle light-cyan prescribed free envelope"),
        ("O", "toggle dark-teal current occupied envelope"),
        ("H", "toggle amber HAA ranges"),
        ("J", "toggle magenta rejected HAA sub-intervals"),
        ("R", "toggle blue pre-obstacle reachable reference"),
        ("C", "cycle overview and top cameras"),
        ("S", "capture screenshot and JSON evidence"),
        ("Esc", "exit"),
    )
    width = max(len(key) for key, _ in rows)
    print("\nViewer controls")
    for key, action in rows:
        print(f"  {key:<{width}}  {action}")


def add_segments(gym, viewer, env, segments, color) -> None:
    vertices = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
    count = vertices.shape[0] // 2
    colors = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
    gym.add_lines(viewer, env, count, vertices, colors)


def add_bold_segments(gym, viewer, env, segments, color, *, z_offset=0.008) -> None:
    """Match the comparison viewer's two vertically offset line strokes."""
    vertices = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
    add_segments(gym, viewer, env, vertices, color)
    raised = vertices.copy()
    raised[:, 2] += z_offset
    add_segments(gym, viewer, env, raised, color)


def draw_boundary(gym, viewer, env, polygon, height, color) -> None:
    points = np.column_stack((polygon, np.full(len(polygon), height)))
    add_bold_segments(
        gym, viewer, env, polyline_segments(points, closed=True), color,
    )


def draw_cloud(gym, viewer, env, problem) -> None:
    points = problem.cloud.points_xy.detach().cpu().numpy()
    size = 0.040
    z = 0.120
    crosses = []
    for x, y in points:
        crosses.extend((
            (x - size, y, z), (x + size, y, z),
            (x, y - size, z), (x, y + size, z),
            (x - 0.72 * size, y - 0.72 * size, z),
            (x + 0.72 * size, y + 0.72 * size, z),
            (x - 0.72 * size, y + 0.72 * size, z),
            (x + 0.72 * size, y - 0.72 * size, z),
        ))
    crosses = np.asarray(crosses)
    add_bold_segments(gym, viewer, env, crosses, WHITE)
    limiting = problem.free_envelope.limiting_points_xy.detach().cpu().numpy()
    feet = problem.free_envelope.clearance_feet_xy.detach().cpu().numpy()
    spokes = np.stack((
        np.column_stack((limiting, np.full(len(limiting), z))),
        np.column_stack((feet, np.full(len(feet), z))),
    ), axis=1).reshape(-1, 3)
    add_bold_segments(gym, viewer, env, spokes, LIGHT_CYAN)


def draw_haa(gym, viewer, env, kinematics, problem, pose, rejection=None) -> None:
    origins, arcs, markers = haa_arc_geometry(
        kinematics, pose, problem.haa_ranges, radius=0.25, samples=41,
    )
    translation = np.array((0.0, 0.0, BASE_HEIGHT + 0.12), dtype=np.float32)
    origins = origins.detach().cpu().numpy() + translation
    arcs = arcs.detach().cpu().numpy() + translation
    markers = markers.detach().cpu().numpy() + translation
    haa_indices = [EL4090_JOINT_NAMES.index(f"{leg}_HAA") for leg in EL4090_LEG_NAMES]
    for index in range(6):
        add_bold_segments(
            gym, viewer, env, polyline_segments(arcs[index]), AMBER,
        )
        bounds = np.stack((origins[index], arcs[index, 0], origins[index], arcs[index, -1]))
        add_bold_segments(gym, viewer, env, bounds, AMBER)
        endpoint = origins[index] + 1.28 * (markers[index] - origins[index])
        add_bold_segments(
            gym, viewer, env, np.stack((origins[index], endpoint)), AMBER,
        )
        if rejection is not None and rejection.feasible_reference:
            for lo_v, hi_v in rejection.rejected_intervals[haa_indices[index]]:
                sub = haa_arc_geometry_interval(
                    kinematics, pose, problem.haa_ranges, index, lo_v, hi_v,
                    radius=0.25, samples=17,
                )
                sub = sub.detach().cpu().numpy() + translation
                # Elevate the rejected band so it renders above the amber arc
                # instead of depth-fighting at the same z.
                elevated = sub.copy()
                elevated[:, 2] += 0.02
                add_bold_segments(
                    gym, viewer, env, polyline_segments(elevated), REJECTION_RED,
                )


def draw_scene(gym, viewer, env, kinematics, directions, problem, pose, state) -> None:
    gym.clear_lines(viewer)
    if state["lidar"]:
        draw_cloud(gym, viewer, env, problem)
    if state["prescribed"]:
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, problem.free_envelope.support_m),
            0.080, LIGHT_CYAN,
        )
    if state["occupied"]:
        # Occupied envelope drawn from the torso shape only, so the red/teal
        # polygon reflects the body rather than the full leg reach. The color
        # follows the torso's OWN envelope excess (not the full-robot flag):
        # a torso that fits inside the declared envelope stays teal even when
        # the legs poke out.
        occupied = capsule_support(
            kinematics, pose.unsqueeze(0), default_el4090_torso_capsules(), directions,
        )[0]
        torso_violation = float((occupied - problem.free_envelope.support_m).max()) > TOLERANCE
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, occupied),
            0.112, RED if torso_violation else DARK_TEAL,
        )
    if state["haa"]:
        rejection = problem.rejection_ranges if state.get("rejection") else None
        draw_haa(gym, viewer, env, kinematics, problem, pose, rejection=rejection)
    if state["reachable"]:
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, problem.reference_reachable_support),
            0.048, REACHABLE_BLUE,
        )


def set_camera(gym, viewer, mode: int) -> None:
    if mode == 0:
        position, target = (3.60, -2.80, 3.80), (0.0, 0.0, 0.24)
    else:
        position, target = (0.04, -0.04, 3.80), (0.0, 0.0, 0.10)
    gym.viewer_camera_look_at(
        viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target),
    )


def create_simulation(args, initial_q):
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, params,
    )
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create the simulator")

    ground_options = gymapi.AssetOptions()
    ground_options.fix_base_link = True
    ground_asset = gym.create_box(sim, 5.0, 5.0, 0.04, ground_options)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_root = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf"
    robot_asset = gym.load_asset(sim, str(asset_root), "el_4090.urdf", asset_options)
    if robot_asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError(f"Failed to load EL4090 asset from {asset_root}")
    asset_names = tuple(gym.get_asset_dof_names(robot_asset))
    if set(asset_names) != set(EL4090_JOINT_NAMES):
        gym.destroy_sim(sim)
        raise RuntimeError(f"Unexpected EL4090 DOF names: {asset_names}")
    q_indices = tuple(EL4090_JOINT_NAMES.index(name) for name in asset_names)

    env = gym.create_env(
        sim, gymapi.Vec3(-3.0, -3.0, 0.0), gymapi.Vec3(3.0, 3.0, 3.0), 1,
    )
    ground_pose = gymapi.Transform()
    ground_pose.p = gymapi.Vec3(0.0, 0.0, -0.03)
    ground = gym.create_actor(env, ground_asset, ground_pose, "graphite_ground", 0, 0)
    gym.set_rigid_body_color(env, ground, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*GROUND))
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, BASE_HEIGHT)
    actor = gym.create_actor(env, robot_asset, pose, "el4090_lidar_envelope", 1, 0)
    for body_index in range(gym.get_asset_rigid_body_count(robot_asset)):
        gym.set_rigid_body_color(
            env, actor, body_index, gymapi.MESH_VISUAL, gymapi.Vec3(*GRAPHITE),
        )
    gym.prepare_sim(sim)
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(0.92, 0.92, 0.92),
        gymapi.Vec3(0.36, 0.36, 0.36),
        gymapi.Vec3(-0.5, -0.4, -1.0),
    )

    camera = gymapi.CameraProperties()
    camera.width = 1600
    camera.height = 900
    camera.horizontal_fov = 58.0
    viewer = gym.create_viewer(sim, camera)
    if viewer is None:
        gym.destroy_sim(sim)
        raise RuntimeError("Isaac Gym failed to create the viewer; check DISPLAY")
    bindings = (
        (gymapi.KEY_ESCAPE, "quit"),
        (gymapi.KEY_G, "regenerate"),
        (gymapi.KEY_M, "motion"),
        (gymapi.KEY_X, "reset"),
        (gymapi.KEY_L, "lidar"),
        (gymapi.KEY_P, "prescribed"),
        (gymapi.KEY_O, "occupied"),
        (gymapi.KEY_H, "haa"),
        (gymapi.KEY_J, "rejection"),
        (gymapi.KEY_R, "reachable"),
        (gymapi.KEY_C, "camera"),
        (gymapi.KEY_S, "capture"),
    )
    for key, action in bindings:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)
    return gym, sim, viewer, env, actor, q_indices


def apply_pose(gym, env, actor, q_indices, pose) -> None:
    states = np.zeros(len(q_indices), dtype=gymapi.DofState.dtype)
    states["pos"] = pose.detach().cpu().numpy()[list(q_indices)]
    gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)


def write_evidence(gym, viewer, path, problem, directions, state, stats, step) -> None:
    path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("generated evidence must be outside extended_legged_gym")
    path.parent.mkdir(parents=True, exist_ok=True)
    gym.write_viewer_image_to_file(viewer, str(path))
    clearances = assigned_point_clearances(
        problem.cloud, directions, problem.free_envelope.support_m,
    )
    baseline_clearances = (
        problem.cloud.points_xy * directions[problem.cloud.sector_indices]
    ).sum(dim=-1) - problem.baseline_support[problem.cloud.sector_indices]
    baseline_polygon_excess = polygon_support_excess(
        problem.cloud.points_xy, directions, problem.baseline_support,
    )
    reference_excess = polygon_support_excess(
        problem.cloud.points_xy, directions, problem.reference_reachable_support,
    )
    polygon = support_polygon(directions, problem.free_envelope.support_m)
    evidence = {
        "screenshot": str(path),
        "step": step,
        "cloud": {
            "seed": problem.seed,
            "count": int(problem.cloud.points_xy.shape[0]),
            "angular_coverage": (
                "sparse nearest-sector assignment; faces without returns retain "
                "the pre-obstacle reachable support cap"
            ),
            "structure": {
                "randomization": (
                    "seeded uneven sector density, randomized cluster/gap "
                    "centers with 0.60 rad circular separation, wide angular "
                    "jitter, and a configurable near-inner radial band"
                ),
                "near_cluster_centers_rad": (
                    problem.cloud.near_cluster_centers_rad.detach().cpu().tolist()
                ),
                "far_gap_centers_rad": (
                    problem.cloud.far_gap_centers_rad.detach().cpu().tolist()
                ),
                "sector_count_bounds": [
                    int(problem.cloud.sector_counts.min()),
                    int(problem.cloud.sector_counts.max()),
                ],
                "lateral_anchor_sectors": (
                    problem.cloud.lateral_anchor_sectors.detach().cpu().tolist()
                ),
                "near_band_fraction": problem.cloud.near_band_fraction,
                "lateral_anchors_per_side": int(
                    problem.cloud.lateral_anchor_sectors.numel() // 2
                ),
            },
            "radius_bounds_m": [problem.cloud.min_radius_m, problem.cloud.max_radius_m],
            "observed_radius_bounds_m": [
                float(problem.cloud.radii_m.min()), float(problem.cloud.radii_m.max()),
            ],
            "required_baseline_clearance_m": problem.cloud.robot_clearance_m,
            "required_lateral_baseline_clearance_m": (
                problem.cloud.lateral_robot_clearance_m
            ),
            "per_return_required_baseline_clearance_m": (
                problem.cloud.required_clearance_m.detach().cpu().tolist()
            ),
            "minimum_baseline_clearance_m": float(baseline_clearances.min()),
            "minimum_baseline_clearance_surplus_m": float(
                (baseline_clearances - problem.cloud.required_clearance_m).min()
            ),
            "minimum_baseline_polygon_outside_excess_m": float(
                baseline_polygon_excess.min()
            ),
            "all_points_outside_baseline_occupied_envelope": bool(
                (baseline_polygon_excess > 0.0).all()
            ),
            "required_point_clearance_m": problem.free_envelope.point_clearance_m,
            "minimum_point_clearance_m": float(clearances.min()),
            "reference_containment_margin_m": problem.cloud.reference_containment_margin_m,
            "minimum_reference_inward_margin_m": -float(reference_excess.max()),
            "all_points_inside_reference_reachable_envelope": bool(
                (reference_excess <= TOLERANCE).all()
            ),
            "ray_inner_radius_bounds_m": [
                float(problem.cloud.ray_inner_radius_m.min()),
                float(problem.cloud.ray_inner_radius_m.max()),
            ],
            "ray_outer_radius_bounds_m": [
                float(problem.cloud.ray_outer_radius_m.min()),
                float(problem.cloud.ray_outer_radius_m.max()),
            ],
        },
        "optimality_scope": problem.free_envelope.optimality_scope,
        "directions_xy": directions.detach().cpu().tolist(),
        "prescribed_support_m": problem.free_envelope.support_m.detach().cpu().tolist(),
        "prescribed_vertices_xy": polygon.tolist(),
        "reference_reachable_support_m": problem.reference_reachable_support.detach().cpu().tolist(),
        "reference_reachable_vertices_xy": support_polygon(
            directions, problem.reference_reachable_support,
        ).tolist(),
        "prescribed_inside_reference_max_support_excess_m": float(
            (problem.free_envelope.support_m - problem.reference_reachable_support).max()
        ),
        "limiting_point_indices": problem.free_envelope.limiting_point_indices.detach().cpu().tolist(),
        "constrained_face_indices": (
            problem.free_envelope.constrained_face_indices.detach().cpu().tolist()
        ),
        "unconstrained_face_indices": (
            problem.free_envelope.unconstrained_face_indices.detach().cpu().tolist()
        ),
        "constrained_face_count": int(
            problem.free_envelope.constrained_face_indices.numel()
        ),
        "unconstrained_face_count": int(
            problem.free_envelope.unconstrained_face_indices.numel()
        ),
        "joint_order": list(EL4090_JOINT_NAMES),
        "exported_joint_lower_rad": problem.range_export.lower.detach().cpu().tolist(),
        "exported_joint_upper_rad": problem.range_export.upper.detach().cpu().tolist(),
        "range_impact": {
            "unconstrained_candidate_count": problem.range_export.diagnostics.candidate_samples,
            "constrained_feasible_count": int(
                problem.range_export.diagnostics.candidate_feasible_count
            ),
            "candidate_reduction_fraction": problem.candidate_reduction_fraction,
            "required_candidate_reduction_fraction": (
                problem.required_candidate_reduction_fraction
            ),
            "unconstrained_candidate_lower_rad": problem.candidate_lower.detach().cpu().tolist(),
            "unconstrained_candidate_upper_rad": problem.candidate_upper.detach().cpu().tolist(),
            "per_joint_interval_shrinkage_rad": problem.joint_shrinkage.detach().cpu().tolist(),
            "maximum_joint_interval_shrinkage_rad": float(problem.joint_shrinkage.max()),
            "required_joint_interval_shrinkage_rad": problem.required_joint_shrink_rad,
        },
        "motion_summary": compact_stats(stats),
        "visible_layers": {key: state[key] for key in ("lidar", "prescribed", "occupied", "haa", "rejection", "reachable")},
        "rejection_ranges": (
            problem.rejection_ranges.to_evidence_dict()
            if problem.rejection_ranges is not None else None
        ),
        "semantic_mapping": {
            "white": "LiDAR returns",
            "light_cyan": "prescribed point-free envelope and active clearance spokes",
            "dark_teal": "current occupied capsule envelope",
            "amber": (
                "exported HAA intervals and current markers directed in body XY "
                "from each URDF hip origin toward its physical FOOT link"
            ),
            "magenta": "rejected sub-intervals of the exported HAA ranges",
            "blue": "pre-obstacle unconstrained reachable-foot reference",
            "red": "torso occupied envelope exceeds the declared envelope (full-robot excess tracked separately in stats)",
        },
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Screenshot: {path}")
    print(f"Evidence:   {json_path}")


def validate_args(args) -> None:
    if args.directions < 8:
        raise ValueError("--directions must be at least 8")
    if args.point_count < 1:
        raise ValueError("--point_count must be positive")
    if args.motion_period_steps < 2:
        raise ValueError("--motion_period_steps must be at least 2")
    if args.point_clearance >= args.robot_clearance:
        raise ValueError("--point_clearance must be smaller than --robot_clearance")
    if not args.point_clearance < args.lateral_robot_clearance <= args.robot_clearance:
        raise ValueError(
            "--lateral_robot_clearance must be greater than --point_clearance "
            "and no greater than --robot_clearance"
        )
    if args.lateral_anchors_per_side < 1:
        raise ValueError("--lateral_anchors_per_side must be positive")
    if args.reference_containment_margin <= 0.0:
        raise ValueError("--reference_containment_margin must be positive")
    if not 0.0 < args.near_band_fraction <= 1.0:
        raise ValueError("--near_band_fraction must be in (0,1]")
    if not 0.0 < args.min_candidate_reduction_fraction < 1.0:
        raise ValueError("--min_candidate_reduction_fraction must be in (0,1)")
    if args.min_joint_shrink_rad <= 0.0:
        raise ValueError("--min_joint_shrink_rad must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    urdf = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    directions = support_directions(args.directions)
    problem = build_problem(args, kinematics, directions, args.seed)
    print_problem(problem, directions)

    trajectory = build_motion_trajectory(
        problem, kinematics, directions, args.motion_period_steps,
    )
    cyclic_q = torch.cat((trajectory.joint_positions, trajectory.joint_positions[:1]))
    maximum_step = float(torch.diff(cyclic_q, dim=0).abs().max())
    print(
        "  continuous motion: fixed feasibility scale "
        f"{float(trajectory.accepted_scale[0]):.6f}; "
        f"maximum cyclic joint step {maximum_step:.6f} rad"
    )

    if args.compute_only:
        print_exported_box(problem)
        print(
            f"Compute-only motion: {trajectory.proposed_q.shape[0]} frames; "
            f"{int((trajectory.naive_excess_m > TOLERANCE).sum())} naive violations; "
            f"{int((trajectory.envelope_excess_m > TOLERANCE).sum())} accepted violations; "
            f"fixed scale {float(trajectory.accepted_scale[0]):.6f}; "
            f"maximum cyclic joint step {maximum_step:.6f} rad"
        )
        return

    print_controls()
    gym, sim, viewer, env, actor, q_indices = create_simulation(args, problem.baseline_q)
    state = {
        "motion": not args.no_motion,
        "motion_step": 0,
        "lidar": True,
        "prescribed": True,
        "occupied": True,
        "haa": True,
        "rejection": args.show_rejection,
        "reachable": True,
        "camera": 0,
    }
    set_camera(gym, viewer, state["camera"])
    stats = new_stats(trajectory)
    step = 0
    captured = False
    running = True
    try:
        while running and not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    running = False
                elif event.action == "regenerate":
                    problem = build_problem(args, kinematics, directions, problem.seed + 1)
                    trajectory = build_motion_trajectory(
                        problem, kinematics, directions, args.motion_period_steps,
                    )
                    state["motion_step"] = 0
                    stats = new_stats(trajectory)
                    print_problem(problem, directions)
                elif event.action == "motion":
                    state["motion"] = not state["motion"]
                    print(f"Feasible motion: {state['motion']}")
                elif event.action == "reset":
                    state["motion_step"] = 0
                    print("Motion phase reset")
                elif event.action in ("lidar", "prescribed", "occupied", "haa", "rejection", "reachable"):
                    state[event.action] = not state[event.action]
                    print(f"{event.action} visible: {state[event.action]}")
                elif event.action == "camera":
                    state["camera"] = (state["camera"] + 1) % 2
                    set_camera(gym, viewer, state["camera"])
                elif event.action == "capture" and args.screenshot is not None:
                    write_evidence(gym, viewer, args.screenshot, problem, directions, state, stats, step)
                    captured = True

            _, pose, naive_excess, accepted = accepted_motion_pose(
                trajectory, state["motion_step"],
            )
            update_stats(stats, problem, pose, naive_excess, accepted)
            apply_pose(gym, env, actor, q_indices, pose)
            draw_scene(gym, viewer, env, kinematics, directions, problem, pose, state)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            if args.screenshot is not None and not captured and step >= max(0, args.screenshot_step):
                write_evidence(gym, viewer, args.screenshot, problem, directions, state, stats, step)
                captured = True
            if state["motion"]:
                state["motion_step"] += 1
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                running = False
        if args.screenshot is not None and not captured:
            write_evidence(gym, viewer, args.screenshot, problem, directions, state, stats, step)
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    print(f"Viewer exited naturally after {step} steps.")
    print(
        f"Accepted compliance: {stats['joint_sample_count']} joint samples; "
        f"{stats['joint_range_violation_count']} joint violations; "
        f"{stats['envelope_violation_count']} envelope violations; "
        f"max support excess {stats['max_occupied_support_excess_m']:.9g} m"
    )
    print(
        f"Naive-box regression: {stats['naive_envelope_violation_count']} violating frames; "
        f"{stats['backtracked_frame_count']} frames backtracked"
    )


if __name__ == "__main__":
    main()
