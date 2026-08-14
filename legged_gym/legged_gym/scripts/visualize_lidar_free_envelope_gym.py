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
    interpolate_joint_ranges,
    polyline_segments,
    support_polygon,
)
from kinematic_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES,
    BatchedUrdfKinematics,
    capsule_support,
    default_el4090_capsules,
    deterministic_joint_samples,
    export_envelope_joint_ranges,
    haa_ranges_from_joint_export,
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


GRAPHITE = (0.125, 0.149, 0.180)
GROUND = (0.55, 0.57, 0.58)
WHITE = (0.96, 0.97, 0.98)
LIGHT_CYAN = (0.32, 0.86, 0.90)
DARK_TEAL = (0.00, 0.38, 0.40)
AMBER = (0.90, 0.62, 0.15)
REACHABLE_BLUE = (0.28, 0.45, 0.72)
RED = (0.86, 0.16, 0.13)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--point_count", type=int, default=192)
    parser.add_argument("--directions", type=int, default=48)
    parser.add_argument("--min_radius", type=float, default=0.0)
    parser.add_argument("--max_radius", type=float, default=2.10)
    parser.add_argument("--robot_clearance", type=float, default=0.05)
    parser.add_argument("--point_clearance", type=float, default=0.02)
    parser.add_argument("--reference_containment_margin", type=float, default=0.005)
    parser.add_argument("--min_candidate_reduction_fraction", type=float, default=0.05)
    parser.add_argument("--min_joint_shrink_rad", type=float, default=0.03)
    parser.add_argument("--motion_period_steps", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps the viewer interactive")
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
    effective_lower, effective_upper = kinematics.joint_limits(soft_fraction=0.9)
    half_width = torch.tensor([0.95, 0.42, 0.48] * 6)
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
        reference_containment_margin=args.reference_containment_margin,
    )
    reference_excess = polygon_support_excess(
        cloud.points_xy, directions, reference_reachable_support,
    )
    if float(reference_excess.max()) > -args.reference_containment_margin + 5e-6:
        raise RuntimeError("LiDAR return escaped the eroded reference reachable envelope")
    free_envelope = maximum_sector_point_free_envelope(
        cloud, directions, point_clearance=args.point_clearance,
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
    )


def motion_offsets(reference: torch.Tensor) -> torch.Tensor:
    leg_offsets = (0.00, 0.50, 0.25, 0.75, 0.50, 0.00)
    joint_offsets = (0.00, 0.18, 0.68)
    return torch.tensor(
        [leg + joint for leg in leg_offsets for joint in joint_offsets],
        dtype=reference.dtype,
        device=reference.device,
    )


def accepted_motion_pose(problem, kinematics, directions, motion_step, period_steps):
    phase = motion_step / period_steps
    proposed = interpolate_joint_ranges(
        problem.range_export.lower,
        problem.range_export.upper,
        phase,
        phase_offsets=motion_offsets(problem.baseline_q),
    ).unsqueeze(0)
    naive_excess = envelope_excess(
        kinematics,
        proposed,
        default_el4090_capsules(),
        directions,
        problem.free_envelope.support_m,
    )[0]
    accepted = backtrack_to_feasible_anchor(
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
    return proposed[0], accepted.joint_positions[0], naive_excess, accepted


def new_stats() -> dict:
    return {
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
    print(f"  returns: {problem.cloud.points_xy.shape[0]} across {directions.shape[0]} angular sectors")
    cluster_text = ", ".join(
        f"{float(value):.2f}" for value in problem.cloud.near_cluster_centers_rad
    )
    gap_text = ", ".join(
        f"{float(value):.2f}" for value in problem.cloud.far_gap_centers_rad
    )
    print(
        "  structure: randomized sector density; guaranteed full coverage; "
        f"near clusters at {cluster_text} rad; far gaps at {gap_text} rad"
    )
    print(f"  baseline capsule-envelope clearance: {float(baseline_clearance.min()):.6f} m minimum")
    print(f"  prescribed point clearance: {float(clearance.min()):.6f} m minimum")
    print(
        "  reference reachable containment: "
        f"{-float(reference_excess.max()):.6f} m minimum inward margin"
    )
    print(f"  optimality: {problem.free_envelope.optimality_scope}")
    print(f"  exported candidates: {int(diagnostics.candidate_feasible_count)}/{diagnostics.candidate_samples} feasible")
    print(f"  candidate reduction: {100.0 * problem.candidate_reduction_fraction:.2f}%")
    print(f"  maximum joint-interval shrinkage: {float(problem.joint_shrinkage.max()):.6f} rad")
    print("  colors: white returns inside blue pre-obstacle reachable reference; light cyan prescribed; dark teal occupied; amber HAA; red violations only")


def print_controls() -> None:
    rows = (
        ("G", "regenerate with seed + 1"),
        ("M", "pause or resume feasible motion"),
        ("X", "reset deterministic motion phase"),
        ("L", "toggle white LiDAR returns and clearance spokes"),
        ("P", "toggle light-cyan prescribed free envelope"),
        ("O", "toggle dark-teal current occupied envelope"),
        ("H", "toggle amber HAA ranges"),
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


def draw_boundary(gym, viewer, env, polygon, height, color) -> None:
    points = np.column_stack((polygon, np.full(len(polygon), height)))
    add_segments(gym, viewer, env, polyline_segments(points, closed=True), color)
    raised = points.copy()
    raised[:, 2] += 0.007
    add_segments(gym, viewer, env, polyline_segments(raised, closed=True), color)


def draw_cloud(gym, viewer, env, problem) -> None:
    points = problem.cloud.points_xy.detach().cpu().numpy()
    size = 0.018
    z = 0.045
    crosses = []
    for x, y in points:
        crosses.extend(((x - size, y, z), (x + size, y, z), (x, y - size, z), (x, y + size, z)))
    add_segments(gym, viewer, env, np.asarray(crosses), WHITE)
    limiting = problem.free_envelope.limiting_points_xy.detach().cpu().numpy()
    feet = problem.free_envelope.clearance_feet_xy.detach().cpu().numpy()
    spokes = np.stack((
        np.column_stack((limiting, np.full(len(limiting), z))),
        np.column_stack((feet, np.full(len(feet), z))),
    ), axis=1).reshape(-1, 3)
    add_segments(gym, viewer, env, spokes, LIGHT_CYAN)


def draw_haa(gym, viewer, env, kinematics, problem, pose) -> None:
    origins, arcs, markers = haa_arc_geometry(
        kinematics, pose, problem.haa_ranges, radius=0.25, samples=41,
    )
    translation = np.array((0.0, 0.0, BASE_HEIGHT + 0.12), dtype=np.float32)
    origins = origins.detach().cpu().numpy() + translation
    arcs = arcs.detach().cpu().numpy() + translation
    markers = markers.detach().cpu().numpy() + translation
    for index in range(6):
        add_segments(gym, viewer, env, polyline_segments(arcs[index]), AMBER)
        bounds = np.stack((origins[index], arcs[index, 0], origins[index], arcs[index, -1]))
        add_segments(gym, viewer, env, bounds, GRAPHITE)
        endpoint = origins[index] + 1.28 * (markers[index] - origins[index])
        add_segments(gym, viewer, env, np.stack((origins[index], endpoint)), AMBER)


def draw_scene(gym, viewer, env, kinematics, directions, problem, pose, state, violation) -> None:
    gym.clear_lines(viewer)
    if state["lidar"]:
        draw_cloud(gym, viewer, env, problem)
    if state["prescribed"]:
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, problem.free_envelope.support_m),
            0.060, LIGHT_CYAN,
        )
    if state["occupied"]:
        occupied = capsule_support(
            kinematics, pose.unsqueeze(0), default_el4090_capsules(), directions,
        )[0]
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, occupied),
            0.086, RED if violation else DARK_TEAL,
        )
    if state["haa"]:
        draw_haa(gym, viewer, env, kinematics, problem, pose)
    if state["reachable"]:
        draw_boundary(
            gym, viewer, env,
            support_polygon(directions, problem.reference_reachable_support),
            0.030, REACHABLE_BLUE,
        )


def set_camera(gym, viewer, mode: int) -> None:
    if mode == 0:
        position, target = (3.45, -3.25, 3.20), (0.0, 0.0, 0.28)
    else:
        position, target = (0.05, -0.05, 5.2), (0.0, 0.0, 0.12)
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
            "angular_coverage": "every fixed-normal sector has at least one jittered return",
            "structure": {
                "randomization": (
                    "seeded uneven sector density, randomized cluster/gap "
                    "centers with 0.60 rad circular separation, wide angular "
                    "jitter, and broad radial scatter"
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
            },
            "radius_bounds_m": [problem.cloud.min_radius_m, problem.cloud.max_radius_m],
            "observed_radius_bounds_m": [
                float(problem.cloud.radii_m.min()), float(problem.cloud.radii_m.max()),
            ],
            "required_baseline_clearance_m": problem.cloud.robot_clearance_m,
            "minimum_baseline_clearance_m": float(baseline_clearances.min()),
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
        "visible_layers": {key: state[key] for key in ("lidar", "prescribed", "occupied", "haa", "reachable")},
        "semantic_mapping": {
            "white": "LiDAR returns",
            "light_cyan": "prescribed point-free envelope and active clearance spokes",
            "dark_teal": "current occupied capsule envelope",
            "amber": (
                "exported HAA intervals and current markers directed from each "
                "hip toward its URDF HFE attachment"
            ),
            "blue": "pre-obstacle unconstrained reachable-foot reference",
            "red": "actual constraint violation only",
        },
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Screenshot: {path}")
    print(f"Evidence:   {json_path}")


def validate_args(args) -> None:
    if args.directions < 8:
        raise ValueError("--directions must be at least 8")
    if args.point_count < args.directions:
        raise ValueError("--point_count must cover every direction sector")
    if args.motion_period_steps <= 0:
        raise ValueError("--motion_period_steps must be positive")
    if args.point_clearance >= args.robot_clearance:
        raise ValueError("--point_clearance must be smaller than --robot_clearance")
    if args.reference_containment_margin <= 0.0:
        raise ValueError("--reference_containment_margin must be positive")
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

    if args.compute_only:
        proposed = []
        for step in range(args.motion_period_steps):
            proposed.append(interpolate_joint_ranges(
                problem.range_export.lower,
                problem.range_export.upper,
                step / args.motion_period_steps,
                phase_offsets=motion_offsets(problem.baseline_q),
            ))
        proposed = torch.stack(proposed)
        accepted = backtrack_to_feasible_anchor(
            kinematics, proposed, problem.baseline_q,
            problem.range_export.lower, problem.range_export.upper,
            default_el4090_capsules(), directions, problem.free_envelope.support_m,
            tolerance=TOLERANCE,
        )
        naive = envelope_excess(
            kinematics, proposed, default_el4090_capsules(), directions,
            problem.free_envelope.support_m,
        )
        print(
            f"Compute-only motion: {proposed.shape[0]} frames; "
            f"{int((naive > TOLERANCE).sum())} naive violations; "
            f"{int((accepted.envelope_excess_m > TOLERANCE).sum())} accepted violations; "
            f"minimum scale {float(accepted.accepted_scale.min()):.6f}"
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
        "reachable": True,
        "camera": 0,
    }
    set_camera(gym, viewer, state["camera"])
    stats = new_stats()
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
                    state["motion_step"] = 0
                    stats = new_stats()
                    print_problem(problem, directions)
                elif event.action == "motion":
                    state["motion"] = not state["motion"]
                    print(f"Feasible motion: {state['motion']}")
                elif event.action == "reset":
                    state["motion_step"] = 0
                    print("Motion phase reset")
                elif event.action in ("lidar", "prescribed", "occupied", "haa", "reachable"):
                    state[event.action] = not state[event.action]
                    print(f"{event.action} visible: {state[event.action]}")
                elif event.action == "camera":
                    state["camera"] = (state["camera"] + 1) % 2
                    set_camera(gym, viewer, state["camera"])
                elif event.action == "capture" and args.screenshot is not None:
                    write_evidence(gym, viewer, args.screenshot, problem, directions, state, stats, step)
                    captured = True

            _, pose, naive_excess, accepted = accepted_motion_pose(
                problem, kinematics, directions, state["motion_step"], args.motion_period_steps,
            )
            update_stats(stats, problem, pose, naive_excess, accepted)
            apply_pose(gym, env, actor, q_indices, pose)
            violation = (
                float(accepted.envelope_excess_m[0]) > TOLERANCE
                or float(accepted.joint_excess_rad[0]) > TOLERANCE
            )
            draw_scene(gym, viewer, env, kinematics, directions, problem, pose, state, violation)
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
