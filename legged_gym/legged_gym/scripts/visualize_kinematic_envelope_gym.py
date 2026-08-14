#!/usr/bin/env python3
"""Standalone Isaac Gym comparison viewer for EL4090 kinematic envelopes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaacgym import gymapi  # Must precede Torch imports for Isaac Gym Preview 4.
import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ENVELOPE_DIR = PACKAGE_ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))

from gym_envelope_geometry import (  # noqa: E402
    DemoPreset,
    build_demo_preset,
    haa_arc_geometry,
    polyline_segments,
    support_polygon,
)
from kinematic_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES,
    EL4090_LEG_NAMES,
    BatchedUrdfKinematics,
    load_urdf_joints,
    support_directions,
)


GRAPHITE = (0.125, 0.149, 0.180)
TEAL = (0.000, 0.486, 0.514)
CYAN = (0.220, 0.749, 0.765)
AMBER = (0.851, 0.604, 0.169)
RED = (0.851, 0.314, 0.263)
NEUTRAL_GROUND = (0.56, 0.58, 0.59)
ROBOT_NEUTRAL = (0.32, 0.35, 0.38)
PRESET_OFFSETS_Y = (-2.4, 0.0, 2.4)
BASE_HEIGHT = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--directions", type=int, default=48)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps the viewer interactive")
    parser.add_argument("--auto_cycle_steps", type=int, default=180)
    parser.add_argument("--no_auto_cycle", action="store_true")
    parser.add_argument("--compute_only", action="store_true", help="Build and validate presets without creating a simulator")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot_step", type=int, default=5)
    return parser.parse_args()


def _joint_vector(haa, hfe: float, kfe: float) -> torch.Tensor:
    values = []
    for angle in haa:
        values.extend((angle, hfe, kfe))
    return torch.tensor(values, dtype=torch.float32)


def build_presets(kinematics, directions):
    lower, upper = kinematics.joint_limits(soft_fraction=0.9)
    specs = (
        (
            "compact-mammal",
            _joint_vector((1.308, -1.308, 1.308, 1.308, -1.308, 1.308), 1.0, -0.608),
            torch.tensor([0.24, 0.10, 0.08] * 6),
            0.045,
            RED,
            31,
        ),
        (
            "nominal-spider",
            _joint_vector((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.6, -0.6),
            torch.tensor([0.52, 0.18, 0.15] * 6),
            0.105,
            AMBER,
            47,
        ),
        (
            "wide-low",
            _joint_vector((0.35, -0.35, 0.20, -0.35, 0.35, -0.20), 0.25, -1.10),
            torch.tensor([0.86, 0.28, 0.24] * 6),
            0.185,
            CYAN,
            59,
        ),
    )
    return tuple(
        build_demo_preset(
            name,
            kinematics,
            directions,
            current,
            half_width,
            lower,
            upper,
            support_margin=margin,
            accent_rgb=accent,
            seed=seed,
        )
        for name, current, half_width, margin, accent, seed in specs
    )


def preset_record(preset: DemoPreset, directions: torch.Tensor) -> dict:
    diagnostics = preset.range_export.diagnostics
    return {
        "name": preset.name,
        "frame": "+x forward, +y left, body-yaw",
        "directions_xy": directions.detach().cpu().tolist(),
        "current_joint_positions_rad": {
            name: float(preset.current_q[index]) for index, name in enumerate(EL4090_JOINT_NAMES)
        },
        "support_margin_m": preset.support_margin,
        "occupied_support_m": preset.occupied_support.detach().cpu().tolist(),
        "allowed_support_m": preset.allowed_support.detach().cpu().tolist(),
        "reachable_support_m": preset.reachable_support.detach().cpu().tolist(),
        "haa_ranges_rad_simulator_order": {
            leg: [float(value) for value in preset.haa_ranges[index]]
            for index, leg in enumerate(EL4090_LEG_NAMES)
        },
        "diagnostics": {
            "candidate_samples": diagnostics.candidate_samples,
            "candidate_feasible_count": int(diagnostics.candidate_feasible_count),
            "validation_samples": diagnostics.validation_samples,
            "validation_feasible_count": int(diagnostics.validation_feasible_count),
            "false_exclusion_count": int(diagnostics.false_exclusion_count),
            "box_validation_samples": diagnostics.box_validation_samples,
            "box_envelope_violation_count": int(diagnostics.box_envelope_violation_count),
            "max_box_envelope_violation_m": float(diagnostics.max_box_envelope_violation),
            "label": diagnostics.label,
        },
    }


def print_records(records) -> None:
    print("\nEL4090 envelope comparison: exact deterministic preset data")
    print(json.dumps(records, indent=2, sort_keys=True))


def print_controls() -> None:
    rows = (
        ("1 / 2 / 3", "select compact / nominal / wide preset"),
        ("Space", "select next preset"),
        ("A", "toggle automatic selection cycle"),
        ("O", "toggle occupied capsule boundary"),
        ("R", "toggle reachable-foot boundary"),
        ("H", "toggle six HAA interval arcs and markers"),
        ("C", "cycle overview / top / selected camera"),
        ("P", "write screenshot and matching JSON evidence"),
        ("Esc", "exit"),
    )
    width = max(len(key) for key, _ in rows)
    print("\nViewer controls")
    for key, action in rows:
        print(f"  {key:<{width}}  {action}")


def add_segments(gym, viewer, env, segments: np.ndarray, color) -> None:
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
    count = segments.shape[0] // 2
    colors = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
    gym.add_lines(viewer, env, count, segments, colors)


def draw_boundary(gym, viewer, env, polygon, offset_y, height, color) -> None:
    points = np.column_stack((polygon, np.full(len(polygon), height, dtype=np.float64)))
    points[:, 1] += offset_y
    add_segments(gym, viewer, env, polyline_segments(points, closed=True), color)
    second = points.copy()
    second[:, 2] += 0.008
    add_segments(gym, viewer, env, polyline_segments(second, closed=True), color)


def draw_datum(gym, viewer, env, offset_y, accent, selected) -> None:
    length = 0.82 if selected else 0.68
    z = 0.035
    points = np.array((
        (-length, offset_y, z), (length, offset_y, z),
        (0.0, offset_y - length, z), (0.0, offset_y + length, z),
    ), dtype=np.float32)
    add_segments(gym, viewer, env, points, accent)


def draw_haa_intervals(gym, viewer, env, kinematics, preset, offset_y, selected) -> None:
    origins, arcs, markers = haa_arc_geometry(
        kinematics, preset.current_q, preset.haa_ranges, radius=0.26, samples=49,
    )
    translation = np.array((0.0, offset_y, BASE_HEIGHT), dtype=np.float32)
    origins_np = origins.detach().cpu().numpy() + translation
    arcs_np = arcs.detach().cpu().numpy() + translation
    markers_np = markers.detach().cpu().numpy() + translation
    origins_np[:, 2] += 0.12
    arcs_np[:, :, 2] += 0.12
    markers_np[:, 2] += 0.12
    for leg_index in range(6):
        add_segments(gym, viewer, env, polyline_segments(arcs_np[leg_index]), AMBER)
        raised_arc = arcs_np[leg_index].copy()
        raised_arc[:, 2] += 0.006
        add_segments(gym, viewer, env, polyline_segments(raised_arc), AMBER)
        bounds = np.stack((
            origins_np[leg_index], arcs_np[leg_index, 0],
            origins_np[leg_index], arcs_np[leg_index, -1],
        ))
        add_segments(gym, viewer, env, bounds, GRAPHITE)
        raised_bounds = bounds.copy()
        raised_bounds[:, 2] += 0.006
        add_segments(gym, viewer, env, raised_bounds, GRAPHITE)
        scale = 1.38 if selected else 1.20
        endpoint = origins_np[leg_index] + scale * (markers_np[leg_index] - origins_np[leg_index])
        add_segments(
            gym, viewer, env,
            np.stack((origins_np[leg_index], endpoint)),
            RED,
        )


def draw_scene(gym, viewer, env, kinematics, presets, directions, state) -> None:
    gym.clear_lines(viewer)
    for index, (preset, offset_y) in enumerate(zip(presets, PRESET_OFFSETS_Y)):
        selected = index == state["active"]
        draw_datum(gym, viewer, env, offset_y, preset.accent_rgb, selected)
        if state["occupied"]:
            draw_boundary(
                gym, viewer, env,
                support_polygon(directions, preset.occupied_support),
                offset_y, BASE_HEIGHT + 0.25, TEAL,
            )
        if state["reachable"]:
            draw_boundary(
                gym, viewer, env,
                support_polygon(directions, preset.reachable_support),
                offset_y, 0.075, CYAN,
            )
        if state["haa"]:
            draw_haa_intervals(gym, viewer, env, kinematics, preset, offset_y, selected)


def set_camera(gym, viewer, state) -> None:
    mode = state["camera"]
    if mode == 0:
        position, target = (3.90, -0.2, 4.70), (0.0, 0.0, 0.30)
    elif mode == 1:
        position, target = (0.1, -0.05, 8.2), (0.0, 0.0, 0.18)
    else:
        y = PRESET_OFFSETS_Y[state["active"]]
        position, target = (3.1, y - 1.55, 2.65), (0.0, y, 0.36)
    gym.viewer_camera_look_at(
        viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target),
    )


def write_evidence(gym, viewer, screenshot: Path, records, state, step) -> None:
    screenshot = screenshot.resolve()
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    gym.write_viewer_image_to_file(viewer, str(screenshot))
    try:
        recorded_path = str(screenshot.relative_to(Path.cwd().resolve()))
    except ValueError:
        recorded_path = str(screenshot)
    evidence = {
        "screenshot": recorded_path,
        "step": step,
        "active_preset": records[state["active"]]["name"],
        "visibility": {key: state[key] for key in ("occupied", "reachable", "haa")},
        "camera_mode": state["camera"],
        "presets": records,
    }
    evidence_path = screenshot.with_suffix(".json")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Screenshot: {screenshot}")
    print(f"Evidence:   {evidence_path}")


def create_simulation(args, presets):
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 4
    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, params,
    )
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create the PhysX simulator")

    ground_options = gymapi.AssetOptions()
    ground_options.fix_base_link = True
    ground = gym.create_box(sim, 3.4, 7.7, 0.04, ground_options)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = False
    asset_options.flip_visual_attachments = False
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_root = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf"
    robot_asset = gym.load_asset(sim, str(asset_root), "el_4090.urdf", asset_options)
    if robot_asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError(f"Failed to load EL4090 asset from {asset_root}")
    asset_dof_names = tuple(gym.get_asset_dof_names(robot_asset))
    if set(asset_dof_names) != set(EL4090_JOINT_NAMES):
        gym.destroy_sim(sim)
        raise RuntimeError(f"Unexpected EL4090 DOF names: {asset_dof_names}")

    env = gym.create_env(
        sim, gymapi.Vec3(-4.0, -5.0, 0.0), gymapi.Vec3(4.0, 5.0, 3.0), 1,
    )
    ground_pose = gymapi.Transform()
    ground_pose.p = gymapi.Vec3(0.0, 0.0, -0.03)
    ground_actor = gym.create_actor(env, ground, ground_pose, "neutral_ground", 0, 0)
    gym.set_rigid_body_color(env, ground_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*NEUTRAL_GROUND))

    actor_handles = []
    body_names = tuple(gym.get_asset_rigid_body_names(robot_asset))
    for index, (preset, offset_y) in enumerate(zip(presets, PRESET_OFFSETS_Y)):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, offset_y, BASE_HEIGHT)
        actor = gym.create_actor(env, robot_asset, pose, preset.name, index + 1, 0)
        actor_handles.append(actor)
        states = np.zeros(len(asset_dof_names), dtype=gymapi.DofState.dtype)
        states["pos"] = np.asarray([
            float(preset.current_q[EL4090_JOINT_NAMES.index(name)]) for name in asset_dof_names
        ], dtype=np.float32)
        gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)
        for body_index, body_name in enumerate(body_names):
            color = preset.accent_rgb if body_name.endswith("_HIP") else ROBOT_NEUTRAL
            gym.set_rigid_body_color(
                env, actor, body_index, gymapi.MESH_VISUAL, gymapi.Vec3(*color),
            )
    gym.prepare_sim(sim)
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(0.88, 0.88, 0.88),
        gymapi.Vec3(0.42, 0.42, 0.42),
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
        (gymapi.KEY_1, "preset_1"),
        (gymapi.KEY_2, "preset_2"),
        (gymapi.KEY_3, "preset_3"),
        (gymapi.KEY_SPACE, "next"),
        (gymapi.KEY_A, "auto"),
        (gymapi.KEY_O, "occupied"),
        (gymapi.KEY_R, "reachable"),
        (gymapi.KEY_H, "haa"),
        (gymapi.KEY_C, "camera"),
        (gymapi.KEY_P, "screenshot"),
    )
    for key, action in bindings:
        gym.subscribe_viewer_keyboard_event(viewer, key, action)
    return gym, sim, viewer, env, actor_handles


def main() -> None:
    args = parse_args()
    urdf = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    directions = support_directions(args.directions)
    presets = build_presets(kinematics, directions)
    records = [preset_record(preset, directions) for preset in presets]
    print_records(records)
    if args.compute_only:
        print("Compute-only validation complete; no simulator or policy checkpoint used.")
        return

    print_controls()
    gym, sim, viewer, env, _ = create_simulation(args, presets)
    state = {
        "active": 0,
        "auto": not args.no_auto_cycle,
        "occupied": True,
        "reachable": True,
        "haa": True,
        "camera": 0,
    }
    set_camera(gym, viewer, state)
    screenshot = args.screenshot
    captured = False
    step = 0
    try:
        running = True
        while running and not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    running = False
                elif event.action.startswith("preset_"):
                    state["active"] = int(event.action[-1]) - 1
                    print(f"Selected preset: {presets[state['active']].name}")
                elif event.action == "next":
                    state["active"] = (state["active"] + 1) % len(presets)
                    print(f"Selected preset: {presets[state['active']].name}")
                elif event.action == "auto":
                    state["auto"] = not state["auto"]
                    print(f"Automatic cycle: {state['auto']}")
                elif event.action in ("occupied", "reachable", "haa"):
                    state[event.action] = not state[event.action]
                    print(f"{event.action} visible: {state[event.action]}")
                elif event.action == "camera":
                    state["camera"] = (state["camera"] + 1) % 3
                    set_camera(gym, viewer, state)
                elif event.action == "screenshot" and screenshot is not None:
                    write_evidence(gym, viewer, screenshot, records, state, step)
                    captured = True
            if state["auto"] and args.auto_cycle_steps > 0 and step > 0 and step % args.auto_cycle_steps == 0:
                state["active"] = (state["active"] + 1) % len(presets)
                print(f"Automatic preset: {presets[state['active']].name}")
            draw_scene(gym, viewer, env, kinematics, presets, directions, state)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            if screenshot is not None and not captured and step >= max(0, args.screenshot_step):
                write_evidence(gym, viewer, screenshot, records, state, step)
                captured = True
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                running = False
        if screenshot is not None and not captured:
            write_evidence(gym, viewer, screenshot, records, state, step)
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    print(f"Viewer exited naturally after {step} steps.")


if __name__ == "__main__":
    main()
