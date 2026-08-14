#!/usr/bin/env python3
"""Run the unchanged LiDAR envelope model with ZhangHT slider border points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
import time
from pathlib import Path
import tkinter as tk

from isaacgym import gymapi  # Must precede Torch imports for Isaac Gym Preview 4.
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ENVELOPE_DIR = PACKAGE_ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))

from kinematic_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES, EL4090_LEG_NAMES, BatchedUrdfKinematics,
    capsule_support, default_el4090_capsules, deterministic_joint_samples,
    export_envelope_joint_ranges, haa_ranges_from_joint_export,
    load_urdf_joints, reachable_foot_support, support_directions,
)
from legacy_slider_envelope import (  # noqa: E402
    LEGACY_MAXIMUM, LEGACY_MIDPOINT, LEGACY_PARAMETER_ORDER,
    LEGACY_PARAMETER_RANGES, legacy_border_lidar_cloud,
    legacy_border_vertices, parameter_tensor,
)
from lidar_free_envelope import (  # noqa: E402
    maximum_sector_point_free_envelope, polygon_support_excess,
)
from visualize_lidar_free_envelope_gym import (  # noqa: E402
    LidarProblem, TOLERANCE, WHITE, accepted_motion_pose, apply_pose,
    build_motion_trajectory, create_simulation, draw_boundary,
    draw_scene as draw_lidar_scene, new_stats, set_camera, update_stats,
    write_evidence as write_lidar_evidence,
)

GRAPHITE = "#20262e"
PANEL_TEXT = "#edf3f5"
PANEL_MUTED = "#aebcc2"


@dataclass(frozen=True)
class SliderResult:
    parameters: tuple[float, ...]
    border_vertices_xy: torch.Tensor
    problem: LidarProblem
    trajectory: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--directions", type=int, default=48)
    parser.add_argument("--candidate_count", type=int, default=768)
    parser.add_argument("--validation_count", type=int, default=257)
    parser.add_argument("--box_validation_samples", type=int, default=256)
    parser.add_argument("--point_clearance", type=float, default=0.02)
    parser.add_argument("--reference_containment_margin", type=float, default=0.005)
    parser.add_argument("--min_candidate_reduction_fraction", type=float, default=0.05)
    parser.add_argument("--min_joint_shrink_rad", type=float, default=0.03)
    parser.add_argument("--motion_period_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps both windows interactive")
    parser.add_argument("--auto_sweep_steps", type=int, default=0)
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--no_motion", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot_step", type=int, default=5)
    return parser.parse_args()


def baseline_joint_pose():
    return torch.tensor([0.0, 0.60, -0.60] * 6)


def build_context(kinematics, args):
    directions = support_directions(args.directions)
    capsules = default_el4090_capsules()
    baseline_q = baseline_joint_pose()
    baseline_support = capsule_support(
        kinematics, baseline_q.unsqueeze(0), capsules, directions,
    )[0]
    lower, upper = kinematics.joint_limits(soft_fraction=0.9)
    half = torch.tensor([0.95, 0.42, 0.48] * 6)
    candidate_lower = torch.maximum(baseline_q - half, lower)
    candidate_upper = torch.minimum(baseline_q + half, upper)
    candidate_q = torch.cat((
        baseline_q.unsqueeze(0),
        deterministic_joint_samples(
            candidate_lower, candidate_upper, args.candidate_count,
            seed=args.seed + 100,
        ),
    ))
    validation_q = deterministic_joint_samples(
        candidate_lower, candidate_upper, args.validation_count,
        seed=args.seed + 200,
    )
    reference_support = reachable_foot_support(
        kinematics, candidate_q.unsqueeze(0), directions,
    )[0]
    return {
        "directions": directions, "capsules": capsules,
        "baseline_q": baseline_q, "baseline_support": baseline_support,
        "lower": lower, "upper": upper,
        "candidate_lower": candidate_lower, "candidate_upper": candidate_upper,
        "candidate_q": candidate_q, "validation_q": validation_q,
        "reference_support": reference_support,
    }


def compute_result(kinematics, context, values, args, generation):
    directions = context["directions"]
    parameters = parameter_tensor(values, dtype=directions.dtype, device=directions.device)
    cloud = legacy_border_lidar_cloud(
        parameters, directions, context["baseline_support"],
        context["reference_support"], seed=args.seed + generation,
        reference_containment_margin=args.reference_containment_margin,
    )
    reference_excess = polygon_support_excess(
        cloud.points_xy, directions, context["reference_support"],
    )
    if float(reference_excess.max()) > -args.reference_containment_margin + 5e-6:
        raise ValueError("ZhangHT border points exceed the pre-obstacle reachable reference")
    free_envelope = maximum_sector_point_free_envelope(
        cloud, directions, point_clearance=args.point_clearance,
        cap_support=context["reference_support"],
    )
    if float((context["baseline_support"] - free_envelope.support_m).max()) > TOLERANCE:
        raise ValueError("ZhangHT border is too small for the feasible capsule anchor")
    export = export_envelope_joint_ranges(
        kinematics, context["candidate_q"], context["validation_q"], directions,
        free_envelope.support_m, context["lower"], context["upper"],
        capsules=context["capsules"],
        box_validation_samples=args.box_validation_samples,
        box_validation_seed=args.seed + 300,
    )
    if not bool(export.valid):
        raise RuntimeError("slider border produced no feasible exported joint range")
    feasible = (
        capsule_support(
            kinematics, context["candidate_q"], context["capsules"], directions,
        ) <= free_envelope.support_m.unsqueeze(0) + TOLERANCE
    ).all(-1)
    feasible_count = int(feasible.sum())
    reduction = 1.0 - feasible_count / context["candidate_q"].shape[0]
    shrinkage = (context["candidate_upper"] - context["candidate_lower"]) - (
        export.upper - export.lower
    )
    problem = LidarProblem(
        seed=args.seed + generation,
        baseline_q=context["baseline_q"],
        baseline_support=context["baseline_support"],
        cloud=cloud,
        free_envelope=free_envelope,
        range_export=export,
        haa_ranges=haa_ranges_from_joint_export(export),
        reference_reachable_support=context["reference_support"],
        candidate_lower=context["candidate_lower"],
        candidate_upper=context["candidate_upper"],
        joint_shrinkage=shrinkage,
        candidate_reduction_fraction=reduction,
        required_candidate_reduction_fraction=args.min_candidate_reduction_fraction,
        required_joint_shrink_rad=args.min_joint_shrink_rad,
    )
    trajectory = build_motion_trajectory(
        problem, kinematics, directions, args.motion_period_steps,
    )
    return SliderResult(
        parameters=tuple(float(value) for value in parameters),
        border_vertices_xy=legacy_border_vertices(parameters),
        problem=problem,
        trajectory=trajectory,
    )


def result_summary(result):
    diagnostics = result.problem.range_export.diagnostics
    return (
        f"feasible {int(diagnostics.candidate_feasible_count)}/"
        f"{diagnostics.candidate_samples}; box violations "
        f"{int(diagnostics.box_envelope_violation_count)}/"
        f"{diagnostics.box_validation_samples}; fixed motion scale "
        f"{float(result.trajectory.accepted_scale[0]):.4f}"
    )


class SliderPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EL4090 | ZhangHT Border Point Source")
        self.root.geometry("610x720+32+32")
        self.root.configure(bg=GRAPHITE)
        self.running = True
        self.capture_requested = False
        self.requested_generation = 0
        self.applied_generation = -1
        self.deadline = time.monotonic()
        self.suspend = False
        self.variables = {}
        self.range_labels = {}
        tk.Label(
            self.root, text="ZHANGHT BORDER POINT SOURCE", bg=GRAPHITE,
            fg=PANEL_TEXT, font=("TkDefaultFont", 15, "bold"), anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 2))
        tk.Label(
            self.root,
            text="LiDAR math unchanged   White: border points   Cyan: maximum   Teal: capsule",
            bg=GRAPHITE, fg=PANEL_MUTED, font=("TkDefaultFont", 9), anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 10))
        sliders = tk.Frame(self.root, bg=GRAPHITE)
        sliders.pack(fill="x", padx=18)
        for name, initial in zip(LEGACY_PARAMETER_ORDER, LEGACY_MAXIMUM):
            lower, upper = LEGACY_PARAMETER_RANGES[name]
            row = tk.Frame(sliders, bg=GRAPHITE)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=name.replace("_", " ").title(), width=17,
                anchor="w", bg=GRAPHITE, fg=PANEL_TEXT,
            ).pack(side="left")
            variable = tk.DoubleVar(value=initial)
            self.variables[name] = variable
            scale = tk.Scale(
                row, variable=variable, from_=lower, to=upper, resolution=0.01,
                orient="horizontal", length=330, showvalue=True, digits=3,
                command=self._slider_changed, bg=GRAPHITE, fg=PANEL_TEXT,
                troughcolor="#46525b", activebackground="#52dbe6",
                highlightthickness=0, bd=0, font=("TkFixedFont", 9),
            )
            scale.pack(side="left", fill="x", expand=True)
            scale.bind("<ButtonRelease-1>", self._slider_released)
        controls = tk.Frame(self.root, bg=GRAPHITE)
        controls.pack(fill="x", padx=18, pady=(10, 8))
        self._button(controls, "Reset midpoint", lambda: self.set_values(LEGACY_MIDPOINT)).pack(side="left")
        self._button(controls, "Maximum border", lambda: self.set_values(LEGACY_MAXIMUM)).pack(side="left", padx=8)
        self._button(controls, "Capture", self._capture).pack(side="left")
        tk.Frame(self.root, height=1, bg="#46525b").pack(fill="x", padx=18, pady=6)
        tk.Label(
            self.root, text="EXPORTED JOINT RANGES [rad]", bg=GRAPHITE,
            fg=PANEL_TEXT, font=("TkDefaultFont", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=18, pady=(3, 5))
        for leg in EL4090_LEG_NAMES:
            value = tk.StringVar(value=f"{leg}  computing...")
            self.range_labels[leg] = value
            tk.Label(
                self.root, textvariable=value, bg=GRAPHITE, fg=PANEL_TEXT,
                font=("TkFixedFont", 9), anchor="w",
            ).pack(fill="x", padx=18, pady=2)
        self.status = tk.StringVar(value="Waiting for first unchanged-model export")
        tk.Label(
            self.root, textvariable=self.status, bg=GRAPHITE, fg="#52dbe6",
            font=("TkDefaultFont", 9), anchor="w", wraplength=570,
        ).pack(fill="x", padx=18, pady=(12, 6))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.enqueue(immediate=True)

    def _button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, bg="#46525b", fg=PANEL_TEXT,
            activebackground="#52dbe6", activeforeground=GRAPHITE,
            relief="flat", padx=10, pady=5,
        )

    def _slider_changed(self, _value=None):
        if not self.suspend:
            self.enqueue(immediate=False)

    def _slider_released(self, _event=None):
        self.enqueue(immediate=True)

    def _capture(self):
        self.capture_requested = True

    def enqueue(self, *, immediate):
        self.requested_generation += 1
        self.deadline = time.monotonic() if immediate else time.monotonic() + 0.08
        self.status.set("Recompute queued...")

    def set_values(self, values):
        self.suspend = True
        for name, value in zip(LEGACY_PARAMETER_ORDER, values):
            self.variables[name].set(value)
        self.suspend = False
        self.enqueue(immediate=True)

    def snapshot(self):
        return tuple(self.variables[name].get() for name in LEGACY_PARAMETER_ORDER)

    def pending(self):
        return self.requested_generation != self.applied_generation and time.monotonic() >= self.deadline

    def update_result(self, result, elapsed_ms):
        lower = result.problem.range_export.lower
        upper = result.problem.range_export.upper
        for leg in EL4090_LEG_NAMES:
            indices = [EL4090_JOINT_NAMES.index(f"{leg}_{joint}") for joint in ("HAA", "HFE", "KFE")]
            pieces = [
                f"{joint} [{float(lower[index]):+.3f}, {float(upper[index]):+.3f}]"
                for joint, index in zip(("HAA", "HFE", "KFE"), indices)
            ]
            self.range_labels[leg].set(f"{leg}  " + "   ".join(pieces))
        self.status.set(f"{result_summary(result)}; recompute {elapsed_ms:.1f} ms")
        self.applied_generation = self.requested_generation

    def update_error(self, message):
        self.status.set(f"Last valid envelope retained: {message}")
        self.applied_generation = self.requested_generation

    def pump(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.running = False

    def close(self):
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def write_evidence(gym, viewer, screenshot, result, state, stats, history, step):
    write_lidar_evidence(
        gym, viewer, screenshot, result.problem, state["directions"],
        state, stats, step,
    )
    path = screenshot.resolve().with_suffix(".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["point_source"] = "ZhangHT slider border ray intersections"
    record["cloud"]["angular_coverage"] = (
        "one ZhangHT-border-derived return in every registered sector"
    )
    record["cloud"]["structure"]["randomization"] = (
        "none; deterministic ray intersections controlled by the five sliders"
    )
    record["envelope_math_model"] = result.problem.free_envelope.optimality_scope
    record["slider_parameters"] = dict(zip(LEGACY_PARAMETER_ORDER, result.parameters))
    record["slider_border_vertices_xy"] = result.border_vertices_xy.tolist()
    record["recompute_history"] = history
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.directions < 8 or args.candidate_count < 1 or args.validation_count < 0:
        raise ValueError("invalid direction or sample count")
    if args.reference_containment_margin <= 0.0:
        raise ValueError("reference containment margin must be positive")
    if args.motion_period_steps < 2 or args.point_clearance <= 0.0:
        raise ValueError("motion period must be at least 2 and point clearance positive")
    urdf = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    context = build_context(kinematics, args)
    result = compute_result(kinematics, context, LEGACY_MAXIMUM, args, 0)
    print("ZhangHT border point source -> unchanged LiDAR envelope model")
    print(f"  {result_summary(result)}")
    if args.compute_only:
        cyclic = torch.cat((result.trajectory.joint_positions, result.trajectory.joint_positions[:1]))
        print(f"  maximum cyclic joint step {float(torch.diff(cyclic, dim=0).abs().max()):.6f} rad")
        return

    panel = SliderPanel()
    gym, sim, viewer, env, actor, q_indices = create_simulation(args, result.problem.baseline_q)
    state = {
        "motion": not args.no_motion, "motion_step": 0,
        "lidar": True, "prescribed": True, "occupied": True,
        "haa": True, "reachable": True, "camera": 0,
        "directions": context["directions"],
    }
    set_camera(gym, viewer, 0)
    stats = new_stats(result.trajectory)
    history = []
    captured = False
    step = 0
    sweep = (LEGACY_MAXIMUM, (0.55, 0.62, 0.55, 0.85, -0.85), LEGACY_MIDPOINT)
    try:
        while panel.running and not gym.query_viewer_has_closed(viewer):
            panel.pump()
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    panel.running = False
                elif event.action == "motion":
                    state["motion"] = not state["motion"]
                elif event.action == "reset":
                    state["motion_step"] = 0
                elif event.action in ("lidar", "prescribed", "occupied", "haa", "reachable"):
                    state[event.action] = not state[event.action]
                elif event.action == "camera":
                    state["camera"] = (state["camera"] + 1) % 2
                    set_camera(gym, viewer, state["camera"])
                elif event.action == "capture":
                    panel.capture_requested = True
            if args.auto_sweep_steps > 0 and step > 0 and step % args.auto_sweep_steps == 0:
                panel.set_values(sweep[(step // args.auto_sweep_steps) % len(sweep)])
            if panel.pending():
                values = panel.snapshot()
                started = time.perf_counter()
                try:
                    updated = compute_result(
                        kinematics, context, values, args, panel.requested_generation,
                    )
                except (ValueError, RuntimeError) as error:
                    panel.update_error(str(error))
                else:
                    elapsed_ms = 1000.0 * (time.perf_counter() - started)
                    result = updated
                    state["motion_step"] = 0
                    stats = new_stats(result.trajectory)
                    panel.update_result(result, elapsed_ms)
                    history.append({
                        "generation": panel.applied_generation,
                        "parameters": dict(zip(LEGACY_PARAMETER_ORDER, result.parameters)),
                        "feasible_candidate_count": int(
                            result.problem.range_export.diagnostics.candidate_feasible_count
                        ),
                        "elapsed_ms": elapsed_ms,
                    })
                    print(f"  update {len(history)}: {result_summary(result)}; {elapsed_ms:.1f} ms")
            _, pose, naive_excess, accepted = accepted_motion_pose(
                result.trajectory, state["motion_step"],
            )
            update_stats(stats, result.problem, pose, naive_excess, accepted)
            apply_pose(gym, env, actor, q_indices, pose)
            violation = (
                float(accepted.envelope_excess_m[0]) > TOLERANCE
                or float(accepted.joint_excess_rad[0]) > TOLERANCE
            )
            draw_lidar_scene(
                gym, viewer, env, kinematics, context["directions"],
                result.problem, pose, state, violation,
            )
            draw_boundary(
                gym, viewer, env, result.border_vertices_xy.detach().cpu().numpy(),
                0.035, WHITE,
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            if args.screenshot is not None and (
                panel.capture_requested or (not captured and step >= args.screenshot_step)
            ):
                write_evidence(gym, viewer, args.screenshot, result, state, stats, history, step)
                panel.capture_requested = False
                captured = True
            if state["motion"]:
                state["motion_step"] += 1
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                panel.running = False
    finally:
        panel.close()
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    print(f"Viewer and Tk panel exited naturally after {step} steps; {len(history)} recomputes")


if __name__ == "__main__":
    main()
