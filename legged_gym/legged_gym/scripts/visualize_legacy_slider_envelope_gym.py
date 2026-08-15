#!/usr/bin/env python3
"""Run the unchanged LiDAR envelope model with ZhangHT slider border points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
    export_envelope_joint_ranges_at_reference, feasible_reference_q,
    haa_ranges_from_joint_export, joint_rejection_ranges, load_urdf_joints,
    reachable_foot_support, support_directions,
)
from legacy_slider_envelope import (  # noqa: E402
    LEGACY_MAXIMUM, LEGACY_MIDPOINT, LEGACY_PARAMETER_ORDER,
    LEGACY_PARAMETER_RANGES, legacy_border_lidar_cloud,
    legacy_border_support, legacy_border_vertices, parameter_tensor,
)
from lidar_free_envelope import (  # noqa: E402
    maximum_sector_point_free_envelope, polygon_support_excess,
)
from visualize_lidar_free_envelope_gym import (  # noqa: E402
    LidarProblem, TOLERANCE, WHITE, accepted_motion_pose, apply_pose,
    build_motion_trajectory, create_simulation, draw_boundary, draw_cloud,
    draw_scene as draw_lidar_scene, new_stats, print_exported_box,
    print_rejection, set_camera, update_stats,
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
    # None when the resting pose cannot satisfy this envelope; the viewer then
    # freezes at the resting pose (the occupied torso turns red only when the
    # torso itself exceeds the envelope).
    trajectory: object | None


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
    parser.add_argument("--motion_period_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps both windows interactive")
    parser.add_argument("--auto_sweep_steps", type=int, default=0)
    parser.add_argument(
        "--show_rejection", action="store_true",
        help="show rejected sub-intervals of the exported joint ranges",
    )
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--no_motion", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot_step", type=int, default=5)
    parser.add_argument(
        "--border_preset", choices=("max", "fwd_060", "bwd_060", "tight_030", "narrow"),
        default=None,
        help="override the auto-sweep with a single border for evidence capture",
    )
    return parser.parse_args()


def baseline_joint_pose():
    return torch.tensor([0.0, 0.60, -0.60] * 6)


# MasterYip requested that the joint ranges genuinely reach the URDF
# mechanical limits (±3 rad for every EL4090 joint). The candidate box and the
# reference reachable cap therefore span the full URDF range. Consequence: the
# width sliders no longer tighten the exported ranges until the border narrows
# below the body width (the robot can tuck its legs inside any wider border),
# which reverses the earlier 0.50 rad effective-HAA cap.


def build_context(kinematics, args):
    directions = support_directions(args.directions)
    capsules = default_el4090_capsules()
    baseline_q = baseline_joint_pose()
    baseline_support = capsule_support(
        kinematics, baseline_q.unsqueeze(0), capsules, directions,
    )[0]
    lower, upper = kinematics.joint_limits(soft_fraction=1.0)
    half = (upper - lower) / 2
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


def inflated_envelope_cap(parameters, directions, reference_support, margin):
    """ENV-RECT-CONTROL-012: the reference cap relaxed where it is tighter than
    the rectangle border.

    The pre-obstacle reachable (foot) reference is a foot-reach bound; it must
    not shrink the free envelope below the declared border. The cap is therefore
    ``max(reference_support, border_support + margin)`` so every border support
    point stays at least ``margin`` inside the cap and the envelope can reach
    the border's full rectangle.
    """
    border_support = legacy_border_support(parameters, directions)
    return torch.maximum(reference_support, border_support + margin)


def compute_result(kinematics, context, values, args, generation):
    directions = context["directions"]
    parameters = parameter_tensor(values, dtype=directions.dtype, device=directions.device)
    envelope_cap = inflated_envelope_cap(
        parameters, directions, context["reference_support"],
        args.reference_containment_margin,
    )
    cloud = legacy_border_lidar_cloud(
        parameters, directions, context["baseline_support"], envelope_cap,
        seed=args.seed + generation,
        reference_containment_margin=args.reference_containment_margin,
    )
    reference_excess = polygon_support_excess(
        cloud.points_xy, directions, envelope_cap,
    )
    if float(reference_excess.max()) > -args.reference_containment_margin + 5e-6:
        raise ValueError("ZhangHT border points exceed the (inflated) reachable cap")
    free_envelope = maximum_sector_point_free_envelope(
        cloud, directions, point_clearance=args.point_clearance,
        cap_support=envelope_cap,
    )
    # The envelope is computed from the live border unconditionally. A border
    # that cannot contain the resting pose is a real, visualized condition
    # (red capsule, frozen motion), not a reason to retain the previous result.
    #
    # The exported box and the rejection bands both sweep each joint at a single
    # feasible reference (resting pose -> box center -> nearest feasible
    # candidate sample), so the displayed ranges are the exact per-axis feasible
    # reach and stay meaningful even when few candidate samples are
    # envelope-feasible under the full-URDF reach window.
    reference, _reference_source = feasible_reference_q(
        kinematics, context["capsules"], directions, free_envelope.support_m,
        context["lower"], context["upper"], context["baseline_q"],
        tolerance=TOLERANCE, fallback_candidates=context["candidate_q"],
    )
    if reference is None:
        # No feasible candidate at all: keep a NaN export box (the rejection and
        # the panel both render it as "infeasible") rather than crashing.
        reference = context["lower"]
    export = export_envelope_joint_ranges_at_reference(
        kinematics, context["capsules"], directions, free_envelope.support_m,
        context["lower"], context["upper"], reference,
        tolerance=TOLERANCE,
        box_validation_samples=args.box_validation_samples,
        box_validation_seed=args.seed + 300,
    )
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
        required_candidate_reduction_fraction=0.0,
        required_joint_shrink_rad=0.0,
        rejection_ranges=joint_rejection_ranges(
            kinematics, context["capsules"], directions,
            free_envelope.support_m, context["lower"], context["upper"],
            reference, tolerance=TOLERANCE,
            fallback_candidates=context["candidate_q"],
        ),
    )
    resting_inside = float(
        (context["baseline_support"] - free_envelope.support_m).max()
    ) <= TOLERANCE
    if bool(export.valid) and resting_inside:
        try:
            trajectory = build_motion_trajectory(
                problem, kinematics, directions, args.motion_period_steps,
            )
        except (ValueError, RuntimeError):
            # Exported ranges are still shown even when no smooth motion fits;
            # the viewer then freezes at the resting pose.
            trajectory = None
    else:
        trajectory = None
    return SliderResult(
        parameters=tuple(float(value) for value in parameters),
        border_vertices_xy=legacy_border_vertices(parameters),
        problem=problem,
        trajectory=trajectory,
    )


def result_summary(result):
    diagnostics = result.problem.range_export.diagnostics
    motion = (
        f"fixed motion scale {float(result.trajectory.accepted_scale[0]):.4f}"
        if result.trajectory is not None else "motion frozen"
    )
    return (
        f"feasible {int(diagnostics.candidate_feasible_count)}/"
        f"{diagnostics.candidate_samples}; box violations "
        f"{int(diagnostics.box_envelope_violation_count)}/"
        f"{diagnostics.box_validation_samples}; {motion}"
    )


def average_haa_range_deg(export):
    """Average per-leg HAA span in degrees, or None for an empty export."""
    haa = haa_ranges_from_joint_export(export)
    if torch.isnan(haa).any():
        return None
    return float((haa[..., 1] - haa[..., 0]).mean() * 180.0 / math.pi)


def live_point_source(values, context, args, generation):
    """Return the current slider border and its border-derived returns.

    The ZhangHT border is the live point source: it must always track the five
    sliders. The envelope export below may be rejected when the border is too
    small for the baseline capsule; this pure-geometry computation never is.
    """
    directions = context["directions"]
    parameters = parameter_tensor(
        values, dtype=directions.dtype, device=directions.device,
    )
    border = legacy_border_vertices(parameters)
    envelope_cap = inflated_envelope_cap(
        parameters, directions, context["reference_support"],
        args.reference_containment_margin,
    )
    cloud = legacy_border_lidar_cloud(
        parameters, directions, context["baseline_support"], envelope_cap,
        seed=args.seed + generation,
        reference_containment_margin=args.reference_containment_margin,
    )
    return border, cloud


class SliderPanel:
    def __init__(self, show_rejection=False):
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
        self.show_rejection = bool(show_rejection)
        self._last_result = None
        self._last_elapsed = 0.0
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
        self._last_result = result
        self._last_elapsed = elapsed_ms
        self._render_range_labels(result)
        motion = (
            "motion frozen: resting pose outside envelope"
            if result.trajectory is None else ""
        )
        suffix = f"; {motion}" if motion else ""
        self.status.set(
            f"{result_summary(result)}; recompute {elapsed_ms:.1f} ms{suffix}",
        )
        self.applied_generation = self.requested_generation

    def _render_range_labels(self, result):
        lower = result.problem.range_export.lower
        upper = result.problem.range_export.upper
        rejection = result.problem.rejection_ranges
        for leg in EL4090_LEG_NAMES:
            indices = [EL4090_JOINT_NAMES.index(f"{leg}_{joint}") for joint in ("HAA", "HFE", "KFE")]
            pieces = []
            for joint, index in zip(("HAA", "HFE", "KFE"), indices):
                lo, hi = float(lower[index]), float(upper[index])
                if math.isnan(lo) or math.isnan(hi):
                    pieces.append(f"{joint} infeasible")
                else:
                    text = f"{joint} reach[{lo:+.3f},{hi:+.3f}]"
                    if self.show_rejection and rejection is not None and rejection.feasible_reference:
                        intervals = rejection.rejected_intervals[index]
                        if intervals:
                            marks = ";".join(f"{a:+.3f}-{b:+.3f}" for a, b in intervals)
                            # Rejection is the complement of the reach over the
                            # full URDF range, so marks can extend beyond the box
                            # (the wings where a tightened border rejects).
                            text += f" rej({marks})"
                    pieces.append(text)
            self.range_labels[leg].set(f"{leg}  " + "   ".join(pieces))

    def refresh_labels(self):
        # Re-render the range rows only; never touch generation state so a
        # pending slider recompute is not accidentally marked applied.
        if self._last_result is not None:
            self._render_range_labels(self._last_result)

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
        print_exported_box(result.problem)
        print_rejection(result.problem.rejection_ranges)
        if result.trajectory is not None:
            cyclic = torch.cat((result.trajectory.joint_positions, result.trajectory.joint_positions[:1]))
            print(f"  maximum cyclic joint step {float(torch.diff(cyclic, dim=0).abs().max()):.6f} rad")
        else:
            print("  motion frozen: resting pose outside the envelope")
        return

    panel = SliderPanel(show_rejection=args.show_rejection)
    gym, sim, viewer, env, actor, q_indices = create_simulation(args, result.problem.baseline_q)
    state = {
        "motion": not args.no_motion, "motion_step": 0,
        "lidar": True, "prescribed": True, "occupied": True,
        "haa": True, "rejection": args.show_rejection,
        "reachable": True, "camera": 0,
        "directions": context["directions"],
    }
    set_camera(gym, viewer, 0)
    stats = new_stats(result.trajectory)
    history = []
    captured = False
    step = 0
    # The sweep walks the front width through the slider range and records the
    # exported per-leg HAA span on every recompute: the MAX border keeps the
    # resting pose feasible (motion accepted), 0.50 and 0.40 still admit poses
    # (0.40 freezes the resting pose), 0.30 tightens the HAA range strongly,
    # and the final very narrow border has no feasible pose. All recompute
    # live now, and the HAA range tightens instead of staying flat.
    if args.border_preset is not None:
        # Single-border evidence capture: override both the initial border and
        # the sweep so the rejection/accessible behavior of a specific border
        # (e.g. fwd_060 for the front-leg wings) can be screenshotted directly.
        presets = {
            "max": LEGACY_MAXIMUM,
            "fwd_060": (0.60, 0.70, 0.60, 0.60, -0.90),
            "bwd_060": (0.60, 0.70, 0.60, 0.90, -0.60),
            "tight_030": (0.30, 0.70, 0.60, 0.84, -0.84),
            "narrow": (0.30, 0.50, 0.30, 0.70, -0.70),
        }
        sweep = (presets[args.border_preset],)
        result = compute_result(
            kinematics, context, sweep[0], args, panel.requested_generation,
        )
        # Mark the preset applied so the panel's pending() stays False and the
        # loop does not immediately recompute the slider's initial MAX border.
        panel.update_result(result, 0.0)
    else:
        sweep = (
            LEGACY_MAXIMUM,
            (0.50, 0.70, 0.60, 0.88, -0.88),
            (0.40, 0.70, 0.60, 0.86, -0.86),
            (0.30, 0.70, 0.60, 0.84, -0.84),
            (0.30, 0.50, 0.30, 0.70, -0.70),
        )
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
                elif event.action == "rejection":
                    state["rejection"] = not state["rejection"]
                    panel.show_rejection = state["rejection"]
                    panel.refresh_labels()
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
                    haa_span = average_haa_range_deg(result.problem.range_export)
                    history.append({
                        "generation": panel.applied_generation,
                        "parameters": dict(zip(LEGACY_PARAMETER_ORDER, result.parameters)),
                        "feasible_candidate_count": int(
                            result.problem.range_export.diagnostics.candidate_feasible_count
                        ),
                        "haa_range_deg": haa_span,
                        "elapsed_ms": elapsed_ms,
                    })
                    haa_text = (
                        f" HAA {haa_span:.1f} deg" if haa_span is not None
                        else " HAA infeasible"
                    )
                    print(f"  update {len(history)}: {result_summary(result)};{haa_text}; {elapsed_ms:.1f} ms")
            if result.trajectory is not None:
                _, pose, naive_excess, accepted = accepted_motion_pose(
                    result.trajectory, state["motion_step"],
                )
                update_stats(stats, result.problem, pose, naive_excess, accepted)
            else:
                # Infeasible border: show the resting pose against the fresh
                # envelope and freeze motion. The occupied torso polygon turns
                # red only when the torso itself exceeds the envelope.
                pose = result.problem.baseline_q
                naive_excess = 0.0
            apply_pose(gym, env, actor, q_indices, pose)
            live_border, live_cloud = live_point_source(
                panel.snapshot(), context, args, panel.requested_generation,
            )
            scene_state = dict(state)
            scene_state["lidar"] = False  # the point source is drawn live below
            if not bool(result.problem.range_export.valid):
                scene_state["haa"] = False  # empty ranges are NaN and must not be drawn
            draw_lidar_scene(
                gym, viewer, env, kinematics, context["directions"],
                result.problem, pose, scene_state,
            )
            draw_boundary(
                gym, viewer, env, live_border.detach().cpu().numpy(), 0.035, WHITE,
            )
            if state["lidar"]:
                draw_cloud(
                    gym, viewer, env,
                    SimpleNamespace(
                        cloud=live_cloud,
                        free_envelope=result.problem.free_envelope,
                    ),
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
            if result.trajectory is not None and state["motion"]:
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
