#!/usr/bin/env python3
"""Interactively recompute EL4090 joint ranges inside ZhangHT's legacy border."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import tkinter as tk

from isaacgym import gymapi  # Must precede Torch imports for Isaac Gym Preview 4.
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ENVELOPE_DIR = PACKAGE_ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))

from gym_envelope_geometry import haa_arc_geometry, polyline_segments  # noqa: E402
from kinematic_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES, EL4090_LEG_NAMES, BatchedUrdfKinematics,
    deterministic_joint_samples, haa_ranges_from_joint_export,
    load_urdf_joints,
)
from legacy_slider_envelope import (  # noqa: E402
    LEGACY_MAXIMUM, LEGACY_MIDPOINT, LEGACY_PARAMETER_ORDER,
    LEGACY_PARAMETER_RANGES, LEGACY_VISUAL_SEMANTICS,
    compute_legacy_admissible_envelope, convex_hull_2d, legacy_feasible,
    parameter_tensor,
)
from visualize_lidar_free_envelope_gym import (  # noqa: E402
    add_bold_segments, apply_pose, create_simulation, draw_boundary, set_camera,
)

GRAPHITE = "#20262e"
PANEL_TEXT = "#edf3f5"
PANEL_MUTED = "#aebcc2"
WHITE = (1.00, 1.00, 1.00)
LIGHT_CYAN = (0.32, 0.86, 0.90)
DARK_TEAL = (0.00, 0.38, 0.39)
AMBER = (0.90, 0.62, 0.15)
RED = (0.92, 0.18, 0.14)
BASE_HEIGHT = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute_device_id", type=int, default=0)
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--candidate_count", type=int, default=4096)
    parser.add_argument("--validation_count", type=int, default=512)
    parser.add_argument("--box_validation_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--max_steps", type=int, default=0, help="0 keeps both windows interactive")
    parser.add_argument("--auto_sweep_steps", type=int, default=0)
    parser.add_argument("--compute_only", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--screenshot_step", type=int, default=5)
    return parser.parse_args()


def build_samples(kinematics, args):
    lower, upper = kinematics.joint_limits(soft_fraction=0.9)
    anchor = torch.tensor([0.0, 0.60, -0.60] * 6)
    half = torch.tensor([0.95, 0.42, 0.48] * 6)
    candidate_lower = torch.maximum(anchor - half, lower)
    candidate_upper = torch.minimum(anchor + half, upper)
    candidates = torch.cat((
        anchor.unsqueeze(0),
        deterministic_joint_samples(
            candidate_lower, candidate_upper, args.candidate_count, seed=args.seed,
        ),
    ))
    validation = deterministic_joint_samples(
        candidate_lower, candidate_upper, args.validation_count, seed=args.seed + 100,
    )
    return candidates, validation, lower, upper


def compute_result(kinematics, samples, values, args):
    candidates, validation, lower, upper = samples
    return compute_legacy_admissible_envelope(
        kinematics,
        parameter_tensor(values, dtype=candidates.dtype, device=candidates.device),
        candidates, validation, lower, upper,
        box_validation_samples=args.box_validation_samples,
        box_validation_seed=args.seed + 200,
    )


def result_summary(result) -> str:
    diagnostics = result.range_export.diagnostics
    return (
        f"feasible {int(result.feasible_mask.sum())}; "
        f"box violations {result.box_foot_violation_count}/{result.box_validation_samples}; "
        f"registered exclusions {diagnostics.violation_count}/{diagnostics.validation_samples}"
    )


class SliderPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EL4090 | ZhangHT Foot Envelope")
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
            self.root, text="ZHANGHT FOOT WORKSPACE", bg=GRAPHITE, fg=PANEL_TEXT,
            font=("TkDefaultFont", 15, "bold"), anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 2))
        tk.Label(
            self.root,
            text="White: legacy border   Cyan: sampled maximum   Teal: current feet",
            bg=GRAPHITE, fg=PANEL_MUTED, font=("TkDefaultFont", 9), anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 10))
        slider_frame = tk.Frame(self.root, bg=GRAPHITE)
        slider_frame.pack(fill="x", padx=18)
        for name, initial in zip(LEGACY_PARAMETER_ORDER, LEGACY_MAXIMUM):
            lower, upper = LEGACY_PARAMETER_RANGES[name]
            row = tk.Frame(slider_frame, bg=GRAPHITE)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=name.replace("_", " ").title(), width=17, anchor="w",
                bg=GRAPHITE, fg=PANEL_TEXT, font=("TkDefaultFont", 10),
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
                font=("TkFixedFont", 9), anchor="w", justify="left",
            ).pack(fill="x", padx=18, pady=2)
        self.status = tk.StringVar(value="Waiting for first sampled export")
        tk.Label(
            self.root, textvariable=self.status, bg=GRAPHITE, fg="#52dbe6",
            font=("TkDefaultFont", 9), anchor="w", justify="left", wraplength=570,
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
        lower = result.range_export.lower
        upper = result.range_export.upper
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
        self.status.set(f"No replacement applied: {message}")
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


def draw_haa(gym, viewer, env, kinematics, result):
    ranges = haa_ranges_from_joint_export(result.range_export)
    origins, arcs, markers = haa_arc_geometry(
        kinematics, result.current_q, ranges, radius=0.25, samples=41,
    )
    translation = np.array((0.0, 0.0, BASE_HEIGHT + 0.12), dtype=np.float32)
    origins = origins.detach().cpu().numpy() + translation
    arcs = arcs.detach().cpu().numpy() + translation
    markers = markers.detach().cpu().numpy() + translation
    for index in range(6):
        add_bold_segments(gym, viewer, env, polyline_segments(arcs[index]), AMBER)
        bounds = np.stack((origins[index], arcs[index, 0], origins[index], arcs[index, -1]))
        add_bold_segments(gym, viewer, env, bounds, AMBER)
        endpoint = origins[index] + 1.28 * (markers[index] - origins[index])
        add_bold_segments(gym, viewer, env, np.stack((origins[index], endpoint)), AMBER)


def draw_foot_markers(gym, viewer, env, feet_xy, color):
    size = 0.035
    segments = []
    for x, y in feet_xy.detach().cpu().numpy():
        segments.extend(((x - size, y, 0.125), (x + size, y, 0.125), (x, y - size, 0.125), (x, y + size, 0.125)))
    add_bold_segments(gym, viewer, env, np.asarray(segments), color)


def draw_scene(gym, viewer, env, kinematics, result):
    gym.clear_lines(viewer)
    draw_boundary(gym, viewer, env, result.border_vertices_xy.detach().cpu().numpy(), 0.050, WHITE)
    for hull in (result.maximal_rear_vertices_xy, result.maximal_front_vertices_xy):
        draw_boundary(gym, viewer, env, hull.detach().cpu().numpy(), 0.080, LIGHT_CYAN)
    parameters = parameter_tensor(result.parameters, dtype=result.current_feet_xy.dtype)
    violation = not bool(legacy_feasible(result.current_feet_xy.unsqueeze(0), parameters)[0])
    color = RED if violation else DARK_TEAL
    current_halves = (
        result.current_feet_xy[result.current_feet_xy[:, 0] <= 0.0],
        result.current_feet_xy[result.current_feet_xy[:, 0] >= 0.0],
    )
    for points in current_halves:
        if points.shape[0] >= 2:
            hull = convex_hull_2d(points)
            draw_boundary(
                gym, viewer, env, hull.detach().cpu().numpy(), 0.110, color,
            )
    draw_foot_markers(gym, viewer, env, result.current_feet_xy, color)
    draw_haa(gym, viewer, env, kinematics, result)


def write_evidence(gym, viewer, screenshot, result, history, step):
    screenshot = screenshot.resolve()
    try:
        screenshot.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("generated evidence must be outside extended_legged_gym")
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    gym.write_viewer_image_to_file(viewer, str(screenshot))
    export = result.range_export
    record = {
        "screenshot": str(screenshot), "step": step,
        "semantics": LEGACY_VISUAL_SEMANTICS,
        "maximality_scope": "separate rear/front convex hulls of registered feasible FK foot samples",
        "parameters": dict(zip(LEGACY_PARAMETER_ORDER, result.parameters)),
        "feasible_candidate_count": int(result.feasible_mask.sum()),
        "candidate_count": int(result.feasible_mask.numel()),
        "maximal_rear_vertices_xy": result.maximal_rear_vertices_xy.tolist(),
        "maximal_front_vertices_xy": result.maximal_front_vertices_xy.tolist(),
        "current_feet_xy": result.current_feet_xy.tolist(),
        "joint_order": list(EL4090_JOINT_NAMES),
        "joint_lower_rad": export.lower.tolist(),
        "joint_upper_rad": export.upper.tolist(),
        "registered_validation_count": export.diagnostics.validation_samples,
        "registered_false_exclusion_count": export.diagnostics.violation_count,
        "box_validation_samples": result.box_validation_samples,
        "box_foot_violation_count": result.box_foot_violation_count,
        "max_box_foot_violation_m": result.max_box_foot_violation_m,
        "recompute_history": history,
    }
    evidence = screenshot.with_suffix(".json")
    evidence.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Screenshot: {screenshot}")
    print(f"Evidence:   {evidence}")


def main():
    args = parse_args()
    if args.candidate_count < 1 or args.validation_count < 0:
        raise ValueError("candidate_count must be positive and validation_count nonnegative")
    urdf = PROJECT_ROOT / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    samples = build_samples(kinematics, args)
    result = compute_result(kinematics, samples, LEGACY_MAXIMUM, args)
    print("ZhangHT legacy foot-workspace export")
    print(f"  parameters: {dict(zip(LEGACY_PARAMETER_ORDER, result.parameters))}")
    print(f"  sampled maximality: separate rear/front hulls; {result_summary(result)}")
    if args.compute_only:
        return

    panel = SliderPanel()
    gym, sim, viewer, env, actor, q_indices = create_simulation(args, result.current_q)
    set_camera(gym, viewer, 0)
    history = []
    captured = False
    step = 0
    sweep_values = (LEGACY_MAXIMUM, LEGACY_MIDPOINT, (0.35, 0.36, 0.35, 0.65, -0.65))
    try:
        while panel.running and not gym.query_viewer_has_closed(viewer):
            panel.pump()
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    panel.running = False
                elif event.action == "camera":
                    set_camera(gym, viewer, 1)
                elif event.action == "capture":
                    panel.capture_requested = True
            if args.auto_sweep_steps > 0 and step > 0 and step % args.auto_sweep_steps == 0:
                panel.set_values(sweep_values[(step // args.auto_sweep_steps) % len(sweep_values)])
            if panel.pending():
                values = panel.snapshot()
                started = time.perf_counter()
                try:
                    updated = compute_result(kinematics, samples, values, args)
                except (ValueError, RuntimeError) as error:
                    panel.update_error(str(error))
                else:
                    elapsed_ms = 1000.0 * (time.perf_counter() - started)
                    result = updated
                    panel.update_result(result, elapsed_ms)
                    history.append({
                        "generation": panel.applied_generation,
                        "parameters": dict(zip(LEGACY_PARAMETER_ORDER, result.parameters)),
                        "feasible_candidate_count": int(result.feasible_mask.sum()),
                        "elapsed_ms": elapsed_ms,
                    })
                    print(f"  update {len(history)}: {result_summary(result)}; {elapsed_ms:.1f} ms")
            apply_pose(gym, env, actor, q_indices, result.current_q)
            draw_scene(gym, viewer, env, kinematics, result)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            if args.screenshot is not None and (panel.capture_requested or (not captured and step >= args.screenshot_step)):
                write_evidence(gym, viewer, args.screenshot, result, history, step)
                panel.capture_requested = False
                captured = True
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
