#!/usr/bin/env python3
"""Run the unchanged LiDAR envelope model with ZhangHT slider border points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk

from isaacgym import gymapi  # Must precede Torch imports for Isaac Gym Preview 4.
import torch

DISTRIBUTION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = DISTRIBUTION_ROOT.parent

from el4090_envelope import (
    EL4090_JOINT_NAMES, EL4090_LEG_NAMES, BatchedUrdfKinematics,
    capsule_support, default_el4090_capsules, deterministic_joint_samples,
    export_envelope_joint_ranges_at_reference, feasible_reference_q,
    haa_ranges_from_joint_export, joint_rejection_ranges, load_urdf_joints,
    reachable_foot_support, support_directions,
)
from legacy_slider_envelope import (
    LEGACY_MAXIMUM, LEGACY_MIDPOINT, LEGACY_PARAMETER_ORDER,
    LEGACY_PARAMETER_RANGES, legacy_border_lidar_cloud,
    legacy_border_support, legacy_border_vertices, parameter_tensor,
)
from lidar_free_envelope import (
    envelope_excess, maximum_sector_point_free_envelope,
    polygon_support_excess,
)
from pd_control import (
    ema_reference_update, nearest_outside_rejection, pd_integrate,
    pd_settled,
)
from visualize_lidar_free_envelope_gym import (
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
    parser.add_argument(
        "--qe", type=str, default=None,
        help="comma-separated initial q_e slider values (18; default all 0)",
    )
    # ENV-RECT-CONTROL-012: PD joint control and live rejection reference.
    parser.add_argument(
        "--legacy_motion", action="store_true",
        help="keep the old trajectory motion instead of the default PD joint control",
    )
    parser.add_argument("--pd_kp", type=float, default=64.0)
    parser.add_argument("--pd_kd", type=float, default=16.0)
    parser.add_argument("--pd_max_rate", type=float, default=3.0)
    parser.add_argument("--pd_dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--pd_alpha", type=float, default=0.5)
    parser.add_argument("--pd_settle_rad", type=float, default=2e-3)
    parser.add_argument(
        "--rejection_recompute_rad", type=float, default=0.03,
        help="recompute rejection with the live reference after the EMA joint "
             "moves by this much (throttled)",
    )
    parser.add_argument(
        "--rejection_min_interval_steps", type=int, default=20,
        help="minimum steps between throttled rejection recomputes",
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


def compute_result(
    kinematics, context, values, args, generation, reference_q=None,
    build_trajectory=True,
):
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
    # feasible reference. ENV-RECT-CONTROL-012: the preferred reference is the
    # live EMA joint when the caller provides one (``reference_q``), otherwise
    # the resting pose. ``feasible_reference_q`` validates it (resting -> box
    # center -> nearest feasible candidate sample), so the displayed ranges stay
    # the exact per-axis feasible reach even when the live joint drifts outside
    # the envelope.
    if reference_q is None:
        reference_q = context["baseline_q"]
    reference, _reference_source = feasible_reference_q(
        kinematics, context["capsules"], directions, free_envelope.support_m,
        context["lower"], context["upper"], reference_q,
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
    trajectory = None
    if build_trajectory and bool(export.valid) and resting_inside:
        try:
            trajectory = build_motion_trajectory(
                problem, kinematics, directions, args.motion_period_steps,
            )
        except (ValueError, RuntimeError):
            # Exported ranges are still shown even when no smooth motion fits;
            # the viewer then freezes at the resting pose.
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
        if result.trajectory is not None else "PD joint control"
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


# ---------------------------------------------------------------------------
# ENV-RECT-CONTROL-012: PD joint control + live (EMA) rejection reference.
# ---------------------------------------------------------------------------
#
# MasterYip: "joint control: use pd to let the joint reach calculated position
# q_c, the expect position is human-set through slider q_e, q_c is the nearest
# position to q_e but outside the rejection range. Default q_e is all 0. For
# rejection range computation: use the current joint (or ema to smooth) as ref
# r, use this r to calculate the rejection range."
#
# The pure PD helpers live in ``pd_control`` (no Isaac Gym dependency, unit
# tested). The rejection reference r is an EMA of the actual joint and is
# re-validated by feasible_reference_q (resting -> box center -> nearest
# feasible candidate) so the demo never crashes when the live joint drifts
# outside the envelope.


def envelope_feasible_q_c(
    q_e, q_c, rejected_intervals, kinematics, capsules, directions, allowed,
    lower, upper,
):
    """Snap q_c joints clamped off a rejected interval to an envelope-feasible value.

    ``nearest_outside_rejection`` moves a joint exactly onto the rejected
    interval endpoint, but those endpoints are half-sweep-step interpolations:
    the true feasibility boundary is up to half a grid step inside the interval,
    so the exact endpoint can sit a few millimeters outside the envelope. This
    nudges every moved joint one sweep step further away from its interval until
    the full pose is envelope-feasible (the first feasible grid value). Joints
    that were not clamped (q_c == q_e) are never touched, and when the endpoint
    pose is already feasible the value is returned unchanged.
    """
    moved = (q_c != q_e)
    if not bool(moved.any()):
        return q_c
    step = (upper - lower) / 100.0
    current = q_c.clone()
    best = current.clone()
    best_excess = float(envelope_excess(
        kinematics, current.unsqueeze(0), capsules, directions, allowed,
    )[0])
    if best_excess <= TOLERANCE:
        return current
    # The true boundary is within one sweep step of the interpolated endpoint,
    # so only a few nudges should be needed. Track the minimum-excess pose so a
    # reference mismatch (fallback) can never push the joint to a URDF limit.
    for _ in range(6):
        nudged = current.clone()
        for joint in moved.nonzero().flatten().tolist():
            lo = hi = None
            for a, b in rejected_intervals[joint]:
                if a <= float(q_e[joint]) <= b:
                    lo, hi = a, b
                    break
            if lo is None:
                continue
            # ``nearest_outside_rejection`` moved the joint to the nearer
            # endpoint; nudge one sweep step further AWAY from the interval.
            # The direction is derived from q_e (not from comparing the float32
            # tensor value to the float64 endpoint, which drifts by ~3e-8).
            if (float(q_e[joint]) - lo) <= (hi - float(q_e[joint])):
                nudged[joint] = current[joint] - float(step[joint])
            else:
                nudged[joint] = current[joint] + float(step[joint])
        nudged = torch.maximum(lower, torch.minimum(upper, nudged))
        excess = float(envelope_excess(
            kinematics, nudged.unsqueeze(0), capsules, directions, allowed,
        )[0])
        if excess <= TOLERANCE:
            return nudged
        if excess < best_excess:
            best, best_excess = nudged, excess
        current = nudged
    return best


def recompute_ranges_at_reference(
    result, kinematics, context, reference, args, generation,
):
    """Recompute a slider result's export and rejection with the live reference.

    The envelope (cloud + free support polygon) does not depend on the joint
    reference, so only the range export, HAA ranges, and rejection intervals are
    refreshed. ``reference`` is the preferred live (EMA) joint; it is validated
    by ``joint_rejection_ranges`` / ``feasible_reference_q`` and falls back to
    the box center or nearest feasible candidate when the live joint drifts
    outside the envelope, so the demo stays robust.
    """
    problem = result.problem
    directions = context["directions"]
    allowed = problem.free_envelope.support_m
    # ENV-RECT-CONTROL-012: validate the live EMA reference. When it drifts
    # outside the envelope, ``feasible_reference_q`` would fall back to the box
    # center, whose rejection bands can be inconsistent with the pose actually
    # being controlled. Prefer the previous live reference (still near the
    # actual pose) when it remains envelope-feasible.
    validated, source = feasible_reference_q(
        kinematics, context["capsules"], directions, allowed,
        context["lower"], context["upper"], reference,
        tolerance=TOLERANCE, fallback_candidates=context["candidate_q"],
    )
    previous = (
        problem.rejection_ranges.reference_q
        if problem.rejection_ranges is not None
        and problem.rejection_ranges.feasible_reference
        and problem.rejection_ranges.reference_q is not None
        else None
    )
    if (
        validated is None and previous is not None
        and bool((capsule_support(
            kinematics, previous.unsqueeze(0), context["capsules"], directions,
        )[0] <= allowed + TOLERANCE).all())
    ):
        validated, source = previous, "reference_q (previous live EMA pose)"
    if validated is None:
        reference = context["lower"]
    else:
        reference = validated
    rejection = joint_rejection_ranges(
        kinematics, context["capsules"], directions, allowed,
        context["lower"], context["upper"], reference,
        tolerance=TOLERANCE, fallback_candidates=context["candidate_q"],
    )
    if rejection.feasible_reference and rejection.reference_q is not None:
        export_reference = rejection.reference_q
    else:
        export_reference = context["lower"]
    export = export_envelope_joint_ranges_at_reference(
        kinematics, context["capsules"], directions, allowed,
        context["lower"], context["upper"], export_reference,
        tolerance=TOLERANCE,
        box_validation_samples=args.box_validation_samples,
        box_validation_seed=args.seed + 300,
    )
    new_problem = replace(
        problem,
        range_export=export,
        haa_ranges=haa_ranges_from_joint_export(export),
        rejection_ranges=rejection,
    )
    return replace(result, problem=new_problem)


def new_pd_stats() -> dict:
    return {
        "frame_count": 0,
        "settled_frames": 0,
        "max_envelope_excess_m": 0.0,
        "max_joint_limit_excess_rad": 0.0,
        "settled_max_envelope_excess_m": 0.0,
    }


def update_pd_stats(
    pd_stats, kinematics, capsules, directions, problem, q, settled,
) -> None:
    """Record PD-frame occupancy without raising on transient violations.

    The settled pose is checked separately: a settled pose that still exceeds
    the envelope would indicate that the rejection reference and the actual
    pose disagree, which the EMA lag can cause briefly; it is reported, never
    fatal.
    """
    pd_stats["frame_count"] += 1
    pd_stats["settled_frames"] += int(settled)
    excess = float(envelope_excess(
        kinematics, q.unsqueeze(0), capsules, directions,
        problem.free_envelope.support_m,
    )[0])
    joint_excess = float(torch.maximum(
        problem.range_export.lower - q,
        q - problem.range_export.upper,
    ).clamp_min(0.0).max())
    pd_stats["max_envelope_excess_m"] = max(
        pd_stats["max_envelope_excess_m"], max(0.0, excess),
    )
    pd_stats["max_joint_limit_excess_rad"] = max(
        pd_stats["max_joint_limit_excess_rad"], joint_excess,
    )
    if settled:
        pd_stats["settled_max_envelope_excess_m"] = max(
            pd_stats["settled_max_envelope_excess_m"], max(0.0, excess),
        )


class SliderPanel:
    def __init__(
        self, show_rejection=False, joint_lower=None, joint_upper=None,
        initial_qe=None,
    ):
        self.root = tk.Tk()
        self.root.title("EL4090 | ZhangHT Border Point Source")
        self.root.geometry("1240x1080+32+32")
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
        self.qe_variables = {}
        self._joint_lower = joint_lower
        self._joint_upper = joint_upper
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
            self.root,
            text="JOINT TARGETS q_e [rad]  (PD drives the actual joint q to q_c = "
                 "the nearest value to q_e outside the rejected intervals)",
            bg=GRAPHITE, fg=PANEL_TEXT, font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(3, 4))
        joint_columns = tk.Frame(self.root, bg=GRAPHITE)
        joint_columns.pack(fill="x", padx=18)
        for column in range(2):
            col = tk.Frame(joint_columns, bg=GRAPHITE)
            col.pack(side="left", fill="x", expand=True, padx=(0, 10))
            for row in range(9):
                joint = EL4090_JOINT_NAMES[column * 9 + row]
                jrow = tk.Frame(col, bg=GRAPHITE)
                jrow.pack(fill="x", pady=1)
                tk.Label(
                    jrow, text=joint, width=7, anchor="w", bg=GRAPHITE,
                    fg=PANEL_TEXT, font=("TkFixedFont", 8),
                ).pack(side="left")
                initial = (
                    float(initial_qe[column * 9 + row])
                    if initial_qe is not None else 0.0
                )
                variable = tk.DoubleVar(value=initial)
                self.qe_variables[joint] = variable
                qe_scale = tk.Scale(
                    jrow, variable=variable, from_=-3.0, to=3.0, resolution=0.01,
                    orient="horizontal", length=160, showvalue=True, digits=3,
                    command=self._qe_slider_changed, bg=GRAPHITE, fg=PANEL_TEXT,
                    troughcolor="#46525b", activebackground="#52dbe6",
                    highlightthickness=0, bd=0, font=("TkFixedFont", 8),
                )
                qe_scale.pack(side="left", fill="x", expand=True)
        tk.Frame(self.root, height=1, bg="#46525b").pack(fill="x", padx=18, pady=6)
        self.pd_readout = tk.StringVar(value="PD: q_e -> q_c readout appears after the first export")
        tk.Label(
            self.root, textvariable=self.pd_readout, bg=GRAPHITE, fg=PANEL_TEXT,
            font=("TkFixedFont", 8), anchor="w", justify="left",
            wraplength=1200,
        ).pack(fill="x", padx=18, pady=(2, 6))
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
            # Tk Scale widgets can fire their -command when the panel re-lays
            # out after new widgets are packed, with the value unchanged. Only
            # queue a recompute when a border value actually changed.
            current = self.snapshot()
            if (
                self._last_result is None
                or tuple(self._last_result.parameters) != current
            ):
                self.enqueue(immediate=False)

    def _slider_released(self, _event=None):
        self.enqueue(immediate=True)

    def _qe_slider_changed(self, _value=None):
        # The q_e sliders set the PD target; they never change the envelope, so
        # they only refresh the live q_e -> q_c readout instead of queueing a
        # border recompute.
        if not self.suspend:
            self.refresh_pd_readout()

    def qe_snapshot(self):
        return tuple(self.qe_variables[name].get() for name in EL4090_JOINT_NAMES)

    def refresh_pd_readout(self):
        if self._last_result is None or self._joint_lower is None:
            return
        qe = torch.tensor(self.qe_snapshot(), dtype=torch.float32)
        rejection = self._last_result.problem.rejection_ranges
        intervals = (
            rejection.rejected_intervals if rejection is not None else None
        )
        if intervals is None:
            qc = torch.clamp(qe, self._joint_lower, self._joint_upper)
        else:
            qc = nearest_outside_rejection(
                qe, intervals, self._joint_lower, self._joint_upper,
            )
        lines = []
        for block in range(3):
            qe_values = ", ".join(f"{float(qe[j]):+.2f}" for j in range(block * 6, block * 6 + 6))
            qc_values = ", ".join(f"{float(qc[j]):+.2f}" for j in range(block * 6, block * 6 + 6))
            names = " ".join(f"{EL4090_JOINT_NAMES[j]:>7}" for j in range(block * 6, block * 6 + 6))
            lines.append(names)
            lines.append(f"q_e [{qe_values}]")
            lines.append(f"q_c [{qc_values}]")
        if rejection is not None and rejection.feasible_reference:
            rejected_names = [
                EL4090_JOINT_NAMES[j]
                for j in range(len(rejection.rejected_intervals))
                if rejection.rejected_intervals[j]
            ]
            lines.append(
                "rejection reference: "
                f"{rejection.reference_source}; {len(rejected_names)} joints "
                f"rejected: {', '.join(rejected_names) if rejected_names else 'none'}"
            )
        else:
            lines.append("rejection: not applicable (no feasible reference)")
        self.pd_readout.set("\n".join(lines))

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
        self.refresh_pd_readout()
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


def write_evidence(
    gym, viewer, screenshot, result, state, stats, history, step,
    pd_state=None,
):
    write_lidar_evidence(
        gym, viewer, screenshot, result.problem, state["directions"],
        state, stats, step,
    )
    path = screenshot.resolve().with_suffix(".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["point_source"] = "ZhangHT slider rectangle border support points"
    record["cloud"]["angular_coverage"] = (
        "one ZhangHT-border-support return in every registered sector"
    )
    record["cloud"]["structure"]["randomization"] = (
        "none; deterministic border support points controlled by the five sliders"
    )
    record["envelope_math_model"] = result.problem.free_envelope.optimality_scope
    record["slider_parameters"] = dict(zip(LEGACY_PARAMETER_ORDER, result.parameters))
    record["slider_border_vertices_xy"] = result.border_vertices_xy.tolist()
    record["recompute_history"] = history
    if pd_state is not None:
        record["pd_control"] = {
            "mode": "pd",
            "kp": pd_state["kp"],
            "kd": pd_state["kd"],
            "dt": pd_state["dt"],
            "max_rate_rad_s": pd_state["max_rate"],
            "ema_alpha": pd_state["alpha"],
            "settle_rad": pd_state["settle_rad"],
            "q_e_rad": pd_state["q_e"].detach().cpu().tolist(),
            "q_c_rad": pd_state["q_c"].detach().cpu().tolist(),
            "q_actual_rad": pd_state["q"].detach().cpu().tolist(),
            "q_dot_rad_s": pd_state["q_dot"].detach().cpu().tolist(),
            "rejection_reference_q_rad": pd_state["reference_r"].detach().cpu().tolist(),
            "rejection_reference_source": (
                result.problem.rejection_ranges.reference_source
                if result.problem.rejection_ranges is not None else None
            ),
            "settled": bool(pd_state["settled"]),
            "settled_max_envelope_excess_m": pd_state["stats"]["settled_max_envelope_excess_m"],
            "max_envelope_excess_m": pd_state["stats"]["max_envelope_excess_m"],
            "max_joint_limit_excess_rad": pd_state["stats"]["max_joint_limit_excess_rad"],
            "frame_count": pd_state["stats"]["frame_count"],
            "settled_frames": pd_state["stats"]["settled_frames"],
        }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.directions < 8 or args.candidate_count < 1 or args.validation_count < 0:
        raise ValueError("invalid direction or sample count")
    if args.reference_containment_margin <= 0.0:
        raise ValueError("reference containment margin must be positive")
    if args.motion_period_steps < 2 or args.point_clearance <= 0.0:
        raise ValueError("motion period must be at least 2 and point clearance positive")
    urdf = REPOSITORY_ROOT / "legged_gym" / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
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

    initial_qe = None
    if args.qe is not None:
        raw = [float(value) for value in args.qe.split(",")]
        if len(raw) != 18:
            raise ValueError("--qe requires exactly 18 comma-separated values")
        initial_qe = raw
    panel = SliderPanel(
        show_rejection=args.show_rejection,
        joint_lower=context["lower"], joint_upper=context["upper"],
        initial_qe=initial_qe,
    )
    gym, sim, viewer, env, actor, q_indices = create_simulation(args, result.problem.baseline_q)
    state = {
        "motion": not args.no_motion, "motion_step": 0,
        "lidar": True, "prescribed": True, "occupied": True,
        "haa": True, "rejection": args.show_rejection,
        "reachable": True, "camera": 0,
        "directions": context["directions"],
    }
    set_camera(gym, viewer, 0)
    history = []
    captured = False
    step = 0
    # ENV-RECT-CONTROL-012: the default control is PD joint reaching. The old
    # trajectory motion stays available behind --legacy_motion.
    pd_state = None
    if not args.legacy_motion:
        pd_state = {
            "kp": args.pd_kp, "kd": args.pd_kd, "dt": args.pd_dt,
            "max_rate": args.pd_max_rate, "alpha": args.pd_alpha,
            "settle_rad": args.pd_settle_rad,
            "q": context["baseline_q"].clone(),
            "q_dot": torch.zeros_like(context["baseline_q"]),
            "reference_r": context["baseline_q"].clone(),
            "reference_r_at_rejection": context["baseline_q"].clone(),
            "q_e": torch.zeros_like(context["baseline_q"]),
            "q_c": torch.zeros_like(context["baseline_q"]),
            "settled": False,
            "last_recompute_step": -args.rejection_min_interval_steps,
            "stats": new_pd_stats(),
        }
        stats = pd_state["stats"]
        build_trajectory = False
    else:
        stats = new_stats(result.trajectory)
        build_trajectory = True
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
            reference_q=(pd_state["reference_r"] if pd_state is not None else None),
            build_trajectory=build_trajectory,
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
                        kinematics, context, values, args,
                        panel.requested_generation,
                        reference_q=(
                            pd_state["reference_r"] if pd_state is not None else None
                        ),
                        build_trajectory=build_trajectory,
                    )
                except (ValueError, RuntimeError) as error:
                    panel.update_error(str(error))
                else:
                    elapsed_ms = 1000.0 * (time.perf_counter() - started)
                    result = updated
                    state["motion_step"] = 0
                    if pd_state is not None:
                        # The border/envelope changed; the PD continues from the
                        # current joint but the rejection reference must be
                        # re-validated against the fresh envelope.
                        pd_state["reference_r_at_rejection"] = (
                            pd_state["reference_r"].clone()
                        )
                        pd_state["last_recompute_step"] = step
                    else:
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
            if pd_state is not None:
                # ENV-RECT-CONTROL-012: PD joint control. q_e comes from the 18
                # sliders, q_c is the nearest value outside the rejected
                # intervals, and the rejection reference is the EMA of the actual
                # joint, recomputed on a throttle when the reference has moved.
                pd_state["q_e"] = torch.tensor(
                    panel.qe_snapshot(), dtype=context["baseline_q"].dtype,
                )
                pd_state["reference_r"] = ema_reference_update(
                    pd_state["q"], pd_state["reference_r"], args.pd_alpha,
                )
                reference_moved = float((
                    pd_state["reference_r"] - pd_state["reference_r_at_rejection"]
                ).abs().max())
                if (
                    reference_moved > args.rejection_recompute_rad
                    and step - pd_state["last_recompute_step"]
                    >= args.rejection_min_interval_steps
                ):
                    started = time.perf_counter()
                    result = recompute_ranges_at_reference(
                        result, kinematics, context,
                        pd_state["reference_r"], args,
                        panel.requested_generation,
                    )
                    pd_state["last_recompute_step"] = step
                    pd_state["reference_r_at_rejection"] = (
                        pd_state["reference_r"].clone()
                    )
                    panel._last_result = result
                    panel._render_range_labels(result)
                    panel.refresh_pd_readout()
                    print(
                        "  rejection reference updated (EMA moved "
                        f"{reference_moved:.4f} rad) in "
                        f"{1000.0 * (time.perf_counter() - started):.0f} ms"
                    )
                intervals = result.problem.rejection_ranges.rejected_intervals
                pd_state["q_c"] = nearest_outside_rejection(
                    pd_state["q_e"], intervals,
                    context["lower"], context["upper"],
                )
                pd_state["q_c"] = envelope_feasible_q_c(
                    pd_state["q_e"], pd_state["q_c"], intervals,
                    kinematics, context["capsules"], context["directions"],
                    result.problem.free_envelope.support_m,
                    context["lower"], context["upper"],
                )
                pd_state["q"], pd_state["q_dot"] = pd_integrate(
                    pd_state["q"], pd_state["q_dot"], pd_state["q_c"],
                    kp=args.pd_kp, kd=args.pd_kd, dt=args.pd_dt,
                    max_rate=args.pd_max_rate,
                    lower=context["lower"], upper=context["upper"],
                )
                pd_state["settled"] = pd_settled(
                    pd_state["q"], pd_state["q_c"], pd_state["q_dot"],
                    args.pd_settle_rad,
                )
                if pd_state["settled"]:
                    pd_state["q"] = pd_state["q_c"].clone()
                    pd_state["q_dot"] = torch.zeros_like(pd_state["q_dot"])
                update_pd_stats(
                    pd_state["stats"], kinematics, context["capsules"],
                    context["directions"], result.problem, pd_state["q"],
                    pd_state["settled"],
                )
                pose = pd_state["q"]
            elif result.trajectory is not None:
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
                write_evidence(
                    gym, viewer, args.screenshot, result, state, stats,
                    history, step, pd_state=pd_state,
                )
                panel.capture_requested = False
                captured = True
            if pd_state is None and result.trajectory is not None and state["motion"]:
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
