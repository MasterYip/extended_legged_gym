#!/usr/bin/env python3
"""EL4090 Occupied Body Support Envelope -- interactive web explorer server.

A stdlib-only ``ThreadingHTTPServer`` bound to ``127.0.0.1`` that serves the
ENV-VIS-015 frontend and exposes compute endpoints. All model math comes from
the installed ``el4090_envelope`` package (or this source checkout's ``src``
tree), so the web example contains no duplicate mathematical implementation.

Endpoints
---------
* ``GET  /``                 -> ``index.html`` (plus ``/app.js``, ``/styles.css``)
* ``GET  /api/health``       -> model/runtime record
* ``POST /api/envelope``     -> Mode A: pose ``q[18]`` -> occupied envelope
* ``POST /api/rejection``    -> Mode B: 5-parameter envelope -> per-joint
                                rejection ranges (the headline mode)
* ``POST /api/haa_rejection`` -> Mode C: 5-parameter envelope + EXACT HFE/KFE
                                pins -> per-leg HAA rejection ranges (the HAA
                                band moves as the HFE/KFE sliders are dragged)

Mode B math
-----------
Given ``V={(x_f,±w_f),(0,±w_m),(x_b,±w_b)}`` the allowed half-space cap is

    h_k^allowed = max_{v in V} u_k^T v  +  margin,

then at a validated feasible reference ``r`` (resting -> box center -> nearest
feasible candidate) the per-joint rejected sub-intervals are

    R_j(r) = { q_j in [l_j,u_j] : g((q_j, r_-j); h^allowed) > tau },
    g(q; h) = max_k ( h_k^occ(q) - h_k ),

over the full URDF box with other joints pinned (ENV-REJECT-006 /
ENV-RANGE-REJECT-010 semantics; reference-pinned per ENV-MATH-LIMITS-011).  The
per-axis accessible box is the same sweep's feasible projection
(``export_envelope_joint_ranges_at_reference``).  When no feasible reference
exists the response is an honest ``{feasible_reference:false, reason}``.

Run
---
    /home/user/miniforge3/envs/isaacgym/bin/python envelope_server.py [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

# --- locate the package and default robot asset -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGGED = Path(os.environ.get("LEGGED_GYM_ROOT", _REPO_ROOT / "legged_gym"))
_URDF = Path(os.environ.get(
    "EL4090_URDF", _LEGGED / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"))
_TASK_DIR = Path(__file__).resolve().parent

_SOURCE_ROOT = _PROJECT_ROOT / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from el4090_envelope.geometry import (  # noqa: E402
    accessible_interval_complement,
    haa_arc_geometry,
    joint_arc_geometry_interval,
    support_polygon,
)
from el4090_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES,
    EL4090_LEG_NAMES,
    BatchedUrdfKinematics,
    capsule_support,
    default_el4090_capsules,
    default_el4090_torso_capsules,
    deterministic_joint_samples,
    export_envelope_joint_ranges_at_reference,
    feasible_reference_q,
    foot_positions,
    haa_ranges_from_joint_export,
    haa_rejection_ranges,
    joint_rejection_ranges,
    legacy_condition_from_support,
    load_urdf_joints,
    reachable_foot_support,
    support_directions,
)

# --- constants --------------------------------------------------------------

JOINT_INDEX = {name: i for i, name in enumerate(EL4090_JOINT_NAMES)}
LEG_INDEX = {leg: i for i, leg in enumerate(EL4090_LEG_NAMES)}
JOINT_KINDS = ("HAA", "HFE", "KFE")

_REACH_SAMPLES = 256          # reachable-foot overlay sample count (URDF box)
_REACH_SEED = 4090            # reachable-foot overlay Sobol seed
_CANDIDATE_SAMPLES = 257      # Mode B fallback reference candidates
_CANDIDATE_SEED = 4090        # Mode B fallback Sobol seed
_ARC_RADIUS = 0.22            # stylized foot-direction arc radius (m)
_ARC_SAMPLES = 17             # arc polyline / true-foot-trajectory resolution
_HAA_ARC_SAMPLES = 33
_LEG_ANIM_FRAMES = 9          # per-rejected-interval swept-leg skeleton keyframes

# --- model wiring (module-level, read-only) ---------------------------------

_kine = BatchedUrdfKinematics(load_urdf_joints(_URDF))
_CAPSULE_FULL = tuple(default_el4090_capsules())
_CAPSULE_TORSO = tuple(default_el4090_torso_capsules())
_reach_cache = {}


def _capsules(capsule_set: str):
    return _CAPSULE_TORSO if capsule_set == "torso" else _CAPSULE_FULL


def capsule_identity(capsule) -> tuple:
    """Return (leg_or_base, part) -- the binding body-part label."""
    name = capsule.name
    if name == "base_x":
        return ("BASE", "base")
    leg = name.split("_")[0]
    part = name.split("_", 1)[1]
    return (leg, part)


# --- 3D geometry helpers (shared Mode A / Mode B) ----------------------------

def _skeleton(q_t) -> dict:
    """Joint skeleton at pose ``q_t [1,18]``: base marker + per-leg 4-point
    polyline HAA origin -> HFE origin -> KFE origin -> foot.

    ``haa`` = the leg's HAA joint origin (``*_HIP`` link frame origin);
    ``hfe`` = the HFE joint origin (``*_THIGH`` link frame origin);
    ``kfe`` = the KFE joint origin (``*_SHANK`` link frame origin);
    ``foot`` = the ``*_FOOT`` link origin.  All in the body-yaw world frame,
    same FK as the capsule endpoints so the skeleton is always consistent
    with the drawn proxies.  The three segments are HAA-HFE, HFE-KFE,
    KFE-Foot (exactly three per leg; no base->hip bone, no direct hip->foot).
    """
    local = q_t.new_zeros((1, 1, 3))
    legs = []
    for leg in EL4090_LEG_NAMES:
        haa = _kine.points(q_t, (f"{leg}_HIP",), local)[0, 0, 0]
        hfe = _kine.points(q_t, (f"{leg}_THIGH",), local)[0, 0, 0]
        kfe = _kine.points(q_t, (f"{leg}_SHANK",), local)[0, 0, 0]
        foot = _kine.points(q_t, (f"{leg}_FOOT",), local)[0, 0, 0]
        legs.append({
            "leg": leg,
            "haa": _to_list(haa),
            "hfe": _to_list(hfe),
            "kfe": _to_list(kfe),
            "foot": _to_list(foot),
        })
    base = _kine.points(q_t, ("BASE",), local)[0, 0, 0]
    return {"base": _to_list(base), "legs": legs}


def _capsule_geometry_payload(capsules, q_t) -> dict:
    """Full 3D capsule proxy geometry (world-space) at pose ``q_t [1,18]``."""
    links = [c.link for c in capsules]
    local = q_t.new_tensor([[c.start, c.end] for c in capsules])   # [L,2,3]
    endpoints = _kine.points(q_t, links, local)[0]                 # [L,2,3]
    return {
        "capsule_names": [c.name for c in capsules],
        "capsule_links": [c.link for c in capsules],
        "capsule_radius": [c.radius for c in capsules],
        "capsules_3d": _to_list(endpoints),
    }


def _sweep_for_interval(chosen, leg_i, joint_kind, lo, hi, samples):
    """Sweep poses for one rejected sub-interval (matches
    ``joint_arc_geometry_interval`` exactly: only the named joint moves, all
    other joints pinned at ``chosen``). Returns ``(sweep_q [samples,18], alpha
    [samples])`` with ``alpha = linspace(0,1,samples)``.
    """
    joint_index = EL4090_JOINT_NAMES.index(
        f"{EL4090_LEG_NAMES[leg_i]}_{joint_kind}")
    alpha = torch.linspace(0.0, 1.0, samples, dtype=chosen.dtype,
                           device=chosen.device)
    sweep_q = chosen.repeat(samples, 1)
    sweep_q[:, joint_index] = lo + alpha * (hi - lo)
    return sweep_q, alpha


def _swept_leg_frames(chosen, leg_i, joint_kind, lo, hi) -> dict:
    """Per-rejected-interval animation keyframes for the swept leg.

    Returns ``sweep`` (rad values at each keyframe), ``trajectory`` (true foot
    world positions ``[N,3]`` along the interval -- the FK foot path, not the
    stylized arc) and ``leg_skeleton_frames`` (HAA/HFE/KFE/foot world
    positions ``[M,4,3]`` at a reduced keyframe resolution -- the same 4-point
    skeleton as ``_skeleton``). All FK, truthful to the model.
    """
    sweep_q, alpha = _sweep_for_interval(chosen, leg_i, joint_kind,
                                         float(lo), float(hi), _ARC_SAMPLES)
    foot_traj = foot_positions(_kine, sweep_q)[:, leg_i]            # [N,3]
    frame_q, _ = _sweep_for_interval(chosen, leg_i, joint_kind,
                                     float(lo), float(hi), _LEG_ANIM_FRAMES)
    local = frame_q.new_zeros((1, 1, 3))
    leg = EL4090_LEG_NAMES[leg_i]
    haa = _kine.points(frame_q, (f"{leg}_HIP",), local)[..., 0, 0, :]
    hfe = _kine.points(frame_q, (f"{leg}_THIGH",), local)[..., 0, 0, :]
    kfe = _kine.points(frame_q, (f"{leg}_SHANK",), local)[..., 0, 0, :]
    foot = _kine.points(frame_q, (f"{leg}_FOOT",), local)[..., 0, 0, :]
    frames = torch.stack((haa, hfe, kfe, foot), dim=1)              # [M,4,3]
    return {
        "sweep": _to_list(lo + alpha * (hi - lo)),
        "trajectory": _to_list(foot_traj),
        "leg_skeleton_frames": _to_list(frames),
    }


# --- presets (mirror ENV-VIS-014) -------------------------------------------

def _pose(**kwargs):
    q = np.zeros(len(EL4090_JOINT_NAMES))
    for name, value in kwargs.items():
        q[JOINT_INDEX[name]] = value
    return q


PRESETS = {
    "spider": {
        "q": _pose(**{f"{leg}_HFE": 0.6 for leg in EL4090_LEG_NAMES},
                   **{f"{leg}_KFE": -0.6 for leg in EL4090_LEG_NAMES}),
        "margin": 0.0,
    },
    "mammal": {
        "q": _pose(
            RF_HAA=-1.308, RM_HAA=1.308, RB_HAA=1.308,
            LF_HAA=-1.308, LM_HAA=1.308, LB_HAA=1.308,
            **{f"{leg}_HFE": 1.0 for leg in EL4090_LEG_NAMES},
            **{f"{leg}_KFE": -0.608 for leg in EL4090_LEG_NAMES}),
        "margin": 0.045,
    },
    "wide-low": {
        "q": _pose(
            RF_HAA=-0.9, RM_HAA=0.9, RB_HAA=0.9,
            LF_HAA=-0.9, LM_HAA=0.9, LB_HAA=0.9,
            **{f"{leg}_HFE": 1.2 for leg in EL4090_LEG_NAMES},
            **{f"{leg}_KFE": -1.2 for leg in EL4090_LEG_NAMES}),
        "margin": 0.185,
    },
    "tuck": {
        "q": _pose(**{f"{leg}_HFE": 2.2 for leg in EL4090_LEG_NAMES},
                   **{f"{leg}_KFE": -2.2 for leg in EL4090_LEG_NAMES}),
        "margin": 0.0,
    },
    "zero": {"q": _pose(), "margin": 0.0},
}

# ENV-RANGE-REJECT-010 / ENV-VIS-014 resting pose: [0.0, +0.60, -0.60] per leg
RESTING_Q = [0.0, 0.60, -0.60] * 6

# Mode B default envelope: just large enough that the resting spider pose
# stays feasible (reference source = "reference_q") while still rejecting the
# KFE wings and the lateral HAA bands -- a legible first-load headline.  The
# symmetric legacy projection (w~0.55, x_f~0.60, x_b~-0.57) does NOT contain
# the resting pose (the asymmetric occupied shape reaches h_occ=0.721 m), so
# the default is deliberately larger than that projection.
DEFAULT_ENVELOPE = {"w_f": 0.70, "w_m": 0.70, "w_b": 0.70, "x_f": 0.75, "x_b": -0.72}
DEFAULT_MARGIN_B = 0.02


def _to_list(value):
    return value.detach().cpu().numpy().tolist()


def _clean(obj):
    """Replace non-JSON-safe values (NaN/inf/bytes) with None recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, tuple):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return _clean(obj.detach().cpu().tolist())
    return obj


# --- Mode A: pose -> envelope -----------------------------------------------

def _reachable_polygon(k: int):
    """Support polygon of FK feet over the full-URDF-box Sobol sample set.

    Cached per ``k`` (independent of pose / capsule set), exactly like the
    ENV-VIS-014 explorer.  A finite-sample support, not a reachability cert.
    """
    cached = _reach_cache.get(k)
    if cached is None:
        lower, upper = _kine.joint_limits(soft_fraction=1.0)
        samples = deterministic_joint_samples(lower, upper, _REACH_SAMPLES, seed=_REACH_SEED)
        directions = support_directions(k)
        h_foot = reachable_foot_support(_kine, samples.unsqueeze(0), directions)[0]
        poly = support_polygon(directions, _to_list(h_foot))
        cached = (poly, _to_list(h_foot))
        _reach_cache[k] = cached
    return cached


def _edge_bindings(polygon, directions, support, binding):
    """Binding capsule index per polygon edge (the ENV-VIS-014 closure test)."""
    poly = np.asarray(polygon)
    support = np.asarray(support)
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1 = poly[i], poly[(i + 1) % n]
        mid = 0.5 * (p0 + p1)
        hit = None
        for k in range(len(directions)):
            val = float(directions[k] @ mid) - support[k]
            if abs(val) < 1e-6:
                hit = int(binding[k])
                break
        out.append(hit)
    return out


def compute_pose(payload: dict) -> dict:
    """Exact FK-only preview for live joint-slider interaction.

    This intentionally excludes support, envelope, reachability, and rejection
    computation. The committed endpoints remain the sole source for derived
    envelope and rejection state.
    """
    q = np.asarray(payload.get("q"), dtype=np.float64)
    if q.shape != (len(EL4090_JOINT_NAMES),):
        raise ValueError("q must be a length-18 joint vector")
    capsule_set = str(payload.get("capsule_set", "full"))
    if capsule_set not in ("full", "torso"):
        raise ValueError("capsule_set must be 'full' or 'torso'")

    capsules = _capsules(capsule_set)
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    geometry = _capsule_geometry_payload(capsules, q_t)
    feet = foot_positions(_kine, q_t)[0]
    return {
        "q": [float(v) for v in q],
        "capsule_set": capsule_set,
        "n_capsules": int(len(capsules)),
        **geometry,
        "skeleton": _skeleton(q_t),
        "feet": _to_list(feet),
    }


def compute_envelope(payload: dict) -> dict:
    """Mode A: pose -> occupied envelope ``P_K(q)`` + overlays."""
    q = np.asarray(payload.get("q"), dtype=np.float64)
    if q.shape != (len(EL4090_JOINT_NAMES),):
        raise ValueError("q must be a length-18 joint vector")
    margin = float(payload.get("margin", 0.0))
    K = int(payload.get("K", 32))
    capsule_set = str(payload.get("capsule_set", "full"))
    show_reachable = bool(payload.get("show_reachable", True))
    show_legacy = bool(payload.get("show_legacy", True))

    capsules = _capsules(capsule_set)
    directions = support_directions(K)
    dirs_np = directions.detach().cpu().numpy()
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)  # [1,18]

    h_occ = capsule_support(_kine, q_t, capsules, directions)[0]

    links = [c.link for c in capsules]
    local = q_t.new_tensor([[c.start, c.end] for c in capsules])  # [L,2,3]
    endpoints = _kine.points(q_t, links, local)[0]                # [L,2,3]
    proj = torch.einsum("lpi,ki->lpk", endpoints[..., :2], directions)
    radii = q_t.new_tensor([c.radius for c in capsules]).unsqueeze(-1)
    per_capsule = proj.max(dim=1).values + radii                 # [L,K]
    binding = per_capsule.argmax(dim=0)                          # [K]

    h_margin = h_occ + margin
    poly = support_polygon(directions, _to_list(h_occ))
    poly_margin = support_polygon(directions, _to_list(h_margin))
    feet = foot_positions(_kine, q_t)[0]                          # [6,3]

    cond = legacy_condition_from_support(
        h_occ.unsqueeze(0), directions, clamp_to_training_ranges=False).squeeze(0)
    cond_clamped = legacy_condition_from_support(
        h_occ.unsqueeze(0), directions).squeeze(0)
    cond_l = _to_list(cond)
    cond_c_l = _to_list(cond_clamped)

    card_targets = {"+x": np.array([1.0, 0.0]), "+y": np.array([0.0, 1.0]),
                    "-x": np.array([-1.0, 0.0]), "-y": np.array([0.0, -1.0])}
    cardinal_bindings = {}
    cardinal_indices = {}
    for label, target in card_targets.items():
        kidx = int(np.argmax(dirs_np @ target))
        leg, part = capsule_identity(capsules[int(binding[kidx])])
        cardinal_bindings[label] = {"leg": leg, "part": part}
        cardinal_indices[label] = kidx

    kmax = int(h_occ.argmax().item())
    ang_max = float(np.degrees(np.arctan2(dirs_np[kmax, 1], dirs_np[kmax, 0])))
    leg_max, part_max = capsule_identity(capsules[int(binding[kmax])])

    reach_poly, h_foot = None, None
    if show_reachable:
        reach_poly, h_foot = _reachable_polygon(K)

    edge_b = _edge_bindings(poly, dirs_np, _to_list(h_occ), binding)
    binding_l = [int(v) for v in binding]

    return {
        "q": [float(v) for v in q],
        "K": K,
        "margin": float(margin),
        "capsule_set": capsule_set,
        "n_capsules": int(len(capsules)),
        "capsule_names": [c.name for c in capsules],
        "capsule_links": [c.link for c in capsules],
        "capsule_radius": [c.radius for c in capsules],
        "capsules_3d": _to_list(endpoints),              # [L,2,3] world-space
        "capsules_xy": _to_list(endpoints[..., :2]),     # [L,2,2] plan projection
        "skeleton": _skeleton(q_t),
        "h_occ": _to_list(h_occ),
        "directions": dirs_np.tolist(),
        "per_capsule": _to_list(per_capsule),
        "binding": binding_l,
        "binding_names": [capsules[i].name for i in binding_l],
        "polygon": poly.tolist(),
        "margin_polygon": poly_margin.tolist(),
        "edge_bindings": [binding_l[e] if e is not None else None for e in edge_b],
        "feet": _to_list(feet),
        "legacy_condition": cond_l,
        "legacy_condition_clamped": cond_c_l,
        "legacy_width": float(cond_l[0]),
        "legacy_x_front": float(cond_l[3]),
        "legacy_x_back": float(cond_l[4]),
        "priors": cond_l[5:8],
        "reachable_support": h_foot,
        "reachable_polygon": reach_poly.tolist() if reach_poly is not None else None,
        "reachable_sample_count": _REACH_SAMPLES,
        "reachable_sample_seed": _REACH_SEED,
        "cardinal_bindings": cardinal_bindings,
        "cardinal_indices": cardinal_indices,
        "max_h_occ": float(h_occ.max().item()),
        "max_h_occ_direction_deg": ang_max,
        "max_h_occ_binding": {"leg": leg_max, "part": part_max},
        "leg_order": list(EL4090_LEG_NAMES),
        "joint_names": list(EL4090_JOINT_NAMES),
        "error": None,
    }


# --- Mode B: envelope -> per-joint rejection ranges --------------------------

def _hexagon_vertices(env: dict) -> np.ndarray:
    """Convex CCW 5-parameter hexagon ``V`` (front -> mid -> back)."""
    return np.array([
        (env["x_f"], env["w_f"]),
        (0.0, env["w_m"]),
        (env["x_b"], env["w_b"]),
        (env["x_b"], -env["w_b"]),
        (0.0, -env["w_m"]),
        (env["x_f"], -env["w_f"]),
    ], dtype=np.float64)


def _allowed_support(env: dict, margin: float, directions: np.ndarray) -> np.ndarray:
    """``h_k^allowed = max_{v in V} u_k^T v + margin`` over the hexagon vertices."""
    V = _hexagon_vertices(env)
    projections = directions @ V.T          # [K,6]
    return projections.max(axis=1) + margin


def _zero_rejection_response(env, margin, K, capsule_set, directions, reason):
    """Clean ``feasible_reference:false`` state -- never invent a fallback."""
    allowed = _allowed_support(env, margin, directions)
    try:
        allowed_poly = support_polygon(directions, allowed)
    except Exception:
        allowed_poly = None
    return {
        "feasible_reference": False,
        "reason": reason,
        "reference_source": "none",
        "reference_q": None,
        "reference_requested_q": None,
        "reference_used_q": None,
        "reference_infeasible_requested": False,
        "reference_support": None,
        "feet": None,
        "capsule_names": [],
        "capsule_links": [],
        "capsule_radius": [],
        "capsules_3d": [],
        "skeleton": None,
        "allowed_support": allowed.tolist(),
        "envelope_polygon": _hexagon_vertices(env).tolist(),
        "allowed_polygon": allowed_poly.tolist() if allowed_poly is not None else None,
        "accessible_box": None,
        "rejected": {
            "per_joint_intervals_rad": [[] for _ in EL4090_JOINT_NAMES],
            "rejected_joint_count": 0,
            "max_rejected_span_rad": 0.0,
            "max_rejected_joint_index": -1,
            "max_rejected_joint_name": None,
        },
        "magenta_arcs": [],
        "amber_arcs": [],
        "haa_arc_geometry": None,
        "envelope": {**env, "margin": margin},
        "K": K,
        "capsule_set": capsule_set,
        "leg_order": list(EL4090_LEG_NAMES),
        "joint_names": list(EL4090_JOINT_NAMES),
        "error": None,
    }


def compute_rejection(payload: dict) -> dict:
    """Mode B: envelope sliders -> per-joint rejected sub-intervals + arcs."""
    env = payload.get("envelope") or {}
    env = {
        "w_f": float(env.get("w_f", DEFAULT_ENVELOPE["w_f"])),
        "w_m": float(env.get("w_m", DEFAULT_ENVELOPE["w_m"])),
        "w_b": float(env.get("w_b", DEFAULT_ENVELOPE["w_b"])),
        "x_f": float(env.get("x_f", DEFAULT_ENVELOPE["x_f"])),
        "x_b": float(env.get("x_b", DEFAULT_ENVELOPE["x_b"])),
    }
    margin = float(payload.get("margin", DEFAULT_MARGIN_B))
    K = int(payload.get("K", 32))
    capsule_set = str(payload.get("capsule_set", "full"))
    reference = str(payload.get("reference", "resting"))
    reference_q = payload.get("reference_q")
    steps = int(payload.get("steps", 101))
    tol = float(payload.get("tolerance", 1e-6))

    capsules = _capsules(capsule_set)
    directions = support_directions(K)
    dirs_np = directions.detach().cpu().numpy()
    lower, upper = _kine.joint_limits(soft_fraction=1.0)

    if reference == "custom" and reference_q is not None:
        preferred = torch.tensor(reference_q, dtype=torch.float32)
        requested_q = list(reference_q)
    else:
        preferred = torch.tensor(RESTING_Q, dtype=torch.float32)
        requested_q = None

    allowed_np = _allowed_support(env, margin, dirs_np)
    allowed = torch.tensor(allowed_np, dtype=torch.float32)
    try:
        allowed_poly = support_polygon(directions, allowed_np)
    except Exception:
        allowed_poly = None

    fallback = deterministic_joint_samples(lower, upper, _CANDIDATE_SAMPLES, seed=_CANDIDATE_SEED)

    chosen, source = feasible_reference_q(
        _kine, capsules, directions, allowed, lower, upper, preferred,
        tolerance=tol, fallback_candidates=fallback,
    )
    if chosen is None:
        return _zero_rejection_response(
            env, margin, K, capsule_set, dirs_np,
            "no feasible reference among reference pose, box center, and "
            "deterministic candidate samples at this envelope",
        )
    # Honest reference reporting: a requested custom reference that was not
    # feasible silently fell back before (source != "reference_q"); surface it.
    reference_infeasible_requested = bool(
        reference == "custom" and requested_q is not None and source != "reference_q"
    )

    rej = joint_rejection_ranges(
        _kine, capsules, directions, allowed, lower, upper, chosen,
        tolerance=tol, steps=steps, fallback_candidates=fallback,
    )
    box = export_envelope_joint_ranges_at_reference(
        _kine, capsules, directions, allowed, lower, upper, chosen,
        tolerance=tol, steps=steps,
    )

    ref_support = capsule_support(_kine, chosen.unsqueeze(0), capsules, directions)[0]
    feet = foot_positions(_kine, chosen.unsqueeze(0))[0]

    # --- magenta rejected arcs (every joint, every rejected interval) --------
    # Each arc keeps the stylized foot-direction band (``points``, radius-style
    # like the HAA arcs) plus the TRUE FK foot path (``trajectory``) swept along
    # the interval with all other joints pinned -- the foot sweep the rejection
    # animation follows -- and a compact swept-leg skeleton keyframe set.
    magenta_arcs = []
    for j, intervals in enumerate(rej.rejected_intervals):
        joint_name = EL4090_JOINT_NAMES[j]
        leg, kind = joint_name.split("_")
        leg_i = int(LEG_INDEX[leg])
        for lo, hi in intervals:
            if hi - lo < 1e-9:
                continue
            pts = joint_arc_geometry_interval(
                _kine, chosen, leg_i, kind, float(lo), float(hi),
                radius=_ARC_RADIUS, samples=_ARC_SAMPLES,
            )
            swept = _swept_leg_frames(chosen, leg_i, kind, lo, hi)
            magenta_arcs.append({
                "leg": leg,
                "leg_index": leg_i,
                "joint_kind": kind,
                "joint_name": joint_name,
                "interval": [float(lo), float(hi)],
                "points": _to_list(pts),
                "sweep": swept["sweep"],
                "trajectory": swept["trajectory"],
                "leg_skeleton_frames": swept["leg_skeleton_frames"],
            })

    # --- amber accessible HAA arcs (box minus rejected, tiles exactly) -------
    amber_arcs = []
    haa_geom = None
    haa_ranges = haa_ranges_from_joint_export(box)          # [6,2]
    for leg_i, leg in enumerate(EL4090_LEG_NAMES):
        haa_idx = int(JOINT_INDEX[f"{leg}_HAA"])
        box_lo = float(box.lower[haa_idx].item())
        box_hi = float(box.upper[haa_idx].item())
        if not (math.isfinite(box_lo) and math.isfinite(box_hi)):
            continue
        accessible = accessible_interval_complement(
            box_lo, box_hi, rej.rejected_intervals[haa_idx],
        )
        for lo, hi in accessible:
            if hi - lo < 1e-9:
                continue
            pts = joint_arc_geometry_interval(
                _kine, chosen, leg_i, "HAA", float(lo), float(hi),
                radius=_ARC_RADIUS, samples=_ARC_SAMPLES,
            )
            amber_arcs.append({
                "leg": leg,
                "leg_index": leg_i,
                "joint_kind": "HAA",
                "joint_name": f"{leg}_HAA",
                "interval": [float(lo), float(hi)],
                "points": _to_list(pts),
            })
    try:
        origins, arcs, markers = haa_arc_geometry(
            _kine, chosen, haa_ranges, radius=_ARC_RADIUS, samples=_HAA_ARC_SAMPLES,
        )
        haa_geom = {
            "origins": _to_list(origins),
            "arcs": _to_list(arcs),
            "markers": _to_list(markers),
        }
    except Exception:
        haa_geom = None

    rejected = {
        "per_joint_intervals_rad": [
            [[float(lo), float(hi)] for lo, hi in joint]
            for joint in rej.rejected_intervals
        ],
        "rejected_joint_count": int(rej.rejected_joint_count),
        "max_rejected_span_rad": float(rej.max_rejected_span_rad),
        "max_rejected_joint_index": int(rej.max_rejected_joint_index),
        "max_rejected_joint_name": (
            EL4090_JOINT_NAMES[rej.max_rejected_joint_index]
            if rej.max_rejected_joint_index >= 0 else None
        ),
    }

    geom = _capsule_geometry_payload(capsules, chosen.unsqueeze(0))
    return {
        "feasible_reference": bool(rej.feasible_reference),
        "reason": None,
        "reference_source": source,
        "reference_source_revalidated": rej.reference_source,
        "reference_q": _to_list(chosen),
        "reference_requested_q": requested_q,
        "reference_used_q": _to_list(chosen),
        "reference_infeasible_requested": reference_infeasible_requested,
        "reference_support": _to_list(ref_support),
        "feet": _to_list(feet),
        "capsule_names": geom["capsule_names"],
        "capsule_links": geom["capsule_links"],
        "capsule_radius": geom["capsule_radius"],
        "capsules_3d": geom["capsules_3d"],
        "skeleton": _skeleton(chosen.unsqueeze(0)),
        "allowed_support": allowed_np.tolist(),
        "envelope_polygon": _hexagon_vertices(env).tolist(),
        "allowed_polygon": allowed_poly.tolist() if allowed_poly is not None else None,
        "accessible_box": {
            "lower": _to_list(box.lower),
            "upper": _to_list(box.upper),
            "center": _to_list(box.center),
            "half_range": _to_list(box.half_range),
            "valid": bool(box.valid.item()),
            "diagnostics_label": box.diagnostics.label,
        },
        "rejected": rejected,
        "magenta_arcs": magenta_arcs,
        "amber_arcs": amber_arcs,
        "haa_arc_geometry": haa_geom,
        "envelope": {**env, "margin": margin},
        "K": K,
        "capsule_set": capsule_set,
        "reference_mode": reference,
        "steps": steps,
        "tolerance": tol,
        "leg_order": list(EL4090_LEG_NAMES),
        "joint_names": list(EL4090_JOINT_NAMES),
        "error": None,
    }


# --- Mode C: envelope + HFE/KFE pins -> per-leg HAA rejection ranges ----------

def compute_haa_rejection(payload: dict) -> dict:
    """Mode C: per-leg HAA rejection at EXACT HFE/KFE pins.

    MasterYip: "use current HFE/KFE joint positions to discover the HAA
    rejection range; but I can adjust HFE/KFE and the HAA rejection should
    change."  The HFE/KFE pins are honored EXACTLY -- there is no fallback
    that changes them.  The module's ``haa_rejection_ranges`` implements the
    three regimes.  When the pinned (extended) pose fits, the rejection is the
    reference-pinned per-axis sweep at the pins (mode ``"pinned"``, the rev-4..6
    semantics).  When it does not fit, the rejection is the fold-aware
    existential projection (mode ``"fold"``): a leg's HAA is rejected only when
    NO fold of the other legs' HAA fits -- so in a narrow envelope the tuck
    |HAA| > ~1.57 shows up accessible (MasterYip: "mammal can let the legged
    fold in ... don't need to change HFE/KFE").  When no HAA tuple fits at all,
    the honest full-range answer is returned (mode ``"none"``).

    The response mirrors ``compute_rejection``'s schema so the frontend reuses
    the Mode B renderers unchanged (``magenta_arcs``/``amber_arcs``/
    ``haa_arc_geometry``/``skeleton``/``capsules_3d``/``allowed_polygon``...);
    only the HAA rows of ``rejected.per_joint_intervals_rad`` are non-empty.
    """
    env = payload.get("envelope") or {}
    env = {
        "w_f": float(env.get("w_f", DEFAULT_ENVELOPE["w_f"])),
        "w_m": float(env.get("w_m", DEFAULT_ENVELOPE["w_m"])),
        "w_b": float(env.get("w_b", DEFAULT_ENVELOPE["w_b"])),
        "x_f": float(env.get("x_f", DEFAULT_ENVELOPE["x_f"])),
        "x_b": float(env.get("x_b", DEFAULT_ENVELOPE["x_b"])),
    }
    margin = float(payload.get("margin", DEFAULT_MARGIN_B))
    K = int(payload.get("K", 32))
    capsule_set = str(payload.get("capsule_set", "full"))
    hfe = payload.get("hfe") or [0.6] * 6
    kfe = payload.get("kfe") or [-0.6] * 6
    # ``haa`` is the CURRENT HAA check value per leg -- used ONLY to draw the
    # skeleton/feet/capsules and a marker on each HAA rejection arc so MasterYip
    # can visually verify the rejection range. It never changes the band (the
    # band is the HAA sweep at the HFE/KFE pins, independent of the check value).
    haa = payload.get("haa") or [0.0] * 6
    steps = int(payload.get("steps", 201))
    tol = float(payload.get("tolerance", 1e-6))

    if len(hfe) != len(EL4090_LEG_NAMES) or len(kfe) != len(EL4090_LEG_NAMES):
        raise ValueError(
            "hfe and kfe must each be length-6 lists in EL4090_LEG_NAMES order")
    if len(haa) != len(EL4090_LEG_NAMES):
        raise ValueError("haa must be a length-6 list in EL4090_LEG_NAMES order")
    hfe = [float(v) for v in hfe]
    kfe = [float(v) for v in kfe]
    haa = [float(v) for v in haa]

    capsules = _capsules(capsule_set)
    directions = support_directions(K)
    dirs_np = directions.detach().cpu().numpy()
    lower, upper = _kine.joint_limits(soft_fraction=1.0)

    allowed_np = _allowed_support(env, margin, dirs_np)
    allowed = torch.tensor(allowed_np, dtype=torch.float32)
    try:
        allowed_poly = support_polygon(directions, allowed_np)
    except Exception:
        allowed_poly = None

    # Exact pins: HAA stays 0.0 (it is swept; its value here is irrelevant),
    # HFE/KFE come straight from the sliders and are ALWAYS honored exactly --
    # there is no fallback that changes them (MasterYip: "don't need to change
    # HFE/KFE").
    pinned = torch.zeros(len(EL4090_JOINT_NAMES), dtype=torch.float32)
    for i, leg in enumerate(EL4090_LEG_NAMES):
        pinned[JOINT_INDEX[f"{leg}_HFE"]] = hfe[i]
        pinned[JOINT_INDEX[f"{leg}_KFE"]] = kfe[i]

    haa_indices = [int(JOINT_INDEX[f"{leg}_HAA"]) for leg in EL4090_LEG_NAMES]

    # Per-HAA-joint rejected sub-intervals (empty for HFE/KFE rows).
    #
    # MasterYip: "decrease the width(all width) the HAA rejection region turns
    # out to be the full range, this is incorrect, mammal can let the legged
    # fold in ... don't need to change HFE/KFE".  The rejection must reflect
    # that a real mammal FOLDS ITS LEGS IN VIA HAA to fit a narrow envelope,
    # with HFE/KFE pinned exactly.  The module's ``haa_rejection_ranges``
    # implements the three regimes:
    #
    #  * pins FEASIBLE    -> reference-pinned per-axis sweep at the exact pins
    #    (mode "pinned"; the verified rev-4..6 semantics).
    #  * pins INFEASIBLE  -> fold-aware existential projection
    #    (mode "fold"): a leg's HAA value v is rejected only when NO fold of
    #    the other legs' HAA (HFE/KFE pinned) fits -- so the tuck
    #    |HAA| > ~1.57 shows up accessible instead of the old full range.
    #  * no HAA tuple fits -> full-range rejection (mode "none", honest).
    haa_result = haa_rejection_ranges(
        _kine, capsules, directions, allowed, lower, upper, pinned, haa_indices,
        tolerance=tol, sweep_steps=steps,
    )
    haa_intervals = haa_result.per_haa_joint_intervals
    pins_feasible = haa_result.pins_feasible
    rejection_mode = haa_result.mode

    # Current visualization pose: the EXACT pins with HAA = the user's check
    # value (never a folded reference -- HFE/KFE are pinned by request).  The
    # skeleton/feet/capsules are drawn here so the user sees their own pose.
    base_ref = pinned
    current = base_ref.clone()
    for i, haa_idx in enumerate(haa_indices):
        current[haa_idx] = haa[i]

    per_joint = []
    for j in range(len(EL4090_JOINT_NAMES)):
        if j in haa_intervals:
            per_joint.append(
                [[float(lo), float(hi)] for lo, hi in haa_intervals[j]])
        else:
            per_joint.append([])

    spans = [
        (hi - lo, j)
        for j in haa_indices
        for lo, hi in haa_intervals[j]
    ]
    if spans:
        max_span, max_joint = max(spans)
    else:
        max_span, max_joint = 0.0, -1
    rejected = {
        "per_joint_intervals_rad": per_joint,
        "rejected_joint_count": sum(1 for j in haa_indices if per_joint[j]),
        "max_rejected_span_rad": float(max_span),
        "max_rejected_joint_index": int(max_joint),
        "max_rejected_joint_name": (
            EL4090_JOINT_NAMES[max_joint] if max_joint >= 0 else None),
    }

    # --- magenta rejected HAA arcs (only HAA; HFE/KFE are the reference, not swept) ---
    magenta_arcs = []
    for haa_idx in haa_indices:
        joint_name = EL4090_JOINT_NAMES[haa_idx]
        leg, _kind = joint_name.split("_")
        leg_i = int(LEG_INDEX[leg])
        for lo, hi in haa_intervals[haa_idx]:
            if hi - lo < 1e-9:
                continue
            pts = joint_arc_geometry_interval(
                _kine, base_ref, leg_i, "HAA", float(lo), float(hi),
                radius=_ARC_RADIUS, samples=_ARC_SAMPLES,
            )
            swept = _swept_leg_frames(base_ref, leg_i, "HAA", lo, hi)
            magenta_arcs.append({
                "leg": leg,
                "leg_index": leg_i,
                "joint_kind": "HAA",
                "joint_name": joint_name,
                "interval": [float(lo), float(hi)],
                "points": _to_list(pts),
                "sweep": swept["sweep"],
                "trajectory": swept["trajectory"],
                "leg_skeleton_frames": swept["leg_skeleton_frames"],
            })

    # --- amber accessible HAA arcs (full box minus rejected, tiles exactly) ----
    amber_arcs = []
    haa_geom = None
    haa_ranges = torch.tensor(
        [[float(lower[haa_idx]), float(upper[haa_idx])] for haa_idx in haa_indices],
        dtype=torch.float32,
    )
    for leg_i, leg in enumerate(EL4090_LEG_NAMES):
        haa_idx = int(JOINT_INDEX[f"{leg}_HAA"])
        box_lo = float(lower[haa_idx].item())
        box_hi = float(upper[haa_idx].item())
        if not (math.isfinite(box_lo) and math.isfinite(box_hi)):
            continue
        accessible = accessible_interval_complement(
            box_lo, box_hi, haa_intervals[haa_idx],
        )
        for lo, hi in accessible:
            if hi - lo < 1e-9:
                continue
            pts = joint_arc_geometry_interval(
                _kine, base_ref, leg_i, "HAA", float(lo), float(hi),
                radius=_ARC_RADIUS, samples=_ARC_SAMPLES,
            )
            amber_arcs.append({
                "leg": leg,
                "leg_index": leg_i,
                "joint_kind": "HAA",
                "joint_name": f"{leg}_HAA",
                "interval": [float(lo), float(hi)],
                "points": _to_list(pts),
            })
    try:
        origins, arcs, markers = haa_arc_geometry(
            _kine, base_ref, haa_ranges, radius=_ARC_RADIUS, samples=_HAA_ARC_SAMPLES,
        )
        haa_geom = {
            "origins": _to_list(origins),
            "arcs": _to_list(arcs),
            "markers": _to_list(markers),
        }
    except Exception:
        haa_geom = None

    ref_support = capsule_support(_kine, base_ref.unsqueeze(0), capsules, directions)[0]
    current_support = capsule_support(_kine, current.unsqueeze(0), capsules, directions)[0]
    current_feasible = bool((current_support <= allowed + tol).all().item())
    feet = foot_positions(_kine, current.unsqueeze(0))[0]
    geom = _capsule_geometry_payload(capsules, current.unsqueeze(0))

    # Current-HAA markers on each leg's rejection arc (same stylization as the
    # band: hip + radius*unit(hip-to-foot XY) at HAA=haa[i], other joints at the
    # reference actually used so the marker sits exactly on the band).
    haa_markers = []
    for i, leg in enumerate(EL4090_LEG_NAMES):
        v = haa[i]
        ref_marker = base_ref.clone()
        ref_marker[haa_indices[i]] = v
        try:
            mpts = joint_arc_geometry_interval(
                _kine, ref_marker, i, "HAA", float(v), float(v),
                radius=_ARC_RADIUS, samples=3,
            )
            point = mpts[0]
        except Exception:
            point = None
        haa_markers.append({
            "leg": leg,
            "leg_index": i,
            "value": v,
            "point": _to_list(point) if point is not None else None,
        })

    # Rejection-mode label for the frontend + "some posture fits" flag.  The
    # mode was computed by the module function; keep the flag derivation here.
    posture_exists = rejection_mode != "none"

    return {
        "feasible_reference": posture_exists,
        "reference_pins_infeasible": not pins_feasible,
        "rejection_mode": rejection_mode,
        "reason": (
            None if rejection_mode == "pinned" else
            ("no feasible posture fits this envelope — HAA rejection is the full range"
             if rejection_mode == "none" else
             "HFE/KFE pins infeasible at this envelope — the HAA rejection accounts "
             "for the other legs folding in via HAA (HFE/KFE pinned)")
        ),
        "reference_source": ("pinned" if pins_feasible else "fold over the HAA group"),
        "reference_q": _to_list(base_ref),
        "reference_requested_q": _to_list(pinned),
        "reference_used_q": _to_list(pinned),
        "reference_infeasible_requested": False,   # pins are never silently swapped
        "reference_support": _to_list(ref_support),
        "current_support": _to_list(current_support),
        "current_feasible": current_feasible,
        "haa_current": haa,
        "haa_markers": haa_markers,
        "feet": _to_list(feet),
        "capsule_names": geom["capsule_names"],
        "capsule_links": geom["capsule_links"],
        "capsule_radius": geom["capsule_radius"],
        "capsules_3d": geom["capsules_3d"],
        "skeleton": _skeleton(current.unsqueeze(0)),
        "allowed_support": allowed_np.tolist(),
        "envelope_polygon": _hexagon_vertices(env).tolist(),
        "allowed_polygon": allowed_poly.tolist() if allowed_poly is not None else None,
        "accessible_box": None,
        "rejected": rejected,
        "magenta_arcs": magenta_arcs,
        "amber_arcs": amber_arcs,
        "haa_arc_geometry": haa_geom,
        "envelope": {**env, "margin": margin},
        "K": K,
        "capsule_set": capsule_set,
        "reference_mode": rejection_mode,
        "steps": steps,
        "tolerance": tol,
        "leg_order": list(EL4090_LEG_NAMES),
        "joint_names": list(EL4090_JOINT_NAMES),
        "error": None,
    }


# --- health ------------------------------------------------------------------

def health_payload() -> dict:
    return {
        "ok": True,
        "mode_list": ["envelope", "rejection", "haa_rejection"],
        "model": "Occupied Body Support Envelope (ENV-DESIGN-003)",
        "urdf": str(_URDF),
        "joint_names": list(EL4090_JOINT_NAMES),
        "leg_names": list(EL4090_LEG_NAMES),
        "presets": list(PRESETS),
        "resting_q": RESTING_Q,
        "default_envelope": DEFAULT_ENVELOPE,
        "reachable_overlay": {"sample_count": _REACH_SAMPLES, "seed": _REACH_SEED},
        "rejection_fallback": {"sample_count": _CANDIDATE_SAMPLES, "seed": _CANDIDATE_SEED},
        "arc_radius": _ARC_RADIUS,
    }


# --- HTTP layer --------------------------------------------------------------

class EnvelopeHandler(BaseHTTPRequestHandler):
    server_version = "EL4090EnvelopeExplorer/1.0"
    # HTTP/1.1 with explicit Content-Length (sent by _send_json): the browser
    # may otherwise reuse a keep-alive socket that the HTTP/1.0 default closes
    # after the first response, silently hanging the next request during rapid
    # slider bursts (a measured rev-2 issue).
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet by default; errors still visible
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(_clean(obj)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name, content_type):
        path = _TASK_DIR / name
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif route == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
        elif route == "/styles.css":
            self._serve_static("styles.css", "text/css; charset=utf-8")
        elif route == "/api/health":
            self._send_json(health_payload())
        else:
            self._send_json({"error": f"unknown route {route}"}, status=404)

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > (1 << 20):
                raise ValueError("payload too large")
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"bad request: {exc}"}, status=400)
            return
        try:
            if route == "/api/pose":
                result = compute_pose(payload)
            elif route == "/api/envelope":
                result = compute_envelope(payload)
            elif route == "/api/rejection":
                result = compute_rejection(payload)
            elif route == "/api/haa_rejection":
                result = compute_haa_rejection(payload)
            else:
                self._send_json({"error": f"unknown route {route}"}, status=404)
                return
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self._send_json(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = None
    for port in range(args.port, args.port + 32):
        try:
            server = ThreadingHTTPServer((args.host, port), EnvelopeHandler)
            bound_port = port
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {args.port}..{args.port + 31}")
    server.daemon_threads = True

    def _shutdown(_signum, _frame):
        print("shutting down EL4090 envelope server...", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"EL4090 envelope explorer: http://{args.host}:{bound_port}/", flush=True)
    print(f"  model: {_URDF}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
