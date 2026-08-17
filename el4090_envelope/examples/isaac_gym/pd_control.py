"""PD joint control helpers for the legacy slider demo (ENV-RECT-CONTROL-012).

MasterYip: "joint control: use pd to let the joint reach calculated position
q_c, the expect position is human-set through slider q_e, q_c is the nearest
position to q_e but outside the rejection range."

The rejection intervals are swept per joint over the full URDF box with all
other joints pinned at the reference (see ``joint_rejection_ranges``), so q_c
for joint j is q_e[j] clamped outside every rejected interval of j. The PD loop
is a discrete critically-damped integrator; the EMA reference lags the actual
joint and is re-validated by ``feasible_reference_q`` in the caller.

This module intentionally has no Isaac Gym dependency so it can be unit tested
without the ``isaacgym``-before-``torch`` import ordering constraint.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor


def nearest_outside_rejection(
    q_e: Tensor,
    rejected_intervals: Sequence[Sequence[Tuple[float, float]]],
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Per-joint nearest value to ``q_e`` outside every rejected interval.

    A joint value inside a rejected interval ``[a, b]`` is moved to the nearer
    endpoint ``a`` or ``b`` (the nearest position outside the rejection range);
    values already outside all rejected intervals are returned unchanged. The
    result is clamped to ``[lower, upper]`` so it always stays inside the URDF
    box. ``rejected_intervals`` is ``rejection_ranges.rejected_intervals``.
    """
    q_c = q_e.clone()
    for joint, intervals in enumerate(rejected_intervals):
        value = float(q_e[joint])
        for lo, hi in intervals:
            if lo <= value <= hi:
                q_c[joint] = lo if (value - lo) <= (hi - value) else hi
    return torch.maximum(lower, torch.minimum(upper, q_c))


def ema_reference_update(q: Tensor, reference: Tensor, alpha: float) -> Tensor:
    """Exponential moving average of the actual joint used as the reference r."""
    return alpha * q + (1.0 - alpha) * reference


def pd_integrate(
    q: Tensor,
    q_dot: Tensor,
    q_c: Tensor,
    *,
    kp: float,
    kd: float,
    dt: float,
    max_rate: float,
    lower: Tensor,
    upper: Tensor,
) -> Tuple[Tensor, Tensor]:
    """One discrete PD integration step toward ``q_c``.

    ``accel = kp * (q_c - q) - kd * q_dot``, then velocity and position are
    integrated with the sim ``dt``. Velocity is clamped to ``max_rate`` and the
    position to the URDF limits. With ``kd = 2 * sqrt(kp)`` the continuous
    system is critically damped (no overshoot); the velocity clamp turns large
    moves into a bounded-rate approach that settles monotonically.
    """
    accel = kp * (q_c - q) - kd * q_dot
    q_dot = q_dot + accel * dt
    q_dot = torch.clamp(q_dot, -max_rate, max_rate)
    q = q + q_dot * dt
    return torch.clamp(q, lower, upper), q_dot


def pd_settled(
    q: Tensor, q_c: Tensor, q_dot: Tensor, settle_rad: float,
) -> bool:
    """True when the joint has converged to ``q_c`` and is at rest."""
    return bool(
        (q - q_c).abs().max() <= settle_rad
        and q_dot.abs().max() <= settle_rad,
    )
