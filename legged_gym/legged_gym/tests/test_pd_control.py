"""ENV-RECT-CONTROL-012: PD joint control + q_c + EMA rejection reference."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_DIR = ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))

SPEC = importlib.util.spec_from_file_location(
    "pd_control_under_test", ENVELOPE_DIR / "pd_control.py",
)
assert SPEC is not None and SPEC.loader is not None
PD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PD
SPEC.loader.exec_module(PD)

LOWER = torch.full((18,), -3.0)
UPPER = torch.full((18,), 3.0)
EMPTY_INTERVALS = ((),) * 18


def intervals_for(*pairs):
    """Return a rejected-intervals tuple with the given (joint, (lo, hi)) pairs."""
    intervals = [() for _ in range(18)]
    for joint, interval in pairs:
        intervals[joint] = (interval,)
    return tuple(intervals)


class TestNearestOutsideRejection(unittest.TestCase):
    def test_value_outside_all_intervals_is_unchanged(self):
        q_e = torch.tensor([0.0, 0.5, -1.0, 2.5, 0.05] + [0.0] * 13)
        q_c = PD.nearest_outside_rejection(
            q_e, intervals_for((1, (0.1, 0.4))), LOWER, UPPER,
        )
        self.assertTrue(torch.equal(q_c, q_e))

    def test_value_inside_moves_to_nearer_endpoint(self):
        q_e = torch.zeros(18)
        q_e[0] = 0.35  # inside [0.1, 0.4]; nearer to 0.4 (0.05 vs 0.25)
        q_e[1] = 0.15  # inside [0.1, 0.4]; nearer to 0.1 (0.05 vs 0.25)
        q_c = PD.nearest_outside_rejection(
            q_e, intervals_for((0, (0.1, 0.4)), (1, (0.1, 0.4))), LOWER, UPPER,
        )
        self.assertAlmostEqual(float(q_c[0]), 0.4, places=6)
        self.assertAlmostEqual(float(q_c[1]), 0.1, places=6)

    def test_result_stays_inside_the_urdf_box(self):
        q_e = torch.full((18,), 5.0)  # beyond the box everywhere
        q_e[0] = 0.3  # inside a rejected interval
        q_c = PD.nearest_outside_rejection(
            q_e, intervals_for((0, (0.1, 0.4))), LOWER, UPPER,
        )
        self.assertTrue(bool((q_c >= LOWER - 1e-6).all()))
        self.assertTrue(bool((q_c <= UPPER + 1e-6).all()))
        self.assertAlmostEqual(float(q_c[0]), 0.4, places=6)
        self.assertAlmostEqual(float(q_c[1]), 3.0, places=6)

    def test_empty_intervals_keep_q_e(self):
        q_e = torch.full((18,), 0.7)
        q_c = PD.nearest_outside_rejection(q_e, EMPTY_INTERVALS, LOWER, UPPER)
        self.assertTrue(torch.equal(q_c, q_e))


class TestEmaReferenceUpdate(unittest.TestCase):
    def test_blend_and_edge_alphas(self):
        q = torch.full((18,), 1.0)
        reference = torch.full((18,), 0.0)
        blended = PD.ema_reference_update(q, reference, 0.5)
        self.assertTrue(torch.allclose(blended, torch.full((18,), 0.5)))
        self.assertTrue(torch.allclose(
            PD.ema_reference_update(q, reference, 0.0), reference,
        ))
        self.assertTrue(torch.allclose(
            PD.ema_reference_update(q, reference, 1.0), q,
        ))

    def test_reference_tracks_a_moving_joint(self):
        q = torch.zeros(18)
        reference = torch.zeros(18)
        alpha = 0.5
        for _ in range(200):
            q = torch.full((18,), 1.5)
            reference = PD.ema_reference_update(q, reference, alpha)
        self.assertTrue(torch.allclose(reference, torch.full((18,), 1.5), atol=1e-6))


class TestPdControl(unittest.TestCase):
    def setUp(self):
        self.kp, self.kd, self.dt, self.max_rate = 64.0, 16.0, 1.0 / 60.0, 3.0
        self.settle = 2e-3

    def integrate(self, q, q_dot, q_c, max_steps=900):
        for steps in range(1, max_steps + 1):
            q, q_dot = PD.pd_integrate(
                q, q_dot, q_c, kp=self.kp, kd=self.kd, dt=self.dt,
                max_rate=self.max_rate, lower=LOWER, upper=UPPER,
            )
            if PD.pd_settled(q, q_c, q_dot, self.settle):
                return q, q_dot, steps
        return q, q_dot, max_steps

    def test_converges_to_target_in_bounded_steps_within_limits(self):
        q = torch.zeros(18)
        q_dot = torch.zeros(18)
        q_c = torch.full((18,), 2.5)
        q, q_dot, steps = self.integrate(q, q_dot, q_c)
        self.assertLess(steps, 900)
        self.assertTrue(torch.allclose(q, q_c, atol=2e-3))
        self.assertTrue(bool((q >= LOWER - 1e-6).all()))
        self.assertTrue(bool((q <= UPPER + 1e-6).all()))
        self.assertLessEqual(float(q_dot.abs().max()), self.max_rate + 1e-6)

    def test_no_overshoot_beyond_target(self):
        q = torch.zeros(18)
        q_dot = torch.zeros(18)
        q_c = torch.full((18,), 2.5)
        q, q_dot, steps = self.integrate(q, q_dot, q_c)
        # With a critically damped PD and a bounded velocity, the joint never
        # overshoots past the target on the way in.
        self.assertTrue(bool((q <= q_c + self.settle).all()))
        # Reverse direction: from +2.5 down to -2.0 must not undershoot.
        q, q_dot = torch.full((18,), 2.5), torch.zeros(18)
        q_c = torch.full((18,), -2.0)
        q, q_dot, steps = self.integrate(q, q_dot, q_c)
        self.assertTrue(bool((q >= q_c - self.settle).all()))

    def test_settles_at_the_rejection_boundary_for_q_e_inside_rejection(self):
        # A2 acceptance: with q_e inside a rejected interval, the joint settles
        # at the nearest reachable boundary (inside the envelope) - here the
        # nearer endpoint 0.4 of the rejected [0.1, 0.4].
        q_e = torch.zeros(18)
        q_e[0] = 0.3
        q_c = PD.nearest_outside_rejection(
            q_e, intervals_for((0, (0.1, 0.4))), LOWER, UPPER,
        )
        self.assertAlmostEqual(float(q_c[0]), 0.4, places=6)
        q, q_dot = torch.zeros(18), torch.zeros(18)
        q, q_dot, steps = self.integrate(q, q_dot, q_c)
        self.assertLess(steps, 900)
        self.assertAlmostEqual(float(q[0]), 0.4, places=3)

    def test_pd_integration_is_deterministic(self):
        q = torch.zeros(18)
        q_dot = torch.zeros(18)
        q_c = torch.full((18,), 1.2)
        first, _, _ = self.integrate(q.clone(), q_dot.clone(), q_c)
        second, _, _ = self.integrate(q.clone(), q_dot.clone(), q_c)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
