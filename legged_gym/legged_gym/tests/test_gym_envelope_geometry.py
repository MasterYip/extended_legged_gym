import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_DIR = ROOT / "utils" / "envelop"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KE = load_module("kinematic_envelope", ENVELOPE_DIR / "kinematic_envelope.py")
GEOM = load_module("gym_envelope_geometry", ENVELOPE_DIR / "gym_envelope_geometry.py")
URDF = ROOT.parent / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"


class TestGymEnvelopeGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = KE.BatchedUrdfKinematics(KE.load_urdf_joints(URDF))

    def test_support_polygon_satisfies_registered_half_spaces(self):
        directions = KE.support_directions(32, dtype=torch.float64)
        support = KE.capsule_support(
            self.kinematics,
            torch.zeros(1, 18, dtype=torch.float64),
            KE.default_el4090_capsules(),
            directions,
        )[0]
        polygon = GEOM.support_polygon(directions, support)
        violations = polygon @ directions.numpy().T - support.numpy()[None, :]
        self.assertGreaterEqual(polygon.shape[0], 3)
        self.assertLessEqual(float(violations.max()), 2e-10)
        segments = GEOM.polyline_segments(np.column_stack((polygon, np.zeros(len(polygon)))), closed=True)
        self.assertEqual(segments.shape, (2 * len(polygon), 3))

    def test_joint_range_interpolation_is_deterministic_and_compliant(self):
        lower = torch.linspace(-1.2, -0.3, 18, dtype=torch.float64)
        upper = lower + torch.linspace(0.2, 1.1, 18, dtype=torch.float64)
        offsets = torch.linspace(0.0, 0.95, 18, dtype=torch.float64)
        first = torch.stack([
            GEOM.interpolate_joint_ranges(lower, upper, phase, phase_offsets=offsets)
            for phase in torch.linspace(0.0, 2.0, 81, dtype=torch.float64)
        ])
        second = torch.stack([
            GEOM.interpolate_joint_ranges(lower, upper, phase, phase_offsets=offsets)
            for phase in torch.linspace(0.0, 2.0, 81, dtype=torch.float64)
        ])
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool((first >= lower).all()))
        self.assertTrue(bool((first <= upper).all()))
        for pose in first:
            self.assertEqual(
                GEOM.joint_range_violations(pose, lower, upper), (0, 0.0),
            )

        self.assertTrue(torch.equal(
            GEOM.interpolate_joint_ranges(lower, upper, 0.0), lower,
        ))
        self.assertTrue(torch.allclose(
            GEOM.interpolate_joint_ranges(lower, upper, 0.5), upper, atol=1e-15,
        ))

    def test_joint_range_violation_summary_counts_interval_excess(self):
        lower = torch.full((18,), -0.5)
        upper = torch.full((18,), 0.5)
        pose = torch.zeros(18)
        pose[2] = -0.6
        pose[11] = 0.7
        count, maximum = GEOM.joint_range_violations(pose, lower, upper)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(maximum, 0.2, places=6)

    def test_haa_arcs_use_actual_urdf_hip_frames(self):
        current = torch.tensor([0.1, 0.6, -0.6] * 6, dtype=torch.float64)
        ranges = torch.tensor([[-0.4, 0.5]] * 6, dtype=torch.float64)
        origins, arcs, markers = GEOM.haa_arc_geometry(
            self.kinematics, current, ranges, radius=0.2, samples=17,
        )
        self.assertEqual(origins.shape, (6, 3))
        self.assertEqual(arcs.shape, (6, 17, 3))
        self.assertEqual(markers.shape, (6, 3))
        self.assertTrue(torch.allclose((arcs - origins[:, None]).norm(dim=-1), torch.full((6, 17), 0.2, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose((markers - origins).norm(dim=-1), torch.full((6,), 0.2, dtype=torch.float64), atol=1e-12))

        foot_links = [f"{leg}_FOOT" for leg in KE.EL4090_LEG_NAMES]
        local_origins = torch.zeros((6, 1, 3), dtype=torch.float64)
        foot_origins = self.kinematics.points(
            current.unsqueeze(0), foot_links, local_origins,
        )[0, :, 0]
        marker_xy = markers[:, :2] - origins[:, :2]
        physical_xy = foot_origins[:, :2] - origins[:, :2]
        marker_directions = torch.nn.functional.normalize(marker_xy, dim=-1)
        physical_leg_directions = torch.nn.functional.normalize(physical_xy, dim=-1)
        alignment = (marker_directions * physical_leg_directions).sum(dim=-1)
        self.assertTrue(torch.allclose(
            alignment, torch.ones(6, dtype=torch.float64), atol=1e-12,
        ))
        reversed_alignment = ((-marker_directions) * physical_leg_directions).sum(-1)
        self.assertTrue(bool((reversed_alignment < 0.0).all()))
        self.assertTrue(torch.allclose(markers[:, 2], origins[:, 2], atol=1e-12))
        self.assertTrue(torch.allclose(
            arcs[:, :, 2], origins[:, None, 2].expand(-1, 17), atol=1e-12,
        ))

    def test_haa_arc_interval_matches_full_arc_when_interval_is_full_range(self):
        current = torch.tensor([0.1, 0.6, -0.6] * 6, dtype=torch.float64)
        ranges = torch.tensor([[-0.4, 0.5]] * 6, dtype=torch.float64)
        _, arcs, _ = GEOM.haa_arc_geometry(
            self.kinematics, current, ranges, radius=0.2, samples=17,
        )
        for leg in range(6):
            sub = GEOM.haa_arc_geometry_interval(
                self.kinematics, current, ranges, leg, -0.4, 0.5,
                radius=0.2, samples=17,
            )
            self.assertEqual(sub.shape, (17, 3))
            self.assertTrue(torch.allclose(sub, arcs[leg], atol=1e-12))

    def test_haa_arc_interval_sub_range_stays_on_the_hip_arc(self):
        current = torch.tensor([0.1, 0.6, -0.6] * 6, dtype=torch.float64)
        ranges = torch.tensor([[-0.4, 0.5]] * 6, dtype=torch.float64)
        origins, _, _ = GEOM.haa_arc_geometry(
            self.kinematics, current, ranges, radius=0.2, samples=17,
        )
        sub = GEOM.haa_arc_geometry_interval(
            self.kinematics, current, ranges, 3, -0.2, 0.3,
            radius=0.2, samples=17,
        )
        self.assertEqual(sub.shape, (17, 3))
        self.assertTrue(torch.allclose(
            (sub - origins[3]).norm(dim=-1),
            torch.full((17,), 0.2, dtype=torch.float64), atol=1e-12,
        ))
        self.assertTrue(torch.allclose(
            sub[:, 2], origins[3, 2].expand(17), atol=1e-12,
        ))
        angle0 = torch.atan2(sub[0, 1] - origins[3, 1], sub[0, 0] - origins[3, 0])
        angle1 = torch.atan2(sub[-1, 1] - origins[3, 1], sub[-1, 0] - origins[3, 0])
        sweep = abs((angle1 - angle0 + torch.pi) % (2.0 * torch.pi) - torch.pi)
        self.assertGreater(float(sweep), 0.1)
        with self.assertRaises(ValueError):
            GEOM.haa_arc_geometry_interval(
                self.kinematics, current, ranges, 3, 0.3, -0.2,
                radius=0.2, samples=17,
            )

    def test_joint_arc_geometry_interval_haa_matches_haa_arc_interval(self):
        current = torch.tensor([0.1, 0.6, -0.6] * 6, dtype=torch.float64)
        ranges = torch.tensor([[-0.4, 0.5]] * 6, dtype=torch.float64)
        sub_new = GEOM.joint_arc_geometry_interval(
            self.kinematics, current, 3, "HAA", -0.4, 0.5, radius=0.2, samples=17,
        )
        sub_old = GEOM.haa_arc_geometry_interval(
            self.kinematics, current, ranges, 3, -0.4, 0.5, radius=0.2, samples=17,
        )
        self.assertEqual(sub_new.shape, (17, 3))
        self.assertTrue(torch.allclose(sub_new, sub_old, atol=1e-12))

    def test_joint_arc_geometry_interval_hfe_kfe_shapes(self):
        current = torch.tensor([0.1, 0.6, -0.6] * 6, dtype=torch.float64)
        for kind in ("HFE", "KFE"):
            for leg in (0, 3):
                sub = GEOM.joint_arc_geometry_interval(
                    self.kinematics, current, leg, kind, -2.0, -1.0,
                    radius=0.2, samples=17,
                )
                self.assertEqual(sub.shape, (17, 3))
                self.assertTrue(torch.isfinite(sub).all())
        with self.assertRaises(ValueError):
            GEOM.joint_arc_geometry_interval(
                self.kinematics, current, 0, "FOO", -2.0, -1.0,
            )

    def test_demo_preset_is_deterministic_and_has_numeric_haa_ranges(self):
        directions = KE.support_directions(24)
        lower, upper = self.kinematics.joint_limits(soft_fraction=0.9)
        current = torch.tensor([0.0, 0.6, -0.6] * 6)
        half = torch.tensor([0.45, 0.15, 0.12] * 6)
        first = GEOM.build_demo_preset(
            "test", self.kinematics, directions, current, half, lower, upper,
            support_margin=0.10, accent_rgb=(0.0, 0.5, 0.5), seed=11,
            candidate_count=65, validation_count=33, box_validation_samples=32,
        )
        second = GEOM.build_demo_preset(
            "test", self.kinematics, directions, current, half, lower, upper,
            support_margin=0.10, accent_rgb=(0.0, 0.5, 0.5), seed=11,
            candidate_count=65, validation_count=33, box_validation_samples=32,
        )
        self.assertEqual(first.haa_ranges.shape, (6, 2))
        self.assertTrue(torch.isfinite(first.haa_ranges).all())
        self.assertTrue(torch.all(first.haa_ranges[:, 0] <= first.haa_ranges[:, 1]))
        self.assertTrue(torch.equal(first.haa_ranges, second.haa_ranges))
        self.assertTrue(torch.equal(first.reachable_support, second.reachable_support))


class TestAccessibleIntervalComplement(unittest.TestCase):
    def test_no_rejection_keeps_full_range(self):
        self.assertEqual(
            GEOM.accessible_interval_complement(-3.0, 3.0, ()),
            ((-3.0, 3.0),),
        )

    def test_single_rejected_band_splits_range(self):
        self.assertEqual(
            GEOM.accessible_interval_complement(-3.0, 3.0, ((1.05, 2.01),)),
            ((-3.0, 1.05), (2.01, 3.0)),
        )

    def test_multiple_bands_are_complementary_no_overlap(self):
        lo, hi = -3.0, 3.0
        rejected = ((-2.5, -2.0), (0.0, 0.4), (1.1, 2.2))
        accessible = GEOM.accessible_interval_complement(lo, hi, rejected)
        # accessible tiles [lo, hi] exactly with the rejected bands, disjoint.
        accessible_span = sum(b - a for a, b in accessible)
        rejected_span = sum(b - a for a, b in rejected)
        self.assertAlmostEqual(accessible_span + rejected_span, hi - lo, places=9)
        for a_lo, a_hi in accessible:
            for r_lo, r_hi in rejected:
                self.assertFalse(a_lo < r_hi and r_lo < a_hi)  # no overlap
        self.assertAlmostEqual(accessible[0][0], lo, places=9)
        self.assertAlmostEqual(accessible[-1][1], hi, places=9)

    def test_rejected_bands_outside_range_are_ignored(self):
        self.assertEqual(
            GEOM.accessible_interval_complement(-1.5, 2.4, ((-3.0, -2.0), (2.5, 3.0))),
            ((-1.5, 2.4),),
        )

    def test_adjacent_bands_merge_accessible_gap(self):
        accessible = GEOM.accessible_interval_complement(
            0.0, 3.0, ((0.5, 1.0), (1.0, 2.0)),
        )
        self.assertEqual(accessible, ((0.0, 0.5), (2.0, 3.0)))


if __name__ == "__main__":
    unittest.main()
