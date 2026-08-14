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


if __name__ == "__main__":
    unittest.main()
