import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_DIR = ROOT / "utils" / "envelop"
sys.path.insert(0, str(ENVELOPE_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KE = load_module("kinematic_envelope", ENVELOPE_DIR / "kinematic_envelope.py")
LIDAR = load_module("lidar_free_envelope", ENVELOPE_DIR / "lidar_free_envelope.py")
LEGACY = load_module("legacy_slider_envelope", ENVELOPE_DIR / "legacy_slider_envelope.py")


class TestLegacySliderPointSource(unittest.TestCase):
    def setUp(self):
        self.directions = KE.support_directions(48)
        self.baseline = torch.full((48,), 0.20)

    def test_parameter_contract_matches_zhanght_branch(self):
        self.assertEqual(
            LEGACY.LEGACY_PARAMETER_ORDER,
            ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit"),
        )
        self.assertEqual(LEGACY.LEGACY_PARAMETER_RANGES, {
            "front_width": (0.3, 0.6), "middle_width": (0.3, 0.7),
            "back_width": (0.3, 0.6), "forward_limit": (0.6, 0.9),
            "backward_limit": (-0.9, -0.6),
        })
        self.assertEqual(LEGACY.LEGACY_MIDPOINT, (0.45, 0.50, 0.45, 0.75, -0.75))
        self.assertEqual(LEGACY.LEGACY_MAXIMUM, (0.60, 0.70, 0.60, 0.90, -0.90))

    def test_exact_symmetric_six_vertex_parameterization(self):
        parameters = torch.tensor([0.4, 0.5, 0.45, 0.8, -0.7])
        expected = torch.tensor((
            (0.8, 0.4), (0.0, 0.5), (-0.7, 0.45),
            (-0.7, -0.45), (0.0, -0.5), (0.8, -0.4),
        ))
        self.assertTrue(torch.equal(LEGACY.legacy_border_vertices(parameters), expected))

    def test_each_raw_return_is_the_nearest_border_intersection(self):
        parameters = LEGACY.parameter_tensor([0.4, 0.5, 0.45, 0.8, -0.7])
        directions = KE.support_directions(8)
        points = LEGACY.sample_border_points(parameters, directions)
        self.assertTrue(torch.allclose(points[0], torch.tensor([0.8, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(points[2], torch.tensor([0.0, 0.5]), atol=1e-6))
        self.assertTrue(torch.allclose(points[4], torch.tensor([-0.7, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(points[6], torch.tensor([0.0, -0.5]), atol=1e-6))
        cross = points[:, 0] * directions[:, 1] - points[:, 1] * directions[:, 0]
        self.assertLessEqual(float(cross.abs().max()), 1e-6)
        self.assertTrue(bool(((points * directions).sum(-1) > 0.0).all()))

    def test_cloud_is_one_return_per_sector_and_inside_reachable_cap(self):
        reference = torch.full((48,), 0.72)
        cloud = LEGACY.legacy_border_lidar_cloud(
            LEGACY.parameter_tensor(LEGACY.LEGACY_MAXIMUM),
            self.directions, self.baseline, reference, seed=4090,
        )
        self.assertTrue(torch.equal(cloud.sector_indices, torch.arange(48)))
        self.assertTrue(torch.equal(cloud.sector_counts, torch.ones(48, dtype=torch.long)))
        excess = LIDAR.polygon_support_excess(cloud.points_xy, self.directions, reference)
        self.assertLessEqual(float(excess.max()), -0.005 + 2e-6)
        self.assertEqual(cloud.reference_containment_margin_m, 0.005)

    def test_slider_changes_only_the_cloud_geometry(self):
        reference = torch.full((48,), 2.0)
        midpoint = LEGACY.legacy_border_lidar_cloud(
            LEGACY.parameter_tensor(LEGACY.LEGACY_MIDPOINT),
            self.directions, self.baseline, reference, seed=1,
        )
        maximum = LEGACY.legacy_border_lidar_cloud(
            LEGACY.parameter_tensor(LEGACY.LEGACY_MAXIMUM),
            self.directions, self.baseline, reference, seed=2,
        )
        self.assertFalse(torch.equal(midpoint.points_xy, maximum.points_xy))
        self.assertTrue(torch.equal(midpoint.sector_indices, maximum.sector_indices))
        self.assertEqual(midpoint.points_xy.shape, maximum.points_xy.shape)

    def test_existing_lidar_optimizer_formula_is_unchanged(self):
        reference = torch.full((48,), 1.4)
        cloud = LEGACY.legacy_border_lidar_cloud(
            LEGACY.parameter_tensor(LEGACY.LEGACY_MIDPOINT),
            self.directions, self.baseline, reference, seed=4090,
        )
        result = LIDAR.maximum_sector_point_free_envelope(
            cloud, self.directions, point_clearance=0.02, cap_support=reference,
        )
        expected = torch.minimum(
            (cloud.points_xy * self.directions).sum(-1) - 0.02,
            reference,
        )
        self.assertTrue(torch.allclose(result.support_m, expected, atol=1e-7))
        self.assertTrue(torch.equal(result.constrained_face_indices, torch.arange(48)))
        self.assertEqual(result.unconstrained_face_indices.numel(), 0)
        self.assertEqual(
            result.optimality_scope,
            "coordinatewise maximum in the declared fixed-normal capped polygon "
            "family under nearest-angular-sector point assignment",
        )

    def test_invalid_parameters_fail_before_cloud_generation(self):
        with self.assertRaises(ValueError):
            LEGACY.parameter_tensor([0.2, 0.5, 0.5, 0.8, -0.8])
        with self.assertRaises(ValueError):
            LEGACY.parameter_tensor([0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
