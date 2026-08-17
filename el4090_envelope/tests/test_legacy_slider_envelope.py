import sys
import unittest
from pathlib import Path

import torch


DISTRIBUTION_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = DISTRIBUTION_ROOT / "examples" / "isaac_gym"
sys.path.insert(0, str(EXAMPLE_DIR))

import el4090_envelope as KE
import legacy_slider_envelope as LEGACY
import lidar_free_envelope as LIDAR


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

    def test_exact_symmetric_rectangle_parameterization(self):
        # ENV-RECT-CONTROL-012: the legacy border is the four-corner rectangle
        # described by the five parameters (the bounding box of the old
        # six-point hexagon), so the envelope inflates toward the rectangle.
        parameters = torch.tensor([0.4, 0.5, 0.45, 0.8, -0.7])
        expected = torch.tensor((
            (0.8, 0.5), (-0.7, 0.5), (-0.7, -0.5), (0.8, -0.5),
        ))
        self.assertTrue(torch.equal(LEGACY.legacy_border_vertices(parameters), expected))
        self.assertTrue(torch.allclose(
            LEGACY.legacy_border_support(parameters, self.directions),
            (expected @ self.directions.T).amax(dim=0),
            atol=1e-6,
        ))

    def test_each_raw_return_is_the_border_support_point(self):
        parameters = LEGACY.parameter_tensor([0.4, 0.5, 0.45, 0.8, -0.7])
        directions = KE.support_directions(8)
        points = LEGACY.sample_border_points(parameters, directions)
        corners = LEGACY.legacy_border_vertices(parameters)
        expected_support = LEGACY.legacy_border_support(parameters, directions)
        # Diagonal rays have a unique supporting corner (no argmax ties): 45 deg
        # -> front/right corner, 135 deg -> back/left, 225 deg -> back/right,
        # 315 deg -> front/right.
        self.assertTrue(torch.equal(points[1], corners[0]))
        self.assertTrue(torch.equal(points[3], corners[1]))
        self.assertTrue(torch.equal(points[5], corners[2]))
        self.assertTrue(torch.equal(points[7], corners[3]))
        # Every return is a border corner and its projection onto its own ray is
        # exactly the rectangle's support function along that ray.
        self.assertTrue(bool(((points[:, None, :] == corners[None, :, :]).any(dim=1)).all()))
        projection = (points * directions).sum(-1)
        self.assertTrue(torch.allclose(projection, expected_support, atol=1e-6))

    def test_rectangle_envelope_inflates_toward_bounding_box(self):
        # ENV-RECT-CONTROL-012 regression: the rectangle-border envelope must be
        # measurably closer to the border's bounding box than the old fusiform
        # hexagon envelope, while every border-constrained return still clears
        # the envelope by the declared point clearance.
        parameters = LEGACY.parameter_tensor(LEGACY.LEGACY_MAXIMUM)
        directions = self.directions
        reference = torch.full((48,), 1.4)
        front, middle, back, forward, backward = parameters.unbind()
        zero = torch.zeros((), dtype=parameters.dtype, device=parameters.device)
        hexagon = torch.stack((
            torch.stack((forward, front)), torch.stack((zero, middle)),
            torch.stack((backward, back)), torch.stack((backward, -back)),
            torch.stack((zero, -middle)), torch.stack((forward, -front)),
        ))
        width = max(
            float(front), float(middle), float(back),
        )
        rect = torch.tensor((
            (float(forward), width), (float(backward), width),
            (float(backward), -width), (float(forward), -width),
        ))
        rect_support = (rect @ directions.T).amax(dim=0)

        # Old envelope: one return per ray at the hexagon boundary radius.
        def hexagon_radius(ray):
            cross2 = lambda a, b: a[0] * b[1] - a[1] * b[0]
            starts, edges = hexagon, torch.roll(hexagon, -1, dims=0) - hexagon
            denominator = cross2(ray, edges)
            safe = denominator.abs() > 1e-9
            t = cross2(starts, edges) / torch.where(
                safe, denominator, torch.ones_like(denominator),
            )
            u = cross2(starts, ray) / torch.where(
                safe, denominator, torch.ones_like(denominator),
            )
            valid = safe & (t >= 0.0) & (u >= -1e-6) & (u <= 1.0 + 1e-6)
            return t.masked_fill(~valid, torch.inf).amin(dim=0)

        old_radii = torch.stack([
            hexagon_radius(direction) for direction in directions.unbind(0)
        ])
        old_support = torch.minimum(old_radii, reference) - 0.02

        # New envelope through the public cloud path.
        cloud = LEGACY.legacy_border_lidar_cloud(
            parameters, directions, torch.full((48,), 0.2), reference,
            seed=4090,
        )
        envelope = LIDAR.maximum_sector_point_free_envelope(
            cloud, directions, point_clearance=0.02, cap_support=reference,
        )
        new_support = envelope.support_m

        corner = int(torch.argmin(
            (directions - torch.tensor([0.70710678, 0.70710678])).norm(dim=-1),
        ))
        self.assertGreater(
            float(new_support[corner]), float(old_support[corner]),
        )
        self.assertLess(
            float((new_support - rect_support).abs().mean()),
            float((old_support - rect_support).abs().mean()),
        )
        self.assertLessEqual(float((new_support - rect_support).abs().mean()), 0.025)
        clearances = LIDAR.assigned_point_clearances(
            cloud, directions, new_support,
        )
        self.assertGreaterEqual(float(clearances.min()), 0.02 - 2e-6)

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
