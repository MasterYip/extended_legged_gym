import importlib.util
import sys
import unittest
from pathlib import Path

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
LEGACY = load_module("legacy_slider_envelope", ENVELOPE_DIR / "legacy_slider_envelope.py")
URDF = ROOT.parent / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"


class TestLegacySliderEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = KE.BatchedUrdfKinematics(KE.load_urdf_joints(URDF))
        cls.lower, cls.upper = cls.kinematics.joint_limits(soft_fraction=0.9)
        anchor = torch.tensor([0.0, 0.60, -0.60] * 6)
        half = torch.tensor([0.80, 0.34, 0.40] * 6)
        cls.candidate_lower = torch.maximum(anchor - half, cls.lower)
        cls.candidate_upper = torch.minimum(anchor + half, cls.upper)
        cls.candidates = torch.cat((
            anchor.unsqueeze(0),
            KE.deterministic_joint_samples(
                cls.candidate_lower, cls.candidate_upper, 1536, seed=4090,
            ),
        ))
        cls.validation = KE.deterministic_joint_samples(
            cls.candidate_lower, cls.candidate_upper, 257, seed=4190,
        )

    def compute(self, values):
        return LEGACY.compute_legacy_admissible_envelope(
            self.kinematics,
            LEGACY.parameter_tensor(values),
            self.candidates,
            self.validation,
            self.lower,
            self.upper,
            box_validation_samples=128,
        )

    def test_parameter_contract_matches_zhanght_branch(self):
        self.assertEqual(
            LEGACY.LEGACY_PARAMETER_ORDER,
            ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit"),
        )
        expected = {
            "front_width": (0.3, 0.6), "middle_width": (0.3, 0.7),
            "back_width": (0.3, 0.6), "forward_limit": (0.6, 0.9),
            "backward_limit": (-0.9, -0.6),
        }
        self.assertEqual(LEGACY.LEGACY_PARAMETER_RANGES, expected)
        self.assertEqual(LEGACY.LEGACY_MIDPOINT, (0.45, 0.50, 0.45, 0.75, -0.75))
        self.assertEqual(LEGACY.LEGACY_MAXIMUM, (0.60, 0.70, 0.60, 0.90, -0.90))

    def test_exact_symmetric_six_vertex_parameterization(self):
        values = torch.tensor([0.4, 0.5, 0.45, 0.8, -0.7])
        expected = torch.tensor((
            (0.8, 0.4), (0.0, 0.5), (-0.7, 0.45),
            (-0.7, -0.45), (0.0, -0.5), (0.8, -0.4),
        ))
        self.assertTrue(torch.equal(LEGACY.legacy_border_vertices(values), expected))

    def test_piecewise_membership_preserves_concave_middle(self):
        values = LEGACY.parameter_tensor([0.6, 0.3, 0.6, 0.9, -0.9])
        points = torch.tensor([[
            [-0.8, 0.5], [0.8, 0.5], [0.0, 0.5], [0.0, 0.25],
        ]])
        excess = LEGACY.legacy_foot_excess(points, values)[0]
        self.assertLessEqual(float(excess[0]), 0.0)
        self.assertLessEqual(float(excess[1]), 0.0)
        self.assertGreater(float(excess[2]), 0.19)
        self.assertLessEqual(float(excess[3]), 0.0)

    def test_maximal_halves_current_feet_and_export_are_contained(self):
        values = [0.55, 0.65, 0.55, 0.85, -0.85]
        parameters = LEGACY.parameter_tensor(values)
        result = self.compute(values)
        self.assertGreater(int(result.feasible_mask.sum()), 0)
        self.assertTrue(bool(LEGACY.legacy_feasible(
            result.current_feet_xy.unsqueeze(0), parameters,
        )[0]))
        for hull in (
            result.maximal_rear_vertices_xy, result.maximal_front_vertices_xy,
        ):
            self.assertGreaterEqual(hull.shape[0], 3)
            self.assertLessEqual(float(
                LEGACY.legacy_foot_excess(hull.unsqueeze(0), parameters).max()
            ), 1e-6)
        self.assertTrue(torch.isfinite(result.range_export.lower).all())
        self.assertTrue(torch.isfinite(result.range_export.upper).all())
        self.assertTrue(bool((result.range_export.lower <= result.range_export.upper).all()))
        self.assertEqual(result.box_validation_samples, 128)
        self.assertGreaterEqual(result.box_foot_violation_count, 0)
        self.assertGreaterEqual(result.max_box_foot_violation_m, 0.0)

    def test_expanding_all_legacy_bounds_does_not_remove_feasible_samples(self):
        compact = self.compute([0.40, 0.42, 0.40, 0.68, -0.68])
        maximum = self.compute(LEGACY.LEGACY_MAXIMUM)
        self.assertGreaterEqual(
            int(maximum.feasible_mask.sum()), int(compact.feasible_mask.sum()),
        )
        self.assertFalse(bool((compact.feasible_mask & ~maximum.feasible_mask).any()))

    def test_recompute_is_deterministic_and_invalid_parameters_fail(self):
        first = self.compute(LEGACY.LEGACY_MIDPOINT)
        second = self.compute(LEGACY.LEGACY_MIDPOINT)
        self.assertTrue(torch.equal(first.feasible_mask, second.feasible_mask))
        self.assertTrue(torch.equal(first.range_export.lower, second.range_export.lower))
        self.assertTrue(torch.equal(
            first.maximal_front_vertices_xy, second.maximal_front_vertices_xy,
        ))
        with self.assertRaises(ValueError):
            LEGACY.parameter_tensor([0.2, 0.5, 0.5, 0.8, -0.8])

    def test_visual_semantics_keep_three_foot_envelopes_distinct(self):
        semantics = LEGACY.LEGACY_VISUAL_SEMANTICS
        self.assertIn("hard outer foot-workspace", semantics["white"])
        self.assertIn("front/rear foot workspace", semantics["light_cyan"])
        self.assertIn("rear/front six-foot hulls", semantics["dark_teal"])
        self.assertEqual(semantics["red"], "legacy foot-workspace violation only")


if __name__ == "__main__":
    unittest.main()
