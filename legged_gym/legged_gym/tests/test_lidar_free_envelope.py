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
LFE = load_module("lidar_free_envelope", ENVELOPE_DIR / "lidar_free_envelope.py")
URDF = ROOT.parent / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"


class TestLidarFreeEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = KE.BatchedUrdfKinematics(KE.load_urdf_joints(URDF))
        cls.directions = KE.support_directions(48)
        cls.capsules = KE.default_el4090_capsules()
        cls.anchor = torch.tensor([0.0, 0.60, -0.60] * 6)
        cls.anchor_support = KE.capsule_support(
            cls.kinematics, cls.anchor.unsqueeze(0), cls.capsules, cls.directions,
        )[0]
        effective_lower, effective_upper = cls.kinematics.joint_limits(soft_fraction=0.9)
        half_width = torch.tensor([0.95, 0.42, 0.48] * 6)
        cls.candidate_lower = torch.maximum(cls.anchor - half_width, effective_lower)
        cls.candidate_upper = torch.minimum(cls.anchor + half_width, effective_upper)
        cls.candidate_q = torch.cat((
            cls.anchor.unsqueeze(0),
            KE.deterministic_joint_samples(
                cls.candidate_lower, cls.candidate_upper, 768, seed=4190,
            ),
        ))
        cls.reference_support = KE.reachable_foot_support(
            cls.kinematics, cls.candidate_q.unsqueeze(0), cls.directions,
        )[0]

    def make_cloud(self, seed=4090):
        return LFE.generate_synthetic_lidar_cloud(
            self.directions, self.anchor_support, self.reference_support,
            count=192, seed=seed, min_radius=0.0, max_radius=2.10,
            robot_clearance=0.05, reference_containment_margin=0.005,
        )

    def test_cloud_is_deterministic_structured_and_clear_of_robot(self):
        first = self.make_cloud()
        second = self.make_cloud()
        changed = self.make_cloud(seed=4091)
        self.assertTrue(torch.equal(first.points_xy, second.points_xy))
        self.assertFalse(torch.equal(first.points_xy, changed.points_xy))
        self.assertEqual(first.points_xy.shape, (192, 2))
        self.assertTrue(bool((first.radii_m >= first.ray_inner_radius_m).all()))
        self.assertTrue(bool((first.radii_m <= first.ray_outer_radius_m).all()))
        self.assertTrue(bool((first.radii_m <= 2.10).all()))
        self.assertTrue(torch.equal(torch.unique(first.sector_indices), torch.arange(48)))
        projection = (first.points_xy * self.directions[first.sector_indices]).sum(-1)
        clearance = projection - self.anchor_support[first.sector_indices]
        self.assertGreaterEqual(float(clearance.min()), 0.05 - 2e-6)
        baseline_excess = LFE.polygon_support_excess(
            first.points_xy, self.directions, self.anchor_support,
        )
        self.assertGreaterEqual(float(baseline_excess.min()), 0.05 - 2e-6)
        reference_excess = LFE.polygon_support_excess(
            first.points_xy, self.directions, self.reference_support,
        )
        self.assertLessEqual(float(reference_excess.max()), -0.005 + 2e-6)

    def test_restricted_family_envelope_is_point_free_and_maximal(self):
        cloud = self.make_cloud()
        envelope = LFE.maximum_sector_point_free_envelope(
            cloud, self.directions, point_clearance=0.02,
        )
        clearances = LFE.assigned_point_clearances(
            cloud, self.directions, envelope.support_m,
        )
        self.assertGreaterEqual(float(clearances.min()), 0.02 - 2e-6)
        limiting = clearances[envelope.limiting_point_indices]
        self.assertTrue(torch.allclose(limiting, torch.full((48,), 0.02), atol=2e-6))
        self.assertLessEqual(
            float((envelope.support_m - self.reference_support).max()), 1e-6,
        )

        # Raising any face breaks clearance at that face's active return.
        raised = envelope.support_m.unsqueeze(0).repeat(48, 1)
        raised[torch.arange(48), torch.arange(48)] += 1e-4
        points = cloud.points_xy[envelope.limiting_point_indices]
        projection = (points * self.directions).sum(-1)
        self.assertTrue(bool((projection - raised.diagonal() < 0.02).all()))

    def test_obstacles_materially_reduce_candidates_and_joint_ranges(self):
        cloud = self.make_cloud()
        envelope = LFE.maximum_sector_point_free_envelope(
            cloud, self.directions, point_clearance=0.02,
        )
        support = KE.capsule_support(
            self.kinematics, self.candidate_q, self.capsules, self.directions,
        )
        feasible = (support <= envelope.support_m.unsqueeze(0) + 1e-6).all(-1)
        reduction = 1.0 - float(feasible.sum()) / self.candidate_q.shape[0]
        self.assertGreaterEqual(reduction, 0.05)

        validation_q = KE.deterministic_joint_samples(
            self.candidate_lower, self.candidate_upper, 257, seed=4290,
        )
        lower, upper = self.kinematics.joint_limits(soft_fraction=0.9)
        export = KE.export_envelope_joint_ranges(
            self.kinematics,
            self.candidate_q,
            validation_q,
            self.directions,
            envelope.support_m,
            lower,
            upper,
            capsules=self.capsules,
            box_validation_samples=256,
            box_validation_seed=4390,
        )
        shrinkage = (self.candidate_upper - self.candidate_lower) - (
            export.upper - export.lower
        )
        self.assertTrue(bool(export.valid))
        self.assertGreaterEqual(float(shrinkage.max()), 0.03)

    def test_backtracking_rejects_naive_box_pose_and_returns_compliant_pose(self):
        lower, upper = self.kinematics.joint_limits(soft_fraction=0.9)
        proposed = KE.deterministic_joint_samples(lower, upper, 256, seed=881)
        allowed = self.anchor_support + 0.025
        excess = LFE.envelope_excess(
            self.kinematics, proposed, self.capsules, self.directions, allowed,
        )
        violating = proposed[excess.argmax()].unsqueeze(0)
        self.assertGreater(float(excess.max()), 0.01)
        accepted = LFE.backtrack_to_feasible_anchor(
            self.kinematics, violating, self.anchor, lower, upper,
            self.capsules, self.directions, allowed,
        )
        self.assertLess(float(accepted.accepted_scale[0]), 1.0)
        self.assertLessEqual(float(accepted.envelope_excess_m[0]), 1e-6)
        self.assertEqual(float(accepted.joint_excess_rad[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
