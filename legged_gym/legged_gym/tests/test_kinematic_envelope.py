import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "envelop" / "kinematic_envelope.py"
SPEC = importlib.util.spec_from_file_location("kinematic_envelope_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
KE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = KE
SPEC.loader.exec_module(KE)
URDF = ROOT.parent / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"


def numpy_transform(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def numpy_fk(joints, q):
    q_index = {name: i for i, name in enumerate(KE.EL4090_JOINT_NAMES)}
    result = {"BASE": np.eye(4)}
    remaining = list(joints)
    while remaining:
        ready = [joint for joint in remaining if joint.parent in result]
        if not ready:
            raise RuntimeError("disconnected graph")
        for joint in ready:
            local = numpy_transform(joint.origin_xyz, joint.origin_rpy)
            if joint.joint_type == "revolute":
                axis = np.asarray(joint.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = q[q_index[joint.name]]
                rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew
                angle_tf = np.eye(4)
                angle_tf[:3, :3] = rotation
                local = local @ angle_tf
            result[joint.child] = result[joint.parent] @ local
            remaining.remove(joint)
    return result


class TestKinematicEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joints = KE.load_urdf_joints(URDF)
        cls.kin = KE.BatchedUrdfKinematics(cls.joints)

    def test_fk_matches_independent_float64_oracle(self):
        poses = torch.tensor([
            [0.0, 0.6, -0.6] * 6,
            [-1.308, 1.0, -0.608] * 6,
        ], dtype=torch.float64)
        actual = self.kin.forward(poses)
        for batch, pose in enumerate(poses.numpy()):
            expected = numpy_fk(self.joints, pose)
            for link in ("LB_FOOT", "LF_FOOT", "LM_FOOT", "RB_FOOT", "RF_FOOT", "RM_FOOT"):
                np.testing.assert_allclose(actual[link][batch].numpy(), expected[link], atol=2e-12, rtol=2e-12)

    def test_shapes_dtype_limits_and_capsule_containment(self):
        q = torch.zeros(7, 18, dtype=torch.float64)
        directions = KE.support_directions(32, dtype=q.dtype)
        capsules = KE.default_el4090_capsules()
        support = KE.capsule_support(self.kin, q, capsules, directions)
        self.assertEqual(support.shape, (7, 32))
        self.assertEqual(support.dtype, q.dtype)
        links = [capsule.link for capsule in capsules]
        local = q.new_tensor([[capsule.start, capsule.end] for capsule in capsules])
        endpoints = self.kin.points(q, links, local)
        radii = q.new_tensor([capsule.radius for capsule in capsules])
        projections = torch.einsum("blpi,ki->blpk", endpoints[..., :2], directions) + radii[None, :, None, None]
        self.assertTrue(torch.all(projections <= support[:, None, None, :] + 1e-12))
        lower, upper = self.kin.joint_limits(soft_fraction=0.9, dtype=q.dtype)
        self.assertTrue(torch.allclose(lower, torch.full_like(lower, -2.7)))
        self.assertTrue(torch.allclose(upper, torch.full_like(upper, 2.7)))
        support32 = KE.capsule_support(self.kin, q.float(), capsules, directions.float()).double()
        self.assertLess(float((support - support32).abs().max()), 2e-6)

    def test_margin_and_point_queries(self):
        q = torch.zeros(2, 18)
        directions = KE.support_directions(16)
        support = KE.capsule_support(self.kin, q, KE.default_el4090_capsules(), directions)
        safe = KE.add_support_margin(support, 0.125)
        self.assertTrue(torch.equal(safe - support, torch.full_like(support, 0.125)))
        origin = torch.zeros(2, 1, 3)
        self.assertTrue(torch.all(KE.contains_points(origin, directions, support)))
        far = torch.tensor([[[10.0, 0.0, 0.0]], [[0.0, -10.0, 0.0]]])
        self.assertFalse(torch.any(KE.contains_points(far, directions, support)))

    def test_gradients_match_finite_difference_away_from_ties(self):
        q = torch.linspace(-0.31, 0.43, 18, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
        directions = KE.support_directions(19, dtype=q.dtype)
        value = KE.capsule_support(self.kin, q, KE.default_el4090_capsules(), directions)[0, 3]
        value.backward()
        nonzero = torch.nonzero(q.grad[0].abs() > 1e-7).flatten()
        self.assertGreater(nonzero.numel(), 0)
        index = int(nonzero[0])
        eps = 1e-6
        plus, minus = q.detach().clone(), q.detach().clone()
        plus[0, index] += eps
        minus[0, index] -= eps
        fd = (
            KE.capsule_support(self.kin, plus, KE.default_el4090_capsules(), directions)[0, 3]
            - KE.capsule_support(self.kin, minus, KE.default_el4090_capsules(), directions)[0, 3]
        ) / (2 * eps)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertAlmostEqual(float(q.grad[0, index]), float(fd), delta=2e-6)

    def test_reachable_support_and_deterministic_sampling(self):
        lower, upper = self.kin.joint_limits(soft_fraction=0.2)
        samples_a = KE.deterministic_joint_samples(lower, upper, 65, seed=17)
        samples_b = KE.deterministic_joint_samples(lower, upper, 65, seed=17)
        self.assertTrue(torch.equal(samples_a, samples_b))
        support = KE.reachable_foot_support(self.kin, samples_a.unsqueeze(0), KE.support_directions(24))
        self.assertEqual(support.shape, (1, 24))

    def test_range_diagnostics_conservative_and_approximate(self):
        lower = torch.full((18,), -2.7)
        upper = torch.full((18,), 2.7)
        train = KE.deterministic_joint_samples(lower, upper, 128, seed=2)
        export = KE.export_joint_ranges(train, train[2:], lower, upper)
        self.assertEqual(export.lower.shape, (18,))
        self.assertTrue(torch.all(export.half_range >= 0))
        self.assertTrue(export.diagnostics.conservative_on_validation)
        outside = train[2:].clone()
        outside[0, 0] = 2.8
        approximate = KE.export_joint_ranges(train, outside, lower, upper)
        self.assertFalse(approximate.diagnostics.conservative_on_validation)
        self.assertEqual(approximate.diagnostics.label, "approximate")
        self.assertGreater(approximate.diagnostics.violation_rate, 0.0)

    def test_legacy_contract_and_haa_remap(self):
        batch = 5
        directions = KE.support_directions(32)
        support = KE.capsule_support(self.kin, torch.zeros(batch, 18), KE.default_el4090_capsules(), directions)
        priors = torch.tensor([[0.1, 0.2, 0.3]]).repeat(batch, 1)
        condition = KE.legacy_condition_from_support(support, directions, morphology_priors=priors)
        self.assertEqual(condition.shape, (batch, 8))
        self.assertTrue(torch.equal(condition[:, 5:8], priors))
        self.assertTrue(torch.all(condition[:, :5] >= condition.new_tensor((0.15, 0.15, 0.15, 0.25, -0.60))))
        self.assertTrue(torch.all(condition[:, :5] <= condition.new_tensor((0.60, 0.60, 0.60, 0.60, -0.25))))
        lower, upper = self.kin.joint_limits(soft_fraction=0.9)
        samples = KE.deterministic_joint_samples(lower, upper, 32)
        export = KE.export_joint_ranges(samples, samples, lower, upper)
        haa = KE.haa_ranges_from_joint_export(export)
        self.assertEqual(haa.shape, (6, 2))
        haa_batch = haa.unsqueeze(0).repeat(batch, 1, 1)
        obs = KE.append_legacy_envelop2_observation(torch.zeros(batch, 66), condition, haa_batch, torch.zeros(batch))
        self.assertEqual(obs.shape, (batch, 83))
        self.assertTrue(torch.equal(obs[:, 66:69], priors))
        legacy = KE.haa_ranges_from_joint_export(export, output_leg_order=KE.LEGACY_HAA_ORDER)
        remap = [KE.LEGACY_HAA_ORDER.index(leg) for leg in KE.EL4090_LEG_NAMES]
        self.assertTrue(torch.equal(legacy[remap], haa))

        symmetric = KE.capsule_support(
            self.kin,
            torch.zeros(1, 18, dtype=torch.float64),
            KE.default_el4090_capsules(),
            KE.support_directions(64, dtype=torch.float64),
        )[0]
        reflected_indices = torch.remainder(-torch.arange(64), 64)
        self.assertLess(float((symmetric - symmetric[reflected_indices]).abs().max()), 5e-6)

        batched = KE.export_joint_ranges(samples.repeat(batch, 1, 1), samples.repeat(batch, 1, 1), lower, upper)
        self.assertEqual(KE.haa_ranges_from_joint_export(batched).shape, (batch, 6, 2))


if __name__ == "__main__":
    unittest.main()
