import json
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

import el4090_envelope as envelope


REPOSITORY = Path(__file__).resolve().parents[2]
URDF = REPOSITORY / "legged_gym" / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"


class TestPublicApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = envelope.BatchedUrdfKinematics(envelope.load_urdf_joints(URDF))

    def test_root_exports_and_serialization(self):
        self.assertEqual(len(envelope.EL4090_JOINT_NAMES), 18)
        directions = envelope.support_directions(8)
        support = envelope.capsule_support(
            self.kinematics, torch.zeros(1, 18),
            envelope.default_el4090_capsules(), directions,
        )
        self.assertEqual(tuple(support.shape), (1, 8))
        proxy = envelope.default_el4090_capsules()[0]
        self.assertEqual(json.loads(json.dumps(asdict(proxy)))["name"], "base_x")

    def test_degenerate_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            envelope.support_directions(2)
        with self.assertRaises(ValueError):
            self.kinematics.forward(torch.zeros(18))

    def test_infeasible_reference_is_honest(self):
        directions = envelope.support_directions(16)
        lower, upper = self.kinematics.joint_limits()
        result = envelope.joint_rejection_ranges(
            self.kinematics, envelope.default_el4090_capsules(), directions,
            torch.full((16,), -100.0), lower, upper, torch.zeros(18),
        )
        self.assertFalse(result.feasible_reference)
        self.assertIsNone(result.reference_q)
        self.assertEqual(result.rejected_joint_count, 0)


if __name__ == "__main__":
    unittest.main()
