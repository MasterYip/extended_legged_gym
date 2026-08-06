import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "utils"
    / "envelop"
    / "network"
    / "haa_swing_range.py"
)
SPEC = importlib.util.spec_from_file_location("haa_swing_range_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
AnalyticHaaRangeEstimator = MODULE.AnalyticHaaRangeEstimator
HaaRangeConfig = MODULE.HaaRangeConfig
HaaRangeNetwork = MODULE.HaaRangeNetwork
MonteCarloHaaRangeEstimator = MODULE.MonteCarloHaaRangeEstimator
apply_env_morphology_priors = MODULE.apply_env_morphology_priors
load_envelope_condition_spec = MODULE.load_envelope_condition_spec


def conditions() -> torch.Tensor:
    return torch.tensor(
        [
            [0.30, 0.30, 0.30, 0.60, -0.60, 0.0, 0.0, 0.0],
            [0.60, 0.70, 0.60, 0.90, -0.90, 1.0, 1.0, 1.0],
            [0.45, 0.50, 0.45, 0.75, -0.75, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )


class TestHaaSwingRange(unittest.TestCase):
    def test_envelop_2_checkpoint_keeps_training_io_contract(self) -> None:
        checkpoint = (
            Path(__file__).parents[1]
            / "envs"
            / "el_4090"
            / "spider_envelop_2"
            / "envelop_network"
            / "haa_range.pt"
        )
        network = HaaRangeNetwork.from_checkpoint(checkpoint)
        self.assertEqual(network.config.condition_names, HaaRangeConfig().condition_names)
        self.assertEqual(network.config.leg_names, ("RF", "RM", "RB", "LF", "LM", "LB"))
        ranges = network(conditions()[2:3])
        self.assertEqual(ranges.shape, (1, 6, 2))
        simulator_leg_order = ("LB", "LF", "LM", "RB", "RF", "RM")
        output_indices = [network.config.leg_names.index(name) for name in simulator_leg_order]
        self.assertEqual(output_indices, [5, 3, 4, 2, 0, 1])

    def test_training_distribution_matches_spider_envelop_config(self) -> None:
        spec = load_envelope_condition_spec()
        self.assertEqual(spec.condition_names, HaaRangeConfig().condition_names)
        self.assertEqual(spec.low, (0.3, 0.3, 0.3, 0.6, -0.9, 0.0, 0.0, 0.0))
        self.assertEqual(spec.high, (0.6, 0.7, 0.6, 0.9, -0.6, 1.0, 1.0, 1.0))

    def test_training_priors_follow_environment_formula(self) -> None:
        spec = load_envelope_condition_spec()
        condition = conditions()[2:3]
        transformed = apply_env_morphology_priors(condition, spec)
        expected_front = torch.tensor(0.25 / (0.25 + 0.175))
        expected_middle = 0.4 * expected_front + 0.6 * 0.5
        self.assertTrue(torch.allclose(transformed[0, 5], expected_front))
        self.assertTrue(torch.allclose(transformed[0, 6], expected_middle))
        self.assertTrue(torch.allclose(transformed[0, 7], expected_front))

    def test_analytic_ranges_are_ordered_and_bounded(self) -> None:
        config = HaaRangeConfig()
        ranges = AnalyticHaaRangeEstimator(config)(conditions())
        self.assertEqual(ranges.shape, (3, 6, 2))
        self.assertTrue(torch.all(ranges[..., 0] <= ranges[..., 1]))
        self.assertTrue(torch.all(ranges >= config.joint_lower))
        self.assertTrue(torch.all(ranges <= config.joint_upper))

    def test_envelope_width_and_prior_change_range(self) -> None:
        estimator = AnalyticHaaRangeEstimator()
        ranges = estimator(conditions())
        narrow_spider_width = ranges[0, :, 1] - ranges[0, :, 0]
        wide_mammal_width = ranges[1, :, 1] - ranges[1, :, 0]
        # Both envelope width and morphology allowance participate; at least
        # one leg group must differ for these opposite corner conditions.
        self.assertFalse(torch.allclose(narrow_spider_width, wide_mammal_width))
        centers = ranges.mean(dim=-1)
        self.assertTrue(torch.allclose(centers[0], torch.zeros(6), atol=1e-6))
        self.assertTrue(torch.allclose(centers[1], torch.tensor(HaaRangeConfig().mammal_haa), atol=1e-6))

    def test_monte_carlo_converges_to_analytic_bounds(self) -> None:
        condition = conditions()[2:3]
        analytic = AnalyticHaaRangeEstimator()(condition)
        sampled = MonteCarloHaaRangeEstimator(num_samples=20000, seed=7)(condition)
        self.assertTrue(torch.all(sampled[..., 0] >= analytic[..., 0] - 1e-6))
        self.assertTrue(torch.all(sampled[..., 1] <= analytic[..., 1] + 1e-6))
        self.assertTrue(torch.allclose(sampled, analytic, atol=5e-3))

    def test_network_output_and_checkpoint_contract(self) -> None:
        network = HaaRangeNetwork(hidden_dims=(16,))
        result = network(conditions())
        self.assertEqual(result.shape, (3, 6, 2))
        self.assertTrue(torch.all(result[..., 0] <= result[..., 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "range.pt"
            network.save_checkpoint(path)
            restored = HaaRangeNetwork.from_checkpoint(path)
            self.assertTrue(torch.allclose(result, restored(conditions())))


if __name__ == "__main__":
    unittest.main()
