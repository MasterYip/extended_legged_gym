"""Configuration for the second envelope-conditioned EL4090 environment."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop2Cfg(El4090EnvelopCfg):
    class control(El4090EnvelopCfg.control):
        # The environment overrides the P target with the current morphology
        # preset; no temporal low-pass state is used.
        control_type = "P"

    class haa_swing_range:
        # analytic, monte_carlo, or network
        method = "analytic"
        joint_lower = -3.0
        joint_upper = 3.0
        leg_reach = 0.55
        front_hip_offset = 0.10
        middle_hip_offset = 0.20
        back_hip_offset = 0.10
        spider_swing_limit = 1.05
        mammal_swing_limit = 0.45
        minimum_half_range = 0.05
        monte_carlo_samples = 2048
        monte_carlo_quantile = 0.0
        monte_carlo_seed = None
        network_checkpoint = ""

    class rewards(El4090EnvelopCfg.rewards):
        haa_range_min_command = 0.15
        haa_range_velocity_clip = 5.0
        haa_range_margin = 0.0

        class scales(El4090EnvelopCfg.rewards.scales):
            # Replaced by envelope-derived, per-leg limits below.
            morphology_haa_range = 0.0
            haa_swing = 0.0
            haa_swing_in_range = 0.15
            haa_range_violation = -3.0
            envelope_constraint = -20.0


class El4090Envelop2CfgPPO(El4090EnvelopCfgPPO):
    class algorithm(El4090EnvelopCfgPPO.algorithm):
        class symmetry_cfg(El4090EnvelopCfgPPO.algorithm.symmetry_cfg):
            data_augmentation_func = (
                "legged_gym.envs.el_4090.spider_envelop_2.symmetry:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner(El4090EnvelopCfgPPO.runner):
        experiment_name = "el_4090_envelop_2_p_haa_range"
