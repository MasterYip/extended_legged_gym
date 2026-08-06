"""Configuration for the second envelope-conditioned EL4090 environment."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop2Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # 66 proprioceptive observations; envelope/prior is not policy input.
        num_observations = 66

    class commands(El4090EnvelopCfg.commands):
        # Policy command is strictly [vx, vy, yaw_rate].
        num_commands = 3
        heading_command = False
        condition_dim = 0
        condition_names: list = []

    class envelope:
        condition_names: list = list(El4090EnvelopCfg.commands.condition_names)
        morphology_prior_mode = El4090EnvelopCfg.commands.morphology_prior_mode
        morphology_prior_weights = El4090EnvelopCfg.commands.morphology_prior_weights
        morphology_middle_front_follow_weight = (
            El4090EnvelopCfg.commands.morphology_middle_front_follow_weight
        )

        class ranges:
            front_width = list(El4090EnvelopCfg.commands.ranges.front_width)
            middle_width = list(El4090EnvelopCfg.commands.ranges.middle_width)
            back_width = list(El4090EnvelopCfg.commands.ranges.back_width)
            forward_limit = list(El4090EnvelopCfg.commands.ranges.forward_limit)
            backward_limit = list(El4090EnvelopCfg.commands.ranges.backward_limit)
            morphology_front_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_front_prior
            )
            morphology_middle_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_middle_prior
            )
            morphology_back_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_back_prior
            )

    class control(El4090EnvelopCfg.control):
        # The environment overrides the P target with the current morphology
        # preset; no temporal low-pass state is used.
        control_type = "P"

    class haa_swing_range:
        # analytic, monte_carlo, or network
        method = "network"
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
        network_checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "spider_envelop_2/envelop_network/haa_range.pt"
        )

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
