"""Configuration for the second envelope-conditioned EL4090 environment."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop2Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # 66 proprioception + 3 morphology priors + 6 HAA centers + 6 half-ranges.
        num_observations = 81
        num_physical_priors = 3
        num_haa_range_observations = 12

    class commands(El4090EnvelopCfg.commands):
        # Policy command is strictly [vx, vy, yaw_rate].
        num_commands = 3
        heading_command = False
        condition_dim = 0
        condition_names: list = []

    class normalization(El4090EnvelopCfg.normalization):
        class obs_scales(El4090EnvelopCfg.normalization.obs_scales):
            morphology_prior = 1.0
            haa_range_center = 1.0
            haa_range_half = 1.0

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
        haa_range_margin = 0.0

        class scales:
            # Basic locomotion tracking and body stabilization.
            termination = -0.0
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5
            lin_vel_z = -2
            ang_vel_xy = -0.1
            orientation = -5
            base_height = -50

            # Smooth and energy-efficient joint motion.
            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-7
            action_rate = -0.001

            # Basic foot contact and gait quality.
            feet_slip = -0.05
            feet_air_time = 3
            collision = -1.0
            stand_still = -1.5
            feet_contact_forces = -0.03
            feet_async = -0.1
            feet_sync = -0.1
            tripod_contact_pattern = -1

            # Generic actuator safety limits.
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01

            # Keep only the generated per-leg HAA range constraint here. A
            # separate phase-based reward will be introduced in the next step.
            haa_range_violation = -3.0


class El4090Envelop2CfgPPO(El4090EnvelopCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(El4090EnvelopCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = (
                "legged_gym.envs.el_4090.spider_envelop_2.symmetry:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 3000

        # logging
        save_interval = 50
        experiment_name = "el_4090_envelop_2_p_haa_range"
        run_name = ""

        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
