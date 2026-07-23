from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import (
    El4090SpiderCfg,
    El4090SpiderCfgPPO,
)


class El4090Tripod2LowCfg(El4090SpiderCfg):
    """EL4090 2-group tripod gait on flat terrain — low crouching stance."""

    class init_state(El4090SpiderCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.0,
            "RM_HFE": 0.0,
            "RB_HFE": 0.0,
            "LF_HFE": 0.0,
            "LM_HFE": 0.0,
            "LB_HFE": 0.0,

            "RF_KFE": -0.0,
            "RM_KFE": -0.0,
            "RB_KFE": -0.0,
            "LF_KFE": -0.0,
            "LM_KFE": -0.0,
            "LB_KFE": -0.0,
        }

    class rewards(El4090SpiderCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.25
        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait (with / w\o actuator net)
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 2.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -8.0
            ang_vel_xy = -0.5
            orientation = [-8.0, -8.0]
            torques = -0.0001
            dof_vel = [-0.0001, -0.0001]
            dof_acc = [-5e-8]
            base_height = [-2.0, -5.0]
            feet_slip = [-0.0, -0.2]
            feet_air_time = [0.5, 0.1]
            collision = -1.0
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still2 = -0.6
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.05, -0.25]
            shank_perp2ground = -0.05
            gait_2_step = [-1.0, -0.2]
            gait_3_step = 0.0

    class commands(El4090SpiderCfg.commands):
        curriculum = True
        max_curriculum = 2.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = False

        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.2, 1.2]    # min max [rad/s]
            heading = [-3.14, 3.14]


class El4090Tripod2LowCfgPPO(El4090SpiderCfgPPO):
    class runner(El4090SpiderCfgPPO.runner):
        experiment_name = "el4090_tripod2_low"
