from legged_gym.envs.el_4090.pd_gru_lidar.el_4090_lidar_config import (
    El4090LidarCfg,
    El4090LidarCfgPPO,
)


class El4090LidarTripod2LowCfg(El4090LidarCfg):
    """EL_4090 LiDAR + tripod-2 gait, low crouching stance."""

    class init_state(El4090LidarCfg.init_state):
        pos = [0.0, 0.0, 0.4]
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,

            "RF_HFE": 0.0, "RM_HFE": 0.0, "RB_HFE": 0.0,
            "LF_HFE": 0.0, "LM_HFE": 0.0, "LB_HFE": 0.0,

            "RF_KFE": -0.0, "RM_KFE": -0.0, "RB_KFE": -0.0,
            "LF_KFE": -0.0, "LM_KFE": -0.0, "LB_KFE": -0.0,
        }

    class rewards(El4090LidarCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.25
        only_positive_rewards = False
        multi_stage_rewards = True
        reward_stage_threshold = 2.0
        reward_min_stage = 0
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

            # Obstacle-avoidance rewards disabled for now.
            sector_dist_penalty = 0.0

    class commands(El4090LidarCfg.commands):
        curriculum = True
        max_curriculum = 2.5
        num_commands = 4
        resampling_time = 4.
        heading_command = False
        small_command_radio = False

        class ranges(El4090LidarCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [-3.14, 3.14]


class El4090LidarTripod2LowCfgPPO(El4090LidarCfgPPO):
    class policy(El4090LidarCfgPPO.policy):
        gradient_checkpointing_proximal = False  
        gradient_checkpointing_distal = True     

    class algorithm(El4090LidarCfgPPO.algorithm):
        num_mini_batches = 4
        
    class runner(El4090LidarCfgPPO.runner):
        experiment_name = "el_4090_lidar_tripod2_low"
