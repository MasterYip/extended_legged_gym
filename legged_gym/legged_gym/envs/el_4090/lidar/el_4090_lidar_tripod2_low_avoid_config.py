from legged_gym.envs.el_4090.lidar.el_4090_lidar_config import (
    El4090LidarCfg,
    El4090LidarCfgPPO,
)


class El4090LidarTripod2LowAvoidCfg(El4090LidarCfg):
    """EL_4090 LiDAR + tripod-2 gait, low crouching stance."""
    class env(El4090LidarCfg.env):
        max_episode_length_s = 20.0

    class terrain(El4090LidarCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        terrain_length = 16
        terrain_width = 16
        border_size = 5
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 4  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        difficulty_scale = 1.0

        # 柱子参数（pillar_field_terrain 已通过 getattr 读取）
        pillar_count_min = 0
        pillar_count_max = 12
        pillar_size_x_min = 0.5
        pillar_size_x_max = 4.0
        pillar_size_y_min = 0.5
        pillar_size_y_max = 4.0
        pillar_height_min = 1.00
        pillar_height_max = 2.00
        pillar_min_separation = 2.2  
        pillar_center_clear_radius = 3.0
        pillar_spawn_radius = 10.0        #约束范围半径
        pillar_allow_height_variation = True

    class init_state(El4090LidarCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,

            "RF_HFE": 0.0, "RM_HFE": 0.0, "RB_HFE": 0.0,
            "LF_HFE": 0.0, "LM_HFE": 0.0, "LB_HFE": 0.0,

            "RF_KFE": -0.0, "RM_KFE": -0.0, "RB_KFE": -0.0,
            "LF_KFE": -0.0, "LM_KFE": -0.0, "LB_KFE": -0.0,
        }
        randomize_rot = True
        rot_randomization_range = [-3.14, 3.14]
        spawn_offset_range = 0.2

    class cmd_safe(El4090LidarCfg.cmd_safe):
        body_semi_length = 0.26   # EL_4090 half-length, ~0.5 m body
        body_semi_width = 0.15    # EL_4090 half-width,  ~0.24 m body + leg
        z_thresh_high = 0.2      # overhead filter threshold (body-frame z)
        d_safety = 0.10           # additional safety gap (m)
        d_safe_max = 1.0          # distance where safe = 1
        cmd_safe_sigma = 0.25     # gaussian kernel width
        dist_penalty_thresh = 1.0 # penalty activates below this (m)
        exp_sigma = -2.0

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

            # Obstacle-avoidance rewards
            sector_dist_penalty = 0.5

    class commands(El4090LidarCfg.commands):
        curriculum = True
        max_curriculum = 2.5
        num_commands = 4
        resampling_time = 20.
        heading_command = True
        small_command_radio = False

        class ranges(El4090LidarCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [-3.14, 3.14]

    class sim(El4090LidarCfg.sim):
        class physx(El4090LidarCfg.sim.physx):
            max_gpu_contact_pairs = 2**24


class El4090LidarTripod2LowAvoidCfgPPO(El4090LidarCfgPPO):
    class policy(El4090LidarCfgPPO.policy):
        gradient_checkpointing_proximal = False  
        gradient_checkpointing_distal = True     

    class algorithm(El4090LidarCfgPPO.algorithm):
        num_mini_batches = 4
        
    class runner(El4090LidarCfgPPO.runner):
        experiment_name = "el_4090_lidar_tripod2_low_avoid"