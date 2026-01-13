# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Extended for LiDAR-based confined space navigation

"""
Configuration for ElSpider LiDAR Confined Space Navigation Task
基于激光雷达的六足机器人受限空间避障运动控制配置
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ElSpiderLidarConfinedCfg(LeggedRobotCfg):
    """Configuration for ElSpider with LiDAR in confined spaces."""
    
    class env(LeggedRobotCfg.env):
        num_envs = 1024  # Reduced from 4096 to avoid PhysX memory issues
        num_actions = 18  # 6 legs × 3 joints
        
        # Base observations: 3+3+3+3+18+18+18 = 66
        # Height measurements: 17×11 = 187
        # LiDAR observations: 12×8 = 96
        num_lidar_obs = 96  # num_theta_bins × num_phi_bins
        num_observations = 66 + 187 + 96  # 349 total
        
        episode_length_s = 20  # Episode length in seconds

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            max_gpu_contact_pairs = 2**24  # Increase contact pairs capacity
            
    class lidar:
        """LiDAR sensor configuration."""
        sensor_type = "simple_grid"  # Options: simple_grid, avia, mid360, etc.
        
        # Sensor update frequency
        update_frequency = 20.0  # Hz
        
        # Range settings
        max_range = 5.0  # meters
        min_range = 0.1  # meters
        
        # Grid LiDAR settings
        horizontal_line_num = 36  # Number of horizontal rays
        vertical_line_num = 10   # Number of vertical rays
        horizontal_fov_deg_min = -180  # Horizontal FOV min (degrees)
        horizontal_fov_deg_max = 180   # Horizontal FOV max (degrees)
        vertical_fov_deg_min = -30     # Vertical FOV min (degrees)
        vertical_fov_deg_max = 10      # Vertical FOV max (degrees)
        
        # Observation downsampling
        num_theta_bins = 12  # Azimuth bins for observation
        num_phi_bins = 8     # Elevation bins for observation
        
        # Sensor mounting position (relative to robot base frame)
        sensor_offset = [0.0, 0.0, 0.15]  # [x, y, z] in meters
        sensor_rotation_deg = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees

    class terrain(LeggedRobotCfg.terrain):
        """Terrain configuration for confined spaces."""
        mesh_type = 'confined_trimesh'  # Use confined terrain with ceiling
        
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        # Height measurement settings
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                           0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Confined terrain settings
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # Number of terrain difficulty levels
        num_cols = 6   # Number of terrain types
        
        # Confined terrain proportions: [maze, tunnel, barrier, timber_piles, confined_gap, column_obstacles, wall_with_gap]
        # Adjust proportions to feature the new maze terrain significantly
        confined_terrain_proportions = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        # Spawn area size for robot placement
        spawn_area_size = 2.0  # meters
        
        # Difficulty scaling
        difficulty_scale = 0.5  # Start with medium difficulty for maze
        
        slope_treshold = 0.75  # Slopes above this threshold will be corrected

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m] - Increased from 0.4 to prevent spawning into ground
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.6, "RM_HFE": 0.6, "RB_HFE": 0.6,
            "LF_HFE": 0.6, "LM_HFE": 0.6, "LB_HFE": 0.6,
            "RF_KFE": 0.6, "RM_KFE": 0.6, "RB_KFE": 0.6,
            "LF_KFE": 0.6, "LM_KFE": 0.6, "LB_KFE": 0.6,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}       # [N*m*s/rad]
        
        action_scale = 0.5
        decimation = 4
        
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini.urdf"
        name = "elspider_air"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "HIP"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 0  # 1 to disable, 0 to enable
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 1.5
            height_measurements = 0.1
            lidar = 0.05  # LiDAR observation noise

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.25
        max_contact_force = 500.
        only_positive_rewards = False  # Allow negative rewards for collision
        
        # Obstacle avoidance parameters
        safe_obstacle_dist = 0.5    # Distance considered safe (meters)
        danger_obstacle_dist = 0.2  # Distance considered dangerous (meters)
        collision_threshold = 0.08  # Distance for collision termination (meters) - reduced from 0.15
        
        # Termination protection - disable collision termination during early training steps
        collision_termination_after_steps = 10  # Only terminate after this many steps
        allow_initial_contact_steps = 5  # Allow contact termination grace period
        
        # Multi-stage rewards
        multi_stage_rewards = True
        reward_stage_threshold = 5.0
        reward_min_stage = 0
        reward_max_stage = 2

        class scales(LeggedRobotCfg.rewards.scales):
            # Standard locomotion rewards
            termination = -0.0
            tracking_lin_vel = 4.0   # 1.5 -> 4.0: Encorage moving forward
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-8
            base_height = -1.0
            feet_air_time = 0.8
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.002
            stand_still = -0.1
            dof_pos_limits = -1.0
            feet_slip = -0.2
            
            # Confined space specific rewards
            obstacle_avoidance = 5.0     # 2.0 -> 5.0
            collision_penalty = -20.0    # -5.0 -> -20.0
            exploration = 2.0            # 0.5 -> 2.0
            
            # Gait rewards
            gait_2_step = -3.0

        class async_gait_scheduler:
            dof_align = 0.5
            dof_nominal_pos = [0.1, 0.2]
            reward_foot_z_align = [0.2, 0.05]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        
        class ranges:
            lin_vel_x = [-0.5, 0.8]   # Slower speeds for confined space
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]


class ElSpiderLidarConfinedCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_lidar_confined'
        load_run = -1
        max_iterations = 5000
        
        # Enable multi-stage rewards
        multi_stage_rewards = True
        
        # Checkpointing
        save_interval = 100
        
        # Logging
        log_interval = 10


# Alternative configuration with simpler LiDAR for faster training
class ElSpiderLidarConfinedSimpleCfg(ElSpiderLidarConfinedCfg):
    """Simplified configuration with reduced LiDAR resolution."""
    
    class env(ElSpiderLidarConfinedCfg.env):
        # Reduced LiDAR observations: 8×6 = 48
        num_lidar_obs = 48
        num_observations = 66 + 187 + 48  # 301 total

    class lidar(ElSpiderLidarConfinedCfg.lidar):
        # Fewer rays for faster computation
        horizontal_line_num = 24
        vertical_line_num = 6
        
        # Smaller observation bins
        num_theta_bins = 8
        num_phi_bins = 6


class ElSpiderLidarConfinedSimpleCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for simplified LiDAR task."""
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_confined_simple'


# Configuration for timber pile terrain only
class ElSpiderLidarTimberPileCfg(ElSpiderLidarConfinedCfg):
    """Configuration specifically for timber pile terrain."""
    
    class terrain(ElSpiderLidarConfinedCfg.terrain):
        # Only timber pile terrain
        confined_terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        difficulty_scale = 0.8  # Slightly easier


class ElSpiderLidarTimberPileCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for timber pile task."""
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_timber_pile'


# Configuration for tunnel terrain only
class ElSpiderLidarTunnelCfg(ElSpiderLidarConfinedCfg):
    """Configuration specifically for tunnel terrain."""
    
    class terrain(ElSpiderLidarConfinedCfg.terrain):
        # Only tunnel terrain
        confined_terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class ElSpiderLidarTunnelCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for tunnel task."""
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_tunnel'
