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
        num_envs = 2048  # Increased: more envs = more diverse experience per update
        num_actions = 18  # 6 legs × 3 joints
        
        # Base observations: 3+3+3+3+18+18+18 = 66
        # Height measurements: 17×11 = 187
        # LiDAR observations: 12×8 = 96
        # Goal observations: 2 (direction_angle, normalized_distance)
        num_lidar_obs = 96  # num_theta_bins × num_phi_bins
        num_goal_obs = 2    # goal direction angle + normalized distance
        num_observations = 66 + 187 + 96 + 2  # 351 total
        
        episode_length_s = 24  # Longer episode to reach goal

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
        terrain_length = 10.0  # Corridor length
        terrain_width = 10.0
        num_rows = 8   # More difficulty levels for corridor progression
        num_cols = 4   # Terrain type columns
        
        # Confined terrain proportions: [corridor, timber, column, maze, tunnel/barrier, gap, corridor_easy]
        # IMPORTANT: keep tunnel/barrier at 0.0 for goal-reaching training to avoid closed, unreachable maps
        confined_terrain_proportions = [0.45, 0.20, 0.15, 0.10, 0.00, 0.05, 0.05]
        
        # Spawn area size for robot placement
        spawn_area_size = 1.2  # Smaller central free area to reduce local hovering near center
        
        # Difficulty scaling - higher for narrower corridors
        difficulty_scale = 0.8  # Real difficulty: corridors get narrow
        
        slope_treshold = 0.75  # Slopes above this threshold will be corrected
        
        # Goal navigation settings
        goal_navigation = True  # Enable start→goal navigation mode
        goal_offset_y = 3.0     # Goal position Y offset from env_origin [meters]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m] - Closer to ground: less falling damage on spawn
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
        push_robots = False
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
        danger_obstacle_dist = 0.15 # REDUCED from 0.2: penalty only fires when very close
        collision_threshold = 0.03  # REDUCED from 0.05: only terminate on actual collision (3cm)
        
        # Termination protection - generous grace period
        collision_termination_after_steps = 200  # INCREASED from 50: let robot survive much longer
        allow_initial_contact_steps = 30  # Grace period at episode start
        
        # Multi-stage rewards disabled
        multi_stage_rewards = False
        reward_stage_threshold = 80.0
        reward_min_stage = 0
        reward_max_stage = 2
        
        # Goal navigation reward parameters
        goal_reach_threshold = 1.0    # Distance to consider goal reached [meters]
        goal_max_distance = 8.0       # Max expected distance to goal [meters] (for normalization)

        class scales(LeggedRobotCfg.rewards.scales):
            # Standard locomotion rewards
            termination = -2.0         # Penalize episode termination
            tracking_lin_vel = 0.5     # Low: ElSpider convention mismatch, goal_reaching handles movement
            tracking_ang_vel = 0.5     # Low: goal heading system handles turning
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
            action_rate = -0.01        # Moderate smoothness penalty
            stand_still = -0.4         # Stronger penalty for not moving
            dof_pos_limits = -1.0
            feet_slip = -0.2
            
            # Confined space specific rewards
            obstacle_avoidance = 0.3    # Reward keeping safe distance
            collision_penalty = -0.2    # Light penalty, don't dominate
            exploration = 0.0           # Disabled: goal system handles movement
            
            # Goal-directed navigation rewards
            goal_reaching = 15.0        # DOMINANT reward: velocity toward goal
            goal_progress = 6.0         # Dense reward on distance reduction per step
            goal_bonus = 30.0           # Large bonus for reaching goal
            goal_heading = 1.5          # Heading guidance, gated by movement speed
            
            # Gait rewards
            gait_2_step = -0.8

        class async_gait_scheduler:
            dof_align = 0.5
            dof_nominal_pos = [0.1, 0.2]
            reward_foot_z_align = [0.2, 0.05]

    class commands(LeggedRobotCfg.commands):
        curriculum = True  # Enable: start with slow commands, increase over time
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0  # Longer resampling: goal provides consistent direction
        heading_command = True  # Will be overridden to use goal heading
        goal_directed = True    # Use goal position to generate heading commands
        
        class ranges:
            lin_vel_x = [0.0, 1.2]    # Forward only toward goal (no backward)
            lin_vel_y = [-0.3, 0.3]   # Small lateral for obstacle avoidance
            ang_vel_yaw = [-1.0, 1.0] # Allow turning to face goal
            heading = [-3.14, 3.14]   # Will be overridden by goal heading


class ElSpiderLidarConfinedCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001          # Low: prevent entropy bonus from pushing noise_std up
        learning_rate = 1e-3           # Faster learning in early stages
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        desired_kl = 0.012            # RELAXED from 0.008: allow larger updates to escape plateau
        schedule = 'adaptive'         # Use adaptive LR schedule based on KL divergence

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5           # Standard: entropy_coef handles exploration
        actor_hidden_dims = [256, 128, 64]   # Smaller network: easier to train, less overfitting
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_lidar_confined'
        load_run = -1
        max_iterations = 22600  # Extended: resuming from checkpoint with boosted rewards
        
        # Multi-stage rewards disabled (env config controls this)
        multi_stage_rewards = False
        
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
