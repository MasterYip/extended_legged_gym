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

SAME_DIM_POLICY_HIDDEN_DIMS = [128, 64, 32]
SAME_DIM_INIT_NOISE_STD = 0.35


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
        horizontal_line_num = 48  # More horizontal rays to catch thin columns
        vertical_line_num = 12   # More vertical rays to catch low obstacles
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
        
        # Final task: only a narrow uniform corridor, with sparse small columns inside it.
        confined_terrain_proportions = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        
        # Spawn area size for robot placement
        spawn_area_size = 1.0  # Smaller central free area to reduce local hovering near center
        spawn_area_flat = True
        
        # Moderate difficulty so the robot can still complete the course.
        difficulty_scale = 0.16
        corridor_only = True
        corridor_width_override = 1.45
        corridor_uniform_width = True
        corridor_obstacle_density_override = 0.08
        corridor_obstacle_size_override = 0.12
        corridor_obstacle_height_override = 0.08
        
        slope_treshold = 0.75  # Slopes above this threshold will be corrected
        
        # Goal navigation settings
        goal_navigation = True  # Enable start→goal navigation mode
        goal_offset_y = 4.8     # Put goal near the far end without hitting the boundary

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.6,
            "RM_HFE": 0.6,
            "RB_HFE": 0.6,
            "LF_HFE": 0.6,
            "LM_HFE": 0.6,
            "LB_HFE": 0.6,

            "RF_KFE": 0.6,
            "RM_KFE": 0.6,
            "RB_KFE": 0.6,
            "LF_KFE": 0.6,
            "LM_KFE": 0.6,
            "LB_KFE": 0.6,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters
        stiffness = {'HAA': 90., 'HFE': 90., 'KFE': 90.}  # [N*m/rad]
        damping = {'HAA': 3.5, 'HFE': 3.5, 'KFE': 3.5}    # [N*m*s/rad]
        
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
        friction_range = [0.7, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
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
        base_height_target = 0.34
        max_contact_force = 500.
        only_positive_rewards = False  # Allow negative rewards for collision
        
        # Obstacle avoidance parameters
        safe_obstacle_dist = 0.65    # Slightly safer corridor tracking
        danger_obstacle_dist = 0.20  # Penalty starts a bit earlier in the final task
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
        goal_reach_threshold = 0.9    # Tighten goal reach criterion for the final task
        goal_max_distance = 6.5       # Max expected distance to goal [meters] (for normalization)

        class scales(LeggedRobotCfg.rewards.scales):
            # Standard locomotion rewards
            termination = -2.0         # Penalize episode termination
            tracking_lin_vel = 1.0     # Stronger forward gait tracking
            tracking_ang_vel = 0.6     # Slightly stronger to keep heading stable
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -0.45
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-8
            base_height = -1.2
            feet_air_time = 1.0
            collision = -1.5
            feet_stumble = -0.0
            action_rate = -0.012       # Reduce over-smoothing to avoid freezing
            stand_still = -0.30        # Stronger anti-idle pressure
            dof_pos_limits = -1.0
            feet_slip = -0.4
            
            # Confined space specific rewards
            obstacle_avoidance = 0.25   # Keep obstacle margin without dominating gait
            collision_penalty = -0.20   # Penalize danger but avoid over-conservative policy
            corridor_centering = 0.45   # Keep the robot away from walls in open corridors
            exploration = 0.0           # Disabled: goal system handles movement

            # Active obstacle negotiation rewards
            obstacle_maneuvering = 0.12
            retreat = 0.05
            
            # Goal-directed navigation rewards
            goal_reaching = 6.5         # Balance objective against gait preservation
            goal_progress = 4.5         # Increase dense forward incentive
            goal_bonus = 16.0           # Terminal objective remains meaningful
            goal_heading = 0.8          # Heading guidance, gated by movement speed
            
            # Gait rewards
            gait_2_step = 0.8

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
            lin_vel_x = [0.12, 0.95]   # Push a clearer forward command to avoid idle/backward bias
            lin_vel_y = [-0.2, 0.2]   # Moderate lateral commands for cleaner gait
            ang_vel_yaw = [-1.0, 1.0] # Allow turning to face goal
            heading = [-3.14, 3.14]   # Will be overridden by goal heading


class ElSpiderLidarConfinedCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.0006         # Slightly more exploration for robust obstacle handling
        learning_rate = 7e-4          # Smoother policy updates for gait stability
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        desired_kl = 0.008            # Tighter updates for stability
        schedule = 'adaptive'         # Use adaptive LR schedule based on KL divergence

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.25
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
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


class ElSpiderLidarFlatPretrainCfg(ElSpiderLidarConfinedCfg):
    """Flat pretraining task with exactly the same obs/action dimensions as confined task.

    This task is used for stage-1 pretraining, then weights can be resumed on
    `elspider_lidar_confined` directly without network shape mismatch.
    """

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        goal_navigation = False
        goal_offset_y = 2.0

    class commands(ElSpiderLidarConfinedCfg.commands):
        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.8, 0.8]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            orientation = -0.25
            base_height = -1.0
            action_rate = -0.01
            feet_slip = -0.15
            collision = -0.5
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 10.0
            goal_progress = 6.0
            goal_bonus = 25.0
            stand_still = -0.35


class ElSpiderLidarFlatPretrainCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for flat pretraining with same network structure."""

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.0003
        desired_kl = 0.008

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_flat_pretrain'
        max_iterations = 8000


class ElSpiderLidarPoseAdaptSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-A: pose/posture adaptation style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        goal_navigation = False

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [-0.3, 0.3]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.3, 0.3]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        goal_reach_threshold = 0.20
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 0.4
            tracking_ang_vel = 0.2
            orientation = -6.0
            base_height = -10.0
            action_rate = -0.002
            stand_still = -0.05
            collision = -0.3
            gait_2_step = -2.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarPoseAdaptSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for pose/posture adaptation style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.25

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.002
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_pose_adapt_same_dim'
        max_iterations = 1500


class ElSpiderLidarFlatSkillSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-B: flat locomotion style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        goal_navigation = False

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [-1.2, 1.2]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.6, 0.6]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        multi_stage_rewards = True
        reward_stage_threshold = 6.0
        reward_min_stage = 0
        reward_max_stage = 1

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_acc = -5e-8
            base_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            gait_2_step = -5.0
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.1
            dof_pos_limits = -1.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarFlatSkillSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for flat locomotion style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.005
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_flat_same_dim'
        max_iterations = 3000
        multi_stage_rewards = True


class ElSpiderLidarMixedTerrainSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-C: mixed terrain style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        goal_navigation = False
        max_init_terrain_level = 0
        terrain_length = 4.0
        terrain_width = 4.0
        num_rows = 4
        num_cols = 4
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.2]

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = True
        max_curriculum = 1.0
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        multi_stage_rewards = True
        reward_stage_threshold = 6.0
        reward_min_stage = 0
        reward_max_stage = 1

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            termination = -5.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_acc = -5e-8
            base_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            gait_2_step = -5.0
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.05
            dof_pos_limits = -1.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarMixedTerrainSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for mixed terrain style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.004
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_mixed_terrains_same_dim'
        max_iterations = 4000
        multi_stage_rewards = True


class ElSpiderLidarNavBarrierSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-D: barrier obstacle-avoidance style task (same obs/action dimensions)."""

    class env(ElSpiderLidarConfinedCfg.env):
        num_envs = 512

    class sim(ElSpiderLidarConfinedCfg.sim):
        class physx(ElSpiderLidarConfinedCfg.sim.physx):
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 6

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'confined_trimesh'
        curriculum = True
        max_init_terrain_level = 0
        num_rows = 1
        num_cols = 1
        difficulty_scale = 0.6
        spawn_area_size = 6.0
        goal_navigation = False
        goal_offset_y = 3.0
        confined_terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.1, 0.7]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.6, 0.6]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        safe_obstacle_dist = 0.6
        danger_obstacle_dist = 0.22
        collision_threshold = 0.05
        goal_reach_threshold = 0.25
        goal_max_distance = 10.0
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 0.8
            tracking_ang_vel = 0.4
            collision = -2.0
            obstacle_avoidance = 1.2
            collision_penalty = -2.0
            stand_still = -0.3
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarNavBarrierSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for barrier navigation style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.003
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_nav_barrier_same_dim'
        max_iterations = 5000


class ElSpiderLidarWalkFlatSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-1: pure locomotion + posture control on flat terrain (same dimensions)."""

    class init_state(ElSpiderLidarConfinedCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.6,
            "RM_HFE": 0.6,
            "RB_HFE": 0.6,
            "LF_HFE": 0.6,
            "LM_HFE": 0.6,
            "LB_HFE": 0.6,

            "RF_KFE": 0.6,
            "RM_KFE": 0.6,
            "RB_KFE": 0.6,
            "LF_KFE": 0.6,
            "LM_KFE": 0.6,
            "LB_KFE": 0.6,
        }

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        goal_navigation = False
        goal_offset_y = 2.0

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.15, 0.55]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.4, 0.4]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = False
        push_robots = False

    class control(ElSpiderLidarConfinedCfg.control):
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}
        action_scale = 0.5
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        multi_stage_rewards = True
        reward_stage_threshold = 6.0
        reward_min_stage = 0
        reward_max_stage = 1
        goal_reach_threshold = 0.15
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_acc = -5e-8
            base_height = -12.0
            feet_slip = -0.4
            feet_air_time = 0.8
            gait_2_step = -5.0
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarWalkFlatSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for stage-1 pure locomotion."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.20

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.0005
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_walk_flat_same_dim'
        max_iterations = 3000
        multi_stage_rewards = True


class ElSpiderLidarNavFlatSameDimCfg(ElSpiderLidarWalkFlatSameDimCfg):
    """Stage-2: flat navigation with goal-directed policy (same dimensions)."""

    class terrain(ElSpiderLidarWalkFlatSameDimCfg.terrain):
        goal_navigation = True
        goal_offset_y = 3.5

    class commands(ElSpiderLidarWalkFlatSameDimCfg.commands):
        goal_directed = True
        heading_command = True

        class ranges(ElSpiderLidarWalkFlatSameDimCfg.commands.ranges):
            lin_vel_x = [0.0, 0.7]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.6, 0.6]

    class rewards(ElSpiderLidarWalkFlatSameDimCfg.rewards):
        base_height_target = 0.34
        goal_reach_threshold = 0.25
        goal_max_distance = 10.0
        terminate_on_goal_reached = True

        class scales(ElSpiderLidarWalkFlatSameDimCfg.rewards.scales):
            base_height = -12.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            feet_slip = -0.4
            feet_air_time = 0.8
            gait_2_step = -5.0
            action_rate = -0.001
            stand_still = -0.0
            goal_reaching = 6.0
            goal_progress = 4.0
            goal_bonus = 12.0
            goal_heading = 1.0


class ElSpiderLidarNavFlatSameDimCfgPPO(ElSpiderLidarFlatPretrainCfgPPO):
    """PPO config for stage-2 flat navigation."""

    class policy(ElSpiderLidarFlatPretrainCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class runner(ElSpiderLidarFlatPretrainCfgPPO.runner):
        experiment_name = 'elspider_lidar_nav_flat_same_dim'
        max_iterations = 8000


class ElSpiderLidarConfinedEasySameDimCfg(ElSpiderLidarConfinedCfg):
    """Easy confined stage with same obs/action dimensions for curriculum transfer.

    Stage-2 after flat pretraining and before full confined training.
    """

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'confined_trimesh'
        use_terrain_obj = False
        curriculum = True
        num_rows = 6
        num_cols = 4
        difficulty_scale = 0.12
        goal_offset_y = 4.5
        spawn_area_size = 1.0
        corridor_only = True
        corridor_width_override = 1.30
        confined_terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        goal_max_distance = 6.0
        safe_obstacle_dist = 0.70
        danger_obstacle_dist = 0.40

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            goal_reaching = 7.0
            goal_progress = 3.5
            goal_bonus = 15.0
            goal_heading = 0.6
            corridor_centering = 0.5
            obstacle_avoidance = 0.30
            collision = -1.1
            collision_penalty = -0.30


class ElSpiderLidarConfinedEasySameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for easy confined same-dim stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_confined_easy_same_dim'
        max_iterations = 12000
