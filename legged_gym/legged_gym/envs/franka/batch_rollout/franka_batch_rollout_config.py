# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept_config import RobotBatchRolloutPerceptCfg, RobotBatchRolloutPerceptCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class FrankaBatchRolloutCfg(RobotBatchRolloutPerceptCfg):
    """Configuration for Franka Panda robot arm batch rollout with perception."""
    
    class env(RobotBatchRolloutPerceptCfg.env):
        num_envs = 40  # Main environments
        rollout_envs = 0  # Rollout environments per main env
        num_observations = 38 + 7  # Will be updated based on perception features
        num_privileged_obs = None
        num_actions = 7  # 7 DOF arm + 2 finger joints
        env_spacing = 1.0
        episode_length_s = 15  # Longer episodes for manipulation planning
        send_timeouts = True

    class terrain(RobotBatchRolloutPerceptCfg.terrain):
        mesh_type = 'confined_trimesh'  # Generate manipulation workspace with obstacles
        use_terrain_obj = False  # Use procedural generation
        measure_heights = False  # Not needed for arm manipulation
        curriculum = False
        horizontal_scale = 0.03  # Very fine resolution for manipulation
        vertical_scale = 0.005
        border_size = 2
        static_friction = 0.6
        dynamic_friction = 0.6
        restitution = 0.0
        # Confined terrain for manipulation workspace
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 2  # starting curriculum state
        terrain_length = 2.
        terrain_width = 2.
        num_rows = 3  # number of terrain rows (levels)
        num_cols = 3  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap, column_obstacles, wall_with_gap]
        confined_terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces


    # Ray caster configuration for obstacle detection
    class raycaster(RobotBatchRolloutPerceptCfg.raycaster):
        enable_raycast = False
        ray_pattern = "spherical"  # Spherical pattern for 3D awareness
        spherical_num_azimuth = 8  # 8 rays in horizontal direction
        spherical_num_elevation = 4  # 4 rays in vertical direction
        max_distance = 2.0  # 2m max range for workspace
        attach_yaw_only = False  # Full 3D orientation
        offset_pos = [0.0, 0.0, 0.1]  # Offset from robot base
        terrain_file = ""  # Will use procedurally generated terrain

    # SDF configuration for precise obstacle avoidance
    class sdf(RobotBatchRolloutPerceptCfg.sdf):
        enable_sdf = True  # Disable SDF for now since it requires mesh files
        mesh_paths = []  # Will use procedurally generated terrain when supported
        max_distance = 1.0  # 1m max SDF distance
        enable_caching = True
        update_freq = 1  # Update every 2 steps
        
        # Query SDF for key arm links
        query_bodies = [
            # "panda_link0",  # Base
            "panda_link1",  # Shoulder
            "panda_link2",  # Upper arm
            "panda_link3",  # Elbow
            "panda_link4",  # Forearm
            "panda_link5",  # Wrist
            "panda_link6",  # Wrist 2
            "panda_link7",  # Hand
            "panda_link8"   # End-effector
        ]
        collision_sphere_radius = [0.2]*4 + [0.2]*4  # List of radii, one per query body
        collision_sphere_pos = []     # List of [x, y, z] offsets, one per query body

        
        compute_gradients = True  # For gradient-based avoidance
        compute_nearest_points = True  # For visualization
        include_in_obs = True  # Include SDF in observations

    class commands(RobotBatchRolloutPerceptCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 7  # End-effector pose commands
        resampling_time = 5.0  # Resample targets every 8 seconds
        randomize_ee_orientation = True  # Enable orientation targets
        heading_command = False  # No heading command for arm
        class ranges(RobotBatchRolloutPerceptCfg.commands.ranges):
            # End-effector workspace bounds (relative to robot base)
            ee_pos_min = [0.2, -0.5, 0.1]   # Conservative workspace
            ee_pos_max = [0.9, 0.5, 0.9]    # Conservative workspace
            ee_rot_range = 0.5  # Allow more rotation variation

    class init_state:
        pos = [-0.03, -0.06, 0.0]  # Fixed base at origin
        rot = [0.0, 0.0, 0.0, 1.0]  # No rotation
        lin_vel = [0.0, 0.0, 0.0]  # No velocity (fixed base)
        ang_vel = [0.0, 0.0, 0.0]
        
        # Default joint angles for Franka Panda (ready position)
        default_joint_angles = {
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,  # -45 degrees
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,  # -135 degrees
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,   # 90 degrees
            "panda_joint7": 0.785,   # 45 degrees
            "panda_finger_joint1": 0.04,  # Open gripper (4cm)
            "panda_finger_joint2": 0.04,  # Open gripper (4cm)
        }

    class control:
        control_type = 'P'  # Position control
        # PD gains tuned for Franka manipulation
        stiffness = {
            'panda_joint1': 100.0,
            'panda_joint2': 100.0, 
            'panda_joint3': 100.0,
            'panda_joint4': 100.0,
            'panda_joint5': 40.0,
            'panda_joint6': 40.0,
            'panda_joint7': 40.0,
            # 'panda_finger_joint1': 0.0,  # Finger stiffness
            # 'panda_finger_joint2': 0.0,  # Finger stiffness
        }
        damping = {
            'panda_joint1': 15.0,
            'panda_joint2': 15.0,
            'panda_joint3': 15.0, 
            'panda_joint4': 15.0,
            'panda_joint5': 4.0,
            'panda_joint6': 4.0,
            'panda_joint7': 4.0,
            # 'panda_finger_joint1': 0.0,  # Finger damping
            # 'panda_finger_joint2': 0.0,  # Finger damping
        }
        action_scale = 1.0  # Moderate action scaling
        decimation = 4  # Higher control frequency for manipulation

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/franka/urdf/franka_panda.urdf"
        name = "franka"
        foot_name = "None"  # No feet for arm robot
        ee_name = "panda_link8"  # End-effector link name
        
        # Penalize contact on arm links (except end-effector for manipulation)
        penalize_contacts_on = [
            "panda_link0", "panda_link1", "panda_link2", 
            "panda_link3", "panda_link4", "panda_link5",
            "panda_link6", "panda_link7", "panda_link8"
        ]
        
        # Terminate if base or major arm links hit something hard
        terminate_after_contacts_on = ["panda_link0", "panda_link1"]
        
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = True  # Fixed base robot
        default_dof_drive_mode = 3  # Effort control
        self_collisions = 0  # Enable self-collision checking
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand(RobotBatchRolloutPerceptCfg.domain_rand):
        randomize_friction = False  # Keep deterministic for now
        randomize_base_mass = False  # Fixed base
        push_robots = False  # No external disturbances
        rollout_envs_sync_pos_drift = 0.0  # No position drift for manipulation

    class rewards(RobotBatchRolloutPerceptCfg.rewards):
        class scales:
            # Primary objectives - end-effector tracking
            ee_position_tracking = 20.0  # High reward for position accuracy
            # ee_orientation_tracking = 10.0  # Moderate reward for orientation
            target_reached = 50.0  # High bonus for reaching target
            # target_reached_dofvel = -0.01  # Penalize if DOF/EE velocity is high at target

            # Motion quality and efficiency
            # dof_vel = -0.01  # Penalize excessive joint velocities
            # dof_acc = -1e-6  # Penalize large accelerations
            # action_rate = -0.3  # Encourage smooth actions
            # torques = -1e-5  # Penalize high torques
            # ee_velocity = -0.05  # Penalize fast end-effector motion
            
            # Safety and obstacle avoidance
            collision_avoidance = 0.01  # Strong penalty for collisions
            # workspace_bounds = -50.0  # Penalty for leaving workspace
            obstacle_avoidance_sdf = -10.0  # Reward for SDF-based avoidance
            # raycast_obstacle_avoidance = 5.0  # Reward for maintaining clearance
            
            # Termination penalty
            # termination = -50.0

        only_positive_rewards = False  # Allow negative rewards for safety
        tracking_sigma = 1.0  # Very tight tracking for manipulation
        soft_dof_pos_limit = 0.95  # Stay within 95% of joint limits
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        
        # Manipulation-specific tolerances
        ee_pos_tolerance = 0.05  # 2cm position tolerance
        ee_rot_tolerance = 0.05  # Tight rotation tolerance
        safe_distance = 0.2     # 3cm minimum safe distance
        clearance_threshold = 0.15  # 15cm desired clearance
        
        # Multi-stage rewards for curriculum
        multi_stage_rewards = True
        reward_stage_threshold = 15.0
        reward_min_stage = 0
        reward_max_stage = 2

    class normalization(RobotBatchRolloutPerceptCfg.normalization):
        class obs_scales(RobotBatchRolloutPerceptCfg.normalization.obs_scales):
            dof_pos = 1.0
            dof_vel = 0.1
            ee_pos = 1.0
            ee_quat = 1.0
            ee_target = 1.0
            ee_error = 5.0  # Amplify error signals
            raycast = 1.0
            sdf = 10.0  # Amplify SDF signals

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(RobotBatchRolloutPerceptCfg.noise):
        add_noise = True
        noise_level = 0.3  # Moderate noise for robustness
        
        class noise_scales(RobotBatchRolloutPerceptCfg.noise.noise_scales):
            dof_pos = 0.01  # Small joint position noise
            dof_vel = 0.1
            ee_pos = 0.002  # Very small end-effector noise
            ee_quat = 0.002
            raycast = 0.01  # Small raycast noise
            sdf = 0.005  # Small SDF noise

    class viewer(RobotBatchRolloutPerceptCfg.viewer):
        ref_env = 0
        pos = [1.3, 0.0, 0.5]  # Camera position for manipulation workspace
        lookat = [0.0, 0.0, 0.5]  # Look at center of workspace
        render_rollouts = False  # Don't render rollouts for performance


class FrankaBatchRolloutCfgPPO(RobotBatchRolloutPerceptCfgPPO):
    """PPO configuration for Franka manipulation with batch rollout."""
    
    class policy:
        init_noise_std = 0.3  # Lower noise for precise manipulation
        actor_hidden_dims = [512, 256, 128]  # Larger network for complex observations
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 2.0  # Higher value loss weight
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001  # Very low entropy for deterministic manipulation
        num_learning_epochs = 8  # More learning epochs
        num_mini_batches = 8  # More mini-batches
        learning_rate = 1e-4  # Lower learning rate for stability
        schedule = 'adaptive'
        gamma = 0.999  # Very high discount for long-horizon planning
        lam = 0.98  # High lambda for TD advantage
        desired_kl = 0.008  # Lower KL divergence limit
        max_grad_norm = 0.5  # Lower gradient norm for stability

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 64  # Longer rollouts for manipulation
        max_iterations = 5000  # More iterations for complex manipulation
        save_interval = 100
        experiment_name = 'franka_batch_manipulation'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
