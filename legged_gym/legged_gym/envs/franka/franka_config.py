# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class FrankaCfg(LeggedRobotCfg):
    """Configuration for Franka Panda robot arm manipulation tasks."""
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 38  # Will be updated based on actual observation size
        num_privileged_obs = None
        num_actions = 7  # 7 DOF arm + 2 finger joints
        env_spacing = 1.0
        episode_length_s = 10  # Shorter episodes for manipulation tasks
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # Generate manipulation workspace with obstacles
        use_terrain_obj = False  # Use procedural generation
        measure_heights = False  # Not needed for arm manipulation
        curriculum = False
        horizontal_scale = 0.05  # Fine resolution for manipulation
        vertical_scale = 0.005
        border_size = 5
        static_friction = 0.6
        dynamic_friction = 0.6
        restitution = 0.0
        # Confined terrain for manipulation workspace
        terrain_length = 2.0  # 2m x 2m workspace
        terrain_width = 2.0
        num_rows = 1  # Single terrain type
        num_cols = 1
        # Use confined terrain types for manipulation obstacles
        confined_terrain_proportions = [0.25, 0.5, 0.75, 1.0]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 7  # End-effector pose commands (3 pos + 4 quat)
        resampling_time = 5.0  # Resample targets every 5 seconds
        randomize_ee_orientation = False  # Keep simple position control for now
        
        class ranges(LeggedRobotCfg.commands.ranges):
            # End-effector workspace bounds (relative to robot base)
            ee_pos_min = [0.3, -0.4, 0.2]   # [x, y, z] minimum position
            ee_pos_max = [0.8, 0.4, 0.8]    # [x, y, z] maximum position  
            ee_rot_range = 0.2  # Maximum rotation angle for random orientations

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.0]  # Fixed base at origin
        rot = [0.0, 0.0, 0.0, 1.0]  # No rotation
        lin_vel = [0.0, 0.0, 0.0]  # No linear velocity (fixed base)
        ang_vel = [0.0, 0.0, 0.0]  # No angular velocity (fixed base)
        
        # Default joint angles for Franka Panda (home position)
        default_joint_angles = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,  # -45 degrees
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,  # -135 degrees  
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,   # 90 degrees
            "panda_joint7": 0.785,   # 45 degrees
            "panda_finger_joint1": 0.04,  # Open gripper (4cm)
            "panda_finger_joint2": 0.04,  # Open gripper (4cm)
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'  # Position control
        # PD gains for Franka Panda joints
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
        action_scale = 1.0  # Conservative action scaling for manipulation
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/franka/urdf/franka_panda.urdf"
        name = "franka"
        foot_name = "None"  # No feet for arm robot
        ee_name = "panda_hand"  # End-effector link name
        penalize_contacts_on = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6"]
        terminate_after_contacts_on = []  # Terminate if base link hits something
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

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False  # Not important for manipulation
        randomize_base_mass = False  # Fixed base
        push_robots = False  # No pushing for arm robot

    class rewards:
        class scales:
            # End-effector tracking rewards
            ee_position_tracking = 10.0
            ee_orientation_tracking = 2.0
            # target_reached = 50.0
            
            # Motion smoothness
            dof_vel = -0.01
            dof_acc = -2.5e-7
            action_rate = -0.01
            torques = -1e-5
            ee_velocity = -0.1
            
            # Safety and constraints
            # collision_avoidance = -10.0
            # workspace_bounds = -10.0
            # obstacle_avoidance_sdf = 5.0
            # raycast_obstacle_avoidance = 2.0
            
            # Termination
            termination = -0.0

        only_positive_rewards = False  # Allow negative rewards for penalties
        tracking_sigma = 0.1  # Tighter tracking for manipulation
        soft_dof_pos_limit = 0.9  # Keep within 90% of joint limits
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        
        # Manipulation-specific parameters
        ee_pos_tolerance = 0.02  # 2cm position tolerance
        ee_rot_tolerance = 0.1   # Rotation tolerance
        safe_distance = 0.05     # 5cm safe distance from obstacles
        clearance_threshold = 0.2  # 20cm desired clearance
        
        # Multi-stage rewards
        multi_stage_rewards = False
        reward_stage_threshold = 10.0
        reward_min_stage = 0
        reward_max_stage = 0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            dof_pos = 1.0
            dof_vel = 0.1
            ee_pos = 1.0
            ee_quat = 1.0
            ee_target = 1.0
            ee_error = 2.0
            # FIXME: Legacy need
            lin_vel = 2.0
            ang_vel = 0.25
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False  # FIXME: No noise for manipulation tasks
        noise_level = 0.5  # Lower noise for manipulation tasks
        
        class noise_scales:
            dof_pos = 0.005  # Very small joint position noise
            dof_vel = 0.1
            ee_pos = 0.001   # Very small end-effector position noise
            ee_quat = 0.001  # Very small orientation noise
            # FIXME: Legacy need
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [2, 0, 1.5]  # Camera position for arm manipulation
        lookat = [0, 0, 0.5]  # Look at workspace


class FrankaCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for Franka manipulation tasks."""
    
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5  # Lower noise for manipulation
        actor_hidden_dims = [256, 128, 64]  # Smaller network for arm control
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005  # Lower entropy for more deterministic control
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3e-4  # Slightly lower learning rate
        schedule = 'adaptive'
        gamma = 0.995  # Higher discount factor for manipulation
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32  # More steps for manipulation episodes
        max_iterations = 2000
        save_interval = 100
        experiment_name = 'franka_manipulation'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None