# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import inspect

from legged_gym.envs.batch_rollout.robot_batch_rollout_nav_config import RobotBatchRolloutNavCfg, RobotBatchRolloutNavCfgPPO
from legged_gym.utils.gait_scheduler import AsyncGaitSchedulerCfg


class AnymalCNavCfg(RobotBatchRolloutNavCfg):
    # Gait scheduler configurations
    class gait_scheduler:
        period = 1.0
        duty = 0.5
        foot_phases = [0.0, 0.5, 0.0, 0.5]  # For 4-legged robot
        dt = 0.02
        swing_height = 0.04
        track_sigma = 0.25

    class async_gait_scheduler(AsyncGaitSchedulerCfg):
        # DOF configuration
        dof_names = ['LF_HAA', 'LF_HFE', 'LF_KFE',
                     'RF_HAA', 'RF_HFE', 'RF_KFE',
                     'LH_HAA', 'LH_HFE', 'LH_KFE',
                     'RH_HAA', 'RH_HFE', 'RH_KFE']

        # DOF alignment sets for trot gait pattern
        dof_align_sets = [['LF_HFE', 'RH_HFE'],
                          ['RF_HFE', 'LH_HFE'],
                          ['LF_KFE', 'RH_KFE'],
                          ['RF_KFE', 'LH_KFE']]

        # Nominal joint positions
        dof_nominal_pos = [0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8]  # HAA, HFE, KFE values

        # Foot configuration
        foot_names = ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT']

        # Foot alignment sets for trot gait pattern
        foot_z_align_sets = [['LF_FOOT', 'RH_FOOT'],
                             ['RF_FOOT', 'LH_FOOT']]

    class env(RobotBatchRolloutNavCfg.env):
        num_envs = 32        # Number of main environments
        rollout_envs = 1     # Number of rollout environments per main env

        # AnymalC specific settings
        num_observations = 48 # 48 + 5 + 128
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class navi_opt(RobotBatchRolloutNavCfg.navi_opt):
        # ElSpider Air specific navigation settings with multiple start/goal positions
        # Example: Multiple start positions for different environments
        start_pos = [
            [2.5, 2.5, 0.32],  # Environment 0
            [3.5, 2.5, 0.32],  # Environment 1
            [2.5, 3.5, 0.32],  # Environment 2
            [3.5, 3.5, 0.32],  # Environment 3
        ]
        
        # Corresponding goal positions
        goal_pos = [
            [3.0, 0.0, 0.32],  # Goal for Environment 0
            [6.0, 3.0, 0.32],  # Goal for Environment 1  
            [0.0, 3.0, 0.32],  # Goal for Environment 2
            [3.0, 6.0, 0.32],  # Goal for Environment 3
        ]
        
        # Single orientation for all (can also be a list of orientations)
        start_quat = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.707, 0.707],
            [0.0, 0.0, -0.707, 0.707],
            [0.0, 0.0, -1.0, 0.0],
        ]
        
        tolerance_rad = 0.3           # Smaller tolerance for precision
        max_linear_vel = 1.0          # m/s
        max_angular_vel = 0.6         # rad/s (matching command ranges)
        kp_linear = 1.2               # Slightly higher gain
        kp_angular = 2.5              # Higher angular gain for better turning

    class terrain(RobotBatchRolloutNavCfg.terrain):
        use_terrain_obj = False  # use TerrainObj class to create terrain
        mesh_type = 'confined_trimesh'  # Options: plane, heightfield, trimesh
        # path to the terrain file
        terrain_file = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"
        measure_heights = False
        curriculum = False
        difficulty_scale = 0.0
        max_init_terrain_level = 2  # starting curriculum state
        terrain_length = 6.
        terrain_width = 6.
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.3, 0.2]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 0.0, 1.0, 0.0]
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    # Ray caster configuration
    class raycaster:
        enable_raycast = False  # Set to True to enable ray casting
        ray_pattern = "spherical"    # Options: single, grid, cone, spherical
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = None       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = False  # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8

    # SDF configuration
    class sdf:
        enable_sdf = False      # Set to True to enable SDF calculations
        # Paths to mesh files for SDF calculation
        mesh_paths = []
        max_distance = 10.0     # Maximum SDF distance to compute
        enable_caching = True   # Enable SDF query caching for performance
        update_freq = 5         # Update SDF values every N steps

        # Robot body parts to compute SDF for
        query_bodies = ["base", "LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]

        # Whether to compute SDF gradients
        compute_gradients = True

        # Whether to compute nearest points on mesh
        compute_nearest_points = True

        # Whether to include SDF values in observations
        include_in_obs = True

    class commands(RobotBatchRolloutNavCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # Command ranges
        class ranges(RobotBatchRolloutNavCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotBatchRolloutNavCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'LF_HAA': 0.0,
            'LF_HFE': 0.4,
            'LF_KFE': -1.1,
            'RF_HAA': 0.0,
            'RF_HFE': 0.4,
            'RF_KFE': -1.1,
            'LH_HAA': 0.0,
            'LH_HFE': -0.4,
            'LH_KFE': 1.1,
            'RH_HAA': 0.0,
            'RH_HFE': -0.4,
            'RH_KFE': 1.1,
        }

    class control(RobotBatchRolloutNavCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        jointpos_action_normalization = False
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # IMPORTANT: for batch rollout, do not use the actuator network
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(RobotBatchRolloutNavCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH", "base"]
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(RobotBatchRolloutNavCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.5
        only_positive_rewards = False
        # Multi-stage rewards
        multi_stage_rewards = False  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales():
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.5
            orientation = -4.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            # base_height = -5
            base_foot_height = -10
            feet_air_time = 0.4
            collision = -0.6
            feet_stumble = -0.8
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 1.0
            dof_nominal_pos = [0.05, 0.2]
            reward_foot_z_align = [0.1, 0.6]

    class domain_rand(RobotBatchRolloutNavCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(RobotBatchRolloutNavCfg.viewer):
        ref_env = 0
        pos = [1.0, -2.0, 2.0]  # [m]
        lookat = [1.0, 0.0, 0.]  # [m]
        render_rollouts=False


class AnymalCNavCfgPPO(RobotBatchRolloutNavCfgPPO):
    class policy(RobotBatchRolloutNavCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotBatchRolloutNavCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(RobotBatchRolloutNavCfgPPO.runner):
        run_name = ''
        experiment_name = 'anymal_c_nav'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = False
