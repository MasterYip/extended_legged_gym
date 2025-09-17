# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import inspect

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept_config import RobotBatchRolloutPerceptCfg, RobotBatchRolloutPerceptCfgPPO
from legged_gym.utils.gait_scheduler import AsyncGaitSchedulerCfg


class ElSpiderAirBatchRolloutCfg(RobotBatchRolloutPerceptCfg):
    # Gait scheduler configurations
    class gait_scheduler:
        period = 1.4
        duty = 0.5
        foot_phases = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # For 6-legged robot
        dt = 0.005
        swing_height = 0.07
        track_sigma = 0.25

    class async_gait_scheduler(AsyncGaitSchedulerCfg):
        # DOF configuration
        dof_names = ['LB_HAA', 'LB_HFE', 'LB_KFE',
                     'LF_HAA', 'LF_HFE', 'LF_KFE',
                     'LM_HAA', 'LM_HFE', 'LM_KFE',
                     'RB_HAA', 'RB_HFE', 'RB_KFE',
                     'RF_HAA', 'RF_HFE', 'RF_KFE',
                     'RM_HAA', 'RM_HFE', 'RM_KFE']

        # DOF alignment sets for tripod gait pattern
        dof_align_sets = [['RF_HFE', 'RB_HFE', 'LM_HFE'],
                          ['LF_HFE', 'LB_HFE', 'RM_HFE'],
                          ['RF_KFE', 'RB_KFE', 'LM_KFE'],
                          ['LF_KFE', 'LB_KFE', 'RM_KFE']]

        # Nominal joint positions
        dof_nominal_pos = [0.0, 1.0, 1.0]*6  # HAA, HFE, KFE repeated for 6 legs

        # Foot configuration
        foot_names = ['LB_FOOT', 'LF_FOOT', 'LM_FOOT', 'RB_FOOT', 'RF_FOOT', 'RM_FOOT']

        # Foot alignment sets for tripod gait pattern
        foot_z_align_sets = [['RF_FOOT', 'RB_FOOT', 'LM_FOOT'],
                             ['LF_FOOT', 'LB_FOOT', 'RM_FOOT']]

    class env(RobotBatchRolloutPerceptCfg.env):
        num_envs = 32        # Number of main environments
        rollout_envs = 0    # Number of rollout environments per main env

        # ElSpider Air specific settings
        num_observations = 66 # 66 + 128 + 6 + 187
        num_actions = 18
        episode_length_s = 20  # episode length in seconds

    class terrain(RobotBatchRolloutPerceptCfg.terrain):
        mesh_type = 'confined_trimesh'
        # path to the terrain file
        measure_heights = False

        # Curriculum Settings
        curriculum = True
        max_init_terrain_level = 2
        terrain_length = 6.
        terrain_width = 6.
        num_rows = 2  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        difficulty_scale = 0.6
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.3, 0.2, 0.1] # FIXME: when a type=0, error might show up
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 1.0, 0.0, 0.0]

        # TerrainObj Settings
        use_terrain_obj = False  # use TerrainObj class to create terrain
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        # origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        # origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        # height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height
        origins_x_range: list = [0.5, 1.5]  # min/max range for random x position
        origins_y_range: list = [-5, 1]  # min/max range for random y position
        height_clearance_factor: float = 1.5  # Required clearance as multiple of nominal_height


    # Ray caster configuration
    class raycaster:
        enable_raycast = False  # Set to True to enable ray casting
        ray_pattern = "spherical2"    # Options: single, grid, cone, spherical
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = None       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = False  # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8
        # For spherical2 pattern (uniform)
        spherical2_num_points = 128     # Number of points for uniform spherical distribution
        spherical2_polar_axis = [0.0, 0.0, 1.0]  # Direction of polar axis

    # SDF configuration
    class sdf:
        enable_sdf = False      # Set to True to enable SDF calculations
        # Paths to mesh files for SDF calculation
        mesh_paths = []
        max_distance = 10.0     # Maximum SDF distance to compute
        enable_caching = True   # Enable SDF query caching for performance
        update_freq = 5         # Update SDF values every N steps

        # Robot body parts to compute SDF for
        # BUG: trunk cannot be found by gym.find_actor_rigid_body_handle
        query_bodies = ["trunk", "RF_SHANK", "RM_SHANK", "RB_SHANK", "LF_SHANK", "LM_SHANK", "LB_SHANK"]

        # Whether to compute SDF gradients
        compute_gradients = True

        # Whether to compute nearest points on mesh
        compute_nearest_points = True

        # Whether to include SDF values in observations
        include_in_obs = True

    class commands(RobotBatchRolloutPerceptCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # FIXME: for policy playing, the init command do not cover all main envs
        class ranges(RobotBatchRolloutPerceptCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotBatchRolloutPerceptCfg.init_state):
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
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

    class control(RobotBatchRolloutPerceptCfg.control):
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2  # Enable Network-0.3 | Disable Network-0.2

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # IMPORTANT: for batch rollout, do not use the actuator network
        # because when rollout envs step, the actuator network will not be freezed
        # for main envs.
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(RobotBatchRolloutPerceptCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini_collsp.urdf"
        name = "elspider"
        foot_name = "FOOT"
        penalize_contacts_on = ["base", "HIP", "THIGH", "SHANK"]  # "SHANK" may collide with the ground through foot
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    # class rewards(RobotBatchRolloutPerceptCfg.rewards):
    #     max_contact_force = 500.
    #     base_height_target = 0.28
    #     only_positive_rewards = True
    #     # Multi-stage
    #     # Stage 0: Learn to walk with tripod gait
    #     # Stage 1: Correct DOF and FootZ positions / Prevent Slip
    #     multi_stage_rewards = True  # if true, reward scales should be list
    #     reward_stage_threshold = 6.0
    #     reward_min_stage = 0  # Start from 0
    #     reward_max_stage = 1

    #     class scales():
    #         termination = -0.0
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 0.5
    #         lin_vel_z = -1.0
    #         ang_vel_xy = -0.05
    #         dof_vel = -0.
    #         stand_still = -0.
    #         dof_pos_limits = -1.0
    #         orientation = -3.0
    #         torques = -0.00001
    #         action_rate = -0.001
    #         dof_acc = -1.5e-8
    #         feet_slip = [-0.0, -0.4]  # Before feet_air_time
    #         feet_air_time = 0.8
    #         async_gait_scheduler = [-0.2, -0.1]
    #         # feet_contact_forces = -0.01
    #         collision = -1.0
    #         base_height = -0.0 # If measure_height is False, this should be 0.0

    #     class async_gait_scheduler:
    #         # Reward for the async gait scheduler
    #         dof_align = 1.0
    #         dof_nominal_pos = [0.05, 0.0]
    #         reward_foot_z_align = [0.1, 0.0]

    class rewards(RobotBatchRolloutPerceptCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.34
        only_positive_rewards = False

        # No multi-stage rewards for flat environment
        multi_stage_rewards = True

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        class scales:
            termination = -0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -0.5e-8
            base_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            collision = -0.05
            feet_stumble = -0.4
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            # async_gait_scheduler = [-0.2, -0.4]
            gait_2_step = -1.0

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 1.0
            dof_nominal_pos = [0.05, 0.2]
            reward_foot_z_align = [0.1, 0.6]

    class domain_rand(RobotBatchRolloutPerceptCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        # NOTE: avoid sim slow down (foundLostAggregatePairsCapacity)
        rollout_envs_sync_pos_drift = 0.0

    class viewer(RobotBatchRolloutPerceptCfg.viewer):
        ref_env = 0
        pos = [2, 2, 2.0]  # [m]
        lookat = [0.0, -2.0, 0.0]  # [m]
        render_rollouts = False

class ElSpiderAirBatchRolloutCfgPPO(RobotBatchRolloutPerceptCfgPPO):
    class policy(RobotBatchRolloutPerceptCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotBatchRolloutPerceptCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (RobotBatchRolloutPerceptCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_batch_rollout'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True
