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


class Go2BatchRolloutCfg(RobotBatchRolloutPerceptCfg):
    # Gait scheduler configurations
    class gait_scheduler:
        period = 0.6
        duty = 0.5
        foot_phases = [0.0, 0.5, 0.5, 0.0]  # For 4-legged robot (Go2 pattern)
        dt = 0.005
        swing_height = 0.15
        track_sigma = 0.25

    class async_gait_scheduler(AsyncGaitSchedulerCfg):
        # DOF configuration for Go2
        dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                     'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                     'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                     'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

        # DOF alignment sets for trot gait pattern
        dof_align_sets = [['FL_thigh_joint', 'RR_thigh_joint'],
                          ['FR_thigh_joint', 'RL_thigh_joint'],
                          ['FL_calf_joint', 'RR_calf_joint'],
                          ['FR_calf_joint', 'RL_calf_joint']]

        # Nominal joint positions for Go2
        dof_nominal_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]  # hip, thigh, calf values

        # Foot configuration for Go2
        foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

        # Foot alignment sets for trot gait pattern
        foot_z_align_sets = [['FL_foot', 'RR_foot'],
                             ['FR_foot', 'RL_foot']]

    class env(RobotBatchRolloutPerceptCfg.env):
        num_envs = 32        # Number of main environments
        rollout_envs = 1     # Number of rollout environments per main env

        # Go2 specific settings
        num_observations = 181
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class terrain(RobotBatchRolloutPerceptCfg.terrain):
        use_terrain_obj = True  # use TerrainObj class to create terrain
        # path to the terrain file
        terrain_file = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"
        measure_heights = False
        curriculum = False
        # Origin generation method
        random_origins: bool = True  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    # Ray caster configuration
    class raycaster:
        enable_raycast = True  # Set to True to enable ray casting
        ray_pattern = "spherical"    # Options: single, grid, cone, spherical
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = False  # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8

    # SDF configuration
    class sdf:
        enable_sdf = True      # Set to True to enable SDF calculations
        # Paths to mesh files for SDF calculation
        mesh_paths = ["/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"]
        max_distance = 10.0     # Maximum SDF distance to compute
        enable_caching = True   # Enable SDF query caching for performance
        update_freq = 5         # Update SDF values every N steps

        # Robot body parts to compute SDF for
        query_bodies = ["base", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]

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
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # Command ranges
        class ranges(RobotBatchRolloutPerceptCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotBatchRolloutPerceptCfg.init_state):
        pos = [0.0, 0.0, 0.43]  # x,y,z [m] - Go2 height
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            'FR_hip_joint': -0.1,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,
            'RL_hip_joint': 0.1,
            'RL_thigh_joint': 1.0,
            'RL_calf_joint': -1.5,
            'RR_hip_joint': -0.1,
            'RR_thigh_joint': 1.0,
            'RR_calf_joint': -1.5,
        }

    class control(RobotBatchRolloutPerceptCfg.control):
        # PD Drive parameters:
        stiffness = {'joint': 55.0}  # [N*m/rad]
        damping = {'joint': 0.8}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # IMPORTANT: for batch rollout, do not use the actuator network
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/go2_actuator_net.pt"

    class asset(RobotBatchRolloutPerceptCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_description.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(RobotBatchRolloutPerceptCfg.rewards):
        max_contact_force = 350.
        base_height_target = 0.43
        only_positive_rewards = True
        # Multi-stage rewards
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales(RobotBatchRolloutPerceptCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_slip = [-0.0, -0.4]  # Before feet_air_time
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.003
            stand_still = -0.
            dof_pos_limits = -1.0

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 1.0
            dof_nominal_pos = [0.05, 0.2]
            reward_foot_z_align = [0.1, 0.6]

    class domain_rand(RobotBatchRolloutPerceptCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1., 1.]  # Go2 is lighter than Anymal

    class viewer(RobotBatchRolloutPerceptCfg.viewer):
        ref_env = 0
        pos = [2.0, 0.0, 2.0]  # [m]
        lookat = [0.5, 0.0, 0.]  # [m]


class Go2BatchRolloutCfgPPO(RobotBatchRolloutPerceptCfgPPO):
    class policy(RobotBatchRolloutPerceptCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotBatchRolloutPerceptCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(RobotBatchRolloutPerceptCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_batch_rollout'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True
