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

from legged_gym.envs.anymal_c.batch_rollout.anymal_c_batch_rollout_config import AnymalCBatchRolloutCfg, AnymalCBatchRolloutCfgPPO

class AnymalCBatchRolloutFlatCfg(AnymalCBatchRolloutCfg):
    class env(AnymalCBatchRolloutCfg.env):
        num_envs = 32        # Number of main environments
        rollout_envs = 0     # Number of rollout environments per main env

        # AnymalC specific settings
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class terrain(AnymalCBatchRolloutCfg.terrain):
        use_terrain_obj = False  # use TerrainObj class to create terrain
        mesh_type = 'plane' 
        # path to the terrain file
        terrain_file = "/home/chen/extended_legged_gym/legged_gym/resources/terrains/confined/confined_terrain.obj"
        measure_heights = False
        curriculum = False
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    # Ray caster configuration
    class raycaster(AnymalCBatchRolloutCfg.raycaster):
        enable_raycast = False  # Set to True to enable ray casting
        ray_pattern = "spherical"    # Options: single, grid, cone, spherical
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = "/home/chen/extended_legged_gym/legged_gym/resources/terrains/confined/confined_terrain.obj"       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = False  # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8

    # SDF configuration
    class sdf(AnymalCBatchRolloutCfg.sdf):
        enable_sdf = False      # Set to True to enable SDF calculations
        # Paths to mesh files for SDF calculation
        mesh_paths = ["/home/chen/extended_legged_gym/legged_gym/resources/terrains/confined/confined_terrain.obj"]
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

    class commands(AnymalCBatchRolloutCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # Command ranges
        class ranges(AnymalCBatchRolloutCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(AnymalCBatchRolloutCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'LF_HAA': 0.0,
            'LF_HFE': 0.4,
            'LF_KFE': -0.8,
            'RF_HAA': 0.0,
            'RF_HFE': 0.4,
            'RF_KFE': -0.8,
            'LH_HAA': 0.0,
            'LH_HFE': -0.4,
            'LH_KFE': 0.8,
            'RH_HAA': 0.0,
            'RH_HFE': -0.4,
            'RH_KFE': 0.8,
        }

    class control(AnymalCBatchRolloutCfg.control):
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

    class asset( AnymalCBatchRolloutCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards(AnymalCBatchRolloutCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.5
        only_positive_rewards = True
        # Multi-stage rewards
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales():
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

            # Override default values
            # orientation = -5.0
            # torques = -0.00001
            # action_rate = -0.001
            # dof_acc = -0.5e-8
            # feet_slip = [-0.0, -0.4]
            # feet_air_time = 0.8
            # collision = -1.0
            # base_height = -8.0

    class domain_rand(AnymalCBatchRolloutCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(AnymalCBatchRolloutCfg.viewer):
        ref_env = 0
        pos = [2.0, 0.0, 2.0]  # [m]
        lookat = [0.5, 0.0, 0.]  # [m]


class AnymalCBatchRolloutFlatCfgPPO(AnymalCBatchRolloutCfgPPO):
    class policy(AnymalCBatchRolloutCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(AnymalCBatchRolloutCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AnymalCBatchRolloutCfgPPO.runner):
        run_name = ''
        experiment_name = 'anymal_c_batch_rollout'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True
