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

from legged_gym.envs.elspider_air.batch_rollout.elspider_air_batch_rollout_config import ElSpiderAirBatchRolloutCfg, ElSpiderAirBatchRolloutCfgPPO


class ElSpiderAirDialMPCCfg(ElSpiderAirBatchRolloutCfg):
    class env(ElSpiderAirBatchRolloutCfg.env):
        num_envs = 32        # Number of main environments
        rollout_envs = 0     # Number of rollout environments per main env
        env_spacing = 2.0
        # ElSpider Air specific settings
        num_observations = 72
        num_actions = 18
        episode_length_s = 20  # episode length in seconds

    class terrain(ElSpiderAirBatchRolloutCfg.terrain):
        use_terrain_obj = True  # use TerrainObj class to create terrain
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        # No height measurements for flat terrain
        measure_heights = False
        curriculum = False
        mesh_type = 'trimesh'  # Use a simple plane
        terrain_length = 5.
        terrain_width = 5.
        # Origin generation method
        random_origins: bool = True  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [0.5, 0.5]  # min/max range for random x position
        origins_y_range: list = [-2.0, -2.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    class raycaster:
        enable_raycast = False  # Set to True to enable ray casting
        ray_pattern = "spherical"    # Options: single, grid, cone, spherical
        num_rays = 10           # Number of rays for cone pattern
        ray_angle = 30.0        # Cone angle in degrees
        terrain_file = "resources/terrains/confined/confined_terrain.obj"       # Path to terrain mesh file
        max_distance = 10.0     # Maximum ray cast distance
        attach_yaw_only = False  # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8

    class sdf:
        enable_sdf = True      # Set to True to enable SDF calculations
        mesh_paths = ["resources/terrains/confined/confined_terrain.obj"]         # Paths to mesh files for SDF calculation
        max_distance = 10.0     # Maximum SDF distance to compute
        enable_caching = True   # Enable SDF query caching for performance
        update_freq = 1         # Update SDF values every N steps

        # Robot body parts to compute SDF for
        # BUG: trunk cannot be found by gym.find_actor_rigid_body_handle
        query_bodies = ["trunk", "RF_SHANK", "RM_SHANK", "RB_SHANK", "LF_SHANK", "LM_SHANK", "LB_SHANK"]

        # Whether to compute SDF gradients
        compute_gradients = True
        # Whether to compute nearest points on mesh
        compute_nearest_points = True
        # Whether to include SDF values in observations
        include_in_obs = True

    class commands(ElSpiderAirBatchRolloutCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(ElSpiderAirBatchRolloutCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(ElSpiderAirBatchRolloutCfg.init_state):
        # FIXME: this 2 do not take effect in dialmpc
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 1.0,
            "RM_HFE": 1.0,
            "RB_HFE": 1.0,
            "LF_HFE": 1.0,
            "LM_HFE": 1.0,
            "LB_HFE": 1.0,

            "RF_KFE": 1.0,
            "RM_KFE": 1.0,
            "RB_KFE": 1.0,
            "LF_KFE": 1.0,
            "LM_KFE": 1.0,
            "LB_KFE": 1.0,
        }

    class control(ElSpiderAirBatchRolloutCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0  # Enable Network-0.3 | Disable Network-0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # No actuator network for flat environment
        use_actuator_network = False

    class asset(ElSpiderAirBatchRolloutCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini.urdf"
        name = "elspider"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "HIP"]  # "SHANK" may collide with the ground through foot
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class rewards(ElSpiderAirBatchRolloutCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.24
        only_positive_rewards = False

        # No multi-stage rewards for flat environment
        multi_stage_rewards = False

        tracking_sigma = 2.0  # tracking reward = exp(-error^2/sigma)

        class scales(ElSpiderAirBatchRolloutCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -0.5e-8
            base_height = -8.0
            feet_slip = -0.0
            feet_air_time = 0.8
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.
            dof_pos_limits = -1.0
            async_gait_scheduler = -0.4

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 0.5
            dof_nominal_pos = [0.2, 0.2]
            reward_foot_z_align = [0.1, 0.05]

    class domain_rand(ElSpiderAirBatchRolloutCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(ElSpiderAirBatchRolloutCfg.viewer):
        ref_env = 0
        pos = [2, 2, 2.0]  # [m]
        lookat = [0.0, 0.0, 0.0]  # [m]


class ElSpiderAirDialMPCCfgPPO(ElSpiderAirBatchRolloutCfgPPO):
    class policy(ElSpiderAirBatchRolloutCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(ElSpiderAirBatchRolloutCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(ElSpiderAirBatchRolloutCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_batch_rollout_flat'
        load_run = -1
        max_iterations = 3000
