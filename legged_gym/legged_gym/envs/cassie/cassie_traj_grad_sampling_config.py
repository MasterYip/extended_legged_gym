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

from legged_gym.envs.batch_rollout.robot_traj_grad_sampling_config import RobotTrajGradSamplingCfg, RobotTrajGradSamplingCfgPPO


class CassieTrajGradSamplingCfg(RobotTrajGradSamplingCfg):
    class env(RobotTrajGradSamplingCfg.env):
        num_envs = 1               # Number of main environments
        rollout_envs = 64          # Number of rollout environments per main env (for trajectory optimization)
        env_spacing = 0.2
        # Cassie specific settings
        num_observations = 169
        num_actions = 12
        episode_length_s = 20      # episode length in seconds

    class trajectory_opt(RobotTrajGradSamplingCfg.trajectory_opt):
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters
        num_diffuse_steps = 2      # Number of diffusion steps (Ndiffuse)
        num_diffuse_steps_init = 6  # Number of initial diffusion steps (Ndiffuse_init)

        # Sampling parameters
        num_samples = 63           # Number of samples to generate at each diffusion step (Nsample)
        temp_sample = 0.2          # Temperature parameter for softmax weighting

        # Control parameters
        horizon_samples = 64       # Horizon length in samples (Hsample)
        horizon_nodes = 16         # Number of control nodes within horizon (Hnode)
        horizon_diffuse_factor = 0.9  # How much more noise to add for further horizon
        traj_diffuse_factor = 0.5  # Diffusion factor for trajectory

        # Update method
        update_method = "wbfo"     # Update method, options: ["mppi", "wbfo"]

        # Whether to compute and store predicted trajectories
        compute_predictions = True

    class terrain(RobotTrajGradSamplingCfg.terrain):
        use_terrain_obj = True     # use TerrainObj class to create terrain
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        measure_heights = True
        curriculum = False
        mesh_type = 'plane'        # Use a simple plane
        terrain_length = 8.
        terrain_width = 8.
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class raycaster(RobotTrajGradSamplingCfg.raycaster):
        enable_raycast = False      # Set to True to enable ray casting
        ray_pattern = "spherical"  # Options: single, grid, cone, spherical
        num_rays = 128             # Number of rays
        ray_angle = 35.0           # Cone angle in degrees
        terrain_file = "resources/terrains/confined/confined_terrain.obj"  # Path to terrain mesh file
        max_distance = 10.0        # Maximum ray cast distance
        attach_yaw_only = False    # If True, only yaw rotation is applied to rays
        offset_pos = [0.0, 0.0, 0.0]  # Offset from robot base
        # For spherical pattern
        spherical_num_azimuth = 16
        spherical_num_elevation = 8

    class sdf(RobotTrajGradSamplingCfg.sdf):
        enable_sdf = False          # Set to True to enable SDF calculations
        mesh_paths = ["resources/terrains/confined/confined_terrain.obj"]  # Paths to mesh files for SDF calculation
        max_distance = 10.0        # Maximum SDF distance to compute
        enable_caching = True      # Enable SDF query caching for performance
        update_freq = 1            # Update SDF values every N steps

        # Robot body parts to compute SDF for
        query_bodies = ["pelvis", "toe_left", "toe_right"]

        # Whether to compute SDF gradients
        compute_gradients = True
        # Whether to compute nearest points on mesh
        compute_nearest_points = True
        # Whether to include SDF values in observations
        include_in_obs = True

    class commands(RobotTrajGradSamplingCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 5.       # time before command are changed[s]
        heading_command = False    # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-2.0, 2.0]     # min max [m/s]
            lin_vel_y = [-0.8, 0.8]     # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotTrajGradSamplingCfg.init_state):
        pos = [0.0, 0.0, 0.9]      # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {   # = target angles [rad] when action = 0.0
            'hip_abduction_left': 0.0,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.5,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.0,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.5,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        }

    class control(RobotTrajGradSamplingCfg.control):
        control_type = 'P'         # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'hip_abduction': 100.0, 'hip_rotation': 100.0,
                     'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                     'toe_joint': 40.}     # [N*m/rad]
        damping = {'hip_abduction': 3.0, 'hip_rotation': 3.0,
                   'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                   'toe_joint': 1.}      # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # No actuator network for this robot configuration
        use_actuator_network = False

    class asset(RobotTrajGradSamplingCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        penalize_contacts_on = []
        terminate_after_contacts_on = ['pelvis']
        self_collisions = 1        # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(RobotTrajGradSamplingCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        # Multi-stage reward system for trajectory planning
        multi_stage_rewards = True
        tracking_sigma = 2.0       # tracking reward = exp(-error^2/sigma)

        class scales(RobotTrajGradSamplingCfg.rewards.scales):
            termination = -200.
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -5.e-6
            dof_vel = -0.0
            dof_acc = -2.e-7
            base_height = -0.0
            feet_slip = -0.0
            feet_air_time = 5.0
            collision = -1.0
            action_rate = -0.01
            stand_still = -0.0
            dof_pos_limits = -1.0
            no_fly = 0.25

    class domain_rand(RobotTrajGradSamplingCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(RobotTrajGradSamplingCfg.viewer):
        ref_env = 0
        pos = [2, 2, 2.0]         # [m]
        lookat = [0.0, 0.0, 0.0]  # [m]


class CassieTrajGradSamplingCfgPPO(RobotTrajGradSamplingCfgPPO):
    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        entropy_coef = 0.01

    class runner:
        run_name = ''
        experiment_name = 'cassie_traj_grad_sampling'
        load_run = -1
        max_iterations = 3000
