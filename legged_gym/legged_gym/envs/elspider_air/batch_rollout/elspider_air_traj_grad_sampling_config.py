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
from legged_gym import LEGGED_GYM_ROOT_DIR

class ElSpiderAirTrajGradSamplingCfg(RobotTrajGradSamplingCfg):
    class env(RobotTrajGradSamplingCfg.env):
        num_envs = 1               # Number of main environments
        rollout_envs = 128         # Number of rollout environments per main env (for trajectory optimization)
        env_spacing = 1.0
        # ElSpider Air specific settings
        num_observations = 66
        num_actions = 18
        episode_length_s = 20  # episode length in seconds

    class trajectory_opt(RobotTrajGradSamplingCfg.trajectory_opt):
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters (similar to dial_config.py)
        # BUG: When this is set to 1, action will converge to 0.0
        num_diffuse_steps = 1  # Number of diffusion steps (Ndiffuse)
        num_diffuse_steps_init = 6  # Number of initial diffusion steps (Ndiffuse_init)

        # Sampling parameters
        num_samples = 127  # Number of samples to generate at each diffusion step (Nsample)
        temp_sample = 0.1  # Temperature parameter for softmax weighting

        # Control parameters
        horizon_samples = 16  # Horizon length in samples (Hsample)
        horizon_nodes = 4  # Number of control nodes within horizon (Hnode)
        horizon_diffuse_factor = 0.9  # How much more noise to add for further horizon
        traj_diffuse_factor = 0.5  # Diffusion factor for trajectory
        noise_scaling = 1.5

        # Update method
        update_method = "avwbfo"  # Update method, options: ["mppi", "wbfo", "avwbfo"]
        gamma = 0.999  # Discount factor for rewards in avwbfo

        # Interpolation method for trajectory conversion
        interp_method = "spline"  # Options: ["linear", "spline"]

        # Whether to compute and store predicted trajectories
        compute_predictions = False

    class rl_warmstart(RobotTrajGradSamplingCfg.rl_warmstart):
        enable = True
        policy_checkpoint = f"{LEGGED_GYM_ROOT_DIR}/ckpt/elspider_air/plane_walk_300.pt"
        actor_network = "mlp"  # options: ["mlp", "lstm"]
        device = "cuda:0"
        # Network architecture settings
        actor_hidden_dims = [128, 64, 32]    # Hidden dimensions for actor network
        critic_hidden_dims = [128, 64, 32]   # Hidden dimensions for critic network
        activation = 'elu'                   # Activation function: elu, relu, selu, etc.
        # Whether to use RL policy for appending new actions during shift
        use_for_append = True
        # Whether to standardize observations for policy input
        standardize_obs = True
        # Input type for the policy
        obs_type = "non_privileged"  # options: ["privileged", "non_privileged"]

    class terrain(RobotTrajGradSamplingCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = False

        # Curriculum Settings
        curriculum = False
        max_init_terrain_level = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 8  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)

        # TerrainObj Settings
        use_terrain_obj = True  # use TerrainObj class to create terrain
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        # Origin generation method
        random_origins: bool = True  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        # BUG: lowbd and upbd should NOT be the SAME (It may affect simulation time) (Unknown reason)
        # origins_x_range: list = [0.8, 0.9]  # min/max range for random x position
        # origins_y_range: list = [-2.0, -1.9]  # min/max range for random y position
        origins_x_range: list = [0.5, 1.5]  # min/max range for random x position
        origins_y_range: list = [-5, 1]  # min/max range for random y position
        height_clearance_factor: float = 1.5  # Required clearance as multiple of nominal_height

    class raycaster(RobotTrajGradSamplingCfg.raycaster):
        enable_raycast = False      # Set to True to enable ray casting
        ray_pattern = "spherical"  # Options: single, grid, cone, spherical
        num_rays = 10              # Number of rays for cone pattern
        ray_angle = 30.0           # Cone angle in degrees
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
        query_bodies = ["trunk", "RF_SHANK", "RM_SHANK", "RB_SHANK", "LF_SHANK", "LM_SHANK", "LB_SHANK"]

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
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotTrajGradSamplingCfg.init_state):
        pos = [0.0, 0.0, 0.30]  # x,y,z [m]
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

    class control(RobotTrajGradSamplingCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2  # Enable Network-0.3 | Disable Network-0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # No actuator network for flat environment
        use_actuator_network = False

    class asset(RobotTrajGradSamplingCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini_collsp.urdf"
        name = "elspider"
        foot_name = "FOOT"
        penalize_contacts_on = ["base", "HIP", "THIGH", "SHANK"]  # "SHANK" may collide with the ground through foot
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class rewards(RobotTrajGradSamplingCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.40
        only_positive_rewards = False

        # No multi-stage rewards for flat environment
        multi_stage_rewards = True

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
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
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            async_gait_scheduler = [-0.2, -0.4]

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 1.0
            dof_nominal_pos = [0.05, 0.2]
            reward_foot_z_align = [0.1, 0.6]

    class domain_rand(RobotTrajGradSamplingCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(RobotTrajGradSamplingCfg.viewer):
        ref_env = 0
        pos = [2, 2, 2.0]  # [m]
        lookat = [0.0, 0.0, 0.0]  # [m]
        render_rollouts = False


class ElSpiderAirTrajGradSamplingCfgPPO(RobotTrajGradSamplingCfgPPO):
    class policy(RobotTrajGradSamplingCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotTrajGradSamplingCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(RobotTrajGradSamplingCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_traj_grad_sampling'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True
