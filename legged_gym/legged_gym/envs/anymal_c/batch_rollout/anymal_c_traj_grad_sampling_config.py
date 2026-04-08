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
from traj_sampling.config.trajectory_optimization_config import AnymalCTrajectoryOptCfg

class AnymalCTrajGradSamplingCfg(RobotTrajGradSamplingCfg):
    class env(RobotTrajGradSamplingCfg.env):
        num_envs = 1        # Number of main environments
        rollout_envs = 1   # Number of rollout environments per main env
        env_spacing = 0.4

        # AnymalC specific settings
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class trajectory_opt(AnymalCTrajectoryOptCfg.trajectory_opt):
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters
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
        gamma = 1.00  # Discount factor for rewards in avwbfo

        # Interpolation method for trajectory conversion
        interp_method = "spline"  # Options: ["linear", "spline"]

        # Whether to compute and store predicted trajectories
        compute_predictions = True
        

    class rl_warmstart(AnymalCTrajectoryOptCfg.rl_warmstart):
        enable = True
        policy_checkpoint = f"{LEGGED_GYM_ROOT_DIR}/ckpt/anymal_c/plane_walk_200.pt"
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
        use_terrain_obj = False  # use TerrainObj class to create terrain
        mesh_type = 'plane'
        # path to the terrain file
        terrain_file = "/home/user/CodeSpace/Python/terrains/confined_terrain3.obj"
        measure_heights = False
        curriculum = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.25, 0.5, 0.75, 1.0]
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [15.0, 15.0]  # min/max range for random x position
        origins_y_range: list = [18.0, 18.0]  # min/max range for random y position
        height_clearance_factor: float = 1.0  # Required clearance as multiple of nominal_height


    # Ray caster configuration
    class raycaster(RobotTrajGradSamplingCfg.raycaster):
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
    class sdf(RobotTrajGradSamplingCfg.sdf):
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

    class commands(RobotTrajGradSamplingCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # Command ranges
        class ranges(RobotTrajGradSamplingCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(RobotTrajGradSamplingCfg.init_state):
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

    class control(RobotTrajGradSamplingCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        jointpos_action_normalization = False

        # # Traj Grad Sampling specific settings:
        # stiffness = {'HAA': 200., 'HFE': 200., 'KFE': 200.}  # [N*m/rad]
        # damping = {'HAA': 4, 'HFE': 4, 'KFE': 4}     # [N*m*s/rad]
        # # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 1.0

        # RL Training
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # IMPORTANT: for batch rollout, do not use the actuator network
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(RobotTrajGradSamplingCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c_boldshankcoll.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH", "base"]
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(RobotTrajGradSamplingCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.5
        only_positive_rewards = False
        # Multi-stage rewards
        multi_stage_rewards = False  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1
        tracking_sigma = 0.25

        class scales():
            # Set reward weights matching unitree_go2_env.py
            termination = -0.0
            
            # Reward terms from unitree_go2_env.py
            # gaits = 0.1          # Control foot positions for gait pattern
            # air_time = 0.0       # Reward for proper foot air time
            # pos = 0.0            # Position tracking
            # upright = 1.3        # Maintain upright orientation 
            # yaw = 0.3            # Yaw orientation tracking
            # vel = 0.6            # Linear velocity tracking
            # ang_vel = 1.0        # Angular velocity tracking
            # height = 1.0         # Height maintenance
            # energy = 0.0         # Energy consumption penalty
            # alive = 0.0          # Stay alive reward

            # Inherited rewards with zero weights (not used in unitree_go2_env)
            # tracking_lin_vel = 0.0
            # tracking_ang_vel = 0.0
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0
            # feet_slip = 0.0
            # feet_air_time = 0.0
            # collision = 0.0
            # feet_stumble = 0.0
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            termination = -0.0
            tracking_lin_vel = 5.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.5
            orientation = -2.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            # base_height = -10
            feet_air_time = 1.0
            collision = -2
            feet_stumble = -0.0
            action_rate = -0.001
            stand_still = -0.

            # New reward terms for AnymalC
            # no_fly = 0.0          # Reward for having at least one foot on the ground
            # gait_scheduler = 0.0  # Reward for following gait scheduler

    class gait_scheduler:
        period = 1.0
        duty = 0.5
        foot_phases = [0.0, 0.5, 0.0, 0.5]  # For 4-legged robot
        dt = 0.02
        swing_height = 0.1
        track_sigma = 0.25

    class domain_rand(RobotTrajGradSamplingCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class viewer(RobotTrajGradSamplingCfg.viewer):
        ref_env = 0
        pos = [15, 15, 4.0]  # [m]
        lookat = [20, 20, 0.0]  # [m]
        render_rollouts = False

class AnymalCTrajGradSamplingCfgPPO(RobotTrajGradSamplingCfgPPO):
    class policy(RobotTrajGradSamplingCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotTrajGradSamplingCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(RobotTrajGradSamplingCfgPPO.runner):
        run_name = ''
        experiment_name = 'anymal_c_traj_grad_sampling'
        load_run = -1
        max_iterations = 3000
