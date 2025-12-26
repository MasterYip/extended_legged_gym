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


class Go2TrajGradSamplingCfg(RobotTrajGradSamplingCfg):
    class env(RobotTrajGradSamplingCfg.env):
        num_envs = 1        # Number of main environments
        rollout_envs = 256   # Number of rollout environments per main env
        env_spacing = 2.0

        # Go2 specific settings
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class trajectory_opt(RobotTrajGradSamplingCfg.trajectory_opt):
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters
        num_diffuse_steps = 2  # Number of diffusion steps (Ndiffuse)
        num_diffuse_steps_init = 6  # Number of initial diffusion steps (Ndiffuse_init)

        # Sampling parameters
        num_samples = 255  # Number of samples to generate at each diffusion step (Nsample)
        temp_sample = 0.05  # Temperature parameter for softmax weighting

        # Control parameters
        horizon_samples = 16  # Horizon length in samples (Hsample)
        horizon_nodes = 4  # Number of control nodes within horizon (Hnode)
        horizon_diffuse_factor = 0.9  # How much more noise to add for further horizon
        traj_diffuse_factor = 0.5  # Diffusion factor for trajectory
        noise_scaling = 0.8
        # Update method
        update_method = "mppi"  # Update method, options: ["mppi", "wbfo", "avwbfo"]
        gamma = 1.00  # Discount factor for rewards in avwbfo

        # Interpolation method for trajectory conversion
        interp_method = "spline"  # Options: ["linear", "spline"]

        # Whether to compute and store predicted trajectories
        compute_predictions = False

    class rl_warmstart(RobotTrajGradSamplingCfg.rl_warmstart):
        enable = False
        policy_checkpoint = "/home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/logs/go2_traj_grad_sampling/model_200.pt"
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
        terrain_file = "/home/user/CodeSpace/Python/terrains/confined_terrain2.obj"
        measure_heights = False
        curriculum = False
        # Origin generation method
        random_origins: bool = False  # Use random origins instead of grid
        origin_generation_max_attempts: int = 10000  # Maximum attempts to generate valid random origins
        origins_x_range: list = [-20.0, 20.0]  # min/max range for random x position
        origins_y_range: list = [-20.0, 20.0]  # min/max range for random y position
        height_clearance_factor: float = 2.0  # Required clearance as multiple of nominal_height

    # Ray caster configuration
    class raycaster(RobotTrajGradSamplingCfg.raycaster):
        enable_raycast = False  # Set to True to enable ray casting
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
    class sdf(RobotTrajGradSamplingCfg.sdf):
        enable_sdf = False      # Set to True to enable SDF calculations
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

    class commands(RobotTrajGradSamplingCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        # Command ranges for Go2 (conservative for trajectory optimization)
        class ranges(RobotTrajGradSamplingCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]  # min max [m/s] - reduced for trajectory optimization
            lin_vel_y = [-0.6, 0.6]   # min max [m/s] - reduced for trajectory optimization
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s] - reduced for trajectory optimization
            heading = [-3.14, 3.14]

    class init_state(RobotTrajGradSamplingCfg.init_state):
        pos = [0.0, 0.0, 0.27]  # x,y,z [m] - Go2 height
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.9,
            'FL_calf_joint': -1.8,
            'FR_hip_joint': 0.0,
            'FR_thigh_joint': 0.9,
            'FR_calf_joint': -1.8,
            'RL_hip_joint': 0.0,
            'RL_thigh_joint': 0.9,
            'RL_calf_joint': -1.8,
            'RR_hip_joint': 0.0,
            'RR_thigh_joint': 0.9,
            'RR_calf_joint': -1.8,
        }

    class control(RobotTrajGradSamplingCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        jointpos_action_normalization = False
        # PD Drive parameters:
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # IMPORTANT: for trajectory optimization, do not use the actuator network
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/go2_actuator_net.pt"

    class asset(RobotTrajGradSamplingCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_description.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(RobotTrajGradSamplingCfg.rewards):
        max_contact_force = 350.
        base_height_target = 0.30
        only_positive_rewards = False
        # Multi-stage rewards optimized for trajectory optimization
        multi_stage_rewards = False  # if true, reward scales should be list
        reward_stage_threshold = 5.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1
        tracking_sigma = 0.25

        class scales():
            # Set reward weights optimized for Go2 trajectory optimization
            termination = -0.0

            # Reward terms from unitree_go2_env.py adapted for trajectory optimization
            gaits = 0.1          # Control foot positions for gait pattern
            air_time = 0.0       # Reward for proper foot air time
            pos = 0.0            # Position tracking
            upright = 0.5        # Maintain upright orientation
            yaw = 0.3            # Yaw orientation tracking
            vel = 1.0            # Linear velocity tracking
            ang_vel = 0.3        # Angular velocity tracking
            height = 10.0         # Height maintenance
            energy = 0.0         # Energy consumption penalty
            alive = 0.0          # Stay alive reward

            # Standard tracking rewards for trajectory optimization
            # tracking_lin_vel = 0.8  # Higher weight for trajectory optimization
            # tracking_ang_vel = 0.4
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.03
            # orientation = -8.0
            # torques = -0.00001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -8.0
            # feet_air_time = 0.5
            # collision = -1.
            # feet_stumble = -0.0
            # action_rate = -0.001
            # stand_still = -0.

            # # Trajectory optimization specific rewards
            # action_smoothness = -0.005  # Encourage smooth actions for trajectory optimization
            # base_stability = 0.1        # Reward stable base motion

    class gait_scheduler:
        period = 1.0
        duty = 0.5
        foot_phases = [0.0, 0.5, 0.0, 0.5]  # For 4-legged robot
        dt = 0.02
        swing_height = 0.1
        track_sigma = 0.25

    class domain_rand(RobotTrajGradSamplingCfg.domain_rand):
        randomize_base_mass = False
        added_mass_range = [-1., 1.]  # Go2 is lighter than Anymal
        # Reduced randomization for trajectory optimization stability
        randomize_friction = False
        randomize_restitution = False

    class viewer(RobotTrajGradSamplingCfg.viewer):
        ref_env = 0
        pos = [-14.0, -14.0, 2.0]  # [m]
        lookat = [-16.0, -16.0, 0.0]  # [m]
        render_rollouts = False

    class sim(RobotTrajGradSamplingCfg.sim):
        dt = 0.005  # Smaller timestep for trajectory optimization precision
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(RobotTrajGradSamplingCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class Go2TrajGradSamplingCfgPPO(RobotTrajGradSamplingCfgPPO):
    class policy(RobotTrajGradSamplingCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(RobotTrajGradSamplingCfgPPO.algorithm):
        entropy_coef = 0.01
        # Trajectory optimization specific training parameters
        learning_rate = 3e-4
        num_learning_epochs = 8  # More epochs for stable learning
        mini_batch_size = 4096   # Larger batch size for stability

    class runner(RobotTrajGradSamplingCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_traj_grad_sampling'
        load_run = -1
        max_iterations = 2000  # Fewer iterations for trajectory optimization
        multi_stage_rewards = False
        save_interval = 100  # More frequent saves
