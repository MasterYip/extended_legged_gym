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

from legged_gym.envs.batch_rollout.robot_plan_grad_sampling_config import RobotPlanGradSamplingCfg, RobotPlanGradSamplingCfgPPO


class ElSpiderAirPlanGradSamplingCfg(RobotPlanGradSamplingCfg):
    """Configuration for ElSpider Air planning-based trajectory gradient sampling environment."""

    class env(RobotPlanGradSamplingCfg.env):
        num_envs = 1               # Number of main environments
        rollout_envs = 256          # Number of rollout environments per main env (for trajectory optimization)
        env_spacing = 1.0
        # ElSpider Air specific settings
        num_observations = 66      # Standard ElSpider Air observation space
        num_actions = 24           # 6 legs × 3 joints per leg
        episode_length_s = 20      # Episode length in seconds

    class planning(RobotPlanGradSamplingCfg.planning):
        # State velocity optimization parameters
        state_vel_dim = 24         # 3 base_lin_vel + 3 base_ang_vel + 18 joint_vel
        integration_method = "euler"  # Options: "euler", "rk4"
        use_sim_step_for_viewer = True  # Step simulation once for viewer updates

        # State velocity limits for ElSpider Air
        max_base_lin_vel = 2.0     # Maximum base linear velocity [m/s]
        max_base_ang_vel = 3.0     # Maximum base angular velocity [rad/s]
        max_joint_vel = 10.0       # Maximum joint velocity [rad/s]

        # Integration parameters
        max_integration_step = 0.01  # Maximum integration time step [s]
        enforce_joint_limits = False  # Enforce joint position limits during integration

        # Planning-specific features
        include_foot_predictions = False  # Include predicted foot positions in observations

    class trajectory_opt(RobotPlanGradSamplingCfg.trajectory_opt):
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters optimized for planning
        num_diffuse_steps = 2      # Reduced for faster planning with integration
        num_diffuse_steps_init = 8  # Initial diffusion steps for better convergence

        # Sampling parameters
        num_samples = 255           # Number of samples per diffusion step
        temp_sample = 0.10         # Temperature for softmax weighting

        # Control parameters for state velocity optimization
        horizon_samples = 24       # Shorter horizon for faster planning
        horizon_nodes = 6          # Fewer control nodes for efficiency
        horizon_diffuse_factor = 0.85
        traj_diffuse_factor = 0.4

        # Update method
        update_method = "avwbfo"     # MPPI works well with state velocity optimization
        gamma = 1.0

        # Interpolation method
        interp_method = "spline"   # Linear interpolation for state velocities

        # Predictions
        compute_predictions = True  # Compute state predictions for visualization

        noise_scaling = 2.0

    class rl_warmstart(RobotPlanGradSamplingCfg.rl_warmstart):
        enable = False             # Disabled by default for pure planning approach
        policy_checkpoint = ""     # Path to RL policy checkpoint if using warmstart
        actor_network = "mlp"
        device = "cuda:0"
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'
        use_for_append = False     # Don't use RL for appending new actions
        standardize_obs = True
        obs_type = "non_privileged"

    class terrain(RobotPlanGradSamplingCfg.terrain):
        use_terrain_obj = True
        terrain_file = "resources/terrains/confined/confined_terrain.obj"
        measure_heights = False    # Simplified for planning
        curriculum = False
        mesh_type = 'trimesh'
        terrain_length = 5.
        terrain_width = 5.
        # Origin generation
        random_origins: bool = True
        origin_generation_max_attempts: int = 5000
        origins_x_range: list = [0, 0]
        origins_y_range: list = [-2, -2]
        height_clearance_factor: float = 2.0

    class raycaster(RobotPlanGradSamplingCfg.raycaster):
        enable_raycast = False     # Simplified for planning

    class sdf(RobotPlanGradSamplingCfg.sdf):
        enable_sdf = True          # Set to True to enable SDF calculations
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

    class commands(RobotPlanGradSamplingCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 6.       # Longer resampling time for planning
        heading_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]    # Conservative ranges for planning
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.4, 0.4]
            heading = [-3.14, 3.14]

    class init_state(RobotPlanGradSamplingCfg.init_state):
        pos = [0.0, 0.0, 0.30]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # ElSpider Air default joint angles
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.6, "RM_HFE": 0.6, "RB_HFE": 0.6,
            "LF_HFE": 0.6, "LM_HFE": 0.6, "LB_HFE": 0.6,
            "RF_KFE": 0.6, "RM_KFE": 0.6, "RB_KFE": 0.6,
            "LF_KFE": 0.6, "LM_KFE": 0.6, "LB_KFE": 0.6,
        }

    class control(RobotPlanGradSamplingCfg.control):
        control_type = 'P'  # Position control
        # PD gains for ElSpider Air
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}
        action_scale = 0.2
        decimation = 4
        use_actuator_network = False

    class asset(RobotPlanGradSamplingCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini_collsp.urdf"
        name = "elspider"
        foot_name = "FOOT"
        penalize_contacts_on = ["trunk", "HIP", "THIGH", "SHANK"]
        terminate_after_contacts_on = []
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(RobotPlanGradSamplingCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.28
        only_positive_rewards = True
        multi_stage_rewards = False  # Simplified for planning
        tracking_sigma = 0.25

        class scales:
            # Planning-focused reward structure
            # termination = -0.0
            tracking_lin_vel = 2.0      # Higher weight for velocity tracking
            tracking_ang_vel = 1.0      # Higher weight for angular tracking
            lin_vel_z = -0.5            # Penalize vertical velocity
            ang_vel_xy = -0.5           # Penalize roll/pitch rates
            orientation = -1.0         # Strong penalty for tilting
            # torques = -0.00002          # Small torque penalty
            # FIXME: dof_vel affect planning results
            # dof_vel = -0.01              # No joint velocity penalty for planning
            # dof_acc = -1e-7             # Small acceleration penalty
            base_height = -10.0         # Strong height maintenance
            feet_slip = -0.5            # Penalize foot slipping
            # feet_air_time = 1.0         # Reward appropriate air time
            # FIXME: collision penalty is not continuous
            collision = -0.10           # Strong collision penalty
            # action_rate = -0.002        # Smooth state velocity changes
            # stand_still = -0.0
            dof_pos_limits = -2.0       # Strong penalty for joint limits

            # Planning-specific rewards
            # gaits = 0.05  # Reward for maintaining gaits
            # state_vel_smoothness = -0.001  # Penalize rapid state velocity changes
            # energy_efficiency = -0.0001   # Reward energy-efficient motions
            async_gait_scheduler = -0.1

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 0.0
            dof_nominal_pos = 1.0
            reward_foot_z_align = 0.0

    class domain_rand(RobotPlanGradSamplingCfg.domain_rand):
        randomize_base_mass = False  # Simplified for planning
        added_mass_range = [0., 0.]
        randomize_friction = False
        push_robots = False

    class viewer(RobotPlanGradSamplingCfg.viewer):
        ref_env = 0
        pos = [2, 2, 2.0]
        lookat = [0.0, 0.0, 0.0]
        render_rollouts = False  # Visualize rollouts for planning

    class normalization(RobotPlanGradSamplingCfg.normalization):
        clip_observations = 100.
        clip_actions = 10.0  # Higher clip for state velocities

        class obs_scales(RobotPlanGradSamplingCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(RobotPlanGradSamplingCfg.noise):
        add_noise = False  # No noise for planning
        noise_level = 0.0

        class noise_scales:
            dof_pos = 0.0
            dof_vel = 0.0
            lin_vel = 0.0
            ang_vel = 0.0
            gravity = 0.0
            height_measurements = 0.0


class ElSpiderAirPlanGradSamplingCfgPPO(RobotPlanGradSamplingCfgPPO):
    """PPO configuration for ElSpider Air planning environment (if using RL warmstart)."""

    class policy(RobotPlanGradSamplingCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm(RobotPlanGradSamplingCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(RobotPlanGradSamplingCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_plan_grad_sampling'
        load_run = -1
        max_iterations = 1000  # Fewer iterations since planning is the main focus
        multi_stage_rewards = False
