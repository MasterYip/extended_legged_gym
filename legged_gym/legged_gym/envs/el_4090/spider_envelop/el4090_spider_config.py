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

from legged_gym.envs.elspider_air.mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg, ElSpiderAirRoughCfgPPO


class El4090EnvelopCfg(ElSpiderAirRoughCfg):
    class env(ElSpiderAirRoughCfg.env):
        num_envs = 4096
        num_observations = 74 + 11 * 17
        num_actions = 18
        # Debug settings
        debug_mode = False  # Enable debug output
        debug_interval = 100  # Print debug info every N steps
        debug_env_id = 0  # Which environment to debug (0-based index)

    class terrain(ElSpiderAirRoughCfg.terrain):
        mesh_type = 'plane'  # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]


        curriculum = True
        # Move to a harder terrain if traveled distance > terrain_length * this ratio.
        terrain_curriculum_move_up_distance_ratio = 0.8
        # Move to an easier terrain if traveled distance < command_speed * episode_time * this ratio.
        terrain_curriculum_move_down_command_distance_ratio = 0.5
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.


        measure_heights = False  # If True, the environment will measure the terrain heights at the LiDAR points and provide them as observations.
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]


        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        terrain_length = 5.
        terrain_width = 5.
        num_rows = 8  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete，stepping stones, ]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

        # stepping stones
        stepping_stones_size = 0.5
        stepping_stones_distance = 1.0
        stepping_stones_max_height = 0.2
        stepping_stones_platform_size = 3.0

        difficulty_scale = 0.3
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class lidar:
        enable = True
        # For terrain.mesh_type='plane', Isaac's plane has no triangle mesh for Warp raycasts.
        # "raycast" analytically intersects each LiDAR beam with z=0; "max_range" means no hits.
        # "zero" preserves the old all-zero behavior for ablations only.
        plane_mode = "raycast"  # "raycast", "max_range", "zero"
        zero_on_plane = False
        num_sensors = 1
        horizontal_line_num = 17
        vertical_line_num = 11
        horizontal_fov_deg_min = -60.0
        horizontal_fov_deg_max = 60.0
        vertical_fov_deg_min = -30.0
        vertical_fov_deg_max = 10.0
        min_range = 0.05
        max_range = 5.0
        synchronize = False

        pointcloud_in_world_frame = False
        position = [0.22, 0.0, 0.12]
        orientation_euler_deg = [0.0, 0.0, 0.0]

        debug_viz = False
        debug_env_ids = [0]
        debug_max_points = 341
        debug_point_size = 0.015
        debug_point_color = (0.0, 1.0, 0.2)

        debug_print = False
        debug_print_interval = 50
        debug_print_num_points = 10

    class lighting:
        base_light_color = [0.45, 0.45, 0.45]
        base_light_ambient = [0.05, 0.05, 0.05]
        base_light_direction = [0.0, 0.0, 1.0]

        enhanced_lighting = False
        key_light_color = [0.35, 0.35, 0.35]
        key_light_ambient = [0.08, 0.08, 0.08]
        key_light_direction = [1.0, 1.0, -1.0]

        side_light_color = [0.12, 0.12, 0.12]
        side_light_ambient = [0.02, 0.02, 0.02]
        side_light_direction = [-1.0, 1.0, -0.5]

        fill_light_color = [0.12, 0.12, 0.12]
        fill_light_ambient = [0.02, 0.02, 0.02]
        fill_light_direction = [1.0, -1.0, -0.5]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 10
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2

    class control(ElSpiderAirRoughCfg.control):
        control_type = 'P' #P， P_LOWPASS
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 130., 
                     'HFE': 130., 
                     'KFE': 130.}  # [N*m/rad]
        damping = {'HAA': 4., 
                   'HFE': 4., 
                   'KFE': 4.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        default_dof_pos_filter_tau = 0.4
        default_dof_pos_filter_done_threshold = 0.02


    class asset(ElSpiderAirRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_4090/urdf/el_4090.urdf"
        name = "el_4090"
        foot_name = "FOOT"
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        shoulder_name = "shoulder"
        penalize_contacts_on = ["BASE","SHANK","THIGH"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class init_state(ElSpiderAirRoughCfg.init_state):
        pos = [0.0, 0.0, 0.52]  # x,y,z [m]
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

            "RF_KFE": -0.6,
            "RM_KFE": -0.6,
            "RB_KFE": -0.6,
            "LF_KFE": -0.6,
            "LM_KFE": -0.6,
            "LB_KFE": -0.6,
            
        }
        mammal_default_joint_angles = {
            "RF_HAA": -1.308,
            "RM_HAA": 1.308,
            "RB_HAA": 1.308,
            "LF_HAA": -1.308,
            "LM_HAA": 1.308,
            "LB_HAA": 1.308,

            "RF_HFE": 1.,
            "RM_HFE": 1.,
            "RB_HFE": 1.,
            "LF_HFE": 1.,
            "LM_HFE": 1.,
            "LB_HFE": 1.,

            "RF_KFE": -0.608,
            "RM_KFE": -0.608,
            "RB_KFE": -0.608,
            "LF_KFE": -0.608,
            "LM_KFE": -0.608,
            "LB_KFE": -0.608,
        }

    ## Rewards V2 (faster&smoother gait, zzl-style)
    class rewards(ElSpiderAirRoughCfg.rewards):
        max_contact_force = 400.
        base_height_spider_target = 0.53
        base_height_mammal_target = 0.64

        feet_air_time_target = 0.25

        tripod_contact_threshold = 1.0
        tripod_contact_min_command = 0.1
        envelope_constraint_margin = 0.0
        envelope_constraint_min_command = 0.15
        structure_transition_error_threshold = 0.10
        structure_transition_reward_ramp_time = 2.0
        reset_structure_transition_on_resample = True
        embedded_state_dof_pos_tolerance = 0.12
        embedded_state_haa_pos_tolerance = 0.35

        morphology_haa_range_mammal_limit = 0.38
        morphology_haa_range_relaxed_limit = 0.90
        morphology_haa_range_active_threshold = 0.60
        morphology_haa_range_weight_exponent = 2.0
        haa_swing_min_command = 0.15
        haa_swing_velocity_clip = 4.0
        haa_swing_morphology_relief = 0.7
        
        reset_base_height_with_morphology = True

        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = False  # if true, reward scales should be list
        reward_stage_threshold = 2.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 0

        class scales:
            termination = -0.0

            tracking_lin_vel = 6
            tracking_ang_vel = 2.5

            lin_vel_z = -2
            ang_vel_xy = -0.1
            orientation = -5
            base_height = -50

            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-7

            feet_slip = -0.05 
            feet_air_time = 1.5

            embedded_state_dof_pos = -5.0
            embedded_state_dof_vel = -0.02
            morphology_haa_range = -8.0
            haa_swing = 0.08

            collision = -1.0
            action_rate = -0.001
            stand_still = -1.5

            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01

            feet_contact_forces = -0.03

            feet_async = -0.1
            feet_sync = -0.1
            tripod_contact_pattern = -1
            
            envelope_constraint = -10.0


    class commands(ElSpiderAirRoughCfg.commands):
        curriculum = False
        max_curriculum = 3.0
        # Expand lin_vel_x range when tracking_lin_vel episode average exceeds this fraction of max reward.
        tracking_lin_vel_curriculum_threshold = 0.8
        # Amount added to both positive and negative lin_vel_x range limits per curriculum update.
        command_curriculum_step = 0.5
        # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, envelope boundary conditions,
        # and three morphology priors inferred from front/middle/back envelope parts.
        # Condition order: front_width, middle_width, back_width,
        # forward_limit, backward_limit,
        # morphology_front_prior, morphology_middle_prior, morphology_back_prior.
        num_commands = 4 + 8
        condition_dim = 8
        condition_names = [
            "front_width",
            "middle_width",
            "back_width",
            "forward_limit",
            "backward_limit",
            "morphology_front_prior",
            "morphology_middle_prior",
            "morphology_back_prior",
        ]
        # directional_ratio: lateral width is spider-like, fore/aft reach is mammal-like.
        # weighted_sum preserves the older direct weighted normalization.
        morphology_prior_mode = "directional_ratio"
        morphology_prior_weights = {
            "front": {"lateral": 0.5, "longitudinal": 0.5},
            "middle": {"lateral": 1.0},
            "back": {"lateral": 0.5, "longitudinal": 0.5},
        }
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        print_sampled_condition = False
        sampled_condition_print_interval = 1
        envelope_debug_viz = True
        envelope_debug_env_ids = [0]
        envelope_debug_ground_z_offset = 0.02
        envelope_debug_color = (0.0, 0.85, 1.0)
        envelope_debug_line_radius = 0.012
        envelope_debug_line_samples = 8
        morphology_reachability_test = False
        morphology_reachability_test_mode = "corners"  # "corners", "random", "center"
        morphology_reachability_resample_steps = 600
        morphology_reachability_print_interval = 100
        morphology_reachability_env_id = 0
        morphology_reachability_dof_error_threshold = 0.08
        morphology_reachability_foot_margin = 0.02

        small_command_radio = True

        class ranges(ElSpiderAirRoughCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1., 1.]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
            front_width = [0.3, 0.6]
            middle_width = [0.3, 0.7]
            back_width = [0.3, 0.6]
            forward_limit = [0.6, 0.9]
            backward_limit = [-0.9, -0.6]
            morphology_front_prior = [0.0, 1.0]
            morphology_middle_prior = [0.0, 1.0]
            morphology_back_prior = [0.0, 1.0]

    class domain_rand(ElSpiderAirRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-10., 10.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 3.0
            lidar = 0.05 #1.0
            embedded_state = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise(ElSpiderAirRoughCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.02
            lidar = 0.02

class El4090EnvelopCfgPPO(ElSpiderAirRoughCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm(ElSpiderAirRoughCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = "legged_gym.envs.el_4090.spider_envelop.symmetry:get_elair_lidar_xsym_obs_act"
        
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 3000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'el_4090_envelop'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
