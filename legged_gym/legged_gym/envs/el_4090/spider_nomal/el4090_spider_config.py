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


class El4090SpiderCfg(ElSpiderAirRoughCfg):
    class env(ElSpiderAirRoughCfg.env):
        num_envs = 4096
        # num_envs = 2048
        num_observations = 66 + 187
        num_actions = 18
        # Debug settings
        debug_mode = False  # Enable debug output
        debug_interval = 100  # Print debug info every N steps
        debug_env_id = 0  # Which environment to debug (0-based index)

    class terrain(ElSpiderAirRoughCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
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


        measure_heights = True
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

        difficulty_scale = 0.8
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces


    class control(ElSpiderAirRoughCfg.control):
        control_type = 'P'
        # PD Drive parameters matching Anymal:
        stiffness = {'HAA': 130., 
                     'HFE': 130., 
                     'KFE': 130.}  # [N*m/rad]
        damping = {'HAA': 2., 
                   'HFE': 2., 
                   'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


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
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.0,
            "RM_HFE": 0.0,
            "RB_HFE": 0.0,
            "LF_HFE": 0.0,
            "LM_HFE": 0.0,
            "LB_HFE": 0.0,

            "RF_KFE": 0.0,
            "RM_KFE": 0.0,
            "RB_KFE": 0.0,
            "LF_KFE": 0.0,
            "LM_KFE": 0.0,
            "LB_KFE": 0.0,
        }

    ## Rewards V2 (faster&smoother gait, zzl-style)
    class rewards(ElSpiderAirRoughCfg.rewards):
        max_contact_force = 350.
        base_height_target = 0.5

        feet_clearance_target = 0.12
        feet_clearance_lookahead = 0.2
        feet_clearance_contact_threshold = 1.0
        feet_clearance_min_command = 0.1
        debug_feet_clearance = False
        debug_feet_clearance_env_id = 0
        debug_feet_clearance_interval = 10
        feet_air_time_target = 0.25
        tripod_contact_threshold = 1.0
        tripod_contact_min_command = 0.1

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

            tracking_lin_vel = 5
            tracking_ang_vel = 2.5

            lin_vel_z = -2
            ang_vel_xy = -0.5
            orientation = -5
            torques = -1e-6#-1e-5
            dof_vel = -1e-6#-1e-5
            dof_acc = -1e-8#-1e-7
            base_height = -50
            feet_slip = -0.05 
            feet_air_time = 3

            tripod_contact_pattern = -5
            feet_terrain_clearance = -5

            collision = -1.
            feet_stumble = -5
            action_rate = -0.01
            stand_still = -3
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01
            feet_contact_forces = -0.01
            # stand_on_six_legs = -1
            shank_vertical = -2
            feet_async = -5.
            feet_sync = -5.
            # contact_force_balance = -1


    class commands(ElSpiderAirRoughCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        # Expand lin_vel_x range when tracking_lin_vel episode average exceeds this fraction of max reward.
        tracking_lin_vel_curriculum_threshold = 0.8
        # Amount added to both positive and negative lin_vel_x range limits per curriculum update.
        command_curriculum_step = 0.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(ElSpiderAirRoughCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1., 1.]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

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
            height_measurements = 5.0
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

class El4090SpiderCfgPPO(ElSpiderAirRoughCfgPPO):
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
            data_augmentation_func = "legged_gym.envs.el_4090.thirdparty.symmetry:get_elair_xsym_obs_act"
        
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 3000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'el_4090_spider_trimesh'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
