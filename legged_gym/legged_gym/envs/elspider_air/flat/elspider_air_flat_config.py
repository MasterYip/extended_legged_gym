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

from legged_gym.envs import ElSpiderAirRoughCfg, ElSpiderAirRoughCfgPPO


class ElSpiderAirFlatCfg(ElSpiderAirRoughCfg):
    class env(ElSpiderAirRoughCfg.env):
        num_observations = 66

    class terrain(ElSpiderAirRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(ElSpiderAirRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class control(ElSpiderAirRoughCfg.control):
        # PD Drive parameters matching Anymal:
        # stiffness = {'HAA': 50., 'HFE': 50., 'KFE': 50.}  # [N*m/rad]
        # damping = {'HAA': 1.5, 'HFE': 1.5, 'KFE': 1.5}     # [N*m*s/rad]
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # Enable Network-0.5 | Disable Network-0.3

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class init_state(ElSpiderAirRoughCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.2,
            "RM_HFE": 0.2,
            "RB_HFE": 0.2,
            "LF_HFE": 0.2,
            "LM_HFE": 0.2,
            "LB_HFE": 0.2,

            "RF_KFE": 0.3,
            "RM_KFE": 0.3,
            "RB_KFE": 0.3,
            "LF_KFE": 0.3,
            "LM_KFE": 0.3,
            "LB_KFE": 0.3,
        }

    ## Rewards V1 (normal dof_acc)
    class rewards(ElSpiderAirRoughCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.28
        only_positive_rewards = False
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait (with / w\o actuator net)
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = [-3.0, -3.0]
            torques = -0.00005
            dof_vel = [-0.0002, -0.001]
            dof_acc = [-5e-8, -2.5e-7]
            base_height = [-4.0, -1.0]
            feet_slip = [-0.0, -0.2]  # Before feet_air_time
            feet_air_time = [0.8, 1.0]
            collision = -1.
            feet_stumble = -0.0
            action_rate = [-0.005, -0.01]
            stand_still = -0.4  # May affect spot turning
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.05, -0.1]
            
            # gait_scheduler = -18.0
            # async_gait_scheduler = -0.4
            gait_2_step = [-3.0, -0.0]

    ## Rewards V0 (small dof_acc)
    # class rewards(ElSpiderAirRoughCfg.rewards):
    #     max_contact_force = 500.
    #     base_height_target = 0.28
    #     only_positive_rewards = True
    #     # Multi-stage
    #     # Stage 0: Learn to walk with tripod gait
    #     # Stage 1: Correct DOF and FootZ positions / Prevent Slip
    #     multi_stage_rewards = True  # if true, reward scales should be list
    #     reward_stage_threshold = 6.0
    #     reward_min_stage = 0  # Start from 0
    #     reward_max_stage = 1

    #     class scales:
    #         termination = -0.0
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 0.5
    #         lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         orientation = -5.0
    #         torques = -0.00001
    #         dof_vel = -0.
    #         dof_acc = -5e-8
    #         base_height = -8.0
    #         feet_slip = [-0.0, -0.4]  # Before feet_air_time
    #         feet_air_time = 0.8
    #         collision = -1.
    #         feet_stumble = -0.0
    #         action_rate = -0.001
    #         stand_still = -0.
    #         dof_pos_limits = -1.0
            
    #         # gait_scheduler = -18.0
    #         # async_gait_scheduler = -0.4
    #         gait_2_step = -5.0
    #         # feet_contact_forces = -0.01

    #     class async_gait_scheduler:
    #         # Reward for the async gait scheduler
    #         dof_align = 1.0
    #         dof_nominal_pos = [0.0, 0.2]
    #         reward_foot_z_align = [0.0, 0.6]


    class commands(ElSpiderAirRoughCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges(ElSpiderAirRoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(ElSpiderAirRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class noise(ElSpiderAirRoughCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

class ElSpiderAirFlatCfgPPO(ElSpiderAirRoughCfgPPO):
    class policy(ElSpiderAirRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(ElSpiderAirRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (ElSpiderAirRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_elspider_air'
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True

    class algorithm(ElSpiderAirRoughCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = "legged_gym.envs.elspider_air.elspider:get_symmetric_observation_action"
        