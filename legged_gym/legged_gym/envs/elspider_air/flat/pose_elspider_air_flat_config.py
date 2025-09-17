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


class PoseElSpiderAirFlatCfg(ElSpiderAirRoughCfg):
    class env(ElSpiderAirRoughCfg.env):
        num_observations = 70

    class terrain(ElSpiderAirRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(ElSpiderAirRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(ElSpiderAirRoughCfg.rewards):
        base_height_target = 0.33
        max_contact_force = 500.
        only_positive_rewards = True
        # Multi-stage rewards
        # Stage 0: Learn to walk with tripod gait
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        # Stage 2: Learn to control pose
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 5.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 2

        class scales (ElSpiderAirRoughCfg.rewards.scales):
            orientation = [-0.5, -0.5, -3.0]
            torques = -0.00001
            action_rate = -0.001
            dof_acc = -5e-8
            feet_slip = [-0.0, -0.3]  # Before feet_air_time
            feet_air_time = 0.8
            # gait_scheduler = -18.0
            async_gait_scheduler = -0.3
            # feet_contact_forces = -0.01
            collision = -1.0
            base_height = [-8.0, -8.0, -12.0]
            # jump_air = -5.0

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 1.0
            dof_nominal_pos = [0.0, 0.2]
            reward_foot_z_align = [0.0, 0.6]

    class commands(ElSpiderAirRoughCfg.commands):
        curriculum = True  # Only for lin_vel_x
        max_curriculum = 2.5  # Maximum value for lin_vel_x
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 8
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        pose_command = True

        class ranges(ElSpiderAirRoughCfg.commands.ranges):
            lin_vel_x = [-1.6, 1.6]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # Base Pose commands
            # For stage 1 - 300 episodes
            # base_yaw_shift = [-0., 0.]
            # base_pitch_shift = [-0., 0.]
            # base_roll_shift = [-0., 0.]
            # base_height = [0.25, 0.25]
            # For stage 2 - 300-1000 episodes
            base_yaw_shift = [-0., 0.]
            base_pitch_shift = [-0.3, 0.3]
            base_roll_shift = [-0.6, 0.6]
            base_height = [0.19, 0.32]
            # Test
            # base_yaw_shift = [-0., 0.]
            # base_pitch_shift = [0., 0.]
            # base_roll_shift = [0.4, 0.5]
            # base_height = [0.3, 0.7]

    class domain_rand(ElSpiderAirRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        friction_range = [0.5, 1.5]


class PoseElSpiderAirFlatCfgPPO(ElSpiderAirRoughCfgPPO):
    class policy(ElSpiderAirRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(ElSpiderAirRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (ElSpiderAirRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'pose_elspider_air_flat'
        load_run = -1
        max_iterations = 3000

        # multi-stage reward (in case of multi-stage rewards)
        multi_stage_rewards = True
