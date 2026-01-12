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

from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO


class PoseAnymalCFlatCfg(AnymalCRoughCfg):
    class env(AnymalCRoughCfg.env):
        num_observations = 52

    class terrain(AnymalCRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(AnymalCRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class normalization(AnymalCRoughCfg.normalization):
        class obs_scales(AnymalCRoughCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class rewards(AnymalCRoughCfg.rewards):
        max_contact_force = 350.

        class scales (AnymalCRoughCfg.rewards.scales):
            orientation = -5.0
            base_height = -30.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01

    class commands(AnymalCRoughCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 8
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        pose_command = True

        class ranges:
            lin_vel_x = [-1.3, 1.3]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # Base Pose commands
            # For stage 1 - 300 episodes
            # base_yaw_shift = [-0., 0.]
            # base_pitch_shift = [-0., 0.]
            # base_roll_shift = [-0., 0.]
            # base_height = [0.4, 0.5]
            # For stage 2 - 300-1000 episodes
            base_yaw_shift = [-0., 0.]
            base_pitch_shift = [-0.5, 0.5]
            base_roll_shift = [-0.3, 0.3]
            base_height = [0.3, 0.7]
            # Test
            # base_yaw_shift = [-0., 0.]
            # base_pitch_shift = [0., 0.]
            # base_roll_shift = [0.4, 0.5]
            # base_height = [0.3, 0.7]

    class domain_rand(AnymalCRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        friction_range = [0., 1.5]


class PoseAnymalCFlatCfgPPO(AnymalCRoughCfgPPO):
    class policy(AnymalCRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (AnymalCRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'pose_anymal_c_flat'
        load_run = -1
        max_iterations = 1000
