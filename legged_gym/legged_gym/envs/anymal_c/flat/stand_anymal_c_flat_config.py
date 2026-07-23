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


class StandAnymalCFlatCfg(AnymalCRoughCfg):
    class env(AnymalCRoughCfg.env):
        num_observations = 48

    class terrain(AnymalCRoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class asset(AnymalCRoughCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]  # "base"

    class init_state(AnymalCRoughCfg.init_state):
        pos = [0.0, 0.0, 0.9]  # x,y,z [m]
        rot = [0.0, -0.707, 0.0, 0.707]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": -0.0,
            "RH_HAA": -0.0,

            "LF_HFE": 0.8,
            "LH_HFE": 1.0,
            "RF_HFE": 0.8,
            "RH_HFE": 1.0,

            "LF_KFE": -2.0,
            "LH_KFE": 0.8,
            "RF_KFE": -2.0,
            "RH_KFE": 0.8,
        }

    class rewards(AnymalCRoughCfg.rewards):
        max_contact_force = 700.
        base_height_target = 0.9
        class scales (AnymalCRoughCfg.rewards.scales):
            orientation = -4.0
            torques = -0.000025
            feet_air_time = 1.
            base_height = -4.
            collision = -2.
            penalty_in_the_air = -4.
            # feet_contact_forces = -0.01

    class commands(AnymalCRoughCfg.commands):
        heading_command = False
        resampling_time = 4.

        class ranges(AnymalCRoughCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(AnymalCRoughCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        friction_range = [0., 1.5]


class StandAnymalCFlatCfgPPO(AnymalCRoughCfgPPO):
    class policy(AnymalCRoughCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner (AnymalCRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'stand_flat_anymal_c'
        load_run = -1
        max_iterations = 1500
