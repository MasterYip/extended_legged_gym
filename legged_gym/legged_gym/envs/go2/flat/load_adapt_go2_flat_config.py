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

from legged_gym.envs.go2.flat.go2_flat_config import Go2FlatCfg, Go2FlatCfgPPO


class LoadAdaptGo2FlatCfg(Go2FlatCfg):
    class env(Go2FlatCfg.env):
        num_observations = 48
        num_actions = 12

    class commands(Go2FlatCfg.commands):
        resampling_time = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards(Go2FlatCfg.rewards):
        # Negative coefficients represent costs
        class scales(Go2FlatCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            collision = -1.
            action_rate = -0.01
            stumble = -0.0
            stand_still = -0.

    class domain_rand(Go2FlatCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.


class LoadAdaptGo2FlatCfgPPO(Go2FlatCfgPPO):
    class algorithm(Go2FlatCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(Go2FlatCfgPPO.runner):
        run_name = ''
        experiment_name = 'load_adapt_go2'
        max_iterations = 300
