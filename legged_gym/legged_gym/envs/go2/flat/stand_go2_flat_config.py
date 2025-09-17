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


class StandGo2FlatCfg(Go2FlatCfg):
    class env(Go2FlatCfg.env):
        num_observations = 48
        num_actions = 12
        episode_length_s = 5.

    class commands(Go2FlatCfg.commands):
        num_commands = 4
        resampling_time = 5.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.1, 0.1]  # min max [m/s]
            lin_vel_y = [-0.1, 0.1]  # min max [m/s]
            ang_vel_yaw = [-0.2, 0.2]  # min max [rad/s]
            heading = [-0.0, 0.0]

    class rewards(Go2FlatCfg.rewards):
        class scales(Go2FlatCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.5
            orientation = -10.0
            torques = -0.000025
            action_rate = -0.01
            # Additional rewards
            standing = -10.0
            penalty_in_the_air = -5.0

    class init_state(Go2FlatCfg.init_state):
        pos = [0.0, 0.0, 0.43]  # x,y,z [m]
        # Override default angles, force robot to stand up
        default_joint_angles = {  # [rad]
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 0.9,
            'RL_thigh_joint': 0.9,
            'FR_thigh_joint': 0.9,
            'RR_thigh_joint': 0.9,

            'FL_calf_joint': -1.8,
            'RL_calf_joint': -1.8,
            'FR_calf_joint': -1.8,
            'RR_calf_joint': -1.8,
        }


class StandGo2FlatCfgPPO(Go2FlatCfgPPO):
    class runner(Go2FlatCfgPPO.runner):
        run_name = ''
        experiment_name = 'stand_go2'
        max_iterations = 300
