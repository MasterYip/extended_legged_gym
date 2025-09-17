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


class PoseGo2FlatCfg(Go2FlatCfg):
    class env(Go2FlatCfg.env):
        num_observations = 60
        num_actions = 12

    class commands(Go2FlatCfg.commands):
        num_commands = 8
        resampling_time = 5.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-0.0, 0.0]
            # Pose command ranges
            base_yaw_shift = [-0.0, 0.0]  # min max [rad] - base additional rotation
            base_pitch_shift = [-0.3, 0.3]  # min max [rad] - base pitch
            base_roll_shift = [-0.3, 0.3]  # min max [rad] - base roll
            base_height = [0.25, 0.42]  # min max [m] - base height

    class rewards(Go2FlatCfg.rewards):
        class scales(Go2FlatCfg.rewards.scales):
            orientation = -5.0
            base_height = -5.0

    class init_state(Go2FlatCfg.init_state):
        reset_mode = 'reset_to_range'
        # Use height ranges, start with higher pos
        pos = [0.0, 0.0, 0.52]  # x,y,z [m]


class PoseGo2FlatCfgPPO(Go2FlatCfgPPO):
    class runner(Go2FlatCfgPPO.runner):
        run_name = ''
        experiment_name = 'pose_go2'
        load_run = -1
        max_iterations = 300
