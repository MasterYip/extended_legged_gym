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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .anymal_c_rough_config import AnymalCRoughCfg


class AnymalCRoughStudentCfg(AnymalCRoughCfg):

    class env(AnymalCRoughCfg.env):
        # Student observations: 48 (proprio) * 3 (history) = 144
        num_observations = 144
        # Privileged observations for distillation: 48 (proprio) + 187 (height scan) = 235
        num_privileged_obs = 235
        # History length for student observations
        history_length = 3

    class terrain(AnymalCRoughCfg.terrain):
        # Enable terrain height measurements for privileged observations
        mesh_type = 'trimesh'
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class rewards(AnymalCRoughCfg.rewards):
        # Keep same reward settings as base config
        pass


class AnymalCRoughStudentCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        teacher_hidden_dims = [512, 256, 128]
        student_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm( ):
        # Distillation algorithm parameters (matching distillation.py)
        num_learning_epochs = 1
        gradient_length = 15
        learning_rate = 1e-3
        max_grad_norm = 1.0
        loss_type = "mse"

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'StudentTeacher'  # Use StudentTeacher for distillation
        algorithm_class_name = 'Distillation'  # Use Distillation instead of PPO
        num_steps_per_env = 24
        max_iterations = 1500

        # Teacher model path (update this with actual teacher model path)
        teacher_model_path = "/home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/legged_gym_cmp/legged_gym/logs/rough_anymal_c_teacher/Jun06_11-09-08_/model_550.pt"

        # Logging
        save_interval = 50
        experiment_name = 'rough_anymal_c_student'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
