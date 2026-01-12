#!/usr/bin/env python3

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

# autopep8:off
import os
import sys
import numpy as np
from isaacgym import gymapi, gymutil
from legged_gym.envs.base.base_task import BaseTask
import torch
import time

from legged_gym.utils.math_utils import quat_apply_yaw
from legged_gym.utils import task_registry, get_args
from legged_gym.envs.batch_rollout.robot_batch_rollout import RobotBatchRollout
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.helpers import class_to_dict, get_load_path
# autopep8:on


def test_robot_batch_rollout(args):
    """Test the RobotBatchRollout class with main step and rollout step functions."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Define noise scales for rollout environments (one per action dimension)
    noise_scales = 5*torch.ones(env_cfg.env.num_actions, device=env.device) * 0.1

    print("Starting simulation loop...")

    # Run for 100 steps
    for i in range(1000):
        # Generate random actions for main environments
        # main_actions = torch.rand((env_cfg.env.num_envs, env_cfg.env.num_actions), device=env.device) * 2 - 1
        main_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions), device=env.device)
        print("Step main environments...")
        for j in range(5):
            # Step main environments
            main_obs, main_privileged_obs, main_rew, main_reset, main_extras = env.step(main_actions)
            time.sleep(1.0)

        # Use the same actions as mean for rollout environments with noise
        print("Step rollout environments...")
        for j in range(20):
            rollout_obs, rollout_privileged_obs, rollout_rew, rollout_reset, rollout_extras = env.step_rollout(
                main_actions, noise_scales)
            time.sleep(0.05)

    print("Test completed successfully!")


if __name__ == "__main__":
    args = get_args()
    test_robot_batch_rollout(args)
