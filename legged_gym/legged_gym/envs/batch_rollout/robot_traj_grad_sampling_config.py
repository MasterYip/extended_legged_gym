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

import os
import inspect

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept_config import RobotBatchRolloutPerceptCfg, RobotBatchRolloutPerceptCfgPPO


class RobotTrajGradSamplingCfg(RobotBatchRolloutPerceptCfg):
    class env(RobotBatchRolloutPerceptCfg.env):
        # Same as parent class, with additions for trajectory optimization
        num_envs = 1  # Number of main environments
        rollout_envs = 128  # Number of parallel rollout environments for trajectory optimization

    # Trajectory optimization configuration
    class trajectory_opt:
        # Enable trajectory optimization
        enable_traj_opt = True

        # Diffusion parameters (similar to dial_config.py)
        num_diffuse_steps = 2  # Number of diffusion steps (Ndiffuse)
        num_diffuse_steps_init = 10  # Number of initial diffusion steps (Ndiffuse_init)

        # Sampling parameters
        num_samples = 127  # Number of samples to generate at each diffusion step (Nsample)
        temp_sample = 0.05  # Temperature parameter for softmax weighting

        # Control parameters
        horizon_samples = 16  # Horizon length in samples (Hsample)
        horizon_nodes = 4  # Number of control nodes within horizon (Hnode)
        horizon_diffuse_factor = 0.9  # How much more noise to add for further horizon
        traj_diffuse_factor = 0.5  # Diffusion factor for trajectory
        noise_scaling = 1.0  # Scaling factor for noise in trajectory optimization

        # Update method
        update_method = "mppi"  # Update method, options: ["mppi", "wbfo", "avwbfo"]
        gamma = 0.99  # Discount factor for rewards in avwbfo

        # Interpolation method for trajectory conversion (wbfo/avwbfo only supports spline)
        interp_method = "spline"  # Options: ["linear", "spline"]

        # Whether to compute and store predicted trajectories
        compute_predictions = True

    # RL warmstart configuration

    class rl_warmstart:
        # Enable RL warmstart
        enable = False

        # Path to policy checkpoint
        policy_checkpoint = ""

        # Actor network configuration
        actor_network = "mlp"  # options: ["mlp", "lstm"]

        # Network architecture settings
        actor_hidden_dims = [128, 64, 32]    # Hidden dimensions for actor network
        critic_hidden_dims = [128, 64, 32]   # Hidden dimensions for critic network
        activation = 'elu'                   # Activation function: elu, relu, selu, etc.

        # Device to load the policy on
        device = "cuda:0"

        # Whether to use RL policy for appending new actions during shift
        use_for_append = True

        # Whether to standardize observations for policy input
        standardize_obs = True

        # Input type for the policy
        obs_type = "privileged"  # options: ["privileged", "non_privileged"]

    class raycaster(RobotBatchRolloutPerceptCfg.raycaster):
        # Same as parent class
        pass

    class sdf(RobotBatchRolloutPerceptCfg.sdf):
        # Same as parent class
        pass

    class terrain(RobotBatchRolloutPerceptCfg.terrain):
        # Same as parent class
        pass

    class commands(RobotBatchRolloutPerceptCfg.commands):
        # Same as parent class
        pass

    class init_state(RobotBatchRolloutPerceptCfg.init_state):
        # Same as parent class
        pass

    class control(RobotBatchRolloutPerceptCfg.control):
        # Same as parent class
        # Add option for normalized action space
        jointpos_action_normalization = False  # If True, use normalized actions [-1, 1] and project to joint range

    class asset(RobotBatchRolloutPerceptCfg.asset):
        # Same as parent class
        pass

    class rewards(RobotBatchRolloutPerceptCfg.rewards):
        # Same as parent class
        pass

    class domain_rand(RobotBatchRolloutPerceptCfg.domain_rand):
        # Same as parent class
        pass

    class viewer(RobotBatchRolloutPerceptCfg.viewer):
        # Same as parent class
        pass


class RobotTrajGradSamplingCfgPPO(RobotBatchRolloutPerceptCfgPPO):
    # Same as parent class with updated environment config
    env = RobotTrajGradSamplingCfg
