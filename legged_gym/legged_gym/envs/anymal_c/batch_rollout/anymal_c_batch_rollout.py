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

import numpy as np
import os
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import warn

from legged_gym.envs.batch_rollout.robot_batch_rollout_percept import RobotBatchRolloutPercept
from isaacgym.torch_utils import quat_rotate
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, quat_apply
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.anymal_c.batch_rollout.anymal_c_batch_rollout_config import AnymalCBatchRolloutCfg
from legged_gym.utils import GaitScheduler, AsyncGaitScheduler

from legged_gym.utils.math_utils import quat_apply_yaw

class AnymalCBatchRollout(RobotBatchRolloutPercept):
    """AnymalC batch rollout environment using the RobotBatchRolloutPercept class."""

    cfg = AnymalCBatchRolloutCfg()

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize the environment with AnymalC-specific configuration."""

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         sim_device=sim_device,
                         headless=headless)

        # load actuator network if configured
        if hasattr(self.cfg.control, "use_actuator_network") and self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

        # Initialize gait schedulers using configuration from config file
        # Regular gait scheduler
        self.gait_scheduler = GaitScheduler(
            self.height_samples,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.foot_positions,
            self.foot_velocities,
            self.total_num_envs,
            self.device,
            gait_cfg=self.cfg.gait_scheduler
        )

        # Async gait scheduler
        self.async_gait_scheduler = AsyncGaitScheduler(
            self.height_samples,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.foot_positions,
            self.foot_velocities,
            self.total_num_envs,
            self.device,
            gait_cfg=self.cfg.async_gait_scheduler
        )

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _draw_debug_vis(self):
        """Draw debug visualization including base velocity and command velocity."""
        # Call parent class visualization first
        super()._draw_debug_vis()

        # Get velocity data
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        cmd_vel_world = quat_apply_yaw(self.base_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0
        # Draw arrows for all main envs
        for j in range(self.num_envs):
            i = self.main_env_indices[j]
            base_pos = self.root_states[i, :3].cpu().numpy()
            # Draw current velocity (green) and command velocity (red)
            self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))

    def post_physics_step_rollout(self):
        """Update after physics steps for rollout environments."""
        super().post_physics_step_rollout()
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands, self.t_rollout)

    def post_physics_step(self):
        super().post_physics_step()
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands, self.t_main)

    def _init_buffers(self):
        """Initialize buffers including any AnymalC-specific ones."""
        # Initialize parent class buffers
        super()._init_buffers()

        # Initialize actuator network state tensors if using actuator network
        if hasattr(self.cfg.control, "use_actuator_network") and self.cfg.control.use_actuator_network:
            self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
            self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
            self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
            self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
            self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def reset_idx(self, env_ids):
        """Reset environments with given ids."""
        # Reset parent class state
        super().reset_idx(env_ids)

        # Reset actuator network states if using it
        if hasattr(self, 'sea_hidden_state_per_env'):
            self.sea_hidden_state_per_env[:, env_ids] = 0.
            self.sea_cell_state_per_env[:, env_ids] = 0.

    def _compute_torques(self, actions, env_ids=None):
        """Compute joint torques from actions."""
        # Use actuator network if configured
        if hasattr(self.cfg.control, "use_actuator_network") and self.cfg.control.use_actuator_network and hasattr(self, 'actuator_network'):
            self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale +
                                       self.default_dof_pos - self.dof_pos).flatten()
            self.sea_input[:, 0, 1] = self.dof_vel.flatten()
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            torques = torques.view(self.total_num_envs, self.num_actions)
            if env_ids is not None:
                # Select only the torques for the given env_ids
                return torques[env_ids]
            return torques
        else:
            # Use default PD controller from parent class
            return super()._compute_torques(actions, env_ids=env_ids)

    def check_termination(self):
        """Check if environments need to be reset."""
        # Call parent class termination check first
        super().check_termination()

        # Add AnymalC-specific termination conditions
        # Terminate if robot is upside down (z-component of projected gravity > 0)
        self.reset_buf[self.main_env_indices] |= (self.projected_gravity[self.main_env_indices, 2] > 0)

    def _reward_orientation(self):
        """Reward for maintaining the proper orientation."""
        # Penalize non-flat base orientation
        # For Anymal, we want the gravity to be aligned with the z-axis (pointed down)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_async_gait_scheduler(self):
        """Reward for AsyncGaitScheduler's performance."""
        # Reward for Async Gait Scheduler
        gait_scheduler_scales = class_to_dict(self.cfg.rewards.async_gait_scheduler)

        def get_weight(key, stage):
            if isinstance(gait_scheduler_scales[key], list):
                return gait_scheduler_scales[key][min(stage, len(gait_scheduler_scales[key])-1)]
            else:
                return gait_scheduler_scales[key]

        return self.async_gait_scheduler.reward_dof_align()*get_weight('dof_align', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_dof_nominal_pos()*get_weight('dof_nominal_pos', self.reward_scales_stage) + \
            self.async_gait_scheduler.reward_foot_z_align()*get_weight('reward_foot_z_align', self.reward_scales_stage)

    def _reward_gait_scheduler(self):
        """Reward for tracking the gait scheduler's patterns."""
        # Reward for tracking the gait scheduler
        return self.gait_scheduler.reward_foot_z_track()
