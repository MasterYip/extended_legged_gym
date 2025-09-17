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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg


class Anymal(LeggedRobot):
    cfg: AnymalCRoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

        # Init gait scheduler
        cfg = GaitSchedulerCfg()
        cfg.dt = self.dt
        cfg.period = 0.6
        cfg.foot_phases = [0.0, 0.5, 0.5, 0.0]
        cfg.swing_height = 0.15
        self.gait_scheduler = GaitScheduler(self.height_samples,
                                            self.base_quat,
                                            self.base_lin_vel,
                                            self.base_ang_vel,
                                            self.projected_gravity,
                                            self.dof_pos,
                                            self.dof_vel,
                                            self.foot_positions,
                                            self.foot_velocities,
                                            self.num_envs,
                                            self.device,
                                            cfg)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale +
                                           self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                    self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)

    def post_physics_step(self):
        super().post_physics_step()
        # Update gait scheduler
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands)

    def _reward_gait_scheduler(self):
        # Reward for tracking the gait scheduler
        return self.gait_scheduler.reward_foot_z_track()


class LoadAdaptAnymal(Anymal):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _draw_debug_vis(self):
        # draw base vel
        self.gym.clear_lines(self.viewer)
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        lin_acc = quat_rotate(self.base_quat[:], self.base_lin_acc).cpu().numpy()
        z_base = quat_rotate(self.base_quat[:], self.gravity_vec).cpu().numpy()
        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            acc_tot = lin_acc[i] + np.array([0, 0, 9.8])
            self.vis.draw_arrow(i, base_pos, base_pos + acc_tot/np.linalg.norm(acc_tot)*2, color=(1, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos - z_base[i]/np.linalg.norm(z_base[i])*2, color=(0, 1, 1))
        return super()._draw_debug_vis()

    # Rewards
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Stable orientation reward
        # Penalize base orientation perpendicular to acc+gravity
        return torch.sum(torch.square(self.projected_gravity[:, :2] - self.base_lin_acc[:, :2]/9.81), dim=1)

    # def _reward_orientation(self):
    #     # Velocity orientation reward
    #     return torch.sum(torch.square(self.projected_gravity[:, :2] - self.base_lin_vel[:, :2]*0.6), dim=1)


class PoseAnymal(Anymal):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  self.commands[:, 4:],  # TODO: add scales for pose commands
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 -
                                 self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _init_buffers(self):
        super()._init_buffers()
        # Additional buffers for pose commands
        self.exp_quat = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

    def _draw_debug_vis(self):
        # draw base vel
        self.gym.clear_lines(self.viewer)
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        z_base_exp = quat_rotate(self.exp_quat, self.gravity_vec).cpu().numpy()
        z_base = quat_rotate(self.base_quat, self.gravity_vec).cpu().numpy()
        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            base_pos[2] = self.commands[i, 7]  # Expected base height
            self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos - z_base_exp[i]/np.linalg.norm(z_base_exp[i]), color=(1, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos - z_base[i]/np.linalg.norm(z_base[i]), color=(0, 1, 1))
        return super()._draw_debug_vis()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # Pose commands
        self.commands[env_ids, 4] = torch_rand_float(
            self.command_ranges["base_yaw_shift"][0], self.command_ranges["base_yaw_shift"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 5] = torch_rand_float(
            self.command_ranges["base_pitch_shift"][0], self.command_ranges["base_pitch_shift"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 6] = torch_rand_float(
            self.command_ranges["base_roll_shift"][0], self.command_ranges["base_roll_shift"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 7] = torch_rand_float(
            self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        pitch_shift = self.commands[:, 5]
        roll_shift = self.commands[:, 6]
        cos_pitch_2 = torch.cos(pitch_shift/2)
        sin_pitch_2 = torch.sin(pitch_shift/2)
        cos_roll_2 = torch.cos(roll_shift/2)
        sin_roll_2 = torch.sin(roll_shift/2)
        quat_pitch = torch.stack([torch.zeros_like(sin_pitch_2), -sin_pitch_2,
                                  torch.zeros_like(sin_pitch_2), cos_pitch_2], dim=-1)

        quat_roll = torch.stack([sin_roll_2, torch.zeros_like(sin_pitch_2),
                                 torch.zeros_like(sin_pitch_2), cos_roll_2], dim=-1)

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.quat_heading = torch.stack([torch.zeros_like(heading), torch.zeros_like(heading),
                                         torch.sin(heading/2), torch.cos(heading/2)], dim=-1)

        self.exp_quat = quat_mul(self.quat_heading, quat_mul(quat_pitch, quat_roll))

    def _reward_orientation(self):
        expect_projected_gravity = quat_rotate_inverse(self.exp_quat, self.gravity_vec).squeeze(-1)
        gravity_diff = expect_projected_gravity - self.projected_gravity
        return torch.sum(torch.square(gravity_diff[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.commands[:, 7])


class StandAnymal(Anymal):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Buf reshape 2
        self.last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

    # Rewards

    def _reward_ang_vel_xy(self):
        # Penalize yz axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, 1:]), dim=1)

    def _reward_orientation(self):
        # Penalize base orientation
        # Projected Gravity should align with -x
        return torch.sum(torch.square(self.projected_gravity[:, 1:]), dim=1)

    def _reward_standing(self):
        # Reward for standing
        return torch.sum(torch.square(self.base_lin_acc[:, 2] - 9.81), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (yz axes in base frame)
        # TODO: check
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] + self.base_lin_vel[:, 1:]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        hind_feet_indices = [self.feet_indices[1], self.feet_indices[3]]
        contact = self.contact_forces[:, hind_feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_penalty_in_the_air(self):
        hind_feet_indices = [self.feet_indices[1], self.feet_indices[3]]
        contact = self.contact_forces[:, hind_feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_foot_contact = contact_filt[:, 0]
        second_foot_contact = contact_filt[:, 1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward


class AnymalStudent(Anymal):
    """AnymalStudent class for distillation training with observation history."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Set history length from config
        self.history_length = getattr(cfg.env, 'history_length', 5)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # Initialize observation history buffer
        # Student obs: 48 (proprio) * history_length
        self.proprio_obs_size = 48
        self.obs_history = torch.zeros(
            self.num_envs, self.history_length, self.proprio_obs_size,
            device=self.device, dtype=torch.float, requires_grad=False
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Clear observation history for reset environments
        self.obs_history[env_ids] = 0.

    def compute_observations(self):
        """Compute observations for student (history) and privileged observations for teacher."""
        # Compute current proprioceptive observations (48 dim)
        current_proprio = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,                    # 3
            self.base_ang_vel * self.obs_scales.ang_vel,                    # 3
            self.projected_gravity,                                         # 3
            self.commands[:, :3] * self.commands_scale,                     # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,                         # 12
            self.actions                                                    # 12
        ), dim=-1)  # Total: 48 dims

        # Update observation history (shift and add new observation)
        self.obs_history = torch.roll(self.obs_history, shifts=1, dims=1)
        self.obs_history[:, 0] = current_proprio

        # Student observations: flattened history (48 * history_length)
        self.obs_buf = self.obs_history.view(self.num_envs, -1)

        # Privileged observations for teacher: current proprio + height measurements (48 + 187 = 235)
        privileged_obs = current_proprio.clone()

        # Add height measurements if enabled
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            privileged_obs = torch.cat((privileged_obs, heights), dim=-1)

        # Store privileged observations
        if hasattr(self, 'privileged_obs_buf'):
            self.privileged_obs_buf = privileged_obs

        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:self.obs_buf.shape[1]]
            # if hasattr(self, 'privileged_obs_buf'):
            #     # Create noise for privileged obs with same scale as original obs
            #     priv_noise_scale = torch.cat([
            #         self.noise_scale_vec[:48],  # proprio noise
            #         self.noise_scale_vec[48:] if self.noise_scale_vec.shape[0] > 48 else torch.zeros(
            #             187, device=self.device)  # height noise
            #     ])
            #     self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * priv_noise_scale

    def get_observations(self):
        """Return observations for student (new interface)."""
        return self.obs_buf

    def get_privileged_observations(self):
        """Return privileged observations for teacher (old interface)."""
        if hasattr(self, 'privileged_obs_buf'):
            return self.privileged_obs_buf
        return None
