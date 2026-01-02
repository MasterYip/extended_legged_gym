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

"""Task-specific ElSpider variants."""

import numpy as np
import torch
from isaacgym.torch_utils import *
from legged_gym.utils import SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain
from .elspider import ElSpider


class LoadAdaptElSpider(ElSpider):
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


class PoseElSpider(ElSpider):
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


class FootTrackElSpider(ElSpider):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.cfg.rewards.raibert_planner.planner_type == 0:
            cfg = SimpleRaibertPlannerConfig()
            cfg.dt = self.dt
            self.raibert_planner = SimpleRaibertPlanner(self.num_envs, self.device, cfg)
            self.raibert_planner.init(self.base_pos, self.base_quat)
        elif self.cfg.rewards.raibert_planner.planner_type == 1:
            cfg = RaibertPlannerConfig()
            cfg.dt = self.dt
            self.raibert_planner = RaibertPlanner(self.num_envs, self.device, cfg)
            self.raibert_planner.init(self.base_pos, self.base_quat)
        else:
            raise ValueError("Invalid planner type")

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.raibert_planner.get_obs_tensor(self.base_pos, self.base_quat),
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 -
                                 self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def check_termination(self):
        """ Check if environments need to be reset
        """
        super().check_termination()
        self.raibert_pos_diff = torch.norm(self.base_pos - self.raibert_planner.base_pos, dim=1)
        self.reset_buf |= self.raibert_pos_diff > 0.5

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.raibert_planner.reset_idx(self.base_pos, self.base_quat, env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        # Update raiber planner
        self.raibert_planner.step(self.commands[:, :3])

    def _draw_debug_vis(self):
        # draw base vel
        self.gym.clear_lines(self.viewer)
        # lin_vel = self.root_states[:, 7:10].cpu().numpy()
        # z_base = quat_rotate(self.base_quat, self.gravity_vec).cpu().numpy()
        raibert_base_pos = self.raibert_planner.base_pos_shift.cpu().numpy()
        raibert_foot_pos = self.raibert_planner.foot_pos.view(-1, 3).cpu().numpy()
        for i in range(self.num_envs):
            base_quat_shift = self.raibert_planner.base_quat_shift[i].cpu().numpy()
            base_pos_shift = self.raibert_planner.base_pos_shift[i].cpu().numpy()
            self.vis.draw_frame_from_quat(i, base_quat_shift, base_pos_shift, length=0.4)
        self.vis.draw_points(0, raibert_base_pos, color=(1, 0, 0), size=0.03)
        self.vis.draw_points(0, raibert_foot_pos, color=(0, 1, 1), size=0.03)
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

        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_raibert_planner(self):
        # Reward for RaiBert Planner
        reward_scales = class_to_dict(self.cfg.rewards.raibert_planner)

        def get_weight(key, stage):
            if isinstance(reward_scales[key], list):
                return reward_scales[key][min(stage, len(reward_scales[key])-1)]
            else:
                return reward_scales[key]

        return self.raibert_planner.reward_base_pos_track(self.root_states[:, :3])*get_weight('base_pos_track', self.reward_scales_stage) + \
            self.raibert_planner.reward_base_quat_track(self.base_quat)*get_weight('base_quat_track', self.reward_scales_stage) + \
            self.raibert_planner.reward_foot_pos_track(self.foot_positions)*get_weight('foot_pos_track', self.reward_scales_stage)

    # Separate reward functions for base pos, base quat and foot pos
    def _reward_raibert_base_pos_track(self):
        return self.raibert_planner.penalty_base_pos_track(self.root_states[:, :3])

    def _reward_raibert_base_quat_track(self):
        return self.raibert_planner.penalty_base_quat_track(self.base_quat)

    def _reward_raibert_foot_swing_contact(self):
        return self.raibert_planner.penalty_foot_swing_contact(self.contact_forces, self.feet_indices)

    def _reward_raibert_foot_pos_track(self):
        return self.raibert_planner.reward_foot_pos_track(self.foot_positions)

    def _reward_raibert_foot_pos_track_z(self):
        return self.raibert_planner.penalty_foot_pos_track_z(self.foot_positions)


class StandElSpider(ElSpider):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
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
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_foot_contact = contact_filt[:, 0]
        second_foot_contact = contact_filt[:, 1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward
