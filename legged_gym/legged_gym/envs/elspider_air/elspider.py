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
from .mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler, \
    SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw

@torch.no_grad()
def get_symmetric_observation_action(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply symmetry transformation to observations and actions for the ElSpider robot.
    
    This function augments the dataset by mirroring the robot's left-right sides.
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for reference)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    # Original and mirrored observations/actions
    # [batch*2, dim] where first batch is original, second batch is mirrored
    
    if obs is not None:
        # Mirror the observations for ElSpider which has 6 legs
        # For policy observation, the structure is:
        # [0:3] - base_lin_vel (mirror y)
        # [3:6] - base_ang_vel (mirror x, z)
        # [6:9] - projected_gravity (mirror y)
        # [9:12] - commands (mirror y for lin_vel, mirror ang_vel_z)
        # [12:30] - dof_pos (swap left-right sides)
        # [30:48] - dof_vel (swap left-right sides)
        # [48:66] - previous actions (swap left-right sides)
        # [66:253] - height measurements (mirror left-right pattern)
        
        # Create mirrored observations
        obs_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_mirrored[:, 3] = -obs[:, 3]
        obs_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_mirrored[:, 10] = -obs[:, 10]
        obs_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions - ElSpider has 6 legs with 3 DOFs each
        # Right side DOFs: 0-8, Left side DOFs: 9-17
        # HAA joints need to be negated when swapped, HFE and KFE can be directly swapped
        
        # Map for mirroring DOF positions (12:30)
        # RF to LF, RM to LM, RB to LB
        # HAA joints need sign flip, HFE and KFE don't
        
        # Right to Left mapping (index of right-side DOF → index of left-side DOF)
        # RF_HAA(0) → LF_HAA(9)
        # RF_HFE(1) → LF_HFE(10)
        # RF_KFE(2) → LF_KFE(11)
        # RM_HAA(3) → LM_HAA(12)
        # RM_HFE(4) → LM_HFE(13)
        # RM_KFE(5) → LM_KFE(14)
        # RB_HAA(6) → LB_HAA(15)
        # RB_HFE(7) → LB_HFE(16)
        # RB_KFE(8) → LB_KFE(17)
        
        # Swap right and left DOF positions
        for i in range(3):  # Three leg pairs (front, middle, back)
            for j in range(3):  # Three joints per leg (HAA, HFE, KFE)
                right_idx = 12 + i*3 + j  # DOF position indices start at 12
                left_idx = 12 + (i+3)*3 + j  # Left legs are offset by 3 legs
                
                # Store right value temporarily
                temp = obs_mirrored[:, right_idx].clone()
                
                # For HAA joints (j=0), negate the values when swapping
                if j == 0:  # HAA joint
                    obs_mirrored[:, right_idx] = -obs[:, left_idx]
                    obs_mirrored[:, left_idx] = -obs[:, right_idx]
                else:  # HFE, KFE joints - direct swap without negation
                    obs_mirrored[:, right_idx] = obs[:, left_idx]
                    obs_mirrored[:, left_idx] = obs[:, right_idx]
        
        # Mirror DOF velocities (30:48) using the same mapping as positions
        for i in range(3):  # Three leg pairs
            for j in range(3):  # Three joints per leg
                right_idx = 30 + i*3 + j  # DOF velocity indices start at 30
                left_idx = 30 + (i+3)*3 + j
                
                # Store right value temporarily
                temp = obs_mirrored[:, right_idx].clone()
                
                # For HAA joints, negate the values when swapping
                if j == 0:  # HAA joint
                    obs_mirrored[:, right_idx] = -obs[:, left_idx]
                    obs_mirrored[:, left_idx] = -obs[:, right_idx]
                else:  # HFE, KFE joints - direct swap
                    obs_mirrored[:, right_idx] = obs[:, left_idx]
                    obs_mirrored[:, left_idx] = obs[:, right_idx]
        
        # Mirror previous actions (48:66) using the same mapping
        for i in range(3):  # Three leg pairs
            for j in range(3):  # Three joints per leg
                right_idx = 48 + i*3 + j  # Action indices start at 48
                left_idx = 48 + (i+3)*3 + j
                
                # Store right value temporarily
                temp = obs_mirrored[:, right_idx].clone()
                
                # For HAA joints, negate the values when swapping
                if j == 0:  # HAA joint
                    obs_mirrored[:, right_idx] = -obs[:, left_idx]
                    obs_mirrored[:, left_idx] = -obs[:, right_idx]
                else:  # HFE, KFE joints - direct swap
                    obs_mirrored[:, right_idx] = obs[:, left_idx]
                    obs_mirrored[:, left_idx] = obs[:, right_idx]
        
        # Mirror height measurements (66:253) if present
        if obs.shape[1] > 66:
            # The height measurements are in a grid pattern
            # Original grid pattern: measured_points_x × measured_points_y
            # For ElSpider, this is typically 17×11 = 187 points
            
            # We need to mirror the points along the y-axis
            # If we have 17 points in x and 11 in y, the indices form a 17×11 grid
            
            height_measurements_start = 66
            x_points = 17  # Number of points along x-axis (from config)
            y_points = 11  # Number of points along y-axis (from config)
            
            for x in range(x_points):
                for y in range(y_points):
                    # Calculate original and mirrored indices
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_y = y_points - y - 1  # Flip y coordinate
                    mirrored_idx = height_measurements_start + x*y_points + mirrored_y
                    
                    # Swap the height measurements
                    obs_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # Combine original and mirrored observations
        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # Mirror the actions
        # ElSpider has 18 actions (6 legs × 3 joints)
        # Right legs: 0-8, Left legs: 9-17
        actions_mirrored = actions.clone()
        
        # Apply the same logic as DOF positions
        for i in range(3):  # Three leg pairs
            for j in range(1):  # Three joints per leg
                right_idx = i*3 + j
                left_idx = (i+3)*3 + j
                
                # For HAA joints, negate the values when swapping
                if j == 0:  # HAA joint
                    actions_mirrored[:, right_idx] = -actions[:, left_idx]
                    actions_mirrored[:, left_idx] = -actions[:, right_idx]
                # else:  # HFE, KFE joints - direct swap
                #     actions_mirrored[:, right_idx] = actions[:, left_idx]
                #     actions_mirrored[:, left_idx] = actions[:, right_idx]
        
        # Combine original and mirrored actions
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0) if actions is not None else None
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented

class ElSpider(LeggedRobot):
    cfg: ElSpiderAirRoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

        # Init gait scheduler
        cfg = GaitSchedulerCfg()
        cfg.dt = self.dt
        cfg.period = 1.4
        cfg.swing_height = 0.07
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

        cfg = AsyncGaitSchedulerCfg()
        self.async_gait_scheduler = AsyncGaitScheduler(self.height_samples,
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

    def _draw_debug_vis(self):
        # draw base vel
        self.gym.clear_lines(self.viewer)
        lin_vel = self.root_states[:, 7:10].cpu().numpy()
        cmd_vel_world = quat_apply_yaw(self.base_quat, self.commands[:, :3]).cpu().numpy()
        cmd_vel_world[:, 2] = 0.0
        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            self.vis.draw_arrow(i, base_pos, base_pos + lin_vel[i], color=(0, 1, 0))
            self.vis.draw_arrow(i, base_pos, base_pos + cmd_vel_world[i], color=(1, 0, 0))
        return super()._draw_debug_vis()

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
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[66:253] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def post_physics_step(self):
        super().post_physics_step()
        # Update gait scheduler
        self.gait_scheduler.step(self.foot_positions, self.foot_velocities, self.commands)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        super().check_termination()
        
        # Add new termination condition - terminate if robot is upside down (z-component of projected gravity > 0)
        self.reset_buf |= (self.projected_gravity[:, 2] > 0)

    def _reward_gait_scheduler(self):
        # Reward for tracking the gait scheduler
        return self.gait_scheduler.reward_foot_z_track()

    def _reward_async_gait_scheduler(self):
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

    def _reward_gait_2_step(self):
        # Foot index (alphabet): 0 LB, 1 LF, 2 LM, 3 RB, 4 RF, 5 RM
        # Hexapod 2-step gait: first group (0-1-5) synchronized, second group (2-3-4) synchronized
        # The two groups are asynchronized with each other
        
        # First group internal synchronization rewards (0-1-5)
        sync_lb_lf = self._sync_reward_func(0, 1)
        sync_lb_rm = self._sync_reward_func(0, 5)
        sync_lf_rm = self._sync_reward_func(1, 5)
        sync_group1 = (sync_lb_lf + sync_lb_rm + sync_lf_rm) / 3
        
        # Second group internal synchronization rewards (2-3-4)
        sync_lm_rb = self._sync_reward_func(2, 3)
        sync_lm_rf = self._sync_reward_func(2, 4)
        sync_rb_rf = self._sync_reward_func(3, 4)
        sync_group2 = (sync_lm_rb + sync_lm_rf + sync_rb_rf) / 3
        
        # Asynchronization rewards between the two groups
        async_lb_lm = self._async_reward_func(0, 2)
        async_lb_rb = self._async_reward_func(0, 3)
        async_lb_rf = self._async_reward_func(0, 4)
        async_lf_lm = self._async_reward_func(1, 2)
        async_lf_rb = self._async_reward_func(1, 3)
        async_lf_rf = self._async_reward_func(1, 4)
        async_rm_lm = self._async_reward_func(5, 2)
        async_rm_rb = self._async_reward_func(5, 3)
        async_rm_rf = self._async_reward_func(5, 4)
        
        # Calculate average asynchronization reward
        async_reward = (async_lb_lm + async_lb_rb + async_lb_rf + 
                         async_lf_lm + async_lf_rb + async_lf_rf + 
                         async_rm_lm + async_rm_rb + async_rm_rf) / 9
        
        # Calculate total synchronization reward
        sync_reward = (sync_group1 + sync_group2) / 2
        
        re = sync_reward + async_reward
        if self.cfg.commands.heading_command:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 3]) >= self.speed_min/ 2)
        else:
            re = re * torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                    torch.abs(self.commands[:, 2]) >= self.speed_min/ 2)
        return re
    

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
