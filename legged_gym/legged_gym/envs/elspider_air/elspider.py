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
def get_elair_xysym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply both left-right and back-forth symmetry transformations for the ElSpider robot.
    
    This function augments the dataset by mirroring the robot's left-right sides and front-back directions.
    Returns [batch*3, dim] where first batch is original, second batch is left-right mirrored, 
    third batch is back-forth mirrored.
    
    Foot index order: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
    Robot heading: x-axis forward, y-axis left, -y-axis right
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for reference)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors [batch*3, dim]
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]

    if obs_type not in ("policy", "critic"):
        raise ValueError(
            f"get_elair_xysym_obs_act: obs_type must be 'policy' or 'critic', "
            f"got '{obs_type}'"
        )

    if obs is not None:
        # --- Left-Right Mirrored Observations ---
        obs_lr_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_lr_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_lr_mirrored[:, 3] = -obs[:, 3]
        obs_lr_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_lr_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_lr_mirrored[:, 10] = -obs[:, 10]
        obs_lr_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions: L(0-8) <-> R(9-17)
        # LB(12:15) LF(15:18) LM(18:21) <-> RB(21:24) RF(24:27) RM(27:30)
        obs_lr_mirrored[:, 12:21] = obs[:, 21:30]  # Left legs get right leg positions
        obs_lr_mirrored[:, 21:30] = obs[:, 12:21]  # Right legs get left leg positions
        
        # Mirror DOF velocities (30:48)
        obs_lr_mirrored[:, 30:39] = obs[:, 39:48]
        obs_lr_mirrored[:, 39:48] = obs[:, 30:39]
        
        # Mirror previous actions (48:66)
        obs_lr_mirrored[:, 48:57] = obs[:, 57:66]
        obs_lr_mirrored[:, 57:66] = obs[:, 48:57]
        
        # Mirror height measurements (66:253) along y-axis
        if obs.shape[1] > 66:
            height_measurements_start = 66
            x_points = 17
            y_points = 11
            
            for x in range(x_points):
                for y in range(y_points):
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_y = y_points - y - 1
                    mirrored_idx = height_measurements_start + x*y_points + mirrored_y
                    obs_lr_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # --- Back-Forth Mirrored Observations ---
        obs_bf_mirrored = obs.clone()
        
        # Mirror linear velocity x-component
        obs_bf_mirrored[:, 0] = -obs[:, 0]
        
        # Mirror angular velocity y and z components
        obs_bf_mirrored[:, 4] = -obs[:, 4]
        obs_bf_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity x-component
        obs_bf_mirrored[:, 6] = -obs[:, 6]
        
        # Mirror command velocities (x and angular z)
        obs_bf_mirrored[:, 9] = -obs[:, 9]
        obs_bf_mirrored[:, 11] = -obs[:, 11]
        
        # Swap back-front DOF positions: Back(LB,RB) <-> Front(LF,RF)
        # LB(12:15) <-> LF(15:18), RB(21:24) <-> RF(24:27), LM stays as is
        obs_bf_mirrored[:, 12:15] = obs[:, 15:18]  # LB gets LF positions
        obs_bf_mirrored[:, 15:18] = obs[:, 12:15]  # LF gets LB positions
        obs_bf_mirrored[:, 21:24] = obs[:, 24:27]  # RB gets RF positions
        obs_bf_mirrored[:, 24:27] = obs[:, 21:24]  # RF gets RB positions
        
        # Mirror DOF velocities (30:48) - swap back-front
        obs_bf_mirrored[:, 30:33] = obs[:, 33:36]  # LB gets LF velocities
        obs_bf_mirrored[:, 33:36] = obs[:, 30:33]  # LF gets LB velocities
        obs_bf_mirrored[:, 39:42] = obs[:, 42:45]  # RB gets RF velocities
        obs_bf_mirrored[:, 42:45] = obs[:, 39:42]  # RF gets RB velocities
        
        # Mirror previous actions (48:66) - swap back-front
        obs_bf_mirrored[:, 48:51] = obs[:, 51:54]  # LB gets LF actions
        obs_bf_mirrored[:, 51:54] = obs[:, 48:51]  # LF gets LB actions
        obs_bf_mirrored[:, 57:60] = obs[:, 60:63]  # RB gets RF actions
        obs_bf_mirrored[:, 60:63] = obs[:, 57:60]  # RF gets RB actions
        
        # Mirror height measurements (66:253) along x-axis
        if obs.shape[1] > 66:
            height_measurements_start = 66
            x_points = 17
            y_points = 11
            
            for x in range(x_points):
                for y in range(y_points):
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_x = x_points - x - 1
                    mirrored_idx = height_measurements_start + mirrored_x*y_points + y
                    obs_bf_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # Combine original, left-right mirrored, and back-forth mirrored observations
        obs_augmented = torch.cat([obs, obs_lr_mirrored, obs_bf_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # --- Left-Right Mirrored Actions ---
        # Foot index: LB(0-2), LF(3-5), LM(6-8), RB(9-11), RF(12-14), RM(15-17)
        actions_lr_mirrored = actions.clone()
        
        # Swap left and right legs
        actions_lr_mirrored[:, 0:9] = actions[:, 9:18]   # Left legs get right leg actions
        actions_lr_mirrored[:, 9:18] = actions[:, 0:9]   # Right legs get left leg actions
        
        # --- Back-Forth Mirrored Actions ---
        actions_bf_mirrored = actions.clone()
        
        # Swap back and front legs: LB<->LF, RB<->RF, LM stays as is
        actions_bf_mirrored[:, 0:3] = actions[:, 3:6]    # LB gets LF actions
        actions_bf_mirrored[:, 3:6] = actions[:, 0:3]    # LF gets LB actions
        actions_bf_mirrored[:, 9:12] = actions[:, 12:15] # RB gets RF actions
        actions_bf_mirrored[:, 12:15] = actions[:, 9:12] # RF gets RB actions
        
        # Combine original, left-right mirrored, and back-forth mirrored actions
        actions_augmented = torch.cat([actions, actions_lr_mirrored, actions_bf_mirrored], dim=0)
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented


@torch.no_grad()
def get_elair_xsym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
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

    if obs_type not in ("policy", "critic"):
        raise ValueError(
            f"get_elair_xsym_obs_act: obs_type must be 'policy' or 'critic', "
            f"got '{obs_type}'"
        )

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

        # Swap right and left DOF positions
        obs_mirrored[:, 12:21] = obs[:, 21:30]  # Right legs get left leg positions
        obs_mirrored[:, 21:30] = obs[:, 12:21]  # Left legs get right leg positions
        
        # Mirror DOF velocities (30:48) using the same mapping as positions
        obs_mirrored[:, 30:39] = obs[:, 39:48]  # Right legs get left leg velocities
        obs_mirrored[:, 39:48] = obs[:, 30:39]  # Left legs get right leg velocities
        
        # Mirror previous actions (48:66) using the same mapping
        obs_mirrored[:, 48:57] = obs[:, 57:66]  # Right legs get left leg actions
        obs_mirrored[:, 57:66] = obs[:, 48:57]  # Left legs get right leg actions
        
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
        
        actions_mirrored[:, 0:9] = actions[:, 9:18]  # Right legs get left leg actions
        actions_mirrored[:, 9:18] = actions[:, 0:9]  # Left legs get right leg actions
        
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
        # Make sure shanks are perpendicular to the ground
        cfg.dof_align_sets = [['RF_HFE', 'RF_KFE'],
                    ['RM_HFE', 'RM_KFE'],
                    ['RB_HFE', 'RB_KFE'],
                    ['LF_HFE', 'LF_KFE'],
                    ['LM_HFE', 'LM_KFE'],
                    ['LB_HFE', 'LB_KFE'],]
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

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length * 0.6
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    # Rewards
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
        
        # Calculate sync all legs reward for small commands (standing still)
        sync_all_pairs = [
            sync_lb_lf, sync_lb_rm, sync_lf_rm,  # Group 1 internal
            sync_lm_rb, sync_lm_rf, sync_rb_rf,  # Group 2 internal
            self._sync_reward_func(0, 2), self._sync_reward_func(0, 3), self._sync_reward_func(0, 4),  # LB with group 2
            self._sync_reward_func(1, 2), self._sync_reward_func(1, 3), self._sync_reward_func(1, 4),  # LF with group 2
            self._sync_reward_func(5, 2), self._sync_reward_func(5, 3), self._sync_reward_func(5, 4)   # RM with group 2
        ]
        sync_all_reward = sum(sync_all_pairs) / len(sync_all_pairs)
        
        # Determine command magnitude
        command_magnitude = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                            torch.abs(self.commands[:, 2]) > self.speed_min)
        
        # Use gait reward for large commands, sync all legs reward for small commands
        re = torch.where(command_magnitude, 
                        sync_reward + async_reward,  # 2-step gait for movement
                        sync_all_reward)             # sync all legs for standing still
        
        return re
    
    def _reward_gait_3_step(self):
        # Foot index (alphabet): 0 LB, 1 LF, 2 LM, 3 RB, 4 RF, 5 RM
        # Hexapod 3-step gait: first group (1-4) synchronized, second group (2-5) synchronized, third group (0-3) synchronized
        # The three groups are asynchronized with each other
        
        # First group internal synchronization rewards (1-4): LF, RF
        sync_group1 = self._sync_reward_func(1, 4)
        
        # Second group internal synchronization rewards (2-5): LM, RM
        sync_group2 = self._sync_reward_func(2, 5)
        
        # Third group internal synchronization rewards (0-3): LB, RB
        sync_group3 = self._sync_reward_func(0, 3)
        
        # Asynchronization rewards between group 1 and group 2
        async_lf_lm = self._async_reward_func(1, 2)
        async_lf_rm = self._async_reward_func(1, 5)
        async_rf_lm = self._async_reward_func(4, 2)
        async_rf_rm = self._async_reward_func(4, 5)
        async_group1_group2 = (async_lf_lm + async_lf_rm + async_rf_lm + async_rf_rm) / 4
        
        # Asynchronization rewards between group 1 and group 3
        async_lf_lb = self._async_reward_func(1, 0)
        async_lf_rb = self._async_reward_func(1, 3)
        async_rf_lb = self._async_reward_func(4, 0)
        async_rf_rb = self._async_reward_func(4, 3)
        async_group1_group3 = (async_lf_lb + async_lf_rb + async_rf_lb + async_rf_rb) / 4
        
        # Asynchronization rewards between group 2 and group 3
        async_lm_lb = self._async_reward_func(2, 0)
        async_lm_rb = self._async_reward_func(2, 3)
        async_rm_lb = self._async_reward_func(5, 0)
        async_rm_rb = self._async_reward_func(5, 3)
        async_group2_group3 = (async_lm_lb + async_lm_rb + async_rm_lb + async_rm_rb) / 4
        
        # Calculate average asynchronization reward across all group pairs
        async_reward = (async_group1_group2 + async_group1_group3 + async_group2_group3) / 3
        async_reward *= 0.0 # 3-step gait does not require strong asynchronization

        # Calculate total synchronization reward
        sync_reward = (sync_group1 + sync_group2 + sync_group3) / 3
        
        # Calculate sync all legs reward for small commands (standing still)
        sync_all_pairs = [
            sync_group1, sync_group2, sync_group3,  # Within-group sync
            self._sync_reward_func(1, 2), self._sync_reward_func(1, 5),  # Group 1 with group 2
            self._sync_reward_func(4, 2), self._sync_reward_func(4, 5),
            self._sync_reward_func(1, 0), self._sync_reward_func(1, 3),  # Group 1 with group 3
            self._sync_reward_func(4, 0), self._sync_reward_func(4, 3),
            self._sync_reward_func(2, 0), self._sync_reward_func(2, 3),  # Group 2 with group 3
            self._sync_reward_func(5, 0), self._sync_reward_func(5, 3)
        ]
        sync_all_reward = sum(sync_all_pairs) / len(sync_all_pairs)
        
        # Determine command magnitude
        command_magnitude = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.speed_min, 
                                            torch.abs(self.commands[:, 2]) > self.speed_min)
        
        # Use gait reward for large commands, sync all legs reward for small commands
        re = torch.where(command_magnitude, 
                        sync_reward + async_reward,  # 3-step gait for movement
                        sync_all_reward)             # sync all legs for standing still
        
        return re
    
    def _reward_shank_perp2ground(self):
        if not hasattr(self, 'hfe_indices'):
            self.hfe_indices = [self.dof_names.index(name) for name in [
                'RF_HFE', 'RM_HFE', 'RB_HFE', 'LF_HFE', 'LM_HFE', 'LB_HFE']]
        if not hasattr(self, 'kfe_indices'):
            self.kfe_indices = [self.dof_names.index(name) for name in [
                'RF_KFE', 'RM_KFE', 'RB_KFE', 'LF_KFE', 'LM_KFE', 'LB_KFE']]
        return torch.square(self.dof_pos[:, self.hfe_indices] - self.dof_pos[:, self.kfe_indices]).sum(dim=1)

    def _reward_haa_nominal_pos(self):
        if not hasattr(self, 'haa_indices'):
            self.haa_indices = [self.dof_names.index(name) for name in [
                'RF_HAA', 'RM_HAA', 'RB_HAA', 'LF_HAA', 'LM_HAA', 'LB_HAA']]
        haa_nominal_pos = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        # use ema to smooth the reward
        ema = 0.01
        self.haa_nominal_pos_ema = getattr(self, 'haa_nominal_pos_ema', torch.zeros(self.num_envs, device=self.device))
        current_deviation = torch.square(self.dof_pos[:, self.haa_indices] - haa_nominal_pos).sum(dim=1)
        self.haa_nominal_pos_ema = ema * current_deviation + (1 - ema) * self.haa_nominal_pos_ema
        return self.haa_nominal_pos_ema

class ElSpiderStudent(ElSpider):
    """ElSpiderStudent class for distillation training with observation history."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Set history length from config
        self.history_length = getattr(cfg.env, 'history_length', 3)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # Initialize observation history buffer
        # Student obs: 66 (proprio) * history_length
        self.proprio_obs_size = 66
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
        # Compute current proprioceptive observations (66 dim)
        current_proprio = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,                    # 3
            self.base_ang_vel * self.obs_scales.ang_vel,                    # 3
            self.projected_gravity,                                         # 3
            self.commands[:, :3] * self.commands_scale,                     # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 18
            self.dof_vel * self.obs_scales.dof_vel,                         # 18
            self.actions                                                    # 18
        ), dim=-1)  # Total: 66 dims

        # Update observation history (shift and add new observation)
        self.obs_history = torch.roll(self.obs_history, shifts=1, dims=1)
        self.obs_history[:, 0] = current_proprio

        # Student observations: flattened history (66 * history_length)
        self.obs_buf = self.obs_history.view(self.num_envs, -1)

        # Privileged observations for teacher: current proprio + height measurements (66 + 187 = 253)
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
