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
import math
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.elspider_air.elspider import ElSpider

from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg


class EL_4090(ElSpider):
    cfg : El4090SpiderCfg
     # env Init
    def __init__(self, cfg: El4090SpiderCfg, sim_params, physics_engine, sim_device, headless,task_name="el4090_spider"):
        """ Parses the provided config file,Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.action_des = 0
        # self.action_flag = 1

        # 初始化 group1_contact_time 和 group2_contact_time
        self.group1_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.group2_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.T = 0.5
        
        # Debug counters
        self.debug_step_counter = 0

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # 定义两组三角步态的脚部索引
        # 组1: LF (1), LB (0), RM (5)
        # 组2: RF (4), RB (3), LM (2)
        self.tripod_group1_indices = torch.tensor([1, 0, 5], device=self.device, dtype=torch.long)
        self.tripod_group2_indices = torch.tensor([4, 3, 2], device=self.device, dtype=torch.long)
        
        # 初始化小腿索引 (SHANK)
        # Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
        # 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
        # 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']
        # 小腿顺序: LB_SHANK(3), LF_SHANK(7), LM_SHANK(11), RB_SHANK(15), RF_SHANK(19), RM_SHANK(23)
        shank_names = ['LB_SHANK', 'LF_SHANK', 'LM_SHANK', 'RB_SHANK', 'RF_SHANK', 'RM_SHANK']
        self.shank_indices = torch.zeros(len(shank_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shank_names)):
            self.shank_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], shank_names[i])


    def _compute_torques(self, actions):
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type == "V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains * \
                (self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        elif control_type == "DELAY":
            # First-order lag actuator model (no FIFO/discrete pure delay):
            #   y_dot = (u - y) / tau
            # with exact discrete update:
            #   y[k] = y[k-1] + alpha * (u[k] - y[k-1]), alpha = 1 - exp(-dt/tau)
            tau = 0.015  # seconds
            dt = self.sim_params.dt

            if not hasattr(self, "_delay_action_state"):
                self._delay_action_state = actions_scaled.clone()

            alpha = 1.0 - math.exp(-dt / max(tau, 1e-8))
            self._delay_action_state += alpha * (actions_scaled - self._delay_action_state)

            # Prevent cross-episode leakage for reset envs.
            if hasattr(self, "reset_buf"):
                reset_mask = self.reset_buf.bool()
                if torch.any(reset_mask):
                    self._delay_action_state[reset_mask] = actions_scaled[reset_mask]

            delayed_actions = self._delay_action_state

            # Use lagged target action in the same P-controller form as "P" mode.
            torques = self.p_gains*(delayed_actions + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _debug_info(self):
        """Print debug information for the specified environment(s)"""
        if not self.cfg.env.debug_mode:
            return
        
        self.debug_step_counter += 1
        if self.debug_step_counter % self.cfg.env.debug_interval != 0:
            return
        
        env_id = self.cfg.env.debug_env_id
        
        # Determine which environments to print
        if env_id == -1:
            # Print all environments
            env_ids_to_print = list(range(self.num_envs))
        else:
            # Print specified environment
            if env_id >= self.num_envs:
                return
            env_ids_to_print = [env_id]
        
        print("\n" + "="*80)
        print(f"DEBUG INFO - Step {self.common_step_counter} | Printing {len(env_ids_to_print)} environment(s)")
        print("="*80)
        
        for env_idx in env_ids_to_print:
            print(f"\n{'─'*80}")
            print(f"Environment {env_idx} | Episode Length: {self.episode_length_buf[env_idx].item()}")
            print(f"{'─'*80}")
            
            # Base state
            print(f"\n[Base State]")
            print(f"  Position:     [{self.base_pos[env_idx, 0]:.3f}, {self.base_pos[env_idx, 1]:.3f}, {self.base_pos[env_idx, 2]:.3f}]")
            print(f"  Base Height:  {self.base_pos[env_idx, 2]:.3f} m")

            # Contact info and forces
            contact = self.contact_forces[env_idx, self.feet_indices, 2] > 1.
            contact_forces = self.contact_forces[env_idx, self.feet_indices, :]
            contact_forces_z = contact_forces[:, 2]
            max_contact_force = torch.max(contact_forces_z).item()
            max_force_idx = torch.argmax(contact_forces_z).item()
            
            # Foot names for better readability
            foot_names = ['LB', 'LF', 'LM', 'RB', 'RF', 'RM']
            
            print(f"\n[Contact Info]")
            print(f"  Feet Contact: {contact.cpu().numpy()}")
            print(f"  Contact Forces (X,Y,Z) [N]:")
            for i, name in enumerate(foot_names):
                fx = contact_forces[i, 0].item()
                fy = contact_forces[i, 1].item()
                fz = contact_forces[i, 2].item()
                f_total = torch.norm(contact_forces[i]).item()
                contact_str = "✓" if contact[i].item() else "✗"
                print(f"    {name}: [{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] (Total: {f_total:7.2f}) {contact_str}")
            print(f"  Max Contact Force: {max_contact_force:.2f} N (Foot {foot_names[max_force_idx]})")
            print(f"  Total Ground Force: {torch.sum(contact_forces_z).item():.2f} N")
            print(f"  Feet Air Time: {self.feet_air_time[env_idx].cpu().numpy()}")
            
        
        print("\n" + "="*80 + "\n")
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt
        return rew_airTime
 
    def _sync_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize desynchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        return se_air + se_contact
    
    def _async_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize synchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        return se_act_0 + se_act_1

    def _reward_feet_sync(self):
        """
        Penalize desynchronization within each tripod group by summing all pair-wise sync errors.
        Group 1: LF (1), LB (0), RM (5)
        Group 2: RF (4), RB (3), LM (2)
        """
        # Pairs in Group 1
        sync_g1 = self._sync_reward_func(1, 0) + self._sync_reward_func(1, 5) + self._sync_reward_func(0, 5)
        
        # Pairs in Group 2
        sync_g2 = self._sync_reward_func(4, 3) + self._sync_reward_func(4, 2) + self._sync_reward_func(3, 2)

        # Total sync penalty is the sum of penalties from both groups
        sync_reward = sync_g1 + sync_g2
        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)
        
        return sync_reward * move_condition

    def _reward_feet_async(self):
        """
        Penalize synchronization between the two tripod groups by summing all pair-wise async errors.

        """
        async_reward = 0
        # Sum of async penalties for all pairs between Group 1 and Group 2
        for foot_g1 in self.tripod_group1_indices:
            for foot_g2 in self.tripod_group2_indices:
                async_reward += self._async_reward_func(foot_g1, foot_g2)

        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)

        return async_reward * move_condition
    
    def _reward_contact_force_balance(self):
        """
        核心惩罚逻辑：惩罚接触力方差。
        惩罚同一三角支撑组内接触力的方差。
        这会鼓励处于同一支撑相的腿均匀地分担负载。
        """
        # 获取所有脚底的法向接触力（Z轴方向）
        normal_forces = self.contact_forces[:, self.feet_indices, 2]
        
        # 创建一个蒙版，标记哪些脚正在与地面接触（力大于1.0）
        contact_mask = (normal_forces > 1.0).float()

        # --- 处理第一组腿 ---
        # 获取第一组腿的接触力
        forces_g1 = normal_forces[:, self.tripod_group1_indices]
        # 获取第一组腿的接触蒙版
        mask_g1 = contact_mask[:, self.tripod_group1_indices]
        # 将未接触地面的腿的力置零，以便它们不影响均值和方差的计算
        masked_forces_g1 = forces_g1 * mask_g1
        # 计算第一组中接触地面的腿的数量
        num_contacting_g1 = torch.sum(mask_g1, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g1 = (num_contacting_g1 > 1).float()
        # 计算接触腿的平均力（分母加一个小数防止除以零）
        mean_g1 = torch.sum(masked_forces_g1, dim=1) / (num_contacting_g1 + 1e-6)
        # 计算接触腿的力的方差： Variance = mean( (x - mean)^2 )
        variance_g1 = torch.sum(torch.square(masked_forces_g1 - mean_g1.unsqueeze(1)) * mask_g1, dim=1) / (num_contacting_g1 + 1e-6)
        
        # --- 处理第二组腿 ---
        # 获取第二组腿的接触力
        forces_g2 = normal_forces[:, self.tripod_group2_indices]
        # 获取第二组腿的接触蒙版
        mask_g2 = contact_mask[:, self.tripod_group2_indices]
        # 将未接触地面的腿的力置零
        masked_forces_g2 = forces_g2 * mask_g2
        # 计算第二组中接触地面的腿的数量
        num_contacting_g2 = torch.sum(mask_g2, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g2 = (num_contacting_g2 > 1).float()
        # 计算接触腿的平均力
        mean_g2 = torch.sum(masked_forces_g2, dim=1) / (num_contacting_g2 + 1e-6)
        # 计算接触腿的力的方差
        variance_g2 = torch.sum(torch.square(masked_forces_g2 - mean_g2.unsqueeze(1)) * mask_g2, dim=1) / (num_contacting_g2 + 1e-6)

        # 总惩罚是两组方差的总和，仅在对应组有效（接触腿数>1）时计算
        total_variance_penalty = variance_g1 * is_valid_g1 + variance_g2 * is_valid_g2
        
        return total_variance_penalty

    def _reward_shank_vertical(self):
  
        # 获取小腿的刚体状态: [num_envs, num_shanks, 13]
        # 13维度: pos(3), quat(4), lin_vel(3), ang_vel(3)
        shank_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.shank_indices, :]
        foot_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :]

        x_error = shank_states[:, :, 0] - foot_states[:, :, 0]
        y_error = shank_states[:, :, 1] - foot_states[:, :, 1]
        
        # 计算小腿方向向量在XY平面的投影长度    
        horizontal_dist = torch.sum(torch.sqrt(x_error**2 + y_error**2), dim=1)
        

        return horizontal_dist

    
    def _reward_haa_tripod_symmetry(self):
        """
        奖励三角步态的HAA对称性。
        
        要求:
        - Group 1 (LF, RM, LB): 组内HAA角度应该相同
        - Group 2 (RF, LM, RB): 组内HAA角度应该相同
        - 两组之间HAA角度应该相反
        
        DOF顺序 (18个DOF, 每条腿3个关节):
        LB: HAA(0), HFE(1), KFE(2)
        LF: HAA(3), HFE(4), KFE(5)
        LM: HAA(6), HFE(7), KFE(8)
        RB: HAA(9), HFE(10), KFE(11)
        RF: HAA(12), HFE(13), KFE(14)
        RM: HAA(15), HFE(16), KFE(17)
        
        Returns:
            torch.Tensor: 惩罚值,当HAA不对称时返回正值
        """
        # 获取各腿的HAA角度
        LB_HAA = self.dof_pos[:, 0]   # LB
        LF_HAA = self.dof_pos[:, 3]   # LF
        LM_HAA = self.dof_pos[:, 6]   # LM
        RB_HAA = self.dof_pos[:, 9]   # RB
        RF_HAA = self.dof_pos[:, 12]  # RF
        RM_HAA = self.dof_pos[:, 15]  # RM
        
        # Group 1 (LF, RM, LB): 组内应该角度相同
        # 计算组内两两之间的角度差的平方
        g1_penalty = torch.square(LF_HAA - RM_HAA) + \
                     torch.square(LF_HAA - LB_HAA) + \
                     torch.square(RM_HAA - LB_HAA)
        
        # Group 2 (RF, LM, RB): 组内应该角度相同
        g2_penalty = torch.square(RF_HAA - LM_HAA) + \
                     torch.square(RF_HAA - RB_HAA) + \
                     torch.square(LM_HAA - RB_HAA)
        
        # 两组之间应该相反: G1的平均值 + G2的平均值 ≈ 0
        # 计算两组的平均HAA角度
        g1_mean = (LF_HAA + RM_HAA + LB_HAA) / 3.0
        g2_mean = (RF_HAA + LM_HAA + RB_HAA) / 3.0
        
        # 两组平均值应该相反(和接近0)
        inter_group_penalty = torch.square(g1_mean + g2_mean)
        
        # 总惩罚
        total_penalty = g1_penalty + g2_penalty + inter_group_penalty
        
        return total_penalty
    
    
    def post_physics_step(self):
        """Override post_physics_step to add debug info"""
        super().post_physics_step()
        # Call debug info after all computations are done
        self._debug_info()


    def _reward_stand_on_six_legs(self):
        # 低命令下：鼓励六条腿全部着地

        lin_cmd_small = torch.norm(self.commands[:, :2], dim=1) < self.speed_min
        if self.cfg.commands.heading_command:
            yaw_or_heading_small = torch.abs(self.commands[:, 3]) < self.speed_min
        else:
            yaw_or_heading_small = torch.abs(self.commands[:, 2]) < self.speed_min
        small_command_mask = torch.logical_and(lin_cmd_small, yaw_or_heading_small)

        # 足端接触（法向力阈值 1N）
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_feet_in_contact = torch.sum(foot_contact.float(), dim=1)

        # 惩罚未着地的脚数量；六足都着地时为 0
        missing_contact_penalty = len(self.feet_indices) - num_feet_in_contact

        return missing_contact_penalty * small_command_mask.float()

    
    

# Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
# 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
# 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']